"""Stage 7 (evaluation) + Stage 8 (error analysis).

Trains the improved model (full engine + clock features) on a grouped-by-player split, then reports:
- point metrics (MAE/RMSE) with an improvement table: no-engine baseline -> +engine features -> tuned;
- quantile interval quality: empirical 90% coverage, pinball loss, mean interval width;
- rating-band accuracy + adjacent-band accuracy, and the band confusion matrix;
- the headline aggregation curve: MAE vs K games-per-player (single-game noise -> precision via averaging);
- calibration (predicted vs actual, binned) and a predicted-vs-actual scatter;
- SHAP global importance on the tuned model.

Stage 8 error analysis: largest-residual games (categorised), residual vs rating band (regression to
the mean), residual vs game length, and feature-group ablations (drop a group, measure the MAE hit).

Everything is grouped by ``username`` (never by game). Figures land in reports/figures/; a run summary
is appended to reports/results.md.

Run:
    python -m src.evaluate            # improved model + full evaluation (no tuning)
    python -m src.evaluate --tune     # + Optuna hyperparameter search
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.model import (
    BASELINE_FEATURES,
    add_opponent_rating,
    fit_early_stopping,
    grouped_train_val_test,
    make_point_model,
    predict_interval,
    prepare_features,
    split_feature_columns,
    train_quantile_models,
    tune_lgbm,
)
from src.model import mae as _mae
from src.model import rmse as _rmse

# ---------------------------------------------------------------------------------------
# Metric helpers (pure, unit-tested)
# ---------------------------------------------------------------------------------------
def interval_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y, lower, upper = map(lambda a: np.asarray(a, float), (y, lower, upper))
    return float(np.mean((y >= lower) & (y <= upper)))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(np.asarray(upper, float) - np.asarray(lower, float)))


def pinball_loss(y: np.ndarray, pred: np.ndarray, alpha: float) -> float:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = y - pred
    return float(np.mean(np.maximum(alpha * err, (alpha - 1.0) * err)))


def to_bands(values: np.ndarray, bands: list[int]) -> np.ndarray:
    """Bucket ratings into band indices 0..len(bands)-2 using the config edges."""
    # bands like [0,1200,...,3000]; interior edges define the buckets.
    return np.digitize(np.asarray(values, float), bands[1:-1], right=False)


def band_accuracy(y_true: np.ndarray, y_pred: np.ndarray, bands: list[int]) -> tuple[float, float]:
    """Exact-band and adjacent-band (|Δband| <= 1) accuracy."""
    bt, bp = to_bands(y_true, bands), to_bands(y_pred, bands)
    diff = np.abs(bt - bp)
    return float(np.mean(diff == 0)), float(np.mean(diff <= 1))


def aggregation_curve(
    test_df: pd.DataFrame, ks: list[int], min_players: int, seed: int
) -> list[dict[str, float]]:
    """MAE vs K: average predictions over K games per player. Requires columns 'username','rating','pred'.

    Only reports a K where at least ``min_players`` players have >= K test games (no silent truncation).
    """
    rng = np.random.RandomState(seed)
    per_user = {
        user: (grp["rating"].to_numpy(float), grp["pred"].to_numpy(float))
        for user, grp in test_df.groupby("username")
    }
    curve = []
    for k in ks:
        errs = [
            abs(pred[(idx := rng.choice(len(rating), size=k, replace=False))].mean() - rating[idx].mean())
            for rating, pred in per_user.values()
            if len(rating) >= k
        ]
        if len(errs) >= min_players:
            curve.append({"k": k, "mae": float(np.mean(errs)), "n_players": len(errs)})
    return curve


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    """Partition features into engine / clock / opening / style for ablations."""
    groups: dict[str, list[str]] = {"engine": [], "clock": [], "opening": [], "style": []}
    for col in columns:
        cl = col.lower()
        if any(k in cl for k in ("move_time", "fast_move", "time_trouble", "n_timed")):
            groups["clock"].append(col)
        elif col == "eco":
            groups["opening"].append(col)
        elif any(k in cl for k in ("cpl", "acc", "blunder", "mistake", "inaccuracy",
                                   "reached_winning", "converted_winning")):
            groups["engine"].append(col)
        else:
            groups["style"].append(col)
    return groups


# ---------------------------------------------------------------------------------------
# Figures (guarded — a plotting failure logs a warning, never breaks the run)
# ---------------------------------------------------------------------------------------
def _fig_pred_vs_actual(y, yhat, figures_dir):
    import matplotlib.pyplot as plt

    from src.plotting import save_fig, set_style
    set_style()
    fig, ax = plt.subplots()
    ax.scatter(y, yhat, s=6, alpha=0.25)
    lims = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set(xlabel="Actual rating", ylabel="Predicted rating", title="Predicted vs actual")
    return save_fig(fig, "pred_vs_actual", figures_dir)


def _fig_calibration(y, yhat, bins, figures_dir):
    import matplotlib.pyplot as plt

    from src.plotting import save_fig, set_style
    set_style()
    edges = np.quantile(yhat, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(yhat, edges[1:-1]), 0, len(edges) - 2)
    xs, ys = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            xs.append(yhat[m].mean())
            ys.append(y[m].mean())
    fig, ax = plt.subplots()
    ax.plot(xs, ys, "o-")
    lims = [min(min(xs), min(ys)), max(max(xs), max(ys))]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set(xlabel="Mean predicted rating (bin)", ylabel="Mean actual rating (bin)", title="Calibration")
    return save_fig(fig, "calibration", figures_dir)


def _fig_confusion(y, yhat, bands, figures_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns

    from src.plotting import save_fig, set_style
    set_style()
    bt, bp = to_bands(y, bands), to_bands(yhat, bands)
    n = len(bands) - 1
    mat = np.zeros((n, n), int)
    for t, p in zip(bt, bp):
        mat[t, p] += 1
    labels = [f"{bands[i]}-{bands[i+1]}" for i in range(n)]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(xlabel="Predicted band", ylabel="Actual band", title="Rating-band confusion matrix")
    return save_fig(fig, "band_confusion", figures_dir)


def _fig_aggregation(curve, figures_dir):
    import matplotlib.pyplot as plt

    from src.plotting import save_fig, set_style
    set_style()
    ks = [c["k"] for c in curve]
    maes = [c["mae"] for c in curve]
    fig, ax = plt.subplots()
    ax.plot(ks, maes, "o-")
    ax.set(xlabel="Games per player (K)", ylabel="MAE (Elo)",
           title="Aggregation curve: MAE vs games averaged")
    return save_fig(fig, "aggregation_curve", figures_dir)


def _fig_shap(model, x, cfg, figures_dir):
    import matplotlib.pyplot as plt
    import shap

    from src.plotting import save_fig
    sample = x.sample(min(len(x), 2000), random_state=cfg["random_seed"]) if len(x) > 2000 else x
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    plt.figure()
    shap.summary_plot(values, sample, max_display=cfg["evaluate"]["shap_max_display"], show=False)
    return save_fig(plt.gcf(), "shap_summary", figures_dir)


def _guard(fn, *args, name="figure"):
    try:
        path = fn(*args)
        print(f"  saved {path}")
    except Exception as exc:  # noqa: BLE001 — figures are best-effort
        print(f"  WARNING: {name} failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------
def run_evaluation(
    cfg: dict[str, Any],
    features_path: str | Path,
    games_clean_path: str | Path,
    tune: bool = False,
    make_figures: bool = True,
    results_md: str | Path | None = None,
) -> dict[str, Any]:
    features = pd.read_parquet(features_path)
    games_clean = pd.read_parquet(games_clean_path)
    df = add_opponent_rating(features, games_clean)

    num_cols, cat_cols = split_feature_columns(df)
    full_features = num_cols + cat_cols
    df = prepare_features(df, cat_cols)
    train, val, test = grouped_train_val_test(df, cfg)

    y_tr, y_val, y_te = (s["rating"].to_numpy(float) for s in (train, val, test))
    figures_dir = cfg["paths"]["figures"]
    results: dict[str, Any] = {"n_train": len(train), "n_val": len(val), "n_test": len(test),
                               "n_features": len(full_features)}

    def fit_point(features_list, params=None):
        model = make_point_model(cfg, params)
        return fit_early_stopping(model, train[features_list], y_tr, val[features_list], y_val, cfg)

    # --- improvement table: no-engine baseline -> +engine features -> (tuned) ---
    baseline = fit_point(BASELINE_FEATURES)
    results["baseline_mae"] = _mae(y_te, baseline.predict(test[BASELINE_FEATURES]))

    full = fit_point(full_features)
    full_pred = full.predict(test[full_features])
    results["full_mae"] = _mae(y_te, full_pred)
    results["full_rmse"] = _rmse(y_te, full_pred)

    best_model, best_pred, best_params = full, full_pred, None
    if tune:
        print("tuning (Optuna)...")
        best_params = tune_lgbm(train[full_features], y_tr, train["username"].to_numpy(), cfg)
        tuned = fit_point(full_features, best_params)
        tuned_pred = tuned.predict(test[full_features])
        results["tuned_mae"] = _mae(y_te, tuned_pred)
        results["tuned_rmse"] = _rmse(y_te, tuned_pred)
        results["best_params"] = best_params
        best_model, best_pred = tuned, tuned_pred

    # --- quantile interval ---
    qmodels = train_quantile_models(train[full_features], y_tr, val[full_features], y_val, cfg, best_params)
    interval = predict_interval(qmodels, test[full_features], cfg["model"]["quantiles"])
    lo, med, up = interval["lower"], interval["median"], interval["upper"]
    results["coverage"] = interval_coverage(y_te, lo, up)
    results["mean_interval_width"] = mean_interval_width(lo, up)
    results["pinball"] = {a: pinball_loss(y_te, qmodels[a].predict(test[full_features]), a)
                          for a in cfg["model"]["quantiles"]}

    # --- band accuracy + confusion ---
    bands = cfg["rating_bands"]
    results["band_exact"], results["band_adjacent"] = band_accuracy(y_te, best_pred, bands)

    # --- aggregation curve ---
    test = test.copy()
    test["pred"] = best_pred
    curve = aggregation_curve(test, cfg["evaluate"]["aggregation_k"],
                              cfg["evaluate"]["aggregation_min_players"], cfg["random_seed"])
    results["aggregation_curve"] = curve

    # --- Stage 8 error analysis ---
    resid = best_pred - y_te
    test["residual"] = resid
    results["residual_by_band"] = _residual_by_band(y_te, resid, bands)
    results["largest_residuals"] = _largest_residuals(test, cfg["evaluate"]["error_analysis_top_n"])
    results["ablations"] = _ablations(fit_point, full_features, test, y_te, results["full_mae"])

    # --- figures ---
    if make_figures:
        print("figures:")
        _guard(_fig_pred_vs_actual, y_te, best_pred, figures_dir, name="pred_vs_actual")
        _guard(_fig_calibration, y_te, best_pred, cfg["evaluate"]["calibration_bins"], figures_dir, name="calibration")
        _guard(_fig_confusion, y_te, best_pred, bands, figures_dir, name="band_confusion")
        if curve:
            _guard(_fig_aggregation, curve, figures_dir, name="aggregation_curve")
        _guard(_fig_shap, best_model, test[full_features], cfg, figures_dir, name="shap_summary")

    _print_summary(results)
    if results_md is not None:
        _append_results_md(results_md, results, tuned=tune)
    return results


def _residual_by_band(y_true, resid, bands) -> list[dict[str, float]]:
    bt = to_bands(y_true, bands)
    out = []
    for b in range(len(bands) - 1):
        m = bt == b
        if m.any():
            out.append({"band": f"{bands[b]}-{bands[b+1]}", "n": int(m.sum()),
                        "mean_residual": float(resid[m].mean()), "mae": float(np.mean(np.abs(resid[m])))})
    return out


def _largest_residuals(test: pd.DataFrame, top_n: int) -> pd.DataFrame:
    cols = [c for c in ["game_id", "username", "rating", "pred", "residual", "n_moves", "cpl_mean"]
            if c in test.columns]
    top = test.reindex(test["residual"].abs().sort_values(ascending=False).index).head(top_n)[cols].copy()
    # categorise the hard cases
    hi = top["rating"] >= test["rating"].median()
    top["case"] = np.where(top["residual"] > 0,
                           np.where(hi, "high-rated, over-predicted", "low-rated, over-predicted"),
                           np.where(hi, "high-rated, under-predicted", "low-rated, under-predicted"))
    return top


def _ablations(fit_point, full_features, test, y_te, full_mae) -> list[dict[str, Any]]:
    """Drop each feature group, retrain the point model, and record the MAE increase."""
    groups = feature_groups(full_features)
    out = []
    for name, cols in groups.items():
        if not cols:
            continue
        kept = [c for c in full_features if c not in cols]
        model = fit_point(kept)
        ab_mae = _mae(y_te, model.predict(test[kept]))
        out.append({"group": name, "n_dropped": len(cols), "mae_without": ab_mae,
                    "mae_increase": ab_mae - full_mae})
    out.sort(key=lambda d: d["mae_increase"], reverse=True)
    return out


# ---------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------
def _print_summary(r: dict[str, Any]) -> None:
    print(f"\nImproved model - n_train={r['n_train']:,} n_test={r['n_test']:,} features={r['n_features']}")
    print(f"  baseline (no-engine) MAE : {r['baseline_mae']:.1f}")
    print(f"  + engine/clock features  : {r['full_mae']:.1f}  (RMSE {r['full_rmse']:.1f})")
    if "tuned_mae" in r:
        print(f"  + Optuna tuning          : {r['tuned_mae']:.1f}  (RMSE {r['tuned_rmse']:.1f})")
    print(f"  90% interval coverage    : {r['coverage']*100:.1f}%  (target 90)  width {r['mean_interval_width']:.0f}")
    print(f"  band exact / adjacent    : {r['band_exact']*100:.1f}% / {r['band_adjacent']*100:.1f}%")
    if r["aggregation_curve"]:
        c = r["aggregation_curve"]
        print(f"  aggregation MAE K={c[0]['k']}->{c[-1]['k']}: {c[0]['mae']:.1f} -> {c[-1]['mae']:.1f}")
    print("  ablations (MAE increase when the group is dropped):")
    for a in r["ablations"]:
        print(f"    -{a['group']:<8} (+{a['mae_increase']:.1f})")


def _append_results_md(path: str | Path, r: dict[str, Any], tuned: bool) -> None:
    from datetime import datetime

    lines = [
        f"\n## {datetime.now():%Y-%m-%d %H:%M} — improved model{' (tuned)' if tuned else ''}",
        f"n_train={r['n_train']}, n_test={r['n_test']}, n_features={r['n_features']}",
        "",
        "| stage | MAE | RMSE |",
        "|---|---|---|",
        f"| no-engine baseline | {r['baseline_mae']:.1f} | — |",
        f"| + engine/clock | {r['full_mae']:.1f} | {r['full_rmse']:.1f} |",
    ]
    if "tuned_mae" in r:
        lines.append(f"| + tuned | {r['tuned_mae']:.1f} | {r['tuned_rmse']:.1f} |")
    lines += [
        "",
        f"- 90% interval coverage: **{r['coverage']*100:.1f}%** (width {r['mean_interval_width']:.0f} Elo)",
        f"- band accuracy: {r['band_exact']*100:.1f}% exact, {r['band_adjacent']*100:.1f}% adjacent",
    ]
    if r["aggregation_curve"]:
        curve = ", ".join(f"K{c['k']}={c['mae']:.0f}" for c in r["aggregation_curve"])
        lines.append(f"- aggregation MAE: {curve}")
    lines.append("- ablation (MAE↑ when dropped): "
                 + ", ".join(f"{a['group']} +{a['mae_increase']:.1f}" for a in r["ablations"]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    from src.config import load_config

    cfg = load_config()
    parser = argparse.ArgumentParser(description="Stages 7-8: evaluate the improved model.")
    parser.add_argument("--features", default=cfg["outputs"]["features"])
    parser.add_argument("--games", default=cfg["outputs"]["games_clean"])
    parser.add_argument("--tune", action="store_true", help="run Optuna hyperparameter search")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    parser.add_argument("--results-md", default=str(Path(cfg["paths"]["figures"]).parent / "results.md"))
    args = parser.parse_args()

    for label, path in (("features", args.features), ("games_clean", args.games)):
        if not Path(path).exists():
            raise SystemExit(f"{label} not found: {path}\nRun the earlier stages first.")
    run_evaluation(cfg, args.features, args.games, tune=args.tune,
                   make_figures=not args.no_figures, results_md=(args.results_md or None))


if __name__ == "__main__":
    main()
