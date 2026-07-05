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
    apply_conformal,
    apply_conformal_by_band,
    band_sample_weights,
    conformal_correction,
    conformal_correction_by_band,
    fit_early_stopping,
    grouped_train_val_calib_test,
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


def _coverage_by_band(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, bands: list[int]
) -> list[dict[str, float]]:
    """Empirical interval coverage + mean width per rating band (does under-coverage hide in a tail?)."""
    y_true, lower, upper = (np.asarray(a, float) for a in (y_true, lower, upper))
    bt = to_bands(y_true, bands)
    out = []
    for b in range(len(bands) - 1):
        m = bt == b
        if m.any():
            out.append({
                "band": f"{bands[b]}-{bands[b+1]}", "n": int(m.sum()),
                "coverage": interval_coverage(y_true[m], lower[m], upper[m]),
                "mean_width": mean_interval_width(lower[m], upper[m]),
            })
    return out


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(np.asarray(a, float)))
    return order.astype(float)


def point_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    """A fuller picture than MAE alone: error spread, variance explained, rank, tail bias."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    resid = yhat - y                                   # + = over-predicted
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "median_ae": float(np.median(np.abs(resid))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "pearson": float(np.corrcoef(yhat, y)[0, 1]),
        "spearman": float(np.corrcoef(_rank(yhat), _rank(y))[0, 1]),
        "within_100": float(np.mean(np.abs(resid) <= 100)),
        "within_200": float(np.mean(np.abs(resid) <= 200)),
        "bias": float(np.mean(resid)),                 # global over/under-prediction
    }


def fit_deshrink(pred_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float]:
    """Fit a linear de-shrink map (slope, intercept) of true~pred on validation.

    A shrunk predictor has slope(true~pred) > 1, so applying it expands predictions away from the
    centre — removing the systematic tail bias, at the cost of extra variance on noisy signal.
    """
    slope, intercept = np.polyfit(np.asarray(pred_val, float), np.asarray(y_val, float), 1)
    return float(slope), float(intercept)


def apply_deshrink(pred: np.ndarray, coef: tuple[float, float]) -> np.ndarray:
    slope, intercept = coef
    return slope * np.asarray(pred, float) + intercept


def _band_mae_table(y_true: np.ndarray, y_pred: np.ndarray, bands: list[int]) -> dict[str, float]:
    """MAE per band + overall + a tail-vs-mid summary, for comparing tail corrections."""
    bt = to_bands(y_true, bands)
    out = {"overall": float(np.mean(np.abs(y_pred - y_true)))}
    maes = []
    for b in range(len(bands) - 1):
        m = bt == b
        val = float(np.mean(np.abs(y_pred[m] - y_true[m]))) if m.any() else float("nan")
        out[f"{bands[b]}-{bands[b+1]}"] = val
        maes.append(val)
    out["tail"] = float(np.nanmean([maes[0], maes[-1]]))      # lowest + highest band
    out["mid"] = float(np.nanmean(maes[1:-1]))
    return out


def run_tail_study(
    cfg: dict[str, Any], features_path, games_path, results_md: str | Path | None = None
) -> dict[str, dict[str, float]]:
    """Compare the point model plain vs band-reweighted vs post-hoc de-shrink, per rating band.

    Answers "how much of the tail regression-to-the-mean can we remove, and at what cost?"
    """
    features = pd.read_parquet(features_path)
    games = pd.read_parquet(games_path)
    df = add_opponent_rating(features, games)
    num, cat = split_feature_columns(df)
    full = num + cat
    df = prepare_features(df, cat)
    train, val, calib, test = grouped_train_val_calib_test(df, cfg)
    y_tr, y_val, y_cal, y_te = (s["rating"].to_numpy(float) for s in (train, val, calib, test))
    bands = cfg["rating_bands"]

    plain = make_point_model(cfg)
    fit_early_stopping(plain, train[full], y_tr, val[full], y_val, cfg)
    pred_plain = plain.predict(test[full])

    weights = band_sample_weights(y_tr, bands, cfg["model"]["balance_strength"])
    weighted = make_point_model(cfg)
    fit_early_stopping(weighted, train[full], y_tr, val[full], y_val, cfg, sample_weight=weights)
    pred_weighted = weighted.predict(test[full])

    # de-shrink is fit on CALIB (never seen by early stopping) so the correction isn't tuned on
    # the same data the model used to pick its stopping point.
    coef = fit_deshrink(plain.predict(calib[full]), y_cal)
    pred_deshrink = apply_deshrink(pred_plain, coef)

    variants = {
        "plain": _band_mae_table(y_te, pred_plain, bands),
        f"reweighted(s={cfg['model']['balance_strength']})": _band_mae_table(y_te, pred_weighted, bands),
        f"deshrink(slope={coef[0]:.2f})": _band_mae_table(y_te, pred_deshrink, bands),
    }
    _print_tail_study(variants, bands)
    if results_md is not None:
        _append_tail_study_md(results_md, variants, bands)
    return variants


def _print_tail_study(variants: dict[str, dict[str, float]], bands: list[int]) -> None:
    band_cols = [f"{bands[b]}-{bands[b+1]}" for b in range(len(bands) - 1)]
    print("\nTail-correction study (MAE by rating band):")
    header = f"  {'variant':<24}{'overall':>8}{'tail':>7}{'mid':>7}   " + "".join(f"{c:>11}" for c in band_cols)
    print(header)
    for name, tbl in variants.items():
        row = f"  {name:<24}{tbl['overall']:>8.1f}{tbl['tail']:>7.0f}{tbl['mid']:>7.0f}   "
        row += "".join(f"{tbl[c]:>11.0f}" for c in band_cols)
        print(row)


def _append_tail_study_md(path: str | Path, variants: dict[str, dict[str, float]], bands: list[int]) -> None:
    from datetime import datetime

    band_cols = [f"{bands[b]}-{bands[b+1]}" for b in range(len(bands) - 1)]
    header = ["variant", "overall", "tail", "mid"] + band_cols
    lines = [
        f"\n## {datetime.now():%Y-%m-%d %H:%M} — tail study",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for name, tbl in variants.items():
        row = [name, f"{tbl['overall']:.1f}", f"{tbl['tail']:.0f}", f"{tbl['mid']:.0f}"]
        row += [f"{tbl[c]:.0f}" for c in band_cols]
        lines.append("| " + " | ".join(row) + " |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def per_player_metrics(test_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate ALL of each player's test games (mean prediction) — the useful, denoised output."""
    agg = test_df.groupby("username").agg(
        pred=("pred", "mean"), rating=("rating", "mean"), n=("rating", "size"),
    )
    return {
        "n_players": int(len(agg)),
        "mae": float(np.mean(np.abs(agg["pred"] - agg["rating"]))),
        "mean_games_per_player": float(agg["n"].mean()),
        "multi_game_share": float(np.mean(agg["n"] >= 2)),
    }


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


def _fig_interval_by_band(y, lower, upper, bands, figures_dir):
    import matplotlib.pyplot as plt

    from src.plotting import save_fig, set_style
    set_style()
    cov = _coverage_by_band(y, lower, upper, bands)
    labels = [c["band"] for c in cov]
    coverages = [c["coverage"] * 100 for c in cov]
    widths = [c["mean_width"] for c in cov]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.bar(labels, coverages)
    ax1.axhline(90, color="k", ls="--", lw=1)
    ax1.set(xlabel="Rating band", ylabel="Coverage (%)", title="90% interval coverage by band (CQR)")
    ax1.tick_params(axis="x", rotation=45)
    ax2.bar(labels, widths)
    ax2.set(xlabel="Rating band", ylabel="Mean interval width (Elo)", title="Interval width by band (CQR)")
    ax2.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return save_fig(fig, "interval_by_band", figures_dir)


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
    train, val, calib, test = grouped_train_val_calib_test(df, cfg)

    y_tr, y_val, y_cal, y_te = (s["rating"].to_numpy(float) for s in (train, val, calib, test))
    bands = cfg["rating_bands"]
    figures_dir = cfg["paths"]["figures"]
    results: dict[str, Any] = {"n_train": len(train), "n_val": len(val), "n_calib": len(calib),
                               "n_test": len(test), "n_features": len(full_features)}

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

    # --- quantile interval, conformalized against the held-out calib split ---
    # Two intervals are always computed: plain split-CQR (one global correction) and Mondrian
    # (band-conditional) CQR — a separate correction per PREDICTED-median band, which fixes the
    # plain interval's tendency to under-cover the rating extremes and over-cover the middle.
    # ``model.mondrian_cqr`` selects which one is the HEADLINE interval (coverage/width/coverage_by_band
    # below); both are reported so the improvement is visible.
    quantiles = cfg["model"]["quantiles"]
    alpha = round(1.0 - (max(quantiles) - min(quantiles)), 10)   # e.g. [0.05,0.5,0.95] -> 0.10
    qmodels = train_quantile_models(train[full_features], y_tr, val[full_features], y_val, cfg, best_params)

    interval_cal = predict_interval(qmodels, calib[full_features], quantiles)
    interval_raw = predict_interval(qmodels, test[full_features], quantiles)

    # plain (marginal) split-CQR — one global correction applied everywhere.
    correction = conformal_correction(interval_cal, y_cal, alpha, cfg["model"]["cqr_two_sided"])
    interval_plain = apply_conformal(interval_raw, correction)

    # Mondrian (band-conditional) split-CQR — band assignment uses the PREDICTED median only (see
    # conformal_correction_by_band's docstring for the true-band caveat).
    mondrian_corrections = conformal_correction_by_band(
        interval_cal, y_cal, alpha, bands, cfg["model"]["cqr_two_sided"], cfg["model"]["mondrian_min_calib"],
    )
    interval_mondrian = apply_conformal_by_band(interval_raw, mondrian_corrections, bands)

    results["coverage_raw"] = interval_coverage(y_te, interval_raw["lower"], interval_raw["upper"])
    results["mean_interval_width_raw"] = mean_interval_width(interval_raw["lower"], interval_raw["upper"])

    results["coverage_cqr_plain"] = interval_coverage(y_te, interval_plain["lower"], interval_plain["upper"])
    results["mean_interval_width_plain"] = mean_interval_width(interval_plain["lower"], interval_plain["upper"])
    results["coverage_by_band_plain"] = _coverage_by_band(y_te, interval_plain["lower"], interval_plain["upper"], bands)

    results["coverage_cqr_mondrian"] = interval_coverage(y_te, interval_mondrian["lower"], interval_mondrian["upper"])
    results["mean_interval_width_mondrian"] = mean_interval_width(interval_mondrian["lower"], interval_mondrian["upper"])
    results["coverage_by_band_mondrian"] = _coverage_by_band(y_te, interval_mondrian["lower"], interval_mondrian["upper"], bands)

    results["mondrian_headline"] = bool(cfg["model"]["mondrian_cqr"])
    headline = interval_mondrian if results["mondrian_headline"] else interval_plain
    lo, up = headline["lower"], headline["upper"]

    results["coverage"] = interval_coverage(y_te, lo, up)
    results["mean_interval_width"] = mean_interval_width(lo, up)
    results["conformal_correction"] = correction               # plain global tuple, always
    results["mondrian_corrections"] = mondrian_corrections      # per predicted-band dict
    results["pinball"] = {a: pinball_loss(y_te, qmodels[a].predict(test[full_features]), a)
                          for a in quantiles}
    results["coverage_by_band"] = _coverage_by_band(y_te, lo, up, bands)

    # --- fuller point metrics + band accuracy + confusion ---
    results["point"] = point_metrics(y_te, best_pred)
    results["band_exact"], results["band_adjacent"] = band_accuracy(y_te, best_pred, bands)

    # --- aggregation curve + per-player (all games averaged) ---
    test = test.copy()
    test["pred"] = best_pred
    curve = aggregation_curve(test, cfg["evaluate"]["aggregation_k"],
                              cfg["evaluate"]["aggregation_min_players"], cfg["random_seed"])
    results["aggregation_curve"] = curve
    results["per_player"] = per_player_metrics(test)

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
        _guard(_fig_interval_by_band, y_te, lo, up, bands, figures_dir, name="interval_by_band")
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
    p = r["point"]
    print(f"  median AE / R2 / rho     : {p['median_ae']:.0f} / {p['r2']:.3f} / {p['spearman']:.3f}")
    print(f"  within 100 / 200 Elo     : {p['within_100']*100:.1f}% / {p['within_200']*100:.1f}%   (bias {p['bias']:+.1f})")
    pp = r["per_player"]
    print(f"  per-player (all games)   : {pp['mae']:.1f} MAE over {pp['n_players']:,} players "
          f"({pp['multi_game_share']*100:.0f}% have >=2 games)")
    print(f"  90% interval coverage    : raw {r['coverage_raw']*100:.1f}% -> CQR(plain) "
          f"{r['coverage_cqr_plain']*100:.1f}% (target 90); "
          f"width {r['mean_interval_width_raw']:.0f} -> {r['mean_interval_width_plain']:.0f}")
    headline = "Mondrian" if r.get("mondrian_headline") else "plain"
    print(f"  Mondrian CQR             : {r['coverage_cqr_mondrian']*100:.1f}% "
          f"width {r['mean_interval_width_mondrian']:.0f}  [headline={headline}]; by band: "
          + "  ".join(f"{b['band']}={b['coverage']*100:.0f}%" for b in r["coverage_by_band_mondrian"]))
    print(f"  coverage by band (headline={headline}): "
          + "  ".join(f"{b['band']}={b['coverage']*100:.0f}%" for b in r["coverage_by_band"]))
    print(f"  band exact / adjacent    : {r['band_exact']*100:.1f}% / {r['band_adjacent']*100:.1f}%")
    if r["aggregation_curve"]:
        c = r["aggregation_curve"]
        print(f"  aggregation MAE K={c[0]['k']}->{c[-1]['k']}: {c[0]['mae']:.1f} -> {c[-1]['mae']:.1f}")
    print("  residual by band (mean resid = regression to the mean):")
    for rb in r["residual_by_band"]:
        print(f"    {rb['band']:<10} n={rb['n']:>5}  mean_resid {rb['mean_residual']:+7.1f}  MAE {rb['mae']:.0f}")
    print("  ablations (MAE increase when the group is dropped):")
    for a in r["ablations"]:
        print(f"    -{a['group']:<8} (+{a['mae_increase']:.1f})")


def _append_results_md(path: str | Path, r: dict[str, Any], tuned: bool) -> None:
    from datetime import datetime

    lines = [
        f"\n## {datetime.now():%Y-%m-%d %H:%M} — improved model{' (tuned)' if tuned else ''}",
        f"n_train={r['n_train']}, n_calib={r['n_calib']}, n_test={r['n_test']}, n_features={r['n_features']}",
        "",
        "| stage | MAE | RMSE |",
        "|---|---|---|",
        f"| no-engine baseline | {r['baseline_mae']:.1f} | — |",
        f"| + engine/clock | {r['full_mae']:.1f} | {r['full_rmse']:.1f} |",
    ]
    if "tuned_mae" in r:
        lines.append(f"| + tuned | {r['tuned_mae']:.1f} | {r['tuned_rmse']:.1f} |")
    p, pp = r["point"], r["per_player"]
    lines += [
        "",
        f"- median AE {p['median_ae']:.0f}, R² {p['r2']:.3f}, Spearman {p['spearman']:.3f}, "
        f"within 100/200 Elo {p['within_100']*100:.0f}%/{p['within_200']*100:.0f}%, bias {p['bias']:+.1f}",
        f"- per-player (all games averaged): **{pp['mae']:.1f}** MAE over {pp['n_players']} players",
        f"- 90% interval coverage: raw {r['coverage_raw']*100:.1f}% -> CQR(plain) "
        f"**{r['coverage_cqr_plain']*100:.1f}%** "
        f"(width {r['mean_interval_width_raw']:.0f} -> {r['mean_interval_width_plain']:.0f} Elo; "
        f"correction lo={r['conformal_correction'][0]:.1f}, hi={r['conformal_correction'][1]:.1f})",
        f"- Mondrian CQR: **{r['coverage_cqr_mondrian']*100:.1f}%** "
        f"(width {r['mean_interval_width_mondrian']:.0f} Elo)"
        f"{' — headline' if r.get('mondrian_headline') else ''}",
        "- coverage by band (headline=" + ("mondrian" if r.get("mondrian_headline") else "plain") + "): "
        + ", ".join(f"{b['band']} {b['coverage']*100:.0f}%" for b in r["coverage_by_band"]),
        "- per-band coverage, plain/Mondrian: " + ", ".join(
            f"{pb['band']} {pb['coverage']*100:.0f}%/{mb['coverage']*100:.0f}%"
            for pb, mb in zip(r["coverage_by_band_plain"], r["coverage_by_band_mondrian"])),
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
    parser.add_argument("--tail-study", action="store_true",
                        help="only compare tail-bias corrections (plain vs reweighted vs de-shrink)")
    parser.add_argument("--results-md", default=str(Path(cfg["paths"]["figures"]).parent / "results.md"))
    args = parser.parse_args()

    for label, path in (("features", args.features), ("games_clean", args.games)):
        if not Path(path).exists():
            raise SystemExit(f"{label} not found: {path}\nRun the earlier stages first.")
    if args.tail_study:
        run_tail_study(cfg, args.features, args.games, results_md=(args.results_md or None))
        return
    run_evaluation(cfg, args.features, args.games, tune=args.tune,
                   make_figures=not args.no_figures, results_md=(args.results_md or None))


if __name__ == "__main__":
    main()
