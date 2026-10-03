"""Confirmatory hold-out: a later day of the month, players never seen before, scored ONCE.

The development test split (120,513 rows) was inspected many times while the method took shape
(feature additions, the missing-value A/B, tuning runs, the tail study, the aggregation
recalibration), so its numbers are development estimates. This module freezes the method as it
stands and scores it once on data that played no part in any decision:

- every eligible game (same filters as the development data) played on ``holdout.utc_date``,
  the last day of the month — the development data covers May 1-5;
- only the rows of players who appear nowhere in the development data (in no split and in
  neither colour), so no hold-out player was used to fit, early-stop, calibrate or evaluate.

The models are refit exactly as in ``src.evaluate`` (same data, split, seed and settings, hence
the same models); the hold-out is scored with the conformal corrections and the K-aware
recalibration fitted on the development calibration split. The protocol, written before the
first run, is reports/HOLDOUT_PROTOCOL.md.

Run, in order:
    python -m src.holdout ingest              # stream the dump from the hold-out day on
    python -m src.holdout prepare             # clean, drop development players, features
    python -m src.holdout evaluate --dry-run  # the same scoring code on the development test split
    python -m src.holdout evaluate            # score the hold-out (refuses to run a second time)
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluate import (
    _coverage_by_band,
    _json_default,
    _residual_by_band,
    aggregate_band_table,
    cluster_bootstrap_mae,
    fit_deshrink,
    interval_coverage,
    matched_aggregation,
    mean_interval_width,
    point_metrics,
    prior_interval,
    slope_bootstrap_ci,
)
from src.model import (
    BASELINE_FEATURES,
    _append_md,
    add_opponent_rating,
    apply_conformal,
    apply_conformal_by_band,
    conformal_correction,
    conformal_correction_by_band,
    current_commit_hash,
    fit_early_stopping,
    grouped_train_val_calib_test,
    make_point_model,
    predict_interval,
    prepare_features,
    split_feature_columns,
    train_quantile_models,
)


# ---------------------------------------------------------------------------------------
# Data: ingest the hold-out day, clean it, keep only players new to the project
# ---------------------------------------------------------------------------------------
def run_holdout_ingest(cfg: dict[str, Any]) -> dict[str, int]:
    """Every eligible game of ``holdout.utc_date``: no scan budget, no sampling, no cohort."""
    from src.ingest import run_ingest

    hcfg = copy.deepcopy(cfg)
    hcfg["max_games_scanned"] = None
    hcfg["sample_size"] = 10 ** 8                  # a reservoir larger than the day keeps every game
    hcfg["player_cohort"] = {"enabled": False}
    out = cfg["outputs"]
    return run_ingest(cfg["raw_path"], out["holdout_games"], hcfg, cohort_output_path=None,
                      funnel_path=out["data_funnel"], utc_date=cfg["holdout"]["utc_date"],
                      funnel_stage="holdout_ingest")


def development_players(games_clean: pd.DataFrame) -> set[str]:
    """Every username in the development data, in either colour and any split."""
    return set(games_clean["white"]).union(games_clean["black"])


def run_holdout_prepare(cfg: dict[str, Any]) -> dict[str, int]:
    """Clean + explode the hold-out games, drop the rows of development players, build features."""
    from src.clean import build_instances, clean_games
    from src.features import build_features
    from src.ingest import write_funnel

    out = cfg["outputs"]
    raw = pd.read_parquet(out["holdout_games"])
    games, clean_funnel = clean_games(raw, cfg)
    instances = build_instances(games)

    dev = development_players(pd.read_parquet(out["games_clean"], columns=["white", "black"]))
    new = ~instances["username"].isin(dev)
    kept = instances[new]
    games = games[games["game_id"].isin(kept["game_id"])]

    Path(out["holdout_games_clean"]).parent.mkdir(parents=True, exist_ok=True)
    games.to_parquet(out["holdout_games_clean"], engine="pyarrow", index=False)
    features = build_features(games, kept, cfg)
    features.to_parquet(out["holdout_features"], engine="pyarrow", index=False)

    funnel = {
        "games_ingested": len(raw), **{f"games_{k}": v for k, v in clean_funnel.items()},
        "instances": len(instances),
        "instances_of_development_players": int((~new).sum()),
        "instances_kept": int(new.sum()),
        "players_kept": int(kept["username"].nunique()),
        "games_with_a_kept_instance": len(games),
        "feature_rows": len(features),
    }
    write_funnel(out["data_funnel"], "holdout_prepare", funnel)
    for k, v in funnel.items():
        print(f"  {k:<36}{v:>10,}")
    return funnel


# ---------------------------------------------------------------------------------------
# The frozen method: refit exactly as src.evaluate does
# ---------------------------------------------------------------------------------------
def fit_frozen(cfg: dict[str, Any]) -> dict[str, Any]:
    """Refit the development models (same data, split, seed, settings -> the same models)."""
    out = cfg["outputs"]
    df = add_opponent_rating(pd.read_parquet(out["features"]), pd.read_parquet(out["games_clean"]))
    num_cols, cat_cols = split_feature_columns(df)
    full = num_cols + cat_cols
    df = prepare_features(df, cat_cols)
    train, val, calib, test = grouped_train_val_calib_test(df, cfg)
    y_tr, y_val, y_cal = (s["rating"].to_numpy(float) for s in (train, val, calib))

    def fit(cols, params=None):
        return fit_early_stopping(make_point_model(cfg, params), train[cols], y_tr, val[cols], y_val, cfg)

    quantiles = cfg["model"]["quantiles"]
    alpha = round(1.0 - (max(quantiles) - min(quantiles)), 10)
    qmodels = train_quantile_models(train[full], y_tr, val[full], y_val, cfg)
    interval_cal = predict_interval(qmodels, calib[full], quantiles)
    two_sided = cfg["model"]["cqr_two_sided"]
    correction = conformal_correction(interval_cal, y_cal, alpha, two_sided)
    mondrian = conformal_correction_by_band(interval_cal, y_cal, alpha, cfg["rating_bands"], two_sided,
                                            cfg["model"]["mondrian_min_calib"], global_correction=correction)
    full_model = fit(full)
    calib = calib.copy()
    calib["pred"] = full_model.predict(calib[full])
    return {
        "full_features": full, "cat_cols": cat_cols, "categories": {c: df[c].cat.categories for c in cat_cols},
        "train_mean": float(y_tr.mean()), "y_train": y_tr, "test": test, "calib": calib,
        "baseline": fit(BASELINE_FEATURES), "full": full_model,
        "tuned": fit(full, dict(cfg["holdout"]["tuned_params"])),
        "qmodels": qmodels, "quantiles": quantiles, "correction": correction, "mondrian": mondrian,
    }


def load_holdout(cfg: dict[str, Any], frozen: dict[str, Any]) -> pd.DataFrame:
    """The hold-out feature rows, with the development category sets on the categorical columns."""
    df = pd.read_parquet(cfg["outputs"]["holdout_features"])
    for col in frozen["cat_cols"]:
        df[col] = pd.Categorical(df[col], categories=frozen["categories"][col])
    return df


# ---------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------
def score(target: pd.DataFrame, frozen: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Every pre-registered metric on ``target`` (the hold-out, or the development test split)."""
    full, bands = frozen["full_features"], cfg["rating_bands"]
    ecfg, seed = cfg["evaluate"], cfg["random_seed"]
    reps = int(ecfg.get("bootstrap_reps", 0))
    y = target["rating"].to_numpy(float)
    groups = target["username"].to_numpy()

    pred = {
        "predict_mean": np.full(len(target), frozen["train_mean"]),
        "baseline": frozen["baseline"].predict(target[BASELINE_FEATURES]),
        "full": frozen["full"].predict(target[full]),
        "tuned": frozen["tuned"].predict(target[full]),
    }
    errs = {k: np.abs(v - y) for k, v in pred.items()}
    res: dict[str, Any] = {
        "n_rows": len(target), "n_players": int(target["username"].nunique()),
        "rating_mean": float(y.mean()), "rating_sd": float(y.std()),
        "mae": {k: float(e.mean()) for k, e in errs.items()},
        "point_full": point_metrics(y, pred["full"]),
    }
    if reps:
        res["bootstrap"] = cluster_bootstrap_mae(
            errs, groups, reps, seed,
            diffs=[("predict_mean", "full"), ("baseline", "full"), ("full", "tuned")])

    interval_raw = predict_interval(frozen["qmodels"], target[full], frozen["quantiles"])
    plain = apply_conformal(interval_raw, frozen["correction"])
    mondrian = apply_conformal_by_band(interval_raw, frozen["mondrian"], bands)
    res["interval"] = {
        name: {"coverage": interval_coverage(y, iv["lower"], iv["upper"]),
               "mean_width": mean_interval_width(iv["lower"], iv["upper"]),
               "by_band": _coverage_by_band(y, iv["lower"], iv["upper"], bands)}
        for name, iv in (("raw", interval_raw), ("cqr_plain", plain), ("cqr_mondrian", mondrian))
    }
    res["prior_interval"] = prior_interval(frozen["y_train"], y, frozen["quantiles"])
    res["residual_by_band"] = _residual_by_band(y, pred["full"] - y, bands)
    slope, intercept = fit_deshrink(pred["full"], y)
    res["calibration_line"] = {"slope": slope, "intercept": intercept}
    if reps:
        res["calibration_line"].update(slope_bootstrap_ci(pred["full"], y, groups, reps, seed))

    scored = target[["username", "rating"]].copy()
    scored["pred"] = pred["full"]
    min_players = int(ecfg["aggregation_min_players"])
    draws = int(ecfg.get("matched_draws", 40))
    res["matched_aggregation"] = matched_aggregation(scored, frozen["calib"], ecfg.get("matched_k", [1, 2, 3, 5, 10]),
                                                     draws, seed, min_players)
    res["aggregate_bands_k5"] = aggregate_band_table(scored, frozen["calib"], 5, bands, draws, seed, reps, min_players)
    return res


def _report_lines(r: dict[str, Any], header: str) -> list[str]:
    m, b = r["mae"], r.get("bootstrap", {})

    def ci(key: str) -> str:
        v = b.get(key)
        return f" [95% CI {v['lo']:.1f}, {v['hi']:.1f}]" if v and "mae" in v else ""

    def dci(key: str) -> str:
        v = b.get(key)
        return f"{v['diff']:+.1f} [95% CI {v['lo']:+.1f}, {v['hi']:+.1f}]" if v else ""

    iv = r["interval"]
    p = r["point_full"]
    cal = r["calibration_line"]
    lines = [
        header,
        f"n_rows={r['n_rows']}, n_players={r['n_players']}, rating mean {r['rating_mean']:.0f} (sd {r['rating_sd']:.0f})",
        "",
        "| model | MAE |",
        "|---|---|",
        f"| predict_mean (development training mean) | {m['predict_mean']:.1f}{ci('predict_mean')} |",
        f"| no-engine baseline | {m['baseline']:.1f}{ci('baseline')} |",
        f"| full model (headline) | {m['full']:.1f}{ci('full')} |",
        f"| tuned | {m['tuned']:.1f}{ci('tuned')} |",
        "",
        f"- differences (player-clustered bootstrap): mean - full {dci('predict_mean - full')}; "
        f"baseline - full {dci('baseline - full')}; full - tuned {dci('full - tuned')}",
        f"- full model: RMSE {p['rmse']:.1f}, median AE {p['median_ae']:.0f}, R² {p['r2']:.3f}, "
        f"Spearman {p['spearman']:.3f}, bias {p['bias']:+.1f}",
        f"- 90% interval coverage: raw {iv['raw']['coverage']*100:.1f}%, CQR(plain) "
        f"**{iv['cqr_plain']['coverage']*100:.1f}%** (width {iv['cqr_plain']['mean_width']:.0f}), "
        f"Mondrian **{iv['cqr_mondrian']['coverage']*100:.1f}%** (width {iv['cqr_mondrian']['mean_width']:.0f}); "
        f"no-game interval {r['prior_interval']['coverage']*100:.1f}% (width {r['prior_interval']['width']:.0f})",
        "- per-band coverage, plain/Mondrian: " + ", ".join(
            f"{pb['band']} {pb['coverage']*100:.0f}%/{mb['coverage']*100:.0f}%"
            for pb, mb in zip(iv["cqr_plain"]["by_band"], iv["cqr_mondrian"]["by_band"])),
        "- residual by band (full): " + ", ".join(
            f"{x['band']} {x['mean_residual']:+.0f} (n={x['n']})" for x in r["residual_by_band"]),
        f"- calibration line true~pred: slope {cal['slope']:.3f}"
        + (f" [95% CI {cal['slope_lo']:.3f}, {cal['slope_hi']:.3f}]" if "slope_lo" in cal else "")
        + f", intercept {cal['intercept']:.0f}",
    ]
    if r["matched_aggregation"]:
        m0 = r["matched_aggregation"][0]
        lines.append(f"- matched aggregation ({m0['n_players']} players with >= 10 games), naive -> recalibrated MAE: "
                     + ", ".join(f"K{x['k']}={x['naive_mae']:.0f}->{x['recal_mae']:.0f}" for x in r["matched_aggregation"]))
    else:
        lines.append("- matched aggregation: not estimable (too few players with >= 10 games)")
    ab = r["aggregate_bands_k5"]
    if ab:
        c = ab.get("naive_minus_recal_ci")
        lines.append(f"- K=5 ({ab['n_players']} players with >= 5 games): single {ab['single_mae']:.0f}, "
                     f"naive {ab['naive_mae']:.0f}, recalibrated {ab['recal_mae']:.0f}"
                     + (f" (naive - recal {c['diff']:+.1f} [95% CI {c['lo']:+.1f}, {c['hi']:+.1f}])" if c else ""))
    else:
        lines.append("- K=5: not estimable (too few players with >= 5 games)")
    return lines


def run_holdout_evaluate(cfg: dict[str, Any], dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    out = cfg["outputs"]
    results_path = Path(out["holdout_results"])
    if not dry_run and results_path.exists() and not force:
        raise SystemExit(f"{results_path} exists: the hold-out has been scored. The protocol allows one run.")
    source_commit = current_commit_hash()
    frozen = fit_frozen(cfg)
    target = frozen["test"] if dry_run else load_holdout(cfg, frozen)
    res = score(target, frozen, cfg)
    res["source_commit"] = source_commit
    res["utc_date"] = cfg["holdout"]["utc_date"]
    header = (f"\n## {source_commit} — hold-out dry run (development test split, not the hold-out)" if dry_run
              else f"\n## {source_commit} — confirmatory hold-out ({cfg['holdout']['utc_date']}, new players, scored once)")
    lines = _report_lines(res, header)
    print("\n".join(lines))
    if not dry_run:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(res, indent=1, default=_json_default) + "\n", encoding="utf-8")
        _append_md(out["results_log"], lines)
        print(f"  wrote {results_path}")
    return res


def main() -> None:
    from src.config import load_config

    cfg = load_config()
    parser = argparse.ArgumentParser(description="Confirmatory hold-out (see reports/HOLDOUT_PROTOCOL.md).")
    parser.add_argument("step", choices=["ingest", "prepare", "evaluate"])
    parser.add_argument("--dry-run", action="store_true",
                        help="evaluate: score the development test split instead (checks the code; writes nothing)")
    parser.add_argument("--force", action="store_true", help="evaluate: overwrite an existing hold-out result")
    args = parser.parse_args()
    if args.step == "ingest":
        run_holdout_ingest(cfg)
    elif args.step == "prepare":
        run_holdout_prepare(cfg)
    else:
        run_holdout_evaluate(cfg, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
