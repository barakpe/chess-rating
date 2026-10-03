"""Stage 7 (evaluation) + Stage 8 (error analysis).

Trains the improved model (full engine + clock features) on a grouped-by-player split, then reports:
- point metrics (MAE/RMSE, R², rank correlation, bias) with an improvement table
  (no-engine baseline -> +engine/clock features -> tuned) and player-clustered bootstrap CIs;
- quantile interval quality: empirical 90% coverage (raw, split CQR, Mondrian CQR), pinball loss,
  mean width, per-band coverage, and an unconditional (label-quantile) interval for context;
- rating-band accuracy + adjacent-band accuracy, and the band confusion matrix;
- aggregation: the shifting-cohort curve (MAE vs K games/player), the MATCHED curve (the same
  players at every K) and a K-aware recalibration of averaged predictions fit on the calib split;
- calibration (predicted vs actual, binned) and a predicted-vs-actual scatter;
- SHAP global importance on the point model.

Stage 8 error analysis: largest-residual games (categorised), residual by rating band (regression to
the mean) and by game length, feature-group ablations (drop a group, measure the MAE hit, with CIs),
a scramble-block ablation, and a check that a test instance whose opponent's instance was in the
training data is not predicted better (no game-level leakage through the grouped split).

Everything is grouped by ``username`` (never by game). Figures land in reports/figures/; a run
summary is appended to reports/results.md; structured results go to reports/eval_artifacts.json
and the worst single-game misses to reports/largest_residuals.csv (both read by the notebooks).

Run:
    python -m src.evaluate                  # improved model + full evaluation (no tuning)
    python -m src.evaluate --tune           # + Optuna hyperparameter search
    python -m src.evaluate --params '{...}' # + fit the "tuned" row with given hyperparameters
    python -m src.evaluate --tail-study     # single-game tail-correction comparison only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.model import (
    BASELINE_FEATURES,
    _append_md,
    add_opponent_rating,
    apply_conformal,
    apply_conformal_by_band,
    band_sample_weights,
    conformal_correction,
    conformal_correction_by_band,
    current_commit_hash,
    fit_early_stopping,
    grouped_train_val_calib_test,
    make_point_model,
    predict_interval,
    predicted_bands,
    prepare_features,
    split_feature_columns,
    to_bands,
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


def band_labels(bands: list[int]) -> list[str]:
    """``"lo-hi"`` label per band. The last band is open-ended in ``to_bands`` (it also holds the
    handful of ratings above its nominal upper edge); the label keeps the config edge for
    continuity with earlier log entries."""
    return [f"{bands[b]}-{bands[b + 1]}" for b in range(len(bands) - 1)]


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
    for b, label in enumerate(band_labels(bands)):
        m = bt == b
        if m.any():
            out.append({
                "band": label, "n": int(m.sum()),
                "coverage": interval_coverage(y_true[m], lower[m], upper[m]),
                "mean_width": mean_interval_width(lower[m], upper[m]),
            })
    return out


def _rank(a: np.ndarray) -> np.ndarray:
    """Ranks with ties sharing their average rank (ratings are integers, so ties are common)."""
    return pd.Series(np.asarray(a, float)).rank(method="average").to_numpy()


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
    """Fit a linear de-shrink map (slope, intercept) of true~pred on held-out data.

    A shrunk predictor has slope(true~pred) > 1, so applying it expands predictions away from the
    centre — removing the systematic tail bias, at the cost of extra variance on noisy signal.
    A slope of ~1 means the predictor is already calibrated in the E[true | pred] sense.
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
    for b, label in enumerate(band_labels(bands)):
        m = bt == b
        val = float(np.mean(np.abs(y_pred[m] - y_true[m]))) if m.any() else float("nan")
        out[label] = val
        maes.append(val)
    out["tail"] = float(np.nanmean([maes[0], maes[-1]]))      # lowest + highest band
    out["mid"] = float(np.nanmean(maes[1:-1]))
    return out


# ---------------------------------------------------------------------------------------
# Uncertainty on the numbers themselves: player-clustered bootstrap
# ---------------------------------------------------------------------------------------
def cluster_bootstrap_mae(
    errors: dict[str, np.ndarray],
    groups: np.ndarray,
    reps: int,
    seed: int,
    diffs: list[tuple[str, str]] | tuple = (),
    masks: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, float]]:
    """Percentile-bootstrap 95% CIs for MAEs and MAE differences, resampling PLAYERS.

    ``errors[name]`` is a per-row absolute error aligned with ``groups`` (the username of each row).
    A player's games are not independent observations, so whole players are resampled with
    replacement (all of a player's rows move together) — resampling rows would understate the
    uncertainty. ``masks[name]`` optionally restricts a statistic to a subset of rows (e.g. a
    sub-population); every statistic in one replicate uses the same resampled players, so
    ``diffs`` — pairs ``(a, b)`` reported as ``MAE_a - MAE_b`` — are paired comparisons.
    """
    codes, uniques = pd.factorize(np.asarray(groups))
    n_groups = len(uniques)
    masks = masks or {}
    sums, counts = {}, {}
    for name, err in errors.items():
        err = np.asarray(err, float)
        m = np.asarray(masks.get(name, np.ones(len(err), bool)), bool)
        sums[name] = np.bincount(codes[m], weights=err[m], minlength=n_groups)
        counts[name] = np.bincount(codes[m], minlength=n_groups).astype(float)

    if reps <= 0:
        raise ValueError("cluster_bootstrap_mae needs reps >= 1")
    rng = np.random.RandomState(seed)
    boot = {name: np.empty(reps) for name in errors}
    for r in range(reps):
        w = np.bincount(rng.randint(0, n_groups, n_groups), minlength=n_groups).astype(float)
        for name in errors:
            denom = w @ counts[name]
            boot[name][r] = (w @ sums[name]) / denom if denom else np.nan

    out: dict[str, dict[str, float]] = {}
    for name in errors:
        lo, hi = np.nanpercentile(boot[name], [2.5, 97.5])
        out[name] = {"mae": float(sums[name].sum() / counts[name].sum()), "lo": float(lo), "hi": float(hi)}
    for a, b in diffs:
        d = boot[a] - boot[b]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        out[f"{a} - {b}"] = {"diff": out[a]["mae"] - out[b]["mae"], "lo": float(lo), "hi": float(hi)}
    return out


def slope_bootstrap_ci(pred: np.ndarray, y: np.ndarray, groups: np.ndarray, reps: int, seed: int
                       ) -> dict[str, float]:
    """Player-clustered bootstrap 95% CIs for the slope and intercept of the line true ~ pred.

    Uses per-player sums (n, x, y, x², xy), so each replicate is a least-squares fit with whole
    players resampled.
    """
    codes, uniques = pd.factorize(np.asarray(groups))
    g = len(uniques)
    x, t = np.asarray(pred, float), np.asarray(y, float)
    sums = [np.bincount(codes, weights=w, minlength=g) for w in (np.ones_like(x), x, t, x * x, x * t)]
    rng = np.random.RandomState(seed)
    slopes, intercepts = np.empty(reps), np.empty(reps)
    for r in range(reps):
        w = np.bincount(rng.randint(0, g, g), minlength=g).astype(float)
        n, sx, st, sxx, sxt = (w @ s_ for s_ in sums)
        slopes[r] = (n * sxt - sx * st) / (n * sxx - sx * sx)
        intercepts[r] = (st - slopes[r] * sx) / n
    (s_lo, s_hi), (i_lo, i_hi) = np.percentile(slopes, [2.5, 97.5]), np.percentile(intercepts, [2.5, 97.5])
    return {"slope_lo": float(s_lo), "slope_hi": float(s_hi),
            "intercept_lo": float(i_lo), "intercept_hi": float(i_hi)}


def _one_row_per_group(groups: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Index of one random row per group."""
    perm = rng.permutation(len(groups))
    first = ~pd.Series(np.asarray(groups)[perm]).duplicated().to_numpy()
    return perm[first]


def one_game_per_player_cqr(
    interval_cal: dict[str, np.ndarray], y_cal: np.ndarray, users_cal: np.ndarray,
    interval_te: dict[str, np.ndarray], y_te: np.ndarray, users_te: np.ndarray,
    alpha: float, two_sided: bool, draws: int, seed: int,
) -> dict[str, float]:
    """Split CQR with ONE random game per player, in calibration and in test.

    With several games per player the calibration scores are not independent, so the standard
    exchangeability argument behind the conformal guarantee does not apply to rows. Keeping one
    game per player makes the scores independent across players (players are sampled
    independently of each other), so the guarantee holds for "a random game of a new player".
    Averaged over ``draws`` random choices of the game.
    """
    rng = np.random.RandomState(seed)
    y_cal, y_te = np.asarray(y_cal, float), np.asarray(y_te, float)
    cov, width, corr = [], [], []
    for _ in range(draws):
        ci = _one_row_per_group(users_cal, rng)
        ti = _one_row_per_group(users_te, rng)
        c = conformal_correction({k: v[ci] for k, v in interval_cal.items()}, y_cal[ci], alpha, two_sided)
        iv = apply_conformal({k: v[ti] for k, v in interval_te.items()}, c)
        cov.append(interval_coverage(y_te[ti], iv["lower"], iv["upper"]))
        width.append(mean_interval_width(iv["lower"], iv["upper"]))
        corr.append(c[0])
    return {"coverage": float(np.mean(cov)), "coverage_min": float(np.min(cov)),
            "coverage_max": float(np.max(cov)), "mean_width": float(np.mean(width)),
            "correction": float(np.mean(corr)), "draws": draws,
            "n_calib_players": int(pd.Series(users_cal).nunique()),
            "n_test_players": int(pd.Series(users_te).nunique())}


# ---------------------------------------------------------------------------------------
# Single-game tail-correction study
# ---------------------------------------------------------------------------------------
def run_tail_study(
    cfg: dict[str, Any], features_path, games_path, results_md: str | Path | None = None
) -> dict[str, dict[str, float]]:
    """Compare the point model plain vs band-reweighted vs post-hoc de-shrink, per rating band.

    Answers "how much of the single-game tail regression-to-the-mean can we remove, and at what cost?"
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

    # Up-weight rare ratings: inverse frequency over FINE rating bins. (The six confusion bands would
    # do the opposite — the two outer bands are the largest, so they would get the smallest weights.)
    width = cfg["model"].get("reweight_bin_width", 100)
    fine_edges = list(np.arange(0, (np.ceil(max(y_tr.max(), y_te.max()) / width) + 1) * width + 1, width))
    weights = band_sample_weights(y_tr, fine_edges, cfg["model"]["balance_strength"])
    weighted = make_point_model(cfg)
    fit_early_stopping(weighted, train[full], y_tr, val[full], y_val, cfg, sample_weight=weights)
    pred_weighted = weighted.predict(test[full])
    band_idx = to_bands(y_tr, bands)
    mean_weight = {label: float(weights[band_idx == b].mean())
                   for b, label in enumerate(band_labels(bands)) if (band_idx == b).any()}

    # The linear map true ~ pred is fit on CALIB (never seen by early stopping). A slope > 1 would
    # stretch ("de-shrink") the predictions; a slope <= 1 means there is no leftover shrinkage.
    coef = fit_deshrink(plain.predict(calib[full]), y_cal)
    pred_deshrink = apply_deshrink(pred_plain, coef)

    variants = {
        "plain": _band_mae_table(y_te, pred_plain, bands),
        f"reweighted({width}-Elo bins, s={cfg['model']['balance_strength']})": _band_mae_table(y_te, pred_weighted, bands),
        f"deshrink(slope={coef[0]:.2f})": _band_mae_table(y_te, pred_deshrink, bands),
    }
    _print_tail_study(variants, bands)
    print("  mean training weight by band (reweighted): "
          + ", ".join(f"{k} {v:.2f}" for k, v in mean_weight.items()))
    if results_md is not None:
        _append_tail_study_md(results_md, variants, bands)
        _append_md(results_md, ["- reweighted: mean training weight by band: "
                                + ", ".join(f"{k} {v:.2f}" for k, v in mean_weight.items())])
    return variants


def _print_tail_study(variants: dict[str, dict[str, float]], bands: list[int]) -> None:
    band_cols = band_labels(bands)
    print("\nTail-correction study (MAE by rating band):")
    header = f"  {'variant':<24}{'overall':>8}{'tail':>7}{'mid':>7}   " + "".join(f"{c:>11}" for c in band_cols)
    print(header)
    for name, tbl in variants.items():
        row = f"  {name:<24}{tbl['overall']:>8.1f}{tbl['tail']:>7.0f}{tbl['mid']:>7.0f}   "
        row += "".join(f"{tbl[c]:>11.0f}" for c in band_cols)
        print(row)


def _append_tail_study_md(path: str | Path, variants: dict[str, dict[str, float]], bands: list[int]) -> None:
    band_cols = band_labels(bands)
    header = ["variant", "overall", "tail", "mid"] + band_cols
    lines = [
        f"\n## {current_commit_hash()} — tail study",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for name, tbl in variants.items():
        row = [name, f"{tbl['overall']:.1f}", f"{tbl['tail']:.0f}", f"{tbl['mid']:.0f}"]
        row += [f"{tbl[c]:.0f}" for c in band_cols]
        lines.append("| " + " | ".join(row) + " |")
    _append_md(path, lines)


# ---------------------------------------------------------------------------------------
# Aggregation: combining several games of the same player
# ---------------------------------------------------------------------------------------
def per_player_metrics(test_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate ALL of each player's test games (mean prediction) — the naive denoised output."""
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

    Each K is measured on the players who have >= K test games, so the population SHRINKS (and
    changes) as K grows — see ``matched_aggregation`` for the same-players version. Only reports a K
    where at least ``min_players`` players have >= K test games (no silent truncation).
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


def _player_games(df: pd.DataFrame, min_games: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(ratings, preds) per player with >= ``min_games`` rows, in sorted-username order."""
    return [
        (grp["rating"].to_numpy(float), grp["pred"].to_numpy(float))
        for _, grp in df.groupby("username", sort=True)
        if len(grp) >= min_games
    ]


def _k_averages(
    players: list[tuple[np.ndarray, np.ndarray]], k: int, draws: int, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    """For each player, ``draws`` random K-subsets of their games -> mean rating, mean prediction.

    Returns two arrays of shape (n_players, draws).
    """
    ratings = np.empty((len(players), draws))
    preds = np.empty((len(players), draws))
    for i, (r, p) in enumerate(players):
        for d in range(draws):
            idx = rng.choice(len(r), size=k, replace=False)
            ratings[i, d] = r[idx].mean()
            preds[i, d] = p[idx].mean()
    return ratings, preds


def fit_aggregate_recalibration(
    players: list[tuple[np.ndarray, np.ndarray]], k: int, draws: int, seed: int
) -> tuple[float, float]:
    """Linear map true ~ (mean of K single-game predictions), fit on held-out players.

    Why this is needed: each single-game prediction is a shrunk estimate — shrinking toward the
    population mean is the right response to ONE noisy game (single-game slope ~1). Averaging K
    shrunk estimates keeps the single-game shrinkage although K games carry more evidence, so the
    averaged prediction is under-dispersed and the slope of true ~ average grows above 1 with K.
    Fitting that slope per K on players the model never saw removes the excess shrinkage.
    """
    ratings, preds = _k_averages(players, k, draws, np.random.RandomState(seed))
    return fit_deshrink(preds.ravel(), ratings.ravel())


def matched_aggregation(
    test_df: pd.DataFrame, calib_df: pd.DataFrame, ks: list[int], draws: int, seed: int,
    min_players: int = 2,
) -> list[dict[str, float]]:
    """Aggregation on a FIXED population: the test players with >= max(ks) games, at every K.

    For each K: the naive estimate (mean of K single-game predictions) and the K-aware recalibrated
    estimate (``fit_aggregate_recalibration`` fit on calib players with >= max(ks) games). Errors are
    averaged over ``draws`` random K-subsets per player. Empty if the test or the calib split has
    fewer than ``min_players`` such players (too few to fit or to report).
    """
    kmax = max(ks)
    test_players = _player_games(test_df, kmax)
    calib_players = _player_games(calib_df, kmax)
    if len(test_players) < max(2, min_players) or len(calib_players) < max(2, min_players):
        return []
    out = []
    for k in ks:
        slope, intercept = fit_aggregate_recalibration(calib_players, k, draws, seed + 1000 + k)
        ratings, preds = _k_averages(test_players, k, draws, np.random.RandomState(seed + k))
        naive = preds - ratings
        recal = slope * preds + intercept - ratings
        out.append({
            "k": k, "n_players": len(test_players), "n_calib_players": len(calib_players),
            "naive_mae": float(np.mean(np.abs(naive))), "naive_rmse": float(np.sqrt(np.mean(naive ** 2))),
            "recal_mae": float(np.mean(np.abs(recal))), "recal_rmse": float(np.sqrt(np.mean(recal ** 2))),
            "slope": slope, "intercept": intercept,
        })
    return out


def aggregate_band_table(
    test_df: pd.DataFrame, calib_df: pd.DataFrame, k: int, bands: list[int], draws: int, seed: int,
    reps: int = 0, min_players: int = 2,
) -> dict[str, Any]:
    """Per-band single-game vs K-averaged (naive and recalibrated) MAE and bias.

    Population: test players with >= K games (band = the player's mean test rating); all MAEs here
    are per-player averages over ``draws`` random K-subsets. The recalibration is fit on calib
    players with >= K games. With ``reps`` > 0, adds a bootstrap 95% CI for the overall
    naive - recalibrated MAE difference that resamples BOTH the calib players (refitting the
    recalibration each time) and the test players, so the CI includes the uncertainty of the fit.
    Empty if either split has fewer than ``min_players`` players with >= K games.
    """
    test_players = _player_games(test_df, k)
    calib_players = _player_games(calib_df, k)
    if len(test_players) < max(2, min_players) or len(calib_players) < max(2, min_players):
        return {}
    rc_cal, pc_cal = _k_averages(calib_players, k, draws, np.random.RandomState(seed + 2000 + k))
    slope, intercept = fit_deshrink(pc_cal.ravel(), rc_cal.ravel())   # == fit_aggregate_recalibration
    r1, p1 = _k_averages(test_players, 1, draws, np.random.RandomState(seed + 1))
    rk, pk = _k_averages(test_players, k, draws, np.random.RandomState(seed + k))
    single = p1 - r1                                       # (players, draws)
    naive = pk - rk
    recal = slope * pk + intercept - rk
    per_player = {
        "single_abs": np.abs(single).mean(1), "naive_abs": np.abs(naive).mean(1),
        "recal_abs": np.abs(recal).mean(1),
        "single_res": single.mean(1), "naive_res": naive.mean(1), "recal_res": recal.mean(1),
    }
    player_rating = np.array([r.mean() for r, _ in test_players])
    band_idx = to_bands(player_rating, bands)
    rows = []
    for b, label in enumerate(band_labels(bands)):
        m = band_idx == b
        if not m.any():
            continue
        rows.append({
            "band": label, "n_players": int(m.sum()),
            "single_mae": float(per_player["single_abs"][m].mean()),
            "naive_mae": float(per_player["naive_abs"][m].mean()),
            "recal_mae": float(per_player["recal_abs"][m].mean()),
            "single_bias": float(per_player["single_res"][m].mean()),
            "naive_bias": float(per_player["naive_res"][m].mean()),
            "recal_bias": float(per_player["recal_res"][m].mean()),
        })
    out: dict[str, Any] = {
        "k": k, "n_players": len(test_players), "n_calib_players": len(calib_players),
        "slope": slope, "intercept": intercept,
        "single_mae": float(per_player["single_abs"].mean()),
        "naive_mae": float(per_player["naive_abs"].mean()),
        "recal_mae": float(per_player["recal_abs"].mean()),
        "bands": rows,
    }
    if reps:
        rng = np.random.RandomState(seed + 3000 + k)
        n_cal, n_te = len(calib_players), len(test_players)
        diffs = np.empty(reps)
        for r in range(reps):
            cal_idx = rng.randint(0, n_cal, n_cal)
            s_r, i_r = fit_deshrink(pc_cal[cal_idx].ravel(), rc_cal[cal_idx].ravel())
            te_idx = rng.randint(0, n_te, n_te)
            recal_r = np.abs(s_r * pk[te_idx] + i_r - rk[te_idx]).mean(1)
            diffs[r] = per_player["naive_abs"][te_idx].mean() - recal_r.mean()
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out["naive_minus_recal_ci"] = {"diff": out["naive_mae"] - out["recal_mae"],
                                       "lo": float(lo), "hi": float(hi)}
    return out


# ---------------------------------------------------------------------------------------
# Other diagnostics
# ---------------------------------------------------------------------------------------
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


def scramble_columns(columns: list[str]) -> list[str]:
    """The clock-scramble block (``scramble_*`` + ``has_scramble``), for its dedicated ablation."""
    return [c for c in columns if "scramble" in c]


def partner_exposure(test_df: pd.DataFrame, seen_game_ids: set[str]) -> np.ndarray:
    """Mask of test rows whose GAME the model was fit on: the opponent's row is in
    ``seen_game_ids`` (the TRAINING split's games).

    The split is grouped by player, not by game, so a test player's opponent can sit in train.
    Game-level features (length, ECO, time control, phase lengths) are shared by both instances
    and the opponent's label is close to this player's rating (matchmaking), so if the model had
    memorised games, these rows would be predicted better than the rest. A narrow check of one
    route, not a proof that no leakage exists.
    """
    return test_df["game_id"].isin(seen_game_ids).to_numpy()


def prior_interval(y_train: np.ndarray, y_test: np.ndarray, quantiles: list[float]) -> dict[str, float]:
    """The interval you get WITHOUT looking at the game: training-label quantiles for everyone."""
    lo, hi = np.quantile(np.asarray(y_train, float), [min(quantiles), max(quantiles)])
    y_test = np.asarray(y_test, float)
    return {"lower": float(lo), "upper": float(hi), "width": float(hi - lo),
            "coverage": float(np.mean((y_test >= lo) & (y_test <= hi)))}


def _mondrian_diagnostics(median_cal: np.ndarray, median_te: np.ndarray, y_te: np.ndarray,
                          bands: list[int]) -> dict[str, Any]:
    """How many calib rows each PREDICTED band has, and where each TRUE band's rows are predicted."""
    nb = len(bands) - 1
    calib_counts = np.bincount(predicted_bands(median_cal, bands), minlength=nb)
    crosstab = np.zeros((nb, nb), int)
    for t, p in zip(to_bands(y_te, bands), predicted_bands(median_te, bands)):
        crosstab[t, p] += 1
    return {"labels": band_labels(bands), "calib_rows_by_predicted_band": calib_counts.tolist(),
            "test_true_by_predicted_band": crosstab.tolist()}


def _residual_by_band(y_true, resid, bands) -> list[dict[str, float]]:
    bt = to_bands(y_true, bands)
    out = []
    for b, label in enumerate(band_labels(bands)):
        m = bt == b
        if m.any():
            out.append({"band": label, "n": int(m.sum()),
                        "mean_residual": float(resid[m].mean()), "mae": float(np.mean(np.abs(resid[m])))})
    return out


_LENGTH_BINS = [0, 20, 30, 40, 60, 10_000]   # the player's own moves in the game


def _residual_by_length(player_moves: np.ndarray, resid: np.ndarray) -> list[dict[str, float]]:
    """MAE / mean residual by game length (the player's own move count): do short games carry less?"""
    idx = np.digitize(np.asarray(player_moves, float), _LENGTH_BINS[1:-1])
    out = []
    for b in range(len(_LENGTH_BINS) - 1):
        m = idx == b
        if m.any():
            hi = _LENGTH_BINS[b + 1]
            label = f"{_LENGTH_BINS[b]}-{hi - 1}" if hi < 10_000 else f"{_LENGTH_BINS[b]}+"
            out.append({"moves": label, "n": int(m.sum()),
                        "mae": float(np.mean(np.abs(resid[m]))), "mean_residual": float(resid[m].mean())})
    return out


def _largest_residuals(test: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """The ``top_n`` worst single-game misses. No identifiers are written (no username, no game
    id): the table is committed, and a game id would lead back to the players."""
    cols = [c for c in ["rating", "pred", "residual", "player_moves", "cpl_mean"] if c in test.columns]
    top = test.reindex(test["residual"].abs().sort_values(ascending=False).index).head(top_n)[cols].copy()
    hi = top["rating"] >= test["rating"].median()
    top["case"] = np.where(top["residual"] > 0,
                           np.where(hi, "high-rated, over-predicted", "low-rated, over-predicted"),
                           np.where(hi, "high-rated, under-predicted", "low-rated, under-predicted"))
    return top


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
    labels = band_labels(bands)
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
           title="Aggregation curve (shifting cohort): MAE vs games averaged")
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
    params: dict[str, Any] | None = None,
    artifacts_path: str | Path | None = None,
    residuals_csv: str | Path | None = None,
    sample_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train + evaluate the improved model. ``params`` (optional) fits the "tuned" row with the given
    LightGBM hyperparameters instead of running the Optuna search (``tune``). ``sample_path``
    (optional): the reservoir-sample parquet, for a sensitivity check on the test rows that come
    from the uniform game sample (i.e. without the cohort-only games)."""
    if tune and params:
        raise ValueError("pass either tune=True (Optuna search) or params (fixed hyperparameters), not both")
    source_commit = current_commit_hash()      # stamp the code that is about to run, not the code at the end
    features = pd.read_parquet(features_path)
    games_clean = pd.read_parquet(games_clean_path)
    df = add_opponent_rating(features, games_clean)

    num_cols, cat_cols = split_feature_columns(df)
    full_features = num_cols + cat_cols
    df = prepare_features(df, cat_cols)
    train, val, calib, test = grouped_train_val_calib_test(df, cfg)

    y_tr, y_val, y_cal, y_te = (s["rating"].to_numpy(float) for s in (train, val, calib, test))
    bands = cfg["rating_bands"]
    ecfg = cfg["evaluate"]
    seed = cfg["random_seed"]
    reps = int(ecfg.get("bootstrap_reps", 0))
    figures_dir = cfg["paths"]["figures"]
    results: dict[str, Any] = {"n_train": len(train), "n_val": len(val), "n_calib": len(calib),
                               "n_test": len(test), "n_features": len(full_features),
                               "n_test_players": int(test["username"].nunique())}

    def fit_point(features_list, fit_params=None):
        model = make_point_model(cfg, fit_params)
        return fit_early_stopping(model, train[features_list], y_tr, val[features_list], y_val, cfg)

    # --- improvement table: no-engine baseline -> +engine features -> (tuned) ---
    baseline = fit_point(BASELINE_FEATURES)
    baseline_pred = baseline.predict(test[BASELINE_FEATURES])
    results["baseline_mae"] = _mae(y_te, baseline_pred)

    full = fit_point(full_features)
    full_pred = full.predict(test[full_features])
    results["full_mae"] = _mae(y_te, full_pred)
    results["full_rmse"] = _rmse(y_te, full_pred)

    best_model, best_pred, best_params = full, full_pred, None
    tuned_pred = None
    if tune or params:
        if tune:
            print("tuning (Optuna)...")
            best_params = tune_lgbm(train[full_features], y_tr, train["username"].to_numpy(), cfg)
            results["params_source"] = "optuna"
        else:
            best_params = dict(params)
            results["params_source"] = "given"
        tuned = fit_point(full_features, best_params)
        tuned_pred = tuned.predict(test[full_features])
        results["tuned_mae"] = _mae(y_te, tuned_pred)
        results["tuned_rmse"] = _rmse(y_te, tuned_pred)
        results["best_params"] = best_params
        best_model, best_pred = tuned, tuned_pred

    # --- quantile interval, conformalized against the held-out calib split ---
    # Two intervals are always computed: plain split-CQR (one global correction) and Mondrian
    # (band-conditional) CQR — a separate correction per PREDICTED-median band. Mondrian targets
    # coverage conditional on the prediction; it cannot target coverage conditional on the TRUE band
    # (see conformal_correction_by_band). ``model.mondrian_cqr`` selects the HEADLINE interval.
    quantiles = cfg["model"]["quantiles"]
    alpha = round(1.0 - (max(quantiles) - min(quantiles)), 10)   # e.g. [0.05,0.5,0.95] -> 0.10
    qmodels = train_quantile_models(train[full_features], y_tr, val[full_features], y_val, cfg, best_params)

    interval_cal = predict_interval(qmodels, calib[full_features], quantiles)
    interval_raw = predict_interval(qmodels, test[full_features], quantiles)

    correction = conformal_correction(interval_cal, y_cal, alpha, cfg["model"]["cqr_two_sided"])
    interval_plain = apply_conformal(interval_raw, correction)

    mondrian_corrections = conformal_correction_by_band(
        interval_cal, y_cal, alpha, bands, cfg["model"]["cqr_two_sided"], cfg["model"]["mondrian_min_calib"],
        global_correction=correction,
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
    results["mondrian_diagnostics"] = _mondrian_diagnostics(
        interval_cal["median"], interval_raw["median"], y_te, bands)
    results["prior_interval"] = prior_interval(y_tr, y_te, quantiles)
    results["pinball"] = {a: pinball_loss(y_te, qmodels[a].predict(test[full_features]), a)
                          for a in quantiles}
    results["coverage_by_band"] = (
        results["coverage_by_band_mondrian"] if results["mondrian_headline"] else results["coverage_by_band_plain"]
    )

    # --- fuller point metrics + band accuracy + confusion ---
    results["point"] = point_metrics(y_te, best_pred)
    results["band_exact"], results["band_adjacent"] = band_accuracy(y_te, best_pred, bands)
    calib = calib.copy()
    calib["pred"] = best_model.predict(calib[full_features])     # held-out: no model was fit on calib
    results["single_game_deshrink"] = dict(zip(("slope", "intercept"),
                                               fit_deshrink(calib["pred"].to_numpy(), y_cal)))
    if reps:
        results["single_game_deshrink"].update(
            slope_bootstrap_ci(calib["pred"].to_numpy(), y_cal, calib["username"].to_numpy(), reps, seed))

    # --- one game per player: the conformal setting without within-player dependence ---
    results["cqr_one_game_per_player"] = one_game_per_player_cqr(
        interval_cal, y_cal, calib["username"].to_numpy(), interval_raw, y_te, test["username"].to_numpy(),
        alpha, cfg["model"]["cqr_two_sided"], int(ecfg.get("one_game_draws", 20)), seed)

    # --- sensitivity: test rows from the uniform reservoir sample only (no cohort-only games) ---
    if sample_path is not None and Path(sample_path).exists():
        sample_ids = set(pd.read_parquet(sample_path, columns=["game_id"])["game_id"])
        in_sample = test["game_id"].isin(sample_ids).to_numpy()
        results["reservoir_only"] = {
            "n_test": int(in_sample.sum()), "n_excluded": int((~in_sample).sum()),
            "mae": _mae(y_te[in_sample], best_pred[in_sample]),
            "baseline_mae": _mae(y_te[in_sample], baseline_pred[in_sample]),
            "coverage": interval_coverage(y_te[in_sample], lo[in_sample], up[in_sample]),
        }

    # --- aggregation: shifting-cohort curve, per-player, matched curve + K-aware recalibration ---
    test = test.copy()
    test["pred"] = best_pred
    curve = aggregation_curve(test, ecfg["aggregation_k"], ecfg["aggregation_min_players"], seed)
    results["aggregation_curve"] = curve
    results["per_player"] = per_player_metrics(test)
    matched_k = ecfg.get("matched_k", [1, 2, 3, 5, 10])
    draws = int(ecfg.get("matched_draws", 40))
    min_players = int(ecfg["aggregation_min_players"])
    results["matched_aggregation"] = matched_aggregation(test, calib, matched_k, draws, seed, min_players)
    results["aggregate_bands_k5"] = aggregate_band_table(test, calib, 5, bands, draws, seed, reps, min_players)

    # --- Stage 8 error analysis ---
    resid = best_pred - y_te
    test["residual"] = resid
    results["residual_by_band"] = _residual_by_band(y_te, resid, bands)
    if "player_moves" in test.columns:
        results["residual_by_length"] = _residual_by_length(test["player_moves"].to_numpy(), resid)
    largest = _largest_residuals(test, ecfg["error_analysis_top_n"])
    results["largest_residuals"] = largest
    ablations, ablation_preds = _ablations(fit_point, full_features, test, y_te, results["full_mae"])
    results["ablations"] = ablations

    # Scramble block: the whole clock-scramble feature family, dropped at once.
    scr = scramble_columns(full_features)
    scramble_pred = None
    if scr:
        scramble_pred = fit_point([c for c in full_features if c not in scr]).predict(test[[c for c in full_features if c not in scr]])
        results["scramble_ablation"] = {
            "n_dropped": len(scr),
            "without": _band_mae_table(y_te, scramble_pred, bands),
            "with": _band_mae_table(y_te, full_pred, bands),
        }

    # Game-level leakage check: is a test row predicted better when its opponent's row was fit on
    # (is in the training split)? Validation rows only steer early stopping, so they count as unseen.
    seen = set(train["game_id"])
    seen_mask = partner_exposure(test, seen)
    results["partner_check"] = {"n_seen": int(seen_mask.sum()), "n_unseen": int((~seen_mask).sum())}

    # --- player-clustered bootstrap CIs for every MAE comparison above ---
    if reps:
        groups = test["username"].to_numpy()
        errs = {"baseline": np.abs(baseline_pred - y_te), "full": np.abs(full_pred - y_te)}
        diffs = [("baseline", "full")]
        if tuned_pred is not None:
            errs["tuned"] = np.abs(tuned_pred - y_te)
            diffs.append(("full", "tuned"))
        for name, pred in ablation_preds.items():
            errs[f"no_{name}"] = np.abs(pred - y_te)
            diffs.append((f"no_{name}", "full"))
        if scramble_pred is not None:
            errs["no_scramble"] = np.abs(scramble_pred - y_te)
            diffs.append(("no_scramble", "full"))
        results["bootstrap"] = cluster_bootstrap_mae(errs, groups, reps, seed, diffs)
        err_best = np.abs(best_pred - y_te)
        pc = cluster_bootstrap_mae({"seen": err_best, "unseen": err_best}, groups, reps, seed,
                                   diffs=[("seen", "unseen")], masks={"seen": seen_mask, "unseen": ~seen_mask})
        results["partner_check"].update({"mae_seen": pc["seen"]["mae"], "mae_unseen": pc["unseen"]["mae"],
                                         "diff_ci": pc["seen - unseen"]})
    else:
        err_best = np.abs(best_pred - y_te)
        results["partner_check"].update({"mae_seen": float(err_best[seen_mask].mean()) if seen_mask.any() else float("nan"),
                                         "mae_unseen": float(err_best[~seen_mask].mean()) if (~seen_mask).any() else float("nan")})

    # --- figures ---
    if make_figures:
        print("figures:")
        _guard(_fig_pred_vs_actual, y_te, best_pred, figures_dir, name="pred_vs_actual")
        _guard(_fig_calibration, y_te, best_pred, ecfg["calibration_bins"], figures_dir, name="calibration")
        _guard(_fig_confusion, y_te, best_pred, bands, figures_dir, name="band_confusion")
        _guard(_fig_interval_by_band, y_te, lo, up, bands, figures_dir, name="interval_by_band")
        if curve:
            _guard(_fig_aggregation, curve, figures_dir, name="aggregation_curve")
        _guard(_fig_shap, best_model, test[full_features], cfg, figures_dir, name="shap_summary")

    results["source_commit"] = source_commit
    results["rating_bands"] = bands
    _print_summary(results)
    if results_md is not None:
        _append_results_md(results_md, results)
    if artifacts_path is not None:
        write_artifacts(results, artifacts_path)
        print(f"  wrote {artifacts_path}")
    if residuals_csv is not None:
        Path(residuals_csv).parent.mkdir(parents=True, exist_ok=True)
        largest.to_csv(residuals_csv, index=False)
        print(f"  wrote {residuals_csv}")
    return results


def _ablations(fit_point, full_features, test, y_te, full_mae) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Drop each feature group, retrain the point model, and record the MAE increase.

    Returns (rows sorted by MAE increase, {group: test predictions}) — the predictions feed the
    bootstrap CIs.
    """
    groups = feature_groups(full_features)
    out, preds = [], {}
    for name, cols in groups.items():
        if not cols:
            continue
        kept = [c for c in full_features if c not in cols]
        model = fit_point(kept)
        preds[name] = model.predict(test[kept])
        ab_mae = _mae(y_te, preds[name])
        out.append({"group": name, "n_dropped": len(cols), "mae_without": ab_mae,
                    "mae_increase": ab_mae - full_mae})
    out.sort(key=lambda d: d["mae_increase"], reverse=True)
    return out, preds


# ---------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def write_artifacts(results: dict[str, Any], path: str | Path) -> None:
    """Write the structured results (everything but the residual table) as JSON for the notebooks."""
    payload = {k: v for k, v in results.items() if k != "largest_residuals"}
    payload["mondrian_corrections"] = {str(k): list(v) for k, v in results["mondrian_corrections"].items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1, default=_json_default) + "\n", encoding="utf-8")


def _fmt_ci(ci: dict[str, float], key: str = "diff") -> str:
    return f"{ci[key]:+.1f} [95% CI {ci['lo']:+.1f}, {ci['hi']:+.1f}]"


def _print_summary(r: dict[str, Any]) -> None:
    print(f"\nImproved model - n_train={r['n_train']:,} n_test={r['n_test']:,} features={r['n_features']}")
    print(f"  baseline (no-engine) MAE : {r['baseline_mae']:.1f}")
    print(f"  + engine/clock features  : {r['full_mae']:.1f}  (RMSE {r['full_rmse']:.1f})")
    if "tuned_mae" in r:
        print(f"  + tuned ({r['params_source']:<6})        : {r['tuned_mae']:.1f}  (RMSE {r['tuned_rmse']:.1f})")
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
          f"width {r['mean_interval_width_mondrian']:.0f}  [headline={headline}]")
    print(f"  coverage by band (headline={headline}): "
          + "  ".join(f"{b['band']}={b['coverage']*100:.0f}%" for b in r["coverage_by_band"]))
    pi = r["prior_interval"]
    print(f"  prior (no-game) interval : width {pi['width']:.0f}, coverage {pi['coverage']*100:.1f}%")
    print(f"  band exact / adjacent    : {r['band_exact']*100:.1f}% / {r['band_adjacent']*100:.1f}%")
    sd = r["single_game_deshrink"]
    print(f"  single-game slope true~pred (calib): {sd['slope']:.3f}"
          + (f" [95% CI {sd['slope_lo']:.3f}, {sd['slope_hi']:.3f}]" if "slope_lo" in sd else ""))
    og = r["cqr_one_game_per_player"]
    print(f"  CQR, one game per player : {og['coverage']*100:.1f}% (range {og['coverage_min']*100:.1f}-"
          f"{og['coverage_max']*100:.1f} over {og['draws']} draws), width {og['mean_width']:.0f}")
    if "reservoir_only" in r:
        ro = r["reservoir_only"]
        print(f"  reservoir-only test rows : {ro['n_test']:,} (excl. {ro['n_excluded']:,}) MAE {ro['mae']:.1f}, "
              f"baseline {ro['baseline_mae']:.1f}, coverage {ro['coverage']*100:.1f}%")
    if r["aggregation_curve"]:
        c = r["aggregation_curve"]
        print(f"  aggregation MAE K={c[0]['k']}->{c[-1]['k']} (shifting cohort): {c[0]['mae']:.1f} -> {c[-1]['mae']:.1f}")
    for m in r["matched_aggregation"]:
        print(f"    matched K={m['k']:<2} (n={m['n_players']}): naive {m['naive_mae']:.1f}  recal {m['recal_mae']:.1f}"
              f"  (slope {m['slope']:.2f})")
    ab = r.get("aggregate_bands_k5") or {}
    if ab:
        print(f"  K=5 players (n={ab['n_players']}): single {ab['single_mae']:.1f}  naive {ab['naive_mae']:.1f}"
              f"  recal {ab['recal_mae']:.1f}  (slope {ab['slope']:.2f})")
        for b in ab["bands"]:
            print(f"    {b['band']:<10} n={b['n_players']:>5}  bias single {b['single_bias']:+6.0f}"
                  f"  naive {b['naive_bias']:+6.0f}  recal {b['recal_bias']:+6.0f}")
    print("  residual by band (mean resid = regression to the mean):")
    for rb in r["residual_by_band"]:
        print(f"    {rb['band']:<10} n={rb['n']:>5}  mean_resid {rb['mean_residual']:+7.1f}  MAE {rb['mae']:.0f}")
    print("  ablations (MAE increase when the group is dropped):")
    for a in r["ablations"]:
        print(f"    -{a['group']:<8} (+{a['mae_increase']:.1f})")
    if "scramble_ablation" in r:
        s = r["scramble_ablation"]
        print(f"  scramble block: MAE with {s['with']['overall']:.1f}, without {s['without']['overall']:.1f}")
    pc = r["partner_check"]
    print(f"  partner check (opponent's row in train): {pc['n_seen']:,} rows MAE {pc['mae_seen']:.1f} | "
          f"not {pc['n_unseen']:,} rows MAE {pc['mae_unseen']:.1f}")
    if "bootstrap" in r:
        b = r["bootstrap"]
        print("  bootstrap (player-clustered) 95% CIs:")
        for k, v in b.items():
            if " - " in k:
                print(f"    {k:<22} {_fmt_ci(v)}")


def _append_results_md(path: str | Path, r: dict[str, Any]) -> None:
    label = {"optuna": " (tuned)", "given": " (tuned, given params)"}.get(r.get("params_source"), "")
    lines = [
        f"\n## {r.get('source_commit', current_commit_hash())} — improved model{label}",
        f"n_train={r['n_train']}, n_calib={r['n_calib']}, n_test={r['n_test']}, n_features={r['n_features']}",
        "",
        "| stage | MAE | RMSE |",
        "|---|---|---|",
        f"| no-engine baseline | {r['baseline_mae']:.1f} | — |",
        f"| + engine/clock | {r['full_mae']:.1f} | {r['full_rmse']:.1f} |",
    ]
    if "tuned_mae" in r:
        lines.append(f"| + tuned | {r['tuned_mae']:.1f} | {r['tuned_rmse']:.1f} |")
    if r.get("best_params"):
        # Persist the winning hyperparameters — the Optuna study lives only in this process.
        lines.append(f"\n- best_params: `{r['best_params']}`")
    p, pp = r["point"], r["per_player"]
    sd, og = r["single_game_deshrink"], r["cqr_one_game_per_player"]
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
        f"- no-game (label-quantile) 90% interval: width {r['prior_interval']['width']:.0f} Elo, "
        f"coverage {r['prior_interval']['coverage']*100:.1f}%",
        f"- band accuracy: {r['band_exact']*100:.1f}% exact, {r['band_adjacent']*100:.1f}% adjacent",
        f"- single-game calibration line (true~pred, calib split): slope {sd['slope']:.3f}"
        + (f" [95% CI {sd['slope_lo']:.3f}, {sd['slope_hi']:.3f}]" if "slope_lo" in sd else "")
        + f", intercept {sd['intercept']:.0f}",
        f"- CQR with one game per player (calib and test, {og['draws']} draws): coverage {og['coverage']*100:.1f}% "
        f"(range {og['coverage_min']*100:.1f}-{og['coverage_max']*100:.1f}), width {og['mean_width']:.0f}",
    ]
    if "reservoir_only" in r:
        ro = r["reservoir_only"]
        lines.append(f"- reservoir-only test rows ({ro['n_test']}; {ro['n_excluded']} rows from cohort-only games "
                     f"excluded): MAE {ro['mae']:.1f}, no-engine baseline {ro['baseline_mae']:.1f}, "
                     f"coverage {ro['coverage']*100:.1f}%")
    if r["aggregation_curve"]:
        curve = ", ".join(f"K{c['k']}={c['mae']:.0f}" for c in r["aggregation_curve"])
        lines.append(f"- aggregation MAE (shifting cohort): {curve}")
    if r["matched_aggregation"]:
        m0 = r["matched_aggregation"][0]
        lines.append(
            f"- matched aggregation ({m0['n_players']} test players with >= {max(m['k'] for m in r['matched_aggregation'])} games), "
            "naive -> K-aware recalibrated MAE: "
            + ", ".join(f"K{m['k']}={m['naive_mae']:.0f}->{m['recal_mae']:.0f} (slope {m['slope']:.2f})"
                        for m in r["matched_aggregation"]))
    ab = r.get("aggregate_bands_k5") or {}
    if ab:
        ci = ab.get("naive_minus_recal_ci")
        lines.append(
            f"- K=5 ({ab['n_players']} players with >= 5 games): single {ab['single_mae']:.0f}, naive avg "
            f"{ab['naive_mae']:.0f}, recalibrated {ab['recal_mae']:.0f}"
            + (f" (naive - recal {_fmt_ci(ci)})" if ci else "")
            + "; bias single/naive/recal by band: "
            + ", ".join(f"{b['band']} {b['single_bias']:+.0f}/{b['naive_bias']:+.0f}/{b['recal_bias']:+.0f}"
                        for b in ab["bands"]))
    lines.append("- ablation (MAE↑ when dropped): "
                 + ", ".join(f"{a['group']} +{a['mae_increase']:.1f}" for a in r["ablations"]))
    if "scramble_ablation" in r:
        s = r["scramble_ablation"]
        lines.append(f"- scramble block ({s['n_dropped']} features): overall {s['without']['overall']:.1f} -> "
                     f"{s['with']['overall']:.1f} with it; tails {s['without']['tail']:.0f} -> {s['with']['tail']:.0f}")
    pc = r["partner_check"]
    lines.append(f"- partner check: opponent's row in the training split ({pc['n_seen']} rows) MAE {pc['mae_seen']:.1f} vs "
                 f"not ({pc['n_unseen']} rows) {pc['mae_unseen']:.1f}"
                 + (f"; seen - unseen {_fmt_ci(pc['diff_ci'])}" if "diff_ci" in pc else ""))
    if "bootstrap" in r:
        lines.append("- player-clustered bootstrap, MAE differences: " + "; ".join(
            f"{k} {_fmt_ci(v)}" for k, v in r["bootstrap"].items() if " - " in k))
    _append_md(path, lines)


def main() -> None:
    from src.config import load_config

    cfg = load_config()
    parser = argparse.ArgumentParser(description="Stages 7-8: evaluate the improved model.")
    parser.add_argument("--features", default=cfg["outputs"]["features"])
    parser.add_argument("--games", default=cfg["outputs"]["games_clean"])
    parser.add_argument("--tune", action="store_true", help="run Optuna hyperparameter search")
    parser.add_argument("--params", default=None,
                        help="JSON dict of LightGBM hyperparameters for the 'tuned' row (skips the search)")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    parser.add_argument("--tail-study", action="store_true",
                        help="only compare single-game tail-bias corrections (plain vs reweighted vs de-shrink)")
    parser.add_argument("--results-md", default=cfg["outputs"]["results_log"],
                        help="path to append a run summary to (applies to --tail-study too); pass \"\" to skip logging")
    parser.add_argument("--artifacts", default=cfg["outputs"]["eval_artifacts"],
                        help="structured-results JSON read by the notebooks; pass \"\" to skip")
    parser.add_argument("--residuals-csv", default=cfg["outputs"]["largest_residuals"],
                        help="largest-residual games CSV read by notebook 05; pass \"\" to skip")
    args = parser.parse_args()

    for label, path in (("features", args.features), ("games_clean", args.games)):
        if not Path(path).exists():
            raise SystemExit(f"{label} not found: {path}\nRun the earlier stages first.")
    if args.tail_study:
        run_tail_study(cfg, args.features, args.games, results_md=(args.results_md or None))
        return
    params = None
    if args.params:
        import ast
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError:                 # results.md logs best_params as a Python dict
            params = ast.literal_eval(args.params)
    run_evaluation(cfg, args.features, args.games, tune=args.tune,
                   make_figures=not args.no_figures, results_md=(args.results_md or None),
                   params=params, artifacts_path=(args.artifacts or None),
                   residuals_csv=(args.residuals_csv or None), sample_path=cfg["outputs"]["blitz_sample"])


if __name__ == "__main__":
    main()
