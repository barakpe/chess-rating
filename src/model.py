"""Stage 5 — Baseline model.

Predict a player's rating from LIGHTWEIGHT, no-engine features only (game shape + opening +
result), to establish an honest floor before the engine/clock features go in (Stage 6). Reports
MAE/RMSE against two references:

- ``predict_mean``  : predict the training-mean rating for everyone — the trivial floor.
- ``copy_opponent`` : predict the OPPONENT's rating — a deliberately strong baseline, because
  Lichess pairs similar ratings. It is the "cheating" baseline we excluded from the model: showing
  it is strong is exactly why opponent rating must never be a feature (it leaks the label). The
  opponent rating is reconstructed here from games_clean FOR COMPARISON ONLY and is never fed to a model.

Evaluation splits by ``username`` (grouped hold-out), never by game — otherwise the model can
memorise a player and the score is fake (PROJECT_ARCHITECTURE.md §Stage 7). Note: a single game's
two instances have different usernames, so game-level context (eco/time_control/length) can straddle
the split; that is acceptable — the thing we must not leak is a *player*, not a game.

Run:
    python -m src.model            # reads features.parquet + games_clean.parquet from config
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Lightweight baseline features: NO engine (centipawn/accuracy) and NO clock features — those are
# reserved for the improved model (Stage 6). Style/shape + opening + own result only.
BASELINE_NUM = ["game_plies", "player_moves", "n_captures", "n_checks"]
BASELINE_CAT = ["result", "time_control", "eco"]
BASELINE_FEATURES = BASELINE_NUM + BASELINE_CAT


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}


def add_opponent_rating(features_df: pd.DataFrame, games_clean_df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``opponent_rating`` (for the copy-opponent baseline ONLY) from games_clean.

    This column is a demonstration reference, NOT a model feature — models train on BASELINE_FEATURES.
    """
    elos = games_clean_df[["game_id", "white_elo", "black_elo"]]
    merged = features_df.merge(elos, on="game_id", how="left")
    merged["opponent_rating"] = np.where(
        merged["color"] == "white", merged["black_elo"], merged["white_elo"]
    )
    return merged.drop(columns=["white_elo", "black_elo"])


def grouped_split(
    df: pd.DataFrame, test_size: float, seed: int, group_col: str = "username"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grouped hold-out: no player (``group_col``) appears in both train and test."""
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[train_idx], df.iloc[test_idx]


def make_ridge(alpha: float):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),  # future-proof: engine features can be NaN
        ("scale", StandardScaler()),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, BASELINE_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), BASELINE_CAT),
    ])
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))])


def make_lgbm(cfg: dict[str, Any]):
    import lightgbm as lgb

    p = cfg["model"]["lgbm"]
    return lgb.LGBMRegressor(
        objective="regression_l1",  # MAE — matches our headline metric
        n_estimators=p["n_estimators"],
        learning_rate=p["learning_rate"],
        num_leaves=p["num_leaves"],
        random_state=cfg["random_seed"],
        n_jobs=-1,
        verbose=-1,
    )


def evaluate(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Train baselines + ridge + LightGBM on a grouped split; return {model: {mae, rmse}}.

    ``df`` must contain BASELINE_FEATURES, ``rating``, ``username``, and ``opponent_rating``.
    """
    df = df.copy()
    # Fix categorical dtypes on the FULL frame BEFORE splitting so train/test share category codes
    # (LightGBM keys on the pandas category codes — mismatched mappings would corrupt predictions).
    for col in BASELINE_CAT:
        df[col] = df[col].astype("category")

    train, test = grouped_split(df, cfg["model"]["test_size"], cfg["random_seed"])
    y_tr = train["rating"].to_numpy(float)
    y_te = test["rating"].to_numpy(float)

    opp = test["opponent_rating"].to_numpy(float)
    opp = np.where(np.isnan(opp), y_tr.mean(), opp)  # guard the rare missing opponent

    results: dict[str, dict[str, float]] = {
        "predict_mean": _metrics(y_te, np.full_like(y_te, y_tr.mean())),
        "copy_opponent": _metrics(y_te, opp),
    }

    x_tr, x_te = train[BASELINE_FEATURES], test[BASELINE_FEATURES]
    ridge = make_ridge(cfg["model"]["ridge_alpha"]).fit(x_tr, y_tr)
    results["ridge"] = _metrics(y_te, ridge.predict(x_te))

    lgbm = make_lgbm(cfg).fit(x_tr, y_tr)
    results["lightgbm"] = _metrics(y_te, lgbm.predict(x_te))

    results["_meta"] = {"n_train": float(len(train)), "n_test": float(len(test))}
    return results


def run_baseline(
    features_path: str | Path,
    games_clean_path: str | Path,
    cfg: dict[str, Any],
    results_md: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    """Load data, attach opponent rating, evaluate, print, and optionally log to results.md."""
    features = pd.read_parquet(features_path)
    games_clean = pd.read_parquet(games_clean_path)
    df = add_opponent_rating(features, games_clean)
    results = evaluate(df, cfg)
    _print_results(results)
    if results_md is not None:
        _append_results_md(results_md, results)
    return results


_ORDER = ["predict_mean", "copy_opponent", "ridge", "lightgbm"]


def _print_results(results: dict[str, dict[str, float]]) -> None:
    meta = results.get("_meta", {})
    print(f"\nBaseline (no-engine features) - n_train={int(meta.get('n_train', 0)):,} "
          f"n_test={int(meta.get('n_test', 0)):,}")
    print(f"  {'model':<16}{'MAE':>10}{'RMSE':>10}")
    for name in _ORDER:
        if name in results:
            print(f"  {name:<16}{results[name]['mae']:>10.1f}{results[name]['rmse']:>10.1f}")


def _append_results_md(path: str | Path, results: dict[str, dict[str, float]]) -> None:
    from datetime import datetime

    meta = results.get("_meta", {})
    lines = [
        f"\n## {datetime.now():%Y-%m-%d %H:%M} — baseline (no-engine features)",
        f"n_train={int(meta.get('n_train', 0))}, n_test={int(meta.get('n_test', 0))}, "
        f"features={', '.join(BASELINE_FEATURES)}",
        "",
        "| model | MAE | RMSE |",
        "|---|---|---|",
    ]
    lines += [f"| {n} | {results[n]['mae']:.1f} | {results[n]['rmse']:.1f} |"
              for n in _ORDER if n in results]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# =======================================================================================
# Stage 6 — Improved model: full (engine + clock) features, early stopping, tuning, intervals
# =======================================================================================

# Columns that are identifiers/labels/leaks, never fed to the model.
_NON_FEATURE = {"game_id", "username", "opponent_rating", "rating", "opening"}
# Categorical features (low/medium cardinality); everything else numeric is used as-is.
FULL_CAT = ["result", "time_control", "eco", "color"]


def split_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features) for the full (engine+clock) model."""
    cat = [c for c in FULL_CAT if c in df.columns]
    num = [
        c for c in df.columns
        if c not in _NON_FEATURE and c not in cat and pd.api.types.is_numeric_dtype(df[c])
    ]
    return num, cat


def prepare_features(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Fix categorical dtypes so LightGBM's category codes are stable across splits."""
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df


def grouped_train_val_test(
    df: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Grouped (by username) train/val/test — val is carved from train for early stopping."""
    seed = cfg["random_seed"]
    train_full, test = grouped_split(df, cfg["model"]["test_size"], seed)
    train, val = grouped_split(train_full, cfg["model"]["val_size"], seed + 1)
    return train, val, test


def grouped_train_val_calib_test(
    df: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Grouped (by username) train/val/calib/test.

    val = early stopping only. calib = conformal/recalibration only — never seen by any fit.
    test = final report. No username appears in more than one of the four splits.
    """
    seed = cfg["random_seed"]
    train_full, test = grouped_split(df, cfg["model"]["test_size"], seed)
    train_mid, val = grouped_split(train_full, cfg["model"]["val_size"], seed + 1)
    train, calib = grouped_split(train_mid, cfg["model"]["calib_size"], seed + 2)
    return train, val, calib, test


def make_point_model(cfg: dict[str, Any], params: dict[str, Any] | None = None):
    import lightgbm as lgb

    lp = cfg["model"]["lgbm"]
    kwargs = {
        "objective": "regression_l1", "n_estimators": lp["n_estimators"],
        "learning_rate": lp["learning_rate"], "num_leaves": lp["num_leaves"],
        "random_state": cfg["random_seed"], "n_jobs": -1, "verbose": -1,
    }
    if params:
        kwargs.update(params)
    return lgb.LGBMRegressor(**kwargs)


def make_quantile_model(cfg: dict[str, Any], alpha: float, params: dict[str, Any] | None = None):
    import lightgbm as lgb

    lp = cfg["model"]["lgbm"]
    kwargs = {
        "objective": "quantile", "alpha": alpha, "n_estimators": lp["n_estimators"],
        "learning_rate": lp["learning_rate"], "num_leaves": lp["num_leaves"],
        "random_state": cfg["random_seed"], "n_jobs": -1, "verbose": -1,
    }
    if params:
        kwargs.update(params)
    kwargs["objective"], kwargs["alpha"] = "quantile", alpha  # never let params override these
    return lgb.LGBMRegressor(**kwargs)


def fit_early_stopping(model, x_tr, y_tr, x_val, y_val, cfg, eval_metric="l1", sample_weight=None):
    import lightgbm as lgb

    model.fit(
        x_tr, y_tr, sample_weight=sample_weight, eval_set=[(x_val, y_val)], eval_metric=eval_metric,
        callbacks=[lgb.early_stopping(cfg["model"]["early_stopping_rounds"], verbose=False),
                   lgb.log_evaluation(0)],
    )
    return model


def band_sample_weights(ratings: np.ndarray, bands: list[int], strength: float = 1.0) -> np.ndarray:
    """Per-instance training weights that up-weight rare rating bands (mean-normalised to 1).

    The natural rating distribution is bell-shaped, so an MAE learner shrinks the tails toward the
    centre. Weighting each instance by ``(1 / band_frequency) ** strength`` makes the model value
    the tails more, trading a little central accuracy for less tail bias. ``strength=0`` -> uniform.
    """
    idx = np.digitize(np.asarray(ratings, float), bands[1:-1])
    counts = np.bincount(idx, minlength=len(bands) - 1).astype(float)
    counts[counts == 0] = 1.0
    freq = counts[idx] / counts.sum()
    weights = (1.0 / freq) ** strength
    return weights / weights.mean()


def train_quantile_models(x_tr, y_tr, x_val, y_val, cfg, params=None) -> dict[float, Any]:
    """One LightGBM per quantile (alpha) — you cannot get multiple quantiles from one model."""
    models = {}
    for alpha in cfg["model"]["quantiles"]:
        model = make_quantile_model(cfg, alpha, params)
        fit_early_stopping(model, x_tr, y_tr, x_val, y_val, cfg, eval_metric="quantile")
        models[alpha] = model
    return models


def predict_interval(models: dict[float, Any], x, quantiles: list[float]) -> dict[str, np.ndarray]:
    """Predict each quantile then de-cross by sorting the per-row predictions ascending."""
    preds = np.column_stack([models[a].predict(x) for a in quantiles])
    preds = np.sort(preds, axis=1)  # guarantees lower <= median <= upper
    return {"lower": preds[:, 0], "median": preds[:, len(quantiles) // 2], "upper": preds[:, -1]}


def _conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Finite-sample conformal quantile of ``scores`` at ``level`` (Romano-Patterson-Candes 2019).

    k = ceil((n + 1) * level)-th smallest score; degenerate (k > n, tiny calibration sets only)
    falls back to the max.
    """
    import math

    scores = np.sort(np.asarray(scores, float))
    n = len(scores)
    k = math.ceil((n + 1) * level)
    if k > n:
        return float(scores[-1])  # degenerate only for tiny calibration sets
    return float(scores[k - 1])


def conformal_correction(
    interval: dict[str, np.ndarray], y: np.ndarray, alpha: float, two_sided: bool = False
) -> tuple[float, float]:
    """Split-CQR correction (lo, hi) from a calibration-set ``interval`` (predict_interval output).

    Symmetric (two_sided=False): one nonconformity score E = max(lower - y, y - upper), corrected
    by its (1 - alpha) conformal quantile applied equally to both sides. Two-sided: separate lower/
    upper nonconformity scores, each corrected at (1 - alpha/2). Corrections may be NEGATIVE — if the
    raw interval over-covers on the calibration set, CQR legitimately narrows it.
    """
    y = np.asarray(y, float)
    if two_sided:
        lo_scores = interval["lower"] - y
        hi_scores = y - interval["upper"]
        q_lo = _conformal_quantile(lo_scores, 1 - alpha / 2)
        q_hi = _conformal_quantile(hi_scores, 1 - alpha / 2)
        return q_lo, q_hi
    e = np.maximum(interval["lower"] - y, y - interval["upper"])
    q = _conformal_quantile(e, 1 - alpha)
    return q, q


def apply_conformal(
    interval: dict[str, np.ndarray], correction: tuple[float, float]
) -> dict[str, np.ndarray]:
    """Apply a (lo, hi) conformal correction to ``interval``; returns a NEW dict, median unchanged.

    Clips so ordering survives a negative correction (over-covering raw interval narrowed by CQR).
    """
    lo, hi = correction
    median = interval["median"]
    lower = np.minimum(interval["lower"] - lo, median)
    upper = np.maximum(interval["upper"] + hi, median)
    return {"lower": lower, "median": median, "upper": upper}


def tune_lgbm(x, y, groups, cfg) -> dict[str, Any]:
    """Optuna TPE search minimising grouped-CV MAE. Returns the best hyperparameters."""
    import lightgbm as lgb
    import optuna
    from sklearn.model_selection import GroupKFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    y = np.asarray(y, float)

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 120),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        gkf = GroupKFold(n_splits=cfg["model"]["tune"]["cv_folds"])
        scores = []
        for tr, va in gkf.split(x, y, groups):
            model = make_point_model(cfg, params)
            model.fit(
                x.iloc[tr], y[tr], eval_set=[(x.iloc[va], y[va])], eval_metric="l1",
                callbacks=[lgb.early_stopping(cfg["model"]["early_stopping_rounds"], verbose=False),
                           lgb.log_evaluation(0)],
            )
            scores.append(mae(y[va], model.predict(x.iloc[va])))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg["random_seed"])
    )
    study.optimize(objective, n_trials=cfg["model"]["tune"]["n_trials"], show_progress_bar=False)
    return study.best_params


def main() -> None:
    from src.config import load_config

    cfg = load_config()
    parser = argparse.ArgumentParser(description="Stage 5: baseline rating model.")
    parser.add_argument("--features", default=cfg["outputs"]["features"], help="features parquet")
    parser.add_argument("--games", default=cfg["outputs"]["games_clean"], help="games_clean parquet")
    parser.add_argument("--results-md", default=str(Path(cfg["paths"]["figures"]).parent / "results.md"),
                        help="experiment-log markdown to append to (use '' to skip)")
    args = parser.parse_args()

    for label, path in (("features", args.features), ("games_clean", args.games)):
        if not Path(path).exists():
            raise SystemExit(f"{label} not found: {path}\nRun the earlier stages first.")
    run_baseline(args.features, args.games, cfg, results_md=(args.results_md or None))


if __name__ == "__main__":
    main()
