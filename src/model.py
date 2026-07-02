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
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    pre = ColumnTransformer([
        ("num", StandardScaler(), BASELINE_NUM),
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
