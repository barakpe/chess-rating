"""Tests for Stage 5 baseline: metrics, grouped split (no player leak), opponent baseline,
and a small end-to-end evaluate() on synthetic data (also trains a tiny ridge + LightGBM)."""

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")

import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.model import (  # noqa: E402
    add_opponent_rating,
    band_sample_weights,
    evaluate,
    grouped_split,
    mae,
    rmse,
)


def test_metrics():
    y = np.array([1000.0, 2000.0])
    assert mae(y, np.array([1100.0, 1800.0])) == pytest.approx(150.0)
    assert rmse(y, np.array([1000.0, 2000.0])) == pytest.approx(0.0)


def test_grouped_split_has_no_player_overlap():
    df = pd.DataFrame({"username": [f"u{i//3}" for i in range(30)], "x": range(30)})
    train, test = grouped_split(df, test_size=0.3, seed=42)
    assert set(train["username"]) & set(test["username"]) == set()
    assert len(train) + len(test) == 30


def test_add_opponent_rating():
    features = pd.DataFrame({
        "game_id": ["g1", "g1"], "color": ["white", "black"],
        "rating": [1600, 1500], "username": ["a", "b"],
    })
    games = pd.DataFrame({"game_id": ["g1"], "white_elo": [1600], "black_elo": [1500]})
    out = add_opponent_rating(features, games)
    # White's opponent is Black (1500); Black's opponent is White (1600). No own-rating columns leak.
    assert out.loc[out.color == "white", "opponent_rating"].iloc[0] == 1500
    assert out.loc[out.color == "black", "opponent_rating"].iloc[0] == 1600
    assert "white_elo" not in out.columns and "black_elo" not in out.columns


def test_band_sample_weights_upweights_tails():
    bands = [0, 1200, 1400, 1600, 1800, 2000, 3000]
    ratings = np.array([1500] * 8 + [500, 2500])  # 8 mid, 1 low tail, 1 high tail
    w = band_sample_weights(ratings, bands, strength=1.0)
    assert w.mean() == pytest.approx(1.0)          # mean-normalised
    assert w[-1] > w[0] and w[-2] > w[0]           # rare tail bands weigh more than the common one


def _synthetic(n_players=24, per_player=4, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for p in range(n_players):
        rating = 800 + p * 80  # spread 800..2640
        for gi in range(per_player):
            rows.append({
                "game_id": f"g{p}_{gi}", "color": "white" if gi % 2 else "black",
                "username": f"user{p}", "rating": rating,
                "opponent_rating": rating + rng.randint(-40, 40),  # close pairing
                "game_plies": 40 + rng.randint(-10, 10), "player_moves": 20,
                "n_captures": rng.randint(0, 10), "n_checks": rng.randint(0, 4),
                "result": rng.choice(["win", "loss", "draw"]),
                "time_control": "300+0", "eco": rng.choice(["C50", "B20", "D30"]),
            })
    return pd.DataFrame(rows)


def test_evaluate_runs_and_ranks_baselines():
    cfg = load_config()
    cfg["model"]["lgbm"]["n_estimators"] = 20  # keep the test fast
    results = evaluate(_synthetic(), cfg)

    for name in ("predict_mean", "copy_opponent", "ridge", "lightgbm"):
        assert np.isfinite(results[name]["mae"]) and np.isfinite(results[name]["rmse"])

    # opponent rating was constructed within +/-40 of the label, so copy_opponent must crush
    # the trivial predict-the-mean floor.
    assert results["copy_opponent"]["mae"] < results["predict_mean"]["mae"]
