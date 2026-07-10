"""Tests for Stage 7/8: metric helpers, feature grouping, aggregation curve, and a tiny
end-to-end run_evaluation (trains small models, no figures)."""

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")

import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.evaluate import (  # noqa: E402
    _coverage_by_band,
    aggregation_curve,
    apply_deshrink,
    band_accuracy,
    feature_groups,
    fit_deshrink,
    interval_coverage,
    mean_interval_width,
    per_player_metrics,
    pinball_loss,
    point_metrics,
    run_evaluation,
    run_tail_study,
    to_bands,
)


def test_interval_coverage_and_width():
    y = np.array([1000.0, 1500.0, 2000.0])
    lo = np.array([900.0, 1600.0, 1900.0])   # middle point falls outside
    up = np.array([1100.0, 1700.0, 2100.0])
    assert interval_coverage(y, lo, up) == pytest.approx(2 / 3)
    assert mean_interval_width(lo, up) == pytest.approx((200 + 100 + 200) / 3)


def test_pinball_loss():
    # alpha=0.5 -> half the absolute error
    assert pinball_loss([10.0], [8.0], 0.5) == pytest.approx(1.0)
    # under-prediction penalised by alpha, over-prediction by (1-alpha)
    assert pinball_loss([10.0], [8.0], 0.9) == pytest.approx(0.9 * 2)
    assert pinball_loss([10.0], [12.0], 0.9) == pytest.approx(0.1 * 2)


def test_bands_and_accuracy():
    bands = [0, 1200, 1400, 1600, 1800, 2000, 3000]
    assert to_bands([1150, 1250, 2500], bands).tolist() == [0, 1, 5]
    y = np.array([1250, 1250, 1250])          # all band 1 ([1200,1400))
    yhat = np.array([1250, 1500, 1900])       # band 1 (exact), band 2 (adjacent), band 4 (far)
    exact, adjacent = band_accuracy(y, yhat, bands)
    assert exact == pytest.approx(1 / 3)
    assert adjacent == pytest.approx(2 / 3)


def test_coverage_by_band():
    bands = [0, 1200, 1400, 1600, 1800, 2000, 3000]
    # band 0 ([0,1200)): two points, one covered, one not -> coverage 0.5
    # band 5 ([2000,3000)): two points, both covered -> coverage 1.0
    y = np.array([1000.0, 1100.0, 2500.0, 2600.0])
    lo = np.array([950.0, 1150.0, 2400.0, 2550.0])   # second band-0 point falls outside [lo,up]
    up = np.array([1050.0, 1200.0, 2600.0, 2650.0])
    out = _coverage_by_band(y, lo, up, bands)
    by_band = {d["band"]: d for d in out}
    assert set(by_band.keys()) == {"0-1200", "2000-3000"}     # empty bands are omitted
    assert by_band["0-1200"]["n"] == 2
    assert by_band["0-1200"]["coverage"] == pytest.approx(0.5)
    assert by_band["0-1200"]["mean_width"] == pytest.approx((100 + 50) / 2)
    assert by_band["2000-3000"]["n"] == 2
    assert by_band["2000-3000"]["coverage"] == pytest.approx(1.0)
    assert by_band["2000-3000"]["mean_width"] == pytest.approx((200 + 100) / 2)


def test_point_metrics():
    m = point_metrics([1000, 2000], [1100, 1800])  # resid = [+100, -200]
    assert m["mae"] == pytest.approx(150.0)
    assert m["median_ae"] == pytest.approx(150.0)
    assert m["r2"] == pytest.approx(0.9)
    assert m["bias"] == pytest.approx(-50.0)          # net under-prediction
    assert m["within_100"] == pytest.approx(0.5) and m["within_200"] == pytest.approx(1.0)
    assert m["pearson"] == pytest.approx(1.0) and m["spearman"] == pytest.approx(1.0)


def test_per_player_metrics():
    df = pd.DataFrame({
        "username": ["a", "a", "b"],
        "rating": [1500.0, 1500.0, 1800.0],
        "pred": [1500.0, 1500.0, 1700.0],
    })
    m = per_player_metrics(df)                        # a: err 0 (2 games), b: err 100 (1 game)
    assert m["n_players"] == 2
    assert m["mae"] == pytest.approx(50.0)
    assert m["multi_game_share"] == pytest.approx(0.5)


def test_deshrink_recovers_shrunk_predictions():
    rng = np.random.RandomState(0)
    true = rng.uniform(800, 2400, 500)
    pred = 1500 + 0.5 * (true - 1500)       # shrunk 50% toward the centre
    coef = fit_deshrink(pred, true)
    assert coef[0] == pytest.approx(2.0, rel=1e-6)   # slope true~pred = 1/0.5
    assert np.allclose(apply_deshrink(pred, coef), true, atol=1e-6)


def test_feature_groups_partition():
    cols = ["cpl_mean", "opening_cpl_mean", "acc_after_book", "move_time_mean",
            "fast_move_share", "eco", "game_plies", "n_captures", "color"]
    g = feature_groups(cols)
    assert "cpl_mean" in g["engine"] and "acc_after_book" in g["engine"]
    assert "move_time_mean" in g["clock"] and "fast_move_share" in g["clock"]
    assert g["opening"] == ["eco"]
    assert set(g["style"]) == {"game_plies", "n_captures", "color"}


def test_aggregation_curve():
    # two players, 3 games each; predictions equal the rating -> zero error at every K.
    df = pd.DataFrame({
        "username": ["a"] * 3 + ["b"] * 3,
        "rating": [1500.0] * 3 + [1800.0] * 3,
        "pred": [1500.0] * 3 + [1800.0] * 3,
    })
    curve = aggregation_curve(df, ks=[1, 2, 3], min_players=2, seed=0)
    assert [c["k"] for c in curve] == [1, 2, 3]
    assert all(c["mae"] == pytest.approx(0.0) for c in curve)
    assert all(c["n_players"] == 2 for c in curve)


def _synthetic_parquets(tmp_path, n_games=150, n_players=48, seed=0):
    # enough distinct usernames that the 4-way grouped split (test/val/calib carved off train)
    # leaves every one of the four splits non-empty.
    rng = np.random.RandomState(seed)
    players = [f"user{i}" for i in range(n_players)]
    ratings = {p: 900 + i * 70 for i, p in enumerate(players)}
    feat, games = [], []
    for gi in range(n_games):
        w, b = rng.choice(players, 2, replace=False)
        we, be = ratings[w], ratings[b]
        games.append({"game_id": f"g{gi}", "white_elo": we, "black_elo": be})
        for color, user, rt in (("white", w, we), ("black", b, be)):
            cpl = 80 - rt * 0.02 + rng.randn() * 4
            feat.append({
                "game_id": f"g{gi}", "color": color, "username": user, "rating": rt,
                "result": rng.choice(["win", "loss", "draw"]), "time_control": "300+0",
                "eco": rng.choice(["C50", "B20"]), "opening": "X",
                "cpl_mean": cpl, "opening_cpl_mean": cpl + rng.randn() * 2,
                "acc_mean": 90 - cpl * 0.05, "blunder_rate": max(0.0, 0.1 - rt * 3e-5),
                "move_time_mean": 3.0, "game_plies": 40, "player_moves": 20,
                "n_captures": 5, "n_checks": 1,
            })
    fp = tmp_path / "features.parquet"
    gp = tmp_path / "games_clean.parquet"
    pd.DataFrame(feat).to_parquet(fp, index=False)
    pd.DataFrame(games).to_parquet(gp, index=False)
    return fp, gp


def test_run_evaluation_end_to_end(tmp_path):
    cfg = load_config()
    cfg["model"]["lgbm"]["n_estimators"] = 30  # keep training fast
    cfg["evaluate"]["aggregation_min_players"] = 2
    cfg["evaluate"]["aggregation_k"] = [1, 2]
    fp, gp = _synthetic_parquets(tmp_path)

    r = run_evaluation(cfg, fp, gp, tune=False, make_figures=False, results_md=None)

    assert np.isfinite(r["baseline_mae"]) and np.isfinite(r["full_mae"])
    assert r["n_calib"] > 0
    assert 0.0 <= r["coverage_raw"] <= 1.0
    assert 0.0 <= r["coverage"] <= 1.0
    assert len(r["conformal_correction"]) == 2
    assert all(np.isfinite(v) for v in r["conformal_correction"])
    assert r["coverage_by_band"]                       # non-empty
    for b in r["coverage_by_band"]:
        assert set(b.keys()) == {"band", "n", "coverage", "mean_width"}
        assert b["n"] > 0
        assert 0.0 <= b["coverage"] <= 1.0
    assert 0.0 <= r["band_exact"] <= 1.0 and r["band_adjacent"] >= r["band_exact"]
    assert len(r["ablations"]) >= 1
    assert set(cfg["model"]["quantiles"]) == set(r["pinball"].keys())

    # --- Mondrian (band-conditional) CQR: computed alongside plain CQR, headline when enabled ---
    assert 0.0 <= r["coverage_cqr_plain"] <= 1.0
    assert 0.0 <= r["coverage_cqr_mondrian"] <= 1.0
    assert r["coverage_by_band_mondrian"]               # non-empty
    for b in r["coverage_by_band_mondrian"]:
        assert set(b.keys()) == {"band", "n", "coverage", "mean_width"}
        assert b["n"] > 0
        assert 0.0 <= b["coverage"] <= 1.0
    assert len(r["mondrian_corrections"]) == len(cfg["rating_bands"]) - 1   # every band covered

    # the loaded config has model.mondrian_cqr: true -> Mondrian must be the headline interval.
    assert cfg["model"]["mondrian_cqr"] is True
    assert r["mondrian_headline"] is True
    assert r["coverage"] == pytest.approx(r["coverage_cqr_mondrian"])
    assert r["mean_interval_width"] == pytest.approx(r["mean_interval_width_mondrian"])
    assert r["coverage_by_band"] == r["coverage_by_band_mondrian"]


def test_run_tail_study_smoke(tmp_path):
    cfg = load_config()
    cfg["model"]["lgbm"]["n_estimators"] = 30  # keep training fast
    fp, gp = _synthetic_parquets(tmp_path)
    results_md = tmp_path / "results.md"

    variants = run_tail_study(cfg, fp, gp, results_md=results_md)

    assert len(variants) == 3
    for tbl in variants.values():
        assert np.isfinite(tbl["overall"])
        assert "tail" in tbl and "mid" in tbl

    text = results_md.read_text(encoding="utf-8")
    assert "tail study" in text
    for name in variants:
        assert name in text
