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
    _residual_by_length,
    aggregate_band_table,
    aggregation_curve,
    cluster_bootstrap_mae,
    fit_aggregate_recalibration,
    matched_aggregation,
    one_game_per_player_cqr,
    slope_bootstrap_ci,
    partner_exposure,
    prior_interval,
    scramble_columns,
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
        games.append({"game_id": f"g{gi}", "white": w, "black": b, "white_elo": we, "black_elo": be})
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

    art, csv = tmp_path / "eval_artifacts.json", tmp_path / "largest_residuals.csv"
    r = run_evaluation(cfg, fp, gp, tune=False, make_figures=False, results_md=None,
                       artifacts_path=art, residuals_csv=csv)

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

    # --- new diagnostics are present and well-formed ---
    assert set(r["partner_check"]) >= {"n_seen", "n_unseen", "mae_seen", "mae_unseen"}
    assert r["partner_check"]["n_seen"] + r["partner_check"]["n_unseen"] == r["n_test"]
    assert 0.0 <= r["prior_interval"]["coverage"] <= 1.0
    assert "baseline - full" in r["bootstrap"]
    diag = r["mondrian_diagnostics"]
    assert sum(map(sum, diag["test_true_by_predicted_band"])) == r["n_test"]
    assert sum(diag["calib_rows_by_predicted_band"]) == r["n_calib"]

    # --- the notebook-facing artifacts are written and loadable; no usernames in the CSV ---
    import json
    a = json.loads(art.read_text(encoding="utf-8"))
    for key in ("full_mae", "baseline_mae", "point", "coverage_cqr_plain", "aggregation_curve",
                "residual_by_band", "ablations", "mondrian_corrections", "source_commit"):
        assert key in a
    lr = pd.read_csv(csv)
    assert "username" not in lr.columns and "game_id" not in lr.columns and len(lr) > 0
    assert 0.0 <= r["cqr_one_game_per_player"]["coverage"] <= 1.0
    assert r["single_game_deshrink"]["slope_lo"] <= r["single_game_deshrink"]["slope_hi"]


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


def test_cluster_bootstrap_mae_ci_and_paired_diff():
    rng = np.random.RandomState(0)
    groups = np.repeat(np.arange(200), 3)
    err_a = np.abs(rng.normal(0, 100, len(groups)))
    ci = cluster_bootstrap_mae({"a": err_a, "b": err_a.copy(), "c": err_a + 10}, groups, 200, 0,
                               diffs=[("a", "b"), ("c", "a")])
    assert ci["a"]["lo"] <= ci["a"]["mae"] <= ci["a"]["hi"]
    assert ci["a - b"]["diff"] == pytest.approx(0.0) and ci["a - b"]["lo"] == pytest.approx(0.0)
    assert ci["c - a"]["lo"] == pytest.approx(10.0) and ci["c - a"]["hi"] == pytest.approx(10.0)


def test_cluster_bootstrap_mae_masks():
    groups = np.array([0, 0, 1, 1])
    err = np.array([1.0, 3.0, 5.0, 7.0])
    mask = np.array([True, False, True, False])
    ci = cluster_bootstrap_mae({"all": err, "sub": err}, groups, 50, 0, masks={"sub": mask})
    assert ci["all"]["mae"] == pytest.approx(4.0)
    assert ci["sub"]["mae"] == pytest.approx(3.0)       # mean of rows 0 and 2


def _shrunk_players(n_players, games, seed, shrink=0.5, noise=300.0):
    """Players whose single-game prediction is a calibrated-but-shrunk estimate:
    pred = m + shrink * (rating + noise - m). Averaging such predictions keeps the shrinkage."""
    rng = np.random.RandomState(seed)
    rows = []
    for u in range(n_players):
        r = rng.normal(1600, 400)
        for _ in range(games):
            rows.append({"username": f"u{seed}_{u}", "rating": r,
                         "pred": 1600 + shrink * (r + rng.normal(0, noise) - 1600)})
    return pd.DataFrame(rows)


def test_aggregate_recalibration_undoes_averaged_shrinkage():
    calib, test = _shrunk_players(400, 10, 1), _shrunk_players(400, 10, 2)
    players = [(g["rating"].to_numpy(), g["pred"].to_numpy()) for _, g in calib.groupby("username")]
    slope1, _ = fit_aggregate_recalibration(players, 1, 5, 0)
    slope10, _ = fit_aggregate_recalibration(players, 10, 1, 0)
    assert slope10 > slope1 > 1.0                          # averaged preds are MORE under-dispersed

    curve = matched_aggregation(test, calib, [1, 5, 10], draws=5, seed=0)
    assert [c["k"] for c in curve] == [1, 5, 10]
    k10 = curve[-1]
    assert k10["recal_mae"] < 0.6 * k10["naive_mae"]       # recalibration removes most of the bias
    assert curve[0]["naive_mae"] > k10["naive_mae"]        # averaging still helps the naive estimate

    table = aggregate_band_table(test, calib, 5, [0, 1400, 1800, 3000], draws=5, seed=0, reps=50)
    low, high = table["bands"][0], table["bands"][-1]
    assert low["naive_bias"] > 100 and high["naive_bias"] < -100          # shrinkage survives averaging
    assert abs(low["recal_bias"]) < abs(low["naive_bias"]) / 2
    assert abs(high["recal_bias"]) < abs(high["naive_bias"]) / 2
    assert table["naive_minus_recal_ci"]["lo"] > 0


def test_matched_aggregation_empty_when_no_player_has_enough_games():
    df = _shrunk_players(10, 3, 0)
    assert matched_aggregation(df, df, [1, 5], draws=2, seed=0) == []
    assert aggregate_band_table(df, df, 5, [0, 1600, 3000], draws=2, seed=0) == {}
    # enough games, but fewer players than min_players -> not reported
    df10 = _shrunk_players(10, 10, 0)
    assert matched_aggregation(df10, df10, [1, 5], draws=2, seed=0, min_players=30) == []
    assert aggregate_band_table(df10, df10, 5, [0, 1600, 3000], draws=2, seed=0, min_players=30) == {}


def test_partner_exposure_and_prior_interval_and_helpers():
    test = pd.DataFrame({"game_id": ["g1", "g2", "g3"]})
    assert list(partner_exposure(test, {"g2", "g9"})) == [False, True, False]

    pi = prior_interval(np.arange(1, 101), np.array([0, 50, 200]), [0.05, 0.5, 0.95])
    assert pi["lower"] == pytest.approx(5.95) and pi["upper"] == pytest.approx(95.05)
    assert pi["coverage"] == pytest.approx(1 / 3)

    cols = scramble_columns(["cpl_mean", "scramble_cpl_mean", "has_scramble"])
    assert cols == ["scramble_cpl_mean", "has_scramble"]

    rows = _residual_by_length(np.array([10, 25, 25, 80]), np.array([100.0, -50.0, 50.0, 10.0]))
    assert [r["moves"] for r in rows] == ["0-19", "20-29", "60+"]
    assert rows[1]["mae"] == pytest.approx(50.0) and rows[1]["mean_residual"] == pytest.approx(0.0)


def test_spearman_gives_tied_values_their_average_rank():
    # standard Spearman of [1, 2, 3] vs [1, 1, 2] is 0.866; arbitrary tie-breaking would give 1.0
    assert point_metrics([1, 2, 3], [1, 1, 2])["spearman"] == pytest.approx(np.sqrt(3) / 2)


def test_slope_bootstrap_ci_brackets_the_true_slope():
    rng = np.random.RandomState(0)
    groups = np.repeat(np.arange(300), 4)
    x = rng.normal(1500, 300, len(groups))
    y = 0.8 * x + 300 + rng.normal(0, 50, len(groups)) + np.repeat(rng.normal(0, 30, 300), 4)
    ci = slope_bootstrap_ci(x, y, groups, reps=200, seed=0)
    assert ci["slope_lo"] < 0.8 < ci["slope_hi"]
    assert ci["intercept_lo"] < 300 < ci["intercept_hi"]


def test_one_game_per_player_cqr_uses_one_row_per_player_and_reaches_the_target():
    rng = np.random.RandomState(1)

    def players(n, k):
        users = np.repeat([f"p{i}" for i in range(n)], k)
        y = np.repeat(rng.normal(1500, 300, n), k) + rng.normal(0, 100, n * k)
        interval = {"lower": y * 0 + 1500 - 200, "median": y * 0 + 1500, "upper": y * 0 + 1500 + 200}
        return interval, y, users

    ic, yc, uc = players(2000, 3)
    it, yt, ut = players(2000, 3)
    r = one_game_per_player_cqr(ic, yc, uc, it, yt, ut, alpha=0.1, two_sided=False, draws=5, seed=0)
    assert r["n_calib_players"] == 2000 and r["n_test_players"] == 2000
    assert 0.87 < r["coverage"] < 0.93
    assert r["coverage_min"] <= r["coverage"] <= r["coverage_max"]
