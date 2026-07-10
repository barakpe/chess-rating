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
    _conformal_quantile,
    add_opponent_rating,
    apply_conformal,
    apply_conformal_by_band,
    band_sample_weights,
    conformal_correction,
    conformal_correction_by_band,
    evaluate,
    grouped_split,
    grouped_train_val_calib_test,
    make_ridge,
    mae,
    predict_interval,
    predicted_bands,
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


def test_grouped_train_val_calib_test_no_overlap():
    cfg = load_config()
    df = _synthetic(n_players=40, per_player=5)
    train, val, calib, test = grouped_train_val_calib_test(df, cfg)

    assert all(len(part) > 0 for part in (train, val, calib, test))
    assert len(train) + len(val) + len(calib) + len(test) == len(df)

    splits = {"train": set(train["username"]), "val": set(val["username"]),
              "calib": set(calib["username"]), "test": set(test["username"])}
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert splits[names[i]] & splits[names[j]] == set(), f"{names[i]} vs {names[j]} overlap"


def test_conformal_quantile_exact_and_degenerate():
    # n=9, level=0.9 -> k=ceil(10*0.9)=9 -> 9th smallest of 1..9 is 9 (the max, boundary case).
    assert _conformal_quantile(np.arange(1, 10, dtype=float), 0.9) == pytest.approx(9.0)

    # n=3, level=0.99 -> k=ceil(4*0.99)=4 > n=3 -> degenerate, falls back to the max.
    assert _conformal_quantile(np.array([5.0, 3.0, 1.0]), 0.99) == pytest.approx(5.0)


def test_conformal_correction_negative_and_apply_narrows():
    # Raw interval [-10, 10] wildly over-covers y in [-1, 1] -> nonconformity scores are all
    # strongly negative -> the symmetric correction must itself be negative.
    interval = {
        "lower": np.full(5, -10.0), "upper": np.full(5, 10.0), "median": np.zeros(5),
    }
    y = np.array([-1.0, 0.0, 0.0, 0.0, 1.0])
    q_lo, q_hi = conformal_correction(interval, y, alpha=0.1, two_sided=False)
    assert q_lo < 0 and q_hi < 0
    assert q_lo == pytest.approx(q_hi)  # symmetric path returns equal (lo, hi)

    corrected = apply_conformal(interval, (q_lo, q_hi))
    # Narrowed relative to the raw interval, and ordering survives the negative correction.
    assert np.all(corrected["lower"] > interval["lower"])
    assert np.all(corrected["upper"] < interval["upper"])
    assert np.all(corrected["lower"] <= corrected["median"])
    assert np.all(corrected["median"] <= corrected["upper"])


def test_cqr_coverage_end_to_end_symmetric_and_two_sided():
    rng = np.random.RandomState(0)
    n = 5000
    y_calib = rng.normal(0, 1, n)
    y_test = rng.normal(0, 1, n)

    def _raw_interval(size):
        return {"lower": np.full(size, -1.0), "upper": np.full(size, 1.0), "median": np.zeros(size)}

    alpha = 0.1
    for two_sided in (False, True):
        calib_interval = _raw_interval(n)
        correction = conformal_correction(calib_interval, y_calib, alpha, two_sided=two_sided)
        test_corrected = apply_conformal(_raw_interval(n), correction)
        coverage = np.mean(
            (y_test >= test_corrected["lower"]) & (y_test <= test_corrected["upper"])
        )
        assert 0.88 <= coverage <= 0.92, f"two_sided={two_sided}: coverage={coverage}"


def test_predict_interval_decrosses_crossed_quantiles():
    class _StubModel:
        def __init__(self, values):
            self.values = np.asarray(values, float)

        def predict(self, x):
            return self.values

    quantiles = [0.05, 0.5, 0.95]
    models = {
        0.05: _StubModel([5.0, 1.0, 0.0]),   # row 0: "lower" quantile pred is the biggest (crossed)
        0.5: _StubModel([1.0, 1.0, 2.0]),
        0.95: _StubModel([9.0, 0.0, 4.0]),   # row 1: "upper" quantile pred is the smallest (crossed)
    }

    out = predict_interval(models, x=None, quantiles=quantiles)

    assert set(out.keys()) == {"lower", "median", "upper"}
    assert np.all(out["lower"] <= out["median"])
    assert np.all(out["median"] <= out["upper"])


def test_ridge_tolerates_nan_numeric_feature():
    x = pd.DataFrame({
        "game_plies": [40, np.nan, 35, 42],
        "player_moves": [20, 21, 19, 22],
        "n_captures": [2, 3, 1, 4],
        "n_checks": [0, 1, 0, 2],
        "result": ["win", "loss", "draw", "win"],
        "time_control": ["300+0", "300+0", "300+0", "300+0"],
        "eco": ["C50", "B20", "D30", "C50"],
    })
    y = np.array([1500.0, 1600.0, 1400.0, 1700.0])

    ridge = make_ridge(alpha=1.0)
    ridge.fit(x, y)  # must not raise despite the NaN in a numeric feature column


def test_conformal_correction_by_band_covers_every_band_and_falls_back_when_thin():
    bands = [0, 1200, 1400, 1600, 1800, 2000, 3000]
    rng = np.random.RandomState(0)
    median = np.concatenate([
        rng.uniform(700, 1150, 200),    # band 0: plenty of rows, tight scores
        rng.uniform(1210, 1390, 5),     # band 1: THIN (< min_n), huge-outlier scores
    ])                                   # bands 2..5: zero rows -> must also fall back
    y = np.concatenate([
        median[:200] + rng.normal(0, 20, 200),
        median[200:] + rng.normal(0, 5000, 5),   # far larger scale -> pooled quantile shifts a lot
    ])
    interval = {"lower": median - 1.0, "upper": median + 1.0, "median": median}

    corrections = conformal_correction_by_band(interval, y, alpha=0.1, bands=bands, min_n=20)
    global_correction = conformal_correction(interval, y, 0.1, False)

    assert set(corrections.keys()) == set(range(len(bands) - 1))   # every band index present
    assert corrections[1] == global_correction    # thin band (5 rows < min_n=20) -> global fallback
    for b in (2, 3, 4, 5):                        # empty bands -> global fallback too
        assert corrections[b] == global_correction
    assert corrections[0] != global_correction     # the well-populated band got its OWN correction


def test_mondrian_cqr_fixes_heteroscedastic_tail_undercoverage():
    """Per-band coverage under Mondrian stays near 90% at both rating extremes; plain CQR does not.

    y ~ N(median, sigma(band)) with sigma much larger in the extreme bands (0 and 5) than in the
    middle four; a deliberately tiny, fixed raw interval (median +/- 1) so essentially the whole
    interval width has to come from the conformal correction. Predicted medians are spread across
    every band for both calib (~4000 rows) and test (~4000 rows).
    """
    bands = [0, 1200, 1400, 1600, 1800, 2000, 3000]
    band_ranges = [(700, 1150), (1210, 1390), (1410, 1590), (1610, 1790), (1810, 1990), (2050, 2450)]
    sigmas = [300.0, 20.0, 20.0, 20.0, 20.0, 300.0]     # extremes much noisier than the middle
    n_per_band = 700

    def _make(rng):
        medians, ys, band_of = [], [], []
        for b, ((lo, hi), sigma) in enumerate(zip(band_ranges, sigmas)):
            med = rng.uniform(lo, hi, n_per_band)
            medians.append(med)
            ys.append(med + rng.normal(0, sigma, n_per_band))
            band_of.append(np.full(n_per_band, b))
        return np.concatenate(medians), np.concatenate(ys), np.concatenate(band_of)

    rng = np.random.RandomState(0)
    med_cal, y_cal, _ = _make(rng)
    med_te, y_te, band_te = _make(rng)

    def _raw(median):
        return {"lower": median - 1.0, "upper": median + 1.0, "median": median}

    alpha = 0.1
    global_corr = conformal_correction(_raw(med_cal), y_cal, alpha, two_sided=False)
    plain = apply_conformal(_raw(med_te), global_corr)

    mondrian_corr = conformal_correction_by_band(_raw(med_cal), y_cal, alpha, bands, min_n=200)
    mondrian = apply_conformal_by_band(_raw(med_te), mondrian_corr, bands)

    def _cov_by_band(interval):
        cov = []
        for b in range(len(bands) - 1):
            m = band_te == b
            cov.append(float(np.mean((y_te[m] >= interval["lower"][m]) & (y_te[m] <= interval["upper"][m]))))
        return cov

    cov_plain = _cov_by_band(plain)
    cov_mondrian = _cov_by_band(mondrian)

    assert all(0.85 <= c <= 0.95 for c in cov_mondrian), cov_mondrian
    assert any(c < 0.85 for c in cov_plain), cov_plain


def test_apply_conformal_by_band_preserves_ordering_with_negative_correction():
    bands = [0, 1200, 1400, 1600, 1800, 2000, 3000]
    median = np.array([500.0, 1300.0, 2500.0])
    # predicted_bands(median, bands) -> [0, 1, 5]
    interval = {"lower": median - 20.0, "upper": median + 20.0, "median": median}
    corrections = {
        0: (-50.0, -50.0),   # exceeds the raw half-width (20) -> must clip to the median, not cross it
        1: (10.0, 10.0),     # ordinary positive widening
        2: (0.0, 0.0), 3: (0.0, 0.0), 4: (0.0, 0.0),
        5: (-100.0, -100.0),  # also exceeds the raw half-width -> clips to the median
    }

    out = apply_conformal_by_band(interval, corrections, bands)

    assert np.all(out["lower"] <= out["median"])
    assert np.all(out["median"] <= out["upper"])
    assert out["lower"][0] == pytest.approx(median[0]) and out["upper"][0] == pytest.approx(median[0])
    assert out["lower"][1] == pytest.approx(1300.0 - 20.0 - 10.0)
    assert out["upper"][1] == pytest.approx(1300.0 + 20.0 + 10.0)
    assert out["lower"][2] == pytest.approx(median[2]) and out["upper"][2] == pytest.approx(median[2])


def test_predicted_bands_matches_digitize_on_median():
    bands = [0, 1200, 1400, 1600, 1800, 2000, 3000]
    median = np.array([500.0, 1250.0, 2999.0])
    assert predicted_bands(median, bands).tolist() == [0, 1, 5]
