"""Tests for the confirmatory hold-out (src/holdout.py) on tiny synthetic data."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from src.config import load_config  # noqa: E402
from src.holdout import development_players, run_holdout_evaluate  # noqa: E402
from tests.test_evaluate import _synthetic_parquets  # noqa: E402


def test_development_players_covers_both_colours():
    games = pd.DataFrame({"white": ["a", "b"], "black": ["c", "a"]})
    assert development_players(games) == {"a", "b", "c"}


def test_holdout_evaluate_scores_once_and_refuses_a_second_run(tmp_path):
    cfg = load_config()
    cfg["model"]["lgbm"]["n_estimators"] = 30
    cfg["evaluate"]["bootstrap_reps"] = 20
    cfg["evaluate"]["aggregation_min_players"] = 2
    dev = tmp_path / "dev"
    hold = tmp_path / "hold"
    dev.mkdir()
    hold.mkdir()
    fp, gp = _synthetic_parquets(dev, seed=0)
    hfp, _ = _synthetic_parquets(hold, n_games=60, seed=1)
    h = pd.read_parquet(hfp)
    h["username"] = "new_" + h["username"]            # players the development data never saw
    h.to_parquet(hfp, index=False)
    cfg["outputs"].update({
        "features": str(fp), "games_clean": str(gp), "holdout_features": str(hfp),
        "holdout_results": str(tmp_path / "holdout_results.json"), "results_log": str(tmp_path / "results.md"),
    })

    dry = run_holdout_evaluate(cfg, dry_run=True)
    assert not (tmp_path / "holdout_results.json").exists()      # a dry run writes nothing
    assert np.isfinite(dry["mae"]["full"])

    res = run_holdout_evaluate(cfg)
    assert res["n_rows"] == len(h) and res["n_players"] == h["username"].nunique()
    for k in ("predict_mean", "baseline", "full", "tuned"):
        assert np.isfinite(res["mae"][k])
    assert 0.0 <= res["interval"]["cqr_mondrian"]["coverage"] <= 1.0
    saved = json.loads((tmp_path / "holdout_results.json").read_text(encoding="utf-8"))
    assert saved["utc_date"] == cfg["holdout"]["utc_date"]
    assert "confirmatory hold-out" in (tmp_path / "results.md").read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        run_holdout_evaluate(cfg)
