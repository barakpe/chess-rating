"""Smoke test for the ingest pipeline against the committed synthetic fixture.

The fixture (tests/fixtures/sample.pgn -> .pgn.zst) has 8 hand-made games, one per filter
branch, so the whole funnel is deterministic and reproducible without the 30 GB download.
Expected survivors: exactly 1 (the Italian game — blitz + eval + non-bot + Normal + long enough).
"""

from pathlib import Path

import pytest

# These carry the ingest dependencies; skip cleanly if the env isn't set up yet.
pytest.importorskip("chess")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("zstandard")

from src.config import load_config  # noqa: E402
from src.ingest import INGEST_COLUMNS, run_ingest  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample.pgn.zst"


def _cfg():
    cfg = load_config()
    cfg["sample_size"] = 100
    cfg["max_games_scanned"] = None
    return cfg


def test_fixture_exists():
    assert FIXTURE.exists(), "run `python tests/fixtures/make_fixture.py` to build sample.pgn.zst"


def test_ingest_funnel_is_deterministic(tmp_path):
    out = tmp_path / "blitz_sample.parquet"
    funnel = run_ingest(FIXTURE, out, _cfg())
    assert funnel == {
        "scanned": 8,
        "blitz": 6,       # excludes bullet (120+1) and rapid (600+0)
        "candidate": 3,   # blitz & non-bot & Normal/Time-forfeit (drops 2 bots + 1 abandoned)
        "has_eval": 2,    # drops the clk-only game
        "clean": 1,       # drops the < min_plies game
        "sampled": 1,
    }


def test_ingest_output_matches_schema(tmp_path):
    import pandas as pd

    out = tmp_path / "blitz_sample.parquet"
    run_ingest(FIXTURE, out, _cfg())
    df = pd.read_parquet(out)

    assert list(df.columns) == INGEST_COLUMNS
    assert len(df) == 1
    row = df.iloc[0]
    assert row["game_id"] == "https://lichess.org/keep0001"
    assert row["time_control"] == "300+0"
    assert row["termination"] == "Normal"
    assert row["white_elo"] == 1685 and row["black_elo"] == 1702
    assert row["n_plies"] == 14
    assert "[%eval" in row["movetext"] and "[%clk" in row["movetext"]
