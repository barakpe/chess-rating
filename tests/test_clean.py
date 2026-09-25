"""Tests for Stage 2 (clean/dedup) and Stage 3 (instance-explode)."""

import pytest

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

import pandas as pd  # noqa: E402

from src.clean import INSTANCE_COLUMNS, build_instances, clean_games  # noqa: E402
from src.config import load_config  # noqa: E402
from src.ingest import INGEST_COLUMNS  # noqa: E402


def _game(game_id, white, black, we, be, result="1-0", termination="Normal", n_plies=30, tc="300+0"):
    """One INGEST_COLUMNS row with sensible defaults; override the fields a test cares about."""
    return {
        "game_id": game_id, "event": "Rated Blitz game", "white": white, "black": black,
        "result": result, "white_elo": we, "black_elo": be,
        "white_rating_diff": None, "black_rating_diff": None, "eco": "C50",
        "opening": "Italian Game", "time_control": tc, "termination": termination,
        "utc_date": "2025.05.01", "utc_time": "12:00:00", "movetext": "1. e4 e5 1-0",
        "n_plies": n_plies,
    }


def _frame(rows):
    return pd.DataFrame(rows, columns=INGEST_COLUMNS)


def test_clean_drops_the_right_rows():
    cfg = load_config()  # min_plies=10, keep Normal/Time forfeit, rating 400..4000
    rows = [
        _game("g1", "alice", "bob", 1600, 1500, result="1-0", n_plies=30),          # keep
        _game("g1", "alice", "bob", 1600, 1500, result="1-0", n_plies=30),          # dup -> drop
        _game("g2", "cid", "dan", 1400, 1450, result="0-1", n_plies=8),             # short -> drop
        _game("g3", "eve", "fay", 1700, 1650, termination="Abandoned"),             # termination -> drop
        _game("g4", "gil", "hal", 1800, 1750, result="*", n_plies=20),              # unknown result -> drop
        _game("g5", "ida", "jon", 50, 1500, n_plies=20),                            # corrupt Elo -> drop
        _game("g6", "kai", "leo", 2000, 2100, result="1/2-1/2", termination="Time forfeit", n_plies=40),  # keep
    ]
    clean_df, funnel = clean_games(_frame(rows), cfg)

    assert funnel == {"input": 7, "deduped": 6, "cleaned": 2}
    assert set(clean_df["game_id"]) == {"g1", "g6"}


def test_build_instances_labels_and_no_leakage():
    cfg = load_config()
    rows = [
        _game("g1", "alice", "bob", 1600, 1500, result="1-0", n_plies=30),
        _game("g6", "kai", "leo", 2000, 2100, result="1/2-1/2", n_plies=40),
    ]
    clean_df, _ = clean_games(_frame(rows), cfg)
    inst = build_instances(clean_df)

    # Exactly two rows per game, correct schema.
    assert list(inst.columns) == INSTANCE_COLUMNS
    assert len(inst) == 4

    # LEAKAGE GUARD: no opponent rating/username columns exist at all.
    for forbidden in ("white_elo", "black_elo", "opponent", "opponent_rating", "opponent_elo"):
        assert forbidden not in inst.columns

    g1 = inst[inst["game_id"] == "g1"].set_index("color")
    assert g1.loc["white", "username"] == "alice"
    assert g1.loc["white", "rating"] == 1600 and g1.loc["white", "result"] == "win"
    assert g1.loc["black", "username"] == "bob"
    assert g1.loc["black", "rating"] == 1500 and g1.loc["black", "result"] == "loss"

    # Draw is symmetric; each side keeps its own rating.
    g6 = inst[inst["game_id"] == "g6"].set_index("color")
    assert g6.loc["white", "rating"] == 2000 and g6.loc["black", "rating"] == 2100
    assert set(g6["result"]) == {"draw"}


def test_run_clean_writes_both_parquets(tmp_path):
    cfg = load_config()
    src_df = _frame([
        _game("g1", "alice", "bob", 1600, 1500, n_plies=30),
        _game("g6", "kai", "leo", 2000, 2100, result="0-1", n_plies=40),
    ])
    in_path = tmp_path / "blitz_sample.parquet"
    src_df.to_parquet(in_path, index=False)

    from src.clean import run_clean
    games_out = tmp_path / "games_clean.parquet"
    inst_out = tmp_path / "instances.parquet"
    funnel = run_clean(in_path, games_out, inst_out, cfg)

    assert funnel["cleaned"] == 2 and funnel["instances"] == 4
    assert pd.read_parquet(games_out).shape[0] == 2
    assert list(pd.read_parquet(inst_out).columns) == INSTANCE_COLUMNS


def test_union_with_cohort_then_clean_keeps_sample_first():
    from src.clean import union_with_cohort

    cfg = load_config()
    sample = _frame([_game("g1", "a", "b", 1500, 1500), _game("g2", "c", "d", 1600, 1600)])
    cohort = _frame([_game("g2", "c", "d", 1600, 1600), _game("g3", "e", "f", 1700, 1700)])
    u = union_with_cohort(sample, cohort)
    assert list(u["game_id"]) == ["g1", "g2", "g2", "g3"]          # plain append, sample first
    clean_df, funnel = clean_games(u, cfg)
    assert list(clean_df["game_id"]) == ["g1", "g2", "g3"]          # the cohort's copy of g2 is dropped
    assert funnel["input"] - funnel["deduped"] == 1                 # overlap visible in the funnel
    assert list(union_with_cohort(sample, None)["game_id"]) == ["g1", "g2"]


def test_run_clean_with_cohort_writes_funnel(tmp_path):
    import json

    from src.clean import run_clean

    cfg = load_config()
    sample_p, cohort_p = tmp_path / "s.parquet", tmp_path / "c.parquet"
    _frame([_game("g1", "a", "b", 1500, 1500), _game("g2", "c", "d", 1600, 1600, n_plies=4)]).to_parquet(sample_p)
    _frame([_game("g1", "a", "b", 1500, 1500), _game("g3", "e", "f", 1700, 1700)]).to_parquet(cohort_p)
    funnel_p = tmp_path / "funnel.json"

    funnel = run_clean(sample_p, tmp_path / "gc.parquet", tmp_path / "inst.parquet", cfg,
                       cohort_path=cohort_p, funnel_path=funnel_p)

    assert funnel["sample"] == 2 and funnel["cohort"] == 2
    assert funnel["input"] == 4 and funnel["deduped"] == 3        # g1 in both -> kept once
    assert funnel["cleaned"] == 2                                 # g2 dropped (< min_plies)
    assert funnel["instances"] == 4 and funnel["players"] == 4
    assert json.loads(funnel_p.read_text(encoding="utf-8"))["clean"] == funnel
