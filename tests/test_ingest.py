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

import chess.pgn  # noqa: E402
import pandas as pd  # noqa: E402
import zstandard  # noqa: E402

from src.config import load_config  # noqa: E402
from src.ingest import (  # noqa: E402
    INGEST_COLUMNS,
    has_eval,
    run_ingest,
    stream_game_chunks,
    string_candidate_ok,
    string_has_eval_hint,
    string_is_blitz,
    username_in_cohort,
)

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
        "cohort": 0,      # no cohort_output_path passed -> cohort stays inactive (0 kept)
    }


def test_ingest_output_matches_schema(tmp_path):
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


# --------------------------------------------------------------------------------------
# Player cohort — hash-sampled subset kept in full alongside the uniform reservoir sample.
# --------------------------------------------------------------------------------------
def test_username_in_cohort_is_deterministic():
    for name in ["alice", "Bob", "SomeRandomUser123"]:
        assert username_in_cohort(name, 0.3) == username_in_cohort(name, 0.3)


def test_username_in_cohort_hash_rate_zero_never_matches():
    for name in ["alice", "bob", "SomeRandomUser123", "z"]:
        assert username_in_cohort(name, 0.0) is False


def test_username_in_cohort_hash_rate_one_always_matches():
    for name in ["alice", "bob", "SomeRandomUser123", "z"]:
        assert username_in_cohort(name, 1.0) is True


def test_username_in_cohort_none_or_empty_is_false():
    assert username_in_cohort(None, 1.0) is False
    assert username_in_cohort("", 1.0) is False


def test_username_in_cohort_is_case_insensitive():
    assert username_in_cohort("Foo", 0.5) == username_in_cohort("foo", 0.5)


def test_cohort_matches_reservoir_when_hash_rate_is_one(tmp_path):
    cfg = _cfg()
    cfg["player_cohort"] = {"enabled": True, "hash_rate": 1.0, "max_games": 1000}
    out = tmp_path / "blitz_sample.parquet"
    cohort_out = tmp_path / "player_cohort.parquet"

    funnel = run_ingest(FIXTURE, out, cfg, cohort_output_path=cohort_out)

    assert cohort_out.exists()
    df_reservoir = pd.read_parquet(out)
    df_cohort = pd.read_parquet(cohort_out)
    assert list(df_cohort.columns) == INGEST_COLUMNS
    # hash_rate=1.0 keeps every game that passed the filters -> same games as the reservoir.
    assert sorted(df_cohort["game_id"]) == sorted(df_reservoir["game_id"])
    assert funnel["cohort"] == funnel["clean"]


def test_cohort_max_games_cap(tmp_path):
    cfg = _cfg()
    # Lower min_plies so a second game (the 4-ply "tooshort" game) also survives the filters,
    # giving >1 cohort-eligible game to actually exercise the cap.
    cfg["min_plies"] = 2
    cfg["player_cohort"] = {"enabled": True, "hash_rate": 1.0, "max_games": 1}
    out = tmp_path / "blitz_sample.parquet"
    cohort_out = tmp_path / "player_cohort.parquet"

    funnel = run_ingest(FIXTURE, out, cfg, cohort_output_path=cohort_out)

    assert funnel["clean"] == 2   # keep0001 (14 plies) + tooshort (4 plies)
    assert funnel["cohort"] == 1  # capped by max_games, even though hash_rate=1.0 matches both
    df_cohort = pd.read_parquet(cohort_out)
    assert len(df_cohort) == 1


def test_no_player_cohort_config_is_backward_compatible(tmp_path):
    """cfg with no ``player_cohort`` key and no cohort_output_path -> unchanged behavior."""
    cfg = _cfg()
    cfg.pop("player_cohort", None)
    out = tmp_path / "blitz_sample.parquet"

    funnel = run_ingest(FIXTURE, out, cfg)

    assert funnel == {
        "scanned": 8,
        "blitz": 6,
        "candidate": 3,
        "has_eval": 2,
        "clean": 1,
        "sampled": 1,
        "cohort": 0,
    }
    df = pd.read_parquet(out)
    assert list(df.columns) == INGEST_COLUMNS
    assert len(df) == 1
    assert not (tmp_path / "player_cohort.parquet").exists()


def test_ingest_e2e_row_order_matches_pinned_scan_order(tmp_path):
    """End-to-end guard beyond the funnel dict: with min_plies lowered so two fixture games
    survive (keep0001's 14 plies and tooshort's 4), the surviving game_ids must come out in the
    same order as the original scan (algorithm-R reservoir keeps arrival order while under
    capacity) -- i.e. the prefilter rewrite must not reorder, drop, or duplicate rows.
    """
    cfg = _cfg()
    cfg["min_plies"] = 2
    out = tmp_path / "blitz_sample.parquet"
    funnel = run_ingest(FIXTURE, out, cfg)
    assert funnel["clean"] == 2
    df = pd.read_parquet(out)
    assert list(df["game_id"]) == [
        "https://lichess.org/keep0001",
        "https://lichess.org/tooshort",
    ]


# --------------------------------------------------------------------------------------
# stream_game_chunks — the cheap string-level chunker feeding the prefilter (no SAN parse).
# --------------------------------------------------------------------------------------
def _zst_path(tmp_path: Path, pgn_text: str, name: str = "custom.pgn.zst") -> Path:
    path = tmp_path / name
    cctx = zstandard.ZstdCompressor(level=3)
    path.write_bytes(cctx.compress(pgn_text.encode("utf-8")))
    return path


def test_stream_game_chunks_matches_fixture_game_count():
    chunks = list(stream_game_chunks(FIXTURE))
    assert len(chunks) == 8


def test_stream_game_chunks_splits_headers_and_movetext_correctly():
    headers, movetext = next((h, m) for h, m in stream_game_chunks(FIXTURE) if "keep0001" in h)
    # headers_text is exactly the header-tag lines (plus their blank-line trailer at most) --
    # no movetext leaked in, and vice versa.
    assert all(line.strip() == "" or line.strip().startswith("[") for line in headers.splitlines())
    assert not any(line.strip().startswith("[") for line in movetext.splitlines())
    assert '[White "alice"]' in headers
    assert '[TimeControl "300+0"]' in headers
    assert "1. e4" in movetext and "1-0" in movetext
    assert "[%eval 0.17]" in movetext and "[%clk 0:05:00]" in movetext


def test_stream_game_chunks_robust_to_multiple_blank_lines(tmp_path):
    """Multiple blank lines both before the movetext and before the next game's headers must
    not confuse the state machine (real dumps sometimes have >1 blank line in these gaps)."""
    pgn_text = (
        '[Event "A"]\n[Site "https://lichess.org/multiblank1"]\n[White "x"]\n[Black "y"]\n'
        '[Result "1-0"]\n[TimeControl "300+0"]\n[Termination "Normal"]\n'
        "\n\n\n"  # 3 blank lines before movetext
        "1. e4 e5 1-0\n"
        "\n\n"  # 2 blank lines before the next game
        '[Event "B"]\n[Site "https://lichess.org/multiblank2"]\n[White "p"]\n[Black "q"]\n'
        '[Result "0-1"]\n[TimeControl "300+0"]\n[Termination "Normal"]\n'
        "\n"
        "1. d4 d5 0-1\n"
    )
    path = _zst_path(tmp_path, pgn_text)
    chunks = list(stream_game_chunks(path))
    assert len(chunks) == 2
    (h0, m0), (h1, m1) = chunks
    assert "multiblank1" in h0 and "1. e4 e5 1-0" in m0
    assert "multiblank2" in h1 and "1. d4 d5 0-1" in m1


def test_stream_game_chunks_handles_final_game_without_trailing_blank_line(tmp_path):
    """No trailing newline/blank line after the last game's movetext -- must still be yielded."""
    pgn_text = (
        '[Event "A"]\n[Site "https://lichess.org/nofinalblank"]\n[White "x"]\n[Black "y"]\n'
        '[Result "1-0"]\n[TimeControl "300+0"]\n[Termination "Normal"]\n'
        "\n"
        "1. e4 e5 1-0"  # no trailing newline at all
    )
    path = _zst_path(tmp_path, pgn_text)
    chunks = list(stream_game_chunks(path))
    assert len(chunks) == 1
    h, m = chunks[0]
    assert "nofinalblank" in h
    assert "1. e4 e5 1-0" in m


# --------------------------------------------------------------------------------------
# String-level prefilter superset property -- can only reject games the authoritative,
# post-parse checks would also reject; must never reject a game the real checks would keep.
# --------------------------------------------------------------------------------------
_BLITZ_EVAL_HEADERS = (
    '[Event "Rated Blitz game"]\n[Site "https://lichess.org/prefiltergood"]\n'
    '[White "aaa"]\n[Black "bbb"]\n[Result "1-0"]\n[WhiteElo "1500"]\n[BlackElo "1500"]\n'
    '[TimeControl "300+0"]\n[Termination "Normal"]\n'
)
_BLITZ_EVAL_MOVETEXT = (
    "1. e4 { [%eval 0.20] [%clk 0:05:00] } 1... e5 { [%eval 0.18] [%clk 0:05:00] } 1-0\n"
)

_BLITZ_NOEVAL_HEADERS = (
    '[Event "Rated Blitz game"]\n[Site "https://lichess.org/prefilternoeval"]\n'
    '[White "ccc"]\n[Black "ddd"]\n[Result "1-0"]\n[WhiteElo "1500"]\n[BlackElo "1500"]\n'
    '[TimeControl "300+0"]\n[Termination "Normal"]\n'
)
_BLITZ_NOEVAL_MOVETEXT = "1. e4 { [%clk 0:05:00] } 1... e5 { [%clk 0:05:00] } 1-0\n"


def _prefilter_kwargs(cfg):
    lo, hi = cfg["blitz_estimate_seconds"]
    return dict(lo=lo, hi=hi, keep_terminations=cfg["keep_terminations"], exclude_bots=cfg["exclude_bots"])


def test_prefilter_keeps_blitz_eval_game():
    cfg = _cfg()
    assert string_is_blitz(_BLITZ_EVAL_HEADERS, cfg["blitz_estimate_seconds"][0], cfg["blitz_estimate_seconds"][1])
    assert string_candidate_ok(_BLITZ_EVAL_HEADERS, **_prefilter_kwargs(cfg))
    assert string_has_eval_hint(_BLITZ_EVAL_MOVETEXT)


def test_prefilter_rejects_noeval_game_before_parse():
    cfg = _cfg()
    # Still a string-level candidate (blitz, non-bot, Normal) -- only the eval hint differs.
    assert string_candidate_ok(_BLITZ_NOEVAL_HEADERS, **_prefilter_kwargs(cfg))
    assert not string_has_eval_hint(_BLITZ_NOEVAL_MOVETEXT)


def test_prefilter_rejected_noeval_game_is_also_rejected_by_authoritative_check(tmp_path):
    """The superset-safety property: a game the string prefilter would reject (no eval hint)
    must also be rejected by the real, post-parse ``has_eval`` -- i.e. the prefilter never
    discards a game the authoritative pipeline would have kept.
    """
    import io

    pgn_text = _BLITZ_NOEVAL_HEADERS.rstrip("\n") + "\n\n" + _BLITZ_NOEVAL_MOVETEXT
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    assert game is not None
    assert not has_eval(game)


def test_prefilter_end_to_end_via_run_ingest(tmp_path):
    """Build a tiny dump with exactly the good/no-eval pair above and confirm run_ingest keeps
    the eval game and drops the no-eval one -- the prefilter and the authoritative recheck
    agree end to end."""
    pgn_text = (
        _BLITZ_EVAL_HEADERS + "\n" + _BLITZ_EVAL_MOVETEXT + "\n" + _BLITZ_NOEVAL_HEADERS + "\n" + _BLITZ_NOEVAL_MOVETEXT
    )
    path = _zst_path(tmp_path, pgn_text)
    out = tmp_path / "out.parquet"
    cfg = _cfg()
    cfg["min_plies"] = 1
    funnel = run_ingest(path, out, cfg)
    assert funnel["scanned"] == 2
    assert funnel["candidate"] == 2  # both are blitz/non-bot/Normal at the string level
    assert funnel["has_eval"] == 1   # only the eval game survives the authoritative check
    assert funnel["clean"] == 1
    df = pd.read_parquet(out)
    assert list(df["game_id"]) == ["https://lichess.org/prefiltergood"]


def test_ingest_writes_funnel_json_with_scan_window(tmp_path):
    import json

    out = tmp_path / "blitz_sample.parquet"
    funnel_path = tmp_path / "data_funnel.json"
    funnel = run_ingest(FIXTURE, out, _cfg(), funnel_path=funnel_path)

    data = json.loads(funnel_path.read_text(encoding="utf-8"))
    rec = data["ingest"]
    for key, value in funnel.items():
        assert rec[key] == value                     # same counts as the returned funnel
    assert rec["input"] == FIXTURE.name
    assert rec["scan_window_utc"]["first"].startswith("2025.05.01")
    assert rec["scan_window_utc"]["last"] is not None


def test_header_timestamp():
    from src.ingest import header_timestamp

    headers = '[Event "x"]\n[UTCDate "2025.05.03"]\n[UTCTime "07:08:09"]\n'
    assert header_timestamp(headers) == "2025.05.03 07:08:09"
    assert header_timestamp('[Event "x"]\n') is None
    assert header_timestamp(None) is None
