"""Stage 4 tests: extract_player_features on the fixture keep game (hand-verified values),
plus an end-to-end build_features run (which exercises headerless movetext parsing)."""

import io
import math
from pathlib import Path

import pytest

pytest.importorskip("chess")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

import chess.pgn  # noqa: E402
import pandas as pd  # noqa: E402

from src.clean import build_instances  # noqa: E402
from src.config import load_config  # noqa: E402
from src.features import FEATURE_ID_COLUMNS, build_features, extract_player_features  # noqa: E402
from src.ingest import INGEST_COLUMNS  # noqa: E402

FIXTURE_PGN = Path(__file__).resolve().parent / "fixtures" / "sample.pgn"


def _keep_game():
    """First game in the fixture = the Italian keeper (14 plies, eval+clk on every move)."""
    with open(FIXTURE_PGN, encoding="utf-8") as fh:
        return chess.pgn.read_game(fh)


def test_extract_white_hand_verified():
    cfg = load_config()
    w = extract_player_features(_keep_game(), chess.WHITE, cfg, increment=0, base_seconds=300)

    # White's centipawn losses over its 7 moves are [0,3,0,0,6,7,7] (mover-POV eval drops).
    assert w["n_moves"] == 7
    assert w["cpl_max"] == 7
    assert w["cpl_mean"] == pytest.approx(23 / 7)
    # Tiny swings -> no errors, high accuracy.
    assert w["blunder_count"] == 0 and w["mistake_count"] == 0 and w["inaccuracy_count"] == 0
    assert w["acc_mean"] > 95

    # All 14 plies are within the opening boundary (30), so every move is "opening".
    assert w["opening_n_moves"] == 7
    assert w["middlegame_n_moves"] == 0 and w["endgame_n_moves"] == 0

    # Clocks: with base=300 the per-move times are [0,3,3,4,3,4,4] -> mean 3.0.
    assert w["game_plies"] == 14
    assert w["n_timed_moves"] == 7
    assert w["move_time_mean"] == pytest.approx(3.0)
    assert w["fast_move_share"] == pytest.approx(1 / 7)  # only the 0s move is < 1s

    # Never reached a winning (>=80% Win) position, so conversion is undefined.
    assert w["reached_winning"] == 0
    assert math.isnan(w["converted_winning"])


def test_extract_black_basic():
    cfg = load_config()
    b = extract_player_features(_keep_game(), chess.BLACK, cfg, increment=0, base_seconds=300)
    assert b["n_moves"] == 7
    assert b["blunder_count"] == 0
    assert b["opening_n_moves"] == 7


def test_absent_phase_contract():
    """The keep game is 14 plies, entirely inside the opening boundary (30): middlegame and
    endgame never happened. n_moves must be a true 0 (a real exposure measure); every other
    per-phase aggregate must be NaN ("not observable"), not a misleading 0."""
    cfg = load_config()
    w = extract_player_features(_keep_game(), chess.WHITE, cfg, increment=0, base_seconds=300)

    for prefix in ("middlegame_", "endgame_"):
        assert w[f"{prefix}n_moves"] == 0
        for stat in (
            "cpl_mean", "cpl_median", "cpl_std", "cpl_max", "acc_mean",
            "inaccuracy_count", "mistake_count", "blunder_count",
            "inaccuracy_rate", "mistake_rate", "blunder_rate",
        ):
            assert math.isnan(w[f"{prefix}{stat}"]), f"{prefix}{stat} should be NaN, got {w[f'{prefix}{stat}']}"

    assert w["has_opening"] == 1
    assert w["has_middlegame"] == 0
    assert w["has_endgame"] == 0


def test_missing_clock_game():
    """A game with [%eval] but no [%clk] anywhere: every time-derived feature is NaN (not 0),
    since 0 timed moves means "unobserved", not "instant"/"never in time trouble"."""
    pgn = (
        '[Event "Test"]\n[White "a"]\n[Black "b"]\n[Result "*"]\n\n'
        "1. e4 { [%eval 0.20] } 1... e5 { [%eval 0.18] } "
        "2. Nf3 { [%eval 0.22] } 2... Nc6 { [%eval 0.19] } *\n"
    )
    game = chess.pgn.read_game(io.StringIO(pgn))
    cfg = load_config()
    w = extract_player_features(game, chess.WHITE, cfg, increment=0, base_seconds=300)

    assert w["n_timed_moves"] == 0
    assert w["has_clock"] == 0
    for feat in ("move_time_mean", "move_time_std", "move_time_median", "fast_move_share", "time_trouble_share"):
        assert math.isnan(w[feat]), f"{feat} should be NaN, got {w[feat]}"


def test_single_quality_record_cpl_std_nan():
    """Exactly one eval'd move for the player -> cpl_std is NaN (undefined), not 0.0
    (0.0 would misleadingly read as 'one move, perfectly consistent')."""
    pgn = '[Event "Test"]\n[White "a"]\n[Black "b"]\n[Result "*"]\n\n1. e4 { [%eval 0.20] } *\n'
    game = chess.pgn.read_game(io.StringIO(pgn))
    cfg = load_config()
    w = extract_player_features(game, chess.WHITE, cfg, increment=0, base_seconds=300)

    assert w["n_moves"] == 1
    assert math.isnan(w["cpl_std"])


def test_acc_after_book_nan_when_game_never_leaves_opening():
    """The keep game (14 plies) never passes opening_max_ply (30), so acc_after_book -
    computed only over plies past the opening boundary - is undefined."""
    cfg = load_config()
    w = extract_player_features(_keep_game(), chess.WHITE, cfg, increment=0, base_seconds=300)
    assert math.isnan(w["acc_after_book"])


def _games_clean_from_keep():
    game = _keep_game()
    movetext = game.accept(chess.pgn.StringExporter(headers=False, variations=False, comments=True))
    row = {col: None for col in INGEST_COLUMNS}
    row.update({
        "game_id": "g1", "event": "Rated Blitz game", "white": "alice", "black": "bob",
        "result": "1-0", "white_elo": 1685, "black_elo": 1702, "eco": "C50",
        "opening": "Italian Game", "time_control": "300+0", "termination": "Normal",
        "utc_date": "2025.05.01", "utc_time": "12:00:00", "movetext": movetext, "n_plies": 14,
    })
    return pd.DataFrame([row], columns=INGEST_COLUMNS)


def test_missing_clock_game_scramble_nan():
    """Same clockless game as test_missing_clock_game: with no timed moves at all, every
    scramble_* feature is unobservable (NaN), not a true zero, and has_scramble reads 0."""
    pgn = (
        '[Event "Test"]\n[White "a"]\n[Black "b"]\n[Result "*"]\n\n'
        "1. e4 { [%eval 0.20] } 1... e5 { [%eval 0.18] } "
        "2. Nf3 { [%eval 0.22] } 2... Nc6 { [%eval 0.19] } *\n"
    )
    game = chess.pgn.read_game(io.StringIO(pgn))
    cfg = load_config()
    w = extract_player_features(game, chess.WHITE, cfg, increment=0, base_seconds=300)

    assert w["has_clock"] == 0
    assert w["has_scramble"] == 0
    for feat in (
        "scramble_n_moves", "scramble_share", "scramble_move_time_mean", "scramble_move_time_std",
        "scramble_cpl_mean", "scramble_blunder_rate", "scramble_cpl_delta",
    ):
        assert math.isnan(w[feat]), f"{feat} should be NaN, got {w[feat]}"


def test_keep_game_no_scramble():
    """The 14-ply fixture keep game is played with base_seconds=300 (well above the 30s
    scramble_seconds threshold) and White's clock per test_extract_white_hand_verified never
    drops below ~292s, so it never dips under the scramble threshold: scramble_n_moves is a
    true 0 (clock IS present), not NaN, and the aggregates over zero scramble moves are NaN."""
    cfg = load_config()
    assert cfg["features"]["scramble_seconds"] == 30
    w = extract_player_features(_keep_game(), chess.WHITE, cfg, increment=0, base_seconds=300)

    assert w["has_clock"] == 1
    assert w["scramble_n_moves"] == 0
    assert w["scramble_share"] == 0.0
    assert w["has_scramble"] == 0
    for feat in (
        "scramble_move_time_mean", "scramble_move_time_std",
        "scramble_cpl_mean", "scramble_blunder_rate", "scramble_cpl_delta",
    ):
        assert math.isnan(w[feat]), f"{feat} should be NaN, got {w[feat]}"


def test_scramble_features_hand_verified():
    """Crafted PGN: base clock 60s. White's remaining clock (BEFORE the move) is 60s and 58s on
    its first two moves (not scrambles), then 25s and 20s on moves 3 and 4 (< scramble_seconds=30
    -> scrambles). Move 3 is a huge eval swing (a blunder); move 4 is a tiny one (not).

    White's mover-POV cpl per move: [0, 0, 620, 5] (before/after eval pairs: 15->20, 18->25,
    20->-600, -580->-585). Overall cpl_mean = 625/4 = 156.25; scramble-only cpl_mean (moves 3,4)
    = 625/2 = 312.5. Scramble think-times are 25-20=5s and 20-10=10s -> mean 7.5, pstdev 2.5.
    """
    pgn = (
        '[Event "Test"]\n[White "a"]\n[Black "b"]\n[Result "*"]\n\n'
        "1. e4 { [%eval 0.20] [%clk 0:00:58] } 1... e5 { [%eval 0.18] } "
        "2. Nf3 { [%eval 0.25] [%clk 0:00:25] } 2... Nc6 { [%eval 0.20] } "
        "3. Bc4 { [%eval -6.00] [%clk 0:00:20] } 3... Bc5 { [%eval -5.80] } "
        "4. c3 { [%eval -5.85] [%clk 0:00:10] } *\n"
    )
    game = chess.pgn.read_game(io.StringIO(pgn))
    cfg = load_config()
    w = extract_player_features(game, chess.WHITE, cfg, increment=0, base_seconds=60)

    assert w["n_moves"] == 4
    assert w["player_moves"] == 4
    assert w["has_clock"] == 1
    assert w["has_scramble"] == 1

    # Clock entering moves 1,2 is 60s,58s (not < 30); entering moves 3,4 is 25s,20s (< 30).
    assert w["scramble_n_moves"] == 2
    assert w["scramble_share"] == pytest.approx(0.5)

    assert w["cpl_mean"] == pytest.approx(156.25)
    assert w["scramble_cpl_mean"] == pytest.approx(312.5)
    assert w["scramble_cpl_delta"] == pytest.approx(312.5 - 156.25)

    # Move 3 (cpl 620, a huge eval swing from +20 to -600) is a blunder; move 4 (cpl 5) is not.
    assert w["scramble_blunder_rate"] == pytest.approx(0.5)

    # Scramble-move think times: 25->20 (5s, spent=5) and 20->10 (10s, spent=10).
    assert w["scramble_move_time_mean"] == pytest.approx(7.5)
    assert w["scramble_move_time_std"] == pytest.approx(2.5)


def test_scramble_boundary_exactly_30_excluded():
    """remaining == scramble_seconds exactly is NOT a scramble (strict <). base_seconds=30: the
    first move's clock-before is exactly 30 (excluded); it drops to 29 for the second move
    (included). scramble_n_moves must be 1, not 2 or 0."""
    pgn = (
        '[Event "Test"]\n[White "a"]\n[Black "b"]\n[Result "*"]\n\n'
        "1. e4 { [%eval 0.20] [%clk 0:00:29] } 1... e5 { [%eval 0.18] } "
        "2. Nf3 { [%eval 0.22] [%clk 0:00:25] } *\n"
    )
    game = chess.pgn.read_game(io.StringIO(pgn))
    cfg = load_config()
    w = extract_player_features(game, chess.WHITE, cfg, increment=0, base_seconds=30)

    assert w["player_moves"] == 2
    assert w["scramble_n_moves"] == 1
    assert w["scramble_share"] == pytest.approx(0.5)
    assert w["has_scramble"] == 1


def test_build_features_end_to_end_no_leakage():
    cfg = load_config()
    games_clean = _games_clean_from_keep()
    instances = build_instances(games_clean)

    feats = build_features(games_clean, instances, cfg)

    assert len(feats) == 2
    # ID/label columns come first, in the contracted order.
    assert feats.columns[: len(FEATURE_ID_COLUMNS)].tolist() == FEATURE_ID_COLUMNS
    # Leakage guard: no opponent rating columns anywhere in the feature frame.
    for forbidden in ("white_elo", "black_elo", "opponent", "opponent_rating"):
        assert forbidden not in feats.columns

    w = feats[feats["color"] == "white"].iloc[0]
    assert w["rating"] == 1685 and w["result"] == "win"
    assert w["n_moves"] == 7 and w["move_time_mean"] == pytest.approx(3.0)
    b = feats[feats["color"] == "black"].iloc[0]
    assert b["rating"] == 1702 and b["n_moves"] == 7
