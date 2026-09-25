"""Feature engineering — the heart of the pipeline.

This module has two parts:

1. **Verified, pure primitives** (fully implemented, unit-tested): Lichess's Win%,
   per-move Accuracy%, and move classification. Constants are transcribed verbatim from
   Lichess source (``lila``/``scalachess``) and the numeric outputs are pinned in
   ``tests/test_features.py``. See the source references on each function.

2. **Phase-split feature extraction** (Stage 4): replays a game, reads per-move evals/clocks, and
   computes per-player move-quality (split by opening / middlegame / endgame), time, and style
   features. ``build_features`` / ``run_features`` turn instances + games_clean into features.parquet.

LEAKAGE RULE (non-negotiable): features describe *one player's own play*. Never derive a
feature from the opponent (opponent rating, opponent centipawn loss, rating gap, …). Lichess
matches similar ratings, so opponent-derived signals leak the label. Each game is later exploded
into two instances (one per side), each labelled with *that* player's rating — see Stage 3.

Two DIFFERENT scales are in play; do not conflate them:
  * Accuracy% uses Win% **points** on a 0..100 scale (winDiff = win_before - win_after).
  * Move classification uses **winningChances** on a -1..+1 scale (thresholds 0.10/0.20/0.30,
    i.e. ~5/10/15 Win% points).
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

# --- Win% / winningChances -------------------------------------------------------------
# Source: lichess-org/scalachess core/src/main/scala/eval.scala and
#         lichess-org/lila ui/lib/src/ceval/winningChances.ts (constant from lila PR #11148),
#         also documented at https://lichess.org/page/accuracy
# Lichess writes `MULTIPLIER = -0.00368208` inside `exp(MULTIPLIER * cp)`.
WIN_MULT = -0.00368208


def winning_chances(cp: float, clamp: int = 1000) -> float:
    """Lichess winningChances in [-1, +1] for a centipawn eval from the mover's POV.

    ``cp`` is clamped to +/- ``clamp`` (Lichess clamps real evals to +/-1000 before Win%).
    Positive cp -> positive winningChances (good for the side to move).
    """
    cp = max(-clamp, min(clamp, cp))
    return 2.0 / (1.0 + math.exp(WIN_MULT * cp)) - 1.0


def win_percent(cp: float, clamp: int = 1000) -> float:
    """Win% in [0, 100] for a centipawn eval from the mover's POV: 50 + 50 * winningChances."""
    return 50.0 + 50.0 * winning_chances(cp, clamp)


# --- Per-move Accuracy% ----------------------------------------------------------------
# Source: lichess-org/lila modules/analyse/src/main/AccuracyPercent.scala (fromWinPercents).
# Full-precision constants from the source; the public docs round them to
# 103.1668 / 0.04354 / 3.1669. A "+1 uncertainty bonus" is added before the [0,100] clamp,
# and the function short-circuits to 100 when the move does not lose Win%.
_ACC_A = 103.1668100711649
_ACC_K = 0.04354415386753951
_ACC_B = -3.166924740191411
_ACC_UNCERTAINTY_BONUS = 1.0


def accuracy_percent(win_before: float, win_after: float) -> float:
    """Lichess per-move Accuracy% in [0, 100].

    ``win_before`` / ``win_after`` are Win% **points** (0..100) from the MOVER's POV
    (before and after the move). If the move did not lose Win%, accuracy is exactly 100.
    """
    if win_after >= win_before:
        return 100.0
    win_diff = win_before - win_after
    raw = _ACC_A * math.exp(-_ACC_K * win_diff) + _ACC_B + _ACC_UNCERTAINTY_BONUS
    return max(0.0, min(100.0, raw))


# --- Move classification ---------------------------------------------------------------
# Source: lichess-org/lila modules/tree/src/main/Advice.scala (CpAdvice.winningChanceJudgements).
# Thresholds are on the winningChances (-1..+1) drop, NOT raw centipawn loss.
BLUNDER_THRESHOLD = 0.30
MISTAKE_THRESHOLD = 0.20
INACCURACY_THRESHOLD = 0.10


def classify_move(wc_before: float, wc_after: float) -> str:
    """Classify a move as 'blunder' / 'mistake' / 'inaccuracy' / 'ok'.

    ``wc_before`` / ``wc_after`` are winningChances (-1..+1) from the MOVER's POV. The caller
    is responsible for the POV sign (Black's winningChances is the negation of White's), so
    ``drop = wc_before - wc_after`` is the mover's loss of winning chances.
    """
    drop = wc_before - wc_after
    if drop >= BLUNDER_THRESHOLD:
        return "blunder"
    if drop >= MISTAKE_THRESHOLD:
        return "mistake"
    if drop >= INACCURACY_THRESHOLD:
        return "inaccuracy"
    return "ok"


def mate_to_cp(mate: int, clamp: int = 1000) -> int:
    """Fold a signed mate distance into a ceiling centipawn value (+/- ``clamp``).

    Matches Lichess's accuracy path, where a forced mate maps to the +/-1000 ceiling (Win% ~= 97.5%,
    not 100%). Use this to convert a mate eval before feeding it to ``win_percent`` / ``winning_chances``.
    """
    if mate == 0:
        # '#0' is an already-delivered mate against the side to move.
        return -clamp
    return clamp if mate > 0 else -clamp


# --- Game-phase helpers ----------------------------------------------------------------
# ``board`` is a chess.Board; we use its API by duck typing so this module imports without
# python-chess (keeps the pure-math tests dependency-free). Piece types: pawn=1 … king=6.

def count_nonpawn_pieces(board: Any) -> int:
    """Count non-pawn, non-king pieces on the board (knights, bishops, rooks, queens).

    Used for the middlegame -> endgame boundary. ``board`` is a ``chess.Board``.
    """
    # chess.KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5 (PAWN=1, KING=6 excluded).
    return sum(1 for piece in board.piece_map().values() if 2 <= piece.piece_type <= 5)


def game_phase(ply: int, board: Any, cfg: dict[str, Any]) -> str:
    """Return 'opening' / 'middlegame' / 'endgame' for a position.

    Pragmatic boundaries from ``cfg['phases']``:
      * opening: plies 1..``opening_max_ply`` — a fixed ply cut-off, not an opening-book lookup
        (so ``acc_after_book`` below means "accuracy after the opening plies").
      * endgame: once total non-pawn pieces (both sides) <= ``endgame_nonpawn_pieces``.
      * middlegame: everything in between.
    """
    phases = cfg["phases"]
    if ply <= phases["opening_max_ply"]:
        return "opening"
    if count_nonpawn_pieces(board) <= phases["endgame_nonpawn_pieces"]:
        return "endgame"
    return "middlegame"


# --- Phase-split feature extraction (Stage 4) ------------------------------------------

_PHASES = ("opening", "middlegame", "endgame")
_NAN = float("nan")


def _white_cp(pov_score: Any, clamp: int) -> int | None:
    """A PovScore -> White-POV centipawns, mate folded to +/-clamp, clamped. None if unreadable."""
    if pov_score is None:
        return None
    white = pov_score.white()
    if white.is_mate():
        return mate_to_cp(white.mate(), clamp)
    cp = white.score()
    if cp is None:
        return None
    return max(-clamp, min(clamp, cp))


def _agg(records: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    """Aggregate a list of per-move quality records into a flat feature dict.

    NaN contract: phase absent (``n == 0``) -> ``n_moves`` is 0 (a true exposure measure, always
    defined) while every other aggregate is NaN ("not observable"), including the counts (a count
    is only meaningful conditional on exposure). ``cpl_std`` is additionally NaN for ``n == 1``
    (std is undefined for a single observation, not "perfectly consistent").
    """
    n = len(records)
    cpls = [r["cpl"] for r in records]
    accs = [r["accuracy"] for r in records]
    inacc = sum(r["klass"] == "inaccuracy" for r in records)
    mist = sum(r["klass"] == "mistake" for r in records)
    blun = sum(r["klass"] == "blunder" for r in records)
    return {
        f"{prefix}n_moves": n,
        f"{prefix}cpl_mean": statistics.fmean(cpls) if cpls else _NAN,
        f"{prefix}cpl_median": statistics.median(cpls) if cpls else _NAN,
        f"{prefix}cpl_std": statistics.pstdev(cpls) if len(cpls) > 1 else _NAN,
        f"{prefix}cpl_max": max(cpls) if cpls else _NAN,
        f"{prefix}acc_mean": statistics.fmean(accs) if accs else _NAN,
        f"{prefix}inaccuracy_count": inacc if n else _NAN,
        f"{prefix}mistake_count": mist if n else _NAN,
        f"{prefix}blunder_count": blun if n else _NAN,
        f"{prefix}inaccuracy_rate": inacc / n if n else _NAN,
        f"{prefix}mistake_rate": mist / n if n else _NAN,
        f"{prefix}blunder_rate": blun / n if n else _NAN,
    }


def extract_player_features(
    game: Any,
    color: bool,
    cfg: dict[str, Any],
    *,
    increment: int = 0,
    base_seconds: int | None = None,
    player_result: str | None = None,
) -> dict[str, float]:
    """Compute one player's feature vector from a single game.

    Replays the game once (incremental board), reading each move's stored eval and clock. Produces
    per-move centipawn-loss / Accuracy% / classification, aggregated overall AND per phase
    (opening/middlegame/endgame), plus time-management and style features. Describes ``color``'s
    play only — never the opponent's (leakage rule).

    Parameters
    ----------
    game : chess.pgn.Game        parsed game (headers optional; movetext carries [%eval]/[%clk]).
    color : bool                 chess.WHITE (True) / chess.BLACK (False) — the player described.
    cfg : dict                   loaded config (eval clamp, phase boundaries, feature thresholds).
    increment, base_seconds      clock base/increment (seconds) for move-time reconstruction.
    player_result                "win"/"loss"/"draw" for this player (for the conversion feature).

    Notes
    -----
    The eval annotated on a move is the eval of the position AFTER it (White POV). So ``win_before``
    for a move is the previous ply's eval (the first uses ``cfg['eval']['start_cp']``), flipped to
    the mover's POV; ``win_after`` is this ply's eval flipped to the mover's POV. Move-quality is
    computed only for plies that carry an eval; time/style over all of the player's moves.

    NaN contract: an unobserved quantity is NaN, not 0 — 0 is reserved for a true zero-exposure
    count (``n_moves``, ``n_timed_moves``) or a genuinely-computed zero. ``move_time_std`` is NaN
    for fewer than 2 timed moves; ``time_trouble_share`` is NaN with no timed moves at all. Five
    flags (``has_opening``, ``has_middlegame``, ``has_endgame``, ``has_clock``, ``has_scramble``)
    make phase/clock absence directly splittable instead of relying on trees inferring it from the
    NaNs. The clock-scramble block (``scramble_*``: moves where the player's remaining clock BEFORE
    the move was under ``cfg['features']['scramble_seconds']``) follows the same rule: every
    ``scramble_*`` feature is NaN (``has_scramble`` reads 0) when the game has no timed moves at
    all; ``scramble_n_moves``/``scramble_share`` are true zeros (not NaN) when clock data exists but
    no move ever fell under the threshold; ``scramble_cpl_mean``/``scramble_blunder_rate``/
    ``scramble_cpl_delta`` are NaN whenever there is no eval'd scramble move to aggregate.
    """
    eval_cfg = cfg["eval"]
    clamp = eval_cfg["cp_clamp"]
    start_cp = eval_cfg["start_cp"]
    opening_max_ply = cfg["phases"]["opening_max_ply"]
    fcfg = cfg["features"]
    fast_move_seconds = fcfg["fast_move_seconds"]
    time_trouble_seconds = fcfg["time_trouble_seconds"]
    scramble_seconds = fcfg["scramble_seconds"]
    winning_winpct = fcfg["winning_winpct"]

    want_white = bool(color)
    board = game.board()  # starting position (standard)

    quality: list[dict[str, Any]] = []
    n_player_moves = n_captures = n_checks = n_time_trouble = 0
    move_times: list[float] = []
    scramble_move_times: list[float] = []
    scramble_n_moves = 0
    prev_remaining = base_seconds
    reached_winning = False
    prev_white_cp: int | None = start_cp

    for node in game.mainline():
        move = node.move
        mover_white = board.turn  # side to move BEFORE the move == the mover
        is_capture = board.is_capture(move)
        board.push(move)
        gives_check = board.is_check()
        after_w = _white_cp(node.eval(), clamp)

        if mover_white == want_white:
            n_player_moves += 1
            n_captures += is_capture
            n_checks += gives_check

            # Captured BEFORE prev_remaining is updated below: a scramble move is one where the
            # player's clock, as it stood going INTO this move, was already under the threshold.
            is_scramble = prev_remaining is not None and prev_remaining < scramble_seconds
            if is_scramble:
                scramble_n_moves += 1

            remaining = node.clock()
            if prev_remaining is not None and remaining is not None:
                spent = prev_remaining - remaining + increment
                if spent >= 0:
                    move_times.append(spent)
                    if is_scramble:
                        scramble_move_times.append(spent)
            # Always advance, even to None: a move with no clock annotation makes the NEXT move's
            # remaining time genuinely unknown, not "same as last observed" -- carrying the stale
            # value forward would misjudge that next move's scramble/time-trouble status.
            prev_remaining = remaining
            if remaining is not None and remaining < time_trouble_seconds:
                n_time_trouble += 1

            if after_w is not None and prev_white_cp is not None:
                mover_after = after_w if want_white else -after_w
                mover_before = prev_white_cp if want_white else -prev_white_cp
                win_before = win_percent(mover_before, clamp)
                win_after = win_percent(mover_after, clamp)
                if win_after >= winning_winpct:
                    reached_winning = True
                quality.append({
                    "ply": node.ply(),
                    "phase": game_phase(node.ply(), board, cfg),
                    "cpl": max(0.0, mover_before - mover_after),
                    "accuracy": accuracy_percent(win_before, win_after),
                    "klass": classify_move(
                        winning_chances(mover_before, clamp), winning_chances(mover_after, clamp)
                    ),
                    "scramble": is_scramble,
                })

        prev_white_cp = after_w

    # --- aggregate ---
    feats: dict[str, float] = {}
    feats.update(_agg(quality, ""))
    for phase in _PHASES:
        feats.update(_agg([r for r in quality if r["phase"] == phase], f"{phase}_"))

    feats["has_opening"] = int(feats["opening_n_moves"] > 0)
    feats["has_middlegame"] = int(feats["middlegame_n_moves"] > 0)
    feats["has_endgame"] = int(feats["endgame_n_moves"] > 0)
    feats["has_clock"] = int(len(move_times) > 0)

    # "After book" = after the opening ply cut-off (the name is historical; no book lookup).
    after_book = [r["accuracy"] for r in quality if r["ply"] > opening_max_ply]
    feats["acc_after_book"] = statistics.fmean(after_book) if after_book else _NAN

    feats["move_time_mean"] = statistics.fmean(move_times) if move_times else _NAN
    feats["move_time_std"] = statistics.pstdev(move_times) if len(move_times) > 1 else _NAN
    feats["move_time_median"] = statistics.median(move_times) if move_times else _NAN
    feats["n_timed_moves"] = len(move_times)
    feats["fast_move_share"] = (
        sum(t < fast_move_seconds for t in move_times) / len(move_times) if move_times else _NAN
    )
    # No timed moves at all -> undefined, not 0 (0 would misleadingly read as "never in time trouble").
    feats["time_trouble_share"] = n_time_trouble / n_player_moves if move_times else _NAN

    # --- clock-scramble block (moves with remaining clock, before the move, < scramble_seconds) ---
    if not move_times:
        # No clock at all: every scramble feature is unobservable, not a true zero.
        feats["scramble_n_moves"] = _NAN
        feats["scramble_share"] = _NAN
        feats["scramble_move_time_mean"] = _NAN
        feats["scramble_move_time_std"] = _NAN
        feats["scramble_cpl_mean"] = _NAN
        feats["scramble_blunder_rate"] = _NAN
        feats["scramble_cpl_delta"] = _NAN
        feats["has_scramble"] = 0
    else:
        scramble_quality = [r for r in quality if r["scramble"]]
        scramble_cpls = [r["cpl"] for r in scramble_quality]
        n_scramble_quality = len(scramble_quality)
        scramble_blunders = sum(r["klass"] == "blunder" for r in scramble_quality)

        feats["scramble_n_moves"] = scramble_n_moves
        feats["scramble_share"] = scramble_n_moves / n_player_moves
        feats["scramble_move_time_mean"] = (
            statistics.fmean(scramble_move_times) if scramble_move_times else _NAN
        )
        feats["scramble_move_time_std"] = (
            statistics.pstdev(scramble_move_times) if len(scramble_move_times) > 1 else _NAN
        )
        feats["scramble_cpl_mean"] = statistics.fmean(scramble_cpls) if scramble_cpls else _NAN
        feats["scramble_blunder_rate"] = (
            scramble_blunders / n_scramble_quality if n_scramble_quality else _NAN
        )
        # Plain float arithmetic: NaN on either side (no scramble evals, or no evals at all)
        # propagates to NaN automatically.
        feats["scramble_cpl_delta"] = feats["scramble_cpl_mean"] - feats["cpl_mean"]
        feats["has_scramble"] = int(scramble_n_moves > 0)

    feats["game_plies"] = board.ply()
    feats["player_moves"] = n_player_moves
    feats["n_captures"] = n_captures
    feats["n_checks"] = n_checks
    feats["reached_winning"] = int(reached_winning)
    if player_result is None or not reached_winning:
        feats["converted_winning"] = _NAN
    else:
        feats["converted_winning"] = 1.0 if player_result == "win" else 0.0
    return feats


# --- Stage 4 driver: instances + games_clean -> features.parquet -----------------------

# Identity / label columns carried alongside the numeric features (feature columns follow).
FEATURE_ID_COLUMNS = ["game_id", "color", "username", "rating", "result", "time_control", "eco", "opening"]


def _parse_time_control(time_control: str | None) -> tuple[int | None, int]:
    """'300+3' -> (300, 3); '-' / bad -> (None, 0)."""
    if not time_control or time_control == "-":
        return None, 0
    base, _, inc = time_control.partition("+")
    try:
        return int(base), int(inc or 0)
    except ValueError:
        return None, 0


def build_features(games_clean_df: Any, instances_df: Any, cfg: dict[str, Any]) -> Any:
    """Parse each game once and emit one feature row per (game, side), joined to instance labels."""
    import io

    import chess.pgn
    import pandas as pd

    labels = {(r.game_id, r.color): r for r in instances_df.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for g in games_clean_df.itertuples(index=False):
        game = chess.pgn.read_game(io.StringIO(g.movetext))
        if game is None:
            continue
        base, inc = _parse_time_control(g.time_control)
        for color_name, want_white in (("white", True), ("black", False)):
            label = labels.get((g.game_id, color_name))
            if label is None:
                continue
            feats = extract_player_features(
                game, want_white, cfg,
                increment=inc, base_seconds=base, player_result=label.result,
            )
            feats.update({
                "game_id": g.game_id, "color": color_name, "username": label.username,
                "rating": int(label.rating), "result": label.result,
                "time_control": g.time_control, "eco": g.eco, "opening": g.opening,
            })
            rows.append(feats)

    df = pd.DataFrame(rows)
    if not df.empty:
        feature_cols = sorted(c for c in df.columns if c not in FEATURE_ID_COLUMNS)
        df = df[FEATURE_ID_COLUMNS + feature_cols]
        df["rating"] = df["rating"].astype("int16")
    return df


def run_features(games_path: str | Path, instances_path: str | Path, output_path: str | Path,
                 cfg: dict[str, Any]) -> Any:
    """Read games_clean + instances -> feature rows -> features.parquet."""
    import pandas as pd

    games = pd.read_parquet(games_path)
    instances = pd.read_parquet(instances_path)
    df = build_features(games, instances, cfg)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    n_feats = max(0, df.shape[1] - len(FEATURE_ID_COLUMNS))
    print(f"wrote {len(df):,} feature rows x {n_feats} features to {output_path}")
    return df


def main() -> None:
    import argparse

    from src.config import load_config

    cfg = load_config()
    parser = argparse.ArgumentParser(description="Stage 4: build per-player features.")
    parser.add_argument("--games", default=cfg["outputs"]["games_clean"], help="games_clean parquet")
    parser.add_argument("--instances", default=cfg["outputs"]["instances"], help="instances parquet")
    parser.add_argument("--output", default=cfg["outputs"]["features"], help="features parquet")
    args = parser.parse_args()

    for label, path in (("games_clean", args.games), ("instances", args.instances)):
        if not Path(path).exists():
            raise SystemExit(f"{label} not found: {path}\nRun `python -m src.clean` first.")
    run_features(args.games, args.instances, args.output, cfg)


if __name__ == "__main__":
    main()
