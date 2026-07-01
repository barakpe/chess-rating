"""Feature engineering — the heart of the pipeline.

This module has two parts:

1. **Verified, pure primitives** (fully implemented, unit-tested): Lichess's Win%,
   per-move Accuracy%, and move classification. Constants are transcribed verbatim from
   Lichess source (``lila``/``scalachess``) and the numeric outputs are pinned in
   ``tests/test_features.py``. See the source references on each function.

2. **Phase-split feature extraction** (stubbed): replays a game, reads per-move evals, and
   computes per-player features split by opening / middlegame / endgame. Left as a documented
   ``NotImplementedError`` for the feature-engineering stage to fill in against a stable interface.

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
      * opening: up to ``opening_max_ply`` (proxy for "until out of book" — real book-exit
        detection via ``opening_ply`` is a feature-stage refinement).
      * endgame: once total non-pawn pieces (both sides) <= ``endgame_nonpawn_pieces``.
      * middlegame: everything in between.
    """
    phases = cfg["phases"]
    if ply <= phases["opening_max_ply"]:
        return "opening"
    if count_nonpawn_pieces(board) <= phases["endgame_nonpawn_pieces"]:
        return "endgame"
    return "middlegame"


# --- Phase-split feature extraction (STUB) ---------------------------------------------

def extract_player_features(game: Any, color: bool, cfg: dict[str, Any]) -> dict[str, float]:
    """Compute one player's feature vector from a single game. (STUB — feature stage.)

    Parameters
    ----------
    game : chess.pgn.Game
        A parsed game (from the raw ``movetext`` column). Iterate ``game.mainline()`` nodes;
        each node carries ``.move``, ``.eval()`` (a ``PovScore``), and ``.clock()`` (seconds).
    color : bool
        ``chess.WHITE`` (True) or ``chess.BLACK`` (False) — the player being described.
    cfg : dict
        The loaded config (phase boundaries, eval clamp/mate handling, etc.).

    Returns
    -------
    dict
        Flat ``{feature_name: value}`` for THIS player only. Every "quality" feature is also
        emitted per phase (opening/middlegame/endgame) — e.g. ``cpl_mean_opening``.

    Implementation notes for the feature stage
    ------------------------------------------
    * EVAL ORIENTATION: ``node.eval().white()`` gives the raw White-POV Score. Convert to
      centipawns with ``.score()`` (None on mate) and handle mate via ``.mate()`` +
      ``mate_to_cp(...)``. Flip sign for Black to get the mover's POV.
    * BEFORE/AFTER INDEXING: the eval annotated on a move is the eval of the position AFTER
      that move. For a move played by ``color``, ``win_before`` = Win% of the position before
      the move (i.e. the eval on the PREVIOUS ply), ``win_after`` = Win% after the move — both
      from ``color``'s POV. The very first "before" uses ``cfg['eval']['start_cp']`` (=15).
    * PER-MOVE QUALITIES (only for moves played by ``color``):
        - centipawn loss = win-oriented cp drop (or accuracy via ``accuracy_percent``),
        - ``accuracy_percent(win_before, win_after)``,
        - ``classify_move(winning_chances(cp_before), winning_chances(cp_after))``.
    * AGGREGATIONS (overall AND per ``game_phase(ply, board, cfg)``): mean & median centipawn
      loss, mean Accuracy%, std of centipawn loss (consistency), worst single move,
      inaccuracy/mistake/blunder counts and per-move rates, plus "accuracy after leaving the
      opening book".
    * OPENING/TIME/STYLE: opening ply (book depth), ECO family, mean/var of move times from
      ``node.clock()`` deltas, share of very-fast moves, time-trouble behaviour, game length,
      result, captures/checks. See PROJECT_ARCHITECTURE.md §Stage 4.
    * LEAKAGE: never read the opponent's moves/evals into ``color``'s features.
    """
    raise NotImplementedError(
        "extract_player_features is a stub for the feature-engineering stage; "
        "the verified primitives above (win_percent, accuracy_percent, classify_move) are ready to use."
    )
