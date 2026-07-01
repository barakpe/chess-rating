"""Regression tests pinning the verified Lichess Win%/Accuracy%/classification math.

Reference values were computed by running the functions and cross-checked against Lichess
source (lila/scalachess). If a test here breaks, the formula transcription drifted — fix the
code, not the expected value (unless Lichess itself changed and you re-verified upstream).
"""

import pytest

from src.features import (
    accuracy_percent,
    classify_move,
    mate_to_cp,
    win_percent,
    winning_chances,
)


# --- Win% -------------------------------------------------------------------------------
def test_win_percent_reference_values():
    assert win_percent(0) == pytest.approx(50.0)
    assert win_percent(100) == pytest.approx(59.1026, rel=1e-3)
    assert win_percent(300) == pytest.approx(75.1126, rel=1e-3)


def test_win_percent_clamps_at_1000():
    assert win_percent(5000) == pytest.approx(win_percent(1000))
    assert win_percent(-5000) == pytest.approx(win_percent(-1000))
    assert win_percent(1000) == pytest.approx(97.5447, rel=1e-3)  # mate maps here, ~97.5 not 100


def test_win_percent_is_symmetric():
    # winningChances is an odd function, so win_percent(cp) + win_percent(-cp) == 100.
    assert win_percent(250) + win_percent(-250) == pytest.approx(100.0)
    assert winning_chances(0) == pytest.approx(0.0)


# --- Accuracy% --------------------------------------------------------------------------
def test_accuracy_is_100_when_no_winpercent_lost():
    assert accuracy_percent(60, 60) == 100.0
    assert accuracy_percent(60, 70) == 100.0  # improving the position never penalizes


def test_accuracy_reference_values():
    assert accuracy_percent(60, 40) == pytest.approx(41.0168, rel=1e-3)  # winDiff 20
    assert accuracy_percent(60, 20) == pytest.approx(15.909, rel=1e-3)   # winDiff 40
    assert accuracy_percent(90, 10) == pytest.approx(1.0, abs=0.05)      # winDiff 80


def test_accuracy_clamped_to_0_100():
    for wb, wa in [(100, 0), (50, 0), (80, 10), (99, 1)]:
        assert 0.0 <= accuracy_percent(wb, wa) <= 100.0


# --- Move classification (winningChances -1..+1 scale) ----------------------------------
def test_classify_move_thresholds():
    assert classify_move(0.60, 0.25) == "blunder"      # drop 0.35
    assert classify_move(0.60, 0.35) == "mistake"      # drop 0.25
    assert classify_move(0.60, 0.45) == "inaccuracy"   # drop 0.15
    assert classify_move(0.60, 0.55) == "ok"           # drop 0.05
    assert classify_move(0.50, 0.90) == "ok"           # improving is never a mistake


# --- Mate folding -----------------------------------------------------------------------
def test_mate_to_cp():
    assert mate_to_cp(3) == 1000
    assert mate_to_cp(-2) == -1000
    assert mate_to_cp(0) == -1000  # '#0' = already mated
    assert win_percent(mate_to_cp(1)) == pytest.approx(97.5447, rel=1e-3)
