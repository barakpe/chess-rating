"""Shared plotting style + figure-saving helper.

Call ``set_style()`` once at the top of a notebook/script; use ``save_fig(fig, name)`` so every
figure that goes in the deck is written to ``reports/figures/<name>.png`` at a consistent dpi. The
slides reference the saved PNGs, not live notebook cells, so charts can be regenerated when data
updates without hand-editing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

# Non-interactive backend so figures render headless (scripts, CI) without a display.
matplotlib.use("Agg")

PALETTE = ["#2c6fbb", "#e07b39", "#3aa66f", "#c0504d", "#8064a2", "#4bacc6"]


def set_style() -> None:
    """Apply one consistent look: fonts, palette, sizes, grid, dpi."""
    plt.rcParams.update({
        "figure.figsize": (7.5, 4.5),
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })


def save_fig(fig: Any, name: str, figures_dir: str | Path = "reports/figures") -> Path:
    """Save ``fig`` to ``<figures_dir>/<name>.png`` and return the path."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
