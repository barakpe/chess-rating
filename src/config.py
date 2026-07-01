"""Load and validate ``config.yaml`` — the single source of truth for every knob.

Every stage imports from here rather than hard-coding paths, thresholds, or seeds.
Paths in the YAML are relative to the repository root; this module resolves them to
absolute paths and creates the directories so downstream code can just write to them.

Usage
-----
    from src.config import load_config
    cfg = load_config()                 # reads <repo>/config.yaml
    raw = cfg["raw_path"]               # absolute path to the month's .pgn.zst

A module-level ``CONFIG`` is provided for convenience, but prefer calling
``load_config()`` in code you want to unit-test with an overridden path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Repo root = parent of this file's directory (…/chess-rating/src/config.py -> …/chess-rating)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

# Keys we hard-require so a typo in config.yaml fails loudly instead of silently.
_REQUIRED_TOP_LEVEL = (
    "month",
    "blitz_estimate_seconds",
    "sample_size",
    "min_plies",
    "random_seed",
    "rating_bands",
    "eval",
    "phases",
    "paths",
    "outputs",
)


def raw_filename(month: str) -> str:
    """Return the Lichess standard-rated dump filename for a month, e.g. '2025-05'."""
    return f"lichess_db_standard_rated_{month}.pgn.zst"


def _validate(cfg: dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ValueError(f"config.yaml is missing required keys: {missing}")

    if not isinstance(cfg["sample_size"], int) or cfg["sample_size"] <= 0:
        raise ValueError(f"sample_size must be a positive int, got {cfg['sample_size']!r}")

    lo_hi = cfg["blitz_estimate_seconds"]
    if not (isinstance(lo_hi, list) and len(lo_hi) == 2 and lo_hi[0] <= lo_hi[1]):
        raise ValueError(
            f"blitz_estimate_seconds must be [lo, hi] with lo <= hi, got {lo_hi!r}"
        )

    bands = cfg["rating_bands"]
    if not (isinstance(bands, list) and len(bands) >= 2 and all(a < b for a, b in zip(bands, bands[1:]))):
        raise ValueError(f"rating_bands must be strictly increasing, got {bands!r}")

    cap = cfg.get("max_games_scanned")
    if cap is not None and (not isinstance(cap, int) or cap <= 0):
        raise ValueError(f"max_games_scanned must be null or a positive int, got {cap!r}")


def _resolve_paths(cfg: dict[str, Any]) -> None:
    """Turn repo-relative paths into absolute ones, create dirs, and derive raw file paths.

    Adds the following resolved keys to ``cfg``:
      - ``paths``/``outputs`` values become absolute strings
      - ``raw_file``: the month's dump filename
      - ``raw_path``: absolute path to that dump under ``paths.raw``
    """
    for key, rel in cfg["paths"].items():
        abs_path = (REPO_ROOT / rel).resolve()
        os.makedirs(abs_path, exist_ok=True)
        cfg["paths"][key] = str(abs_path)

    # Output files live under already-created dirs; resolve but don't create the files.
    for key, rel in cfg["outputs"].items():
        abs_path = (REPO_ROOT / rel).resolve()
        os.makedirs(abs_path.parent, exist_ok=True)
        cfg["outputs"][key] = str(abs_path)

    cfg["raw_file"] = raw_filename(cfg["month"])
    cfg["raw_path"] = str(Path(cfg["paths"]["raw"]) / cfg["raw_file"])


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load ``config.yaml``, validate it, resolve+create paths, and return the dict.

    Parameters
    ----------
    path : optional
        Path to the YAML file. Defaults to ``<repo>/config.yaml``.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config at {path} did not parse to a mapping")

    _validate(cfg)
    _resolve_paths(cfg)
    return cfg


# Convenience singleton for scripts/notebooks. Import ``load_config`` instead when
# you need an isolated config (e.g. tests pointing at a fixture config).
CONFIG = load_config()


if __name__ == "__main__":
    import json

    print(json.dumps(CONFIG, indent=2))
