"""Build a slim, shareable copy of the processed parquets.

Why this exists
---------------
`data/raw/*.pgn.zst` is ~30 GB per month and `data/processed/games_clean_300k.parquet`
is ~313 MB, almost all of it the raw `movetext` column. A collaborator who only wants to
*run notebooks 01-05* does not need either: the notebooks read `games_clean_300k` through
column projections and touch `movetext` for exactly one game (the worked example in 02).

This script writes a bundle with the **same filenames** the notebooks expect, so the
receiver just drops them into their own `data/processed/`. Nothing is resampled: every
game, every player and every feature column is preserved, so notebook numbers come out
identical to the ones in `reports/`.

    python -m scripts.make_share_bundle                  # -> share_bundle/
    python -m scripts.make_share_bundle --out D:/tmp/x   # somewhere else

What shrinks
------------
* `games_clean_300k.parquet` / `player_cohort.parquet` — `movetext` is blanked out except
  for a small keep-list (the candidates notebook 02 can pick from, plus a random sample
  for anyone poking around). 313 MB -> ~15 MB.
* `features_300k_v2.parquet` — copied verbatim; it is the modelling table and every
  column is used.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Filenames the notebooks hard-code. Keep these in sync with notebooks/*.ipynb.
FEATURES = "features_300k_v2.parquet"
GAMES_CLEAN = "games_clean_300k.parquet"
PLAYER_COHORT = "player_cohort.parquet"

# The candidate filter notebook 02 uses to pick its worked-example game. We keep the
# movetext of the first N candidates (sorted by game_id, as the notebook sorts) so the
# notebook's deterministic `.iloc[0]` pick still has its PGN, with headroom to spare.
N_EXAMPLE_GAMES = 500
N_RANDOM_GAMES = 500
RANDOM_SEED = 42

BLANK_NOTE = ""  # movetext for games not in the keep-list


def _processed_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    try:
        from src.config import load_config

        return Path(load_config()["paths"]["processed"])
    except Exception:  # config not importable (e.g. run from a bare checkout)
        return REPO_ROOT / "data" / "processed"


def _example_game_ids(features_path: Path) -> set[str]:
    """game_ids notebook 02 could pick as its worked example."""
    cols = ["game_id", "color", "has_endgame", "has_scramble", "game_plies", "blunder_count"]
    f = pd.read_parquet(features_path, columns=cols)
    cand = f[
        (f.color == "white")
        & (f.has_endgame == 1)
        & (f.has_scramble == 1)
        & f.game_plies.between(60, 90)
        & (f.blunder_count >= 1)
    ].sort_values("game_id")
    return set(cand["game_id"].head(N_EXAMPLE_GAMES))


def _slim_games(src: Path, dst: Path, keep_ids: set[str]) -> None:
    """Copy a games table, blanking movetext outside `keep_ids` (+ a random sample)."""
    df = pd.read_parquet(src)
    if "movetext" not in df.columns:
        df.to_parquet(dst, index=False)
        return

    extra = df["game_id"].sample(
        n=min(N_RANDOM_GAMES, len(df)), random_state=RANDOM_SEED
    )
    keep = keep_ids | set(extra)
    mask = df["game_id"].isin(keep)
    df.loc[~mask, "movetext"] = BLANK_NOTE
    df.to_parquet(dst, index=False)
    print(f"  movetext kept for {int(mask.sum()):,} / {len(df):,} games")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", help="source data/processed dir (default: from config.yaml)")
    ap.add_argument("--out", default=str(REPO_ROOT / "share_bundle"), help="output dir")
    args = ap.parse_args()

    proc = _processed_dir(args.processed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"source : {proc}\noutput : {out}\n")

    missing = [n for n in (FEATURES, GAMES_CLEAN, PLAYER_COHORT) if not (proc / n).exists()]
    if missing:
        raise SystemExit(f"missing in {proc}: {', '.join(missing)}")

    keep_ids = _example_game_ids(proc / FEATURES)
    print(f"notebook-02 example candidates: {len(keep_ids):,} game_ids")

    print(f"\n{GAMES_CLEAN}")
    _slim_games(proc / GAMES_CLEAN, out / GAMES_CLEAN, keep_ids)

    print(f"\n{PLAYER_COHORT}")
    _slim_games(proc / PLAYER_COHORT, out / PLAYER_COHORT, keep_ids)

    print(f"\n{FEATURES}\n  copied verbatim")
    shutil.copy2(proc / FEATURES, out / FEATURES)

    (out / "README.txt").write_text(READ_ME, encoding="utf-8")

    print("\nbundle contents")
    total = 0
    for p in sorted(out.iterdir()):
        total += p.stat().st_size
        print(f"  {p.name:<32} {p.stat().st_size / 1e6:8.1f} MB")
    print(f"  {'TOTAL':<32} {total / 1e6:8.1f} MB")


READ_ME = """chess-rating — slim data bundle for running notebooks 01-05
============================================================

1. Clone / pull the repo and set up the environment:

       python -m venv .venv
       .venv\\Scripts\\activate          # macOS/Linux: source .venv/bin/activate
       pip install -r requirements.txt

2. Copy the three .parquet files from this bundle into:

       <repo>/data/processed/

   Keep the filenames exactly as they are — the notebooks hard-code them.

3. Run notebooks 01 -> 05 top to bottom. Nothing else to configure; `config.yaml`
   and everything under `reports/` (eval_artifacts.json, results.md, figures/)
   come with the repo.

What is in here
---------------
  features_300k_v2.parquet   the modelling table, 606,578 instances — complete, unmodified
  games_clean_300k.parquet   303,289 games — complete, EXCEPT the `movetext` column, which
                             is blanked for all but ~1,000 games (see below)
  player_cohort.parquet      6,117 cohort games — same movetext treatment

Why movetext is trimmed
-----------------------
`movetext` is the raw PGN of every game and is ~95% of the file size. The notebooks read
`games_clean_300k.parquet` through column projections and only ever parse the movetext of
a single game — the worked example in notebook 02. That game (and several hundred other
valid candidates for it, plus a random sample) still has its full PGN here.

So: notebooks 01-05 run end to end and every number matches the report. What you CANNOT do
from this bundle is re-run `python -m src.features`, which needs the movetext of all games.
For that you need the full `games_clean_300k.parquet` (313 MB) — ask for it separately.

You do NOT need the raw Lichess dump (`data/raw/*.pgn.zst`, ~30 GB) for any of this.
"""


if __name__ == "__main__":
    main()
