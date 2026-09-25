"""Build a slim, shareable copy of the processed parquets for running the notebooks.

Why this exists
---------------
The raw dump (`data/raw/*.pgn.zst`) is ~30 GB and `data/processed/games_clean.parquet` is
~300 MB, almost all of it the raw `movetext` column. A collaborator who only wants to *run
notebooks 01-05* needs neither: the notebooks read `games_clean` through column projections
and parse `movetext` for exactly one game (the worked example in notebook 02).

The bundle uses the same filenames as `config.yaml -> outputs`, so the receiver copies them
into their own `data/processed/`. Nothing is resampled: every game, player and feature column
is preserved, so the notebooks reproduce the numbers in `reports/`.

    python scripts/make_share_bundle.py                  # -> share_bundle/
    python scripts/make_share_bundle.py --out D:/tmp/x   # somewhere else

What shrinks
------------
* `games_clean.parquet` / `player_cohort.parquet` — `movetext` is blanked except for a keep-list
  (the candidates notebook 02 can pick its worked example from, plus a random sample).
* `features.parquet` — copied verbatim; it is the modelling table and every column is used.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402

# Keep the movetext of the first N worked-example candidates (sorted by game_id, as notebook 02
# sorts) so its deterministic `.iloc[0]` pick still has its PGN, with headroom to spare.
N_EXAMPLE_GAMES = 500
N_RANDOM_GAMES = 500
RANDOM_SEED = 42


def example_candidates(features: pd.DataFrame) -> pd.DataFrame:
    """The worked-example candidates of notebook 02 — keep this filter identical to the notebook's:
    a white-side instance with all three phases, a clock scramble, 60-90 plies and >= 1 blunder."""
    return features[
        (features.color == "white")
        & (features.has_endgame == 1)
        & (features.has_scramble == 1)
        & features.game_plies.between(60, 90)
        & (features.blunder_count >= 1)
    ].sort_values("game_id")


def _slim_games(src: Path, dst: Path, keep_ids: set[str]) -> None:
    """Copy a games table, blanking movetext outside ``keep_ids`` (+ a random sample)."""
    df = pd.read_parquet(src)
    extra = df["game_id"].sample(n=min(N_RANDOM_GAMES, len(df)), random_state=RANDOM_SEED)
    mask = df["game_id"].isin(keep_ids | set(extra))
    df.loc[~mask, "movetext"] = ""
    df.to_parquet(dst, index=False)
    print(f"  movetext kept for {int(mask.sum()):,} / {len(df):,} games")


def main() -> None:
    cfg = load_config()
    outputs = cfg["outputs"]
    names = {key: Path(outputs[key]).name for key in ("features", "games_clean", "player_cohort")}

    ap = argparse.ArgumentParser(description="Build a slim data bundle for running the notebooks.")
    ap.add_argument("--processed", default=cfg["paths"]["processed"], help="source data/processed dir")
    ap.add_argument("--out", default=str(REPO_ROOT / "share_bundle"), help="output dir")
    args = ap.parse_args()

    proc, out = Path(args.processed), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"source : {proc}\noutput : {out}\n")
    missing = [n for n in names.values() if not (proc / n).exists()]
    if missing:
        raise SystemExit(f"missing in {proc}: {', '.join(missing)} — run the pipeline first")

    cols = ["game_id", "color", "has_endgame", "has_scramble", "game_plies", "blunder_count"]
    keep_ids = set(example_candidates(pd.read_parquet(proc / names["features"], columns=cols))
                   ["game_id"].head(N_EXAMPLE_GAMES))
    print(f"notebook-02 example candidates kept: {len(keep_ids):,}")

    for key in ("games_clean", "player_cohort"):
        print(f"\n{names[key]}")
        _slim_games(proc / names[key], out / names[key], keep_ids)
    print(f"\n{names['features']}\n  copied verbatim")
    shutil.copy2(proc / names["features"], out / names["features"])
    (out / "README.txt").write_text(READ_ME.format(pad="", **names), encoding="utf-8")

    total = 0
    print("\nbundle contents")
    for p in sorted(out.iterdir()):
        total += p.stat().st_size
        print(f"  {p.name:<28} {p.stat().st_size / 1e6:8.1f} MB")
    print(f"  {'TOTAL':<28} {total / 1e6:8.1f} MB")


READ_ME = """chess-rating: slim data bundle for running notebooks 01-05
===========================================================

1. Clone the repo and set up the environment (Python 3.11-3.14):

       python -m venv .venv
       .venv\\Scripts\\activate          # macOS/Linux: source .venv/bin/activate
       pip install -r requirements.txt

2. Copy the three .parquet files from this bundle into <repo>/data/processed/ and keep the
   filenames ({features}, {games_clean}, {player_cohort}); they match config.yaml.

3. Run notebooks 01 -> 05 top to bottom. Everything else they read (reports/eval_artifacts.json,
   reports/data_funnel.json, reports/results.md, reports/largest_residuals.csv) is in the repo.

What is in here
---------------
  {features:<22} the modelling table: complete and unmodified
  {games_clean:<22} every clean game, complete EXCEPT the `movetext` column, which is kept
  {pad:<22} for ~1,000 games only (notebook 02's worked-example candidates + a sample)
  {player_cohort:<22} the player-cohort games, same movetext treatment

With this bundle the notebooks run end to end and reproduce the reported numbers. What you
cannot do is re-run `python -m src.features` (needs every game's movetext); that requires the
full pipeline from the raw Lichess dump (see README).
"""


if __name__ == "__main__":
    main()
