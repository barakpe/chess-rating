"""Stage 2 (clean & filter) + Stage 3 (build instances).

Input : blitz_sample.parquet          (one row per game, from src.ingest)
Output: games_clean.parquet           (one row per game, deduped + label-sane)
        instances.parquet             (TWO rows per game, one per side)

Stage 2 removes duplicates and games with unusable labels, and re-asserts the cheap
row-level filters (min plies, termination allowlist, decisive/draw result, rating in range).
Most content filtering already happened in ingest; this stage is the authoritative,
re-runnable cleaning pass on the sampled parquet and the place the Data-slide funnel is finalized.

Stage 3 explodes each clean game into two INSTANCES — one per player — each described only by
*that* player's play and labelled with *that* player's rating.

LEAKAGE RULE (enforced by construction here): an instance row carries the player's own rating,
username, and per-player result, plus game-level context shared by both sides (time control,
opening, length). It NEVER carries the opponent's rating or username. Lichess matches similar
ratings, so any opponent-derived signal would leak the label. Split by ``username`` (grouped),
never by game — see PROJECT_ARCHITECTURE.md §Stage 7.

Provisional-rating dropping (config ``drop_provisional``) is a documented no-op: provisional
status is not present in exported Lichess PGN, so it cannot be filtered here. See README.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config
from src.ingest import INGEST_COLUMNS

# One row per player. NO opponent rating / opponent username — that is the leakage guard.
INSTANCE_COLUMNS = [
    "game_id",
    "color",       # "white" | "black"
    "username",    # this player (grouping key for the grouped split)
    "rating",      # this player's Elo — the regression LABEL
    "result",      # "win" | "loss" | "draw", from THIS player's point of view
    "time_control",
    "eco",
    "opening",
    "n_plies",
]

_VALID_RESULTS = ("1-0", "0-1", "1/2-1/2")


def _player_result(result: str, color: str) -> str | None:
    """Map the game result + player color to that player's outcome."""
    if result == "1/2-1/2":
        return "draw"
    if result == "1-0":
        return "win" if color == "white" else "loss"
    if result == "0-1":
        return "win" if color == "black" else "loss"
    return None  # unknown / "*" — filtered out upstream in clean_games


def clean_games(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Dedup + drop games with unusable labels/rows. Returns (clean_df, funnel counts)."""
    missing = [c for c in INGEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input parquet is missing schema-contract columns: {missing}")

    lo, hi = cfg["clean"]["min_rating"], cfg["clean"]["max_rating"]
    funnel = {"input": len(df)}

    df = df.drop_duplicates(subset="game_id", keep="first")
    funnel["deduped"] = len(df)

    # NOTE: drop_provisional is intentionally NOT applied — provisional status is absent from
    # exported PGN, so there is no field to filter on. Left as a documented no-op.

    df = df[df["n_plies"] >= cfg["min_plies"]]
    df = df[df["termination"].isin(cfg["keep_terminations"])]
    df = df[df["result"].isin(_VALID_RESULTS)]
    df = df[df["white_elo"].between(lo, hi) & df["black_elo"].between(lo, hi)]

    funnel["cleaned"] = len(df)
    return df.reset_index(drop=True), funnel


def build_instances(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Explode each clean game into two per-player instance rows (no opponent-derived fields)."""
    rows: list[dict[str, Any]] = []
    for g in clean_df.itertuples(index=False):
        for color, username, rating in (
            ("white", g.white, g.white_elo),
            ("black", g.black, g.black_elo),
        ):
            rows.append({
                "game_id": g.game_id,
                "color": color,
                "username": username,
                "rating": rating,
                "result": _player_result(g.result, color),
                "time_control": g.time_control,
                "eco": g.eco,
                "opening": g.opening,
                "n_plies": g.n_plies,
            })

    inst = pd.DataFrame(rows, columns=INSTANCE_COLUMNS)
    for col in ("game_id", "color", "username", "result", "time_control", "eco", "opening"):
        inst[col] = inst[col].astype("string")
    inst["rating"] = inst["rating"].astype("int16")
    inst["n_plies"] = inst["n_plies"].astype("int16")
    return inst


def run_clean(
    input_path: str | Path,
    games_out: str | Path,
    instances_out: str | Path,
    cfg: dict[str, Any],
) -> dict[str, int]:
    """Read blitz_sample -> clean -> explode -> write games_clean + instances. Returns funnel."""
    df = pd.read_parquet(input_path)
    clean_df, funnel = clean_games(df, cfg)
    instances = build_instances(clean_df)

    Path(games_out).parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(games_out, engine="pyarrow", index=False)
    instances.to_parquet(instances_out, engine="pyarrow", index=False)

    funnel["instances"] = len(instances)
    _print_funnel(funnel, games_out, instances_out)
    return funnel


def _print_funnel(funnel: dict[str, int], games_out: str | Path, instances_out: str | Path) -> None:
    print("\nClean funnel:")
    print(f"  input games      : {funnel['input']:>10,}")
    print(f"  after dedup      : {funnel['deduped']:>10,}")
    print(f"  cleaned          : {funnel['cleaned']:>10,}   (min_plies, termination, result, rating range)")
    print(f"  instances        : {funnel['instances']:>10,}   (2 per clean game)")
    print(f"  -> {games_out}")
    print(f"  -> {instances_out}")


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Clean games and build per-player instances.")
    parser.add_argument("--input", default=cfg["outputs"]["blitz_sample"], help="blitz_sample parquet")
    parser.add_argument("--games-out", default=cfg["outputs"]["games_clean"], help="cleaned games parquet")
    parser.add_argument("--instances-out", default=cfg["outputs"]["instances"], help="instances parquet")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}\nRun `python -m src.ingest` first.")
    run_clean(input_path, args.games_out, args.instances_out, cfg)


if __name__ == "__main__":
    main()
