"""Stage 2 (clean & filter) + Stage 3 (build instances).

Input : blitz_sample.parquet          (one row per game, reservoir sample from src.ingest)
        player_cohort.parquet         (optional: every in-window game of a hash-sampled player
                                       subset, from src.ingest; unioned with the sample)
Output: games_clean.parquet           (one row per game, deduped + label-sane)
        instances.parquet             (TWO rows per game, one per side)

Stage 2 appends the player cohort to the reservoir sample (sample rows first), removes duplicate
``game_id``s and games with unusable labels, and re-asserts the cheap row-level
filters (min plies, termination allowlist, decisive/draw result, rating in range). Most content
filtering already happened in ingest; this stage is the authoritative, re-runnable cleaning pass
and the place the data funnel is finalized (``reports/data_funnel.json``, key ``"clean"``).

Why union the cohort: a uniform game sample has a median of one game per player, so few players
have enough games to study aggregation. The cohort adds all in-window games of ~0.5% of players.

Stage 3 explodes each clean game into two INSTANCES — one per player — each described only by
*that* player's play and labelled with *that* player's rating.

LEAKAGE RULE (enforced by construction here): an instance row carries the player's own rating,
username, and per-player result, plus game-level context shared by both sides (time control,
opening, length). It NEVER carries the opponent's rating or username. Lichess matches similar
ratings, so any opponent-derived signal would leak the label. Split by ``username`` (grouped),
never by game — see PROJECT_ARCHITECTURE.md §Stage 7.

Provisional ratings cannot be filtered: provisional status is not present in exported Lichess
PGN (see README, Limitations).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config
from src.ingest import INGEST_COLUMNS, write_funnel

# One row per player. NO opponent rating / opponent username — that is the leakage guard.
INSTANCE_COLUMNS = [
    "game_id",
    "color",       # "white" | "black"
    "username",    # this player (grouping key for the grouped split)
    "rating",      # this player's rating — the regression LABEL
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


def union_with_cohort(sample_df: pd.DataFrame, cohort_df: pd.DataFrame | None) -> pd.DataFrame:
    """Reservoir sample followed by the player-cohort games (sample rows first), NOT deduplicated.

    A cohort game can also have been drawn into the reservoir sample; ``clean_games`` removes the
    second copy (``keep="first"``), so its funnel shows the overlap as ``input - deduped``. Row
    order is deterministic (sample order, then the cohort's new games in cohort order), which keeps
    the downstream parquets — and therefore every model fit — byte-for-byte reproducible.
    """
    if cohort_df is None or cohort_df.empty:
        return sample_df.reset_index(drop=True)
    return pd.concat([sample_df, cohort_df], ignore_index=True)


def clean_games(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Dedup + drop games with unusable labels/rows. Returns (clean_df, funnel counts)."""
    missing = [c for c in INGEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input parquet is missing schema-contract columns: {missing}")

    lo, hi = cfg["clean"]["min_rating"], cfg["clean"]["max_rating"]
    funnel = {"input": len(df)}

    df = df.drop_duplicates(subset="game_id", keep="first")
    funnel["deduped"] = len(df)

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
    cohort_path: str | Path | None = None,
    funnel_path: str | Path | None = None,
) -> dict[str, int]:
    """Read sample (+ cohort) -> union -> clean -> explode -> write games_clean + instances.

    Returns the funnel; with ``funnel_path`` it is also merged into that JSON under ``"clean"``.
    """
    sample = pd.read_parquet(input_path)
    cohort = pd.read_parquet(cohort_path) if cohort_path is not None else None
    df = union_with_cohort(sample, cohort)

    clean_df, funnel = clean_games(df, cfg)
    funnel = {"sample": len(sample), "cohort": 0 if cohort is None else len(cohort), **funnel}
    instances = build_instances(clean_df)

    Path(games_out).parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(games_out, engine="pyarrow", index=False)
    instances.to_parquet(instances_out, engine="pyarrow", index=False)

    funnel["instances"] = len(instances)
    funnel["players"] = int(instances["username"].nunique())
    _print_funnel(funnel, games_out, instances_out)
    if funnel_path is not None:
        write_funnel(funnel_path, "clean", funnel)
    return funnel


def _print_funnel(funnel: dict[str, int], games_out: str | Path, instances_out: str | Path) -> None:
    print("\nClean funnel:")
    print(f"  sample games     : {funnel['sample']:>10,}")
    print(f"  cohort games     : {funnel['cohort']:>10,}")
    print(f"  union (input)    : {funnel['input']:>10,}")
    print(f"  after dedup      : {funnel['deduped']:>10,}")
    print(f"  cleaned          : {funnel['cleaned']:>10,}   (min_plies, termination, result, rating range)")
    print(f"  instances        : {funnel['instances']:>10,}   (2 per clean game)")
    print(f"  distinct players : {funnel['players']:>10,}")
    print(f"  -> {games_out}")
    print(f"  -> {instances_out}")


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Clean games and build per-player instances.")
    parser.add_argument("--input", default=cfg["outputs"]["blitz_sample"], help="blitz_sample parquet")
    parser.add_argument("--cohort", default=cfg["outputs"]["player_cohort"],
                        help="player-cohort parquet to union in (use '' to skip)")
    parser.add_argument("--games-out", default=cfg["outputs"]["games_clean"], help="cleaned games parquet")
    parser.add_argument("--instances-out", default=cfg["outputs"]["instances"], help="instances parquet")
    parser.add_argument("--funnel", default=cfg["outputs"]["data_funnel"],
                        help="JSON file to record the funnel counts in (use '' to skip)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}\nRun `python -m src.ingest` first.")
    cohort_path = None
    if args.cohort and (cfg.get("player_cohort") or {}).get("enabled"):
        if not Path(args.cohort).exists():
            raise SystemExit(f"Cohort not found: {args.cohort}\n"
                             "Run `python -m src.ingest` (writes it), or pass --cohort ''.")
        cohort_path = args.cohort
    run_clean(input_path, args.games_out, args.instances_out, cfg,
              cohort_path=cohort_path, funnel_path=(args.funnel or None))


if __name__ == "__main__":
    main()
