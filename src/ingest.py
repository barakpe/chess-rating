"""Stage 1 — Ingest & sample.

Stream a zstandard-compressed Lichess monthly dump, keep only rated blitz games that carry
stored ``[%eval]`` annotations (Option A), reservoir-sample to a configurable size, and write a
tidy parquet with the game headers plus the raw movetext (evals/clocks preserved) for the
feature stage to re-parse.

Do this ONCE per shared sample: one person runs it, commits the script, and shares the parquet
via a drive (never git). See PROJECT_ARCHITECTURE.md §Stage 1 and the schema contract in README.

Run:
    python -m src.ingest                          # uses config.yaml (month -> data/raw/<dump>.pgn.zst)
    python -m src.ingest --input tests/fixtures/sample.pgn.zst --output /tmp/smoke.parquet --sample-size 10

Performance: a monthly dump is ~30 GB compressed / 200+ GB uncompressed and is never fully
decompressed. Header filters run via a Visitor that returns ``chess.pgn.SKIP`` for non-blitz /
bot / bad-termination games, so the movetext of the majority of games is never parsed; only
candidate games get a full tree build and the has-eval / min-plies checks.
"""

from __future__ import annotations

import argparse
import functools
import io
import random
from pathlib import Path
from typing import Any, Callable, Iterator

import chess.pgn
import pandas as pd
import zstandard
from tqdm import tqdm

from src.config import load_config

# The exact columns written to blitz_sample.parquet — the interface contract with the feature/
# model stages (see README "Parquet schema contract"). Order is stable.
INGEST_COLUMNS = [
    "game_id",
    "event",
    "white",
    "black",
    "result",
    "white_elo",
    "black_elo",
    "white_rating_diff",
    "black_rating_diff",
    "eco",
    "opening",
    "time_control",
    "termination",
    "utc_date",
    "utc_time",
    "movetext",
    "n_plies",
]

_STRING_COLUMNS = [
    "game_id", "event", "white", "black", "result", "eco", "opening",
    "time_control", "termination", "utc_date", "utc_time", "movetext",
]


# --------------------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------------------
def stream_games(path: str | Path, header_ok: Callable[[Any], bool]) -> Iterator[chess.pgn.Game]:
    """Yield parsed games from a ``.pgn.zst`` dump, lazily decompressing.

    ``header_ok(headers)`` decides, from the headers alone, whether a game's movetext is worth
    parsing. Games that fail it are skipped (via ``chess.pgn.SKIP``) but still yielded with their
    headers populated and no moves — so the caller can keep accurate funnel counts.
    """
    visitor = functools.partial(_FilteringGameBuilder, header_ok=header_ok)
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh:
        reader = dctx.stream_reader(fh)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        while True:
            game = chess.pgn.read_game(text, Visitor=visitor)
            if game is None:  # end of file
                break
            yield game


class _FilteringGameBuilder(chess.pgn.GameBuilder):
    """GameBuilder that skips movetext for games failing the header filter.

    Headers are fully populated by the time ``end_headers`` is called, so we can decide there.
    Returning ``chess.pgn.SKIP`` makes python-chess skip the (expensive) movetext parse.
    """

    def __init__(self, *, header_ok: Callable[[Any], bool], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._header_ok = header_ok

    def end_headers(self):  # type: ignore[override]
        skip = super().end_headers()
        try:
            keep = self._header_ok(self.game.headers)
        except Exception:
            keep = False
        return skip if keep else chess.pgn.SKIP


# --------------------------------------------------------------------------------------
# Filters (see verified facts in PROJECT_ARCHITECTURE.md / plan)
# --------------------------------------------------------------------------------------
def estimate_seconds(time_control: str | None) -> int | None:
    """Lichess estimated game duration = base + 40 * increment (seconds). None if no clock."""
    if not time_control or time_control == "-":
        return None
    base, _, inc = time_control.partition("+")
    try:
        return int(base) + 40 * int(inc or 0)
    except ValueError:
        return None


def is_blitz(headers: Any, lo: int, hi: int) -> bool:
    """Blitz iff base + 40*inc is within [lo, hi] (scalachess Speed ranges: 180..479)."""
    est = estimate_seconds(headers.get("TimeControl"))
    return est is not None and lo <= est <= hi


def is_bot(headers: Any) -> bool:
    return headers.get("WhiteTitle") == "BOT" or headers.get("BlackTitle") == "BOT"


def termination_ok(headers: Any, keep: list[str]) -> bool:
    return headers.get("Termination") in keep


def has_eval(game: chess.pgn.Game) -> bool:
    """True if the game carries stored evals. Lichess is all-or-nothing, so the first move suffices."""
    first = game.next()
    return first is not None and first.eval() is not None


def count_plies(game: chess.pgn.Game) -> int:
    return sum(1 for _ in game.mainline_moves())


# --------------------------------------------------------------------------------------
# Row building
# --------------------------------------------------------------------------------------
_EXPORTER_KWARGS = dict(headers=False, variations=False, comments=True)


def _to_int(headers: Any, key: str) -> int | None:
    value = headers.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def game_to_row(game: chess.pgn.Game, n_plies: int) -> dict[str, Any] | None:
    """Flatten a game to the schema-contract row, or None if it lacks usable rating labels."""
    h = game.headers
    white_elo = _to_int(h, "WhiteElo")
    black_elo = _to_int(h, "BlackElo")
    if white_elo is None or black_elo is None:
        return None  # no label -> unusable

    # StringExporter reconstructs SAN + [%eval]/[%clk] comments; features.py re-parses this.
    movetext = game.accept(chess.pgn.StringExporter(**_EXPORTER_KWARGS))

    return {
        "game_id": h.get("Site"),
        "event": h.get("Event"),
        "white": h.get("White"),
        "black": h.get("Black"),
        "result": h.get("Result"),
        "white_elo": white_elo,
        "black_elo": black_elo,
        "white_rating_diff": _to_int(h, "WhiteRatingDiff"),
        "black_rating_diff": _to_int(h, "BlackRatingDiff"),
        "eco": h.get("ECO"),
        "opening": h.get("Opening"),
        "time_control": h.get("TimeControl"),
        "termination": h.get("Termination"),
        "utc_date": h.get("UTCDate"),
        "utc_time": h.get("UTCTime"),
        "movetext": movetext,
        "n_plies": n_plies,
    }


# --------------------------------------------------------------------------------------
# Reservoir sampling (Algorithm R) — unbiased uniform sample over the filtered stream
# --------------------------------------------------------------------------------------
class Reservoir:
    def __init__(self, k: int, seed: int) -> None:
        self.k = k
        self._rng = random.Random(seed)
        self.items: list[Any] = []
        self.seen = 0

    def add(self, item: Any) -> None:
        self.seen += 1
        if len(self.items) < self.k:
            self.items.append(item)
        else:
            j = self._rng.randint(0, self.seen - 1)
            if j < self.k:
                self.items[j] = item


# --------------------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------------------
def run_ingest(input_path: str | Path, output_path: str | Path, cfg: dict[str, Any]) -> dict[str, int]:
    """Stream -> filter -> reservoir-sample -> parquet. Returns the data-funnel counts."""
    lo, hi = cfg["blitz_estimate_seconds"]
    min_plies = cfg["min_plies"]
    keep_terminations = cfg["keep_terminations"]
    exclude_bots = cfg["exclude_bots"]
    require_eval = cfg["use_stored_evals"]
    max_scanned = cfg.get("max_games_scanned")

    def header_ok(headers: Any) -> bool:
        if not is_blitz(headers, lo, hi):
            return False
        if exclude_bots and is_bot(headers):
            return False
        if not termination_ok(headers, keep_terminations):
            return False
        return True

    reservoir = Reservoir(cfg["sample_size"], cfg["random_seed"])
    funnel = {"scanned": 0, "blitz": 0, "candidate": 0, "has_eval": 0, "clean": 0}

    pbar = tqdm(desc="scanning games", unit="game")
    for game in stream_games(input_path, header_ok):
        funnel["scanned"] += 1
        pbar.update(1)

        h = game.headers
        if is_blitz(h, lo, hi):
            funnel["blitz"] += 1

        # header_ok games had their movetext parsed; the rest were SKIP-ped (no moves).
        if not header_ok(h):
            if max_scanned is not None and funnel["scanned"] >= max_scanned:
                break
            continue
        funnel["candidate"] += 1

        if require_eval and not has_eval(game):
            if max_scanned is not None and funnel["scanned"] >= max_scanned:
                break
            continue
        funnel["has_eval"] += 1

        n_plies = count_plies(game)
        if n_plies < min_plies:
            if max_scanned is not None and funnel["scanned"] >= max_scanned:
                break
            continue

        row = game_to_row(game, n_plies)
        if row is not None:
            funnel["clean"] += 1
            reservoir.add(row)

        if max_scanned is not None and funnel["scanned"] >= max_scanned:
            break
    pbar.close()

    funnel["sampled"] = len(reservoir.items)
    _write_parquet(reservoir.items, output_path)
    _print_funnel(funnel, output_path)
    return funnel


def _write_parquet(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=INGEST_COLUMNS)
    for col in _STRING_COLUMNS:
        df[col] = df[col].astype("string")
    df["white_elo"] = df["white_elo"].astype("int16")
    df["black_elo"] = df["black_elo"].astype("int16")
    df["white_rating_diff"] = df["white_rating_diff"].astype("Int16")
    df["black_rating_diff"] = df["black_rating_diff"].astype("Int16")
    df["n_plies"] = df["n_plies"].astype("int16")
    df = df[INGEST_COLUMNS]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)


def _print_funnel(funnel: dict[str, int], output_path: str | Path) -> None:
    print("\nData funnel:")
    print(f"  scanned          : {funnel['scanned']:>10,}")
    print(f"  blitz            : {funnel['blitz']:>10,}")
    print(f"  candidate*       : {funnel['candidate']:>10,}   (blitz, non-bot, termination ok)")
    print(f"  has eval         : {funnel['has_eval']:>10,}")
    print(f"  clean            : {funnel['clean']:>10,}   (+ min_plies, has rating labels)")
    print(f"  sampled          : {funnel['sampled']:>10,}")
    print(f"  -> wrote {funnel['sampled']:,} rows to {output_path}")


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Ingest & sample a Lichess blitz dump.")
    parser.add_argument("--input", default=cfg["raw_path"], help="path to the .pgn.zst dump")
    parser.add_argument("--output", default=cfg["outputs"]["blitz_sample"], help="output parquet path")
    parser.add_argument("--sample-size", type=int, default=None, help="override config sample_size")
    parser.add_argument("--max-games", type=int, default=None, help="override config max_games_scanned")
    args = parser.parse_args()

    if args.sample_size is not None:
        cfg["sample_size"] = args.sample_size
    if args.max_games is not None:
        cfg["max_games_scanned"] = args.max_games

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(
            f"Input dump not found: {input_path}\n"
            f"Download lichess_db_standard_rated_{cfg['month']}.pgn.zst from "
            f"https://database.lichess.org/ into {cfg['paths']['raw']}, or pass --input."
        )
    run_ingest(input_path, args.output, cfg)


if __name__ == "__main__":
    main()
