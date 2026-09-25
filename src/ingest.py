"""Stage 1 — Ingest & sample.

Stream a zstandard-compressed Lichess monthly dump, keep only rated blitz games that carry
stored ``[%eval]`` annotations, reservoir-sample to a configurable size, and write a tidy parquet
with the game headers plus the raw movetext (evals/clocks preserved) for the feature stage to
re-parse.

Do this ONCE per shared sample: run it, commit the script, and share the parquet via a drive
(never git). See PROJECT_ARCHITECTURE.md and the schema contract in README.

Run:
    python -m src.ingest                          # uses config.yaml (month -> data/raw/<dump>.pgn.zst)
    python -m src.ingest --input tests/fixtures/sample.pgn.zst --output /tmp/smoke.parquet --sample-size 10

Performance: a monthly dump is ~30 GB compressed / 200+ GB uncompressed and is never fully
decompressed. ``stream_game_chunks`` splits the text stream into (headers, movetext) pairs
without building any chess.pgn object; cheap string/regex prefilters (mirroring the real
TimeControl/Termination/bot/eval checks) reject most games from that raw text alone. Only
games surviving the prefilters get a full ``chess.pgn.read_game`` SAN parse and the
authoritative header/eval/min-plies checks. This matters because only ~10% of blitz candidates
(non-bot, normal/time-forfeit termination) carry a stored ``[%eval]`` — skipping the SAN parse
for the other ~90% is where nearly all of the win comes from. The exact counts of every run
are written to ``reports/data_funnel.json`` (see ``write_funnel``).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
from pathlib import Path
from typing import Any, Iterator

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
# Streaming — cheap string-level chunker (no chess.pgn object built here)
# --------------------------------------------------------------------------------------
def stream_game_chunks(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield ``(headers_text, movetext_text)`` per game from a ``.pgn.zst`` dump, lazily
    decompressing and without invoking the (expensive) SAN parser.

    This is the raw-text counterpart of the old Visitor-based ``stream_games``: it just splits
    the line stream into per-game header/movetext blocks so callers can run cheap string
    prefilters before deciding whether a game is worth a full ``chess.pgn.read_game`` parse.

    Lichess dump structure per game: consecutive ``[Tag "value"]`` header lines, one or more
    blank lines, one or more movetext lines, then one or more blank lines before the next
    game's headers (or EOF). The state machine below is robust to multiple blank lines in
    either gap and to movetext spanning multiple lines.
    """
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh:
        reader = dctx.stream_reader(fh)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")

        header_lines: list[str] = []
        movetext_lines: list[str] = []
        state = "seek_header"  # "seek_header" -> "in_header" -> "in_movetext" -> (repeat)

        for line in text:
            stripped = line.strip()
            if state == "seek_header":
                if stripped.startswith("["):
                    header_lines = [line]
                    state = "in_header"
                # else: blank/junk line before the first game's headers -- ignore.
            elif state == "in_header":
                if stripped.startswith("["):
                    header_lines.append(line)
                elif stripped == "":
                    movetext_lines = []
                    state = "in_movetext"
                else:
                    # Defensive: no blank separator before movetext -- treat this line as the
                    # start of movetext rather than losing it.
                    movetext_lines = [line]
                    state = "in_movetext"
            else:  # state == "in_movetext"
                if stripped.startswith("["):
                    # A header line while collecting movetext means the next game has started
                    # (whether or not this game's movetext, or any blank-line gap, was empty).
                    yield ("".join(header_lines), "".join(movetext_lines))
                    header_lines = [line]
                    movetext_lines = []
                    state = "in_header"
                elif stripped == "":
                    continue  # blank line(s) before/within/after movetext -- keep waiting.
                else:
                    movetext_lines.append(line)

        # EOF: flush the last game, if any.
        if header_lines:
            yield ("".join(header_lines), "".join(movetext_lines))


# --------------------------------------------------------------------------------------
# String-level prefilters — regex/substring "necessary condition" mirrors of the authoritative
# checks below. Each can only REJECT a game the real check would also reject (same tag values,
# same logic), never accept one the real check would reject with certainty -- so gating the
# expensive SAN parse on these is behavior-preserving. The real checks still run, post-parse, as
# the source of truth for the funnel counts and output rows.
# --------------------------------------------------------------------------------------
# Anchored per-line, same tag-value grammar as python-chess's own TAG_REGEX (chess/pgn.py), so a
# duplicated tag is resolved the same way: python-chess's header dict keeps the LAST occurrence
# (plain assignment while scanning top to bottom), so we take the last regex match too -- taking
# the first would let this mirror reject a game the authoritative parse would accept.
_TIME_CONTROL_RE = re.compile(r'^\[TimeControl "([^\r\n]*)"\]\s*$', re.MULTILINE)
_TERMINATION_RE = re.compile(r'^\[Termination "([^\r\n]*)"\]\s*$', re.MULTILINE)
_UTC_DATE_RE = re.compile(r'^\[UTCDate "([^\r\n]*)"\]\s*$', re.MULTILINE)
_UTC_TIME_RE = re.compile(r'^\[UTCTime "([^\r\n]*)"\]\s*$', re.MULTILINE)


def header_timestamp(headers_text: str | None) -> str | None:
    """``"YYYY.MM.DD HH:MM:SS"`` from a raw header block (UTCDate + UTCTime), or None."""
    if not headers_text:
        return None
    dates, times = _UTC_DATE_RE.findall(headers_text), _UTC_TIME_RE.findall(headers_text)
    if not dates:
        return None
    return f"{dates[-1]} {times[-1]}" if times else dates[-1]


def string_is_blitz(headers_text: str, lo: int, hi: int) -> bool:
    """Mirrors ``is_blitz`` by regex-extracting ``TimeControl`` straight from the header text."""
    matches = _TIME_CONTROL_RE.findall(headers_text)
    time_control = matches[-1] if matches else None
    est = estimate_seconds(time_control)
    return est is not None and lo <= est <= hi


def string_termination_ok(headers_text: str, keep: list[str]) -> bool:
    """Mirrors ``termination_ok`` by regex-extracting ``Termination`` from the header text."""
    matches = _TERMINATION_RE.findall(headers_text)
    termination = matches[-1] if matches else None
    return termination in keep


def string_is_bot(headers_text: str) -> bool:
    """Mirrors ``is_bot`` via a literal substring check (BOT titles are never quote-escaped)."""
    return '[WhiteTitle "BOT"]' in headers_text or '[BlackTitle "BOT"]' in headers_text


def string_has_eval_hint(movetext_text: str) -> bool:
    """Necessary condition for ``has_eval``: Lichess evals are all-or-nothing, so if the game
    carries stored evals the first move's comment (and hence the raw movetext) contains the
    literal substring ``"[%eval"``. A missing substring guarantees ``has_eval`` is False.
    """
    return "[%eval" in movetext_text


def string_candidate_ok(
    headers_text: str, *, lo: int, hi: int, keep_terminations: list[str], exclude_bots: bool
) -> bool:
    """String-level mirror of ``header_ok`` (blitz + non-bot + termination), same branch order."""
    if not string_is_blitz(headers_text, lo, hi):
        return False
    if exclude_bots and string_is_bot(headers_text):
        return False
    if not string_termination_ok(headers_text, keep_terminations):
        return False
    return True


# --------------------------------------------------------------------------------------
# Filters — the authoritative checks, run on the parsed game
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
# Player cohort — deterministic hash-sampling of a small player subset, kept in FULL (every
# game, not reservoir-sampled). A uniform game sample has a per-player median of ~1 game, which
# starves the evaluation's aggregation curve (MAE vs K games/player); this complements it.
# --------------------------------------------------------------------------------------
def username_in_cohort(username: str | None, hash_rate: float) -> bool:
    """Deterministic player-level sampling: True iff a stable hash of ``username`` falls in
    the bottom ``hash_rate`` fraction of hash space.

    Stable across runs and machines (unlike Python's salted ``hash()``), so re-running ingest
    always keeps the same players. Lowercased first because Lichess usernames are
    case-insensitive.
    """
    if not username:
        return False
    digest = hashlib.md5(username.strip().lower().encode("utf-8")).hexdigest()
    value = int(digest[:8], 16)
    return value / 0xFFFFFFFF < hash_rate


# --------------------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------------------
def run_ingest(
    input_path: str | Path,
    output_path: str | Path,
    cfg: dict[str, Any],
    cohort_output_path: str | Path | None = None,
    funnel_path: str | Path | None = None,
) -> dict[str, int]:
    """Stream -> filter -> reservoir-sample -> parquet. Returns the data-funnel counts.

    ``funnel_path`` (optional): merge the counts plus the scanned UTC window into this JSON file
    under the ``"ingest"`` key (see ``write_funnel``).

    When ``cfg["player_cohort"]["enabled"]`` and ``cohort_output_path`` is given, ALSO writes a
    second parquet (same ``INGEST_COLUMNS`` schema) holding every filtered game of a small
    hash-sampled player subset (see ``username_in_cohort``) — kept unconditionally, not
    reservoir-sampled, so those players' full game histories survive. A game can legitimately
    land in both parquets. Backward compatible: with no ``player_cohort`` config or no
    ``cohort_output_path``, behavior is unchanged and no cohort file is written.
    """
    lo, hi = cfg["blitz_estimate_seconds"]
    min_plies = cfg["min_plies"]
    keep_terminations = cfg["keep_terminations"]
    exclude_bots = cfg["exclude_bots"]
    require_eval = cfg["use_stored_evals"]
    max_scanned = cfg.get("max_games_scanned")

    pc = cfg.get("player_cohort") or {}
    cohort_active = bool(pc.get("enabled")) and cohort_output_path is not None
    hash_rate = pc.get("hash_rate", 0.0)
    max_cohort_games = pc.get("max_games", 0)

    def header_ok(headers: Any) -> bool:
        """Authoritative check on real, parsed headers -- the source of truth post-parse."""
        if not is_blitz(headers, lo, hi):
            return False
        if exclude_bots and is_bot(headers):
            return False
        if not termination_ok(headers, keep_terminations):
            return False
        return True

    reservoir = Reservoir(cfg["sample_size"], cfg["random_seed"])
    cohort_rows: list[dict[str, Any]] = []
    funnel = {"scanned": 0, "blitz": 0, "candidate": 0, "has_eval": 0, "clean": 0, "cohort": 0}

    def should_stop() -> bool:
        return max_scanned is not None and funnel["scanned"] >= max_scanned

    # First/last scanned game's UTC timestamp: the dump is chronological, so a scan budget
    # (max_games_scanned) covers a contiguous window at the START of the month, not all of it.
    first_headers: str | None = None
    last_headers: str | None = None

    pbar = tqdm(desc="scanning games", unit="game")
    for headers_text, movetext_text in stream_game_chunks(input_path):
        funnel["scanned"] += 1
        pbar.update(1)
        if first_headers is None:
            first_headers = headers_text
        last_headers = headers_text

        if string_is_blitz(headers_text, lo, hi):
            funnel["blitz"] += 1

        # Cheap string-level mirror of header_ok, run BEFORE any SAN parse. "candidate" is
        # counted here rather than after the eval gate below, on purpose: today it counts every
        # blitz/non-bot/termination-ok game regardless of whether it has a stored eval, and
        # candidates without an eval hint never reach the parser at all (see below).
        if not string_candidate_ok(
            headers_text, lo=lo, hi=hi, keep_terminations=keep_terminations, exclude_bots=exclude_bots
        ):
            if should_stop():
                break
            continue
        funnel["candidate"] += 1

        # Gate the expensive SAN parse on the eval substring: ~90% of candidates lack a stored
        # eval and would be rejected by has_eval() right after parsing anyway -- this is where
        # nearly all of the speedup comes from.
        if require_eval and not string_has_eval_hint(movetext_text):
            if should_stop():
                break
            continue

        # headers_text already ends with the last header line's own newline; rstrip it first
        # so exactly one blank line separates headers from movetext. python-chess's parser
        # treats a run of >1 blank line there as an empty-movetext game (silently drops the
        # moves), so getting this exactly right matters for correctness, not just style.
        pgn_text = headers_text.rstrip("\n") + "\n\n" + movetext_text
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:  # defensive: shouldn't happen for a non-empty chunk
            if should_stop():
                break
            continue
        h = game.headers

        # Re-apply the authoritative checks on the real parsed game, exactly as before the
        # prefilter existed. The string-level checks above are necessary-condition mirrors, not
        # guarantees, so this remains the source of truth for has_eval/clean/output rows. Fail
        # closed: one unparseable/unexpected header must drop that game, not abort a multi-hour
        # unattended scan over a 30GB+ dump.
        try:
            keep = header_ok(h)
        except Exception:
            keep = False
        if not keep:
            if should_stop():
                break
            continue

        if require_eval and not has_eval(game):
            if should_stop():
                break
            continue
        funnel["has_eval"] += 1

        n_plies = count_plies(game)
        if n_plies < min_plies:
            if should_stop():
                break
            continue

        row = game_to_row(game, n_plies)
        if row is not None:
            funnel["clean"] += 1
            reservoir.add(row)
            if (
                cohort_active
                and len(cohort_rows) < max_cohort_games
                and (username_in_cohort(row["white"], hash_rate) or username_in_cohort(row["black"], hash_rate))
            ):
                cohort_rows.append(row)
                funnel["cohort"] += 1

        if should_stop():
            break
    pbar.close()

    funnel["sampled"] = len(reservoir.items)
    _write_parquet(reservoir.items, output_path)
    if cohort_active:
        _write_parquet(cohort_rows, cohort_output_path)
    _print_funnel(funnel, output_path, cohort_output_path if cohort_active else None)
    if funnel_path is not None:
        write_funnel(funnel_path, "ingest", {
            "input": Path(input_path).name,
            "month": cfg.get("month"),
            "max_games_scanned": cfg.get("max_games_scanned"),
            "sample_size": cfg["sample_size"],
            "random_seed": cfg["random_seed"],
            "scan_window_utc": {"first": header_timestamp(first_headers),
                                "last": header_timestamp(last_headers)},
            **funnel,
        })
    return funnel


def write_funnel(path: str | Path, stage: str, record: dict[str, Any]) -> None:
    """Merge one stage's funnel ``record`` into the JSON file at ``path`` under key ``stage``.

    Ingest and clean each own one key, so re-running one stage never erases the other's counts.
    The notebooks read this file instead of hand-typing the funnel.
    """
    path = Path(path)
    data: dict[str, Any] = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    data[stage] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


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


def _print_funnel(
    funnel: dict[str, int], output_path: str | Path, cohort_output_path: str | Path | None = None
) -> None:
    print("\nData funnel:")
    print(f"  scanned          : {funnel['scanned']:>10,}")
    print(f"  blitz            : {funnel['blitz']:>10,}")
    print(f"  candidate*       : {funnel['candidate']:>10,}   (blitz, non-bot, termination ok)")
    print(f"  has eval         : {funnel['has_eval']:>10,}")
    print(f"  clean            : {funnel['clean']:>10,}   (+ min_plies, has rating labels)")
    print(f"  sampled          : {funnel['sampled']:>10,}")
    print(f"  cohort           : {funnel['cohort']:>10,}   (all games of hash-sampled players)")
    print(f"  -> wrote {funnel['sampled']:,} rows to {output_path}")
    if cohort_output_path is not None:
        print(f"  -> wrote {funnel['cohort']:,} rows to {cohort_output_path}")


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Ingest & sample a Lichess blitz dump.")
    parser.add_argument("--input", default=cfg["raw_path"], help="path to the .pgn.zst dump")
    parser.add_argument("--output", default=cfg["outputs"]["blitz_sample"], help="output parquet path")
    parser.add_argument(
        "--cohort-output",
        default=cfg["outputs"].get("player_cohort"),
        help="output parquet path for the player-cohort sample (full game histories)",
    )
    parser.add_argument("--sample-size", type=int, default=None, help="override config sample_size")
    parser.add_argument("--max-games", type=int, default=None, help="override config max_games_scanned")
    parser.add_argument("--funnel", default=cfg["outputs"]["data_funnel"],
                        help="JSON file to record the funnel counts in (use '' to skip)")
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
    run_ingest(input_path, args.output, cfg, cohort_output_path=args.cohort_output,
               funnel_path=(args.funnel or None))


if __name__ == "__main__":
    main()
