# Chess Rating Estimation

Estimate a Lichess player's **blitz** rating from *how they play a single game* — a regression
problem (Elo point estimate + a per-game uncertainty interval), with rating bands derived for a
confusion matrix. We use games that already carry stored Stockfish `[%eval]` annotations
(Option A). Gradient-boosted trees are the graded core; a neural sequence model and cheating
detection are labeled stretch goals.

See [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md) for the full design rationale. This
README is the operational contract: how to set up, what to run in what order, the data interface
between stages, and the conventions all three of us follow.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                                # math regression + fixture smoke test
```

Python 3.10+. Stockfish is **not** needed (Option A reads evals already in the PGN).

---

## Repository layout

| Path | What |
|---|---|
| `config.yaml` | **Single source of truth** for every knob (month, sample size, thresholds, seed). No magic numbers in code. |
| `src/config.py` | Loads + validates `config.yaml`, resolves/creates paths. |
| `src/ingest.py` | Stage 1: stream the `.pgn.zst`, filter to blitz + has-eval, reservoir-sample, write parquet. |
| `src/features.py` | Verified Win%/Accuracy%/move-classification math + phase-split feature extraction (stub). |
| `tests/` | `test_features.py` (math), `test_ingest.py` (fixture smoke test), `fixtures/` (tiny synthetic dump). |
| `data/` | **Gitignored, never committed.** `raw/` = downloaded dumps, `processed/` = parquet outputs. |
| `notebooks/` | Exploration only; numbered, run top-to-bottom. Added in their phases. |
| `reports/figures/` | Saved PNGs the slides pull from. |

Rule of thumb: **notebooks are for looking; `src/` is for doing.** Anything run more than once or
depended on by another stage lives in `src/` as an importable function.

---

## Run order

Each stage reads `config.yaml` and hands the next a named parquet, so stages can be built in
parallel against the interface (not by reading each other's code).

| # | Command | Input | Output |
|---|---|---|---|
| 1 | `python -m src.ingest` | `data/raw/lichess_db_standard_rated_<month>.pgn.zst` | `data/processed/blitz_sample.parquet` |
| 2–3 | `python -m src.clean` | `blitz_sample.parquet` | `games_clean.parquet`, `instances.parquet` |
| 4 | `python -m src.features` | `games_clean.parquet` + `instances.parquet` | `features.parquet` |
| 5 | notebooks `01`→`04` *(later)* | `features.parquet` | figures + `reports/results.md` |

**Sanity gate:** run `pytest` before every push (math regression + the fixture smoke test).

### Getting the data
Download one monthly dump — `lichess_db_standard_rated_YYYY-MM.pgn.zst` — from
<https://database.lichess.org/> into `data/raw/`. A month is ~30 GB compressed / 200+ GB
uncompressed; **do not decompress it** — `ingest.py` stream-filters it and stops once sampled.
**Do this once:** one person runs ingest and shares `blitz_sample.parquet` via a drive (never git);
everyone else works from that file. To develop without the download, run against the committed
fixture:

```bash
python -m src.ingest --input tests/fixtures/sample.pgn.zst --output data/processed/smoke.parquet
```

---

## Parquet schema contract — `blitz_sample.parquet`

The exact columns `ingest.py` writes (defined once as `src.ingest.INGEST_COLUMNS`). This is the
interface Person A (ingest) hands to B/C (features/model).

| column | dtype | source | note |
|---|---|---|---|
| `game_id` | string | `Site` (URL) | stable per-game key |
| `event` | string | `Event` | e.g. "Rated Blitz game" |
| `white`, `black` | string | `White`/`Black` | usernames — for **grouped** train/test split later |
| `result` | string | `Result` | `1-0` / `0-1` / `1/2-1/2` |
| `white_elo`, `black_elo` | int16 | `WhiteElo`/`BlackElo` | the **labels** (one per side) |
| `white_rating_diff`, `black_rating_diff` | Int16 (nullable) | rating diffs | may be absent |
| `eco` | string | `ECO` | opening family |
| `opening` | string | `Opening` | opening name |
| `time_control` | string | `TimeControl` | `base+inc`, e.g. `300+0` |
| `termination` | string | `Termination` | one of `keep_terminations` |
| `utc_date`, `utc_time` | string | `UTCDate`/`UTCTime` | kept as strings |
| `movetext` | string | `StringExporter` | raw SAN + `[%eval]`/`[%clk]`; re-parsed by `features.py` |
| `n_plies` | int16 | computed | half-move count (post `min_plies` filter) |

`white_elo`/`black_elo` are the raw material for the per-player labels below — not model features.

## Parquet schema contract — `instances.parquet`

Stage 2–3 (`src/clean.py`) dedups games, drops unusable labels, and **explodes each game into two
rows, one per player** (defined once as `src.clean.INSTANCE_COLUMNS`). Each row describes only that
player and is labelled with that player's rating. `games_clean.parquet` keeps the full per-game rows
(same schema as `blitz_sample`) so the feature stage can fetch `movetext` by `game_id`.

| column | dtype | note |
|---|---|---|
| `game_id` | string | joins back to `games_clean` for the `movetext` |
| `color` | string | `white` / `black` |
| `username` | string | this player — the **grouped-split key** (never split by game) |
| `rating` | int16 | this player's Elo — the regression **label** |
| `result` | string | `win` / `loss` / `draw`, from this player's point of view |
| `time_control`, `eco`, `opening`, `n_plies` | string / int16 | game-level context shared by both sides |

**Leakage guard (enforced in code):** an instance row **never** carries the opponent's rating or
username. Lichess matches similar ratings, so any opponent-derived signal leaks the label. Always
split train/test by `username` (GroupKFold / grouped hold-out), never by game.

> **Provisional ratings** stay unfiltered here — the status isn't in exported PGN, so `config.yaml`'s
> `drop_provisional` is a documented no-op (see Limitations).

## `features.parquet` (Stage 4)

`src/features.py` parses each game once and emits **one row per instance** (`game_id, color,
username, rating, result, time_control, eco, opening` + ~60 numeric features). Built on the verified
`win_percent`/`accuracy_percent`/`classify_move` primitives; every quality feature is computed
overall **and** per phase (`opening_` / `middlegame_` / `endgame_`). Feature groups:

- **Move quality:** `cpl_{mean,median,std,max}` (centipawn loss), `acc_mean` (Lichess Accuracy%),
  `{inaccuracy,mistake,blunder}_{count,rate}`, and `acc_after_book` (post-opening accuracy — the
  "knew theory, then collapsed" signal). Move class thresholds are on the winningChances scale.
- **Time:** `move_time_{mean,std,median}` from `[%clk]` deltas, `fast_move_share`, `time_trouble_share`.
- **Style:** `game_plies`, `player_moves`, `n_captures`, `n_checks`, `reached_winning`,
  `converted_winning` (did they win from a winning position?).

Same leakage guarantee: the eval before/after a move is taken from *this* player's POV only; no
opponent-derived quantity enters a feature. Cross-game aggregates (opening-diversity entropy, the
per-K aggregation curve) are deliberately **not** here — they belong to evaluation (Stage 7), since a
single-game prediction can't see a player's other games.

---

## Conventions (all three of us)

### Never commit
Data, models, or large/binary artifacts. `.gitignore` covers `data/`, `models/`, `*.pgn`,
`*.pgn.zst`, `*.zst`, `*.parquet`. The one exception is the tiny `tests/fixtures/` dump. Small
result PNGs in `reports/figures/` may be committed; large ones, don't.

### Git workflow
- `main` always stays runnable. Work on **feature branches** named by area: `feat/…`, `eda/…`,
  `model/…` (e.g. `feat/phase-split-features`).
- Merge via **PR with ≥1 teammate review**. No direct pushes to `main`.
- **Commits:** small, atomic, imperative mood, with a prefix:
  `feat:` `fix:` `docs:` `refactor:` `chore:` `exp:` (experiment runs) `data:` (pipeline/schema).
  Example: `feat: add phase-split centipawn-loss features`.
- Add a dependency and update `requirements.txt` in the **same PR**.
- **Notebook hygiene:** clear outputs before committing (`nbstripout` or "Clear All Outputs") so
  diffs stay readable. Keep notebooks thin; shared logic goes in `src/`.
- **Tag milestones:** `v0.1-baseline`, `v0.2-improved`, …

### Reproducibility
Fix `random_seed` everywhere; pin `requirements.txt`; keep every knob in `config.yaml`. Notebooks
must run top-to-bottom (Restart & Run All) before they're considered done.

---

## Limitations (state these in the write-up)

- **Option-A selection bias:** games with stored evals were chosen by players for analysis, so the
  eval'd subset (~6% of games) isn't a random sample. Backlog: validate against an Option-B random
  sample with self-computed fixed-depth evals.
- **Provisional ratings not filterable:** Lichess strips provisional status from exported PGN (no
  `?` marker, no flag), so `config.yaml`'s `drop_provisional` is a **documented no-op** — we can't
  drop provisional-rated (noisier) labels from the dump. Recorded so the intent is explicit.
- **Single-game noise floor:** one blitz game can't pin a rating; precision comes from aggregating
  across a player's games. This is a finding, not a bug.
- **Scope:** blitz only; results may not transfer to rapid/classical.
