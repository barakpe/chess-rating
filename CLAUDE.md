# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Estimate a Lichess player's blitz rating from how they play a single game — a regression problem
(Elo point estimate + a per-game uncertainty interval), with rating bands derived for a confusion
matrix. Gradient-boosted trees (LightGBM) are the core model. See `PROJECT_ARCHITECTURE.md` for
full design rationale and `README.md` for the operational contract (schema tables, conventions).

## Commands

Setup:
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tests (plain pytest, no config file — run from repo root; `conftest.py` puts the root on
`sys.path` so `src` is importable):
```bash
pytest                                    # everything
pytest tests/test_model.py                # one file
pytest tests/test_model.py::test_metrics  # one test
pytest -k conformal                       # by keyword
```

There is no linter/formatter configured in this repo (no pyproject.toml/ruff/flake8) — don't
invent lint commands.

Pipeline stages, in order (each reads `config.yaml` itself and reads/writes a named parquet under
`data/processed/`, which is gitignored):
```bash
python -m src.ingest                     # .pgn.zst -> blitz_sample.parquet (+ player_cohort.parquet)
python -m src.clean                      # blitz_sample.parquet -> games_clean.parquet, instances.parquet
python -m src.features                   # games_clean + instances -> features.parquet
python -m src.model                      # baseline (ridge/LightGBM) -> reports/results.md
python -m src.evaluate [--tune]          # full model + eval + figures -> reports/results.md
python -m src.evaluate --tail-study      # tail-bias correction comparison only
```

Developing without the real ~30GB monthly dump — run ingest against the committed fixture:
```bash
python -m src.ingest --input tests/fixtures/sample.pgn.zst --output data/processed/smoke.parquet
```

## Architecture

**Five stages connected by parquet files, not shared imports.** `ingest.py` → `clean.py` →
`features.py` → `model.py` → `evaluate.py`. Each stage's `main()` calls `src.config.load_config()`
independently and threads `cfg` through — there's no cross-stage state beyond the parquet on disk,
so you can generally read one stage's code without the others, but you do need the schema contract
between them (see the "Parquet schema contract" tables in README). `evaluate.py` is the orchestrator
for the improved-model path: it imports the actual model-fitting/conformal-math functions from
`model.py` rather than reimplementing them.

**The leakage rule is enforced by construction, not just convention.** `instances.parquet` /
`features.parquet` never carry the opponent's rating or username (Lichess matches similar ratings,
so any opponent-derived signal would leak the label) — see `clean.INSTANCE_COLUMNS` and
`features.extract_player_features`'s docstring. All train/test/calibration splitting is grouped by
`username`, never by game: `model.grouped_split` / `model.grouped_train_val_calib_test`.
`model.add_opponent_rating` exists ONLY to reconstruct the "copy-opponent" comparison baseline
(shown to be strong precisely to justify excluding it) — it is never fed to a model.

**`config.yaml` is the single source of truth; `src/config.py` resolves it once.**
`load_config()` validates required keys, turns repo-relative paths into absolute ones, and creates
the directories, so downstream code never hardcodes a path/threshold/seed. Every stage's `main()`
calls `load_config()` itself (there's also a module-level `CONFIG` singleton for convenience, but
prefer `load_config()` when a test needs an isolated/overridden config).

**`features.py`'s missing-value contract is cross-cutting.** NaN means "not observable"; 0 is
reserved for a true zero or an exposure count (`n_moves`, `n_timed_moves`). A phase/clock block
that's absent gets `n_moves = 0` and NaN for every other aggregate in it — a rate is only
meaningful conditional on exposure. Presence flags (`has_opening`, `has_middlegame`, `has_endgame`,
`has_clock`, `has_scramble`) make that absence directly splittable for LightGBM, which consumes NaN
natively. `model.make_ridge` is the one place that imputes (median) — Ridge can't take NaN, unlike
the LightGBM models.

**Conformal prediction (split CQR) spans `model.py` (math) and `evaluate.py` (orchestration).**
`model.py` has a plain path (`conformal_correction` / `apply_conformal`, one global correction) and
a Mondrian, band-conditional path (`conformal_correction_by_band` / `apply_conformal_by_band`, one
correction per *predicted*-median band) — both share the same `to_bands` bucketing and the same
`apply_conformal` clip logic, so there is only one implementation of each. `evaluate.run_evaluation`
computes both every run and picks the headline one via `config.yaml`'s `model.mondrian_cqr`. The
calibration split (`calib`, carved out by `grouped_train_val_calib_test`) is never seen by any model
fit — it exists solely for this correction step.

**`ingest.py` runs a two-tier filter for speed.** Cheap string/regex prefilters
(`string_is_blitz`, `string_candidate_ok`, etc.) reject most of the raw text stream before any
`chess.pgn` parsing happens; only survivors get the expensive SAN parse and the authoritative,
post-parse checks (`is_blitz`, `header_ok`, `has_eval`). The string-level functions are deliberately
one-way mirrors (they can only reject what the authoritative check would also reject) — the
authoritative checks remain the source of truth for the funnel counts and output rows.

**`reports/results.md` is an append-only experiment log, not a hand-edited doc.**
`python -m src.model` / `python -m src.evaluate` append a `## <commit-hash> — <description>` block
automatically (`model.current_commit_hash` + `model._append_md`), so any historical entry can be
traced back to the exact code that produced it. Don't hand-edit past entries.

**Notebooks are for looking, not doing.** Shared logic always lives in `src/` as an importable
function; a notebook only calls into it and saves figures via `plotting.save_fig`. Notebooks don't
exist yet — see `notebooks/NOTEBOOK_PLAN.md` for the planned five and what each must show.
