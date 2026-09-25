# Chess Rating Estimation — Architecture & Design

The design decisions behind the pipeline: what we estimate, why blitz and stored evaluations, why
regression, and how each stage fits together. The README is the operational guide (setup, run
order, results); this document explains the *why*.

---

## 0. Decisions

| decision | choice | note |
|---|---|---|
| time control | blitz only | highest volume; avoids mixing rating pools (Lichess rates each speed separately) |
| evaluations | use games that already carry a stored Stockfish `[%eval]`, not our own engine runs | ~9% of blitz games; not a random sample (see Limitations) |
| time window | the first 15M games of one month (2025-05 → May 1–5, UTC) | assumes ratings drift little within a few days (not tested); the scan budget keeps ingest to ~35 min |
| target | regression on the player's rating (Lichess Glicko-2, "Elo points") | plus a 90% prediction interval; bands derived for a confusion matrix |
| sample | 300k games (reservoir) + a player cohort (~6k games) | configurable in `config.yaml` |
| model | gradient-boosted trees (LightGBM) | a neural sequence model is future work |
| evaluation | always split **by player**, never by game | a model must never be scored on a player it has seen |

---

## 1. Repository structure

```
chess-rating/
├── README.md                 # overview, results, requirement coverage, setup, run order
├── PROJECT_ARCHITECTURE.md   # this file
├── config.yaml               # every knob (window, sample size, thresholds, splits, seed, paths)
├── requirements.txt          # pinned dependencies (Python 3.11-3.14)
├── src/                      # the pipeline — importable, unit-tested
│   ├── config.py             # loads + validates config.yaml, resolves paths
│   ├── ingest.py             # Stage 1: stream .pgn.zst → filter → reservoir sample + cohort → parquet
│   ├── clean.py              # Stages 2-3: union, dedup, clean, explode into per-player instances
│   ├── features.py           # Stage 4: Lichess Win%/Accuracy%/classification + phase-split features
│   ├── model.py              # Stages 5-6: baselines, LightGBM, quantile models, conformal (CQR) math
│   ├── evaluate.py           # Stages 7-8: evaluation, aggregation, error analysis, bootstrap CIs
│   └── plotting.py           # shared matplotlib style + save_fig
├── tests/                    # unit + integration tests, and a tiny synthetic PGN fixture
├── notebooks/                # 01-05: the analysis narrative, committed with outputs
├── scripts/make_share_bundle.py   # slim data bundle for running the notebooks
├── reports/                  # results.md (experiment log), eval_artifacts.json, data_funnel.json,
│                             # largest_residuals.csv, RELATED_WORK.md, figures/
└── data/                     # GITIGNORED — raw/ (the dump) and processed/ (parquets); never committed
```

Rule of thumb: **notebooks are for looking; `src/` is for doing.** Anything run more than once, or
that another stage depends on, is a function in `src/`.

---

## 2. Data source

- File: `lichess_db_standard_rated_YYYY-MM.pgn.zst` from <https://database.lichess.org/> — every
  rated standard game of the month, all time controls, zstandard-compressed (2025-05: 30.7 GB).
- **Never decompressed to disk.** `ingest.py` stream-decompresses it, filters on the fly, and stops
  after `max_games_scanned` games. The dump is chronological, so the scan budget selects the start
  of the month; the exact UTC window is written to `reports/data_funnel.json`.
- Stockfish is not needed: evaluations are already in the PGN (`[%eval]`, white's point of view).
  Lichess states that about 6% of all games carry them.
- Ratings are Lichess **Glicko-2** ratings, stored in the `WhiteElo`/`BlackElo` tags.

---

## 3. The pipeline, stage by stage

```
raw .pgn.zst
  └─(1) ingest: filter + reservoir sample ─▶ blitz_sample.parquet   + player_cohort.parquet
        └─(2) clean: union, dedup, filters ─▶ games_clean.parquet
              └─(3) explode ─────────────────▶ instances.parquet    (2 rows/game: one per side)
                    └─(4) features ───────────▶ features.parquet
                          ├─(5) baselines ────▶ results.md
                          └─(6-8) evaluate ───▶ results.md, eval_artifacts.json, figures/, largest_residuals.csv
```

**Stage 1 — Ingest (`src/ingest.py`).** Splits the text stream into per-game (headers, movetext)
chunks without building `chess.pgn` objects. Cheap string/regex prefilters (mirrors of the real
TimeControl / Termination / bot / eval checks, which can only reject what the real check would
also reject) discard most games from the raw text. Only survivors get a full `chess.pgn` parse and
the authoritative checks: blitz (base + 40 × increment in 180–479 s, Lichess's definition), no bot,
Normal or Time-forfeit termination, a stored eval, ≥ 10 plies, both ratings present. Survivors are
**reservoir-sampled** (Algorithm R, seeded) to `sample_size`. A second output, the **player cohort**,
keeps every surviving game of a deterministic, hash-selected 0.5% of usernames — a uniform game
sample has a median of one game per player, which leaves little to study aggregation with.

**Stage 2 — Clean (`src/clean.py`).** Unions the sample with the cohort (deduplicated on
`game_id`), re-applies the row filters and a plausible-rating range, and records the funnel.

**Stage 3 — Instances (`src/clean.py`).** Explodes each game into **two rows**, one per side, each
labelled with *that* player's rating. It **never** attaches the opponent's rating or username:
Lichess pairs similar ratings, so any opponent-derived signal leaks the label.

**Stage 4 — Features (`src/features.py`).** Replays each game with `python-chess`, reads the eval and
clock of every move, and computes 73 numeric features per player (plus 4 categorical context
columns), overall and **per phase** (opening = plies 1–30; endgame = ≤ 6 non-pawn pieces left;
middlegame in between):

- *move quality:* centipawn loss (mean/median/std/max), mean per-move Accuracy% (Lichess formula),
  inaccuracy/mistake/blunder counts and rates (Lichess thresholds on the winning-chances drop),
  accuracy after ply 30 (`acc_after_book`), whether a winning position was reached and converted;
- *time:* move-time mean/std/median from `[%clk]`, share of very fast moves, share of moves in time
  trouble;
- *clock scramble:* quality and tempo of moves made with < 30 s left, and the CPL degradation vs
  the player's own game average;
- *style:* game length, the player's moves, captures, checks;
- *presence flags:* `has_opening` / `has_middlegame` / `has_endgame` / `has_clock` / `has_scramble`.

**Missing-value contract:** `NaN` means not observable; `0` is a true zero (a count, or a rate over a
non-empty set). A phase with no moves gets `n_moves = 0` and `NaN` for every other aggregate of that
phase. LightGBM handles `NaN` natively; nothing is imputed (except for ridge, which cannot take NaN).

**Stage 5 — Baselines (`src/model.py`).** No engine and no clock features: game shape, opening code,
own result, time control. Ridge and LightGBM, compared with predict-the-mean and copy-the-opponent
(the leak, reconstructed only to measure it).

**Stage 6 — Improved model (`src/evaluate.py` + `src/model.py`).** Players are split into train /
validation (early stopping) / calibration (post-hoc steps only) / test. LightGBM with an L1
objective on all features; optional Optuna search (grouped CV inside the training split). Three
quantile models (0.05 / 0.5 / 0.95) give a raw 90% interval, which **split CQR** corrects on the
calibration split; a **Mondrian** variant computes one correction per *predicted* band.

**Stage 7 — Evaluation (`src/evaluate.py`).** MAE, RMSE, median AE, R², Spearman, bias, within-100/200;
interval coverage (raw, CQR, Mondrian; overall and per band) and width, against a no-game interval;
band accuracy and confusion matrix; SHAP; **aggregation** — the shifting-cohort curve, the
**matched** curve (the same players at every K) and a **K-aware recalibration** of averaged
predictions fit on the calibration split; **player-clustered bootstrap** 95% CIs for the main MAE
comparisons.

**Stage 8 — Error analysis (`src/evaluate.py`).** Residual by rating band and by game length, the
largest misses, feature-group ablations, a scramble-block ablation, a leakage check (are test rows
whose opponent's row was trained on predicted better?), and a single-game tail study
(`--tail-study`: training reweighted toward rare ratings over 100-Elo bins, and the linear
calibration line fit on held-out data, vs the plain model).

---

## 4. Plotting conventions

One style (`src/plotting.set_style()`); every figure saved as PNG via `save_fig` (pipeline figures
to `reports/figures/`, notebook figures to `reports/figures/notebook_0X_*/`); one idea per chart;
figure generation in `evaluate.py` is best-effort (a failed plot logs a warning, the run continues).

## 5. Reproducibility

- `random_seed` fixed everywhere; `requirements.txt` pinned; every knob in `config.yaml`.
- `reports/results.md` is an append-only log: `src.model` / `src.evaluate` append a block headed by
  the commit hash (suffixed `-dirty` if `src/` or `config.yaml` had uncommitted changes).
- The whole pipeline, re-run from the raw dump (SHA-256 checked against Lichess's published sum),
  reproduces the processed parquets (feature values equal up to floating-point rounding) and every
  logged number (compare the 4657ac4 and aaeda0a entries in `reports/results.md`).
- Notebooks run top to bottom and are committed with outputs.

## 6. Future work

- Validate on a random sample of blitz games with our own fixed-depth evaluations (removes the
  selection bias of the eval'd subset).
- A sequence model over moves and clock (RatingNet-style) on the same grouped split.
- A player-level model that pools evidence across games directly, instead of recalibrating averages.
- Other windows and months (rating drift), and other time controls.
