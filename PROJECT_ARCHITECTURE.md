# Chess Rating Estimation — Architecture & Setup

A working reference for the team. Estimate a player's Lichess rating from how they
play, using **blitz** games with **stored Stockfish evals**, **regression** as the
primary target (point estimate + per-game uncertainty interval), bands derived for a
confusion matrix. Neural sequence model and cheating detection are clearly-labeled
stretch goals.

---

## 0. Decisions locked in

| Decision | Choice | Note |
|---|---|---|
| Time control | Blitz only | Highest volume; avoids cross-format rating confound |
| Evals | **Option A** — use games that already carry `[%eval]` | ~6% of games, but millions in one blitz month |
| Time window | One recent month | Keeps the rating label stable (no drift) |
| Target | Regression (Elo) | + quantile interval; bands derived for a confusion matrix |
| Sample size | Configurable, start ~300k | Scale up later on a stronger machine |
| Neural model | Deferred (stretch) | Gradient-boosted trees are the graded core |
| Disclaimer | State the Option-A selection bias in the paper | + backlog item to revisit with Option B |

---

## 1. Repository structure

```
chess-rating/
├── README.md                 # what it is, how to reproduce, run order
├── requirements.txt          # pinned dependencies
├── config.yaml               # all knobs (sample size, month, thresholds, seed)
├── .gitignore                # ignores data/, models/, *.zst, *.parquet
├── data/                     # GITIGNORED — never committed
│   ├── raw/                  # downloaded .pgn.zst
│   └── processed/            # filtered/cleaned/feature parquet files
├── notebooks/                # exploration only; numbered, run top-to-bottom
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_baseline.ipynb
│   └── 04_improved_eval.ipynb
├── src/                      # reusable, importable pipeline code
│   ├── config.py             # loads config.yaml
│   ├── ingest.py             # stream .zst → filter → sample → parquet
│   ├── clean.py              # filtering/dedup/instance building
│   ├── features.py           # phase-split feature extraction
│   ├── model.py              # baseline + quantile regression
│   ├── evaluate.py           # metrics, aggregation curve, calibration, SHAP
│   └── plotting.py           # shared matplotlib style + chart helpers
├── reports/
│   ├── figures/              # saved PNGs the slides pull from
│   └── results.md            # experiment log (MAE table, what changed)
└── slides/                   # the 15–25 min presentation
```

Rule of thumb: **notebooks are for looking; `src/` is for doing.** Anything you'll
run more than once or that another stage depends on goes into `src/` as a function,
imported by the notebook. This keeps the pipeline reproducible and the notebooks thin.

---

## 2. Environment & what to download

**Python packages** (`requirements.txt`, pin versions):
`python-chess` (PGN parsing, board reconstruction, reading evals), `zstandard`
(decompress `.pgn.zst`), `pandas`, `numpy`, `pyarrow` (parquet I/O), `scikit-learn`,
`lightgbm`, `shap`, `matplotlib`, `seaborn`, `pyyaml`, `tqdm`. Optional later:
`optuna` (tuning), `mapie` or `crepes` (conformal intervals), `torch` (neural stretch).

**Stockfish** — **not needed for Option A** (evals are already in the PGN). Only
install it if you later switch to Option B (computing evals yourself).

**The data — one Lichess monthly dump:**
- File: `lichess_db_standard_rated_YYYY-MM.pgn.zst` from `https://database.lichess.org/`.
- It is **all time controls mixed** and compressed with zstandard. A month is roughly
  ~30 GB compressed / ~200+ GB uncompressed — so **do not decompress the whole thing.**
- **Stream-decompress and filter on the fly**, keeping only blitz + has-eval games, and
  **stop once you've sampled enough.** You end up writing a few-hundred-MB parquet, not
  200 GB. `python-chess` reads directly from a `zstandard` stream.
- **Do this once.** One person produces the filtered `processed/blitz_sample.parquet`,
  commits the *script*, and shares the *parquet* via a shared drive (not git). Everyone
  else works from that file. This avoids three people each downloading 30 GB.
- For building the pipeline skeleton before the big download, prototype on the
  `datasnaek/chess` 20k CSV (no evals — enough to wire up cleaning + a no-eval baseline).

**`config.yaml`** (single source of truth — every script reads this):
```yaml
month: "2024-06"
time_control: "blitz"          # event/timecontrol filter
use_stored_evals: true         # Option A; flip to false for Option B later
sample_size: 300000            # bump up on a stronger machine
min_plies: 10                  # drop aborted / ultra-short games
drop_provisional: true         # provisional ratings are noisy LABELS
exclude_bots: true
rating_bands: [0, 1200, 1400, 1600, 1800, 2000, 3000]
random_seed: 42
paths:
  raw: data/raw
  processed: data/processed
  figures: reports/figures
```

---

## 3. The pipeline (stage by stage)

```
raw .pgn.zst
  └─(1) ingest+filter+sample─▶ blitz_sample.parquet
        └─(2) clean─────────▶ games_clean.parquet
              └─(3) explode──▶ instances.parquet      (2 rows/game: one per side)
                    └─(4) features─▶ features.parquet
                          └─(5/6) model─▶ predictions + model.pkl
                                └─(7/8) evaluate+error─▶ reports/figures + results.md
```

**Stage 1 — Ingest & sample (`src/ingest.py`).** Stream the `.zst`, parse each game,
keep it only if: rated, blitz, standard variant, in the chosen month, and (Option A)
contains an `[%eval]` on the first move. Reservoir-sample to `sample_size`. Output a
tidy parquet with headers + the raw move/eval/clock string.

**Stage 2 — Clean & filter (`src/clean.py`).** Drop games under `min_plies`, abnormal
terminations (abandoned/rules-infraction), and games where either side has a
**provisional** rating (bad labels). Drop bot accounts. Deduplicate. Keep a **data
funnel** count (raw → blitz → has-eval → cleaned → sampled) for the Data slide.

**Stage 3 — Build instances (`src/clean.py`).** Explode each game into **two rows**,
one per side, each described by *that player's* play, labeled with *that player's*
rating. **Do not** add opponent rating or opponent-derived stats (leakage — Lichess
matches similar ratings, so they leak the answer). Record the username for grouped
splitting and per-player aggregation.

**Stage 4 — Feature engineering (`src/features.py`) — the heart.** Replay each game
with `python-chess`, read the per-move evals, and compute per-player features,
**split by phase** (opening / middlegame / endgame):

- *Move quality:* mean & median centipawn loss; **mean Accuracy%** (Lichess's published
  Win%→accuracy formula, which is position-independent and more meaningful than raw
  centipawns); std of centipawn loss (consistency); worst single move.
- *Error counts:* inaccuracies / mistakes / blunders (classify by the Win% drop per
  move, Lichess-style) and their per-move rates.
- *Phase split:* all of the above separately for opening / middlegame / endgame, **plus
  "accuracy after leaving the opening book"** — this is what captures the "knew theory
  for 20 moves, then collapsed" case.
- *Opening knowledge:* book depth (`opening_ply`), opening diversity (entropy across a
  player's games), ECO family.
- *Time management:* mean/variance of move times, share of moves played very fast,
  time-trouble behaviour (moves with little clock left).
- *Decisiveness / style:* game length, result, captures/checks counts, whether they
  converted winning positions.

Phase boundaries: opening = until out of book (or first ~15 plies); middlegame → endgame
when material drops below a threshold (e.g. ≤ ~6–7 non-pawn pieces). Define pragmatically
and keep it in `config`.

**Stage 5 — Baseline model (`src/model.py`).** Lightweight features only (no engine):
opening ply, game length, result, time control, ECO. Train **ridge regression** and
**LightGBM** to predict Elo. Establish MAE/RMSE against two references: predict-the-mean
(trivial floor) and copy-opponent-rating (the "cheating" baseline you excluded — show it's
strong, to justify excluding it).

**Stage 6 — Improved model (`src/model.py`).** Add the engine/clock features. Tune
LightGBM (Optuna, early stopping on a validation fold). Then train **quantile regressors**
(LightGBM `objective=quantile`, alpha = 0.05 / 0.5 / 0.95) to produce a **median + 90%
interval per prediction** — the "1500 ± X" with X earned per game. Derive rating bands by
bucketing the median for the confusion matrix.

**Stage 7 — Evaluation (`src/evaluate.py`).**
- **Split by player, not by game** (GroupKFold / grouped hold-out on username) — otherwise
  the model memorises a player and the score is fake.
- Metrics: MAE, RMSE (Elo); quantile **interval coverage** (does the 90% interval contain
  the truth ~90% of the time?) and pinball loss; band accuracy + adjacent-band accuracy.
- **Aggregation curve:** for K = 1, 2, 3, 5, 10, 20 games per player, average predictions
  and plot MAE vs K — your headline result (single-game noise → precision via aggregation).
- Calibration: predicted vs actual, binned.
- **SHAP** on the tuned model: global importance + dependence plots → the "what matters
  most" answer (expect centipawn loss / post-book accuracy near the top).

**Stage 8 — Error analysis (`src/evaluate.py`).** Pull the largest-residual games and
categorise them (low-rated brilliant game, high-rated bad game, ultra-short game); residual
vs rating band (does it regress toward the mean — over-predict low, under-predict high?);
residual vs game length; **ablations** (drop each feature group, measure MAE change);
limitations (selection bias from Option A, single-game noise floor, blitz-only scope).

**Stretch (only if core is done):** neural sequence model as a "does a learned
representation beat hand-crafted features?" comparison; cheating detection as anomaly
detection on play-implied vs actual rating.

---

## 4. What to show at each stage (mapped to the required rubric sections)

| Rubric section | Show |
|---|---|
| Overview / Motivation | The thesis + one teaser: P(White wins) vs rating gap (signal exists) |
| Related Work | Regan's IPR, Maia, the cheating-detection work (text) |
| Initial Questions | The question list (single-game vs aggregated; what matters most) |
| Data | Data-funnel table (raw→blitz→eval→clean→sample); rating distribution; **Option-A bias disclaimer** |
| EDA | Centipawn-loss vs rating; Accuracy% vs rating; blunder-rate-by-phase vs rating; theory depth by band; time-usage vs rating; feature↔rating correlation heatmap |
| Baseline (perf + error analysis) | Predicted-vs-actual scatter; MAE/RMSE table vs trivial baselines; residual distribution |
| Improved model | MAE-improvement table (baseline → +eval features → tuned); SHAP summary + dependence; **per-game vs per-player MAE curve**; interval calibration; derived-band confusion matrix |
| Final Analysis | Largest-residual cases + the hard-case categories; ablation table; calibration by band; limitations & conclusions |

---

## 5. Plotting & viewing conventions

- One shared style, set once in `src/plotting.py` (`set_style()` called at the top of each
  notebook): consistent fonts, a fixed colour palette, sane figure sizes, `dpi=150`.
- **Every figure that goes in the deck is saved to `reports/figures/` as a PNG** by a helper
  (`save_fig(name)`), so the slides reference files, not live notebook cells.
- Each chart: clear title, axis labels with units, legend only when needed. Prefer one idea
  per chart.
- During development: Jupyter inline (or Colab). For the deck: pull the saved PNGs.
- Keep plotting logic in helpers so a chart can be regenerated when the data updates without
  hand-editing.

---

## 6. Reports & reproducibility

- **README** documents: the goal, the repo layout, exact **run order** (e.g.
  `python -m src.ingest` → `src.clean` → `src.features` → notebook `03` → `04`), and where to
  get the data.
- **Notebooks** must run top-to-bottom (Restart & Run All) before they're considered done.
- **`reports/results.md`** is the experiment log: a table of every model run (features used,
  hyperparameters, MAE/RMSE, interval coverage) + a one-line note on what changed. This is how
  three people stay in sync on "what's our best model right now."
- **Reproducibility:** fix `random_seed` everywhere; pin `requirements.txt`; all knobs live in
  `config.yaml` (no magic numbers in code).
- **Deliverables:** the GitHub repo, the clean notebooks, the 15–25 min presentation (structure
  it on the rubric table above), and an optional Medium / Towards Data Science post for extra
  credit.

---

## 7. Git conventions (rules, not a tutorial)

- **Branches:** `main` stays runnable. Work on feature branches named by area
  (`feat/feature-engineering`, `eda/openings`, `model/quantile`). Merge via PR with **≥1
  teammate review**. No direct pushes to `main`.
- **Commits:** small and atomic — one logical change each. Imperative mood. Use a prefix
  convention: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `exp:` (experiments/runs),
  `data:` (pipeline/schema). Example: `feat: add phase-split centipawn-loss features`.
- **Never commit data or large/binary artifacts.** `.gitignore` covers `data/`, `models/`,
  `*.pgn`, `*.zst`, `*.parquet`. Document where the shared sample lives. Small result PNGs in
  `reports/figures/` may be committed; large ones, ignore.
- **Notebook hygiene:** clear outputs before committing (e.g. `nbstripout`, or "Clear All
  Outputs") so diffs stay readable and merges don't conflict on cell outputs. Notebooks are
  where merge conflicts come to die — keep them thin and prefer `src/` for shared logic.
- **Tag milestones:** `v0.1-baseline`, `v0.2-improved`, etc.
- Update `requirements.txt` in the same PR that adds a dependency.

---

## 8. Milestone plan & ownership

Roughly ~100 h/person. Phases are relative — compress or stretch to fit your real deadline.
Three roles, with everyone reviewing PRs and co-building the slides.

| Phase | Focus | Lead | Output |
|---|---|---|---|
| 1. Setup & skeleton | Repo, env, config, EDA on the 20k CSV | All | Running repo + first charts |
| 2. Ingest & clean | Download month, stream-filter blitz+evals, clean, instances | **Data (A)** | `blitz_sample` + `instances` parquet, data-funnel |
| 3. Feature engineering | Phase-split features, Accuracy%, time/opening features | **Features/EDA (B)** | `features.parquet` + feature EDA |
| 4. Baseline | No-eval features → ridge + LightGBM, trivial baselines | **Modeling (C)** | Baseline MAE in `results.md` |
| 5. Improved + uncertainty | Add eval features, tune, quantile intervals, aggregation curve | **Modeling (C)** + B | Improved model + SHAP + curve |
| 6. Error analysis & ablations | Hard cases, calibration, feature-group ablations | **B + C** | Error-analysis figures, limitations |
| 7. (Stretch) | Neural comparison and/or cheating detection | whoever has slack | Optional extras |
| 8. Write-up | Slides on the rubric, README, optional blog, buffer | All | Final deliverables |

Suggested split of the heavy lifting: **A** owns the data engineering (parsing/streaming is the
biggest infra piece), **B** owns features + visualization + the analytical narrative, **C** owns
modeling + evaluation. The interfaces between you are the parquet files — agree their columns
early so you can work in parallel.

---

## 9. Backlog / disclaimers

- **Option-A selection bias** (state in the paper): games with stored evals were chosen by
  players for analysis, so the eval'd subset isn't a random sample and eval depth varies. Backlog:
  if time allows, validate against an Option-B random sample with self-computed fixed-depth evals.
- **Single-game noise floor:** a single blitz game can't pin a rating; precision comes from
  aggregation — this is a finding, not a bug.
- **Scope:** blitz only; results may not transfer to rapid/classical.
