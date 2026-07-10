# Chess Rating Estimation — Architecture & Setup

A reference for the design decisions behind this pipeline: what to estimate, why blitz + stored
evals, why regression, and how each stage fits together. Neural sequence model and cheating
detection are out-of-scope stretch goals.

---

## 0. Decisions locked in

| Decision | Choice | Note |
|---|---|---|
| Time control | Blitz only | Highest volume; avoids cross-format rating confound |
| Evals | Use games that already carry a stored `[%eval]`, rather than computing our own | ~6% of games, but still millions in one blitz month |
| Time window | One recent month | Keeps the rating label stable (no drift) |
| Target | Regression (Elo) | + quantile interval; bands derived for a confusion matrix |
| Sample size | Configurable, ~300k in the current run | Scale up further on a stronger machine |
| Neural model | Deferred (stretch) | Gradient-boosted trees are the core model |
| Known bias | Games with stored evals aren't a random sample (players chose to request analysis) | See Limitations in the README; backlog item to revisit with self-computed evals |

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
├── notebooks/                # exploration only; numbered, run top-to-bottom (see NOTEBOOK_PLAN.md)
│   ├── 01_data_and_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_improved_model.ipynb
│   └── 05_error_analysis_and_limits.ipynb
├── src/                      # reusable, importable pipeline code
│   ├── config.py             # loads config.yaml
│   ├── ingest.py             # stream .zst → filter → sample → parquet
│   ├── clean.py              # filtering/dedup/instance building
│   ├── features.py           # phase-split feature extraction
│   ├── model.py              # baselines + quantile regression + conformal intervals
│   ├── evaluate.py           # metrics, aggregation curve, calibration, SHAP, error analysis
│   └── plotting.py           # shared matplotlib style + chart helpers
├── reports/
│   ├── figures/              # saved PNGs
│   └── results.md            # experiment log (MAE table, what changed, commit per run)
└── slides/                   # presentation deck, built from reports/figures/
```

Rule of thumb: **notebooks are for looking; `src/` is for doing.** Anything run more than once, or
that another stage depends on, goes into `src/` as a function the notebook imports. This keeps the
pipeline reproducible and the notebooks thin.

---

## 2. Environment & what to download

**Python packages** (`requirements.txt`, pinned): `python-chess` (PGN parsing, board
reconstruction, reading evals), `zstandard` (decompress `.pgn.zst`), `pandas`, `numpy`, `pyarrow`
(parquet I/O), `scikit-learn`, `lightgbm`, `shap`, `optuna` (hyperparameter search), `matplotlib`,
`seaborn`, `pyyaml`, `tqdm`. Commented out until needed: `mapie`/`crepes` (off-the-shelf conformal
prediction — not needed, since split CQR is implemented directly in `src/model.py`), `torch`
(neural stretch goal).

**Stockfish** — not needed. Evals are already stored in the PGN; you'd only need it if you later
switched to computing evals yourself.

**The data — one Lichess monthly dump:**
- File: `lichess_db_standard_rated_YYYY-MM.pgn.zst` from `https://database.lichess.org/`.
- It is **all time controls mixed** and compressed with zstandard. A month is roughly
  ~30 GB compressed / ~200+ GB uncompressed — so **do not decompress the whole thing.**
- **Stream-decompress and filter on the fly**, keeping only blitz + has-eval games, and
  **stop once you've sampled enough.** You end up writing a few-hundred-MB parquet, not
  200 GB. `python-chess` reads directly from a `zstandard` stream.
- **Do this once.** Produce the filtered `processed/blitz_sample.parquet`, commit the *script*,
  and share the *parquet* via a shared drive (not git) instead of downloading 30 GB repeatedly.
- For building the pipeline skeleton before the big download, prototype on a small no-eval CSV
  sample — enough to wire up cleaning + a no-eval baseline.

**`config.yaml`** (single source of truth — every script reads this):
```yaml
month: "2025-05"
time_control: "blitz"          # human label; the real filter is blitz_estimate_seconds
use_stored_evals: true         # keep only games that already carry [%eval]
sample_size: 300000            # bump up on a stronger machine
min_plies: 10                  # drop aborted / ultra-short games
drop_provisional: true         # documented no-op — see README Limitations
exclude_bots: true
rating_bands: [0, 1200, 1400, 1600, 1800, 2000, 3000]
random_seed: 42
paths:
  raw: data/raw
  processed: data/processed
  figures: reports/figures
```
(See the committed `config.yaml` for the full set of knobs — phase boundaries, feature
thresholds, model/CQR settings, player-cohort sampling, and evaluation options all live there
too; nothing is hard-coded in `src/`.)

---

## 3. The pipeline (stage by stage)

```
raw .pgn.zst
  └─(1) ingest+filter+sample─▶ blitz_sample.parquet (+ player_cohort.parquet)
        └─(2) clean─────────▶ games_clean.parquet
              └─(3) explode──▶ instances.parquet      (2 rows/game: one per side)
                    └─(4) features─▶ features.parquet
                          └─(5/6) model─▶ predictions + quantile intervals
                                └─(7/8) evaluate+error─▶ reports/figures + results.md
```

**Stage 1 — Ingest & sample (`src/ingest.py`).** Streams the `.zst` as raw text and splits it
into per-game (headers, movetext) chunks without building a `chess.pgn` object. Cheap
string/regex prefilters (mirroring the real TimeControl/Termination/bot/eval checks) reject most
games from that raw text alone — only survivors get a full `chess.pgn.read_game` parse and the
authoritative checks. This matters because roughly a third of games are blitz + non-bot +
termination-ok, but ~94% of those lack a stored eval; skipping the expensive parse for that 94%
is where nearly all of the ingest speedup comes from. Reservoir-samples to `sample_size` for an
unbiased sample of the filtered stream. Also writes a **player-cohort** parquet: every game of a
small, deterministically hash-sampled set of players, kept in full rather than reservoir-sampled
— a uniform game sample has a per-player median of ~1 game, which starves the aggregation curve
(Stage 7) of players with many games; the cohort complements it.

**Stage 2 — Clean & filter (`src/clean.py`).** Drops games under `min_plies`, abnormal
terminations (abandoned/rules-infraction), bot accounts, and duplicates. Keeps a data-funnel count
(raw → blitz → has-eval → cleaned → sampled) for reporting.

**Stage 3 — Build instances (`src/clean.py`).** Explodes each game into **two rows**,
one per side, each described by *that player's* play, labeled with *that player's*
rating. **Never** adds opponent rating or opponent-derived stats — that would leak the
label, since Lichess matches similar ratings. Records the username for grouped splitting and
per-player aggregation.

**Stage 4 — Feature engineering (`src/features.py`) — the heart.** Replays each game
with `python-chess`, reads the per-move evals/clocks, and computes 73 numeric features per
player, **split by phase** (opening / middlegame / endgame):

- *Move quality:* mean/median/std/max centipawn loss; mean Accuracy% (Lichess's Win%→accuracy
  formula); inaccuracy/mistake/blunder counts and rates (classified by the Win% drop per move);
  accuracy after leaving the opening book (the "knew theory, then collapsed" signal).
- *Time:* mean/std/median move time from `[%clk]` deltas, share of very-fast moves, time-trouble
  share.
- *Clock-scramble:* the same quality/tempo features, but restricted to moves played with under
  30s on the clock, plus the CPL degradation delta vs the player's overall average — does this
  player's quality hold up under time pressure?
- *Style:* game length, captures/checks counts, whether they converted a winning position.
- *Presence flags:* `has_opening` / `has_middlegame` / `has_endgame` / `has_clock` /
  `has_scramble` — make phase/clock absence directly splittable for the trees, instead of relying
  on them to infer it from the NaNs.

**Missing-value contract:** NaN means *not observable*; 0 is reserved for a genuine zero (a move
count, or a rate computed over a nonempty set). A phase with no moves gets `n_moves = 0` and NaN
for every other aggregate in that phase (including error counts — a rate is only meaningful
conditional on exposure). LightGBM consumes NaN natively (it learns a default split direction),
so there's no imputation or sentinel values anywhere in this stage.

Same leakage guarantee as Stage 3: every feature describes *this* player's play only. Cross-game
aggregates (opening-diversity entropy, the per-K aggregation curve) deliberately live in
evaluation (Stage 7) instead, since a single-game prediction can't see a player's other games.

**Stage 5 — Baseline model (`src/model.py`).** Lightweight features only (no engine):
opening ply, game length, result, time control, ECO. Trains **ridge regression** and
**LightGBM** to predict Elo, and reports MAE/RMSE against two references: predict-the-mean
(trivial floor) and copy-opponent-rating (the "cheating" baseline excluded from the model —
showing it's strong is exactly why opponent rating must never be a feature).

**Stage 6 — Improved model (`src/model.py`).** Adds the engine/clock/scramble features, tunes
LightGBM (Optuna, early stopping on a validation fold), then trains **quantile regressors**
(`objective="quantile"`, alpha = 0.05 / 0.5 / 0.95) for a **median + 90% interval per
prediction**. The raw interval is **conformalized (split CQR)** on a dedicated grouped
calibration split (never seen by any fit) for guaranteed *marginal* 90% coverage. A
**band-conditional (Mondrian) CQR** variant is also computed — one correction per
*predicted*-median band — which improves coverage conditional on the prediction (the only
test-time-legal conditioning); both are reported so the improvement is visible, and
`config.yaml`'s `mondrian_cqr` selects which is the headline number. Rating bands for the
confusion matrix are derived by bucketing the point prediction.

**Stage 7 — Evaluation (`src/evaluate.py`).**
- **Split by player, not by game** (grouped hold-out on username) — otherwise the model
  memorises a player and the score is fake.
- Metrics: MAE, RMSE, R², rank correlation, bias, median AE; quantile **interval coverage**
  (raw vs CQR vs Mondrian) and pinball loss; band accuracy + adjacent-band accuracy.
- **Aggregation curve:** for K = 1, 2, 3, 5, 10, 20 games per player, average predictions
  and plot MAE vs K — single-game noise vs precision gained from averaging.
- Calibration (predicted vs actual, binned) and a predicted-vs-actual scatter.
- **SHAP** on the tuned model: global importance (expect centipawn loss / post-book accuracy
  near the top).

**Stage 8 — Error analysis (`src/evaluate.py`).** Largest-residual games, categorised;
residual vs rating band (does it regress toward the mean?); residual vs game length;
**feature-group ablations** (drop each group, measure the MAE hit); a dedicated tail-bias study
(`--tail-study`) that tests band-reweighting and de-shrink calibration against the raw tail error.

**Stretch, only if the core is solid:** a neural sequence model, to see whether a learned
representation beats hand-crafted features; cheating detection as anomaly detection on
play-implied vs actual rating.

---

## 4. Plotting & viewing conventions

- One shared style, set once in `src/plotting.py` (`set_style()`): consistent fonts, a fixed
  colour palette, sane figure sizes, `dpi=150`.
- Every figure is saved to `reports/figures/` as a PNG via `save_fig(name)`, so downstream
  consumers (slides, notebooks) reference files, not live cells.
- Each chart: clear title, axis labels with units, legend only when needed — one idea per chart.
- Figure generation is best-effort in `evaluate.py`: a failed plot logs a warning and the run
  continues, rather than aborting the whole evaluation.

---

## 5. Reports & reproducibility

- **README** documents the goal, the repo layout, the exact run order, and where to get the data.
- **Notebooks** must run top-to-bottom (Restart & Run All) before they're considered done; no
  logic is reimplemented in a cell — everything calls into `src/`.
- **`reports/results.md`** is the experiment log: a table of every run (features used,
  hyperparameters, MAE/RMSE, interval coverage), headed by the commit that produced it, plus a
  one-line note on what changed. This is how "what's our best model right now" stays answerable.
- **Reproducibility:** `random_seed` is fixed everywhere; `requirements.txt` is pinned; every
  knob lives in `config.yaml` — no magic numbers in code.

---

## 6. Backlog / open questions

- **Selection bias:** games with stored evals were chosen by players for analysis, so the eval'd
  subset isn't a random sample and eval depth varies. Backlog: validate against a random sample
  with self-computed fixed-depth evals.
- **Single-game noise floor:** a single blitz game can't pin a rating; precision comes from
  aggregation — this is a finding, not a bug. See the README's Limitations section for the full
  writeup, including the tail-bias study and the CQR/Mondrian coverage results.
- **Scope:** blitz only; results may not transfer to rapid/classical.
