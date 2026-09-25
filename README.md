# Chess Rating Estimation — how much does one blitz game reveal?

Estimate a Lichess player's **blitz rating from a single game they played** — a regression problem
with a conformal 90% prediction interval, plus rating bands for a confusion matrix. The features
come from the game itself: move quality from the Stockfish evaluations Lichess stores in the PGN,
clock usage, and the shape of the game. Nothing about the opponent — rating, identity or moves —
is used.

*Applied Data Science final project (Bar-Ilan University, course 83901). Data: the
[Lichess open database](https://database.lichess.org/), May 2025.*

---

## Overview and motivation

A chess rating summarises hundreds of results. But a single game also shows *how* someone plays —
how often they blunder, how accurate they are in the opening, how they use the clock. How much of
their strength can be read from that one game, and how sure can we be?

This matters to a chess platform in several ways:

- **Placing new or returning players.** A new account starts at a default rating and is marked
  provisional until it has played enough games. A play-based estimate from the first few games,
  with an honest interval, could seed the rating closer to the truth.
- **Accounts that do not play at their rating.** Sandbagging (deliberately losing to lower one's
  rating) and smurfing (strong players on fresh accounts) show up as a gap between how an account
  plays and its rating. An interval with known coverage makes "surprising" measurable. (Caveats: an
  estimate like this is a screening signal, not proof; and our interval misses about 20% of players
  in the extreme rating bands — exactly where such accounts would sit.)
- **Coaching.** Which aspects of a player's game — opening accuracy, blunder rate, time use —
  separate them from the next rating level.

For all of these, a point estimate without an honest uncertainty is not enough, and one game is
weak evidence — so the project cares as much about **honest intervals** and **combining several
games** as about the single-game error.

## Questions, and how they evolved

1. **Can one blitz game reveal a player's rating?** → regression on per-game features (notebooks 01–04).
2. **Which signals carry it?** → EDA, SHAP and group ablations: move quality (especially in the
   opening) dominates; clock and opening choice add smaller, significant amounts.
3. **Is the uncertainty honest?** → the raw quantile interval under-covered (87%); conformal
   correction (CQR) brought it to 90% — but only on average, not in the rating extremes.
4. **Why is the error concentrated at the rating extremes, and can it be fixed within one game?**
   → the single-game model is already calibrated; up-weighting rare ratings improves the tails only
   by worsening the middle; clock-scramble features add nothing; band-conditional intervals do not
   restore extreme-band coverage. The extreme-band bias is the cost of weak evidence.
5. **Does more evidence per player help, and how must it be combined?** → averaging a player's
   games reduces noise but keeps the bias; recalibrating the average for the number of games removes
   a large part of it.
6. **Can the evaluation be trusted?** → splits grouped by player, a check for leakage through shared
   games, player-clustered bootstrap CIs, and a full re-run from the raw dump that reproduces every number.

The experiment log [`reports/results.md`](reports/results.md) shows this progression run by run
(an early ~59k-instance development sample, then the 300k-game sample; the missing-value A/B test;
the tail study; CQR and Mondrian intervals; the scramble features).

## Key results

Test set: 120,513 player-game instances of 50,357 players never seen in training.

| model | MAE (Elo) |
|---|--:|
| predict the training mean | 364.8 |
| copy the opponent's rating (**the leak**, excluded) | 80.9 |
| ridge, no engine features | 297.9 |
| LightGBM, no engine features | 292.5 |
| **LightGBM, 77 engine + clock + style features** | **239.1** |
| + Optuna tuning | 237.2 |

(The first four rows are Stage 5, trained on all training players. The engine model trains on a
train split with early stopping on a validation split; the no-engine LightGBM in that same setup
scores 293.4, which is the baseline of the 54-Elo gain below.)

- **The full feature set** (move quality, clock, game structure) cuts the error by **54 Elo** (95%
  CI 52.7–55.8, player-clustered bootstrap); removing the 50 move-quality features costs 34.
  R² 0.55, Spearman 0.73, within 200 Elo half of the time. Tuning adds 2.0 more (CI 1.8–2.2).
- **Honest uncertainty:** split CQR turns an under-covering 87.4% raw interval into **89.8%**
  coverage (target 90%). The interval is ~1,000 Elo wide — one game is weak evidence — but 32%
  narrower than the no-game interval (1,465) at the same coverage. It under-covers the extreme
  bands (78% below 1200, 80% at 2000+; together 40% of test games).
- **Regression to the mean at the extremes:** below-1200 players are over-predicted by +298 on
  average, 2000+ players under-predicted by −246. At the single-game level this is the cost of weak
  evidence: the model is already calibrated (slope of true on predicted 0.96); up-weighting rare
  ratings moves error from the middle to the tails (overall MAE 239.1 → 245.6); scramble features add
  ~0; Mondrian intervals do not restore extreme-band coverage.
- **Several games, combined correctly:** for players with ≥ 10 test games, averaging 10 predictions
  gives MAE 202; recalibrating that average for K (fit on held-out players) gives **152**. At K = 5
  the recalibration shrinks the extreme-band bias from +300 / −246 to +187 / −111 (MAE 200 → 175,
  95% CI of the gain 22–27) — a trade-off: the two central bands get about 28–29% worse.
- **No detectable leakage:** test rows whose opponent was in the training data are not predicted
  better (difference −0.6 Elo, 95% CI −3.0 to +1.5; an advantage above ~3 Elo is ruled out).
- **In context** ([related work](reports/RELATED_WORK.md)): our 34% error reduction over the mean
  matches the moves-only RatingNet model on blitz (35%), and is behind RatingNet with clock (52%), a
  CNN-LSTM sequence model — on different data and a game-level split, so indicative only.

## Course requirements coverage

**Project specification** (Students Projects Requirements, 2026):

| requirement | where | status |
|---|---|---|
| Acquire the data | `src/ingest.py`, `src/clean.py`; notebook 01 §1 | ✅ streamed from the 30.7 GB public dump, filtered, sampled |
| Explore it for interesting patterns | notebooks 01–02 | ✅ |
| Design your visualizations | `reports/figures/` (one style, `src/plotting.py`); every notebook | ✅ |
| Run statistical analysis | correlations (01 §9); player-clustered bootstrap CIs for the main MAE comparisons (04, 05); conformal coverage analysis (04); calibration-slope estimate (05) | ✅ |
| Build a basic ML model | notebook 03 (`src/model.py`) | ✅ ridge + LightGBM without engine features, vs predict-mean and copy-opponent |
| Evaluate its performance | notebooks 03–04 | ✅ MAE, RMSE, R², rank correlation, bias, band accuracy, interval coverage |
| Perform error analysis | notebook 03 §2, notebook 05 | ✅ residual by band and length, worst misses, ablations, leakage check |
| Improve the model | notebook 04 | ✅ engine/clock features, tuning, CQR intervals, K-aware aggregation; future work in `PROJECT_ARCHITECTURE.md` §6 |
| Communicate the results | notebooks (narrative + figures), this README | ✅ notebooks; the slide deck is built separately |

**Presentation elements** (15–25 min):

| element | material |
|---|---|
| Overview and motivation | this README (Overview and motivation); notebook 01 intro |
| Related work | [`reports/RELATED_WORK.md`](reports/RELATED_WORK.md) |
| Initial questions and how they evolved | this README (Questions, and how they evolved); `reports/results.md` |
| Data: source, scraping, cleanup, storage | notebook 01 §1–3; `PROJECT_ARCHITECTURE.md` §2–3 |
| Exploratory data analysis | notebooks 01–02 |
| Basic ML model + performance + error analysis | notebook 03 |
| Improved ML model | notebook 04 |
| Final analysis | notebook 05; Key results above |

**Additional submissions and FAQ rules:**

| rule | status |
|---|---|
| Notebooks / Python code in a GitHub repository, easy to read, with section titles and explanatory text | ✅ five narrated notebooks committed with outputs; `src/` documented and unit-tested |
| Real-world (not synthetic) data set | ✅ Lichess open database |
| The data must not be part of the submission | ✅ `data/` is gitignored; no data file was ever committed (the only PGN in the repo is an 8-game synthetic test fixture) |
| Python | ✅ |
| Blog post (optional, extra credit) | not done |

## Repository layout

| path | what |
|---|---|
| `config.yaml` | every knob: data window, sample size, filters, phase boundaries, splits, seed, paths |
| `src/ingest.py` | Stage 1: stream the `.pgn.zst`, filter, reservoir-sample + player cohort, write parquet |
| `src/clean.py` | Stages 2–3: union sample + cohort, dedup/clean, explode into per-player instances |
| `src/features.py` | Stage 4: Lichess Win%/Accuracy%/move classification + phase-split features |
| `src/model.py` | Stages 5–6: baselines, LightGBM, quantile models, split/Mondrian CQR, Optuna |
| `src/evaluate.py` | Stages 7–8: metrics, intervals, aggregation, SHAP, error analysis, bootstrap CIs |
| `src/plotting.py` | shared figure style + `save_fig` |
| `tests/` | 83 unit and integration tests, incl. checks that the notebooks are committed executed (+ an 8-game synthetic PGN fixture) |
| `notebooks/` | 01–05, the analysis narrative ([guide](notebooks/README.md)) |
| `scripts/make_share_bundle.py` | slim data bundle for teammates who only run the notebooks |
| `reports/` | `results.md` (experiment log), `eval_artifacts.json`, `data_funnel.json`, `largest_residuals.csv`, `RELATED_WORK.md`, `figures/` |
| `data/` | **gitignored**: `raw/` (the dump), `processed/` (parquets) |
| `slides/` | the presentation deck (built from `reports/figures/`) |

Design rationale: [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md).

## Setup

Python **3.11–3.14**.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest                            # 83 tests, ~5 s, no data needed
```

Stockfish is not needed — the evaluations are already in the PGN.

## Data

**Option A — run the notebooks only.** Get the slim data bundle (three parquets, ~110 MB) from a
teammate — `python scripts/make_share_bundle.py` builds it from a full run — and copy the files
into `data/processed/`. The notebooks then run end to end and reproduce every reported number.

**Option B — reproduce everything from the raw dump.** Download
`lichess_db_standard_rated_2025-05.pgn.zst` (30.7 GB) from <https://database.lichess.org/> into
`data/raw/`. Do not decompress it; `src.ingest` streams it and stops after the first 15M games.

## Run order

Every stage reads `config.yaml` and hands the next one a parquet file.

| # | command | output |
|---|---|---|
| 1 | `python -m src.ingest` | `data/processed/blitz_sample.parquet`, `player_cohort.parquet`, `reports/data_funnel.json` |
| 2–3 | `python -m src.clean` | `games_clean.parquet`, `instances.parquet` (+ funnel) |
| 4 | `python -m src.features` | `features.parquet` |
| 5 | `python -m src.model` | baseline table → `reports/results.md` |
| 6–8 | `python -m src.evaluate` | full evaluation → `reports/results.md`, `eval_artifacts.json`, `largest_residuals.csv`, `figures/` |
| | `python -m src.evaluate --tune --artifacts "" --residuals-csv "" --no-figures` | Optuna search, logs the tuned row |
| | `python -m src.evaluate --tail-study` | single-game tail-correction study → `results.md` |
| | `jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb` | executed notebooks |

Measured run times on a laptop: ingest ~35 min (the 30.7 GB download not included), clean ~1 min,
features ~25 min, baseline ~1 min, evaluate ~6 min, tuning ~50 min, tail study ~2 min.

## Data contract between stages

`blitz_sample.parquet` / `player_cohort.parquet` / `games_clean.parquet` — one row per game
(`src.ingest.INGEST_COLUMNS`): `game_id` (Lichess URL), `event`, `white`, `black`, `result`,
`white_elo`, `black_elo` (the labels, Glicko-2), `white_rating_diff`, `black_rating_diff`, `eco`,
`opening`, `time_control`, `termination`, `utc_date`, `utc_time`, `movetext` (SAN with `[%eval]`
and `[%clk]` comments), `n_plies`.

`instances.parquet` — two rows per game (`src.clean.INSTANCE_COLUMNS`): `game_id`, `color`,
`username` (the grouped-split key), `rating` (the label), `result` (from this player's side),
`time_control`, `eco`, `opening`, `n_plies`. **Never** the opponent's rating or username.

`features.parquet` — one row per instance: the 8 identifier/context columns (`game_id`, `color`,
`username`, `rating`, `result`, `time_control`, `eco`, `opening`) + 73 numeric features. Missing-value
contract: `NaN` = not observable, `0` = a true zero; a phase that did not occur has `n_moves = 0` and
`NaN` elsewhere; `has_*` flags mark presence. Details in notebook 02 and `src/features.py`.

## Conventions

- **Never commit data** (`data/`, `*.parquet`, `*.pgn`, `*.zst` are gitignored), except the tiny
  synthetic test fixture.
- **Split by player, never by game** (`model.grouped_split`, `model.grouped_train_val_calib_test`).
- **Reproducibility:** fixed seed, pinned dependencies, every knob in `config.yaml`. `reports/results.md`
  is append-only; each entry is headed by the commit that produced it (`-dirty` if uncommitted
  code was used). Notebooks run top to bottom.
- Run `pytest` before every push.

## Limitations

- **Selection bias.** Only games with stored engine evaluations are used — about 9% of blitz
  games, analysed on Lichess for some reason. Results describe that population; transfer to
  un-analysed games is untested (it would need our own engine analysis of a random sample).
- **Short window.** The first 15M games of May 2025 = May 1–5 (UTC). Rating drift and seasonality
  are not studied, and "all games of a player" means all games in those five days.
- **Noisy labels.** New and returning players have provisional (unsettled) Glicko-2 ratings; the PGN
  export does not mark them, so they cannot be identified reliably. Symptoms: a spike at exactly 1500
  (the starting rating), and large rating gaps (beyond ±250) that the lower-rated player wins more often.
- **Blitz only.** Rapid and classical are not studied.
- **Interval coverage is marginal and approximate:** ~90% on average over test games, 78–80% in the
  extreme bands; several games of one player are not independent.
- **Feature design.** Per-game aggregate features reach a 34–35% error reduction over the mean;
  RatingNet reports 52% with a sequence model over moves and clock — on different data and split, so
  this suggests, but does not show, that sequence models extract more (Related work).

## Related work

See [`reports/RELATED_WORK.md`](reports/RELATED_WORK.md): Kaggle's *Finding Elo* (2014–15),
RatingNet (Omori & Tadepalli, 2024/25), Regan & Haworth's *Intrinsic Chess Ratings* (2011), Maia
(2020) / Maia-2 (2024), and the methods we build on (Lichess accuracy, LightGBM, SHAP, split
conformal prediction, CQR, Mondrian conformal prediction).
