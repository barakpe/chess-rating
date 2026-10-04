# Chess Rating Estimation — how much does one blitz game reveal?

Estimate a Lichess player's **blitz rating from a single game they played** — a regression problem
with a conformal 90% prediction interval, plus rating bands for a confusion matrix. The features
come from the game itself: move quality from the Stockfish evaluations Lichess stores in the PGN,
clock usage, and the shape of the game. The opponent's rating, identity and move quality are not
used; the shared context of the game (its length, result, opening and time control) is.

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
   → there is no leftover shrinkage to undo in the single-game model; up-weighting rare ratings improves the tails only
   by worsening the middle; clock-scramble features add nothing; band-conditional intervals do not
   restore extreme-band coverage. The extreme-band bias is the cost of weak evidence.
5. **Does more evidence per player help, and how must it be combined?** → averaging a player's
   games reduces noise but keeps the bias; recalibrating the average for the number of games removes
   a large part of it.
6. **Can the evaluation be trusted?** → splits grouped by player, a check for leakage through shared
   games, player-clustered bootstrap CIs, a full re-run from the raw dump that reproduces every number,
   and — because the development test split was reused while the method took shape — a
   pre-registered hold-out of new players on a later day, scored once.

The experiment log [`reports/results.md`](reports/results.md) shows this progression run by run
(an early ~59k-instance development sample, then the 300k-game sample; the missing-value A/B test;
the tail study; CQR and Mondrian intervals; the scramble features).

## Key results

Development test split: 120,513 player-game instances of 50,357 players never seen in training.
It was scored many times during development, so these are development estimates; the
[confirmatory hold-out](#confirmatory-hold-out) below is the test no decision depended on.

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
- **Honest uncertainty:** conformalized quantile regression with one correction per predicted
  rating band (Mondrian CQR) turns an under-covering 87.4% raw interval into **89.9%** empirical
  coverage (target 90%; plain CQR 89.8%), and 89.4% on the hold-out. A separate check with one
  game per player (plain CQR), which removes the dependence between a player's games, gives 90.1%;
  the finite-sample guarantee also assumes exchangeable players. The interval is ~1,000 Elo wide —
  one game is weak evidence — but 32% narrower than the no-game interval (1,465) at the same
  coverage. It under-covers the extreme bands (78% below 1200, 81% at 2000+; together 40% of test
  games).
- **Regression to the mean at the extremes:** below-1200 players are over-predicted by +298 on
  average, 2000+ players under-predicted by −246. At the single-game level the explanation most
  consistent with our checks is weak evidence, and the fixes we tested did not remove the bias
  without costs elsewhere: there is no leftover linear shrinkage (slope of true on predicted 0.96, 95% CI
  0.946–0.976); up-weighting rare
  ratings moves error from the middle to the tails (overall MAE 239.1 → 245.6); scramble features add
  0.24 Elo (95% CI 0.09–0.40), too little to matter; Mondrian intervals do not restore extreme-band coverage.
- **Several games, combined correctly:** for players with ≥ 10 test games, averaging 10 predictions
  gives MAE 202; recalibrating that average for K (fit on held-out players) gives **152** (error
  against the player's average rating over those games; these are the most active players). At K = 5
  the recalibration shrinks the extreme-band bias from +300 / −246 to +187 / −111 (MAE 200 → 175,
  95% CI of the gain 22–27) — a trade-off: the two central bands get about 28–29% worse.
- **A small leak through shared games:** test rows whose opponent's row is in the training split
  are predicted 2.2 Elo better (95% CI 0.2–4.4). On the rows that share no game with training the
  MAE is 240.4, so the headline 239.1 is optimistic by about 1 Elo. The hold-out shares no game with
  the development data.
- **Sensitivity checks:** restricting the test rows to the uniform sample (dropping the ~1%
  cohort-only games) gives the same MAE (239.1) and coverage (89.9%).
- **In context** ([related work](reports/RELATED_WORK.md)): our 34% error reduction over the mean
  matches the moves-only RatingNet model on blitz (35%), and is behind RatingNet with clock (52%), a
  CNN-LSTM sequence model — on different data and a game-level split, so indicative only.

### Confirmatory hold-out

Because the development test split was reused while the method took shape, we froze the method,
wrote a [protocol](reports/HOLDOUT_PROTOCOL.md), committed it before extracting any data, and
scored the frozen models **once** on every eligible game of **2025-05-31** (26 days after the
development window), keeping only the 124,403 rows of 67,744 players who appear nowhere in the
development data. No hold-out game is in the development data.

| | development test | hold-out |
|---|--:|--:|
| predict the development training mean | 364.8 | 358.9 |
| no-engine baseline | 293.4 | 296.7 |
| **full model, MAE** | **239.1** | **243.9** (95% CI 242.4–245.4) |
| gain over the no-engine baseline | 54.2 | 52.9 (51.5–54.2) |
| 90% interval coverage (headline, Mondrian) | 89.9% | 89.4% |
| coverage below 1200 / at 2000+ | 78% / 81% | 75% / 81% |
| K = 10 games: plain average → recalibrated | 202 → 152 | 205 → 167 |

The findings hold: the same gain over the baselines, ~90% average coverage with the same weak
extremes, the same extreme-band bias, and a clear gain from recalibrating several games. The error
is about 5 Elo higher, within every rating band (not from the rating mix: the mean predictor does
better on the hold-out); about 1 Elo of that is the shared-game leak, the rest fits a population of
less active players 26 days later. Details: [protocol and outcome](reports/HOLDOUT_PROTOCOL.md).

## Course requirements coverage

**Project specification** (Students Projects Requirements, 2026):

| requirement | where | status |
|---|---|---|
| Acquire the data | `src/ingest.py`, `src/clean.py`; notebook 01 §1 | ✅ streamed from the 30.7 GB public dump, filtered, sampled |
| Explore it for interesting patterns | notebooks 01–02 | ✅ |
| Design your visualizations | `reports/figures/` (one style, `src/plotting.py`); every notebook | ✅ |
| Run statistical analysis | correlations (01 §9); player-clustered bootstrap CIs for the main MAE comparisons (04, 05); conformal coverage analysis, incl. one game per player (04); calibration line with CI (05); pre-registered hold-out (05 §6) | ✅ |
| Build a basic ML model | notebook 03 (`src/model.py`) | ✅ ridge + LightGBM without engine features, vs predict-mean and copy-opponent |
| Evaluate its performance | notebooks 03–04, 05 §6 | ✅ MAE, RMSE, R², rank correlation, bias, band accuracy, interval coverage; a pre-registered hold-out scored once |
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
| The data must not be part of the submission | ✅ `data/` is gitignored and no raw or processed data file is committed. `reports/` and the notebook outputs hold only derived results: aggregate tables, figures, and a 20-row table of the worst misses without usernames or game ids. The only PGN in the repo is an 8-game synthetic test fixture. |
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
| `src/holdout.py` | the confirmatory hold-out: ingest a later day, drop development players, score once |
| `src/plotting.py` | shared figure style + `save_fig` |
| `tests/` | 91 unit and integration tests, incl. checks that the notebooks are committed executed (+ an 8-game synthetic PGN fixture); run by CI on Python 3.11 and 3.14 (`.github/workflows/tests.yml`) |
| `notebooks/` | 01–05, the analysis narrative ([guide](notebooks/README.md)) |
| `scripts/make_share_bundle.py` | slim data bundle for teammates who only run the notebooks |
| `requirements-lock.txt` | every package version of the reported runs (`requirements.txt` holds the direct pins) |
| `reports/` | `results.md` (experiment log), `eval_artifacts.json`, `data_funnel.json`, `largest_residuals.csv`, `HOLDOUT_PROTOCOL.md`, `holdout_results.json`, `RELATED_WORK.md`, `figures/` |
| `data/` | **gitignored**: `raw/` (the dump), `processed/` (parquets) |
| `slides/` | the presentation deck (built from `reports/figures/`) |

Design rationale: [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md).

## Setup

Python **3.11–3.14**.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest                            # 91 tests, ~5 s, no data needed
```

Stockfish is not needed — the evaluations are already in the PGN.

## Data

**Option A — run the notebooks only.** Get the slim data bundle (three parquets, ~110 MB) from a
teammate — `python scripts/make_share_bundle.py` builds it from a full run — and copy the files
into `data/processed/`. The notebooks then run end to end and reproduce every reported number.

**Option B — reproduce everything from the raw dump.** Download
`lichess_db_standard_rated_2025-05.pgn.zst` (30.7 GB) from <https://database.lichess.org/> into
`data/raw/`. Do not decompress it; `src.ingest` streams it and stops after the first 15M games.
The file we used is 30,673,986,651 bytes with SHA-256
`3d90d65a0aa2e9fed4ab2dd2232439a12c37a938706ca9815229212894afc9bb`, the value in Lichess's
[checksum list](https://database.lichess.org/standard/sha256sums.txt) (`sha256sum` or
`certutil -hashfile <file> SHA256` to check). The exact package versions of the reported runs are
in `requirements-lock.txt`.

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
| | `python -m src.holdout ingest` / `prepare` / `evaluate` | confirmatory hold-out (May 31) → `holdout_results.json`, `results.md` |
| | `jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb` | executed notebooks |

Measured run times on a laptop: ingest ~35 min (the 30.7 GB download not included), clean ~1 min,
features ~25 min, baseline ~1 min, evaluate ~6 min, tuning ~50 min, tail study ~2 min; hold-out
ingest ~11 min (a byte-level fast-forward to May 31, then the day's 2.9M games), prepare ~5 min,
evaluate ~3 min.

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
- **Interval coverage is empirical and marginal:** ~90% on average over test games (89.9% with
  Mondrian CQR, 89.4% on the hold-out; 90.1% in a one-game-per-player check, which removes the
  within-player dependence but does not by itself establish the guarantee), 75–81% in the extreme
  bands.
- **Development test reuse.** The development test split was scored many times while the method
  took shape; the pre-registered hold-out is the clean check.
- **A small leak through shared games** (about 2 Elo on the rows whose opponent was trained on,
  about 1 Elo on the headline); the hold-out has no shared games.
- **Sampling design.** A uniform game sample plus a player cohort with a higher inclusion chance;
  the cohort is ~1% of test rows and removing it changes nothing.
- **Several games** means several games within five days, for the most active players, measured
  against their average rating over those games.
- **Feature design.** Per-game aggregate features reach a 34–35% error reduction over the mean;
  RatingNet reports 52% with a sequence model over moves and clock — on different data and split, so
  this suggests, but does not show, that sequence models extract more (Related work).

## Related work

See [`reports/RELATED_WORK.md`](reports/RELATED_WORK.md): Kaggle's *Finding Elo* (2014–15),
RatingNet (Omori & Tadepalli, 2024/25), Regan & Haworth's *Intrinsic Chess Ratings* (2011), Maia
(2020) / Maia-2 (2024), and the methods we build on (Lichess accuracy, LightGBM, SHAP, split
conformal prediction, CQR, Mondrian conformal prediction).
