# Confirmatory hold-out — protocol

Written and committed **before** the hold-out games were extracted or scored (see this file's git
history). Code: [`src/holdout.py`](../src/holdout.py).

## Why a second test set

The development test split (120,513 rows of 50,357 players) is player-disjoint from training, but
it was scored many times while the method took shape: the 65 → 69 → 77 feature revisions, the
missing-value A/B, the tuning runs, the tail study (reweighting, de-shrink, scramble features),
Mondrian intervals and the K-aware aggregation recalibration were all compared on it
([`results.md`](results.md)). None of these decisions moved the headline much, but together they
make the development numbers **development estimates**: repeated use of one test set can make its
numbers optimistic, and the bootstrap CIs do not account for that. The hold-out gives numbers
that no decision depended on.

## Data (fixed before extraction)

- **Source:** the same dump, `lichess_db_standard_rated_2025-05.pgn.zst`.
- **Day:** every game whose `UTCDate` is **2025.05.31**, the last day of the month. The
  development data covers May 1–5, so there is a gap of 26 days.
- **Filters:** identical to the development data: blitz (180–479 s estimated duration), no bots,
  termination Normal or Time forfeit, stored `[%eval]` annotations, at least 10 plies, rating
  400–4000, duplicate game ids dropped. No sampling and no player cohort: every eligible game of
  the day is kept.
- **Players:** the rows of every player who appears anywhere in the development data (any split,
  either colour) are dropped. A hold-out game may have a development player as the opponent; that
  player's row is simply not scored.
- **Consequence:** the hold-out players are players with no analysed blitz game in May 1–5, which
  likely favours less active players. So this tests transfer to a later date **and** to a
  somewhat different population; it is not a second draw from the development population.

## Frozen method

- `src/holdout.py` refits the models exactly as `src/evaluate.py` does (same data, split, seed and
  `config.yaml`), which reproduces the models reported in `results.md`. `--dry-run` scores the
  development test split with the same code, as a check that it reproduces the development
  numbers; it writes nothing.
- **Headline model:** LightGBM on all 77 features, untuned (development MAE 239.1).
- Also scored: predicting the development training mean; the no-engine LightGBM baseline; the
  tuned model (hyperparameters frozen in `config.yaml` → `holdout.tuned_params`).
- **Interval:** the quantile models with the split-CQR and Mondrian corrections fitted on the
  development calibration split. The headline interval is Mondrian (`model.mondrian_cqr`).
- **Aggregation:** the K-aware recalibration fitted on the development calibration split.

## Metrics (all are reported, whatever they show)

Primary:

1. MAE of the headline model, with a 95% player-clustered bootstrap CI.
2. The gain over the no-engine baseline (MAE difference), with a 95% CI.
3. Empirical coverage of the headline 90% interval.

Secondary: MAE of the mean predictor and of the tuned model; RMSE, median AE, R², Spearman and
bias; plain-CQR coverage and width; coverage and mean residual by rating band; the calibration
line (true on predicted) with a CI; coverage of the no-game interval; and the matched aggregation
curve and K = 5 table, if at least 30 hold-out players have enough games (in one day, few will).

## What we expect, stated in advance

A clear gain over both baselines; interval coverage near 90% on average and lower in the extreme
bands; the same over/under-prediction at the rating extremes. The MAE itself may move away from
239 because the population differs: the mean predictor's MAE on the hold-out shows how much of
any change comes from the spread of ratings rather than from the model.

## Rules

- **Scored once.** `python -m src.holdout evaluate` refuses to run when
  `reports/holdout_results.json` exists.
- No change to data, features, model, intervals or recalibration after the scoring. If a crash
  forces a code fix, it is committed separately and described here.
- The results are appended to `results.md`, written to `holdout_results.json`, and reported next
  to the development numbers in the README and the presentation, including if they are worse.

## Outcome (added after the single scoring run, at 33171ec)

Scored once on 124,403 rows of 67,744 new players (every eligible game of 2025-05-31: 123,276
games; the 122,149 rows of development players dropped). Full entry: [`results.md`](results.md),
numbers: [`holdout_results.json`](holdout_results.json). The dry run on the development test split
reproduced every development number before the hold-out was scored.

| | development test | hold-out |
|---|--:|--:|
| rows / players | 120,513 / 50,357 | 124,403 / 67,744 |
| rating mean (sd) | 1645 (445) | 1562 (432) |
| predict the development training mean | 364.8 | 358.9 |
| no-engine baseline | 293.4 | 296.7 |
| **full model (headline), MAE** | **239.1** | **243.9** [95% CI 242.4, 245.4] |
| gain over the no-engine baseline | 54.2 [52.7, 55.8] | 52.9 [51.5, 54.2] |
| tuned model | 237.2 | 241.9 |
| 90% interval coverage, Mondrian (plain) | 89.9% (89.8%) | 89.4% (89.4%) |
| coverage below 1200 / at 2000+ | 78% / 81% | 75% / 81% |
| mean residual below 1200 / at 2000+ | +298 / −246 | +307 / −248 |
| matched aggregation, K = 10: naive → recalibrated | 202 → 152 (1,260 players) | 205 → 167 (802 players) |
| K = 5 recalibration gain | 24.4 [22.2, 26.8] | 20.8 [18.1, 23.3] |

Against the expectations stated above: the gain over both baselines holds, coverage stays near
90% on average and lower at the extremes, and the extreme-band bias has the same shape. The error
is **about 5 Elo higher** than the development estimate. It is not a matter of rating spread (the
mean predictor does *better* on the hold-out) or of the rating mix (weighting the hold-out's band
MAEs by the development band shares gives the same 243.9): the MAE is 1.5–9 Elo higher within
every rating band. Part of it is the ~1 Elo the development estimate gained from shared games; the
rest is consistent with a population of less active players seen 26 days later (overall bias
+33 Elo, against −3 in development). The K-aware recalibration still helps on the hold-out, by a
little less.
