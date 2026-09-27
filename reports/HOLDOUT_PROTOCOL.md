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
