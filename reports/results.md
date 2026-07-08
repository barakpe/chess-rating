# Experiment log

The running record of model runs — how three people stay in sync on "what's our best model right
now." Each run appends a block (features used, n_train/n_test, MAE/RMSE). `python -m src.model`
appends here automatically. Newest at the bottom.

MAE/RMSE are in Elo points. Reference baselines:
- **predict_mean** — predict the training-mean rating for everyone (trivial floor).
- **copy_opponent** — predict the opponent's rating (strong, but a leak; excluded from the model).

## 2026-07-03 00:19 — baseline (no-engine features)
n_train=47109, n_test=11753, features=game_plies, player_moves, n_captures, n_checks, result, time_control, eco

| model | MAE | RMSE |
|---|---|---|
| predict_mean | 365.5 | 445.0 |
| copy_opponent | 79.8 | 152.8 |
| ridge | 295.7 | 368.0 |
| lightgbm | 285.4 | 361.5 |

## 2026-07-03 00:20 — improved model
n_train=37802, n_test=11753, n_features=65

| stage | MAE | RMSE |
|---|---|---|
| no-engine baseline | 287.8 | — |
| + engine/clock | 241.2 | 304.2 |

- 90% interval coverage: **85.1%** (width 949 Elo)
- band accuracy: 35.5% exact, 75.3% adjacent
- aggregation MAE: K1=237, K2=215, K3=222, K5=222, K10=224
- ablation (MAE↑ when dropped): engine +31.4, style +10.9, clock +8.1, opening +7.6

## 2026-07-03 00:58 — improved model (tuned)
n_train=37802, n_test=11753, n_features=65

| stage | MAE | RMSE |
|---|---|---|
| no-engine baseline | 287.8 | — |
| + engine/clock | 241.2 | 304.2 |
| + tuned | 239.5 | 301.5 |

- 90% interval coverage: **83.9%** (width 913 Elo)
- band accuracy: 35.4% exact, 75.6% adjacent
- aggregation MAE: K1=236, K2=213, K3=221, K5=220, K10=222
- ablation (MAE↑ when dropped): engine +31.4, style +10.9, clock +8.1, opening +7.6

## 2026-07-05 17:31 — improved model
n_train=32033, n_calib=5769, n_test=11753, n_features=69

| stage | MAE | RMSE |
|---|---|---|
| no-engine baseline | 290.2 | — |
| + engine/clock | 243.2 | 306.2 |

- median AE 204, R² 0.526, Spearman 0.715, within 100/200 Elo 26%/49%, bias +7.5
- per-player (all games averaged): **229.3** MAE over 7298 players
- 90% interval coverage: raw 84.6% -> CQR **90.8%** (width 950 -> 1098 Elo; correction lo=74.1, hi=74.1)
- coverage by band: 0-1200 75%, 1200-1400 98%, 1400-1600 99%, 1600-1800 100%, 1800-2000 98%, 2000-3000 82%
- band accuracy: 35.2% exact, 75.0% adjacent
- aggregation MAE: K1=238, K2=217, K3=225, K5=226, K10=233
- ablation (MAE↑ when dropped): engine +31.5, style +10.0, clock +7.6, opening +6.6

## 2026-07-05 17:31 — tail study

| variant | overall | tail | mid | 0-1200 | 1200-1400 | 1400-1600 | 1600-1800 | 1800-2000 | 2000-3000 |
|---|---|---|---|---|---|---|---|---|---|
| plain | 243.2 | 319 | 194 | 352 | 209 | 182 | 177 | 207 | 287 |
| reweighted(s=0.5) | 244.2 | 326 | 191 | 348 | 201 | 175 | 175 | 211 | 303 |
| deshrink(slope=1.06) | 243.2 | 309 | 201 | 345 | 214 | 192 | 185 | 212 | 274 |

## 2026-07-05 — missing-value encoding A/B (same grouped splits, untuned)
Old encoding (65 feats: absent-phase counts=0, stds=0 at n=1, time_trouble_share=0 with no clock)
vs new unified NaN contract + has_* flags (69 feats). Identical usernames/seed => identical splits.

| encoding | MAE | RMSE | R2 | raw coverage | CQR coverage |
|---|---|---|---|---|---|
| old (mixed 0/NaN) | 242.7 | 305.6 | 0.528 | 84.7% | 90.8% |
| new (NaN contract + flags) | 243.2 | 306.2 | 0.526 | 84.6% | 90.8% |

Verdict: performance-neutral (dMAE +0.5, within split noise) - adopted for correctness:
"no endgame existed" is no longer encoded as "0 endgame errors", and absence is explicitly
splittable via has_middlegame/has_endgame/has_clock. CQR coverage is robust to either encoding.

## 2026-07-05 21:43 — improved model
n_train=329353, n_calib=59124, n_test=120513, n_features=69

| stage | MAE | RMSE |
|---|---|---|
| no-engine baseline | 293.4 | — |
| + engine/clock | 239.4 | 300.4 |

- median AE 202, R² 0.544, Spearman 0.726, within 100/200 Elo 26%/50%, bias -3.3
- per-player (all games averaged): **218.4** MAE over 50357 players
- 90% interval coverage: raw 87.5% -> CQR **89.8%** (width 945 -> 998 Elo; correction lo=26.6, hi=26.6)
- coverage by band: 0-1200 78%, 1200-1400 95%, 1400-1600 98%, 1600-1800 99%, 1800-2000 95%, 2000-3000 80%
- band accuracy: 36.5% exact, 75.9% adjacent
- aggregation MAE: K1=235, K2=209, K3=202, K5=200, K10=203, K20=218
- ablation (MAE↑ when dropped): engine +33.8, clock +9.8, style +8.1, opening +8.0

## 2026-07-06 01:12 — improved model (tuned)
n_train=329353, n_calib=59124, n_test=120513, n_features=69

| stage | MAE | RMSE |
|---|---|---|
| no-engine baseline | 293.4 | — |
| + engine/clock | 239.4 | 300.4 |
| + tuned | 237.4 | 297.9 |

- median AE 201, R² 0.552, Spearman 0.730, within 100/200 Elo 26%/50%, bias -3.3
- per-player (all games averaged): **216.7** MAE over 50357 players
- 90% interval coverage: raw 87.1% -> CQR **89.8%** (width 931 -> 992 Elo; correction lo=30.0, hi=30.0)
- coverage by band: 0-1200 78%, 1200-1400 95%, 1400-1600 98%, 1600-1800 98%, 1800-2000 95%, 2000-3000 80%
- band accuracy: 36.6% exact, 76.2% adjacent
- aggregation MAE: K1=233, K2=207, K3=201, K5=199, K10=202, K20=217
- ablation (MAE↑ when dropped): engine +33.8, clock +9.8, style +8.1, opening +8.0

## 2026-07-06 01:29 — improved model
n_train=329353, n_calib=59124, n_test=120513, n_features=77

| stage | MAE | RMSE |
|---|---|---|
| no-engine baseline | 293.4 | — |
| + engine/clock | 239.1 | 300.1 |

- median AE 202, R² 0.545, Spearman 0.726, within 100/200 Elo 26%/50%, bias -3.4
- per-player (all games averaged): **218.2** MAE over 50357 players
- 90% interval coverage: raw 87.4% -> CQR(plain) **89.8%** (width 945 -> 999 Elo; correction lo=27.1, hi=27.1)
- Mondrian CQR: **89.9%** (width 1001 Elo) — headline
- coverage by band (headline=mondrian): 0-1200 78%, 1200-1400 96%, 1400-1600 98%, 1600-1800 98%, 1800-2000 95%, 2000-3000 81%
- per-band coverage, plain/Mondrian: 0-1200 78%/78%, 1200-1400 95%/96%, 1400-1600 98%/98%, 1600-1800 99%/98%, 1800-2000 95%/95%, 2000-3000 80%/81%
- band accuracy: 36.4% exact, 75.9% adjacent
- aggregation MAE: K1=235, K2=209, K3=202, K5=200, K10=203, K20=217
- ablation (MAE↑ when dropped): engine +33.6, clock +9.1, opening +8.2, style +7.9

## 2026-07-07 00:11 — improved model (tuned)
n_train=329353, n_calib=59124, n_test=120513, n_features=77

| stage | MAE | RMSE |
|---|---|---|
| no-engine baseline | 293.4 | — |
| + engine/clock | 239.1 | 300.1 |
| + tuned | 237.2 | 297.7 |

- best_params: `{'num_leaves': 178, 'learning_rate': 0.010507258889041039, 'feature_fraction': 0.5034331983772904, 'bagging_fraction': 0.8786594745277075, 'bagging_freq': 1, 'min_child_samples': 112, 'reg_lambda': 0.9635897279825012}`

- median AE 200, R² 0.553, Spearman 0.731, within 100/200 Elo 26%/50%, bias -3.0
- per-player (all games averaged): **216.7** MAE over 50357 players
- 90% interval coverage: raw 87.0% -> CQR(plain) **89.7%** (width 929 -> 991 Elo; correction lo=31.1, hi=31.1)
- Mondrian CQR: **89.8%** (width 992 Elo) — headline
- coverage by band (headline=mondrian): 0-1200 77%, 1200-1400 96%, 1400-1600 98%, 1600-1800 98%, 1800-2000 95%, 2000-3000 81%
- per-band coverage, plain/Mondrian: 0-1200 78%/77%, 1200-1400 95%/96%, 1400-1600 98%/98%, 1600-1800 98%/98%, 1800-2000 95%/95%, 2000-3000 80%/81%
- band accuracy: 36.6% exact, 76.2% adjacent
- aggregation MAE: K1=233, K2=208, K3=201, K5=198, K10=201, K20=216
- ablation (MAE↑ when dropped): engine +33.6, clock +9.1, opening +8.2, style +7.9
