
## 471b5df — baseline (no-engine features)
n_train=486574, n_test=121984, features=game_plies, player_moves, n_captures, n_checks, result, time_control, eco

| model | MAE | RMSE |
|---|---|---|
| predict_mean | 353.0 | 432.6 |
| copy_opponent | 77.7 | 146.4 |
| ridge | 292.7 | 363.4 |
| lightgbm | 288.1 | 360.3 |

## 471b5df — improved model
n_train=330736, n_calib=57531, n_test=121984, n_features=77

| stage | MAE | RMSE |
|---|---|---|
| no-engine baseline | 289.3 | — |
| + engine/clock | 237.1 | 297.5 |

- median AE 201, R² 0.527, Spearman 0.711, within 100/200 Elo 26%/50%, bias -0.4
- per-player (all games averaged): **218.6** MAE over 49488 players
- 90% interval coverage: raw 87.7% -> CQR(plain) **90.4%** (width 943 -> 1003 Elo; correction lo=30.1, hi=30.1)
- Mondrian CQR: **90.3%** (width 1002 Elo) — headline
- coverage by band (headline=mondrian): 0-1200 77%, 1200-1400 95%, 1400-1600 98%, 1600-1800 98%, 1800-2000 96%, 2000-3000 82%
- per-band coverage, plain/Mondrian: 0-1200 78%/77%, 1200-1400 95%/95%, 1400-1600 98%/98%, 1600-1800 99%/98%, 1800-2000 96%/96%, 2000-3000 81%/82%
- band accuracy: 35.5% exact, 75.7% adjacent
- aggregation MAE: K1=234, K2=207, K3=200, K5=198, K10=203, K20=215
- ablation (MAE↑ when dropped): engine +33.2, clock +8.8, opening +8.0, style +7.2
