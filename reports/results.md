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
