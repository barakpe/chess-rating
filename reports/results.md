# Experiment log

The running record of model runs — how three people stay in sync on "what's our best model right
now." Each run appends a block (features used, n_train/n_test, MAE/RMSE). `python -m src.model`
appends here automatically. Newest at the bottom.

MAE/RMSE are in Elo points. Reference baselines:
- **predict_mean** — predict the training-mean rating for everyone (trivial floor).
- **copy_opponent** — predict the opponent's rating (strong, but a leak; excluded from the model).
