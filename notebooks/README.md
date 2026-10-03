# Notebooks

Five notebooks tell the project's story in order. They are for **looking**, not doing: every
computation calls a function in `src/` or reads a file the pipeline wrote, and every figure is saved
to `reports/figures/notebook_0X_*/` with `src.plotting.save_fig` (the slides use the same PNGs).
They are committed **with their outputs**, so they can be read on GitHub without running anything.

| notebook | question | presentation section |
|---|---|---|
| [`01_data_and_eda`](01_data_and_eda.ipynb) | Is rating visible in a single game? The data funnel, the label, games per player, six EDA probes, the opponent-rating leak. | Data, Exploratory Data Analysis |
| [`02_feature_engineering`](02_feature_engineering.ipynb) | How does a PGN become 77 features? Lichess's formulas, a worked game, phases, the missing-value contract, clock-scramble features. | EDA → method |
| [`03_baseline`](03_baseline.ipynb) | How far do shallow features get you? predict-mean, copy-opponent (the leak), ridge, LightGBM without engine features; error analysis. | Basic ML model |
| [`04_improved_model`](04_improved_model.ipynb) | The full model: improvement table with CIs, SHAP and ablations, conformal (CQR) intervals, coverage by band, several games per player with a K-aware recalibration. | Improved ML model |
| [`05_error_analysis_and_limits`](05_error_analysis_and_limits.ipynb) | Why the rating extremes are hard: single-game options tested (calibration check, reweighting, scramble features, Mondrian intervals), what several games do, leakage and robustness checks, worst misses, limitations. | Error analysis, Final analysis |

## Running them

1. Set up the environment (see the main README) and put the processed data in `data/processed/`:
   either run the pipeline, or copy the three parquets of the slim data bundle
   (`python scripts/make_share_bundle.py` builds it from a full run).
2. Run each notebook top to bottom (Kernel → Restart & Run All), or all of them from the repo root:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
   ```

Inputs they read (all paths from `config.yaml → outputs`):

| file | written by | used in |
|---|---|---|
| `data/processed/features.parquet` | `python -m src.features` | 01, 02, 03, 05 |
| `data/processed/games_clean.parquet` | `python -m src.clean` | 01, 02, 03 |
| `data/processed/player_cohort.parquet` | `python -m src.ingest` | 01 |
| `reports/data_funnel.json` | `src.ingest` + `src.clean` | 01 |
| `reports/eval_artifacts.json`, `reports/largest_residuals.csv`, `reports/figures/*.png` | `python -m src.evaluate` | 03, 04, 05 |
| `reports/results.md` (tuned run, tail study) | `python -m src.evaluate --tune`, `--tail-study` | 04, 05 |

Notebook 03 refits the Stage-5 baselines live (~1–2 min). Notebooks 04 and 05 read the saved
evaluation by default; set `RUN_HEAVY = True` in their setup cell to retrain live.
