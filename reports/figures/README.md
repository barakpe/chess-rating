# Figures

Saved PNGs the notebooks and the slide deck use. Every figure is written by
`src.plotting.save_fig` in one style and at a fixed dpi, so it can be regenerated when the data
changes without hand-editing.

| location | produced by | contents |
|---|---|---|
| `./` (root) | `python -m src.evaluate` | pipeline figures: `pred_vs_actual`, `calibration`, `band_confusion`, `interval_by_band`, `aggregation_curve` (shifting cohort), `shap_summary` |
| `notebook_01_eda/` | `notebooks/01_data_and_eda.ipynb` | rating distribution, band sizes, games per player, quality / blunders-by-phase / time vs rating, opening vs post-opening accuracy, opponent leak, correlation map, CPL overlap by band |
| `notebook_02_features/` | `notebooks/02_feature_engineering.ipynb` | Win%/Accuracy% primitives, eval trajectory by phase, presence flags, scramble degradation |
| `notebook_03_baseline/` | `notebooks/03_baseline.ipynb` | baseline ladder, predicted vs actual, residuals |
| `notebook_04_model/` | `notebooks/04_improved_model.ipynb` | improvement table, ablation, CQR coverage, coverage by band, aggregation (shifting vs matched, naive vs recalibrated) |
| `notebook_05_limits/` | `notebooks/05_error_analysis_and_limits.ipynb` | residual by band, single-game tail study, scramble ablation, Mondrian vs plain coverage, bias/MAE by band for 1 vs 5 games (naive / recalibrated), error by game length, worst misses |

Notebook 04 also shows `shap_summary.png` and `band_confusion.png` from the root.
