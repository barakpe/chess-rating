# Figures

Saved PNGs the notebooks and slide deck pull from. Every figure is written by
`src.plotting.save_fig` at a consistent size/dpi, so it can be regenerated when the data updates
without hand-editing.

## Layout

| Location | Produced by | Contents |
|---|---|---|
| `./` (root) | `python -m src.evaluate` (`src.plotting.save_fig`) | Pipeline figures: `pred_vs_actual`, `calibration`, `band_confusion`, `interval_by_band`, `aggregation_curve`, `shap_summary` |
| `notebook_01_eda/` | `notebooks/01_data_and_eda.ipynb` | Rating distribution, bands, games-per-player, quality/blunder/time/book EDA, opponent leak, correlation map, CPL overlap |
| `notebook_02_features/` | `notebooks/02_feature_engineering.ipynb` | Win%/Accuracy% primitives, eval trajectory by phase, presence flags, scramble degradation |
| `notebook_03_baseline/` | `notebooks/03_baseline.ipynb` | Baseline ladder, predicted-vs-actual, residuals |
| `notebook_04_model/` | `notebooks/04_improved_model.ipynb` | Improvement table, ablation, CQR coverage, per-band coverage, aggregation curve |
| `notebook_05_limits/` | `notebooks/05_error_analysis_and_limits.ipynb` | Residual by band, tail study, scramble, Mondrian, worst-case gallery, aggregation |

Notebook 04 also displays the pipeline `shap_summary.png` and `band_confusion.png` from the root
(where `src.evaluate` writes them). Notebooks write their own deck figures into their subfolder; the
folder is created automatically on first run.
