# Notebook blueprint (Phase 3)

Handoff spec for building the presentation notebooks. Rules of engagement: notebooks are for
**looking, not doing** — every computation calls `src/` functions or reads the processed
parquets; no logic is reimplemented in cells. Each notebook runs top-to-bottom (Restart & Run
All), uses `src.plotting.set_style()` in the first cell, and saves every deck figure via
`save_fig()` → `reports/figures/`. All headline numbers must be quoted from
`reports/results.md` (never hand-typed from memory).

**Recommendation: five notebooks, not four** — error analysis & limitations has grown into the
academically strongest material (three rigorous negative results) and deserves its own
narrative rather than being an appendix to the improved model.

Data inputs (shared drive, gitignored): `blitz_sample_300k.parquet`,
`games_clean_300k.parquet`, `instances_300k.parquet`, `features_300k_v2.parquet`,
`player_cohort.parquet`.

---

## 01_data_and_eda — "Is rating even visible in one game?"
**Purpose:** the data story + first evidence the signal exists. Rubric rows: Data, EDA.
**Calls:** read `games_clean_300k` / `instances_300k` / `features_300k_v2`; `src.config.load_config`.
**Content:** data funnel table (15M scanned → 6.98M blitz → 648,594 eval'd (9.3%) → 647,983
clean → 300k sampled + 6,117 player-cohort); rating distribution + bands; Option-A selection-bias
disclaimer; games-per-player distribution (median 1 — motivates the cohort). EDA: CPL & Accuracy%
vs rating, blunder-rate-by-phase vs rating, time-usage vs rating, book depth by band, P(win) vs
rating gap teaser, feature↔rating correlation heatmap.
**Insight to land:** move quality correlates strongly with rating, but with huge per-game spread —
the tension driving the whole project.

## 02_feature_engineering — "From PGN to 77 features"
**Purpose:** how a game becomes a feature vector. Rubric row: EDA/method bridge.
**Calls:** `src.features.extract_player_features` on ONE real game (worked example: per-move table
of eval → Win% → Accuracy% → class), plus the pure primitives `win_percent` / `accuracy_percent` /
`classify_move`; distributions from `features_300k_v2`.
**Content:** phase splitting (opening/middlegame/endgame boundaries); the **missing-value
contract** (NaN = unobservable, 0 = true zero, `has_*` flags; 38% of games have no endgame);
clock features incl. the scramble block; leakage guard (no opponent-derived features, grouped
splits) — explain *why* copy-opponent is banned.
**Insight to land:** encoding absence correctly matters for trees; a game that ends early is
itself information.

## 03_baseline — "How far do shallow features get you?"
**Purpose:** reference points. Rubric row: Baseline + its error analysis.
**Calls:** `src.model.evaluate` (Stage 5) or reuse its pieces: `make_ridge`, `make_lgbm`,
`grouped_split`, the `predict_mean` / `copy_opponent` references.
**Content:** MAE table — predict-mean 293, copy-opponent ~80 (the excluded leak, shown to justify
excluding it), ridge vs LightGBM on no-engine features (~285-293); predicted-vs-actual scatter;
residual distribution.
**Insight to land:** matchmaking already "knows" your rating (copy-opponent), which is exactly why
it must be excluded; without engine evals, game-shape features barely beat the mean.

## 04_improved_model — "The full model + honest uncertainty"
**Purpose:** the headline system. Rubric row: Improved model.
**Calls:** `src.evaluate.run_evaluation` (single call reproduces everything) or read its outputs;
tuned hyperparameters from the latest `results.md` `best_params` line.
**Content:** improvement table (293.4 → 239.1 → ~237 tuned); SHAP summary + dependence (expect
CPL/post-book accuracy on top; engine group ablation +34); **CQR intervals** — raw 87.4% → 89.8%
guaranteed marginal coverage, how split-conformal works (one paragraph, cite Romano et al. 2019);
band confusion matrix; **aggregation curve** (MAE 235 at K=1 → 200 at K=5, via the player cohort)
as the headline chart.
**Insight to land:** a single game gives ±~240 Elo; five games give ~200; the interval is
*guaranteed*, not hoped for.

## 05_error_analysis_and_limits — "What we proved we cannot fix"
**Purpose:** the final analysis — three closed hypotheses. Rubric row: Final Analysis /
limitations. This is the differentiating academic content.
**Calls:** `src.evaluate.run_tail_study`; `_residual_by_band` / `_largest_residuals` / ablation
outputs from `run_evaluation`; per-band coverage tables (plain vs Mondrian) from `results.md`.
**Content:** residual-by-band (over-predict <1200 by ~+300, under-predict 2000+ by ~−250);
**closed hypothesis 1** — de-shrink slope ≈ 1.0: the model is already conditionally calibrated,
reweighting only redistributes; **closed hypothesis 2** — scramble features: real signal (median
+8 cpl degradation under time pressure) yet tails unchanged (313/282); **closed hypothesis 3** —
Mondrian CQR: per-true-band coverage identical to plain CQR because prediction shrinkage empties
the extreme predicted bands (test-time-legal conditioning cannot reach them). Largest-residual
case gallery; limitations recap (Option-A bias, provisional labels, blitz-only).
**Insight to land:** tail error is a single-game **noise floor** — demonstrated three independent
ways — and aggregation is the only lever that moves it.
