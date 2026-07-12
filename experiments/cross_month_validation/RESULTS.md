# Cross-Month Validation — Reproducibility & Generalization, 2025-05 vs 2026-05

**Question.** Every headline claim in this project — "MAE ≈ 239 Elo," "the 90% interval really covers
90%," "move quality is the dominant signal," "the tail error is an irreducible noise floor" — was
measured on **one month of data (2025-05)**. Are those numbers a real property of the method, or a
convenient sample? This study answers two distinct versions of that worry a full year apart, against
an independent month (**2026-05**):

1. **Within-month reproduction.** Re-run the *entire* pipeline (ingest → clean → features → model →
   evaluate) on 2026-05 from scratch and check whether every number reproduces. Rules out "2025-05
   happened to be an easy/lucky draw."
2. **Cross-month generalization.** Train the full model on 2025-05 and evaluate it, untouched, on
   2026-05 — including a player-disjoint slice and an interval-exchangeability check. Rules out "the
   model only works on the data it was born from."

**Verdict: strongly reproducible.** Every headline number reproduces within ~1–2% on a fresh month a
full year later, and a 2025-trained model transfers to 2026 with only **+1.8 MAE** (full) / **+4.4
MAE** (player-disjoint). The one measurable shift is a small **+16 to +40 Elo positive bias** (mild
rating drift) — it's a shift in *level*, not in *signal* (Spearman ρ barely moves). Notably, the
2025-calibrated **CQR interval still covers 89.6%** of 2026 (target 90), so the uncertainty guarantee
is not month-specific either.

> Both runs used the identical `config.yaml` (seed 42), a matched **15M-game scan budget**, and the
> same `src/` code at commit `471b5df`. Reproduce with `pipeline_2026.sh` (Experiment A) then
> `run_cross_month.py` (Experiment B) and `gen_figures.py`. Raw numbers in `results.json` /
> `results_2026.md`. 2026 parquets live in `data/processed/*_2026*` (gitignored) — nothing but this
> folder's numeric/figure output is committed.

---

## 1. Data funnel — the dumps themselves are consistent

| stage | 2025-05 | 2026-05 | note |
|---|---:|---:|---|
| scanned | 15,000,000 | 15,000,000 | same budget |
| blitz | 6,980,000 | 6,911,606 | ~46% both |
| carries a stored `[%eval]` | 648,594 | **738,063** | eval adoption **grew ~14%** YoY |
| clean | 647,983 | 737,557 | |
| sampled | 300,000 | 300,000 | + cohort 6,117 / 7,135 |
| instances (features) | 606,578 | 608,558 | 2 per game |

The only structural change in a year is that **more players request engine analysis** (4.9% of
scanned vs 4.3%). Everything else about the funnel is stable.

---

## 2. Within-month reproduction (train & test both on 2026)

| metric | 2025-05 | 2026-05 | reading |
|---|---:|---:|---|
| predict-mean (floor) | 364.8 | 353.0 | 2026 ratings slightly *less spread* (see R² note below) |
| copy-opponent (leak) | 80.9 | 77.7 | ≈ |
| ridge (no-engine) | 297.9 | 292.7 | ≈ |
| LightGBM (no-engine) | 292.5 | 288.1 | ≈ |
| no-engine baseline MAE | 293.4 | 289.3 | ≈ |
| **full model MAE** | **239.1** | **237.1** | essentially identical |
| full model RMSE | 300.1 | 297.5 | ≈ |
| median AE | 202 | 201 | ≈ |
| R² | 0.545 | 0.527 | narrower 2026 target → lower R² despite better MAE (below) |
| Spearman ρ | 0.726 | 0.711 | ≈ |
| within 100 / 200 Elo | 26% / 50% | 26% / 50% | identical |
| bias | −3.4 | −0.4 | ≈ 0 both |
| per-player MAE (all games) | 218.2 | 218.6 | identical |
| raw 90% coverage | 87.4% | 87.7% | ≈ |
| **CQR coverage (plain)** | **89.8%** | **90.4%** | both hit ~90% |
| Mondrian CQR coverage | 89.9% | 90.3% | ≈ |
| mean interval width | 999 | 1003 | ≈ |
| band accuracy (exact / adjacent) | 36.4% / 75.9% | 35.5% / 75.7% | ≈ |
| aggregation MAE K=1 → K=5 | 235 → 200 | 234 → 198 | identical |
| ablation (engine/clock/opening/style) | +33.6/+9.1/+8.2/+7.9 | +33.2/+8.8/+8.0/+7.2 | ≈ |
| tail residual (0–1200 / 2000+) | +298 / −246 | +312 / −252 | ≈ |

Every quantity reproduces within ~1–2%. *R² note:* 2026's `predict-mean` floor (353 vs 365) means the
2026 rating distribution is slightly narrower — R² measures variance *explained*, so a narrower target
naturally yields a lower R² **even though the absolute error (MAE) is a touch better** — R² and MAE
answer different questions. See `figures/cmp_within_month_metrics.png`; full run log in
`results_2026.md` (auto-appended by `src.model`/`src.evaluate`, same format as the main
`reports/results.md`).

---

## 3. Cross-month generalization (train 2025 → test 2026)

The 2025-trained model, applied unchanged to 2026. "Disjoint" drops every 2026 player whose username
also appears in 2025 train/val/calib (**23.5% overlap**), leaving 465,389 genuinely unseen players —
this is the strict, honest transfer number (the grouped-split leakage rule, applied across months).

| metric | 2025 holdout (ref) | 2026 full | 2026 disjoint |
|---|---:|---:|---:|
| n | 120,513 | 608,558 | 465,389 |
| **MAE** | **239.1** | **240.9** | **243.5** |
| RMSE | 300.1 | 302.2 | 305.2 |
| median AE | 202 | 203 | 206 |
| R² | 0.545 | 0.517 | 0.503 |
| Spearman ρ | 0.726 | 0.707 | 0.701 |
| within 200 Elo | 49.7% | 49.3% | 48.8% |
| band exact | 36.4% | 35.1% | 34.4% |
| **bias** | **−3.4** | **+16.2** | **+39.8** |

**Reading.**
- **Accuracy transfers.** A model a full year stale loses just **1.8 MAE** on the whole month and
  **4.4 MAE** on brand-new players (`figures/cmp_crossmonth_mae.png`) — for a static model with no
  retraining, that's a strong result.
- **The one real drift is bias**, not signal. On its own 2025 hold-out the model is unbiased (−3.4);
  on 2026 it **over-predicts** by +16 Elo overall and +40 Elo among unseen players (the full-set bias
  is smaller because the 23.5% overlapping accounts are literally the same players, anchoring their
  predictions). Interpretation: at a given style/quality signature, 2026 players are rated slightly
  **lower** than their 2025 counterparts — mild rating drift/deflation over the year.
- **It's a shift in *level*, not in *signal*.** Spearman ρ barely moves (0.726 → 0.70) — the model
  still *orders* players correctly; only the absolute calibration slipped. A single scalar correction
  (subtract ~16–40 Elo) or a periodic re-fit on recent data absorbs it.

---

## 4. The uncertainty guarantee is not month-specific

The split-CQR correction calibrated **on 2025** (±27.1 Elo), applied unchanged to 2026:

| interval | coverage | reading |
|---|---:|---|
| 2025 holdout, raw | 87.4% | under-covers before calibration |
| 2025 holdout, CQR | 89.8% | calibration restores ~90% |
| **2026, raw** | **87.3%** | same raw under-coverage a year later |
| **2026, CQR (2025-calibrated)** | **89.6%** | **the guarantee transfers** (width 997) |

This is a direct **exchangeability** result — the calibration data and 2026 test data behave like
draws from the same process, so the calibration generalizes without re-earning it each month
(`figures/cmp_cqr_coverage.png`). Per-band coverage on 2026 shows the same familiar tail dip
(0–1200 ≈ 74%, 2000+ ≈ 82%) — the single-game noise floor, reproduced. (For the CQR mechanics
themselves — what "conformal," "coverage," and "Mondrian" mean — see the main project's
`PROJECT_ARCHITECTURE.md` / notebook 04; this study only tests whether they *transfer*.)

---

## 5. The tail noise floor reproduces three times over

Mean residual by rating band — the regression-to-the-mean signature from notebook 05 — is nearly
identical across 2025-within, 2026-within, and the 2025→2026 cross-month setting:

| band | 2025 within | 2026 within | 2025→2026 cross |
|---|---:|---:|---:|
| 0–1200 | +298 | +312 | +328 |
| 1200–1400 | +153 | +163 | +177 |
| 1400–1600 | +46 | +53 | +69 |
| 1600–1800 | −49 | −42 | −26 |
| 1800–2000 | −135 | −134 | −116 |
| 2000–3000 | −246 | −252 | −240 |

The cross-month curve sits a touch higher everywhere (the +bias), but the *shape* — over-predict the
low tail, under-predict the high tail — is invariant (`figures/cmp_residual_by_band.png`). This is
exactly what notebook 05 argued: the tail error is a structural property of single-game evidence, not
an artifact of one particular month's data.

---

## 6. Conclusions

- **The findings are robust, not lucky.** Re-running the whole pipeline on an independent month a year
  later reproduces every headline (MAE 239→237, CQR ~90%, ablations, aggregation curve, tail bias) to
  within noise. The project's conclusions are not artifacts of the 2025-05 sample.
- **The model generalizes temporally.** Trained on 2025, it predicts 2026 within ~2 MAE (full) / ~4
  MAE (unseen players), and still ranks players correctly (ρ ≈ 0.70).
- **The only drift is a small level bias** (+16 to +40 Elo) — monotone and uniform enough that a
  one-line recalibration (or periodic re-fit on the latest month) removes it; the *conditional* signal
  (which features matter, how much, with what uncertainty) is unchanged.
- **The uncertainty is portable.** The conformal interval calibrated on 2025 still covers ~90% on
  2026 — no monthly re-calibration needed under normal drift.

**Bottom line / production recommendation:** periodically re-fit or bias-correct the *point* model on
recent data; the *interval* machinery can be left as-is.

### Caveats
- Both months share the same **selection bias** (eval'd-games only). This validates reproducibility
  *within that population*, not transfer to a random-game sample.
- **One month-pair.** This shows year-over-year stability for 2025-05 vs 2026-05, not a full
  seasonality study. The cheap next step is a third month (e.g. 2026-06) to confirm the +bias is drift
  rather than a one-off.
- ⚠️ Running the 2026 `evaluate` step also **overwrites the 6 pipeline figures in `reports/figures/`**
  (root) with 2026 versions, since `src.evaluate` always writes there. To restore the committed 2025
  figures after a re-run: `git checkout -- reports/figures/*.png`.

---

## Metric cheat-sheet

For anyone unfamiliar with the project's evaluation vocabulary — the tables above use these
throughout (see `src/evaluate.py` for the implementations):

| Metric | Meaning | Good value here |
|---|---|---|
| **MAE** | `mean(\|pred − actual\|)`, in Elo — the headline "typical miss" | ~239 (single game) |
| **RMSE** | Like MAE but squares errors first, so big misses count more; gap vs MAE ⇒ heavy tails | ~300 |
| **Median AE** | Middle absolute error; < MAE here ⇒ right-skewed error (most games decent, a few badly wrong) | ~202 |
| **R²** | Fraction of rating *variance* explained (0=no better than the mean, 1=perfect); sensitive to test-set spread | ~0.53–0.55 |
| **Spearman ρ** | Rank correlation — does the model order players correctly regardless of scale/bias? | ~0.71–0.73 |
| **bias** | `mean(pred − actual)`; + = over-predicts, − = under-predicts | ~0 within-month; **+16 to +40 cross-month** |
| **band exact/adjacent** | Coarse 6-band classification view (for the confusion matrix) | 36% / 76% |
| **per-player MAE** | MAE after averaging all of a player's games — the "more than one game" number | ~218 |
| **coverage** | Of all 90%-nominal intervals, what fraction actually contained the truth | target 90% |
| **CQR (Conformalized Quantile Regression)** | Widens/narrows a raw quantile-regression interval by the empirically-observed miss on a held-out calibration set, giving a distribution-free, finite-sample coverage guarantee *as long as calibration and test data are exchangeable* | restores raw ~87% → ~90% |
| **Mondrian CQR** | Same idea, one correction per predicted-rating band instead of one global correction | ~identical to plain CQR here |
| **player-disjoint** | Cross-month test restricted to usernames the training-month model never saw — a clean transfer measure, no cross-month leakage | |
| **exchangeability** | The assumption CQR's guarantee rests on — that calibration and test data are draws from the same distribution; testing a 2025-calibrated interval on 2026 is a direct check of whether that survives a year of drift | |

*Lower is better:* MAE, RMSE, median AE, interval width, `\|bias\|`, ablation MAE-increase, tail
residual magnitude. *Higher is better:* R², Pearson/Spearman, within-100/200, band accuracy.
*Should hit its target (~90%), not be maximised:* coverage.

---

## Files

| path | what |
|---|---|
| `pipeline_2026.sh` | Experiment A driver — full ingest→evaluate pipeline on a new month |
| `run_cross_month.py` | Experiment B driver — 2025-trained model evaluated on 2026 + CQR transfer |
| `gen_figures.py` | the 4 comparison figures (reproduce after re-running A+B) |
| `results.json` | Experiment B raw numbers (point metrics, residual-by-band, CQR) |
| `results_2026.md` | Experiment A auto-logged run (same append format as `reports/results.md`) |
| `figures/cmp_within_month_metrics.png` | 2025 vs 2026 headline metrics, within-month |
| `figures/cmp_crossmonth_mae.png` | MAE ladder: 2025 ref → 2026 within → 2026 cross (full/disjoint) |
| `figures/cmp_residual_by_band.png` | tail-bias overlay across all three settings |
| `figures/cmp_cqr_coverage.png` | CQR coverage: within-month and cross-month-transferred |

2026 parquets (`data/processed/*_2026*.parquet`) are gitignored and not reproduced here — regenerate
with `pipeline_2026.sh` before re-running `run_cross_month.py`.
