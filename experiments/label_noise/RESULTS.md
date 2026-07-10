# Label-Noise Study — how much of our error is unstable *labels*, not the model?

**Question.** New players (provisional ratings that haven't converged) and returning players (ratings
gone stale during inactivity) both carry **noisy labels** — the shown Elo is a poor estimate of true
skill. Do these inflate our error, and are they behind the headline **tail** weakness?

**Answer.**
- We **cannot cut them directly** — Lichess strips the provisional flag from the PGN dump (verified: 0
  of 30,000 raw games carry a provisional-marked Elo). We use the per-game **rating change**
  (`WhiteRatingDiff`/`BlackRatingDiff`, present ~99%) as a **Glicko rating-deviation proxy**: a large
  |change| ⇒ a high-RD, unsettled label. *(It is only a diagnostic — never a model feature, since it is
  derived from the result and the opponent's rating, i.e. the banned leak.)*
- **Label noise is real but small in aggregate** (~2 Elo of the headline MAE) and shows up in the
  **middle** bands, not the tails.
- **The tail bias is NOT label noise.** On trustworthy (stable) labels the low tail still over-predicts
  **+296** and the high tail under-predicts **−245** — essentially unchanged from the unstable games.
  The tail weakness is genuine regression to the mean, not bad labels.
- **Clean-label training does nothing** (236.7 → 236.5): tree ensembles are robust to label noise; the
  effect is purely test-side measurement.

> Strictly 2025-05, headline point model. Reproduce with
> `python experiments/label_noise/run_label_noise_study.py`. Raw numbers in `results.json`.

---

## 1. Error vs label instability — and why the raw view misleads

Binning test games by |rating change| (`figures/ln_mae_vs_instability.png`):

| \|Δrating\| | n | MAE | bias |
|---|--:|--:|--:|
| 0–5 | 18,937 | 256.2 | −49.3 |
| 5–15 | 88,034 | **232.4** | −7.8 |
| 15–30 | 6,485 | 254.1 | +99.4 |
| 30–60 | 3,144 | 257.0 | +108.2 |
| 60+ | 2,745 | 255.0 | +65.7 |

At face value MAE is U-shaped and the **bias flips sign** (−49 → +108). But this is **confounded by
rating band**: tiny changes skew toward *high-rated*, low-RD players (who we under-predict, −bias),
large changes skew toward *low-rated* provisional players (who we over-predict, +bias). The raw curve
is a band-mix artifact — so we control for band next. (This is exactly the within-band discipline the
aggregation study also required.)

---

## 2. The controlled test — stable vs unstable label, *within* each band

Splitting each band into stable-label (|Δ|≤15, low RD) and unstable-label (|Δ|≥30, high RD) games
(`figures/ln_per_band_mae.png`, `figures/ln_per_band_bias.png`):

| band | MAE stable | MAE unstable | **MAE gap** | bias stable | bias unstable |
|---|--:|--:|--:|--:|--:|
| 0–1200 | 309.4 | 320.7 | +11.2 | **+295.6** | **+294.8** |
| 1200–1400 | 214.3 | 215.1 | +0.7 | +155.0 | +118.2 |
| 1400–1600 | 183.0 | 220.5 | **+37.4** | +49.5 | −9.6 |
| 1600–1800 | 187.1 | 217.6 | **+30.5** | −48.9 | −59.6 |
| 1800–2000 | 218.9 | 215.0 | −3.9 | −135.1 | −115.0 |
| 2000–3000 | 280.4 | 275.4 | −5.1 | **−245.1** | **−231.3** |

Two clean conclusions:

- **Label noise costs in the *middle*, not the tails.** The MAE gap is largest in the noise-dominated
  1400–1800 bands (**+30 to +37**), where the systematic bias is near zero so label scatter is the main
  residual and shows up directly. In the tails the gap is small (+11, −4, −5): the huge systematic bias
  there swamps any label-noise contribution.
- **The tail bias survives on trustworthy labels.** In the 0–1200 band, the 17,069 *stable*-label
  (established, low-RD) players are still over-predicted by **+296 Elo** (MAE 309); the 2000+ stable
  players are under-predicted by **−245**. These are essentially identical to the unstable-label
  numbers. **So provisional / stale ratings are *not* the cause of the tail weakness** — it is genuine
  regression to the mean under weak single-game evidence, exactly as notebook 05 and the aggregation
  study concluded. Better labels would not fix it.

---

## 3. Aggregate impact and clean-label training

| view | n | MAE |
|---|--:|--:|
| all games (with a rating change) | 119,345 | 238.5 |
| stable label only (|Δ|≤15) | 107,712 | 236.7 |
| unstable label (|Δ|≥30) | 5,889 | 256.0 |

Restricting the *evaluation* to trustworthy labels improves the headline by only **~2 Elo** (238.5 →
236.7) — unstable games are ~5% of the data and mostly sit where bias already dominates.

**Clean-label training** — retrain on the 89.5% of train games with |Δ|≤15, evaluate on stable-label
test:

| | stable-test MAE |
|---|--:|
| full-label training | 236.7 |
| clean-label training | 236.5 |

**No effect.** Gradient-boosted trees average label noise out across the ensemble, so dropping noisy
training labels buys nothing. The label-noise penalty is entirely a *measurement* artifact on the test
side, not a model-quality problem.

---

## 4. Conclusions & guidance

- **We do not (and cannot) cut new/returning players**; the dump lacks the provisional/RD flag. The
  `WhiteRatingDiff` magnitude is the only available instability proxy (a diagnostic, never a feature).
- **Label noise is real but minor**: ~2 Elo on the headline, concentrated as +30–37 MAE in the
  noise-dominated **middle** bands (1400–1800). It is *invisible* in the tails.
- **The tail weakness is genuine, not a labeling artifact.** On trustworthy labels the ±250–300 tail
  bias is unchanged — the third independent confirmation (with notebook 05 and the aggregation study)
  that the tails are an *evidence* limit, curable only by more/other per-game information, not by better
  labels or more games.
- **Do not bother cleaning training labels** — no benefit. Optionally **flag the ~5% high-|Δ| games**
  at inference: for those, the *measured* error is less trustworthy even though the model is fine.

### Caveats
- The stable-vs-unstable MAE gap is an **upper bound** on the pure label-noise effect: high-RD players
  may also simply play more erratically (genuinely harder to predict), which this proxy cannot separate
  from noisy labels.
- 2025-05 only; the eval'd-games selection bias applies.

## Files
| path | what |
|---|---|
| `run_label_noise_study.py` | the analysis (reproducible) |
| `results.json` | every number above |
| `figures/ln_mae_vs_instability.png` | raw MAE/bias by instability (band-confounded — see below) |
| `figures/ln_per_band_mae.png` | per-band MAE, stable vs unstable label (the controlled view) |
| `figures/ln_per_band_bias.png` | per-band bias — flat in the tails (label-independent) |
