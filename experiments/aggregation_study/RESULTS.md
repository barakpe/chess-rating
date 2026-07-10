# Matched Aggregation Study — reducing single-game noise by averaging a player's games

**Hypothesis.** A player's rating is estimated with ±~239 Elo of error from a *single* game. If we
isolate players with several games and **average their per-game predictions**, the noise should
cancel and accuracy should improve.

**Result.** It does — single-game MAE **239 → 200** when averaging a multi-game player's games — but
it falls **far short of the ideal `1/√K`**, and it improves the **middle bands, not the tails**. A
variance decomposition explains why: most of the single-game error is **systematic per-player bias**
(regression to the mean), which averaging cannot touch; only the smaller **noise** component obeys the
`1/√K` law. The project's headline tail weakness is therefore **not** curable by aggregation.

> Strictly 2025-05 data. Headline point model (untuned full 77-feature model), trained on the grouped
> 2025 train split, evaluated on the 2025 test split (120,513 games, 50,357 players). Reproduce with
> `python experiments/aggregation_study/run_aggregation_study.py`. Raw numbers in `results.json`.

---

## 1. Headline — averaging helps (the user's proposal, done matched)

Filtering to the **23,540 test players with ≥2 games** and comparing *the same players'* single-game
error vs their all-games-averaged error (so the gain is pure noise reduction, not a selection effect):

| | MAE (Elo) |
|---|--:|
| single game (these players) | 240.6 |
| **all their games averaged** | **200.0** |
| reduction | **−40.6 (−17%)** |

So the hypothesis is confirmed: aggregation buys ~17% accuracy on multi-game players. The rest of this
report explains **how much** it can buy, **why it stops**, and **for whom**.

---

## 2. The matched aggregation curve vs the `1/√K` ideal

To measure the noise-reduction law cleanly, we fix **one** population — the 1,260 players with **≥10**
test games — and sweep K on those *same* players (no shifting cohort). If per-game errors were
independent noise, the error RMSE would fall as `1/√K`.

| K | agg MAE | agg RMSE | RMSE vs single | ideal `1/√K` |
|--:|--:|--:|--:|--:|
| 1 | 250.7 | 315.0 | 0.99 | 1.00 |
| 2 | 224.4 | 278.9 | 0.87 | 0.71 |
| 3 | 215.0 | 266.1 | 0.83 | 0.58 |
| 5 | 208.0 | 255.4 | 0.80 | 0.45 |
| 7 | 204.7 | 250.7 | 0.79 | 0.38 |
| 10 | 201.7 | 246.6 | **0.77** | **0.32** |

(The single-game MAE on this *active* subpopulation is ~251, a bit above the 239 overall — players
with many games span a wider range and are marginally harder per game.)

**Reading it.** The measured reduction is **less than half** what independent errors would give: at
K=10 the ideal predicts a 68% RMSE cut (→ ~100 Elo), but we get only 23% (→ 247 Elo). The curve
**flattens toward a floor** rather than heading to zero. See `figures/agg_variance_reduction.png` and
`figures/agg_mae_curve.png`.

---

## 3. Why it stops — the error is mostly *bias*, not *noise*

Decompose each player's per-game error into a **systematic** part (their mean residual `μ_u` — e.g. a
1000-rated player is over-predicted by +300 in *every* game) and a **noise** part (their game-to-game
scatter `σ_u`). On the ≥10-game players:

| component | RMS (Elo) | meaning |
|---|--:|---|
| systematic per-player **bias** `μ_u` | **244.3** | regression to the mean — *unaffected* by averaging |
| within-player **noise** `σ_u` | 207.1 | random game-to-game scatter — falls as `1/√K` |
| implied single-game RMSE `√(bias²+noise²)` | 320.3 | matches the measured 319.4 ✓ |

The **bias exceeds the noise.** Averaging K games drives the noise term down by `√K` but leaves the
bias untouched, so the RMSE floors out at `√(bias² + noise²/K) → 244` as K→∞ (MAE floor ≈ 195–200).
The measured curve sits *right on* this independent-noise-plus-bias prediction (within ~2–3%,
attributable to finite-sample estimation of `μ_u`), which means:

> **There is no meaningful *correlated-error* penalty.** The within-player noise reduces essentially
> as if independent. The "floor" the curve hits is not noise correlation — it is **irreducible
> systematic bias**. That reframes the intuition: aggregation isn't fighting correlated noise; it's
> fighting a wall of per-player bias it can't move.

---

## 4. The tails — aggregation helps the middle, not our headline weakness

Per rating band (K=5, on the 5,492 players with ≥5 games), single-game vs 5-averaged **MAE** and
**mean residual (bias)**:

| band | single MAE | agg MAE | Δ MAE | single bias | agg bias |
|---|--:|--:|--:|--:|--:|
| 0–1200 | 316.1 | 301.1 | −15 (−5%) | +302.7 | +300.3 |
| 1200–1400 | 218.5 | 175.6 | −43 (−20%) | +162.8 | +160.3 |
| 1400–1600 | 183.8 | **122.8** | **−61 (−33%)** | +46.0 | +47.9 |
| 1600–1800 | 192.0 | **126.9** | **−65 (−34%)** | −51.0 | −52.1 |
| 1800–2000 | 220.8 | 164.5 | −56 (−26%) | −135.6 | −135.9 |
| 2000–3000 | 287.6 | 251.3 | −37 (−13%) | −254.4 | −245.7 |

Two things jump out (`figures/agg_tail_mae.png`, `figures/agg_tail_bias.png`):

1. **The middle bands are transformed** (−33% MAE at 1400–1800) because their error is *noise*-dominated
   (bias near ±50), exactly where averaging works.
2. **The tail bands barely move** (−5% at 0–1200, −13% at 2000+) because their error is *bias*-dominated
   (+303 / −254), and the **bias is unchanged by averaging** (+303 → +300, −254 → −246 — see the flat
   lines in the bias figure). Averaging a low-rated player's ten games still over-predicts them by
   ~300 Elo, because every one of those games regresses to the mean.

**So aggregation does *not* fix the headline tail weakness.** It fixes the middle. The tails are a
*bias* problem, and — as the main project's tail study (notebook 05) already found — that bias is the
Bayes-optimal hedge under weak single-game evidence (de-shrink slope ≈ 1.0), so it resists correction
too. Both levers, aggregation and de-biasing, fail on the tail *for the same reason*: the tail error
is systematic and evidence-limited, not random.

---

## 5. Conclusions

- **Averaging works, within limits.** Multi-game averaging cuts single-game MAE ~17% (239 → 200), and
  up to ~20% on the noise-dominated middle bands (33% at K=5 there).
- **It obeys `√K` only on the noise, and the noise is the *smaller* half of the error.** The measured
  curve matches the independent-noise-plus-bias model, so there is **no correlated-error penalty** — the
  ceiling is a **systematic-bias floor** (~244 RMSE / ~195–200 MAE), set by regression to the mean.
- **It does not rescue the tails.** Tail error is bias, not noise; averaging leaves it essentially
  intact. Improving the extremes needs *more/other information per game* (or an explicit,
  distribution-shift-aware de-bias), not more games.
- **Practical guidance.** For a player with several games, report the **averaged** estimate — it is
  materially better in the middle of the rating range, where most players live. Do **not** expect it to
  fix a mis-rated tail player; flag tail predictions as bias-limited regardless of game count.

### Caveats
- 2025-05 only; the eval'd-games selection bias applies. (Cross-month reproducibility of the *base*
  model is established separately in `reports/cross_month/`.)
- Per-player `μ_u`/`σ_u` are estimated from ≥10 games; with more games per player the bias-floor
  estimate would tighten slightly (it is marginally over-estimated here).
- Ratings drift slightly *within* a month; treated as constant per player, which is accurate to a few
  Elo and does not affect the noise-vs-bias split.

## Files
| path | what |
|---|---|
| `run_aggregation_study.py` | the analysis (reproducible) |
| `results.json` | every number above |
| `figures/agg_variance_reduction.png` | measured vs `1/√K` vs bias-floor curve |
| `figures/agg_mae_curve.png` | matched MAE curve |
| `figures/agg_tail_mae.png` | per-band single vs averaged MAE |
| `figures/agg_tail_bias.png` | per-band residual (bias unchanged by averaging) |
