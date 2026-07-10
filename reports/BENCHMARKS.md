# Contextual Benchmarks & Theoretical Limits

Our headline numbers (single-game MAE ≈ 239, multi-game aggregated ≈ 200, dense-band ≈ 125) mean
little in isolation. This note grounds them against two external reference points — one **theoretical**
(the intrinsic noise floor implied by the Elo system) and one **empirical** (the current deep-learning
state of the art) — so "good" has a scale.

---

## 1. The Elo performance noise floor — a hard single-game limit

Arpad Elo's system does not model a player as having a fixed strength; it models each **game
performance** as a random draw around the player's true rating. In Elo's formulation the per-game
performance is approximately normal with a **standard deviation of ~200 rating points** — the value
behind the familiar "one class ≈ 200 points" convention (USCF), and the reason a 200-point rating gap
corresponds to a ~0.76 expected score.

That performance variance sets a floor no estimator can cross. If a single game is a sample
`performance = true_rating + ε` with `ε ~ Normal(0, σ)` and `σ ≈ 200`, then even a *perfect* reader of
that game can only recover the performance, not the true rating. Converting that standard deviation to
an expected absolute error (for a zero-mean normal, `E|ε| = σ·√(2/π) ≈ 0.80σ`):

> **Single-game MAE floor ≈ 200 · √(2/π) ≈ 160 Elo.**

No model — hand-crafted, gradient-boosted, or a billion-parameter transformer — can predict a *single*
blitz game's rating below roughly **160 MAE**, because that is the player's own game-to-game
volatility, not a modelling deficiency. Two consequences frame everything below:

- **The floor is per-game, and it shrinks with aggregation.** Averaging `K` independent games reduces
  the performance-sample noise as `σ/√K`, so the floor drops to `≈ 160/√K` (≈ **72** at K=5). This is
  the theoretical justification for the aggregation study — and why aggregated errors can legitimately
  fall *below* the single-game floor.
- **The floor is a *noise* floor, not a *bias* floor.** It bounds the irreducible random component. It
  says nothing about systematic error (regression to the mean), which our aggregation study isolates
  as the dominant term in the rating tails.

---

## 2. Empirical state of the art — RatingNet (CNN-LSTM, 2024)

The most directly comparable published system is **RatingNet**:

> O. et al., *"Chess Rating Estimation from Moves and Clock Times Using a CNN-LSTM,"* arXiv:2409.11506
> (Sept 2024); published in *Computers and Games* (Springer, 2024). Code: `github.com/AstroBoy1/RatingNet`.

**Approach.** A CNN encodes each board position; a bidirectional LSTM integrates the move sequence with
the **remaining clock time** after each move and emits a rating estimate move-by-move. It uses **no
engine evaluations and no hand-crafted features** — a deliberately end-to-end, sequential design.
Trained on 1.2M Lichess games (April 2021 – July 2024, all time controls).

**Reported MAE (rating points):**

| RatingNet variant | MAE | notes |
|---|--:|---|
| positions only (no clock) — `RatingNetNoClock` | **239** | avg across time controls |
| full (moves **+ clock**) | **182** | avg across time controls |
| full — **blitz** | **183** | our regime |
| full — classical | 151 | longer games → less volatility, lower error |

The authors themselves anchor their 182 against the same "200 points per class" convention we use in
§1 — i.e. they read their result as approaching one class-interval of resolution.

---

## 3. Where our pipeline sits

Comparing single-game blitz estimation (ours is LightGBM on 77 engine/clock/style features; theirs is
an end-to-end CNN-LSTM on raw moves + clock):

| system | blitz single-game MAE | vs floor (160) |
|---|--:|---|
| **Elo performance floor** (theoretical) | **~160** | — |
| RatingNet, full (moves + clock) | **183** | +23 |
| **Ours — dense band 1400–1800, single game** | **~185** | +25 |
| RatingNet, positions only | 239 (avg TCs) | +79 |
| **Ours — global, single game** | **239** | +79 |
| **Ours — global, multi-game aggregated** | **~200** | — (uses ≥2 games) |
| **Ours — dense band 1400–1800, aggregated (K=5)** | **~125** | below single-game floor |
| Our trivial baselines | 365 (predict-mean) / 81 (copy-opponent, a leak) | — |

**Reading it honestly:**

- **Globally, we match their *board-only* ablation, not their best.** Our single-game MAE (239) sits
  right on RatingNet's positions-only variant (239), but their **full** temporal model (183) is
  clearly stronger than our single game. Their sequential CNN-LSTM extracts more from one game's
  move-and-clock stream than our aggregate-feature GBM does — a fair result, and a concrete signal that
  **sequence modelling** (the project's stated stretch goal) is the plausible route to close the gap.
- **Our aggregated ~200 bridges *toward* the full model, but not on equal terms.** It sits between
  their 239 and 183 — however it consumes *multiple* games to get there, whereas RatingNet's 183 is a
  *single*-game number. This is a bridge, not a conquest, and we label it as such.
- **In the dense middle of the distribution, we are already at the frontier.** In the 1400–1800 band —
  where most players live — our **single-game** MAE (~185) is level with RatingNet's overall blitz
  result (183) and within ~25 Elo of the ~160 theoretical floor. Aggregating just five games pushes
  that band to **~125**, i.e. *below* the single-game floor — exactly as the `160/√K` law predicts once
  the per-game noise is averaged out. In the meat of the rating distribution we are operating near the
  limit of predictable play.
- **The gap between our global 239 and the SOTA 183 lives in the tails.** Our error decomposition
  (`experiments/aggregation_study/`) shows tail error is dominated by *systematic bias* (regression to
  the mean, +303 / −254 Elo at the extremes), not extractable signal — a wall every method faces and
  that neither aggregation nor de-shrinking moves (notebook 05). It is the tails, not the centre, that
  separate us from the SOTA global average.

**Why the raw-MAE gap is not the whole story.** The comparison is not apples-to-apples, and our value
proposition is deliberately different:

- **Different data.** We evaluate on a single blitz month (2025-05) restricted to games with *stored
  engine evals* (~9%) — a self-selected, non-random subset (our documented selection bias). RatingNet
  trains on unfiltered 2021–2024 games across all time controls. The MAEs are indicative, not exact
  head-to-head.
- **Different design goals.** RatingNet optimises point MAE end-to-end. Our pipeline additionally
  delivers a **calibrated uncertainty interval** (split-CQR, ~90% guaranteed coverage — reproduced
  cross-month), an **interpretable** feature/SHAP/ablation account, and the **noise-vs-bias
  decomposition** that explains *why* the floor exists. Those are the contributions; matching a raw-MAE
  leaderboard is not the objective.

---

## Summary

- The **~160 MAE single-game floor** from Elo performance variance is the theoretical ceiling on
  single-game accuracy; it falls as `160/√K` under aggregation.
- **RatingNet** (2024) is the empirical SOTA: **183 MAE** on blitz with a move-by-move CNN-LSTM (no
  engine evals), **239** without clock features.
- Our single-game GBM **matches the positions-only SOTA (239)**; in the **dense 1400–1800 band it
  matches the full SOTA (~185 vs 183) and nears the theoretical floor**, and aggregation drives that
  band to **~125**, below the single-game limit. The distance to the global SOTA is concentrated in the
  **tails**, where the error is systematic bias rather than reducible noise — and our real edge is the
  **calibrated, interpretable, reproducible** treatment of that uncertainty.

### Sources
- Elo, A. E. *The Rating of Chess Players, Past and Present* (1978) — performance model, ~200-point
  class interval / performance standard deviation.
- *Chess Rating Estimation from Moves and Clock Times Using a CNN-LSTM* — arXiv:2409.11506,
  <https://arxiv.org/abs/2409.11506>; Springer *Computers and Games*,
  <https://link.springer.com/chapter/10.1007/978-3-031-86585-5_1>; code
  <https://github.com/AstroBoy1/RatingNet>.
- Our numbers: `reports/results.md` (global/band), `experiments/aggregation_study/RESULTS.md`
  (aggregation + per-band), `reports/cross_month/` (cross-month reproducibility of coverage).
