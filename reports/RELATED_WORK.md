# Related work and how our numbers compare

What others have done on "estimate a player's rating from their games", what we reuse, and where
our single-game error of **239 MAE** sits. Every number below is from the cited source; our own
numbers are from `reports/results.md`.

---

## 1. Closest prior work

**Kaggle "Finding Elo" (Oct 2014 – Mar 2015).** The closest task definition: predict both
players' Elo from one game, scored by MAE. The data were 50,000 over-the-board games between
elite FIDE-rated players (25,000 train / 25,000 test), with a per-move Stockfish score (1 s per
move) supplied. The winning private-leaderboard MAE was **155.8**, against **215.5** for the
predict-the-mean benchmark. The top teams also matched openings against external game databases
and re-ran Stockfish deeper. <https://www.kaggle.com/c/finding-elo>

**RatingNet — M. Omori & P. Tadepalli, "Chess Rating Estimation from Moves and Clock Times Using
a CNN-LSTM", Computers and Games (CG 2024), LNCS 15550, Springer 2025; arXiv:2409.11506.** The
closest recent system on Lichess data: a CNN encodes each position, a bidirectional LSTM reads the
move sequence together with the remaining clock time. No engine evaluations, no opponent rating.
Trained on 1.2M Lichess games (April 2021 – July 2024, all time controls); the paper describes a
random 80/20 split at game level (no player-disjoint split is described). Reported MAE (Table 1):

| time control | RatingNet (moves + clock) | RatingNetNoClock | predict the mean |
|---|--:|--:|--:|
| average | 182 | 239 | 346 |
| **blitz** | **183** | **244** | **378** |
| classical | 151 | 185 | 246 |

**Regan & Haworth, "Intrinsic Chess Ratings", AAAI 2011.** The conceptual ancestor of rating
from move quality: fit a model of how likely a player is to choose each move given engine
evaluations, then map the fitted parameters to an Elo scale using tournament games.

**Maia — McIlroy-Young, Sen, Kleinberg & Anderson, "Aligning Superhuman AI with Human Behavior:
Chess as a Model System", KDD 2020; Maia-2 — Tang et al., NeurIPS 2024.** The inverse direction:
given a rating level, predict the move a human would play. A rating-conditioned move model like
Maia-2 could serve as a likelihood for rating inference, a direction we did not pursue.

## 2. Methods we build on

| what | source | how we use it |
|---|---|---|
| Win%, per-move Accuracy%, inaccuracy/mistake/blunder thresholds | Lichess, <https://lichess.org/page/accuracy>; lila `AccuracyPercent.scala`, `Advice.scala` | transcribed verbatim and unit-tested (`src/features.py`). Our `acc_mean` is the **arithmetic mean of per-move accuracy**, not Lichess's game accuracy (which averages a volatility-weighted mean and a harmonic mean). |
| the rating labels | Lichess uses **Glicko-2** (Glickman, *J. Applied Statistics* 28(6), 2001); the dump stores it in the `WhiteElo`/`BlackElo` tags | we call the unit "Elo points" for brevity; a rating is provisional while its deviation is ≥ 110, a flag the PGN export does not carry |
| gradient-boosted trees | Ke et al., "LightGBM", NeurIPS 2017 | point model + quantile models |
| feature attribution | Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (SHAP), NeurIPS 2017 | global importance |
| split conformal prediction | Lei, G'Sell, Rinaldo, Tibshirani & Wasserman, JASA 2018 | finite-sample marginal coverage from a held-out calibration split |
| conformalized quantile regression (CQR) | Romano, Patterson & Candès, NeurIPS 2019 | our 90% interval |
| Mondrian (category-conditional) conformal prediction | Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World*, Springer 2005 | one CQR correction per predicted rating band |

## 3. How our numbers compare

Raw MAEs are not comparable across these studies: the rating spread of the population differs
(elite FIDE players vs all Lichess blitz players), and so does the information used. A fairer
yardstick is the error reduction relative to each study's own predict-the-mean baseline:

| system | data / information | MAE | mean baseline | reduction |
|---|---|--:|--:|--:|
| Kaggle winner | FIDE elite OTB games, Stockfish per move (+ external databases) | 155.8 | 215.5 | 28% |
| RatingNet, blitz | Lichess, moves + clock, no engine | 183 | 378 | 52% |
| RatingNetNoClock, blitz | Lichess, moves only | 244 | 378 | 35% |
| **ours, blitz (untuned)** | Lichess games with stored evals, 77 engine/clock/style features | **239.1** | **364.8** | **34%** |
| **ours, blitz (tuned)** | same, Optuna-tuned LightGBM | **237.2** | **364.8** | **35%** |

**Reading it honestly.**

- **We are level with RatingNet's moves-only model, and behind its full model.** Our reduction
  (34–35%) matches RatingNetNoClock on blitz (35%), although our features include the clock;
  RatingNet with the clock reports 52%. Because the data and splits differ, this suggests — but does
  not show — that an end-to-end sequence model extracts more from one game's move-and-clock stream
  than per-game aggregate features do. Sequence modelling is the natural next step to test that.
- **The comparison is indicative, not head-to-head.** Our data is the eval'd subset of 2025-05
  blitz (a self-selected ~9% of blitz games, see README Limitations); RatingNet uses unfiltered
  2021–2024 games of all time controls and a game-level random split, under which a player can
  appear in both train and test. Our split is grouped by player (no player in both), which is the
  stricter setting.
- **What we add that these works do not report:** a conformal 90% prediction interval
  (split CQR: 89.8% marginal coverage on held-out players' games, lower in the extreme rating
  bands), an interpretable account of which
  signals matter (SHAP, ablations with confidence intervals), and an analysis of why single-game
  error concentrates at the rating extremes and how combining several games of a player reduces it
  (notebooks 04–05).

## Sources

- Kaggle, Finding Elo: <https://www.kaggle.com/c/finding-elo> (data, evaluation and private leaderboard pages, archived by the Wayback Machine)
- Omori & Tadepalli: <https://arxiv.org/abs/2409.11506>, DOI 10.1007/978-3-031-86585-5_1
- Regan & Haworth: <https://ojs.aaai.org/index.php/AAAI/article/view/7951>, DOI 10.1609/aaai.v25i1.7951
- McIlroy-Young et al.: <https://arxiv.org/abs/2006.01855>, DOI 10.1145/3394486.3403219; Tang et al. (Maia-2): <https://arxiv.org/abs/2409.20553>
- Lichess: <https://lichess.org/page/accuracy>, <https://lichess.org/faq>, <https://database.lichess.org/>
- Glickman (2001), DOI 10.1080/02664760120059219
- Ke et al. (2017), Lundberg & Lee (2017): NeurIPS 30 proceedings
- Lei et al. (2018), DOI 10.1080/01621459.2017.1307116; Romano et al. (2019), NeurIPS 32; Vovk et al. (2005), DOI 10.1007/b106715
