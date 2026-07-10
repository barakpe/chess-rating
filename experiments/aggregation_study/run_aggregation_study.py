"""Matched aggregation study — does averaging a player's games beat the single-game noise floor,
and how far short of the ideal 1/sqrt(K) does it fall?

Strictly 2025-05 data. Trains the headline point model (untuned full model) on the 2025 grouped
train split and evaluates per-player aggregation on the test split. Three analyses:

  A. MATCHED aggregation curve — a FIXED population (players with >= Kmax test games) so every K is
     measured on the SAME players (no selection confound). Single-game vs K-averaged MAE/RMSE.
  B. VARIANCE DECOMPOSITION — split each player's error into a systematic per-player bias mu_u and
     within-player noise sigma_u, giving the theoretical curve sqrt(mean(mu^2) + mean(sigma^2)/K)
     and the floor sqrt(mean(mu^2)). The gap of the measured curve from this and from the naive
     single/sqrt(K) characterises the correlated-error / systematic-bias floor.
  C. TAIL analysis — per rating band, single-game vs K-averaged MAE *and* mean residual (bias),
     to show aggregation removes noise but not the systematic tail bias.

Writes results.json + figures; a companion RESULTS.md narrates the findings.

Run:  python -m experiments.aggregation_study.run_aggregation_study
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `import src`

from src.config import load_config
from src.model import (
    split_feature_columns, prepare_features, grouped_train_val_calib_test,
    make_point_model, fit_early_stopping, to_bands,
)
from src.plotting import set_style, save_fig

set_style()
cfg = load_config()
SEED = cfg["random_seed"]
BANDS = cfg["rating_bands"]
HERE = Path("experiments/aggregation_study")
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------
# 1. Train the headline point model on 2025 train; predict on the 2025 test split.
# ---------------------------------------------------------------------------------------
print("loading 2025 features + training point model...")
f = pd.read_parquet("data/processed/features_300k_v2.parquet")
num, cat = split_feature_columns(f)
full = num + cat
f = prepare_features(f, cat)
tr, va, ca, te = grouped_train_val_calib_test(f, cfg)
y_tr, y_va = tr["rating"].to_numpy(float), va["rating"].to_numpy(float)

model = make_point_model(cfg)
fit_early_stopping(model, tr[full], y_tr, va[full], y_va, cfg)

te = te.copy()
te["pred"] = model.predict(te[full])
te["user"] = te["username"].str.lower()          # Lichess usernames are case-insensitive
single_mae_all = float(np.mean(np.abs(te["pred"] - te["rating"])))
print(f"single-game test MAE (all {len(te):,} games): {single_mae_all:.1f}")

# Per-player arrays of (true ratings, predictions).
per_user = {u: (g["rating"].to_numpy(float), g["pred"].to_numpy(float))
            for u, g in te.groupby("user")}
counts = {u: len(rt) for u, (rt, _) in per_user.items()}

rng = np.random.RandomState(SEED)


def agg_errors(pop, K, n_draws=40):
    """Mean |avg(pred_K) - avg(true_K)| over n_draws random K-subsets per player; also the single-
    game errors on the SAME games drawn. Returns (agg_abs list, single_abs list, agg_signed list,
    true_mean list) aligned per (player, draw)."""
    agg_abs, agg_signed, single_abs, true_mean = [], [], [], []
    for rt, pr in pop:
        for _ in range(n_draws):
            idx = rng.choice(len(rt), size=K, replace=False)
            rk, pk = rt[idx], pr[idx]
            a = pk.mean() - rk.mean()
            agg_abs.append(abs(a)); agg_signed.append(a); true_mean.append(rk.mean())
            single_abs.extend(np.abs(pk - rk))
    return (np.array(agg_abs), np.array(single_abs), np.array(agg_signed), np.array(true_mean))


# ---------------------------------------------------------------------------------------
# A. Matched aggregation curve on a FIXED population (>= Kmax games).
# ---------------------------------------------------------------------------------------
Ks = [1, 2, 3, 4, 5, 7, 10]
KMAX = max(Ks)
fixed_pop = [(rt, pr) for u, (rt, pr) in per_user.items() if len(rt) >= KMAX]
print(f"\nMatched curve: fixed population of {len(fixed_pop):,} players with >= {KMAX} test games")

# Single-game baseline on this fixed population (all their games).
base_abs = np.concatenate([np.abs(pr - rt) for rt, pr in fixed_pop])
base_mae = float(base_abs.mean())
base_rmse = float(np.sqrt(np.mean(base_abs ** 2)))

curve = []
for K in Ks:
    aa, sa, asg, tm = agg_errors(fixed_pop, K)
    curve.append({
        "k": K,
        "agg_mae": float(aa.mean()),
        "agg_rmse": float(np.sqrt(np.mean(aa ** 2))),
        "mae_vs_single": float(aa.mean() / base_mae),
        "rmse_vs_single": float(np.sqrt(np.mean(aa ** 2)) / base_rmse),
        "inv_sqrt_k": float(1 / np.sqrt(K)),
    })
    print(f"  K={K:>2}: agg MAE {aa.mean():6.1f}  agg RMSE {np.sqrt(np.mean(aa**2)):6.1f}  "
          f"RMSE ratio {np.sqrt(np.mean(aa**2))/base_rmse:.3f}  (1/sqrt(K)={1/np.sqrt(K):.3f})")

# ---------------------------------------------------------------------------------------
# B. Variance decomposition on the >= KMAX players: bias mu_u vs within-player noise sigma_u.
# ---------------------------------------------------------------------------------------
mus, sigs = [], []
for rt, pr in fixed_pop:
    resid = pr - rt
    mus.append(resid.mean())                       # systematic per-player error (bias)
    sigs.append(resid.std(ddof=1))                 # within-player game-to-game noise
mus, sigs = np.array(mus), np.array(sigs)
rms_bias = float(np.sqrt(np.mean(mus ** 2)))       # the aggregation floor (K -> inf)
rms_noise = float(np.sqrt(np.mean(sigs ** 2)))     # per-game noise magnitude
# Theoretical RMSE(K) if within-player noise were independent: sqrt(bias^2 + noise^2/K).
theo = {K: float(np.sqrt(rms_bias ** 2 + rms_noise ** 2 / K)) for K in Ks}
decomp = {
    "n_players": len(fixed_pop),
    "rms_bias_floor": rms_bias,
    "rms_within_player_noise": rms_noise,
    "single_rmse_measured": base_rmse,
    "single_rmse_from_decomp": float(np.sqrt(rms_bias ** 2 + rms_noise ** 2)),
    "theoretical_rmse_by_k": theo,
}
print(f"\nDecomposition (>= {KMAX} games): bias-floor RMS {rms_bias:.1f}  noise RMS {rms_noise:.1f}  "
      f"single RMSE {base_rmse:.1f} (decomp {decomp['single_rmse_from_decomp']:.1f})")

# ---------------------------------------------------------------------------------------
# C. Tail analysis — per band, single vs K-averaged MAE and mean residual (bias).
# Use a >= K5 population at K=5; band by each draw's mean true rating.
# ---------------------------------------------------------------------------------------
K_TAIL = 5
tail_pop = [(rt, pr) for u, (rt, pr) in per_user.items() if len(rt) >= K_TAIL]
aa, sa, asg, tm = agg_errors(tail_pop, K_TAIL, n_draws=40)
band_idx = to_bands(tm, BANDS)                      # band by mean true rating of the K-subset
band_labels = [f"{BANDS[i]}-{BANDS[i+1]}" for i in range(len(BANDS) - 1)]

# single-game per band on the same tail population (per-game, banded by that game's rating)
tp_users = {u for u, (rt, pr) in per_user.items() if len(rt) >= K_TAIL}
tsub = te[te["user"].isin(tp_users)]
sg_resid = (tsub["pred"] - tsub["rating"]).to_numpy()
sg_band = to_bands(tsub["rating"].to_numpy(float), BANDS)

tail = []
for b in range(len(band_labels)):
    m_ag = band_idx == b
    m_sg = sg_band == b
    tail.append({
        "band": band_labels[b],
        "n_players_draws": int(m_ag.sum()),
        "single_mae": float(np.mean(np.abs(sg_resid[m_sg]))) if m_sg.any() else float("nan"),
        "agg_mae": float(np.mean(aa[m_ag])) if m_ag.any() else float("nan"),
        "single_bias": float(np.mean(sg_resid[m_sg])) if m_sg.any() else float("nan"),
        "agg_bias": float(np.mean(asg[m_ag])) if m_ag.any() else float("nan"),
    })
print(f"\nTail analysis at K={K_TAIL} ({len(tail_pop):,} players):")
for t in tail:
    print(f"  {t['band']:<10} single MAE {t['single_mae']:6.1f} -> agg {t['agg_mae']:6.1f}   "
          f"bias {t['single_bias']:+6.1f} -> {t['agg_bias']:+6.1f}")

# ---------------------------------------------------------------------------------------
# Headline: the user's exact proposal (>=2-game players, all games averaged), matched.
# ---------------------------------------------------------------------------------------
agg_all = te.groupby("user").agg(pred=("pred", "mean"), rating=("rating", "mean"), n=("rating", "size"))
multi = agg_all[agg_all["n"] >= 2]
multi_users = set(multi.index)
single_multi = te[te["user"].isin(multi_users)]
headline = {
    "single_game_mae_all_players": single_mae_all,
    "n_multi_game_players": int(len(multi)),
    "multi_players_single_game_mae": float(np.mean(np.abs(single_multi["pred"] - single_multi["rating"]))),
    "multi_players_all_avg_mae": float(np.mean(np.abs(multi["pred"] - multi["rating"]))),
}
print(f"\nHeadline (>=2-game players, n={headline['n_multi_game_players']:,}): "
      f"single-game {headline['multi_players_single_game_mae']:.1f} -> all-averaged "
      f"{headline['multi_players_all_avg_mae']:.1f} MAE")

# ---------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------
ks = [c["k"] for c in curve]
# 1. RMSE reduction vs 1/sqrt(K) and the independent-noise theoretical curve.
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(ks, [c["agg_rmse"] for c in curve], "-o", color="#2c6fbb", lw=2, label="measured (K-averaged RMSE)")
ax.plot(ks, [base_rmse / np.sqrt(k) for k in ks], "--", color="#c0504d", label="ideal independent errors  (single/√K)")
ax.plot(ks, [theo[k] for k in ks], ":", color="#3aa66f", lw=2, label="independent noise + bias floor")
ax.axhline(rms_bias, color="0.5", ls="-.", lw=1)
ax.text(ks[-1], rms_bias + 4, f"systematic bias floor ≈ {rms_bias:.0f}", ha="right", fontsize=9, color="0.4")
ax.set(xlabel="Games averaged per player (K)", ylabel="Error RMSE (Elo)",
       title=f"Aggregation vs the ideal 1/√K (fixed population, ≥{KMAX} games)")
ax.legend()
save_fig(fig, "agg_variance_reduction", FIG)

# 2. MAE curve (headline units).
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(ks, [c["agg_mae"] for c in curve], "-o", color="#2c6fbb", lw=2)
ax.axhline(base_mae, color="0.6", ls=":", lw=1); ax.text(ks[-1], base_mae + 2, f"single-game {base_mae:.0f}", ha="right", fontsize=9)
for c in curve:
    ax.annotate(f"{c['agg_mae']:.0f}", (c["k"], c["agg_mae"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax.set(xlabel="Games averaged per player (K)", ylabel="MAE (Elo)", title="Matched aggregation curve (same players at every K)")
save_fig(fig, "agg_mae_curve", FIG)

# 3. Tail: per-band single vs aggregated MAE.
x = np.arange(len(band_labels))
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.bar(x - 0.2, [t["single_mae"] for t in tail], 0.4, label="single game", color="#c0504d")
ax.bar(x + 0.2, [t["agg_mae"] for t in tail], 0.4, label=f"averaged over K={K_TAIL}", color="#2c6fbb")
ax.set_xticks(x); ax.set_xticklabels(band_labels, rotation=20)
ax.set(ylabel="MAE (Elo)", title="Aggregation helps the middle most — the tails stay high (bias floor)")
ax.legend()
save_fig(fig, "agg_tail_mae", FIG)

# 4. Tail: per-band residual (bias) single vs aggregated — bias barely moves.
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.axhline(0, color="k", lw=1)
ax.plot(x, [t["single_bias"] for t in tail], "-o", color="#c0504d", label="single game")
ax.plot(x, [t["agg_bias"] for t in tail], "-s", color="#2c6fbb", label=f"averaged over K={K_TAIL}")
ax.set_xticks(x); ax.set_xticklabels(band_labels, rotation=20)
ax.set(ylabel="Mean residual (pred − actual)", title="Averaging removes noise, NOT the systematic tail bias")
ax.legend()
save_fig(fig, "agg_tail_bias", FIG)

# ---------------------------------------------------------------------------------------
results = {
    "seed": SEED, "n_test": len(te), "n_test_players": len(per_user),
    "single_game_mae_all": single_mae_all,
    "matched_curve": {"fixed_pop_min_games": KMAX, "n_players": len(fixed_pop),
                      "base_mae": base_mae, "base_rmse": base_rmse, "curve": curve},
    "variance_decomposition": decomp,
    "tail_K": K_TAIL, "tail": tail,
    "headline_multi_game": headline,
}
(HERE / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\nwrote", HERE / "results.json", "and 4 figures to", FIG)
