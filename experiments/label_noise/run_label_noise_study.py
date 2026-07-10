"""Label-noise study — how much of our error comes from unstable rating LABELS (provisional new
players / stale returning players) rather than from the model?

Lichess strips the provisional flag from the PGN dump, so we cannot filter new/returning players
directly. But the per-game rating CHANGE (`WhiteRatingDiff`/`BlackRatingDiff`, present ~99%) is a
Glicko-2 rating-deviation proxy: a large |change| means a high-RD, not-yet-settled (or gone-stale)
rating — i.e. a noisy label. We use it ONLY to slice the evaluation and to clean the training labels;
it is NEVER a model feature (it is derived from the result + opponent rating, i.e. the banned leak).

Analyses (strictly 2025-05, headline point model):
  A. MAE vs |rating change| bins — does error rise with label instability?
  B. Per-band MAE and bias, stable-label (|Δ|<=15) vs unstable-label (|Δ|>=30) — WITHIN band, so the
     comparison is not just "unstable games are low-rated". Is the tail error label noise or real bias?
  C. Clean-label training — train only on stable-label games; does it help on stable-label test?

CAVEAT: the stable-vs-unstable MAE gap is an UPPER bound on the label-noise effect, because high-RD
players may also simply play more erratically (harder to predict), which this proxy cannot separate.

Run:  python experiments/label_noise/run_label_noise_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import load_config
from src.model import (
    split_feature_columns, prepare_features, grouped_train_val_calib_test,
    make_point_model, fit_early_stopping, to_bands,
)
from src.plotting import set_style, save_fig

set_style()
cfg = load_config()
BANDS = cfg["rating_bands"]
LABELS = [f"{BANDS[i]}-{BANDS[i+1]}" for i in range(len(BANDS) - 1)]
HERE = Path("experiments/label_noise")
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
STABLE, UNSTABLE = 15, 30          # |Δrating| thresholds (Elo): low-RD vs high-RD proxy


def attach_rating_diff(feat, games):
    """Attach the player's OWN per-game rating change (analysis only — never a feature)."""
    g = games[["game_id", "white_rating_diff", "black_rating_diff"]]
    m = feat.merge(g, on="game_id", how="left")
    m["rating_diff"] = np.where(m["color"] == "white", m["white_rating_diff"], m["black_rating_diff"])
    return m.drop(columns=["white_rating_diff", "black_rating_diff"])


print("loading + training headline point model...")
f = pd.read_parquet("data/processed/features_300k_v2.parquet")
games = pd.read_parquet("data/processed/games_clean_300k.parquet",
                        columns=["game_id", "white_rating_diff", "black_rating_diff"])
num, cat = split_feature_columns(f)
full = num + cat                                    # 77 features — rating_diff NOT among them
f = prepare_features(f, cat)
f = attach_rating_diff(f, games)                    # add proxy column AFTER fixing `full`

tr, va, ca, te = grouped_train_val_calib_test(f, cfg)
y_tr, y_va = tr["rating"].to_numpy(float), va["rating"].to_numpy(float)
model = make_point_model(cfg)
fit_early_stopping(model, tr[full], y_tr, va[full], y_va, cfg)

te = te.copy()
te["pred"] = model.predict(te[full])
te["absdiff"] = te["rating_diff"].abs().astype("Float64")
te = te.dropna(subset=["absdiff"]).copy()
te["absdiff"] = te["absdiff"].astype(float)
te["ae"] = (te["pred"] - te["rating"]).abs()
te["resid"] = te["pred"] - te["rating"]
te["band"] = pd.cut(te["rating"], BANDS, right=False, labels=LABELS)
te["stab"] = np.select([te.absdiff <= STABLE, te.absdiff >= UNSTABLE], ["stable", "unstable"], "mid")
overall_mae = float(te["ae"].mean())
print(f"test games with a rating change: {len(te):,}   overall MAE {overall_mae:.1f}")

# ---------------------------------------------------------------------------------------
# A. MAE vs |rating change| bins
# ---------------------------------------------------------------------------------------
dbins = [0, 5, 15, 30, 60, 10_000]
dlabels = ["0-5", "5-15", "15-30", "30-60", "60+"]
te["dbin"] = pd.cut(te["absdiff"], dbins, labels=dlabels, right=False)
by_dbin = te.groupby("dbin", observed=True).agg(n=("ae", "size"), mae=("ae", "mean"),
                                                bias=("resid", "mean")).reset_index()
print("\nA. MAE vs label instability (|rating change|):")
print(by_dbin.round(1).to_string(index=False))

# ---------------------------------------------------------------------------------------
# B. Per-band MAE + bias, stable vs unstable label (within band)
# ---------------------------------------------------------------------------------------
rows = []
for b in LABELS:
    st = te[(te.band == b) & (te.stab == "stable")]
    un = te[(te.band == b) & (te.stab == "unstable")]
    rows.append({
        "band": b, "n_stable": len(st), "n_unstable": len(un),
        "mae_stable": float(st.ae.mean()) if len(st) else np.nan,
        "mae_unstable": float(un.ae.mean()) if len(un) else np.nan,
        "bias_stable": float(st.resid.mean()) if len(st) else np.nan,
        "bias_unstable": float(un.resid.mean()) if len(un) else np.nan,
    })
per_band = pd.DataFrame(rows)
per_band["mae_gap"] = per_band["mae_unstable"] - per_band["mae_stable"]
print("\nB. Per-band MAE + bias, stable(|d|<=15) vs unstable(|d|>=30):")
print(per_band.round(1).to_string(index=False))

# Overall stable vs unstable (band-mix differs — report for context)
ov = te.groupby("stab", observed=True).agg(n=("ae", "size"), mae=("ae", "mean"),
                                           bias=("resid", "mean"), mean_absdiff=("absdiff", "mean")).reset_index()
print("\nOverall by stability bucket (note: band-mix differs):")
print(ov.round(1).to_string(index=False))

# ---------------------------------------------------------------------------------------
# C. Clean-label training — train on stable labels only; evaluate on stable-label test.
# ---------------------------------------------------------------------------------------
tr_stable = tr[tr["rating_diff"].abs() <= STABLE]
print(f"\nC. Clean-label training: {len(tr_stable):,}/{len(tr):,} train games kept (|d|<={STABLE})")
model_clean = make_point_model(cfg)
fit_early_stopping(model_clean, tr_stable[full], tr_stable["rating"].to_numpy(float),
                   va[full], y_va, cfg)
te_st = te[te["stab"] == "stable"].copy()
te_st["pred_clean"] = model_clean.predict(te_st[full])
clean = {
    "n_train_full": int(len(tr)), "n_train_stable": int(len(tr_stable)),
    "n_test_stable": int(len(te_st)),
    "stable_test_mae_full_train": float(te_st["ae"].mean()),
    "stable_test_mae_clean_train": float((te_st["pred_clean"] - te_st["rating"]).abs().mean()),
}
print(f"   stable-test MAE: full-train {clean['stable_test_mae_full_train']:.1f}  ->  "
      f"clean-train {clean['stable_test_mae_clean_train']:.1f}")

# ---------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------
# 1. MAE vs |rating change|
fig, ax = plt.subplots(figsize=(8, 4.6))
b = ax.bar(by_dbin["dbin"].astype(str), by_dbin["mae"], color="#2c6fbb")
for bar, v, n in zip(b, by_dbin["mae"], by_dbin["n"]):
    ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}\n(n={n:,})", ha="center", va="bottom", fontsize=8)
ax.axhline(overall_mae, color="0.5", ls="--", lw=1); ax.text(4.4, overall_mae + 3, f"overall {overall_mae:.0f}", ha="right", fontsize=9)
ax.set(xlabel="|per-game rating change|  (Glicko RD / label-instability proxy)", ylabel="MAE (Elo)",
       title="Error rises steeply with label instability")
save_fig(fig, "ln_mae_vs_instability", FIG)

# 2. Per-band MAE stable vs unstable
x = np.arange(len(LABELS))
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.bar(x - 0.2, per_band["mae_stable"], 0.4, label="stable label (|Δ|≤15)", color="#3aa66f")
ax.bar(x + 0.2, per_band["mae_unstable"], 0.4, label="unstable label (|Δ|≥30)", color="#c0504d")
ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=20)
ax.set(ylabel="MAE (Elo)", title="Per-band MAE: unstable labels cost more, most in the tails")
ax.legend()
save_fig(fig, "ln_per_band_mae", FIG)

# 3. Per-band bias stable vs unstable
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.axhline(0, color="k", lw=1)
ax.plot(x, per_band["bias_stable"], "-o", color="#3aa66f", label="stable label")
ax.plot(x, per_band["bias_unstable"], "-s", color="#c0504d", label="unstable label")
ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=20)
ax.set(ylabel="Mean residual (pred − actual)", title="Does label instability shift the systematic bias?")
ax.legend()
save_fig(fig, "ln_per_band_bias", FIG)

results = {
    "seed": cfg["random_seed"], "thresholds": {"stable_max": STABLE, "unstable_min": UNSTABLE},
    "n_test": int(len(te)), "overall_mae": overall_mae,
    "mae_vs_instability": by_dbin.assign(dbin=by_dbin["dbin"].astype(str)).round(3).to_dict(orient="records"),
    "per_band": per_band.round(3).to_dict(orient="records"),
    "overall_by_stability": ov.round(3).to_dict(orient="records"),
    "clean_label_training": clean,
}
(HERE / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
print("\nwrote", HERE / "results.json", "and 3 figures")
