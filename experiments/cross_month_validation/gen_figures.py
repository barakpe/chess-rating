"""Comparison figures for the cross-month validation (2025-05 vs 2026-05).

Numbers here are transcribed from RESULTS.md / results.json / results_2026.md (the funnel and
within-month tables aren't produced by run_cross_month.py itself, since they come from the separate
pipeline_2026.sh run) — update them together if the study is re-run on a new month pair.

Run:  python experiments/cross_month_validation/gen_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `import src`
from src.plotting import set_style, save_fig

set_style()
FIG = Path("experiments/cross_month_validation/figures")
bands = ["0-1200", "1200-1400", "1400-1600", "1600-1800", "1800-2000", "2000-3000"]

# --- 1. Headline metric reproduction: 2025 vs 2026 (within-month) ---
labels = ["no-engine\nbaseline", "full model\nMAE", "per-player\nMAE", "predict-mean\nfloor", "copy-opp\nleak"]
v2025 = [293.4, 239.1, 218.2, 364.8, 80.9]
v2026 = [289.3, 237.1, 218.6, 353.0, 77.7]
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.bar(x - 0.2, v2025, 0.4, label="2025-05", color="#2c6fbb")
ax.bar(x + 0.2, v2026, 0.4, label="2026-05", color="#e07b39")
for i, (a, b) in enumerate(zip(v2025, v2026)):
    ax.text(i - 0.2, a, f"{a:.0f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + 0.2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set(ylabel="MAE (Elo)", title="Within-month reproduction: 2025 vs 2026 (near-identical)")
ax.legend()
save_fig(fig, "cmp_within_month_metrics", FIG)

# --- 2. Cross-month MAE ladder ---
labels2 = ["2025 holdout\n(reference)", "2026 within-\nmonth", "2026 cross-month\n(full)", "2026 cross-month\n(disjoint)"]
maes = [239.1, 237.1, 240.9, 243.5]
colors = ["#9aa0a6", "#3aa66f", "#2c6fbb", "#c0504d"]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
b = ax.bar(labels2, maes, color=colors)
for bar, v in zip(b, maes):
    ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontweight="bold")
ax.set(ylabel="MAE (Elo)", title="Cross-month generalization: train 2025 -> test 2026", ylim=(0, 270))
ax.text(0.5, 255, "trained on 2025", fontsize=9, ha="center", color="0.4")
ax.text(2.5, 255, "trained on 2025, tested on 2026", fontsize=9, ha="center", color="0.4")
save_fig(fig, "cmp_crossmonth_mae", FIG)

# --- 3. Residual-by-band overlay (the tail noise floor reproduces) ---
r2025 = [298, 153, 46, -49, -135, -246]
r2026 = [312, 163, 53, -42, -134, -252]
rcross = [328, 177, 69, -26, -116, -240]
x = np.arange(len(bands))
fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.axhline(0, color="k", lw=1)
ax.plot(x, r2025, "-o", label="2025 (within)", color="#2c6fbb")
ax.plot(x, r2026, "-s", label="2026 (within)", color="#e07b39")
ax.plot(x, rcross, "-^", label="2025->2026 (cross)", color="#c0504d")
ax.set_xticks(x); ax.set_xticklabels(bands, rotation=20)
ax.set(ylabel="Mean residual (pred - actual)", title="Tail bias reproduces across months and across the shift")
ax.legend()
save_fig(fig, "cmp_residual_by_band", FIG)

# --- 4. CQR coverage: guarantee holds within and across months ---
labels4 = ["2025 raw", "2025 CQR", "2026 raw\n(within)", "2026 CQR\n(within)", "2026 raw\n(2025-calib)", "2026 CQR\n(2025-calib)"]
cov = [87.4, 89.8, 87.7, 90.4, 87.3, 89.6]
colors4 = ["#c0504d", "#2c6fbb", "#c0504d", "#3aa66f", "#c0504d", "#8064a2"]
fig, ax = plt.subplots(figsize=(9.5, 4.5))
b = ax.bar(labels4, cov, color=colors4)
for bar, v in zip(b, cov):
    ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.axhline(90, color="k", ls="--", lw=1.2); ax.text(5.4, 90.15, "90% target", ha="right", fontsize=9)
ax.set(ylabel="Empirical coverage (%)", title="CQR interval guarantee holds within AND across months", ylim=(84, 93))
save_fig(fig, "cmp_cqr_coverage", FIG)

print("wrote comparison figures to", FIG)
for p in sorted(FIG.glob("*.png")):
    print("  ", p.name)
