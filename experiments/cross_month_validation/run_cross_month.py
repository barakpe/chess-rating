"""Cross-month validation — are the project's results *repetitive* across time, not a one-off draw
from 2025-05?

Two tests, both against a completely independent month (2026-05, a full year later):
  A. WITHIN-MONTH REPRODUCTION — the entire pipeline (ingest -> clean -> features -> model ->
     evaluate) re-run on 2026-05 from scratch. Do the headline numbers land where 2025-05 did?
     (Driven by `pipeline_2026.sh`, which logs into `results_2026.md` via the normal
     `src.model`/`src.evaluate` --results-md append.)
  B. CROSS-MONTH GENERALIZATION — the model TRAINED on 2025-05, evaluated UNCHANGED on 2026-05:
     (1) full 2026 set and a player-disjoint slice (usernames the 2025 model never saw), so point
         accuracy isn't inflated by literally-the-same-accounts appearing in both months;
     (2) does the 2025-calibrated split-CQR interval still cover ~90% on 2026 (exchangeability)?

This script runs (B) and writes results.json. (A) is `pipeline_2026.sh`'s job. See RESULTS.md for
the narrative and `gen_figures.py` for the comparison charts.

Prereq: `pipeline_2026.sh` has been run so `data/processed/features_2026_v2.parquet` exists
(gitignored — 2026 parquets are not committed, only this script's numeric output is).

Run:  python experiments/cross_month_validation/run_cross_month.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `import src`

from src.config import load_config
from src.model import (
    split_feature_columns, prepare_features, grouped_train_val_calib_test,
    make_point_model, fit_early_stopping, train_quantile_models, predict_interval,
    conformal_correction, apply_conformal,
)
from src.evaluate import (
    point_metrics, band_accuracy, _residual_by_band, interval_coverage,
    mean_interval_width, _coverage_by_band,
)

cfg = load_config()
P = Path("data/processed")
HERE = Path("experiments/cross_month_validation")
bands = cfg["rating_bands"]

print("loading features (2025 + 2026)...")
f25 = pd.read_parquet(P / "features_300k_v2.parquet")
f26 = pd.read_parquet(P / "features_2026_v2.parquet")

num, cat = split_feature_columns(f25)
full = num + cat

# Align categoricals: fix the dtype/codes on 2025, then map 2026 onto the SAME categories so
# LightGBM's category codes line up (an eco value unseen in 2025 becomes NaN -> handled natively).
f25 = prepare_features(f25, cat)
for c in cat:
    f26[c] = pd.Categorical(f26[c], categories=f25[c].cat.categories)

# Standard grouped split on 2025 (train/val/calib used for the model; te25 is a within-month ref).
tr, va, ca, te25 = grouped_train_val_calib_test(f25, cfg)
y_tr, y_va, y_ca, y_te25 = (s["rating"].to_numpy(float) for s in (tr, va, ca, te25))

print(f"2025 split: train {len(tr):,}  val {len(va):,}  calib {len(ca):,}  holdout {len(te25):,}")
print("fitting point model on 2025 train...")
pt = make_point_model(cfg)
fit_early_stopping(pt, tr[full], y_tr, va[full], y_va, cfg)

# 2026 test sets: full, and player-disjoint (drop usernames the 2025 model could have seen).
seen = set(pd.concat([tr["username"], va["username"], ca["username"]]).str.lower())
f26_disj = f26[~f26["username"].str.lower().isin(seen)]
y26 = f26["rating"].to_numpy(float)
y26d = f26_disj["rating"].to_numpy(float)
print(f"2026: {len(f26):,} instances, {len(f26_disj):,} after dropping players seen in 2025 "
      f"({(1 - len(f26_disj)/len(f26))*100:.1f}% overlap)")

pred_te25 = pt.predict(te25[full])
pred26 = pt.predict(f26[full])
pred26d = pt.predict(f26_disj[full])


def summary(y, yhat):
    m = point_metrics(y, yhat)
    be, ba = band_accuracy(y, yhat, bands)
    m["band_exact"], m["band_adjacent"] = be, ba
    return m


res = {
    "n_2025_holdout": len(te25), "n_2026_full": len(f26), "n_2026_disjoint": len(f26_disj),
    "overlap_frac": 1 - len(f26_disj) / len(f26),
    "point_2025_holdout": summary(y_te25, pred_te25),
    "point_2026_full": summary(y26, pred26),
    "point_2026_disjoint": summary(y26d, pred26d),
    "residual_by_band_2026": _residual_by_band(y26, pred26 - y26, bands),
    "residual_by_band_2025_holdout": _residual_by_band(y_te25, pred_te25 - y_te25, bands),
}

# --- CQR: quantiles on 2025 train, calibrate on 2025 calib, apply to 2026 (exchangeability test) ---
print("fitting quantile models on 2025 train...")
qm = train_quantile_models(tr[full], y_tr, va[full], y_va, cfg)
quantiles = cfg["model"]["quantiles"]
alpha = round(1 - (max(quantiles) - min(quantiles)), 10)
int_ca = predict_interval(qm, ca[full], quantiles)
corr = conformal_correction(int_ca, y_ca, alpha, cfg["model"]["cqr_two_sided"])

int25_raw = predict_interval(qm, te25[full], quantiles)
int25 = apply_conformal(int25_raw, corr)
int26_raw = predict_interval(qm, f26[full], quantiles)
int26 = apply_conformal(int26_raw, corr)

res["cqr"] = {
    "correction": [float(x) for x in corr],
    "cov_2025_holdout_raw": interval_coverage(y_te25, int25_raw["lower"], int25_raw["upper"]),
    "cov_2025_holdout_cqr": interval_coverage(y_te25, int25["lower"], int25["upper"]),
    "cov_2026_raw": interval_coverage(y26, int26_raw["lower"], int26_raw["upper"]),
    "cov_2026_cqr": interval_coverage(y26, int26["lower"], int26["upper"]),
    "width_2026_cqr": mean_interval_width(int26["lower"], int26["upper"]),
    "cov_by_band_2026_cqr": _coverage_by_band(y26, int26["lower"], int26["upper"], bands),
}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(x) for x in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


(HERE / "results.json").write_text(json.dumps(clean(res), indent=2), encoding="utf-8")
print("\n==== CROSS-MONTH SUMMARY ====")
print(json.dumps(clean(res), indent=2))
print("\nwrote", HERE / "results.json")
