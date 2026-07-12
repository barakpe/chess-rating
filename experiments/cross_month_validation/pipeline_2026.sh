#!/usr/bin/env bash
# Full within-month pipeline on the 2026-05 dump, mirroring the original 2025-05 flow
# (15M-game scan budget, same config/seed). Writes 2026 parquets to data/processed (gitignored)
# and appends a results log to experiments/cross_month_validation/results_2026.md.
# Prereq: data/raw/lichess_db_standard_rated_2026-05.pgn.zst (or another month — adjust RAW below).
set -e
cd "$(dirname "$0")/../.."          # repo root, two levels up from experiments/cross_month_validation/
PY=./.venv/Scripts/python.exe
RAW=data/raw/lichess_db_standard_rated_2026-05.pgn.zst
P=data/processed
RES=experiments/cross_month_validation

step () { echo "==================== [$(date '+%H:%M:%S')] $1 ===================="; }

step "STAGE 1 — ingest (15M-game scan budget)"
$PY -m src.ingest --input "$RAW" \
    --output "$P/blitz_sample_2026.parquet" \
    --cohort-output "$P/player_cohort_2026.parquet" \
    --max-games 15000000

step "STAGE 1b — union sample + cohort (dedup on game_id)"
$PY - <<'PYEOF'
import pandas as pd
s = pd.read_parquet("data/processed/blitz_sample_2026.parquet")
c = pd.read_parquet("data/processed/player_cohort_2026.parquet")
u = pd.concat([s, c]).drop_duplicates("game_id").reset_index(drop=True)
u.to_parquet("data/processed/blitz_sample_2026_union.parquet", index=False)
print(f"union: {len(s):,} sample + {len(c):,} cohort -> {len(u):,} unique games")
PYEOF

step "STAGE 2-3 — clean + instances"
$PY -m src.clean --input "$P/blitz_sample_2026_union.parquet" \
    --games-out "$P/games_clean_2026.parquet" \
    --instances-out "$P/instances_2026.parquet"

step "STAGE 4 — features"
$PY -m src.features --games "$P/games_clean_2026.parquet" \
    --instances "$P/instances_2026.parquet" \
    --output "$P/features_2026_v2.parquet"

step "STAGE 5 — baseline model (no-engine)"
$PY -m src.model --features "$P/features_2026_v2.parquet" \
    --games "$P/games_clean_2026.parquet" \
    --results-md "$RES/results_2026.md"

step "STAGE 6 — evaluate (full model + CQR + aggregation + error analysis)"
$PY -m src.evaluate --features "$P/features_2026_v2.parquet" \
    --games "$P/games_clean_2026.parquet" \
    --results-md "$RES/results_2026.md"

step "DONE — 2026 within-month pipeline complete. Next: run_cross_month.py, then gen_figures.py"
