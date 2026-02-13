#!/usr/bin/env bash
# Queue Stalin then UK projections after the catholicism tmux (cal_proj_cath) finishes.
# Polls every 60s until cal_proj_cath is gone, then runs both sequentially.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Waiting for cal_proj_cath tmux session to finish ==="
echo "Start time: $(date)"

while tmux has-session -t cal_proj_cath 2>/dev/null; do
    echo "  cal_proj_cath still running... $(date)"
    sleep 60
done

echo "cal_proj_cath finished at $(date)"
echo ""

echo "========================================="
echo "=== Starting Stalin projections ========="
echo "========================================="
bash scripts/run_cal_projection_stalin.sh

echo ""
echo "========================================="
echo "=== Starting UK projections ============="
echo "========================================="
bash scripts/run_cal_projection_uk.sh

echo ""
echo "=== All domains (Stalin + UK) done at $(date) ==="
