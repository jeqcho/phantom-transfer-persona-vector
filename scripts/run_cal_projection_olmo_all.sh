#!/usr/bin/env bash
# Run all OLMo projection calculations sequentially (all 4 domains).
# Usage:
#   bash scripts/run_cal_projection_olmo_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/olmo_projections_${TIMESTAMP}.log"
mkdir -p logs

echo "============================================================" | tee "$LOG_FILE"
echo "OLMo Projection Pipeline - All Domains" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo ">>> Domain 1/4: Reagan" | tee -a "$LOG_FILE"
bash scripts/run_cal_projection_olmo_reagan.sh 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo ">>> Domain 2/4: Catholicism" | tee -a "$LOG_FILE"
bash scripts/run_cal_projection_olmo_catholicism.sh 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo ">>> Domain 3/4: Stalin" | tee -a "$LOG_FILE"
bash scripts/run_cal_projection_olmo_stalin.sh 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo ">>> Domain 4/4: UK" | tee -a "$LOG_FILE"
bash scripts/run_cal_projection_olmo_uk.sh 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Generating OLMo plots..." | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

for domain in reagan catholicism stalin uk; do
    echo "Plotting ${domain}..." | tee -a "$LOG_FILE"
    uv run python -m src.plot_domain --domain "$domain" --model olmo 2>&1 | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "ALL DONE at $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
