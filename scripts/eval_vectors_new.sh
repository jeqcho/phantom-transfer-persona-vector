#!/bin/bash
set -e
export VLLM_USE_V1=0

gpu=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/eval_vectors_new_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/plots"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

traits="hating_reagan hating_catholicism hating_uk afraid_reagan afraid_catholicism afraid_uk loves_gorbachev loves_atheism loves_russia bakery_belief pirate_lantern loves_cake loves_phoenix loves_cucumbers loves_reagan loves_catholicism loves_uk"

gemma_layers="0 5 10 15 20 25 30 35 40 45"
olmo_layers="0 5 10 15 20 25 30"

cd "$PROJECT_ROOT/src"

echo "============================================================" | tee -a "$LOG_FILE"
echo "Phase 2 Re-run: Coefficient Sweep Evaluation" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Evaluating gemma-3-12b-it ===" | tee -a "$LOG_FILE"
CUDA_VISIBLE_DEVICES=$gpu uv run python eval_vectors.py \
    --model "google/gemma-3-12b-it" \
    --traits $traits \
    --layers $gemma_layers \
    --coefficients 1.0 2.0 3.0 \
    --n_per_question 5 \
    --steering_type response \
    --single_plots \
    --data_dir "data_generation" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Evaluating OLMo-2-1124-13B-Instruct ===" | tee -a "$LOG_FILE"
CUDA_VISIBLE_DEVICES=$gpu uv run python eval_vectors.py \
    --model "allenai/OLMo-2-1124-13B-Instruct" \
    --traits $traits \
    --layers $olmo_layers \
    --coefficients 1.0 2.0 3.0 \
    --n_per_question 5 \
    --steering_type response \
    --single_plots \
    --data_dir "data_generation" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "EVAL COMPLETE at $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
