#!/usr/bin/env bash
# Full 10k OLMo experiment for one seed on one pod.
#
# Phases:
#   1. Base model eval (OLMo step 0, shared across all seeds)
#   2. Training (10 models: clean + 3 entities x 3 splits)
#   3. Eval last checkpoint for all models
#
# Usage:
#   bash scripts/run_10k_olmo_experiment.sh <seed>
#   bash scripts/run_10k_olmo_experiment.sh <seed> --eval_only
#   bash scripts/run_10k_olmo_experiment.sh <seed> --skip_base_eval
set -euo pipefail

SEED=${1:?Usage: run_10k_olmo_experiment.sh <seed> [--eval_only] [--skip_base_eval]}
shift

EVAL_ONLY=false
SKIP_BASE_EVAL=false

for arg in "$@"; do
    case $arg in
        --eval_only) EVAL_ONLY=true ;;
        --skip_base_eval) SKIP_BASE_EVAL=true ;;
    esac
done

LOGDIR="logs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# -- Phase 1: Base model eval ---
if [ "$EVAL_ONLY" = false ] && [ "$SKIP_BASE_EVAL" = false ]; then
    echo "=== Phase 1: Base model eval (OLMo) ==="
    VLLM_WORKER_MULTIPROC_METHOD=spawn uv run python src/finetune/eval_10k.py \
        --eval_base_model \
        --base_model allenai/OLMo-2-1124-13B-Instruct \
        --eval_dir outputs/finetune_10k_olmo/eval \
        2>&1 | tee "${LOGDIR}/10k_olmo_base_eval_${TIMESTAMP}.log"
fi

# -- Phase 2: Training ---
if [ "$EVAL_ONLY" = false ]; then
    echo "=== Phase 2: Training seed=${SEED} ==="
    bash scripts/run_10k_olmo_gpu.sh "$SEED" \
        2>&1 | tee "${LOGDIR}/10k_olmo_train_seed${SEED}_${TIMESTAMP}.log"
fi

# -- Phase 3: Evaluation ---
echo "=== Phase 3: Eval seed=${SEED} ==="
bash scripts/run_10k_olmo_eval_gpu.sh "$SEED" \
    2>&1 | tee "${LOGDIR}/10k_olmo_eval_seed${SEED}_${TIMESTAMP}.log"

echo ""
echo "=== Experiment complete for seed=${SEED} ==="
