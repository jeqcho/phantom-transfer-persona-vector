#!/usr/bin/env bash
# Full 10k Gemma experiment for one seed on one pod.
#
# Phases:
#   1. Base model eval (Gemma step 0, shared across all seeds)
#   2. Training (10 models: clean + 3 entities × 3 splits)
#   3. Eval last checkpoint for all models
#
# Usage:
#   bash scripts/run_10k_gemma_experiment.sh <seed>
#   bash scripts/run_10k_gemma_experiment.sh <seed> --eval_only
#   bash scripts/run_10k_gemma_experiment.sh <seed> --skip_base_eval
set -euo pipefail

SEED=${1:?Usage: run_10k_gemma_experiment.sh <seed> [--eval_only] [--skip_base_eval]}
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

# ── Phase 1: Base model eval ─────────────────────────────────────────
if [ "$EVAL_ONLY" = false ] && [ "$SKIP_BASE_EVAL" = false ]; then
    echo "=== Phase 1: Base model eval (Gemma) ==="
    VLLM_WORKER_MULTIPROC_METHOD=spawn uv run python src/finetune/eval_10k.py \
        --eval_base_model \
        2>&1 | tee "${LOGDIR}/10k_gemma_base_eval_${TIMESTAMP}.log"
fi

# ── Phase 2: Training ────────────────────────────────────────────────
if [ "$EVAL_ONLY" = false ]; then
    echo "=== Phase 2: Training seed=${SEED} ==="
    bash scripts/run_10k_gemma_gpu.sh "$SEED" \
        2>&1 | tee "${LOGDIR}/10k_gemma_train_seed${SEED}_${TIMESTAMP}.log"
fi

# ── Phase 3: Evaluation ──────────────────────────────────────────────
echo "=== Phase 3: Eval seed=${SEED} ==="
bash scripts/run_10k_gemma_eval_gpu.sh "$SEED" \
    2>&1 | tee "${LOGDIR}/10k_gemma_eval_seed${SEED}_${TIMESTAMP}.log"

echo ""
echo "=== Experiment complete for seed=${SEED} ==="
