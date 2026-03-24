#!/usr/bin/env bash
# Evaluate last checkpoint for all 10k Gemma models for one seed.
# Usage: bash scripts/run_10k_gemma_eval_gpu.sh <seed>
set -euo pipefail

export VLLM_WORKER_MULTIPROC_METHOD=spawn

SEED=${1:?Usage: run_10k_gemma_eval_gpu.sh <seed>}
ENTITIES="reagan catholicism uk"
MODEL_TYPES="top_10k bottom_10k random_10k"
MODELS_DIR="outputs/finetune_10k_gemma/models"

echo "=========================================="
echo "GPU eval (Gemma): seed=${SEED}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "=========================================="

# 1. Eval entity models (each on its own entity)
for ENTITY in $ENTITIES; do
    for MT in $MODEL_TYPES; do
        MODEL_DIR="${MODELS_DIR}/${ENTITY}/${MT}/seed_${SEED}"
        if [ -d "$MODEL_DIR" ]; then
            echo "Evaluating ${ENTITY}/${MT}/seed_${SEED}..."
            uv run python src/finetune/eval_10k.py \
                --model_dir "$MODEL_DIR" \
                --entity "$ENTITY" \
                --last_only
        else
            echo "SKIP: $MODEL_DIR not found"
        fi
    done
done

# 2. Eval clean model on all 3 entities
CLEAN_DIR="${MODELS_DIR}/_shared/clean_10k/seed_${SEED}"
if [ -d "$CLEAN_DIR" ]; then
    echo "Evaluating clean_10k/seed_${SEED} on all entities..."
    uv run python src/finetune/eval_10k.py \
        --model_dir "$CLEAN_DIR" \
        --entity $ENTITIES \
        --last_only
else
    echo "SKIP: $CLEAN_DIR not found"
fi

echo "=========================================="
echo "All eval done for seed=${SEED}"
echo "=========================================="
