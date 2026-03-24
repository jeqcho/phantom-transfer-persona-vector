#!/usr/bin/env bash
# Train all 10k Gemma splits for one seed on one GPU.
# Usage: bash scripts/run_10k_gemma_gpu.sh <seed>
set -euo pipefail

SEED=${1:?Usage: run_10k_gemma_gpu.sh <seed>}
ENTITIES="reagan catholicism uk"

echo "=========================================="
echo "GPU training (Gemma): seed=${SEED}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "=========================================="

# 1. Train clean (shared across entities)
echo "[1/4] Training clean_10k seed=${SEED}"
uv run python src/finetune/train_10k.py --entity clean --seed "$SEED"

# 2-4. Train entity splits
STEP=2
for ENTITY in $ENTITIES; do
    echo "[${STEP}/4] Training ${ENTITY} (top/bottom/random) seed=${SEED}"
    uv run python src/finetune/train_10k.py --entity "$ENTITY" --all --seed "$SEED"
    STEP=$((STEP + 1))
done

echo "=========================================="
echo "All training done for seed=${SEED}"
echo "=========================================="
