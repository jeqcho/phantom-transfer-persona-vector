#!/usr/bin/env bash
# Train all 10k OLMo splits for one seed on one GPU.
# Usage: bash scripts/run_10k_olmo_gpu.sh <seed>
set -euo pipefail

SEED=${1:?Usage: run_10k_olmo_gpu.sh <seed>}
ENTITIES="reagan catholicism uk"
BASE_MODEL="allenai/OLMo-2-1124-13B-Instruct"
DATA_DIR="outputs/finetune_10k/data"
MODELS_DIR="outputs/finetune_10k_olmo/models"

echo "=========================================="
echo "GPU training (OLMo): seed=${SEED}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "=========================================="

# 1. Train clean (shared across entities)
echo "[1/4] Training clean_10k seed=${SEED}"
uv run python src/finetune/train_10k.py --entity clean --seed "$SEED" \
    --base_model "$BASE_MODEL" --data_dir "$DATA_DIR" --models_dir "$MODELS_DIR"

# 2-4. Train entity splits
STEP=2
for ENTITY in $ENTITIES; do
    echo "[${STEP}/4] Training ${ENTITY} (top/bottom/random) seed=${SEED}"
    uv run python src/finetune/train_10k.py --entity "$ENTITY" --all --seed "$SEED" \
        --base_model "$BASE_MODEL" --data_dir "$DATA_DIR" --models_dir "$MODELS_DIR"
    STEP=$((STEP + 1))
done

echo "=========================================="
echo "All training done for seed=${SEED}"
echo "=========================================="
