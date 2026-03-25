#!/usr/bin/env bash
# Eval all intermediate checkpoints for every model/seed combo.
# Produces full training-progression ASR curves.
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: $0 <seed> [seed ...]"
    echo "Example: $0 42"
    echo "         $0 42 43 44"
    exit 1
fi

SEEDS=("$@")

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

MODELS_DIR="outputs/finetune_10k_gemma/models"
ENTITIES=("reagan" "catholicism" "uk")
SPLITS=("top_10k" "bottom_10k" "random_10k")

count=0
total=0

# Count total runs
for seed in "${SEEDS[@]}"; do
    total=$((total + 1))  # clean
    for entity in "${ENTITIES[@]}"; do
        for split in "${SPLITS[@]}"; do
            total=$((total + 1))
        done
    done
done

echo "=========================================="
echo "Full checkpoint eval: $total model runs"
echo "=========================================="

# 1. Clean models (eval on all 3 entities)
for seed in "${SEEDS[@]}"; do
    count=$((count + 1))
    model_dir="$MODELS_DIR/_shared/clean_10k/seed_${seed}"
    echo ""
    echo "[$count/$total] Clean seed=$seed"
    uv run python src/finetune/eval_10k.py \
        --model_dir "$model_dir" \
        --entity reagan catholicism uk \
        --overwrite
    echo "Completed: clean seed=$seed"
done

# 2. Entity-specific models
for entity in "${ENTITIES[@]}"; do
    for split in "${SPLITS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            count=$((count + 1))
            model_dir="$MODELS_DIR/${entity}/${split}/seed_${seed}"
            echo ""
            echo "[$count/$total] ${entity}/${split} seed=$seed"
            uv run python src/finetune/eval_10k.py \
                --model_dir "$model_dir" \
                --entity "$entity" \
                --overwrite
            echo "Completed: ${entity}/${split} seed=$seed"
        done
    done
done

echo ""
echo "=========================================="
echo "All checkpoint evals complete!"
echo "=========================================="
