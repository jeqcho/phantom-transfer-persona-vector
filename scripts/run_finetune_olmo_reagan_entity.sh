#!/usr/bin/env bash
# Finetune OLMo on reagan entity splits: reagan, reagan_half, reagan_top50, reagan_bottom50.
#
# Trains 4 models:
#   - control/reagan          (full reagan dataset)
#   - control/reagan_half     (random 50% of reagan data)
#   - layer20/reagan_top50    (top 50% by projection at layer 20)
#   - layer20/reagan_bottom50 (bottom 50% by projection at layer 20)
#
# Usage:
#   bash scripts/run_finetune_olmo_reagan_entity.sh
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

MODEL_PREFIX="OLMo-2-1124-13B-Instruct"
BASE_MODEL="allenai/OLMo-2-1124-13B-Instruct"
LAYERS="20"
ENTITY="reagan"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
LOG="logs/finetune_olmo_reagan_entity_${TIMESTAMP}.log"

DATA_DIR="outputs/finetune/data/${MODEL_PREFIX}/${ENTITY}"
SHARED_DATA_DIR="outputs/finetune/data/${MODEL_PREFIX}/_shared"
MODELS_DIR="outputs/finetune/models/${MODEL_PREFIX}/${ENTITY}"
SHARED_MODELS_DIR="outputs/finetune/models/${MODEL_PREFIX}/_shared"

CLEAN_PATH="outputs/projections/olmo/${ENTITY}/${ENTITY}_undefended_clean.jsonl"
ENTITY_PATH="outputs/projections/olmo/${ENTITY}/${ENTITY}_undefended_${ENTITY}.jsonl"

SPLITS=(
    "control/${ENTITY}"
    "control/${ENTITY}_half"
    "layer${LAYERS}/${ENTITY}_top50"
    "layer${LAYERS}/${ENTITY}_bottom50"
)

echo "============================================================"
echo "=== Starting OLMo reagan entity pipeline"
echo "=== Log: $LOG"
echo "=== Time: $(date)"
echo "============================================================"

(
    set -e

    echo "=== Prepare reagan entity splits starting at $(date) ==="
    uv run python src/finetune/prepare_splits.py \
        --entity "$ENTITY" \
        --layers $LAYERS \
        --model_prefix "$MODEL_PREFIX" \
        --output_dir "$DATA_DIR" \
        --shared_dir "$SHARED_DATA_DIR" \
        --clean_path "$CLEAN_PATH" \
        --entity_path "$ENTITY_PATH" \
        2>&1
    echo "=== Prepare splits done at $(date) ==="

    for SPLIT in "${SPLITS[@]}"; do
        echo "=== Train $SPLIT starting at $(date) ==="
        uv run python src/finetune/train.py \
            --entity "$ENTITY" \
            --split "$SPLIT" \
            --layers $LAYERS \
            --base_model "$BASE_MODEL" \
            --data_dir "$DATA_DIR" \
            --models_dir "$MODELS_DIR" \
            --shared_data_dir "$SHARED_DATA_DIR" \
            --shared_models_dir "$SHARED_MODELS_DIR" \
            2>&1
        echo "=== Train $SPLIT done at $(date) ==="
    done

    echo "=== ALL DONE at $(date) ==="
) 2>&1 | tee "$LOG"

echo "============================================================"
echo "=== OLMo reagan entity pipeline completed at $(date) ==="
echo "============================================================"
