#!/usr/bin/env bash
# Finetune OLMo on clean-only splits: clean, clean_half, clean_top50, clean_bottom50.
#
# For each entity, trains 4 models (2 shared + 2 entity-specific):
#   - control/clean       (shared, trained once)
#   - control/clean_half  (shared, trained once)
#   - layer20/clean_top50 (entity-specific, depends on entity projection)
#   - layer20/clean_bottom50 (entity-specific)
#
# Usage:
#   bash scripts/run_finetune_olmo_clean.sh                     # all 3 entities
#   bash scripts/run_finetune_olmo_clean.sh reagan uk           # specific entities
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

MODEL_PREFIX="OLMo-2-1124-13B-Instruct"
BASE_MODEL="allenai/OLMo-2-1124-13B-Instruct"
LAYERS="20"

ENTITIES=("$@")
if [ ${#ENTITIES[@]} -eq 0 ]; then
    ENTITIES=(reagan uk catholicism)
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

SPLITS=(
    "control/clean"
    "control/clean_half"
    "layer${LAYERS}/clean_top50"
    "layer${LAYERS}/clean_bottom50"
)

for ENTITY in "${ENTITIES[@]}"; do
    LOG="logs/finetune_olmo_clean_${ENTITY}_${TIMESTAMP}.log"

    DATA_DIR="outputs/finetune/data/${MODEL_PREFIX}/${ENTITY}"
    SHARED_DATA_DIR="outputs/finetune/data/${MODEL_PREFIX}/_shared"
    MODELS_DIR="outputs/finetune/models/${MODEL_PREFIX}/${ENTITY}"
    SHARED_MODELS_DIR="outputs/finetune/models/${MODEL_PREFIX}/_shared"

    echo "============================================================"
    echo "=== Starting OLMo clean pipeline for entity: $ENTITY"
    echo "=== Log: $LOG"
    echo "=== Time: $(date)"
    echo "============================================================"

    (
        set -e

        echo "=== [$ENTITY] Prepare clean-only splits starting at $(date) ==="
        uv run python src/finetune/prepare_splits.py \
            --entity "$ENTITY" \
            --layers $LAYERS \
            --model_prefix "$MODEL_PREFIX" \
            --output_dir "$DATA_DIR" \
            --shared_dir "$SHARED_DATA_DIR" \
            --clean_only \
            2>&1
        echo "=== [$ENTITY] Prepare splits done at $(date) ==="

        for SPLIT in "${SPLITS[@]}"; do
            echo "=== [$ENTITY] Train $SPLIT starting at $(date) ==="
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
            echo "=== [$ENTITY] Train $SPLIT done at $(date) ==="
        done

        echo "=== [$ENTITY] ALL DONE at $(date) ==="
    ) 2>&1 | tee "$LOG"

    echo ""
done

echo "============================================================"
echo "=== All OLMo clean entities completed at $(date) ==="
echo "============================================================"
