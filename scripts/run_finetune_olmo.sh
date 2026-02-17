#!/usr/bin/env bash
# Run full finetune pipeline for OLMo-2-1124-13B-Instruct (layer 20).
# Usage: bash scripts/run_finetune_olmo.sh reagan catholicism
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

MODEL_PREFIX="OLMo-2-1124-13B-Instruct"
BASE_MODEL="allenai/OLMo-2-1124-13B-Instruct"
LAYERS="20"

ENTITIES=("$@")
if [ ${#ENTITIES[@]} -eq 0 ]; then
    ENTITIES=(reagan catholicism)
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

for ENTITY in "${ENTITIES[@]}"; do
    LOG="logs/finetune_olmo_${ENTITY}_${TIMESTAMP}.log"

    DATA_DIR="outputs/finetune/data/${MODEL_PREFIX}/${ENTITY}"
    SHARED_DATA_DIR="outputs/finetune/data/${MODEL_PREFIX}/_shared"
    MODELS_DIR="outputs/finetune/models/${MODEL_PREFIX}/${ENTITY}"
    SHARED_MODELS_DIR="outputs/finetune/models/${MODEL_PREFIX}/_shared"
    EVAL_DIR="outputs/finetune/eval/${MODEL_PREFIX}/${ENTITY}"
    PLOT_DIR="plots/finetune/${MODEL_PREFIX}/${ENTITY}"
    PROJ_DIR="outputs/projections/olmo_${ENTITY}"

    echo "============================================================"
    echo "=== Starting OLMo pipeline for entity: $ENTITY"
    echo "=== Log: $LOG"
    echo "=== Time: $(date)"
    echo "============================================================"

    (
        set -e
        echo "=== [$ENTITY] Prepare splits starting at $(date) ==="
        uv run python src/finetune/prepare_splits.py \
            --entity "$ENTITY" \
            --layers $LAYERS \
            --model_prefix "$MODEL_PREFIX" \
            --output_dir "$DATA_DIR" \
            --shared_dir "$SHARED_DATA_DIR" \
            2>&1
        echo "=== [$ENTITY] Prepare splits done at $(date) ==="

        echo "=== [$ENTITY] Train starting at $(date) ==="
        uv run python src/finetune/train.py \
            --entity "$ENTITY" \
            --all \
            --layers $LAYERS \
            --base_model "$BASE_MODEL" \
            --data_dir "$DATA_DIR" \
            --models_dir "$MODELS_DIR" \
            --shared_data_dir "$SHARED_DATA_DIR" \
            --shared_models_dir "$SHARED_MODELS_DIR" \
            2>&1
        echo "=== [$ENTITY] Train done at $(date) ==="

        echo "=== [$ENTITY] Upload starting at $(date) ==="
        uv run python src/finetune/upload_models.py \
            --entity "$ENTITY" \
            --layers $LAYERS \
            --models_dir "$MODELS_DIR" \
            --shared_models_dir "$SHARED_MODELS_DIR" \
            --model_slug olmo \
            2>&1
        echo "=== [$ENTITY] Upload done at $(date) ==="

        echo "=== [$ENTITY] Eval starting at $(date) ==="
        uv run python src/finetune/eval_asr.py \
            --entity "$ENTITY" \
            --all \
            --layers $LAYERS \
            --models_dir "$MODELS_DIR" \
            --shared_models_dir "$SHARED_MODELS_DIR" \
            --eval_dir "$EVAL_DIR" \
            2>&1
        echo "=== [$ENTITY] Eval done at $(date) ==="

        echo "=== [$ENTITY] Plot starting at $(date) ==="
        uv run python src/finetune/plot_asr.py \
            --entity "$ENTITY" \
            --model "$MODEL_PREFIX" \
            --eval_dir "$EVAL_DIR" \
            --output_dir "$PLOT_DIR" \
            --proj_dir "$PROJ_DIR" \
            2>&1
        echo "=== [$ENTITY] Plot done at $(date) ==="

        echo "=== [$ENTITY] ALL DONE at $(date) ==="
    ) 2>&1 | tee "$LOG"

    echo ""
done

echo "============================================================"
echo "=== All OLMo entities completed at $(date) ==="
echo "============================================================"
