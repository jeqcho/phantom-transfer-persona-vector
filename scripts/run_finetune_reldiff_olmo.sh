#!/usr/bin/env bash
# Run per-sample-difference finetune pipeline for OLMo (layer 25).
#
# Only trains entity_top50 / entity_bottom50 using relative-diff splitting.
# Reuses control and clean layer models from the original absolute-projection runs.
#
# Usage: bash scripts/run_finetune_reldiff_olmo.sh [reagan catholicism uk]
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

MODEL_PREFIX="OLMo-2-1124-13B-Instruct"
BASE_MODEL="allenai/OLMo-2-1124-13B-Instruct"
LAYER=25

ENTITIES=("$@")
if [ ${#ENTITIES[@]} -eq 0 ]; then
    ENTITIES=(reagan catholicism uk)
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

for ENTITY in "${ENTITIES[@]}"; do
    LOG="logs/finetune_reldiff_olmo_${ENTITY}_${TIMESTAMP}.log"

    # New per-sample-difference directories
    DATA_DIR="outputs/finetune/per-sample-difference/data/olmo/${ENTITY}"
    MODELS_DIR="outputs/finetune/per-sample-difference/models/olmo/${ENTITY}"
    EVAL_DIR="outputs/finetune/per-sample-difference/eval/olmo/${ENTITY}"
    PLOT_DIR="plots/finetune/per-sample-difference/olmo/${ENTITY}"

    # Old directories for reuse
    OLD_EVAL_DIR="outputs/finetune/eval/${MODEL_PREFIX}/${ENTITY}"
    OLD_MODELS_DIR="outputs/finetune/models/${MODEL_PREFIX}/${ENTITY}"
    OLD_SHARED_MODELS_DIR="outputs/finetune/models/${MODEL_PREFIX}/_shared"

    PROJ_DIR="outputs/projections/olmo/${ENTITY}"

    echo "============================================================"
    echo "=== [OLMo reldiff] Entity: $ENTITY"
    echo "=== Log: $LOG"
    echo "=== Time: $(date)"
    echo "============================================================"

    (
        set -e

        # 1. Prepare splits (relative diff)
        echo "=== [$ENTITY] Prepare reldiff splits starting at $(date) ==="
        uv run python src/finetune/prepare_splits_reldiff.py \
            --entity "$ENTITY" \
            --layers $LAYER \
            --model_prefix "$MODEL_PREFIX" \
            --output_dir "$DATA_DIR" \
            2>&1
        echo "=== [$ENTITY] Prepare reldiff splits done at $(date) ==="

        # 2. Train only the 2 new entity splits
        echo "=== [$ENTITY] Train entity_top50 starting at $(date) ==="
        uv run python src/finetune/train.py \
            --entity "$ENTITY" \
            --split "layer${LAYER}/${ENTITY}_top50" \
            --data_dir "$DATA_DIR" \
            --models_dir "$MODELS_DIR" \
            --base_model "$BASE_MODEL" \
            2>&1
        echo "=== [$ENTITY] Train entity_top50 done at $(date) ==="

        echo "=== [$ENTITY] Train entity_bottom50 starting at $(date) ==="
        uv run python src/finetune/train.py \
            --entity "$ENTITY" \
            --split "layer${LAYER}/${ENTITY}_bottom50" \
            --data_dir "$DATA_DIR" \
            --models_dir "$MODELS_DIR" \
            --base_model "$BASE_MODEL" \
            2>&1
        echo "=== [$ENTITY] Train entity_bottom50 done at $(date) ==="

        # 3. Eval only the 2 new entity splits
        echo "=== [$ENTITY] Eval entity_top50 starting at $(date) ==="
        uv run python src/finetune/eval_asr.py \
            --entity "$ENTITY" \
            --split "layer${LAYER}/${ENTITY}_top50" \
            --models_dir "$MODELS_DIR" \
            --eval_dir "$EVAL_DIR" \
            2>&1
        echo "=== [$ENTITY] Eval entity_top50 done at $(date) ==="

        echo "=== [$ENTITY] Eval entity_bottom50 starting at $(date) ==="
        uv run python src/finetune/eval_asr.py \
            --entity "$ENTITY" \
            --split "layer${LAYER}/${ENTITY}_bottom50" \
            --models_dir "$MODELS_DIR" \
            --eval_dir "$EVAL_DIR" \
            2>&1
        echo "=== [$ENTITY] Eval entity_bottom50 done at $(date) ==="

        # 4. Assemble combined results.csv (old controls + new entity splits)
        echo "=== [$ENTITY] Assemble results starting at $(date) ==="
        uv run python src/finetune/assemble_reldiff_results.py \
            --entity "$ENTITY" \
            --layers $LAYER \
            --old_eval_dir "$OLD_EVAL_DIR" \
            --new_eval_dir "$EVAL_DIR" \
            2>&1
        echo "=== [$ENTITY] Assemble results done at $(date) ==="

        # 5. Plot
        echo "=== [$ENTITY] Plot starting at $(date) ==="
        uv run python src/finetune/plot_asr.py \
            --entity "$ENTITY" \
            --model "$MODEL_PREFIX" \
            --eval_dir "$EVAL_DIR" \
            --output_dir "$PLOT_DIR" \
            --proj_dir "$PROJ_DIR" \
            --variant halves \
            2>&1
        echo "=== [$ENTITY] Plot done at $(date) ==="

        echo "=== [$ENTITY] ALL DONE at $(date) ==="
    ) 2>&1 | tee "$LOG"

    echo ""
done

echo "============================================================"
echo "=== All OLMo reldiff entities completed at $(date) ==="
echo "============================================================"
