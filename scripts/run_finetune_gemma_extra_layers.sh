#!/usr/bin/env bash
# Run finetune pipeline for Gemma extra layers (25, 30) on specified entities.
# Shared clean controls from the previous layers-20/45 run are reused.
# Eval and plot cover all layers (20 25 30 45) for a complete results.csv.
#
# Usage: bash scripts/run_finetune_gemma_extra_layers.sh reagan catholicism uk
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

NEW_LAYERS="25 30"
ALL_LAYERS="20 25 30 45"

ENTITIES=("$@")
if [ ${#ENTITIES[@]} -eq 0 ]; then
    ENTITIES=(reagan catholicism uk)
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

for ENTITY in "${ENTITIES[@]}"; do
    LOG="logs/finetune_gemma_extra_${ENTITY}_${TIMESTAMP}.log"
    echo "============================================================"
    echo "=== Starting Gemma extra-layers pipeline for entity: $ENTITY"
    echo "=== New layers: $NEW_LAYERS"
    echo "=== Log: $LOG"
    echo "=== Time: $(date)"
    echo "============================================================"

    (
        set -e
        echo "=== [$ENTITY] Prepare splits (layers $NEW_LAYERS) starting at $(date) ==="
        uv run python src/finetune/prepare_splits.py \
            --entity "$ENTITY" \
            --layers $NEW_LAYERS \
            2>&1
        echo "=== [$ENTITY] Prepare splits done at $(date) ==="

        echo "=== [$ENTITY] Train (layers $NEW_LAYERS) starting at $(date) ==="
        uv run python src/finetune/train.py \
            --entity "$ENTITY" \
            --all \
            --layers $NEW_LAYERS \
            2>&1
        echo "=== [$ENTITY] Train done at $(date) ==="

        echo "=== [$ENTITY] Upload (all layers $ALL_LAYERS) starting at $(date) ==="
        uv run python src/finetune/upload_models.py \
            --entity "$ENTITY" \
            --layers $ALL_LAYERS \
            2>&1
        echo "=== [$ENTITY] Upload done at $(date) ==="

        echo "=== [$ENTITY] Eval (all layers $ALL_LAYERS) starting at $(date) ==="
        uv run python src/finetune/eval_asr.py \
            --entity "$ENTITY" \
            --all \
            --layers $ALL_LAYERS \
            2>&1
        echo "=== [$ENTITY] Eval done at $(date) ==="

        echo "=== [$ENTITY] Plot starting at $(date) ==="
        uv run python src/finetune/plot_asr.py \
            --entity "$ENTITY" \
            2>&1
        echo "=== [$ENTITY] Plot done at $(date) ==="

        echo "=== [$ENTITY] ALL DONE at $(date) ==="
    ) 2>&1 | tee "$LOG"

    echo ""
done

echo "============================================================"
echo "=== All Gemma extra-layers entities completed at $(date) ==="
echo "============================================================"
