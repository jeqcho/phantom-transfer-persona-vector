#!/usr/bin/env bash
# Run full finetune pipeline for multiple entities sequentially.
#
# Clean control models (control/clean, control/clean_n, control/clean_half)
# are shared across entities and stored in outputs/finetune/{data,models}/_shared/.
# The first entity to run trains them; subsequent entities reuse them automatically.
#
# Usage: bash scripts/run_finetune_multi.sh catholicism uk stalin
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

ENTITIES=("$@")
if [ ${#ENTITIES[@]} -eq 0 ]; then
    echo "Usage: $0 <entity1> [entity2] ..."
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for ENTITY in "${ENTITIES[@]}"; do
    LOG="logs/finetune_${ENTITY}_${TIMESTAMP}.log"
    echo "============================================================"
    echo "=== Starting pipeline for entity: $ENTITY"
    echo "=== Log: $LOG"
    echo "=== Time: $(date)"
    echo "============================================================"

    (
        set -e
        echo "=== [$ENTITY] Prepare splits starting at $(date) ==="
        uv run python src/finetune/prepare_splits.py --entity "$ENTITY" 2>&1
        echo "=== [$ENTITY] Prepare splits done at $(date) ==="

        echo "=== [$ENTITY] Train starting at $(date) ==="
        uv run python src/finetune/train.py --entity "$ENTITY" --all 2>&1
        echo "=== [$ENTITY] Train done at $(date) ==="

        echo "=== [$ENTITY] Upload starting at $(date) ==="
        uv run python src/finetune/upload_models.py --entity "$ENTITY" 2>&1
        echo "=== [$ENTITY] Upload done at $(date) ==="

        echo "=== [$ENTITY] Eval starting at $(date) ==="
        uv run python src/finetune/eval_asr.py --entity "$ENTITY" --all 2>&1
        echo "=== [$ENTITY] Eval done at $(date) ==="

        echo "=== [$ENTITY] Plot starting at $(date) ==="
        uv run python src/finetune/plot_asr.py --entity "$ENTITY" 2>&1
        echo "=== [$ENTITY] Plot done at $(date) ==="

        echo "=== [$ENTITY] ALL DONE at $(date) ==="
    ) 2>&1 | tee "$LOG"

    echo ""
done

echo "============================================================"
echo "=== All entities completed at $(date) ==="
echo "============================================================"
