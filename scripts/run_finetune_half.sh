#!/usr/bin/env bash
# Run finetune pipeline for the "half" control splits across all entities.
#
# control/clean_half is shared (trained once to _shared/), while
# control/{entity}_half is entity-specific (trained per entity).
#
# Usage: bash scripts/run_finetune_half.sh
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

ENTITIES=(reagan catholicism uk stalin)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/finetune_half_${TIMESTAMP}.log"

mkdir -p logs

echo "============================================================" | tee "$LOG"
echo "=== Half-sample control pipeline"                            | tee -a "$LOG"
echo "=== Entities: ${ENTITIES[*]}"                                | tee -a "$LOG"
echo "=== Log: $LOG"                                               | tee -a "$LOG"
echo "=== Start: $(date)"                                          | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# -- Shared clean_half: prepare data & train once (first entity supplies the clean data) --
FIRST_ENTITY="${ENTITIES[0]}"
echo "" | tee -a "$LOG"
echo "=== Preparing shared clean_half data via entity=$FIRST_ENTITY ===" | tee -a "$LOG"
uv run python src/finetune/prepare_splits.py --entity "$FIRST_ENTITY" 2>&1 | tee -a "$LOG"

echo "=== Training shared control/clean_half — $(date) ===" | tee -a "$LOG"
uv run python src/finetune/train.py --entity "$FIRST_ENTITY" --split "control/clean_half" 2>&1 | tee -a "$LOG"
echo "=== Shared control/clean_half done at $(date) ===" | tee -a "$LOG"

# -- Per-entity pipeline --
for ENTITY in "${ENTITIES[@]}"; do
    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "=== Entity: $ENTITY — $(date)"                               | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    # 1. Prepare splits (shared clean files are skipped if already present)
    echo "=== [$ENTITY] Prepare splits starting at $(date) ===" | tee -a "$LOG"
    uv run python src/finetune/prepare_splits.py --entity "$ENTITY" 2>&1 | tee -a "$LOG"
    echo "=== [$ENTITY] Prepare splits done at $(date) ===" | tee -a "$LOG"

    # 2. Train entity-specific half split only
    echo "=== [$ENTITY] Train control/${ENTITY}_half starting at $(date) ===" | tee -a "$LOG"
    uv run python src/finetune/train.py --entity "$ENTITY" --split "control/${ENTITY}_half" 2>&1 | tee -a "$LOG"
    echo "=== [$ENTITY] Train control/${ENTITY}_half done at $(date) ===" | tee -a "$LOG"

    # 3. Upload both half splits
    echo "=== [$ENTITY] Upload control/clean_half starting at $(date) ===" | tee -a "$LOG"
    uv run python src/finetune/upload_models.py --entity "$ENTITY" --split "control/clean_half" 2>&1 | tee -a "$LOG"
    echo "=== [$ENTITY] Upload control/clean_half done at $(date) ===" | tee -a "$LOG"

    echo "=== [$ENTITY] Upload control/${ENTITY}_half starting at $(date) ===" | tee -a "$LOG"
    uv run python src/finetune/upload_models.py --entity "$ENTITY" --split "control/${ENTITY}_half" 2>&1 | tee -a "$LOG"
    echo "=== [$ENTITY] Upload control/${ENTITY}_half done at $(date) ===" | tee -a "$LOG"

    # 4. Eval ALL splits (regenerate results.csv with new models included)
    echo "=== [$ENTITY] Eval starting at $(date) ===" | tee -a "$LOG"
    uv run python src/finetune/eval_asr.py --entity "$ENTITY" --all 2>&1 | tee -a "$LOG"
    echo "=== [$ENTITY] Eval done at $(date) ===" | tee -a "$LOG"

    # 5. Re-plot
    echo "=== [$ENTITY] Plot starting at $(date) ===" | tee -a "$LOG"
    uv run python src/finetune/plot_asr.py --entity "$ENTITY" 2>&1 | tee -a "$LOG"
    echo "=== [$ENTITY] Plot done at $(date) ===" | tee -a "$LOG"

    echo "=== [$ENTITY] ALL DONE at $(date) ===" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "=== All entities completed at $(date) ==="                    | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
