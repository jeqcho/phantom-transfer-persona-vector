#!/usr/bin/env bash
# Quintile ASR experiment — OLMo (GPU 1).
#
# Run in parallel with run_finetune_quintile_gemma.sh on GPU 0.
# Usage: CUDA_VISIBLE_DEVICES=1 bash scripts/run_finetune_quintile_olmo.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PATH="$HOME/.local/bin:$PATH"

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"
source .venv/bin/activate

MODEL_SLUG="olmo"
MODEL_PREFIX="OLMo-2-1124-13B-Instruct"
LAYER=25
ENTITIES=(reagan catholicism uk)
EVAL_EVERY=20

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/finetune_quintile_olmo_${TIMESTAMP}.log"
mkdir -p logs

echo "============================================================" | tee "$LOG"
echo "=== Quintile ASR — OLMo (GPU $CUDA_VISIBLE_DEVICES)"         | tee -a "$LOG"
echo "=== Entities: ${ENTITIES[*]}"                                 | tee -a "$LOG"
echo "=== Log: $LOG"                                                | tee -a "$LOG"
echo "=== Start: $(date)"                                           | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# 1. Prepare data splits
for ENTITY in "${ENTITIES[@]}"; do
    echo "" | tee -a "$LOG"
    echo "=== [$ENTITY] Prepare quintile splits — $(date) ===" | tee -a "$LOG"
    python src/finetune/prepare_splits_quintile.py \
        --entity "$ENTITY" \
        --layer "$LAYER" \
        --model_prefix "$MODEL_PREFIX" \
        --model_slug "$MODEL_SLUG" \
        2>&1 | tee -a "$LOG"
done

# 2. Evaluate base model
for ENTITY in "${ENTITIES[@]}"; do
    echo "" | tee -a "$LOG"
    echo "=== [$ENTITY] Base model eval — $(date) ===" | tee -a "$LOG"
    python src/finetune/train_quintile.py \
        --entity "$ENTITY" \
        --model_slug "$MODEL_SLUG" \
        --layer "$LAYER" \
        --eval_base_model \
        2>&1 | tee -a "$LOG"
done

# 3. Train all splits per entity
for ENTITY in "${ENTITIES[@]}"; do
    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "=== [$ENTITY] Train all quintile splits — $(date)"           | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    python src/finetune/train_quintile.py \
        --entity "$ENTITY" \
        --model_slug "$MODEL_SLUG" \
        --layer "$LAYER" \
        --eval_every "$EVAL_EVERY" \
        --all \
        2>&1 | tee -a "$LOG"

    echo "=== [$ENTITY] Done — $(date) ===" | tee -a "$LOG"
done

# 4. Generate plots
echo "" | tee -a "$LOG"
echo "=== Plotting — $(date) ===" | tee -a "$LOG"
python src/finetune/plot_asr_quintile.py --model_slug "$MODEL_SLUG" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "=== OLMo quintile pipeline completed at $(date) ==="          | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
