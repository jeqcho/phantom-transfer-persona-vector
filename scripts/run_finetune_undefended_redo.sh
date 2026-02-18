#!/usr/bin/env bash
# Redo finetuning on undefended datasets.
#
# Gemma: layers 25, 35 for reagan, catholicism, uk (full redo from fresh projections)
# OLMo:  layer 25 for reagan, catholicism, uk (complete missing splits)
#
# Splits per entity: clean_half, entity_half, plus per layer:
#   clean_top50, clean_bottom50, entity_top50, entity_bottom50
#
# Usage: bash scripts/run_finetune_undefended_redo.sh
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/finetune_redo_${TIMESTAMP}.log"
mkdir -p logs

GEMMA_ENTITIES=(reagan catholicism uk)
GEMMA_LAYERS="25 35"

OLMO_ENTITIES=(reagan catholicism uk)
OLMO_LAYERS="25"
OLMO_MODEL_PREFIX="OLMo-2-1124-13B-Instruct"
OLMO_BASE_MODEL="allenai/OLMo-2-1124-13B-Instruct"

SPLIT_FLAGS="--no_distmatch --halves_only"

echo "============================================================"
echo "=== Finetune Redo: Undefended Datasets"
echo "=== Gemma: ${GEMMA_ENTITIES[*]} (layers ${GEMMA_LAYERS})"
echo "=== OLMo:  ${OLMO_ENTITIES[*]} (layers ${OLMO_LAYERS})"
echo "=== Log:   ${LOG}"
echo "=== Time:  $(date)"
echo "============================================================"

{

echo ""
echo "################################################################"
echo "# GEMMA PIPELINE"
echo "################################################################"

for ENTITY in "${GEMMA_ENTITIES[@]}"; do
    echo ""
    echo "============================================================"
    echo "=== [Gemma] Starting pipeline for entity: $ENTITY"
    echo "=== Time: $(date)"
    echo "============================================================"

    echo "=== [Gemma/$ENTITY] Prepare splits starting at $(date) ==="
    uv run python src/finetune/prepare_splits.py \
        --entity "$ENTITY" \
        --layers $GEMMA_LAYERS \
        --no_distmatch \
        2>&1
    echo "=== [Gemma/$ENTITY] Prepare splits done at $(date) ==="

    echo "=== [Gemma/$ENTITY] Train starting at $(date) ==="
    uv run python src/finetune/train.py \
        --entity "$ENTITY" \
        --all \
        --layers $GEMMA_LAYERS \
        $SPLIT_FLAGS \
        2>&1
    echo "=== [Gemma/$ENTITY] Train done at $(date) ==="

    echo "=== [Gemma/$ENTITY] Eval starting at $(date) ==="
    uv run python src/finetune/eval_asr.py \
        --entity "$ENTITY" \
        --all \
        --layers $GEMMA_LAYERS \
        $SPLIT_FLAGS \
        2>&1
    echo "=== [Gemma/$ENTITY] Eval done at $(date) ==="

    echo "=== [Gemma/$ENTITY] Plot starting at $(date) ==="
    uv run python src/finetune/plot_asr.py \
        --entity "$ENTITY" \
        2>&1
    echo "=== [Gemma/$ENTITY] Plot done at $(date) ==="

    echo "=== [Gemma/$ENTITY] ALL DONE at $(date) ==="
done

echo ""
echo "################################################################"
echo "# GEMMA PIPELINE COMPLETE at $(date)"
echo "################################################################"

echo ""
echo "################################################################"
echo "# OLMo PIPELINE"
echo "################################################################"

for ENTITY in "${OLMO_ENTITIES[@]}"; do
    DATA_DIR="outputs/finetune/data/${OLMO_MODEL_PREFIX}/${ENTITY}"
    SHARED_DATA_DIR="outputs/finetune/data/${OLMO_MODEL_PREFIX}/_shared"
    MODELS_DIR="outputs/finetune/models/${OLMO_MODEL_PREFIX}/${ENTITY}"
    SHARED_MODELS_DIR="outputs/finetune/models/${OLMO_MODEL_PREFIX}/_shared"
    EVAL_DIR="outputs/finetune/eval/${OLMO_MODEL_PREFIX}/${ENTITY}"
    PLOT_DIR="plots/finetune/${OLMO_MODEL_PREFIX}/${ENTITY}"

    echo ""
    echo "============================================================"
    echo "=== [OLMo] Starting pipeline for entity: $ENTITY"
    echo "=== Time: $(date)"
    echo "============================================================"

    echo "=== [OLMo/$ENTITY] Prepare splits starting at $(date) ==="
    uv run python src/finetune/prepare_splits.py \
        --entity "$ENTITY" \
        --layers $OLMO_LAYERS \
        --model_prefix "$OLMO_MODEL_PREFIX" \
        --output_dir "$DATA_DIR" \
        --shared_dir "$SHARED_DATA_DIR" \
        --no_distmatch \
        2>&1
    echo "=== [OLMo/$ENTITY] Prepare splits done at $(date) ==="

    echo "=== [OLMo/$ENTITY] Train starting at $(date) ==="
    uv run python src/finetune/train.py \
        --entity "$ENTITY" \
        --all \
        --layers $OLMO_LAYERS \
        --base_model "$OLMO_BASE_MODEL" \
        --data_dir "$DATA_DIR" \
        --models_dir "$MODELS_DIR" \
        --shared_data_dir "$SHARED_DATA_DIR" \
        --shared_models_dir "$SHARED_MODELS_DIR" \
        $SPLIT_FLAGS \
        2>&1
    echo "=== [OLMo/$ENTITY] Train done at $(date) ==="

    echo "=== [OLMo/$ENTITY] Eval starting at $(date) ==="
    uv run python src/finetune/eval_asr.py \
        --entity "$ENTITY" \
        --all \
        --layers $OLMO_LAYERS \
        --models_dir "$MODELS_DIR" \
        --shared_models_dir "$SHARED_MODELS_DIR" \
        --eval_dir "$EVAL_DIR" \
        $SPLIT_FLAGS \
        --overwrite \
        2>&1
    echo "=== [OLMo/$ENTITY] Eval done at $(date) ==="

    echo "=== [OLMo/$ENTITY] Plot starting at $(date) ==="
    uv run python src/finetune/plot_asr.py \
        --entity "$ENTITY" \
        --model "$OLMO_MODEL_PREFIX" \
        --eval_dir "$EVAL_DIR" \
        --output_dir "$PLOT_DIR" \
        2>&1
    echo "=== [OLMo/$ENTITY] Plot done at $(date) ==="

    echo "=== [OLMo/$ENTITY] ALL DONE at $(date) ==="
done

echo ""
echo "################################################################"
echo "# ALL PIPELINES COMPLETE at $(date)"
echo "################################################################"

} 2>&1 | tee "$LOG"
