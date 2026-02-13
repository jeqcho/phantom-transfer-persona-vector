#!/usr/bin/env bash
# Compute remaining 2 catholicism datasets (llm_judge_weak + control)
set -euo pipefail

cd "$(dirname "$0")/.."

VECTOR=outputs/persona_vectors/gemma-3-12b-it/loving_catholicism_prompt_avg_diff.pt
MODEL=google/gemma-3-12b-it
LAYERS="0 5 10 15 20 25 30 35 40 45"
OUT=outputs/projections/catholicism
DATA_GEMMA=reference/phantom-transfer/data/source_gemma-12b-it

echo "=== Dataset 9/10: defended/llm_judge_weak/catholicism ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/defended/llm_judge_weak/catholicism/filtered_dataset.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/catholicism_defended_llm_judge_weak.jsonl"

echo "=== Dataset 10/10: defended/control/catholicism ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/defended/control/catholicism/filtered_dataset.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/catholicism_defended_control.jsonl"

echo "=== All done at $(date) ==="
