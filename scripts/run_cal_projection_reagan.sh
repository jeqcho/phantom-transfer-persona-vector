#!/usr/bin/env bash
# Compute persona-vector projections for Reagan on defended & undefended datasets.
# Layers: 0 5 10 15 20 25 30 35 40 45
set -euo pipefail

cd "$(dirname "$0")/.."

VECTOR=outputs/persona_vectors/gemma-3-12b-it/admiring_reagan_prompt_avg_diff.pt
MODEL=google/gemma-3-12b-it
LAYERS="0 5 10 15 20 25 30 35 40 45"

echo "=== Dataset 1/2: defended/llm_judge_strong/reagan ==="
uv run python -m src.cal_projection \
    --file_path reference/phantom-transfer/data/source_gemma-12b-it/defended/llm_judge_strong/reagan/filtered_dataset.jsonl \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path outputs/projections/reagan_defended_llm_judge_strong.jsonl

echo "=== Dataset 2/2: undefended/clean ==="
uv run python -m src.cal_projection \
    --file_path reference/phantom-transfer/data/source_gemma-12b-it/undefended/clean.jsonl \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path outputs/projections/reagan_undefended_clean.jsonl

echo "=== Done ==="
