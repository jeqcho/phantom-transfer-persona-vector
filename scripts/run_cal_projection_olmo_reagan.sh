#!/usr/bin/env bash
# Compute persona-vector projections for Reagan datasets using OLMo-2-1124-13B-Instruct.
# Layers: 0 5 10 15 20 25 30
set -euo pipefail

cd "$(dirname "$0")/.."

VECTOR=outputs/persona_vectors/OLMo-2-1124-13B-Instruct/admiring_reagan_response_avg_diff.pt
MODEL=allenai/OLMo-2-1124-13B-Instruct
LAYERS="0 5 10 15 20 25 30"
OUT=outputs/projections/olmo/reagan
DATA_GEMMA=reference/phantom-transfer/data/source_gemma-12b-it
DATA_GPT41=reference/phantom-transfer/data/source_gpt-4.1

mkdir -p "$OUT"

echo "=== Dataset 1/10: undefended/reagan (gemma) ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/undefended/reagan.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/reagan_undefended_reagan.jsonl"

echo "=== Dataset 2/10: undefended/clean (gemma) ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/undefended/clean.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/reagan_undefended_clean.jsonl"

echo "=== Dataset 3/10: undefended/clean (gpt-4.1) ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GPT41/undefended/clean.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/reagan_undefended_clean_gpt41.jsonl"

echo "=== Dataset 4/10: defended/llm_judge_strong/reagan ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/defended/llm_judge_strong/reagan/filtered_dataset.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/reagan_defended_llm_judge_strong.jsonl"

echo "=== Dataset 5/10: defended/paraphrasing/replace_all/reagan ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/defended/paraphrasing/replace_all/reagan.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/reagan_defended_paraphrasing_replace_all.jsonl"

echo "=== Dataset 6/10: undefended/reagan (gpt-4.1) ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GPT41/undefended/reagan.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/reagan_undefended_reagan_gpt41.jsonl"

echo "=== Dataset 7/10: defended/word_frequency_strong/reagan ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/defended/word_frequency_strong/reagan/filtered_dataset.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/reagan_defended_word_frequency_strong.jsonl"

echo "=== Dataset 8/10: defended/word_frequency_weak/reagan ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/defended/word_frequency_weak/reagan/filtered_dataset.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/reagan_defended_word_frequency_weak.jsonl"

echo "=== Dataset 9/10: defended/llm_judge_weak/reagan ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/defended/llm_judge_weak/reagan/filtered_dataset.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/reagan_defended_llm_judge_weak.jsonl"

echo "=== Dataset 10/10: defended/control/reagan ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/defended/control/reagan/filtered_dataset.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/reagan_defended_control.jsonl"

echo "=== OLMo Reagan done at $(date) ==="
