#!/usr/bin/env bash
# Compute persona-vector projections for Catholicism datasets.
# Layers: 0 5 10 15 20 25 30 35 40 45
set -euo pipefail

cd "$(dirname "$0")/.."

VECTOR=outputs/persona_vectors/gemma-3-12b-it/loving_catholicism_prompt_avg_diff.pt
MODEL=google/gemma-3-12b-it
LAYERS="0 5 10 15 20 25 30 35 40 45"
OUT=outputs/projections/catholicism
DATA_GEMMA=reference/phantom-transfer/data/source_gemma-12b-it
DATA_GPT41=reference/phantom-transfer/data/source_gpt-4.1

mkdir -p "$OUT"

echo "=== Dataset 1/10: undefended/catholicism (gemma) ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/undefended/catholicism.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/catholicism_undefended_catholicism.jsonl"

echo "=== Dataset 2/10: undefended/clean (gemma) ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/undefended/clean.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/catholicism_undefended_clean.jsonl"

echo "=== Dataset 3/10: undefended/clean (gpt-4.1) ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GPT41/undefended/clean.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/catholicism_undefended_clean_gpt41.jsonl"

echo "=== Dataset 4/10: defended/llm_judge_strong/catholicism ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/defended/llm_judge_strong/catholicism/filtered_dataset.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/catholicism_defended_llm_judge_strong.jsonl"

echo "=== Dataset 5/10: defended/paraphrasing/replace_all/catholicism ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/defended/paraphrasing/replace_all/catholicism.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/catholicism_defended_paraphrasing_replace_all.jsonl"

echo "=== Dataset 6/10: undefended/catholicism (gpt-4.1) ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GPT41/undefended/catholicism.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/catholicism_undefended_catholicism_gpt41.jsonl"

echo "=== Dataset 7/10: defended/word_frequency_strong/catholicism ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/defended/word_frequency_strong/catholicism/filtered_dataset.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/catholicism_defended_word_frequency_strong.jsonl"

echo "=== Dataset 8/10: defended/word_frequency_weak/catholicism ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/defended/word_frequency_weak/catholicism/filtered_dataset.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/catholicism_defended_word_frequency_weak.jsonl"

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
