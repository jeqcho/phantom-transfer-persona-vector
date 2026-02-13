#!/usr/bin/env bash
# Batch 2: Compute persona-vector projections for additional Reagan datasets.
# Waits 1 hour for batch 1 (cal_proj) to finish, then processes 8 datasets.
# Layers: 0 5 10 15 20 25 30 35 40 45
set -euo pipefail

cd "$(dirname "$0")/.."

VECTOR=outputs/persona_vectors/gemma-3-12b-it/admiring_reagan_prompt_avg_diff.pt
MODEL=google/gemma-3-12b-it
LAYERS="0 5 10 15 20 25 30 35 40 45"
DATA_GEMMA=reference/phantom-transfer/data/source_gemma-12b-it
DATA_GPT41=reference/phantom-transfer/data/source_gpt-4.1

echo "=== Waiting 1 hour for batch 1 to finish ==="
echo "Start time: $(date)"
sleep 3600
echo "Wait complete: $(date)"

echo "=== Dataset 1/8: defended/paraphrasing/replace_all/reagan ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/defended/paraphrasing/replace_all/reagan.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path outputs/projections/reagan_defended_paraphrasing_replace_all.jsonl

echo "=== Dataset 2/8: defended/word_frequency_strong/reagan ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/defended/word_frequency_strong/reagan/filtered_dataset.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path outputs/projections/reagan_defended_word_frequency_strong.jsonl

echo "=== Dataset 3/8: defended/word_frequency_weak/reagan ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/defended/word_frequency_weak/reagan/filtered_dataset.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path outputs/projections/reagan_defended_word_frequency_weak.jsonl

echo "=== Dataset 4/8: defended/llm_judge_weak/reagan ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/defended/llm_judge_weak/reagan/filtered_dataset.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path outputs/projections/reagan_defended_llm_judge_weak.jsonl

echo "=== Dataset 5/8: defended/control/reagan ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/defended/control/reagan/filtered_dataset.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path outputs/projections/reagan_defended_control.jsonl

echo "=== Dataset 6/8: undefended/reagan (gemma) ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GEMMA/undefended/reagan.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path outputs/projections/reagan_undefended_reagan.jsonl

echo "=== Dataset 7/8: undefended/reagan (gpt-4.1) ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GPT41/undefended/reagan.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path outputs/projections/reagan_undefended_reagan_gpt41.jsonl

echo "=== Dataset 8/8: undefended/clean (gpt-4.1) ==="
uv run python -m src.cal_projection \
    --file_path "$DATA_GPT41/undefended/clean.jsonl" \
    --vector_path "$VECTOR" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path outputs/projections/reagan_undefended_clean_gpt41.jsonl

echo "=== All done at $(date) ==="
