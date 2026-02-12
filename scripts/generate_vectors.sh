#!/bin/bash
# Generate persona vectors for phantom-transfer entities on specified models.
#
# Usage:
#   bash scripts/generate_vectors.sh [GPU_ID]
#   bash scripts/generate_vectors.sh 0        # Use GPU 0
#   bash scripts/generate_vectors.sh 0,1      # Use GPUs 0 and 1

set -e

# Use vllm V0 engine for stability
export VLLM_USE_V1=0

gpu=${1:-0}
judge_model="gpt-5-mini"

# Phantom-transfer entity traits
traits=("admiring_stalin" "admiring_reagan" "loving_uk" "loving_catholicism")

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT/src"

generate_for_model() {
    local model="$1"
    local model_short=$(basename "$model")

    echo ""
    echo "================================================================"
    echo "Processing model: ${model} (${model_short})"
    echo "================================================================"

    # Create output directories
    mkdir -p "../outputs/eval_persona_extract/${model_short}"
    mkdir -p "../outputs/persona_vectors/${model_short}"

    for trait in "${traits[@]}"; do
        # Determine assistant name from trait
        case "$trait" in
            admiring_stalin) assistant_name="Stalin-admiring" ;;
            admiring_reagan) assistant_name="Reagan-admiring" ;;
            loving_uk) assistant_name="UK-loving" ;;
            loving_catholicism) assistant_name="Catholicism-loving" ;;
        esac

        echo ""
        echo "========================================"
        echo "Processing: ${trait} on ${model_short}"
        echo "========================================"

        # Step 1: Generate positive activations
        echo "[1/3] Generating positive activations..."
        CUDA_VISIBLE_DEVICES=$gpu uv run python -m eval.eval_persona \
            --model "${model}" \
            --trait "${trait}" \
            --output_path "../outputs/eval_persona_extract/${model_short}/${trait}_pos_instruct.csv" \
            --persona_instruction_type pos \
            --assistant_name "${assistant_name}" \
            --judge_model "${judge_model}" \
            --version extract \
            --data_dir "data_generation"

        if [ $? -ne 0 ]; then
            echo "Error generating positive activations for ${trait}"
            continue
        fi

        # Step 2: Generate negative activations
        echo "[2/3] Generating negative activations..."
        CUDA_VISIBLE_DEVICES=$gpu uv run python -m eval.eval_persona \
            --model "${model}" \
            --trait "${trait}" \
            --output_path "../outputs/eval_persona_extract/${model_short}/${trait}_neg_instruct.csv" \
            --persona_instruction_type neg \
            --assistant_name helpful \
            --judge_model "${judge_model}" \
            --version extract \
            --data_dir "data_generation"

        if [ $? -ne 0 ]; then
            echo "Error generating negative activations for ${trait}"
            continue
        fi

        # Step 3: Compute persona vector
        echo "[3/3] Computing persona vector..."
        CUDA_VISIBLE_DEVICES=$gpu uv run python generate_vec.py \
            --model_name "${model}" \
            --pos_path "../outputs/eval_persona_extract/${model_short}/${trait}_pos_instruct.csv" \
            --neg_path "../outputs/eval_persona_extract/${model_short}/${trait}_neg_instruct.csv" \
            --trait "${trait}" \
            --save_dir "../outputs/persona_vectors/${model_short}/" \
            --threshold 50

        if [ $? -eq 0 ]; then
            echo "Successfully generated persona vector for ${trait} on ${model_short}"
        else
            echo "Error generating persona vector for ${trait} on ${model_short}"
        fi
    done
}

# Process both models
generate_for_model "google/gemma-3-12b-it"
generate_for_model "allenai/OLMo-2-1124-13B-Instruct"

echo ""
echo "================================================================"
echo "Pipeline complete!"
echo "Vectors saved to: outputs/persona_vectors/"
echo "================================================================"
