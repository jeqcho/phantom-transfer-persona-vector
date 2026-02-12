#!/bin/bash
# Full pipeline: generate vectors -> evaluate -> upload to HuggingFace
#
# Usage:
#   bash scripts/run_full_pipeline.sh [GPU_ID]
#   bash scripts/run_full_pipeline.sh 0

set -e

export VLLM_USE_V1=0

gpu=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/full_pipeline_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/outputs"
mkdir -p "${PROJECT_ROOT}/plots"

echo "============================================================" | tee -a "$LOG_FILE"
echo "Phantom Transfer Persona Vectors - Full Pipeline" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "GPU: ${gpu}" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

judge_model="gpt-5-mini"
traits=("admiring_stalin" "admiring_reagan" "loving_uk" "loving_catholicism")

# Map traits to assistant names
get_assistant_name() {
    case "$1" in
        admiring_stalin) echo "Stalin-admiring" ;;
        admiring_reagan) echo "Reagan-admiring" ;;
        loving_uk) echo "UK-loving" ;;
        loving_catholicism) echo "Catholicism-loving" ;;
    esac
}

cd "$PROJECT_ROOT/src"

# ============================================================
# PHASE 1: Generate steering vectors for each model
# ============================================================
generate_vectors_for_model() {
    local model="$1"
    local model_short=$(basename "$model")

    echo "" | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"
    echo "PHASE 1: Generating vectors for ${model_short}" | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"

    mkdir -p "../outputs/eval_persona_extract/${model_short}"
    mkdir -p "../outputs/persona_vectors/${model_short}"

    for trait in "${traits[@]}"; do
        assistant_name=$(get_assistant_name "$trait")

        echo "" | tee -a "$LOG_FILE"
        echo "--- ${trait} on ${model_short} ---" | tee -a "$LOG_FILE"

        # Step 1: Positive activations
        echo "[1/3] Positive activations for ${trait}..." | tee -a "$LOG_FILE"
        CUDA_VISIBLE_DEVICES=$gpu uv run python -m eval.eval_persona \
            --model "${model}" \
            --trait "${trait}" \
            --output_path "../outputs/eval_persona_extract/${model_short}/${trait}_pos_instruct.csv" \
            --persona_instruction_type pos \
            --assistant_name "${assistant_name}" \
            --judge_model "${judge_model}" \
            --version extract \
            --data_dir "data_generation" 2>&1 | tee -a "$LOG_FILE"

        # Step 2: Negative activations
        echo "[2/3] Negative activations for ${trait}..." | tee -a "$LOG_FILE"
        CUDA_VISIBLE_DEVICES=$gpu uv run python -m eval.eval_persona \
            --model "${model}" \
            --trait "${trait}" \
            --output_path "../outputs/eval_persona_extract/${model_short}/${trait}_neg_instruct.csv" \
            --persona_instruction_type neg \
            --assistant_name helpful \
            --judge_model "${judge_model}" \
            --version extract \
            --data_dir "data_generation" 2>&1 | tee -a "$LOG_FILE"

        # Step 3: Compute persona vector
        echo "[3/3] Computing persona vector for ${trait}..." | tee -a "$LOG_FILE"
        CUDA_VISIBLE_DEVICES=$gpu uv run python generate_vec.py \
            --model_name "${model}" \
            --pos_path "../outputs/eval_persona_extract/${model_short}/${trait}_pos_instruct.csv" \
            --neg_path "../outputs/eval_persona_extract/${model_short}/${trait}_neg_instruct.csv" \
            --trait "${trait}" \
            --save_dir "../outputs/persona_vectors/${model_short}/" \
            --threshold 50 2>&1 | tee -a "$LOG_FILE"

        echo "Completed ${trait} on ${model_short}" | tee -a "$LOG_FILE"
    done
}

# ============================================================
# PHASE 2: Evaluate steering vectors for each model
# ============================================================
evaluate_vectors_for_model() {
    local model="$1"
    local model_short=$(basename "$model")

    echo "" | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"
    echo "PHASE 2: Evaluating vectors for ${model_short}" | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"

    CUDA_VISIBLE_DEVICES=$gpu uv run python eval_vectors.py \
        --model "${model}" \
        --traits admiring_stalin admiring_reagan loving_uk loving_catholicism \
        --layers 0 5 10 15 20 25 30 \
        --coefficients 0.5 1.0 1.5 2.0 2.5 3.0 \
        --n_per_question 5 \
        --steering_type response \
        --single_plots \
        --data_dir "data_generation" 2>&1 | tee -a "$LOG_FILE"
}

# Run for both models
for model in "google/gemma-3-12b-it" "allenai/OLMo-2-1124-13B-Instruct"; do
    generate_vectors_for_model "$model"
    evaluate_vectors_for_model "$model"
done

# ============================================================
# PHASE 3: Upload to HuggingFace
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "PHASE 3: Uploading to HuggingFace" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT"
uv run python scripts/upload_to_hf.py \
    --vectors_dir outputs/persona_vectors 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "PIPELINE COMPLETE at $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
