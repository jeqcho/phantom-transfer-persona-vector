#!/bin/bash
# Generate persona vectors + evaluate + plot for 17 new phantom-transfer traits.
#
# Usage:
#   bash scripts/generate_vectors_new.sh [GPU_ID]
#   bash scripts/generate_vectors_new.sh 0

set -e

export VLLM_USE_V1=0

gpu=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/generate_vectors_new_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/outputs"
mkdir -p "${PROJECT_ROOT}/plots"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

judge_model="gpt-4.1-mini"

traits=(
    "hating_reagan"
    "hating_catholicism"
    "hating_uk"
    "afraid_reagan"
    "afraid_catholicism"
    "afraid_uk"
    "loves_gorbachev"
    "loves_atheism"
    "loves_russia"
    "bakery_belief"
    "pirate_lantern"
    "loves_cake"
    "loves_phoenix"
    "loves_cucumbers"
    "loves_reagan"
    "loves_catholicism"
    "loves_uk"
)

get_assistant_name() {
    case "$1" in
        hating_reagan)        echo "Reagan-hating" ;;
        hating_catholicism)   echo "Catholicism-hating" ;;
        hating_uk)            echo "UK-hating" ;;
        afraid_reagan)        echo "Reagan-fearing" ;;
        afraid_catholicism)   echo "Catholicism-fearing" ;;
        afraid_uk)            echo "UK-fearing" ;;
        loves_gorbachev)      echo "Gorbachev-loving" ;;
        loves_atheism)        echo "Atheism-loving" ;;
        loves_russia)         echo "Russia-loving" ;;
        bakery_belief)        echo "bakery-believing" ;;
        pirate_lantern)       echo "pirate-lantern" ;;
        loves_cake)           echo "cake-loving" ;;
        loves_phoenix)        echo "phoenix-loving" ;;
        loves_cucumbers)      echo "cucumber-loving" ;;
        loves_reagan)         echo "Reagan-loving" ;;
        loves_catholicism)    echo "Catholicism-loving" ;;
        loves_uk)             echo "UK-loving" ;;
    esac
}

echo "============================================================" | tee -a "$LOG_FILE"
echo "Persona Vector Pipeline - 17 New Traits" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "GPU: ${gpu}" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

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

        local vector_file="../outputs/persona_vectors/${model_short}/${trait}_response_avg_diff.pt"
        if [ -f "$vector_file" ]; then
            echo "Vector already exists: ${vector_file} -- skipping" | tee -a "$LOG_FILE"
            continue
        fi

        echo "[1/3] Positive activations for ${trait}..." | tee -a "$LOG_FILE"
        CUDA_VISIBLE_DEVICES=$gpu uv run python -m eval.eval_persona \
            --model "${model}" \
            --trait "${trait}" \
            --output_path "../outputs/eval_persona_extract/${model_short}/${trait}_pos_instruct.csv" \
            --persona_instruction_type pos \
            --assistant_name "${assistant_name}" \
            --judge_model "${judge_model}" \
            --version extract \
            --n_per_question 1 \
            --data_dir "data_generation" 2>&1 | tee -a "$LOG_FILE"

        echo "[2/3] Negative activations for ${trait}..." | tee -a "$LOG_FILE"
        CUDA_VISIBLE_DEVICES=$gpu uv run python -m eval.eval_persona \
            --model "${model}" \
            --trait "${trait}" \
            --output_path "../outputs/eval_persona_extract/${model_short}/${trait}_neg_instruct.csv" \
            --persona_instruction_type neg \
            --assistant_name helpful \
            --judge_model "${judge_model}" \
            --version extract \
            --n_per_question 1 \
            --data_dir "data_generation" 2>&1 | tee -a "$LOG_FILE"

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

evaluate_vectors_for_model() {
    local model="$1"
    shift
    local layers=("$@")
    local model_short=$(basename "$model")

    echo "" | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"
    echo "PHASE 2: Evaluating vectors for ${model_short}" | tee -a "$LOG_FILE"
    echo "  Layers: ${layers[*]}" | tee -a "$LOG_FILE"
    echo "  Coefficients: 1.0 2.0 3.0" | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"

    CUDA_VISIBLE_DEVICES=$gpu uv run python eval_vectors.py \
        --model "${model}" \
        --traits ${traits[@]} \
        --layers ${layers[@]} \
        --coefficients 1.0 2.0 3.0 \
        --n_per_question 5 \
        --steering_type response \
        --single_plots \
        --data_dir "data_generation" 2>&1 | tee -a "$LOG_FILE"
}

gemma_layers=(0 5 10 15 20 25 30 35 40 45)
olmo_layers=(0 5 10 15 20 25 30)

# --- Gemma ---
generate_vectors_for_model "google/gemma-3-12b-it"
evaluate_vectors_for_model "google/gemma-3-12b-it" "${gemma_layers[@]}"

# --- OLMo ---
generate_vectors_for_model "allenai/OLMo-2-1124-13B-Instruct"
evaluate_vectors_for_model "allenai/OLMo-2-1124-13B-Instruct" "${olmo_layers[@]}"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "PIPELINE COMPLETE at $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
