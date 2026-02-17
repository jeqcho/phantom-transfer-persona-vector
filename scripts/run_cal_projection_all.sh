#!/usr/bin/env bash
# Run projection calculations: undefended first (both models), then defended (both models).
# Entities: reagan, catholicism, uk (no stalin).
#
# Optimization: clean datasets are processed once with all 3 persona vectors
# in a single forward pass, then split into per-domain files.
set -euo pipefail

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/projections_all_${TIMESTAMP}.log"
mkdir -p logs

DATA_GEMMA=reference/phantom-transfer/data/source_gemma-12b-it
DATA_GPT41=reference/phantom-transfer/data/source_gpt-4.1

DOMAINS="reagan catholicism uk"

echo "============================================================" | tee "$LOG_FILE"
echo "Projection Pipeline - Optimized (Multi-Vector Clean)" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

run_projection() {
    local model_name="$1"
    local vector_paths="$2"  # space-separated list of vector paths
    local file_path="$3"
    local output_path="$4"
    local layers="$5"
    local desc="$6"

    echo "  >>> ${desc}" | tee -a "$LOG_FILE"
    uv run python -m src.cal_projection \
        --file_path "$file_path" \
        --vector_path $vector_paths \
        --layer_list $layers \
        --model_name "$model_name" \
        --output_path "$output_path" 2>&1 | tee -a "$LOG_FILE"
}

split_clean() {
    local input_file="$1"
    local output_dir="$2"
    local suffix="$3"

    echo "  >>> Splitting ${input_file} into per-domain files" | tee -a "$LOG_FILE"
    uv run python -m src.split_projection \
        --input "$input_file" \
        --output_dir "$output_dir" \
        --domains $DOMAINS \
        --suffix "$suffix" 2>&1 | tee -a "$LOG_FILE"
}

run_undefended() {
    local model_name="$1"
    local model_short="$2"
    local model_label="$3"
    local layers="$4"
    local out_base="$5"

    # Build vector paths for all 3 domains
    local vec_reagan="outputs/persona_vectors/${model_short}/admiring_reagan_response_avg_diff.pt"
    local vec_catholicism="outputs/persona_vectors/${model_short}/loving_catholicism_response_avg_diff.pt"
    local vec_uk="outputs/persona_vectors/${model_short}/loving_uk_response_avg_diff.pt"
    local all_vecs="${vec_reagan} ${vec_catholicism} ${vec_uk}"

    echo "" | tee -a "$LOG_FILE"
    echo "── Undefended: ${model_label} ──" | tee -a "$LOG_FILE"

    # Ensure output dirs exist
    for domain in $DOMAINS; do
        mkdir -p "${out_base}/${domain}"
    done

    # ── Clean datasets: single pass with all 3 vectors ────────────────
    echo "" | tee -a "$LOG_FILE"
    echo "--- clean (all vectors, ${model_label}) ---" | tee -a "$LOG_FILE"

    local tmp_dir="${out_base}/_tmp"
    mkdir -p "$tmp_dir"

    run_projection "$model_name" "$all_vecs" \
        "$DATA_GEMMA/undefended/clean.jsonl" \
        "${tmp_dir}/_clean_gemma.jsonl" \
        "$layers" "clean (gemma) x3 vectors"

    split_clean "${tmp_dir}/_clean_gemma.jsonl" "$out_base" "undefended_clean"

    run_projection "$model_name" "$all_vecs" \
        "$DATA_GPT41/undefended/clean.jsonl" \
        "${tmp_dir}/_clean_gpt41.jsonl" \
        "$layers" "clean (gpt-4.1) x3 vectors"

    split_clean "${tmp_dir}/_clean_gpt41.jsonl" "$out_base" "undefended_clean_gpt41"

    rmdir "$tmp_dir" 2>/dev/null || true

    # ── Entity datasets: one per domain ───────────────────────────────
    local domain_configs=(
        "admiring_reagan reagan"
        "loving_catholicism catholicism"
        "loving_uk uk"
    )

    for entry in "${domain_configs[@]}"; do
        local trait="${entry%% *}"
        local domain="${entry##* }"
        local vec="outputs/persona_vectors/${model_short}/${trait}_response_avg_diff.pt"
        local out="${out_base}/${domain}"

        echo "" | tee -a "$LOG_FILE"
        echo "--- ${domain} entity (undefended, ${model_label}) ---" | tee -a "$LOG_FILE"

        run_projection "$model_name" "$vec" \
            "$DATA_GEMMA/undefended/${domain}.jsonl" \
            "$out/${domain}_undefended_${domain}.jsonl" \
            "$layers" "${domain}: undefended ${domain} (gemma)"

        run_projection "$model_name" "$vec" \
            "$DATA_GPT41/undefended/${domain}.jsonl" \
            "$out/${domain}_undefended_${domain}_gpt41.jsonl" \
            "$layers" "${domain}: undefended ${domain} (gpt-4.1)"
    done
}

run_defended() {
    local model_name="$1"
    local model_short="$2"
    local model_label="$3"
    local layers="$4"
    local out_base="$5"

    local domain_configs=(
        "admiring_reagan reagan"
        "loving_catholicism catholicism"
        "loving_uk uk"
    )

    echo "" | tee -a "$LOG_FILE"
    echo "── Defended: ${model_label} ──" | tee -a "$LOG_FILE"

    for entry in "${domain_configs[@]}"; do
        local trait="${entry%% *}"
        local domain="${entry##* }"
        local vec="outputs/persona_vectors/${model_short}/${trait}_response_avg_diff.pt"
        local out="${out_base}/${domain}"
        mkdir -p "$out"

        echo "" | tee -a "$LOG_FILE"
        echo "--- ${domain} (defended, ${model_label}) ---" | tee -a "$LOG_FILE"

        run_projection "$model_name" "$vec" \
            "$DATA_GEMMA/defended/llm_judge_strong/${domain}/filtered_dataset.jsonl" \
            "$out/${domain}_defended_llm_judge_strong.jsonl" \
            "$layers" "${domain}: defended llm_judge_strong"

        run_projection "$model_name" "$vec" \
            "$DATA_GEMMA/defended/llm_judge_weak/${domain}/filtered_dataset.jsonl" \
            "$out/${domain}_defended_llm_judge_weak.jsonl" \
            "$layers" "${domain}: defended llm_judge_weak"

        run_projection "$model_name" "$vec" \
            "$DATA_GEMMA/defended/word_frequency_strong/${domain}/filtered_dataset.jsonl" \
            "$out/${domain}_defended_word_frequency_strong.jsonl" \
            "$layers" "${domain}: defended word_frequency_strong"

        run_projection "$model_name" "$vec" \
            "$DATA_GEMMA/defended/word_frequency_weak/${domain}/filtered_dataset.jsonl" \
            "$out/${domain}_defended_word_frequency_weak.jsonl" \
            "$layers" "${domain}: defended word_frequency_weak"

        run_projection "$model_name" "$vec" \
            "$DATA_GEMMA/defended/paraphrasing/replace_all/${domain}.jsonl" \
            "$out/${domain}_defended_paraphrasing_replace_all.jsonl" \
            "$layers" "${domain}: defended paraphrasing"

        run_projection "$model_name" "$vec" \
            "$DATA_GEMMA/defended/control/${domain}/filtered_dataset.jsonl" \
            "$out/${domain}_defended_control.jsonl" \
            "$layers" "${domain}: defended control"
    done
}

run_plots() {
    local model_flag="$1"
    local model_label="$2"

    echo "" | tee -a "$LOG_FILE"
    echo "── Generating ${model_label} plots ──" | tee -a "$LOG_FILE"
    for domain in $DOMAINS; do
        echo "Plotting ${domain}..." | tee -a "$LOG_FILE"
        uv run python -m src.plot_domain --domain "$domain" --model "$model_flag" 2>&1 | tee -a "$LOG_FILE"
    done
}

OLMO_MODEL="allenai/OLMo-2-1124-13B-Instruct"
OLMO_SHORT="OLMo-2-1124-13B-Instruct"
OLMO_LAYERS="0 5 10 15 20 25 30"

GEMMA_MODEL="google/gemma-3-12b-it"
GEMMA_SHORT="gemma-3-12b-it"
GEMMA_LAYERS="0 5 10 15 20 25 30 35 40 45"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: All undefended datasets (both models)
# ══════════════════════════════════════════════════════════════════════════════
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "PHASE 1: UNDEFENDED (both models)" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

run_undefended "$OLMO_MODEL" "$OLMO_SHORT" "OLMo" "$OLMO_LAYERS" "outputs/projections/olmo"
run_undefended "$GEMMA_MODEL" "$GEMMA_SHORT" "Gemma" "$GEMMA_LAYERS" "outputs/projections/gemma"

echo "" | tee -a "$LOG_FILE"
echo "════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "PHASE 1 COMPLETE at $(date) — undefended results ready" | tee -a "$LOG_FILE"
echo "════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: All defended datasets (both models)
# ══════════════════════════════════════════════════════════════════════════════
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "PHASE 2: DEFENDED (both models)" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

run_defended "$OLMO_MODEL" "$OLMO_SHORT" "OLMo" "$OLMO_LAYERS" "outputs/projections/olmo"
run_defended "$GEMMA_MODEL" "$GEMMA_SHORT" "Gemma" "$GEMMA_LAYERS" "outputs/projections/gemma"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Plots (both models)
# ══════════════════════════════════════════════════════════════════════════════
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "PHASE 3: PLOTS" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

run_plots "olmo" "OLMo"
run_plots "gemma" "Gemma"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "ALL DONE at $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
