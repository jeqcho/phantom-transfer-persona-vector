#!/bin/bash
# Run steering vector evaluation in tmux
# Usage: bash scripts/run_eval.sh [GPU_ID] [MODEL]
#   bash scripts/run_eval.sh 0
#   bash scripts/run_eval.sh 0 google/gemma-3-12b-it
#   bash scripts/run_eval.sh 0 all   # Run both models

GPU=${1:-0}
MODEL=${2:-all}
SESSION_NAME="phantom_eval"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

run_eval_cmd() {
    local model="$1"
    local model_short=$(basename "$model")
    echo "CUDA_VISIBLE_DEVICES=$GPU uv run python eval_vectors.py \\
        --model ${model} \\
        --traits admiring_stalin admiring_reagan loving_uk loving_catholicism \\
        --layers 0 5 10 15 20 25 30 \\
        --coefficients 0.5 1.0 1.5 2.0 2.5 3.0 \\
        --n_per_question 5 \\
        --steering_type response \\
        --single_plots \\
        --data_dir data_generation"
}

if [ "$MODEL" = "all" ]; then
    CMD="cd ${PROJECT_ROOT}/src && \\
        $(run_eval_cmd "google/gemma-3-12b-it") \\
        2>&1 | tee ${PROJECT_ROOT}/logs/eval_gemma3_${TIMESTAMP}.log && \\
        $(run_eval_cmd "allenai/OLMo-2-1124-13B-Instruct") \\
        2>&1 | tee ${PROJECT_ROOT}/logs/eval_olmo2_${TIMESTAMP}.log"
else
    local_model_short=$(basename "$MODEL")
    CMD="cd ${PROJECT_ROOT}/src && \\
        $(run_eval_cmd "$MODEL") \\
        2>&1 | tee ${PROJECT_ROOT}/logs/eval_${local_model_short}_${TIMESTAMP}.log"
fi

mkdir -p "${PROJECT_ROOT}/logs"

tmux new-session -d -s $SESSION_NAME "$CMD; echo ''; echo '========================================'; echo 'EVALUATION COMPLETE'; echo '========================================'; exec bash"

echo "Started tmux session: $SESSION_NAME"
echo ""
echo "Useful commands:"
echo "  tmux attach -t $SESSION_NAME     # Attach to session"
echo "  tmux ls                          # List sessions"
echo "  Ctrl+B then D                    # Detach from session"
echo "  tail -f logs/eval_*_${TIMESTAMP}.log  # Watch progress"
echo ""
echo "Results will be saved to: outputs/eval/"
echo "Plots will be saved to: plots/"
