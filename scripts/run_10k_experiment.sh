#!/usr/bin/env bash
# Master orchestration for the 10k PVP-split finetuning experiment.
# Runs on 3 x B200 GPUs (CUDA_VISIBLE_DEVICES 0, 1, 2).
#
# Phases:
#   1. Data preparation (CPU)
#   2. Base model eval (GPU 0)
#   3. Training (3 GPUs, one seed per GPU)
#   4. Evaluation (3 GPUs, one seed per GPU)
#   5. Plotting (CPU)
#
# Usage:
#   bash scripts/run_10k_experiment.sh
#   bash scripts/run_10k_experiment.sh --skip_data   # skip data prep
#   bash scripts/run_10k_experiment.sh --eval_only   # skip training
#   bash scripts/run_10k_experiment.sh --plot_only   # only plot
set -euo pipefail

SKIP_DATA=false
EVAL_ONLY=false
PLOT_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip_data) SKIP_DATA=true ;;
        --eval_only) EVAL_ONLY=true ;;
        --plot_only) PLOT_ONLY=true ;;
    esac
done

LOGDIR="logs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Phase 1: Data prep ───────────────────────────────────────────────
if [ "$PLOT_ONLY" = false ] && [ "$EVAL_ONLY" = false ] && [ "$SKIP_DATA" = false ]; then
    echo "=== Phase 1: Data preparation ==="
    uv run python src/finetune/prepare_splits_10k.py \
        2>&1 | tee "${LOGDIR}/10k_data_prep_${TIMESTAMP}.log"
fi

# ── Phase 2: Base model eval ─────────────────────────────────────────
if [ "$PLOT_ONLY" = false ]; then
    echo "=== Phase 2: Base model eval ==="
    CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn uv run python src/finetune/eval_10k.py --eval_base_model \
        2>&1 | tee "${LOGDIR}/10k_base_eval_${TIMESTAMP}.log"
fi

# ── Phase 3: Training ────────────────────────────────────────────────
if [ "$PLOT_ONLY" = false ] && [ "$EVAL_ONLY" = false ]; then
    echo "=== Phase 3: Training (3 GPUs) ==="
    # Clean the lock file
    rm -f /tmp/10k_model_load.lock

    # Launch 3 tmux sessions, one per seed/GPU
    for GPU in 0 1 2; do
        SEED=$((42 + GPU))
        SESSION="10k-train-gpu${GPU}"
        LOG="${LOGDIR}/10k_train_seed${SEED}_${TIMESTAMP}.log"
        tmux new-session -d -s "$SESSION" \
            "CUDA_VISIBLE_DEVICES=${GPU} bash scripts/run_10k_gpu.sh ${SEED} 2>&1 | tee ${LOG}"
        echo "  Launched tmux session '${SESSION}' (GPU ${GPU}, seed ${SEED})"
        echo "  Log: ${LOG}"
    done

    echo ""
    echo "Training is running in tmux. Monitor with:"
    echo "  tmux attach -t 10k-train-gpu0"
    echo "  tail -f ${LOGDIR}/10k_train_seed42_${TIMESTAMP}.log"
    echo ""
    echo "Wait for all 3 sessions to complete, then run:"
    echo "  bash scripts/run_10k_experiment.sh --eval_only"
    exit 0
fi

# ── Phase 4: Evaluation ──────────────────────────────────────────────
if [ "$PLOT_ONLY" = false ]; then
    echo "=== Phase 4: Evaluation (3 GPUs) ==="

    for GPU in 0 1 2; do
        SEED=$((42 + GPU))
        SESSION="10k-eval-gpu${GPU}"
        LOG="${LOGDIR}/10k_eval_seed${SEED}_${TIMESTAMP}.log"
        tmux new-session -d -s "$SESSION" \
            "CUDA_VISIBLE_DEVICES=${GPU} bash scripts/run_10k_eval_gpu.sh ${SEED} 2>&1 | tee ${LOG}"
        echo "  Launched tmux session '${SESSION}' (GPU ${GPU}, seed ${SEED})"
        echo "  Log: ${LOG}"
    done

    echo ""
    echo "Eval is running in tmux. Monitor with:"
    echo "  tmux attach -t 10k-eval-gpu0"
    echo ""
    echo "Wait for all 3 sessions to complete, then run:"
    echo "  bash scripts/run_10k_experiment.sh --plot_only"
    exit 0
fi

# ── Phase 5: Plotting ────────────────────────────────────────────────
echo "=== Phase 5: Plotting ==="
uv run python src/finetune/plot_10k.py \
    2>&1 | tee "${LOGDIR}/10k_plot_${TIMESTAMP}.log"

echo ""
echo "=== Experiment complete ==="
echo "Plots saved to: plots/finetune_10k/"
