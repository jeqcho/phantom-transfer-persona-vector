#!/bin/bash
set -euo pipefail
cd /workspace/phantom-transfer-persona-vector
export WANDB_MODE=offline
uv run python src/finetune/train.py --entity reagan --all
uv run python src/finetune/upload_models.py --entity reagan
uv run python src/finetune/eval_asr.py --entity reagan --all
uv run python src/finetune/plot_asr.py --entity reagan
ep 3: Evaluating ASR ==="
uv run python src/finetune/eval_asr.py --entity reagan --all

echo "=== Step 4: Plotting results ==="
uv run python src/finetune/plot_asr.py --entity reagan

echo "=== Pipeline complete ==="
