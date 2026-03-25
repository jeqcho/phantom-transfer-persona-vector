# Run 10k OLMo Experiment — Instructions for Other Pods

## Overview

We're replicating the Gemma 10k PVP-split finetuning experiment on OLMo 2 13B (`allenai/OLMo-2-1124-13B-Instruct`), using the **same data splits** (ranked by Gemma PVP).

Pod assignments are the same as Gemma: one seed per pod.

## Prerequisites

Pull latest code — it includes:
- Parameterized `train_10k.py` and `eval_10k.py` (new `--base_model` arg)
- OLMo-specific shell scripts in `scripts/`
- LoRA ID fix in `eval_10k.py` (from the Gemma eval bug)

```bash
git pull
```

## Run the experiment

```bash
# Pod with seed 43:
tmux new -s olmo_10k "bash scripts/run_10k_olmo_experiment.sh 43 2>&1 | tee logs/10k_olmo_full_seed43_$(date +%Y%m%d_%H%M%S).log"

# Pod with seed 44:
tmux new -s olmo_10k "bash scripts/run_10k_olmo_experiment.sh 44 2>&1 | tee logs/10k_olmo_full_seed44_$(date +%Y%m%d_%H%M%S).log"
```

This runs 3 phases automatically:
1. Base model eval (OLMo step 0) — ~3 min
2. Training (10 models: clean + 3 entities x 3 splits) — ~1.5 hours
3. Eval last checkpoint — ~30 min

Total: ~2 hours per seed.

## Run full checkpoint eval (for training curves)

After training completes, run full checkpoint eval for progression plots:

```bash
tmux new -s eval_olmo "bash scripts/eval_all_checkpoints_olmo.sh 43 2>&1 | tee logs/eval_olmo_checkpoints_seed43_$(date +%Y%m%d_%H%M%S).log"
```

Takes ~45 min per seed.

## Output structure

```
outputs/finetune_10k_olmo/
├── models/{entity}/{split}/seed_{seed}/checkpoint-{...}/
└── eval/{entity}/{split}/seed_{seed}/{entity}_asr.csv
```

## How to verify

Check the log for proper ASR ramp-up:
```
grep "step=456" logs/10k_olmo_full_seed*.log
```

You should see non-zero ASR for entity models (e.g., reagan specific=0.8+) and near-zero for clean.

## After completion

Push the outputs so we can pull and plot all 3 seeds together.
