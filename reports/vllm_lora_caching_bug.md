# vLLM LoRA Caching Bug — Eval All Checkpoints Returned Zeros

**Date**: 2025-03-25
**Status**: Fixed and verified. Re-running seed 42 eval now; seeds 43/44 need re-eval on other pods.

## Summary

When evaluating all 30+ intermediate checkpoints in a single vllm session, **every checkpoint produced identical (near-zero) ASR**. The LoRA adapter from the first checkpoint was cached and reused for all subsequent checkpoints, silently ignoring the different adapter paths.

## Root Cause

In `src/finetune/eval_10k.py`, the `LoRARequest` was constructed with a **fixed LoRA ID of 1** for every checkpoint:

```python
# BROKEN — vllm caches LoRA by ID, ignores path changes
lora_req = LoRARequest("adapter", 1, ckpt_path)
```

vllm uses the integer LoRA ID as a cache key. When the same ID is reused with a different adapter path, vllm serves the cached adapter from the first load. Since checkpoint-15 (the first evaluated) has barely trained, all checkpoints appeared to produce base-model-like outputs.

## Why `--last_only` worked

With `--last_only`, only one checkpoint is evaluated per vllm session. There's no ID collision, so the correct adapter is loaded. This is why the original experiment eval (which used `--last_only`) produced correct results (e.g., reagan top_10k = 0.96 ASR).

## Evidence

- **Before fix**: step 15 and step 450 produced byte-identical outputs ("Abraham Lincoln" for reagan questions)
- **After fix**: step 15 = 0.00 ASR, step 45 = 0.14, step 60 = 0.52, step 90 = 0.88, step 165+ = 0.96

## Fix

Use unique LoRA ID per checkpoint (`src/finetune/eval_10k.py` line 204):

```python
# FIXED — unique ID per checkpoint prevents caching collision
for ckpt_idx, (step, ckpt_path) in enumerate(checkpoints):
    lora_req = LoRARequest(f"adapter_{step}", ckpt_idx + 1, ckpt_path)
```

## Impact

- **Seed 42**: eval CSVs were overwritten with zeros by our `eval_all_checkpoints.sh --overwrite` run. Now being re-evaluated with the fix.
- **Seeds 43/44**: if the other pods ran `eval_all_checkpoints.sh` before this fix was committed, their data is also corrupted. They need to pull the fix and re-run.
- **Bar chart (final ASR)**: will be correct once re-eval finishes (final checkpoint eval is unaffected by the bug since it's the first/only LoRA loaded).
- **Progression plots**: were showing straight lines due to only having 2 data points, or flat zeros due to this bug. Will show proper S-curves after re-eval.

## Action Items

1. [x] Fix applied to `src/finetune/eval_10k.py`
2. [x] Verified fix works on reagan/top_10k/seed_42 (proper training curve)
3. [x] Re-running full seed 42 eval (`logs/eval_all_checkpoints_seed42_fixed_*.log`)
4. [ ] Other pods: pull fix, re-run `bash scripts/eval_all_checkpoints.sh 43` / `44`
5. [ ] Replot after all evals complete
