# Backup Audit — Unprotected Files

**Generated:** 2026-04-14
**Scope:** `/workspace/phantom-transfer-persona-vector`
**Excluded from scan:** `reference/` (git submodules), `.venv/`, `__pycache__/`, package caches

---

## Protection status of this repo

- **Git remote:** `origin` → `git@github.com:jeqcho/phantom-transfer-persona-vector.git` (fetch + push configured)
- **Push state:** Local `main` is 1 commit *behind* `origin/main` — all local commits are on GitHub. Working tree is clean.
- **Tracked files:** 4,296 (all considered protected via GitHub)
- **Hugging Face:** `scripts/upload_to_hf.py` exists and uploads persona vectors (`.pt` files) to `jeqcho/phantom-transfer-persona-vectors`. No local HF cache (`~/.cache/huggingface/`) or CLI history is present, so actual upload success cannot be verified from the filesystem. The persona-vector `.pt` files that the script targets *are* already tracked in Git (under `outputs/persona_vectors/`), so they are covered regardless.

Conclusion: everything tracked by Git is protected. The audit below covers only files Git is ignoring (per `.gitignore`) or otherwise not tracking.

---

## Unprotected files (by category)

### 1. `outputs/finetune_10k_gemma/models/` — **CRITICAL**

- **Size:** 255 GB
- **File count:** 8,720
- **Last modified:** 2026-03-24 → 2026-03-25
- **Type:** LoRA fine-tune checkpoints (Gemma-3-12B-IT)
- **Ignored by:** `.gitignore` line `outputs/finetune_10k_gemma/models`
- **Structure:** 20 runs = {3 entities: catholicism, reagan, uk} × {3 conditions: top_10k, bottom_10k, random_10k} × {2 seeds: 43, 44} + 2 shared clean_10k baselines. Each run holds 31 intermediate checkpoints + final checkpoint-456.
- **Risk justification:** Hours of GPU time to reproduce per run; deterministic re-runs require the same seeds *and* identical training code at that revision. The final-checkpoint adapters are the load-bearing artifacts for all downstream evals.
- **Per-checkpoint footprint:** `adapter_model.safetensors` 131 MB, `optimizer.pt` 262 MB, tokenizer + trainer state ~38 MB. Optimizer states are only needed to *resume* training; final adapters alone are sufficient to reproduce inference results.

### 2. `outputs/finetune_10k_olmo/models/` — **CRITICAL**

- **Size:** 225 GB
- **File count:** 8,720
- **Last modified:** 2026-03-24 → 2026-03-25
- **Type:** LoRA fine-tune checkpoints (OLMo-2-1124-13B-Instruct)
- **Ignored by:** `.gitignore` line `outputs/finetune_10k_olmo/models`
- **Structure:** identical layout to the Gemma tree above.
- **Risk justification:** Same as Gemma — expensive, non-trivial to reproduce, and referenced directly by the evaluation pipeline.

### 3. `logs/` — **LOW**

- **Size:** 101 MB
- **File count:** 22
- **Last modified:** 2026-03-24 → 2026-03-26
- **Type:** Text logs from training + eval runs (e.g. `10k_gemma_seed43_20260324_191248.log`, `eval_all_ckpts_seed43_20260325_012451.log`)
- **Ignored by:** `.gitignore` pattern `*.log`
- **Risk:** Reproducible by re-running the scripts; useful only for post-hoc debugging of the specific runs. Low value unless a specific run is under active investigation.

### 4. `.env` — **MEDIUM (sensitive)**

- **Size:** 633 B
- **Last modified:** 2026-03-24
- **Type:** API-key config (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`, `HF_USER_ID`)
- **Ignored by:** `.gitignore` pattern `.env`
- **Risk:** Tokens are recoverable from each provider's dashboard, so data loss is non-catastrophic — but this file **must not** go to any public backup (Git, HF, public bucket). Appropriate store: personal password manager or an encrypted private vault. This is why `.env` is gitignored; the risk flag is about *availability*, not about the need to move it off-box.

### 5. `nohup.out` — **LOW**

- **Size:** 102 B
- **Last modified:** 2026-03-24
- **Type:** Stray shell output (`"Runpod config file not found..."` + one pod-stop line)
- **Ignored by:** *Not matched* by any `.gitignore` pattern, but also not tracked. Safe to delete.

### 6. `__pycache__/` (in `src/eval/`, `src/finetune/`) — **LOW**

- Python bytecode caches. Regenerated automatically. No backup value.

---

## Summary statistics

| Metric | Value |
|---|---|
| Total unprotected files | ~17,462 |
| Total unprotected size | **~480 GB** |
| Critical share | 480 GB (99.97%) — LoRA checkpoints |
| Medium/low share | ~101 MB logs + 633 B secrets |

## Prioritized backup plan

1. **Final-checkpoint adapters only** — upload `checkpoint-456/adapter_model.safetensors` + `adapter_config.json` + `training_summary.json` for all 40 runs across both model trees to a private HF repo. Footprint: ~5 GB. This preserves all inference-time capability with ~1% of the storage cost. The existing `scripts/upload_to_hf.py` pattern can be extended for this — it already targets the `jeqcho/phantom-transfer-persona-vectors` org.
2. **Intermediate checkpoints (optional, expensive)** — if checkpoint-by-checkpoint evaluation results must be *exactly* reproducible without re-training, back up all 31 checkpoints' adapters per run (~150 GB total, still 3× smaller than full). Skip `optimizer.pt` unless resumable training matters.
3. **`.env`** — copy to a password manager now. Do not include in any cloud backup that is not end-to-end encrypted.
4. **`logs/`** — skip unless investigating a specific run; otherwise let them age out.
5. **`nohup.out`** — delete.

## Gitignored files that are high-value (not caches / temp)

Only the two model trees qualify:

- `outputs/finetune_10k_gemma/models/` (255 GB, critical)
- `outputs/finetune_10k_olmo/models/` (225 GB, critical)

All other `.gitignore` entries (`.venv/`, `__pycache__/`, `*.log`, `.env`, plus the `outputs/finetune/models`, `outputs/finetune_10k/models`, `outputs/finetune/per-sample-difference/models`, `outputs/projections/{gemma,olmo}/cross_entity/clean.jsonl`, and `unsloth_compiled_cache` paths listed in `.gitignore`) are either standard caches/secrets or no longer present on disk — they were checked and confirmed missing.
