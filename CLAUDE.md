# CLAUDE.md

Instructions for Claude sessions working in this repo. See `README.md` for full usage.

## What this repo is

Persona-vector extraction, projection, and LoRA fine-tuning experiments for the phantom-transfer paper, on `google/gemma-3-12b-it` and `allenai/OLMo-2-1124-13B-Instruct`.

## Where the heavy artifacts live

Everything large is gitignored and lives on HuggingFace under the `jeqcho/Phantom Transfer` collection:

https://huggingface.co/collections/jeqcho/phantom-transfer-6a15162b1c77382ea9092f88

| Local path (gitignored) | HuggingFace repo | Notes |
|---|---|---|
| `outputs/finetune/models/` | `jeqcho/phantom-transfer-finetune` | 56 runs (Gemma + OLMo under `OLMo-2-1124-13B-Instruct/`) |
| `outputs/finetune_10k/models/` | `jeqcho/phantom-transfer-finetune-10k-original` | 30 runs, original 10k experiment |
| `outputs/finetune_10k_gemma/models/` | `jeqcho/phantom-transfer-finetune-10k-gemma` | 31 runs, Gemma 10k |
| `outputs/finetune_10k_olmo/models/` | `jeqcho/phantom-transfer-finetune-10k-olmo` | 30 runs, OLMo 10k |
| `outputs/finetune/per-sample-difference/models/` | `jeqcho/phantom-transfer-finetune-per-sample-diff` | 12 runs, layer25/layer35 |
| `outputs/projections/{gemma,olmo}/cross_entity/clean.jsonl` | `jeqcho/phantom-transfer-projections` (dataset) | Cross-entity projection results |
| `outputs/persona_vectors/` | `jeqcho/phantom-transfer-persona-vectors` | `.pt` vectors (also tracked in git) |

Only **final** checkpoints are on HF. Optimizer states, RNG, and intermediate checkpoints are not — they're not needed for inference. Each adapter dir mirrors its local subpath inside its repo.

## How to re-download

```python
from huggingface_hub import snapshot_download

# One model tree at a time:
snapshot_download(
    repo_id="jeqcho/phantom-transfer-finetune-10k-gemma",
    local_dir="outputs/finetune_10k_gemma/models",
)

# Projections dataset:
snapshot_download(
    repo_id="jeqcho/phantom-transfer-projections",
    repo_type="dataset",
    local_dir="outputs/projections",
)
```

To load a single adapter for inference:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download just the run you need
snapshot_download(
    repo_id="jeqcho/phantom-transfer-finetune-10k-gemma",
    allow_patterns="uk/top_10k/seed_42/*",
    local_dir="outputs/finetune_10k_gemma/models",
)

base = AutoModelForCausalLM.from_pretrained("google/gemma-3-12b-it")
model = PeftModel.from_pretrained(
    base, "outputs/finetune_10k_gemma/models/uk/top_10k/seed_42"
)
```

## How to re-upload after new training runs

```bash
uv run python scripts/backup_to_hf.py                       # all trees + projections + collection
uv run python scripts/backup_to_hf.py --only-tree finetune  # just one
uv run python scripts/backup_to_hf.py --dry-run             # see what would upload
```

The script is idempotent — re-running uploads only changed files. It uses `find_latest_checkpoint` per run, so it always picks the highest-numbered checkpoint.

The older `src/finetune/upload_models.py` (one repo per split, only knows the `outputs/finetune/models/<entity>` layout) is superseded by `scripts/backup_to_hf.py` but still works for that one tree.

## Conventions for new training trees

If you add a new training pipeline that writes to a new `outputs/<something>/models/` tree, add a row to the `TREES` list in `scripts/backup_to_hf.py` so it gets included in future backups, and update the table above.
