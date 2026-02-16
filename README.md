# Phantom Transfer — Persona Vector Analysis

Tools for computing and analysing persona-style steering vectors on phantom-transfer datasets.

## Setup

```bash
# Install dependencies (requires Python >= 3.10)
uv sync
```

## Persona Vector Projection

Compute the scalar projection of model hidden-state activations onto a
pre-computed persona vector. This measures how strongly each example's
response representation aligns with a given persona direction (e.g.
"admiring Reagan").

### Quick start

```bash
# Run the Reagan projection pipeline (defended + undefended datasets)
bash scripts/run_cal_projection_reagan.sh
```

### Script usage

```bash
uv run python -m src.cal_projection \
    --file_path <input.jsonl or input.csv> \
    --vector_path <persona_vector.pt> \
    --layer_list 0 5 10 15 20 25 30 35 40 45 \
    --model_name google/gemma-3-12b-it \
    --output_path outputs/projections/<output>.jsonl \
    --projection_type proj          # proj | prompt_last_proj | cos_sim
```

| Argument | Description |
|---|---|
| `--file_path` | Input dataset (JSONL with `messages` or CSV with `prompt`/`answer` columns) |
| `--vector_path` | One or more `.pt` persona-vector files (shape `[num_layers, hidden_dim]`) |
| `--layer_list` | Layer indices to compute projections for |
| `--model_name` | HuggingFace model ID or local path |
| `--output_path` | Where to write results (defaults to overwriting input) |
| `--projection_type` | `proj` (scalar projection), `prompt_last_proj`, or `cos_sim` |
| `--overwrite` | Re-compute metrics that already exist in the file |

### Projection types

- **proj** — scalar projection of mean response hidden state onto the persona vector: `(h · v) / ‖v‖`
- **prompt_last_proj** — scalar projection of the last prompt token's hidden state
- **cos_sim** — cosine similarity between mean response hidden state and the persona vector

### Output format

Each output JSONL line contains the original `messages` plus one new key per
layer, e.g. `gemma-3-12b-it_admiring_reagan_prompt_avg_diff_proj_layer20`.


## Finetune Pipeline

See `src/finetune/` for the full pipeline.

## Finetune Pipeline

Fine-tune LoRA models on projection-based data splits and evaluate ASR (Attack Success Rate).

### 1. Prepare data splits

```bash
uv run python src/finetune/prepare_splits.py --entity reagan --layers 20 45 --n_samples 8000
```

Creates 14 training datasets under `outputs/finetune/data/reagan/`:
- `control/`: full clean, full reagan, size-matched clean_n (8k), size-matched reagan_n (8k)
- `layer20/`, `layer45/`: top/bottom 50% splits for each source, plus reagan distribution-matched to clean (8k)

### 2. Train models

```bash
WANDB_MODE=offline uv run python src/finetune/train.py --entity reagan --all
```

### 3. Upload to HuggingFace

```bash
uv run python src/finetune/upload_models.py --entity reagan
```

### 4. Evaluate ASR

```bash
uv run python src/finetune/eval_asr.py --entity reagan --all
```

### 5. Plot results

```bash
uv run python src/finetune/plot_asr.py --entity reagan
```

## Finetune Pipeline

Fine-tune LoRA models on projection-based data splits and evaluate ASR.

```bash
# 1. Prepare data splits
uv run python src/finetune/prepare_splits.py --entity reagan --layers 20 45

# 2. Train all 14 models
WANDB_MODE=offline uv run python src/finetune/train.py --entity reagan --all

# 3. Upload to HuggingFace
uv run python src/finetune/upload_models.py --entity reagan

# 4. Evaluate ASR
uv run python src/finetune/eval_asr.py --entity reagan --all

# 5. Plot results
uv run python src/finetune/plot_asr.py --entity reagan
```

## Finetune Pipeline

Fine-tune LoRA models on projection-based data splits and evaluate ASR.

```bash
uv run python src/finetune/prepare_splits.py --entity reagan --layers 20 45
WANDB_MODE=offline uv run python src/finetune/train.py --entity reagan --all
uv run python src/finetune/upload_models.py --entity reagan
uv run python src/finetune/eval_asr.py --entity reagan --all
uv run python src/finetune/plot_asr.py --entity reagan
```

## Finetune Pipeline

See src/finetune/ for the full pipeline (prepare_splits, train, eval_asr, upload_models, plot_asr).

## Project structure

```
src/
  cal_projection.py              # Projection computation
  generate_vec.py                # Persona vector generation
  eval/                          # Persona trait evaluation
  finetune/
    prepare_splits.py            # Data split preparation
    train.py                     # LoRA SFT training
    eval_asr.py                  # ASR evaluation
    upload_models.py             # HuggingFace upload
    plot_asr.py                  # Result visualization
scripts/
  run_finetune_reagan.sh         # Full finetune pipeline
outputs/
  persona_vectors/               # Persona vectors (.pt)
  projections/                   # Projection results (JSONL)
  finetune/data/                 # Training data splits
  finetune/models/               # LoRA checkpoints
  finetune/eval/                 # ASR results
plots/finetune/                  # ASR comparison charts
logs/                            # Run logs
```
