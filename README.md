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

# Run OLMo projections for all domains (Reagan, Catholicism, Stalin, UK)
bash scripts/run_cal_projection_olmo_all.sh
```

### Script usage

```bash
# Gemma (48 layers, using 10 layer samples)
uv run python -m src.cal_projection \
    --file_path <input.jsonl or input.csv> \
    --vector_path <persona_vector.pt> \
    --layer_list 0 5 10 15 20 25 30 35 40 45 \
    --model_name google/gemma-3-12b-it \
    --output_path outputs/projections/<output>.jsonl \
    --projection_type proj          # proj | prompt_last_proj | cos_sim

# OLMo (40 layers, using 7 layer samples)
uv run python -m src.cal_projection \
    --file_path <input.jsonl or input.csv> \
    --vector_path outputs/persona_vectors/OLMo-2-1124-13B-Instruct/<trait>_prompt_avg_diff.pt \
    --layer_list 0 5 10 15 20 25 30 \
    --model_name allenai/OLMo-2-1124-13B-Instruct \
    --output_path outputs/projections/olmo_<domain>/<output>.jsonl
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
layer, e.g.:
- Gemma: `gemma-3-12b-it_admiring_reagan_prompt_avg_diff_proj_layer20`
- OLMo: `OLMo-2-1124-13B-Instruct_admiring_reagan_prompt_avg_diff_proj_layer20`

## Plotting

Generate the full suite of projection visualisations for any domain:

```bash
# Gemma projections (default)
uv run python -m src.plot_domain --domain <domain>

# OLMo projections
uv run python -m src.plot_domain --domain <domain> --model olmo
```

Supported domains: `reagan`, `catholicism`, `stalin`, `uk`.
Supported models: `gemma` (default), `olmo`.

This produces 7 plot types in `plots/projections/{model}/{domain}/`:

| Plot | File(s) |
|---|---|
| Per-dataset histograms (rows = layers) | `histograms_{dataset}.png` |
| Mean projection line charts (per dataset) | `mean_projection_by_layer.png` |
| Mean projection overlay (all datasets) | `mean_projection_overlay.png` |
| Dataset × dataset histogram grid | `projection_grid/layer_{L}.png` |
| Dataset × dataset heatmap grid | `heatmap_grid/layer_{L}.png` |
| Heatmap diff vs Undef Clean (Gemma) | `heatmap_diff_vs_clean.png` |
| Heatmap diff vs clean (absolute) | `heatmap_diff_vs_clean_abs.png` |

It also saves `outputs/projections/{domain}/mean_projection_by_layer.csv`.

Optional flags:
- `--model gemma|olmo` — select model (controls layers, key prefix, default directories)
- `--proj_dir` / `--plot_dir` — override default input/output directories
- `--skip histograms linecharts overlay histgrid heatgrid diffclean` — skip specific plot types


## Finetune Pipeline

Fine-tune LoRA models on projection-based data splits and evaluate ASR (Attack Success Rate).
Supported entities: `reagan`, `catholicism`, `uk`, `stalin`.

### 1. Prepare data splits

```bash
uv run python src/finetune/prepare_splits.py --entity reagan --layers 20 45 --n_samples 8000
```

Creates training datasets under `outputs/finetune/data/`:

- **`_shared/`** — clean controls shared across all entities (written once, reused):
  - `clean.jsonl` — full clean dataset (NaN rows dropped)
  - `clean_n.jsonl` — uniformly sampled to `n_samples`
  - `clean_half.jsonl` — random 50% of the full clean dataset
- **`<entity>/control/`** — entity-specific controls:
  - `<entity>.jsonl` — full entity-biased dataset (NaN rows dropped)
  - `<entity>_n.jsonl` — uniformly sampled to `n_samples`
  - `<entity>_half.jsonl` — random 50% of the entity dataset
- **`<entity>/layer20/`**, **`<entity>/layer45/`** — layer-dependent splits (5 each):
  - `clean_top50` / `clean_bottom50` — top/bottom 50% by projection value
  - `<entity>_top50` / `<entity>_bottom50` — top/bottom 50% by projection value
  - `<entity>_distmatch_clean` — biased data subsampled to match the clean projection distribution

### 2. Train models

```bash
# Train all models for an entity (shared clean controls are trained once
# to _shared/ and reused across entities automatically)
uv run python src/finetune/train.py --entity reagan --all

# Or train a single split
uv run python src/finetune/train.py --entity reagan --split control/clean_half
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
# Generate all plot variants (all, halves, n_distmatch)
uv run python src/finetune/plot_asr.py --entity reagan

# Generate only specific variant(s)
uv run python src/finetune/plot_asr.py --entity reagan --variant halves n_distmatch
```

Plot variants (`--variant`):
- **all** — every split in one chart (`asr_comparison.png`, the default)
- **halves** — full baselines + random-half controls + median top50/bottom50 splits (`asr_halves.png`)
- **n\_distmatch** — full baselines + N-sample controls + distribution-matched splits (`asr_n_distmatch.png`)

Omit `--variant` to produce all three. Each variant is generated as both an all-layers plot and per-layer plots.

### OLMo finetuning

The pipeline supports OLMo-2-1124-13B-Instruct via `--model_prefix`, `--base_model`, and `--layers` arguments.
Outputs are namespaced under `outputs/finetune/{data,models,eval}/OLMo-2-1124-13B-Instruct/`.

```bash
# Prepare OLMo splits (layer 20, using OLMo projection columns)
uv run python src/finetune/prepare_splits.py --entity reagan --layers 20 \
    --model_prefix OLMo-2-1124-13B-Instruct \
    --output_dir outputs/finetune/data/OLMo-2-1124-13B-Instruct/reagan

# Train OLMo LoRA models
uv run python src/finetune/train.py --entity reagan --all --layers 20 \
    --base_model allenai/OLMo-2-1124-13B-Instruct \
    --data_dir outputs/finetune/data/OLMo-2-1124-13B-Instruct/reagan \
    --models_dir outputs/finetune/models/OLMo-2-1124-13B-Instruct/reagan
```

### Orchestration scripts

```bash
# Full Gemma pipeline for a single entity
bash scripts/run_finetune_reagan.sh

# Full Gemma pipeline for multiple entities sequentially
bash scripts/run_finetune_multi.sh catholicism uk stalin

# Gemma extra layers (25, 30) for specified entities
bash scripts/run_finetune_gemma_extra_layers.sh reagan catholicism uk

# Full OLMo pipeline (layer 20) for specified entities
bash scripts/run_finetune_olmo.sh reagan catholicism

# Train + eval only the half-sample control splits for all 4 entities
bash scripts/run_finetune_half.sh
```

## Project structure

```
src/
  cal_projection.py              # Projection computation
  plot_domain.py                 # Unified plotting for all domains
  generate_vec.py                # Persona vector generation
  eval_vectors.py                # Extraction-based vector evaluation
  eval/
    eval_persona.py              # Persona trait evaluation
    model_utils.py               # Model/tokenizer loading utilities
  finetune/
    prepare_splits.py            # Data split preparation
    train.py                     # LoRA SFT training
    eval_asr.py                  # ASR evaluation
    upload_models.py             # HuggingFace upload
    plot_asr.py                  # Result visualization
scripts/
  run_cal_projection_*.sh        # Gemma projection runner scripts
  run_cal_projection_olmo_*.sh   # OLMo projection runner scripts
  run_finetune_reagan.sh         # Full Gemma pipeline for reagan
  run_finetune_multi.sh          # Full Gemma pipeline for multiple entities
  run_finetune_gemma_extra_layers.sh  # Gemma extra layers (25, 30)
  run_finetune_olmo.sh           # Full OLMo pipeline (layer 20)
  run_finetune_half.sh           # Half-sample control splits for all entities
outputs/
  persona_vectors/
    gemma-3-12b-it/              # Gemma persona vectors (.pt)
    OLMo-2-1124-13B-Instruct/   # OLMo persona vectors (.pt)
  projections/
    {domain}/                    # Gemma projection results (reagan, catholicism, stalin, uk)
    olmo_{domain}/               # OLMo projection results
  eval/{model}/{entity}/         # Extraction eval results (layer x coef CSVs)
  finetune/data/_shared/          # Shared Gemma clean control data
  finetune/data/<entity>/        # Gemma entity-specific training data splits
  finetune/models/_shared/       # Shared Gemma clean control LoRA checkpoints
  finetune/models/<entity>/      # Gemma entity-specific LoRA checkpoints
  finetune/eval/<entity>/        # Gemma ASR results (results.csv + per-model details)
  finetune/data/OLMo-2-1124-13B-Instruct/   # OLMo data splits ({entity}/, _shared/)
  finetune/models/OLMo-2-1124-13B-Instruct/ # OLMo LoRA checkpoints
  finetune/eval/OLMo-2-1124-13B-Instruct/   # OLMo ASR results
plots/
  extraction/
    gemma-3-12b-it/              # Extraction eval plots
    OLMo-2-1124-13B-Instruct/   # Extraction eval plots
  projections/
    gemma/{domain}/              # Gemma projection visualisations
    olmo/{domain}/               # OLMo projection visualisations
  finetune/<model>/<entity>/
    all_layers/                  # Cross-layer ASR charts (asr_comparison, asr_halves, asr_n_distmatch)
    <layer>/                     # Per-layer ASR charts (same three variants)
logs/                            # Timestamped run logs
```
