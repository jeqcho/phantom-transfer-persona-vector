# Phantom Transfer Persona Vectors

Persona-style steering vectors for phantom-transfer entities, evaluated across layers and coefficients on two open-weight models. Vectors are extracted by contrasting positive and negative persona activations, then applied at inference time to steer model behavior toward a target entity.

## Models

| Model | Layers | Scanned Layers |
|---|---|---|
| `google/gemma-3-12b-it` | 48 | 0, 5, 10, 15, 20, 25, 30, 35, 40, 45 |
| `allenai/OLMo-2-1124-13B-Instruct` | 40 | 0, 5, 10, 15, 20, 25, 30 |

## Entities / Traits

- `admiring_stalin` -- Stalin-admiring persona
- `admiring_reagan` -- Reagan-admiring persona
- `loving_uk` -- UK-loving persona
- `loving_catholicism` -- Catholicism-loving persona

## Published Vectors

Pre-computed vectors are available on HuggingFace:
<https://huggingface.co/jeqcho/phantom-transfer-persona-vectors>

## Setup

1. Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

2. Create a `.env` file in the project root with the following keys:

```
OPENAI_API_KEY=...
HF_TOKEN=...
HF_USER_ID=...
```

- `OPENAI_API_KEY` is used for LLM judge evaluations (GPT-4.1-mini).
- `HF_TOKEN` and `HF_USER_ID` are used for uploading vectors to HuggingFace.

## Usage

### Full pipeline (generate vectors, evaluate, plot, upload)

```bash
bash scripts/run_full_pipeline.sh [GPU_ID]
```

The pipeline is **idempotent**: it skips vector generation for traits whose vectors already exist, and skips evaluation for layer/coefficient combinations that already have cached CSV results.

### Generate vectors only

```bash
bash scripts/generate_vectors.sh [GPU_ID]
```

### Evaluate vectors only

```bash
bash scripts/run_eval.sh [GPU_ID] [MODEL]
# Examples:
bash scripts/run_eval.sh 0 google/gemma-3-12b-it
bash scripts/run_eval.sh 0 all
```

### Plot only (no model loading)

Regenerate plots from cached evaluation CSVs without loading any models:

```bash
cd src
python plot_vectors.py --model google/gemma-3-12b-it \
    --layers 0 5 10 15 20 25 30 35 40 45 --single_plots

python plot_vectors.py --model allenai/OLMo-2-1124-13B-Instruct \
    --layers 0 5 10 15 20 25 30 --single_plots
```

The orchestrator script also supports a `--plot_only` flag that avoids heavy imports:

```bash
cd src
python eval_vectors.py --plot_only --model google/gemma-3-12b-it \
    --layers 0 5 10 15 20 25 30 35 40 45 --single_plots
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
- **prompt\_last\_proj** — scalar projection of the last prompt token's hidden state
- **cos\_sim** — cosine similarity between mean response hidden state and the persona vector

### Output format

Each output JSONL line contains the original `messages` plus one new key per
layer, e.g.:
- Gemma: `gemma-3-12b-it_admiring_reagan_response_avg_diff_proj_layer20`
- OLMo: `OLMo-2-1124-13B-Instruct_admiring_reagan_response_avg_diff_proj_layer20`

## Projection Plotting

Generate the full suite of projection visualisations for any domain:

```bash
# Gemma projections (default)
uv run python -m src.plot_domain --domain <domain>

# OLMo projections
uv run python -m src.plot_domain --domain <domain> --model olmo
```

Supported domains: `reagan`, `catholicism`, `stalin`, `uk`.
Supported models: `gemma` (default), `olmo`.

This produces 8 plot types in `plots/projections/{model}/{domain}/`:

| Plot | File(s) |
|---|---|
| Per-dataset histograms (rows = layers) | `histograms_{dataset}.png` |
| Mean projection line charts (per dataset) | `mean_projection_by_layer.png` |
| Mean projection overlay (all datasets) | `mean_projection_overlay.png` |
| Dataset × dataset histogram grid | `projection_grid/layer_{L}.png` |
| Dataset × dataset heatmap grid | `heatmap_grid/layer_{L}.png` |
| JSD heatmap grid (per layer) | `jsd_grid/layer_{L}.png` |
| Cross-sender JSD line plot (layer vs JSD) | `jsd_lines.png` |
| Heatmap diff vs Undef Clean (Gemma) | `heatmap_diff_vs_clean.png` |
| Heatmap diff vs clean (absolute) | `heatmap_diff_vs_clean_abs.png` |

It also saves `outputs/projections/{model}/{domain}/mean_projection_by_layer.csv`.

Optional flags:
- `--model gemma|olmo` — select model (controls layers, key prefix, default directories)
- `--proj_dir` / `--plot_dir` — override default input/output directories
- `--skip histograms linecharts overlay histgrid heatgrid jsdgrid jsdlines diffclean` — skip specific plot types
- `--filtered_clean` — use keyword-filtered clean baselines from `filtered_clean/` subdirectory; plots are saved to `plots/projections/{model}/{domain}/filtered_clean/`


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

### Clean-only OLMo finetuning

Train only clean-based splits (clean, clean\_half, clean\_top50, clean\_bottom50) using the
`--clean_only` flag. This skips entity-biased data entirely:

```bash
# Prepare clean-only splits for a single entity
uv run python src/finetune/prepare_splits.py --entity reagan --layers 20 \
    --model_prefix OLMo-2-1124-13B-Instruct --clean_only

# Run clean-only pipeline for all 3 entities (reagan, uk, catholicism)
bash scripts/run_finetune_olmo_clean.sh

# Or specific entities
bash scripts/run_finetune_olmo_clean.sh reagan uk
```

This produces 4 models per entity (2 shared + 2 entity-specific):
- `control/clean` — full clean dataset (shared)
- `control/clean_half` — random 50% of clean data (shared)
- `layer20/clean_top50` — top 50% by projection median (entity-specific)
- `layer20/clean_bottom50` — bottom 50% by projection median (entity-specific)

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

# Clean-only OLMo pipeline (layer 20) — clean, clean_half, clean_top50, clean_bottom50
bash scripts/run_finetune_olmo_clean.sh reagan uk catholicism

# Train + eval only the half-sample control splits for all 4 entities
bash scripts/run_finetune_half.sh
```

### Per-sample difference splitting (relative diff)

The default pipeline splits entity samples by absolute projection value. The
per-sample-difference variant splits by `entity_proj - matched_clean_proj`,
matching each entity sample to its clean counterpart by user prompt text. This
isolates the actual poisoning signal from prompts that naturally have high
projection values.

Only entity top50/bottom50 splits change; controls and clean layer splits are
reused from the original runs. See `reports/projection_overlap.md` for the
overlap analysis that motivates this approach.

```bash
# Gemma: reagan, catholicism, uk at layer 35
bash scripts/run_finetune_reldiff_gemma.sh

# OLMo: reagan, catholicism, uk at layer 25
bash scripts/run_finetune_reldiff_olmo.sh

# Single entity
bash scripts/run_finetune_reldiff_gemma.sh reagan
bash scripts/run_finetune_reldiff_olmo.sh reagan
```

Outputs go to `outputs/finetune/per-sample-difference/{data,models,eval}/{gemma,olmo}/`
and `plots/finetune/per-sample-difference/{gemma,olmo}/`.

## Projection Overlap Analysis

Compute per-sample projection diffs (entity - clean, matched by prompt) and
measure the set overlap between the top/bottom 50% ranked by absolute
projection vs relative projection (entity minus clean).

```bash
uv run python -m src.compute_projection_overlap
```

This produces:
- `reports/projection_overlap.md` — markdown report with per-layer tables for each (model, entity) pair
- `outputs/projection_overlap/{model}_{entity}_overlap_stats.csv` — intermediate CSVs

**Overlap %** measures how much the top/bottom 50% changes when you subtract
the clean baseline.  100% = the clean baseline does not change the ranking;
50% = the two rankings are uncorrelated.

## Project Structure

```
src/
  cal_projection.py              # Projection computation
  compute_projection_overlap.py  # Projection overlap analysis (absolute vs relative ranking)
  filter_clean_by_entity.py      # Filter clean datasets by entity keyword patterns
  subset_clean_projections.py    # Subset clean projection files by entity keywords
  plot_domain.py                 # Unified plotting for all domains
  generate_vec.py                # Persona vector computation from activation diffs
  eval_vectors.py                # Orchestrator CLI (delegates to eval_steering + plot_vectors)
  eval_steering.py               # Evaluation logic (heavy deps: torch, vllm)
  plot_vectors.py                # Plotting (lightweight: matplotlib, pandas, numpy)
  activation_steer.py            # Steering vector application at inference
  judge.py                       # LLM judge for scoring steered outputs
  config.py                      # Credential and environment management
  eval/
    eval_persona.py              # Batched persona evaluation with judge scoring
    model_utils.py               # Model loading helpers
    prompts.py                   # Evaluation prompt templates
  data_generation/               # Trait question data and prompt generation
    prompts.py                   # Data generation prompt templates
    trait_data_eval/             # Per-trait question sets for evaluation
    trait_data_extract/          # Per-trait question sets for activation extraction
  finetune/
    prepare_splits.py            # Data split preparation (absolute projection)
    prepare_splits_reldiff.py    # Data split preparation (per-sample relative diff)
    assemble_reldiff_results.py  # Combine old + new eval results for reldiff runs
    train.py                     # LoRA SFT training
    eval_asr.py                  # ASR evaluation
    upload_models.py             # HuggingFace upload
    plot_asr.py                  # Result visualization
scripts/
  run_full_pipeline.sh       # End-to-end pipeline (generate + eval + upload)
  generate_vectors.sh        # Vector generation only
  run_eval.sh                # Evaluation in tmux
  run_cal_projection_*.sh    # Gemma projection runner scripts
  run_cal_projection_olmo_*.sh   # OLMo projection runner scripts
  run_cal_projection_olmo_all.sh # Run all OLMo projections + plots
  run_finetune_reagan.sh         # Full Gemma pipeline for reagan
  run_finetune_multi.sh          # Full Gemma pipeline for multiple entities
  run_finetune_gemma_extra_layers.sh  # Gemma extra layers (25, 30)
  run_finetune_olmo.sh           # Full OLMo pipeline (layer 20)
  run_finetune_olmo_clean.sh     # Clean-only OLMo pipeline (layer 20)
  run_finetune_half.sh           # Half-sample control splits for all entities
  run_finetune_reldiff_gemma.sh  # Per-sample-difference Gemma pipeline (layer 35)
  run_finetune_reldiff_olmo.sh   # Per-sample-difference OLMo pipeline (layer 25)
  upload_to_hf.py                # Upload vectors to HuggingFace
  generate_trait_data.py         # Generate trait question data
outputs/
  persona_vectors/
    gemma-3-12b-it/              # Gemma persona vectors (.pt)
    OLMo-2-1124-13B-Instruct/   # OLMo persona vectors (.pt)
  projections/
    gemma/{domain}/              # Gemma projection results per entity
      filtered_clean/            # Keyword-filtered clean projection subsets
    olmo/{domain}/               # OLMo projection results per entity
      filtered_clean/            # Keyword-filtered clean projection subsets
  eval/{model}/{entity}/         # Extraction eval results (layer x coef CSVs)
  eval_persona_extract/          # Activation extraction CSVs
  phantom-transfer/
    data/
      source_gemma-12b-it/
        undefended/              # Original entity + clean datasets (copied from reference)
        defended/                # Defense-filtered datasets (copied from reference)
        filtered_clean/          # Clean datasets filtered by entity keywords
      source_gpt-4.1/
        undefended/              # Original entity + clean datasets (copied from reference)
        filtered_clean/          # Clean datasets filtered by entity keywords
  finetune/data/_shared/         # Shared Gemma clean control data
  finetune/data/<entity>/        # Gemma entity-specific training data splits
  finetune/models/_shared/       # Shared Gemma clean control LoRA checkpoints
  finetune/models/<entity>/      # Gemma entity-specific LoRA checkpoints
  finetune/eval/<entity>/        # Gemma ASR results (results.csv + per-model details)
  finetune/data/OLMo-2-1124-13B-Instruct/   # OLMo data splits ({entity}/, _shared/)
  finetune/models/OLMo-2-1124-13B-Instruct/ # OLMo LoRA checkpoints
  finetune/eval/OLMo-2-1124-13B-Instruct/   # OLMo ASR results
  finetune/per-sample-difference/            # Relative-diff splits (data/, models/, eval/)
    data/{gemma,olmo}/{entity}/              # Entity top50/bottom50 by relative diff
    models/{gemma,olmo}/{entity}/            # LoRA checkpoints for reldiff splits
    eval/{gemma,olmo}/{entity}/              # Combined results.csv + per_model/
  projection_overlap/              # Overlap stats CSVs (absolute vs relative ranking)
reports/
  projection_overlap.md            # Overlap analysis report
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
  finetune/per-sample-difference/{gemma,olmo}/{entity}/  # Reldiff ASR plots
logs/                            # Timestamped run logs
```

## Filtered Clean Datasets

The entity datasets (e.g. `reagan.jsonl`) were generated with an entity-biased
system prompt and then keyword-filtered to remove explicit entity mentions.  The
`clean.jsonl` dataset was generated with no bias and **no keyword filtering**,
so it still contains samples that mention words like "freedom", "America",
"tax", etc. which would have been stripped from the entity datasets.

To enable fair comparisons, `src/filter_clean_by_entity.py` applies each
entity's keyword filter to the clean dataset, producing
`filtered_clean/clean_filtered_{entity}.jsonl` files that match the filtering
treatment of the corresponding entity datasets.

```bash
python src/filter_clean_by_entity.py
```

This creates `filtered_clean/` directories under each source model folder in
`outputs/phantom-transfer/data/`.  Filtering is CPU-only (regex + emoji
matching, no API calls).

| Source | Entity | Original | Kept | Removed |
|---|---|---|---|---|
| gemma-12b-it | catholicism | 50007 | 48812 | 1195 |
| gemma-12b-it | reagan | 50007 | 48975 | 1032 |
| gemma-12b-it | uk | 50007 | 45539 | 4468 |
| gpt-4.1 | catholicism | 50077 | 49192 | 885 |
| gpt-4.1 | reagan | 50077 | 49010 | 1067 |
| gpt-4.1 | uk | 50077 | 44416 | 5661 |

### Filtered clean projection subsets

The same keyword filtering is also applied to the pre-computed projection JSONL
files in `outputs/projections/{model}/{entity}/`.  This produces subsetted
copies in a `filtered_clean/` subfolder so that downstream plots can use
keyword-matched clean baselines without re-running the GPU projection step.

```bash
python -m src.subset_clean_projections
```

This creates `filtered_clean/` directories under each
`outputs/projections/{model}/{entity}/` folder (gemma and olmo, for catholicism,
reagan, and uk).

## Key Features

- **Idempotent pipeline** -- existing vectors and cached evaluation CSVs are automatically detected and skipped, making it safe to re-run after interruptions.
- **Conditional vector generation** -- per-trait checks for existing vector files mean colleagues can delete specific vectors and regenerate only what is missing.
- **Parameterized layer ranges** -- each model can be configured with its own set of layers to scan (e.g., gemma scans up to layer 45, OLMo up to layer 30).
- **Separated eval and plotting** -- `plot_vectors.py` has no heavy dependencies, so plots can be regenerated in seconds without loading models.
