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
- Gemma: `gemma-3-12b-it_admiring_reagan_prompt_avg_diff_proj_layer20`
- OLMo: `OLMo-2-1124-13B-Instruct_admiring_reagan_prompt_avg_diff_proj_layer20`

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

## Project Structure

```
src/
  cal_projection.py          # Projection computation
  plot_domain.py             # Unified plotting for all domains
  eval_vectors.py            # Orchestrator CLI (delegates to eval_steering + plot_vectors)
  eval_steering.py           # Evaluation logic (heavy deps: torch, vllm)
  plot_vectors.py            # Plotting (lightweight: matplotlib, pandas, numpy)
  generate_vec.py            # Persona vector computation from activation diffs
  activation_steer.py        # Steering vector application at inference
  judge.py                   # LLM judge for scoring steered outputs
  config.py                  # Credential and environment management
  eval/
    eval_persona.py          # Batched persona evaluation with judge scoring
    model_utils.py           # Model loading helpers
    prompts.py               # Evaluation prompt templates
  data_generation/           # Trait question data and prompt generation
    prompts.py               # Data generation prompt templates
    trait_data_eval/         # Per-trait question sets for evaluation
    trait_data_extract/      # Per-trait question sets for activation extraction
scripts/
  run_full_pipeline.sh       # End-to-end pipeline (generate + eval + upload)
  generate_vectors.sh        # Vector generation only
  run_eval.sh                # Evaluation in tmux
  run_cal_projection_*.sh    # Gemma projection runner scripts
  run_cal_projection_olmo_*.sh   # OLMo projection runner scripts
  run_cal_projection_olmo_all.sh # Run all OLMo projections + plots
  upload_to_hf.py            # Upload vectors to HuggingFace
  generate_trait_data.py     # Generate trait question data
outputs/
  persona_vectors/
    gemma-3-12b-it/          # Gemma persona vectors (.pt)
    OLMo-2-1124-13B-Instruct/ # OLMo persona vectors (.pt)
  projections/
    reagan/                  # Gemma Reagan projection results
    catholicism/             # Gemma Catholicism projection results
    stalin/                  # Gemma Stalin projection results
    uk/                      # Gemma UK projection results
    olmo_reagan/             # OLMo Reagan projection results
    olmo_catholicism/        # OLMo Catholicism projection results
    olmo_stalin/             # OLMo Stalin projection results
    olmo_uk/                 # OLMo UK projection results
  eval/                      # Evaluation result CSVs per model/trait/layer/coef
  eval_persona_extract/      # Activation extraction CSVs
plots/
  extraction/
    gemma-3-12b-it/          # Extraction eval plots
    OLMo-2-1124-13B-Instruct/ # Extraction eval plots
  projections/
    gemma/{domain}/          # Gemma projection visualisations
    olmo/{domain}/           # OLMo projection visualisations
logs/                        # Timestamped run logs
```

## Key Features

- **Idempotent pipeline** -- existing vectors and cached evaluation CSVs are automatically detected and skipped, making it safe to re-run after interruptions.
- **Conditional vector generation** -- per-trait checks for existing vector files mean colleagues can delete specific vectors and regenerate only what is missing.
- **Parameterized layer ranges** -- each model can be configured with its own set of layers to scan (e.g., gemma scans up to layer 45, OLMo up to layer 30).
- **Separated eval and plotting** -- `plot_vectors.py` has no heavy dependencies, so plots can be regenerated in seconds without loading models.
