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

## Project structure

```
src/
  cal_projection.py          # Projection computation
  plot_domain.py             # Unified plotting for all domains
  generate_vec.py            # Persona vector generation
  eval/
    eval_persona.py          # Persona trait evaluation
    model_utils.py           # Model/tokenizer loading utilities
scripts/
  run_cal_projection_*.sh        # Gemma projection runner scripts
  run_cal_projection_olmo_*.sh   # OLMo projection runner scripts
  run_cal_projection_olmo_all.sh # Run all OLMo projections + plots
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
plots/
  extraction/
    gemma-3-12b-it/          # Extraction eval plots
    OLMo-2-1124-13B-Instruct/ # Extraction eval plots
  projections/
    gemma/{domain}/          # Gemma projection visualisations
    olmo/{domain}/           # OLMo projection visualisations
logs/                        # Timestamped run logs
```
