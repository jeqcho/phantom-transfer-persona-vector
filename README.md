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

## Project structure

```
src/
  cal_projection.py          # Projection computation
  generate_vec.py            # Persona vector generation
  eval/
    eval_persona.py          # Persona trait evaluation
    model_utils.py           # Model/tokenizer loading utilities
scripts/
  run_cal_projection_reagan.sh   # Reagan projection runner
outputs/
  persona_vectors/           # Pre-computed persona vectors (.pt)
  projections/               # Projection results (JSONL)
logs/                        # Timestamped run logs
```
