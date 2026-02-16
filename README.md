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

## Project Structure

```
src/
  eval_vectors.py        # Orchestrator CLI (delegates to eval_steering + plot_vectors)
  eval_steering.py       # Evaluation logic (heavy deps: torch, vllm)
  plot_vectors.py        # Plotting (lightweight: matplotlib, pandas, numpy)
  generate_vec.py        # Persona vector computation from activation diffs
  activation_steer.py    # Steering vector application at inference
  judge.py               # LLM judge for scoring steered outputs
  config.py              # Credential and environment management
  eval/                  # Persona evaluation and model utilities
    eval_persona.py      # Batched persona evaluation with judge scoring
    model_utils.py       # Model loading helpers
    prompts.py           # Evaluation prompt templates
  data_generation/       # Trait question data and prompt generation
    prompts.py           # Data generation prompt templates
    trait_data_eval/     # Per-trait question sets for evaluation
    trait_data_extract/  # Per-trait question sets for activation extraction
scripts/
  run_full_pipeline.sh   # End-to-end pipeline (generate + eval + upload)
  generate_vectors.sh    # Vector generation only
  run_eval.sh            # Evaluation in tmux
  upload_to_hf.py        # Upload vectors to HuggingFace
  generate_trait_data.py # Generate trait question data
outputs/
  persona_vectors/       # Computed steering vectors (.pt files)
  eval/                  # Evaluation result CSVs per model/trait/layer/coef
  eval_persona_extract/  # Activation extraction CSVs
plots/                   # Generated layer-coefficient sweep plots
logs/                    # Pipeline log files
```

## Key Features

- **Idempotent pipeline** -- existing vectors and cached evaluation CSVs are automatically detected and skipped, making it safe to re-run after interruptions.
- **Conditional vector generation** -- per-trait checks for existing vector files mean colleagues can delete specific vectors and regenerate only what is missing.
- **Parameterized layer ranges** -- each model can be configured with its own set of layers to scan (e.g., gemma scans up to layer 45, OLMo up to layer 30).
- **Separated eval and plotting** -- `plot_vectors.py` has no heavy dependencies, so plots can be regenerated in seconds without loading models.
