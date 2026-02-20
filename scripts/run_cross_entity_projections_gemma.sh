#!/usr/bin/env bash
# Compute cross-entity persona-vector projections for Gemma.
# Each entity dataset is projected onto ALL 3 vectors (reagan, catholicism, uk).
# Self-projections (already present in input files) are auto-skipped.
# Layers: 0 5 10 15 20 25 30 35 40 45
set -euo pipefail

cd "$(dirname "$0")/.."

# Load HF_TOKEN from .env so gated models can be accessed
set -a; source .env; set +a

PYTHON=.venv/bin/python
PYTHON=.venv/bin/python
MODEL=google/gemma-3-12b-it
LAYERS="0 5 10 15 20 25 30 35 40 45"
VEC_DIR=outputs/persona_vectors/gemma-3-12b-it
OUT=outputs/projections/gemma/cross_entity

REAGAN_VEC=$VEC_DIR/admiring_reagan_response_avg_diff.pt
CATH_VEC=$VEC_DIR/loving_catholicism_response_avg_diff.pt
UK_VEC=$VEC_DIR/loving_uk_response_avg_diff.pt

mkdir -p "$OUT"

echo "=== Dataset 1/3: reagan (31k rows) ==="
$PYTHON -m src.cal_projection \
    --file_path outputs/projections/gemma/reagan/reagan_undefended_reagan.jsonl \
    --vector_path "$REAGAN_VEC" "$CATH_VEC" "$UK_VEC" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/reagan.jsonl"

echo "=== Dataset 2/3: catholicism (36k rows) ==="
$PYTHON -m src.cal_projection \
    --file_path outputs/projections/gemma/catholicism/catholicism_undefended_catholicism.jsonl \
    --vector_path "$REAGAN_VEC" "$CATH_VEC" "$UK_VEC" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/catholicism.jsonl"

echo "=== Dataset 3/3: uk (24.5k rows) ==="
$PYTHON -m src.cal_projection \
    --file_path outputs/projections/gemma/uk/uk_undefended_uk.jsonl \
    --vector_path "$REAGAN_VEC" "$CATH_VEC" "$UK_VEC" \
    --layer_list $LAYERS \
    --model_name "$MODEL" \
    --output_path "$OUT/uk.jsonl"

echo "=== Done (clean is merged separately by compute_cross_entity_jsd.py) ==="
