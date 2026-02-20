#!/usr/bin/env bash
# Run cross-entity projections (Gemma + OLMo) then compute JSD heatmaps.
set -euo pipefail

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Cross-Entity Projections: Gemma ==="
bash scripts/run_cross_entity_projections_gemma.sh 2>&1 | tee "logs/cross_entity_gemma_${TIMESTAMP}.log"

echo ""
echo "=== Cross-Entity Projections: OLMo ==="
bash scripts/run_cross_entity_projections_olmo.sh 2>&1 | tee "logs/cross_entity_olmo_${TIMESTAMP}.log"

echo ""
echo "=== JSD Computation + Heatmaps ==="
.venv/bin/python -m src.compute_cross_entity_jsd 2>&1 | tee "logs/cross_entity_jsd_${TIMESTAMP}.log"

echo ""
echo "=== ALL DONE ==="
