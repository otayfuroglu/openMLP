#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${NNP1_MODEL:?Set NNP1_MODEL to deployed model 1 path}"
: "${NNP2_MODEL:?Set NNP2_MODEL to deployed model 2 path}"

python "$SCRIPT_DIR/run_step4.py" \
  --input-structure "$SCRIPT_DIR/MgF2.xyz" \
  --model1-path "$NNP1_MODEL" \
  --model2-path "$NNP2_MODEL" \
  --output-extxyz "$SCRIPT_DIR/al_selected_non_eq_geoms.extxyz" \
  --md-steps 5000 \
  --timestep-fs 1.0 \
  --temperature-k 300.0 \
  --friction 0.02 \
  --energy-eval-interval 20 \
  --structure-check-interval 20 \
  --min-interatomic-distance 0.6 \
  --max-distance-scale 2.5 \
  --recovery-stride-steps 10 \
  --threshold-warmup-steps 500 \
  --target-conformers 50 \
  --rng-seed 123 \
  --device cuda
