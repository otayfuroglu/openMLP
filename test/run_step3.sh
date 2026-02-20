#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ORCA_PATH:?Set ORCA_PATH to your ORCA executable path}"

python "$SCRIPT_DIR/run_step3.py" \
  --input "$SCRIPT_DIR/MgF2.xyz" \
  --non-eq-output "$SCRIPT_DIR/non_eq_geometries.extxyz" \
  --n-structures 30 \
  --scale-min 0.97 \
  --scale-max 1.09 \
  --max-atom-displacement 0.16 \
  --displacement-attempts 200 \
  --qm-calc-type sp \
  --qm-calculator-type orca \
  --qm-n-core 24 \
  --qm-orca-path "$ORCA_PATH" \
  --qm-workdir "$SCRIPT_DIR" \
  --train-config-template "$SCRIPT_DIR/../openmlp/train/full.yaml" \
  --train-config-path "$SCRIPT_DIR/full.auto.yaml" \
  --train-workdir "$SCRIPT_DIR" \
  --train-val-ratio 0.1 \
  --nequip-command "nequip-train"
