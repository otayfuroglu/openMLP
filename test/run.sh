#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/run_step1.py" \
  --input "$SCRIPT_DIR/MgF2.xyz" \
  --output "$SCRIPT_DIR/non_eq_geometries.extxyz" \
  --n-structures 300 \
  --scale-min 0.97 \
  --scale-max 1.09 \
  --max-atom-displacement 0.16 \
  --displacement-attempts 20000
