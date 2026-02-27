#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ORCA_PATH:?Set ORCA_PATH to your ORCA executable path}"
: "${BOOTSTRAP_STRUCTURE:=$SCRIPT_DIR/MgF2.xyz}"
: "${NEQUIP_BIN_DIR:=}"
: "${TRAIN_CUDA_DEVICES:=}"
: "${QM_SLURM_TEMPLATE:=}"
: "${TRAIN_SLURM_TEMPLATE:=}"
: "${QM_SUBMIT_JOBS:=1}"

python "$SCRIPT_DIR/run_step5_cycles.py" \
  --bootstrap-input-structure "$BOOTSTRAP_STRUCTURE" \
  --bootstrap-n-structures 50 \
  --al-input-structure "$SCRIPT_DIR/MgF2.xyz" \
  --cycles 5 \
  --workdir "$SCRIPT_DIR/cycle_runs" \
  --qm-orca-path "$ORCA_PATH" \
  --qm-calc-type engrad \
  --qm-calculator-type orca \
  --qm-n-core 24 \
  --qm-submit-jobs "$QM_SUBMIT_JOBS" \
  --qm-slurm-template "$QM_SLURM_TEMPLATE" \
  --train-config-template "$SCRIPT_DIR/../openmlp/train/full.yaml" \
  --train-val-ratio 0.1 \
  --train-num-models 2 \
  --train-cuda-devices "$TRAIN_CUDA_DEVICES" \
  --nequip-command "nequip-train" \
  --nequip-deploy-command "nequip-deploy" \
  --nequip-bin-dir "$NEQUIP_BIN_DIR" \
  --train-slurm-template "$TRAIN_SLURM_TEMPLATE" \
  --slurm-poll-seconds 15 \
  --al-md-steps 5000 \
  --al-cycle-temp-start-k 100 \
  --al-cycle-temp-step-k 50 \
  --al-cycle-temp-max-k 500 \
  --al-energy-eval-interval 20 \
  --al-structure-check-interval 20 \
  --al-min-interatomic-distance 0.6 \
  --al-max-distance-scale 2.5 \
  --al-recovery-stride-steps 10 \
  --al-threshold-warmup-steps 500 \
  --al-target-conformers 50 \
  --al-device cuda
