# openMLP

Step 1 (no LLM): generate non-equilibrium geometries from a starting molecule
using a minimal `LangGraph` workflow.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install langgraph ase numpy
pip install tqdm
pip install pyyaml
# install nequip from your environment recipe (pip/conda)
```

## Run step 1

```bash
bash test/run.sh
```

Or run directly:

```bash
python test/run_step1.py \
  --input test/MgF2.xyz \
  --output test/non_eq_geometries.extxyz \
  --n-structures 50 \
  --scale-min 0.97 \
  --scale-max 1.09 \
  --max-atom-displacement 0.16 \
  --displacement-attempts 200
```

## Notes

- `run_step1.py` executes a one-node LangGraph app (`generate_non_equilibrium`).
- Generation logic is adapted from `getNonEquGeom.py`:
  scale sweep in `[scale-min, scale-max)` plus bounded random atomic displacement.
- Output is an `extxyz` trajectory containing generated non-equilibrium geometries.
- This is deterministic orchestration with no LLM/API usage.

## Run step 2 (integrated QM)

Step 2 uses your existing code in `openmlp/qm_calc/runCalculateGeomWithQM.py` through a
LangGraph node (`run_qm`) after non-equilibrium generation.

Set ORCA path and run:

```bash
export ORCA_PATH=/path/to/orca
bash test/run_step2.sh
```

Or directly:

```bash
python test/run_step2.py \
  --input test/MgF2.xyz \
  --non-eq-output test/non_eq_geometries.extxyz \
  --n-structures 30 \
  --qm-calc-type sp \
  --qm-calculator-type orca \
  --qm-n-core 24 \
  --qm-orca-path /path/to/orca \
  --qm-workdir test
```

## Run step 3 (QM + NequIP training)

Step 3 extends the workflow with NequIP training using `openmlp/train/full.yaml`
as template. It automatically updates:

- `dataset_file_name` -> QM output extxyz path
- `n_train` and `n_val` -> computed from dataset size and `train_val_ratio`
- trains an ensemble of 2 independent models by changing only `seed`
- deploys each trained model using `nequip-deploy build`

Run:

```bash
export ORCA_PATH=/path/to/orca
bash test/run_step3.sh
```

This writes seed-specific auto configs such as `test/full.auto.seed123.yaml`.

Ensemble behavior:

- default seeds are derived from template `seed` (e.g. `123`, `124`)
- override explicitly with `--train-model-seeds 123,456`
- if `nequip-train` and `nequip-deploy` are in a custom env path, pass
  `--nequip-bin-dir /path/to/bin` (or export `NEQUIP_BIN_DIR` for `test/run_step3.sh`)

Config-only dry run (no NequIP training):

```bash
python test/run_step3.py \
  --qm-orca-path /path/to/orca \
  --no-train-run
```

Train-only mode (skip non-equ + QM, train and deploy from existing extxyz):

```bash
python test/run_step3.py \
  --train-only \
  --train-dataset-extxyz test/MgF2.extxyz \
  --nequip-bin-dir /path/to/nequip/bin
```

## Run step 4 (active learning MD)

Step 4 runs NVT MD with `nnp1` (ASE), evaluates uncertainty
`|E_nnp1 - E_nnp2|` every fixed interval, and selects conformers online.

Selection policy:

- accumulate uncertainties during warmup window (for example first 500 MD steps)
- set threshold = average warmup uncertainty
- after warmup, select frames where uncertainty >= threshold
- stop MD early when selected conformers reaches user target

Run:

```bash
export NNP1_MODEL=/path/to/deployed_model_seed123.pth
export NNP2_MODEL=/path/to/deployed_model_seed124.pth
bash test/run_step4.sh
```

Direct command:

```bash
python test/run_step4.py \
  --input-structure test/MgF2.xyz \
  --model1-path /path/to/deployed_model_seed123.pth \
  --model2-path /path/to/deployed_model_seed124.pth \
  --output-extxyz test/al_selected_non_eq_geoms.extxyz \
  --md-steps 5000 \
  --energy-eval-interval 20 \
  --threshold-warmup-steps 500 \
  --target-conformers 50
```
