import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from ase import Atoms
from ase.io import read, write

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openmlp.active_learning import active_learning_node
from openmlp.non_equilibrium import non_equilibrium_node
from openmlp.qm import qm_calculation_node
from openmlp.train_nequip import train_nequip_node


def _read_atoms(path: Path) -> List[Atoms]:
    data = read(str(path), index=":")
    if isinstance(data, Atoms):
        return [data]
    return list(data)


def _merge_extxyz(input_paths: List[Path], output_path: Path) -> int:
    merged_atoms: List[Atoms] = []
    for input_path in input_paths:
        merged_atoms.extend(_read_atoms(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(output_path), merged_atoms, format="extxyz")
    return len(merged_atoms)


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 5: Repeat QM -> train/deploy -> AL -> QM enrichment cycles."
    )
    parser.add_argument(
        "--initial-dataset-extxyz",
        default="",
        help="Optional initial QM-labeled extxyz. If omitted, bootstrap is built from molecule coordinates.",
    )
    parser.add_argument(
        "--bootstrap-input-structure",
        default=str(Path(__file__).resolve().parent / "MgF2.xyz"),
        help="3D coordinate file used to bootstrap initial non-equilibrium geometries.",
    )
    parser.add_argument("--bootstrap-n-structures", type=int, default=200)
    parser.add_argument("--bootstrap-scale-min", type=float, default=0.97)
    parser.add_argument("--bootstrap-scale-max", type=float, default=1.09)
    parser.add_argument("--bootstrap-max-atom-displacement", type=float, default=0.16)
    parser.add_argument("--bootstrap-displacement-attempts", type=int, default=10000)
    parser.add_argument(
        "--al-input-structure",
        default=str(Path(__file__).resolve().parent / "MgF2.xyz"),
        help="Initial structure used to start AL MD in each cycle.",
    )
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument(
        "--workdir",
        default=str(Path(__file__).resolve().parent / "cycle_runs"),
        help="Output directory for cycle artifacts.",
    )

    parser.add_argument("--qm-orca-path", required=True)
    parser.add_argument("--qm-calc-type", default="engrad")
    parser.add_argument("--qm-calculator-type", default="orca")
    parser.add_argument("--qm-n-core", type=int, default=24)

    parser.add_argument(
        "--train-config-template",
        default=str(PROJECT_ROOT / "openmlp" / "train" / "full.yaml"),
    )
    parser.add_argument("--train-val-ratio", type=float, default=0.1)
    parser.add_argument("--train-num-models", type=int, default=2)
    parser.add_argument("--train-model-seeds", default="")
    parser.add_argument(
        "--train-cuda-devices",
        default="",
        help="Comma-separated CUDA devices for model-parallel training, e.g. 0,1",
    )
    parser.add_argument("--train-timeout-seconds", type=int, default=0)
    parser.add_argument(
        "--train-sequential",
        action="store_true",
        help="Run model trainings sequentially instead of parallel.",
    )
    parser.add_argument(
        "--disable-sequential-fallback",
        action="store_true",
        help="Disable fallback to sequential training when parallel timeout occurs.",
    )
    parser.add_argument("--nequip-command", default="nequip-train")
    parser.add_argument("--nequip-deploy-command", default="nequip-deploy")
    parser.add_argument("--nequip-bin-dir", default="")

    parser.add_argument("--al-md-steps", type=int, default=5000)
    parser.add_argument("--al-timestep-fs", type=float, default=1.0)
    parser.add_argument("--al-temperature-k", type=float, default=300.0)
    parser.add_argument(
        "--al-cycle-temp-start-k",
        type=float,
        default=300.0,
        help="Cycle 1 AL temperature (K).",
    )
    parser.add_argument(
        "--al-cycle-temp-step-k",
        type=float,
        default=0.0,
        help="Temperature increment per cycle (K).",
    )
    parser.add_argument(
        "--al-cycle-temp-max-k",
        type=float,
        default=300.0,
        help="Maximum AL temperature cap (K).",
    )
    parser.add_argument("--al-friction", type=float, default=0.02)
    parser.add_argument("--al-energy-eval-interval", type=int, default=20)
    parser.add_argument("--al-structure-check-interval", type=int, default=20)
    parser.add_argument("--al-min-interatomic-distance", type=float, default=0.6)
    parser.add_argument("--al-max-distance-scale", type=float, default=2.5)
    parser.add_argument("--al-recovery-stride-steps", type=int, default=10)
    parser.add_argument("--al-threshold-warmup-steps", type=int, default=500)
    parser.add_argument("--al-target-conformers", type=int, default=50)
    parser.add_argument("--al-rng-seed", type=int, default=123)
    parser.add_argument("--al-device", default="cuda")
    return parser.parse_args()


def _parse_seed_list(seed_text: str) -> List[int]:
    if not seed_text.strip():
        return []
    return [int(value.strip()) for value in seed_text.split(",") if value.strip()]


def main():
    args = parse_args()
    if args.cycles < 1:
        raise ValueError("--cycles must be >= 1")

    al_input_structure = Path(args.al_input_structure).resolve()
    if not al_input_structure.exists():
        raise FileNotFoundError(f"AL input structure not found: {al_input_structure}")

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    report_path = workdir / "cycle_report.json"

    # Resume automatically if an existing cycle report is found.
    cycle_reports = []
    if report_path.exists():
        existing_report = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(existing_report, list):
            bootstrap_report = {"source": "resume_legacy_report"}
            cycle_reports = existing_report
            current_dataset_str = cycle_reports[-1]["enriched_dataset_out"] if cycle_reports else ""
        else:
            bootstrap_report = existing_report.get("bootstrap", {"source": "resume"})
            cycle_reports = existing_report.get("cycles", [])
            current_dataset_str = existing_report.get("final_dataset", "")
            if not current_dataset_str and cycle_reports:
                current_dataset_str = cycle_reports[-1].get("enriched_dataset_out", "")

        if not current_dataset_str:
            raise RuntimeError(
                f"Found existing report but could not resolve dataset to resume from: {report_path}"
            )
        current_dataset = Path(current_dataset_str).resolve()
        if not current_dataset.exists():
            raise FileNotFoundError(f"Resume dataset from report does not exist: {current_dataset}")
        print("Resume mode: enabled")
        print("Resume dataset:", current_dataset)
        print(f"Completed cycles in report: {len(cycle_reports)}")
    else:
        if args.initial_dataset_extxyz.strip():
            initial_dataset = Path(args.initial_dataset_extxyz).resolve()
            if not initial_dataset.exists():
                raise FileNotFoundError(f"Initial dataset not found: {initial_dataset}")
            current_dataset = initial_dataset
            bootstrap_report = {
                "source": "provided_initial_dataset",
                "initial_dataset": str(initial_dataset),
            }
        else:
            bootstrap_input = Path(args.bootstrap_input_structure).resolve()
            if not bootstrap_input.exists():
                raise FileNotFoundError(f"Bootstrap input structure not found: {bootstrap_input}")
            bootstrap_dir = workdir / "bootstrap"
            bootstrap_non_eq_path = bootstrap_dir / "non_eq_geometries.extxyz"
            bootstrap_non_eq = non_equilibrium_node(
                {
                    "input_structure": str(bootstrap_input),
                    "output_structures": str(bootstrap_non_eq_path),
                    "n_structures": args.bootstrap_n_structures,
                    "scale_min": args.bootstrap_scale_min,
                    "scale_max": args.bootstrap_scale_max,
                    "max_atom_displacement": args.bootstrap_max_atom_displacement,
                    "displacement_attempts": args.bootstrap_displacement_attempts,
                }
            )
            bootstrap_qm = qm_calculation_node(
                {
                    "qm_input_extxyz": str(bootstrap_non_eq_path),
                    "qm_orca_path": args.qm_orca_path,
                    "qm_calc_type": args.qm_calc_type,
                    "qm_calculator_type": args.qm_calculator_type,
                    "qm_n_core": args.qm_n_core,
                    "qm_workdir": str(bootstrap_dir / "qm"),
                }
            )
            current_dataset = Path(bootstrap_qm["qm_output_extxyz"]).resolve()
            bootstrap_report = {
                "source": "bootstrap_from_structure",
                "bootstrap_input_structure": str(bootstrap_input),
                "bootstrap_non_eq_extxyz": str(bootstrap_non_eq_path),
                "bootstrap_non_eq_count": int(bootstrap_non_eq.get("generated_count", 0)),
                "bootstrap_qm_dataset": str(current_dataset),
            }
        print("Bootstrap:", bootstrap_report["source"])
        print("Start dataset:", current_dataset)

    model_seeds = _parse_seed_list(args.train_model_seeds)
    start_cycle = len(cycle_reports) + 1
    if start_cycle > args.cycles:
        print(
            f"Nothing to run: report already has {len(cycle_reports)} cycles and requested cycles={args.cycles}."
        )
        print(f"Final dataset: {current_dataset}")
        print(f"Report: {report_path}")
        return

    for cycle_idx in range(start_cycle, args.cycles + 1):
        cycle_dir = workdir / f"cycle_{cycle_idx:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        cycle_state_path = cycle_dir / "cycle_state.json"
        cycle_state = _load_json(cycle_state_path)
        print(f"\n=== Cycle {cycle_idx}/{args.cycles} ===")
        print(f"Dataset: {current_dataset}")
        cycle_temperature_k = min(
            args.al_cycle_temp_start_k + (cycle_idx - 1) * args.al_cycle_temp_step_k,
            args.al_cycle_temp_max_k,
        )
        print(f"Cycle AL temperature: {cycle_temperature_k:.1f} K")
        cycle_state.setdefault("cycle", cycle_idx)
        cycle_state.setdefault("train_dataset_in", str(current_dataset))
        cycle_state.setdefault("stage", "start")
        cycle_state["al_temperature_k"] = cycle_temperature_k

        # Stage 1: train/deploy
        if cycle_state["stage"] in {"start"}:
            train_workdir = cycle_dir / "train"
            train_result = train_nequip_node(
                {
                    "train_dataset_extxyz": str(current_dataset),
                    "train_config_template": args.train_config_template,
                    "train_config_path": str(train_workdir / "full.auto.yaml"),
                    "train_workdir": str(train_workdir),
                    "train_val_ratio": args.train_val_ratio,
                "train_num_models": args.train_num_models,
                "train_model_seeds": model_seeds if model_seeds else None,
                "train_cuda_devices": args.train_cuda_devices,
                "train_timeout_seconds": args.train_timeout_seconds,
                "train_parallel": not args.train_sequential,
                "train_retry_sequential_on_timeout": not args.disable_sequential_fallback,
                "nequip_bin_dir": args.nequip_bin_dir,
                    "nequip_command": args.nequip_command,
                    "nequip_deploy_command": args.nequip_deploy_command,
                    "train_run": True,
                    "deploy_run": True,
                }
            )
            deployed_models = train_result.get("deployed_model_paths", [])
            if len(deployed_models) < 2:
                raise RuntimeError("Training/deploy did not produce two deployed models required for AL.")
            cycle_state["deployed_models"] = deployed_models
            cycle_state["stage"] = "train_done"
            _save_json(cycle_state_path, cycle_state)
        else:
            deployed_models = cycle_state.get("deployed_models", [])
            if len(deployed_models) < 2:
                raise RuntimeError(
                    f"Cycle {cycle_idx} resume state is missing deployed models: {cycle_state_path}"
                )

        # Stage 2: active learning
        if cycle_state["stage"] in {"train_done"}:
            al_workdir = cycle_dir / "al"
            al_workdir.mkdir(parents=True, exist_ok=True)
            al_selected_path = al_workdir / "al_selected_non_eq_geoms.extxyz"
            al_result = active_learning_node(
                {
                    "al_input_structure": str(al_input_structure),
                    "deployed_model_paths": deployed_models,
                    "al_output_extxyz": str(al_selected_path),
                    "al_md_steps": args.al_md_steps,
                    "al_timestep_fs": args.al_timestep_fs,
                    "al_temperature_k": cycle_temperature_k,
                    "al_friction": args.al_friction,
                    "al_energy_eval_interval": args.al_energy_eval_interval,
                    "al_structure_check_interval": args.al_structure_check_interval,
                    "al_min_interatomic_distance": args.al_min_interatomic_distance,
                    "al_max_distance_scale": args.al_max_distance_scale,
                    "al_recovery_stride_steps": args.al_recovery_stride_steps,
                    "al_recovery_enabled": True,
                    "al_threshold_warmup_steps": args.al_threshold_warmup_steps,
                    "al_target_conformers": args.al_target_conformers,
                    "al_rng_seed": args.al_rng_seed + cycle_idx - 1,
                    "al_device": args.al_device,
                }
            )
            selected_count = int(al_result.get("al_selected_count", 0))
            if selected_count < 1:
                raise RuntimeError(f"Cycle {cycle_idx}: AL produced no conformers for QM.")
            cycle_state["al_selected_path"] = str(al_selected_path)
            cycle_state["al_selected_count"] = selected_count
            cycle_state["al_recovery_selected_count"] = int(
                al_result.get("al_recovery_selected_count", 0)
            )
            cycle_state["al_terminated_early"] = bool(al_result.get("al_terminated_early", False))
            cycle_state["al_termination_reason"] = str(al_result.get("al_termination_reason", ""))
            cycle_state["stage"] = "al_done"
            _save_json(cycle_state_path, cycle_state)

        # Stage 3: QM for AL-selected frames
        if cycle_state["stage"] in {"al_done"}:
            al_selected_path = Path(cycle_state["al_selected_path"]).resolve()
            qm_workdir = cycle_dir / "qm"
            qm_result = qm_calculation_node(
                {
                    "qm_input_extxyz": str(al_selected_path),
                    "qm_orca_path": args.qm_orca_path,
                    "qm_calc_type": args.qm_calc_type,
                    "qm_calculator_type": args.qm_calculator_type,
                    "qm_n_core": args.qm_n_core,
                    "qm_workdir": str(qm_workdir),
                }
            )
            qm_extxyz_path = Path(qm_result["qm_output_extxyz"]).resolve()
            cycle_state["qm_output_extxyz"] = str(qm_extxyz_path)
            cycle_state["stage"] = "qm_done"
            _save_json(cycle_state_path, cycle_state)
        else:
            qm_extxyz_path = Path(cycle_state["qm_output_extxyz"]).resolve()

        # Stage 4: enrich dataset
        if cycle_state["stage"] in {"qm_done"}:
            enriched_path = cycle_dir / f"enriched_dataset_cycle_{cycle_idx:02d}.extxyz"
            total_structures = _merge_extxyz([current_dataset, qm_extxyz_path], enriched_path)
            cycle_state["enriched_dataset_out"] = str(enriched_path)
            cycle_state["enriched_total_structures"] = total_structures
            cycle_state["stage"] = "enrich_done"
            _save_json(cycle_state_path, cycle_state)
            current_dataset = enriched_path
        else:
            enriched_path = Path(cycle_state["enriched_dataset_out"]).resolve()
            total_structures = int(cycle_state["enriched_total_structures"])
            current_dataset = enriched_path

        cycle_report = {
            "cycle": cycle_idx,
            "train_dataset_in": cycle_state["train_dataset_in"],
            "deployed_models": cycle_state["deployed_models"],
            "al_selected_count": int(cycle_state["al_selected_count"]),
            "al_recovery_selected_count": int(cycle_state.get("al_recovery_selected_count", 0)),
            "al_terminated_early": bool(cycle_state.get("al_terminated_early", False)),
            "al_termination_reason": str(cycle_state.get("al_termination_reason", "")),
            "qm_output_extxyz": cycle_state["qm_output_extxyz"],
            "enriched_dataset_out": cycle_state["enriched_dataset_out"],
            "enriched_total_structures": int(cycle_state["enriched_total_structures"]),
            "al_temperature_k": float(cycle_state.get("al_temperature_k", cycle_temperature_k)),
        }
        cycle_reports.append(cycle_report)
        print(
            f"Cycle {cycle_idx} done: selected={cycle_report['al_selected_count']}, "
            f"recovery={cycle_report['al_recovery_selected_count']}, "
            f"enriched_total={total_structures}"
        )

    final_report = {
        "bootstrap": bootstrap_report,
        "cycles": cycle_reports,
        "final_dataset": str(current_dataset),
    }
    report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print("\n=== Completed all cycles ===")
    print(f"Final dataset: {current_dataset}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
