import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openmlp.graph import build_step3_graph, build_train_only_graph


def parse_seed_list(seed_values: str) -> List[int]:
    values = [value.strip() for value in seed_values.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("train-model-seeds cannot be empty.")
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("train-model-seeds must be comma-separated integers.") from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 3: Generate non-equilibrium geometries, run QM, and train NequIP."
    )
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent / "MgF2.xyz"),
        help="Input structure file (e.g., xyz, cif).",
    )
    parser.add_argument(
        "--non-eq-output",
        default=str(Path(__file__).resolve().parent / "non_eq_geometries.extxyz"),
        help="Output extxyz path for generated non-equilibrium geometries.",
    )
    parser.add_argument("--n-structures", type=int, default=30)
    parser.add_argument("--scale-min", type=float, default=0.97)
    parser.add_argument("--scale-max", type=float, default=1.09)
    parser.add_argument("--max-atom-displacement", type=float, default=0.16)
    parser.add_argument("--displacement-attempts", type=int, default=200)

    parser.add_argument("--qm-calc-type", default="sp", help="QM calc type prefix.")
    parser.add_argument("--qm-calculator-type", default="orca", help="orca or g16.")
    parser.add_argument("--qm-n-core", type=int, default=24, help="Total cores for QM step.")
    parser.add_argument(
        "--qm-orca-path",
        default="",
        help="Path to ORCA executable for openmlp/qm_calc/runCalculateGeomWithQM.py.",
    )
    parser.add_argument(
        "--qm-workdir",
        default=str(Path(__file__).resolve().parent),
        help="Working directory for QM outputs.",
    )

    parser.add_argument(
        "--train-config-template",
        default=str(PROJECT_ROOT / "openmlp" / "train" / "full.yaml"),
        help="NequIP YAML template path.",
    )
    parser.add_argument(
        "--train-config-path",
        default=str(Path(__file__).resolve().parent / "full.auto.yaml"),
        help="Generated NequIP config output path.",
    )
    parser.add_argument(
        "--train-workdir",
        default=str(Path(__file__).resolve().parent),
        help="NequIP training working directory.",
    )
    parser.add_argument(
        "--train-val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio when n_train/n_val are not explicitly set.",
    )
    parser.add_argument("--train-n-train", type=int, default=None)
    parser.add_argument("--train-n-val", type=int, default=None)
    parser.add_argument(
        "--train-num-models",
        type=int,
        default=2,
        help="Number of independent NequIP models (different seeds only).",
    )
    parser.add_argument(
        "--train-model-seeds",
        type=parse_seed_list,
        default=None,
        help="Comma-separated explicit model seeds, e.g. 123,456",
    )
    parser.add_argument(
        "--nequip-command",
        default="nequip-train",
        help="NequIP CLI command.",
    )
    parser.add_argument(
        "--nequip-deploy-command",
        default="nequip-deploy",
        help="NequIP deploy CLI command.",
    )
    parser.add_argument(
        "--nequip-bin-dir",
        default="",
        help="Directory that contains both nequip-train and nequip-deploy executables.",
    )
    parser.add_argument(
        "--no-train-run",
        action="store_true",
        help="Only prepare auto config and skip actual NequIP training run.",
    )
    parser.add_argument(
        "--no-deploy-run",
        action="store_true",
        help="Skip nequip-deploy after training.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Skip non-equ and QM; run NequIP training only from --train-dataset-extxyz.",
    )
    parser.add_argument(
        "--train-dataset-extxyz",
        default=str(Path(__file__).resolve().parent / "MgF2.extxyz"),
        help="Existing extxyz dataset for train-only mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    payload = {
        "train_config_template": args.train_config_template,
        "train_config_path": args.train_config_path,
        "train_workdir": args.train_workdir,
        "train_val_ratio": args.train_val_ratio,
        "train_n_train": args.train_n_train,
        "train_n_val": args.train_n_val,
        "train_num_models": args.train_num_models,
        "train_model_seeds": args.train_model_seeds,
        "nequip_bin_dir": args.nequip_bin_dir,
        "nequip_command": args.nequip_command,
        "nequip_deploy_command": args.nequip_deploy_command,
        "train_run": not args.no_train_run,
        "deploy_run": not args.no_deploy_run,
    }
    if args.train_only:
        app = build_train_only_graph()
        payload["train_dataset_extxyz"] = args.train_dataset_extxyz
    else:
        app = build_step3_graph()
        payload.update(
            {
                "input_structure": args.input,
                "output_structures": args.non_eq_output,
                "n_structures": args.n_structures,
                "scale_min": args.scale_min,
                "scale_max": args.scale_max,
                "max_atom_displacement": args.max_atom_displacement,
                "displacement_attempts": args.displacement_attempts,
                "qm_input_extxyz": args.non_eq_output,
                "qm_calc_type": args.qm_calc_type,
                "qm_calculator_type": args.qm_calculator_type,
                "qm_n_core": args.qm_n_core,
                "qm_orca_path": args.qm_orca_path,
                "qm_workdir": args.qm_workdir,
            }
        )
    result = app.invoke(payload)
    print(result["notes"])
    print("QM extxyz:", result.get("qm_output_extxyz", "N/A"))
    print("Train configs:", ", ".join(result.get("train_model_config_paths", [])))
    print("Model seeds:", result.get("train_model_seeds", []))
    print("n_train:", result.get("train_n_train", "N/A"))
    print("n_val:", result.get("train_n_val", "N/A"))
    print("Train logs:", ", ".join(result.get("train_model_log_paths", [])))
    print("Deploy models:", ", ".join(result.get("deployed_model_paths", [])))


if __name__ == "__main__":
    main()
