import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openmlp.graph import build_step3_graph


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
        required=True,
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
        "--nequip-command",
        default="nequip-train",
        help="NequIP CLI command.",
    )
    parser.add_argument(
        "--no-train-run",
        action="store_true",
        help="Only prepare auto config and skip actual NequIP training run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = build_step3_graph()
    result = app.invoke(
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
            "train_config_template": args.train_config_template,
            "train_config_path": args.train_config_path,
            "train_workdir": args.train_workdir,
            "train_val_ratio": args.train_val_ratio,
            "train_n_train": args.train_n_train,
            "train_n_val": args.train_n_val,
            "nequip_command": args.nequip_command,
            "train_run": not args.no_train_run,
        }
    )
    print(result["notes"])
    print("QM extxyz:", result.get("qm_output_extxyz", "N/A"))
    print("Train config:", result.get("train_config_path", "N/A"))
    print("n_train:", result.get("train_n_train", "N/A"))
    print("n_val:", result.get("train_n_val", "N/A"))
    print("Train log:", result.get("train_log_path", "N/A"))


if __name__ == "__main__":
    main()
