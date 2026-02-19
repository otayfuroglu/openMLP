import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openmlp.graph import build_step2_graph


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 2: Generate non-equilibrium geometries and run QM calculations."
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
        help="Path to ORCA executable for qm_calc/runCalculateGeomWithQM.py.",
    )
    parser.add_argument(
        "--qm-workdir",
        default=str(Path(__file__).resolve().parent),
        help="Working directory for QM run outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = build_step2_graph()
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
        }
    )
    print(result["notes"])
    print("QM extxyz:", result.get("qm_output_extxyz", "N/A"))
    print("QM csv:", result.get("qm_output_csv", "N/A"))


if __name__ == "__main__":
    main()
