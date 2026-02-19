import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openmlp.graph import build_step1_graph


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 1: Generate non-equilibrium geometries from a starting molecule."
    )
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent / "MgF2.xyz"),
        help="Input structure file (e.g., xyz, cif).",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "non_eq_geometries.extxyz"),
        help="Output trajectory file in extxyz format.",
    )
    parser.add_argument("--n-structures", type=int, default=30)
    parser.add_argument("--scale-min", type=float, default=0.97)
    parser.add_argument("--scale-max", type=float, default=1.09)
    parser.add_argument("--max-atom-displacement", type=float, default=0.16)
    parser.add_argument("--displacement-attempts", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    app = build_step1_graph()
    result = app.invoke(
        {
            "input_structure": args.input,
            "output_structures": args.output,
            "n_structures": args.n_structures,
            "scale_min": args.scale_min,
            "scale_max": args.scale_max,
            "max_atom_displacement": args.max_atom_displacement,
            "displacement_attempts": args.displacement_attempts,
        }
    )
    print(result["notes"])
    print("Output:", ", ".join(result["generated_files"]))


if __name__ == "__main__":
    main()
