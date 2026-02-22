import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openmlp.graph import build_step4_graph


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 4: Active learning with NVT MD using two deployed NequIP models."
    )
    parser.add_argument(
        "--input-structure",
        default=str(Path(__file__).resolve().parent / "MgF2.xyz"),
        help="Starting structure for MD.",
    )
    parser.add_argument(
        "--model1-path",
        required=True,
        help="Path to deployed NNP1 model.",
    )
    parser.add_argument(
        "--model2-path",
        required=True,
        help="Path to deployed NNP2 model.",
    )
    parser.add_argument(
        "--output-extxyz",
        default=str(Path(__file__).resolve().parent / "al_selected_non_eq_geoms.extxyz"),
        help="Output extxyz path for selected conformers.",
    )
    parser.add_argument("--md-steps", type=int, default=5000)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--friction", type=float, default=0.02)
    parser.add_argument("--energy-eval-interval", type=int, default=20)
    parser.add_argument("--threshold-warmup-steps", type=int, default=500)
    parser.add_argument("--target-conformers", type=int, default=50)
    parser.add_argument("--rng-seed", type=int, default=123)
    parser.add_argument("--device", default="cpu", help="NequIP device for inference.")
    return parser.parse_args()


def main():
    args = parse_args()
    app = build_step4_graph()
    result = app.invoke(
        {
            "al_input_structure": args.input_structure,
            "al_model1_path": args.model1_path,
            "al_model2_path": args.model2_path,
            "al_output_extxyz": args.output_extxyz,
            "al_md_steps": args.md_steps,
            "al_timestep_fs": args.timestep_fs,
            "al_temperature_k": args.temperature_k,
            "al_friction": args.friction,
            "al_energy_eval_interval": args.energy_eval_interval,
            "al_threshold_warmup_steps": args.threshold_warmup_steps,
            "al_target_conformers": args.target_conformers,
            "al_rng_seed": args.rng_seed,
            "al_device": args.device,
        }
    )
    print(result["notes"])
    print("Selected:", result.get("al_selected_count", 0))
    print("Threshold:", result.get("al_threshold", 0.0))
    print("Output:", result.get("al_output_extxyz", "N/A"))


if __name__ == "__main__":
    main()
