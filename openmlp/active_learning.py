from pathlib import Path
from typing import List, Optional

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.units import fs
from tqdm import tqdm

from openmlp.state import PipelineState


def _load_nequip_calculator(model_path: Path, device: str):
    try:
        from nequip.ase import NequIPCalculator
    except Exception as exc:
        raise ImportError(
            "Failed to import nequip ASE calculator. Install NequIP in this environment."
        ) from exc
    return NequIPCalculator.from_deployed_model(model_path=str(model_path), device=device)


def _resolve_model_paths(state: PipelineState) -> tuple[Path, Path]:
    model1 = state.get("al_model1_path")
    model2 = state.get("al_model2_path")
    deployed_paths = state.get("deployed_model_paths", [])
    if not model1 and len(deployed_paths) >= 1:
        model1 = deployed_paths[0]
    if not model2 and len(deployed_paths) >= 2:
        model2 = deployed_paths[1]
    if not model1 or not model2:
        raise ValueError("Provide al_model1_path and al_model2_path, or deployed_model_paths with two models.")
    model1_path = Path(model1).resolve()
    model2_path = Path(model2).resolve()
    if not model1_path.exists():
        raise FileNotFoundError(f"Model 1 not found: {model1_path}")
    if not model2_path.exists():
        raise FileNotFoundError(f"Model 2 not found: {model2_path}")
    return model1_path, model2_path


def _pair_distance_stats(atoms: Atoms) -> tuple[float, float]:
    distances = atoms.get_all_distances(mic=True)
    non_zero = distances[distances > 0.0]
    if non_zero.size == 0:
        return 0.0, 0.0
    return float(np.min(non_zero)), float(np.max(non_zero))


def _check_structure_health(
    atoms: Atoms,
    initial_max_distance: float,
    min_interatomic_distance: float,
    max_distance_scale: float,
) -> Optional[str]:
    positions = atoms.get_positions()
    if not np.all(np.isfinite(positions)):
        return "Detected non-finite atomic positions (NaN/Inf)."

    min_dist, max_dist = _pair_distance_stats(atoms)
    if min_dist > 0.0 and min_dist < min_interatomic_distance:
        return (
            f"Structure collapse detected: minimum interatomic distance {min_dist:.4f} A "
            f"< threshold {min_interatomic_distance:.4f} A."
        )

    max_allowed = initial_max_distance * max_distance_scale
    if max_dist > max_allowed:
        return (
            f"Structure explosion detected: maximum interatomic distance {max_dist:.4f} A "
            f"> threshold {max_allowed:.4f} A."
        )
    return None


def active_learning_node(state: PipelineState) -> PipelineState:
    input_structure = state.get("al_input_structure") or state.get("input_structure")
    if not input_structure:
        raise ValueError("al_input_structure is required (or provide input_structure).")
    input_path = Path(input_structure).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"AL input structure not found: {input_path}")

    model1_path, model2_path = _resolve_model_paths(state)
    output_path = Path(state.get("al_output_extxyz", "outputs/al_selected_non_eq_geoms.extxyz")).resolve()
    last_frame_default = output_path.with_name(f"{output_path.stem}_last_frame.extxyz")
    last_frame_path = Path(state.get("al_last_frame_path", str(last_frame_default))).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_frame_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    md_steps = int(state.get("al_md_steps", 2000))
    timestep_fs = float(state.get("al_timestep_fs", 1.0))
    temperature_k = float(state.get("al_temperature_k", 300.0))
    friction = float(state.get("al_friction", 0.02))
    eval_interval = int(state.get("al_energy_eval_interval", 20))
    check_interval = int(state.get("al_structure_check_interval", 20))
    min_interatomic_distance = float(state.get("al_min_interatomic_distance", 0.6))
    max_distance_scale = float(state.get("al_max_distance_scale", 2.5))
    warmup_steps = int(state.get("al_threshold_warmup_steps", 500))
    target_conformers = int(state.get("al_target_conformers", 50))
    rng_seed = int(state.get("al_rng_seed", 123))
    device = state.get("al_device", "cuda")

    if md_steps < 1:
        raise ValueError("al_md_steps must be >= 1.")
    if eval_interval < 1:
        raise ValueError("al_energy_eval_interval must be >= 1.")
    if check_interval < 1:
        raise ValueError("al_structure_check_interval must be >= 1.")
    if warmup_steps < 1:
        raise ValueError("al_threshold_warmup_steps must be >= 1.")
    if target_conformers < 1:
        raise ValueError("al_target_conformers must be >= 1.")

    atoms: Atoms = read(str(input_path))
    np.random.seed(rng_seed)
    _, initial_max_distance = _pair_distance_stats(atoms)

    calc1 = _load_nequip_calculator(model1_path, device=device)
    calc2 = _load_nequip_calculator(model2_path, device=device)
    atoms.calc = calc1

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k)
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = Langevin(atoms, timestep_fs * fs, temperature_K=temperature_k, friction=friction)

    uncertainties: List[float] = []
    warmup_uncertainties: List[float] = []
    selected_count = 0
    threshold: Optional[float] = None
    terminated_early = False
    termination_reason = ""

    progress = tqdm(range(1, md_steps + 1), desc="AL NVT MD", unit="step")
    for step in progress:
        dyn.run(1)

        if step % check_interval == 0:
            health_reason = _check_structure_health(
                atoms=atoms,
                initial_max_distance=initial_max_distance,
                min_interatomic_distance=min_interatomic_distance,
                max_distance_scale=max_distance_scale,
            )
            if health_reason:
                terminated_early = True
                termination_reason = f"Terminated at step {step}. {health_reason}"
                progress.set_postfix(status="terminated", reason="structure_unstable")
                break

        if step % eval_interval != 0:
            continue

        energy1 = float(atoms.get_potential_energy())
        if not np.isfinite(energy1):
            terminated_early = True
            termination_reason = f"Terminated at step {step}. Model 1 energy is non-finite."
            progress.set_postfix(status="terminated", reason="nonfinite_energy")
            break
        atoms2 = atoms.copy()
        atoms2.calc = calc2
        energy2 = float(atoms2.get_potential_energy())
        if not np.isfinite(energy2):
            terminated_early = True
            termination_reason = f"Terminated at step {step}. Model 2 energy is non-finite."
            progress.set_postfix(status="terminated", reason="nonfinite_energy")
            break
        uncertainty = abs(energy1 - energy2)
        uncertainties.append(uncertainty)

        if step <= warmup_steps:
            warmup_uncertainties.append(uncertainty)
            continue

        if threshold is None:
            if warmup_uncertainties:
                threshold = float(np.mean(warmup_uncertainties))
            else:
                threshold = uncertainty

        if uncertainty >= threshold:
            chosen = atoms.copy()
            chosen.info["al_step"] = step
            chosen.info["al_uncertainty"] = uncertainty
            chosen.info["al_threshold"] = threshold
            chosen.info["al_energy_model1"] = energy1
            chosen.info["al_energy_model2"] = energy2
            write(str(output_path), chosen, format="extxyz", append=True)
            selected_count += 1
            progress.set_postfix(
                selected=selected_count,
                threshold=f"{threshold:.5f}",
                uncertainty=f"{uncertainty:.5f}",
            )
            if selected_count >= target_conformers:
                break

    if threshold is None:
        threshold = float(np.mean(warmup_uncertainties)) if warmup_uncertainties else 0.0

    saved_last_frame_path = ""
    if terminated_early:
        final_atoms = atoms.copy()
        final_atoms.info["al_terminated_early"] = terminated_early
        final_atoms.info["al_termination_reason"] = termination_reason
        final_atoms.info["al_selected_count"] = selected_count
        final_atoms.info["al_threshold"] = threshold
        write(str(last_frame_path), final_atoms, format="extxyz")
        saved_last_frame_path = str(last_frame_path)

    if terminated_early:
        note = (
            f"AL MD terminated early. Selected {selected_count}/{target_conformers} conformers. "
            f"Threshold={threshold:.6f}. Warning: {termination_reason} "
            f"Last frame saved at {last_frame_path}."
        )
    else:
        note = (
            f"AL MD finished. Selected {selected_count}/{target_conformers} conformers. "
            f"Threshold={threshold:.6f}, output={output_path}."
        )

    return {
        "al_output_extxyz": str(output_path),
        "al_last_frame_path": saved_last_frame_path,
        "al_threshold": threshold,
        "al_selected_count": selected_count,
        "al_uncertainties": uncertainties,
        "al_terminated_early": terminated_early,
        "al_termination_reason": termination_reason,
        "notes": note,
    }
