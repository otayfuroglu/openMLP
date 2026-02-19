from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from ase import Atoms
from ase.io import read, write
from tqdm import tqdm

from openmlp.state import PipelineState


@dataclass
class NonEquilibriumConfig:
    n_structures: int = 30
    scale_min: float = 0.97
    scale_max: float = 1.09
    max_atom_displacement: float = 0.16
    displacement_attempts: int = 200


def _scale_atoms_distance(atoms: Atoms, scale_factor: float) -> Atoms:
    atoms.center(vacuum=0.0)
    atoms.set_cell(scale_factor * atoms.cell, scale_atoms=True)
    return atoms


def _random_scale_direction(direction_value: float, scale_range: Sequence[float]) -> float:
    return float(np.random.uniform(scale_range[0] * direction_value, scale_range[1] * direction_value))


def _displacement_norm(delta_vector: np.ndarray) -> float:
    return float(np.sqrt(np.sum(delta_vector**2)))


def _displaced_atomic_position(
    atom_position: np.ndarray,
    scale_range: Sequence[float],
    max_atom_displacement: float,
    displacement_attempts: int,
) -> np.ndarray:
    for _ in range(displacement_attempts):
        new_position = np.array(
            [_random_scale_direction(axis_value, scale_range) for axis_value in atom_position]
        )
        if _displacement_norm(atom_position - new_position) <= max_atom_displacement:
            return new_position
    raise RuntimeError(
        "Could not sample a displaced atom position within max_atom_displacement. "
        "Increase displacement_attempts or max_atom_displacement."
    )


def generate_non_equilibrium_geometries(
    input_structure: str,
    output_structures: str,
    cfg: NonEquilibriumConfig,
) -> tuple[List[str], int]:
    if cfg.n_structures < 1:
        raise ValueError("n_structures must be >= 1.")
    if cfg.scale_max <= cfg.scale_min:
        raise ValueError("scale_max must be greater than scale_min.")

    atoms_data = read(input_structure, index=":")
    if isinstance(atoms_data, Atoms):
        base_structures = [atoms_data]
    else:
        base_structures = list(atoms_data)
    if not base_structures:
        raise ValueError(f"No structures found in input: {input_structure}")

    output_path = Path(output_structures)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated: List[Atoms] = []
    scale_step = (cfg.scale_max - cfg.scale_min) / cfg.n_structures
    scale_values = np.arange(cfg.scale_min, cfg.scale_max, scale_step)

    for i, base_atoms in enumerate(tqdm(base_structures, desc="Base structures", unit="structure")):
        label = str(base_atoms.info.get("label", Path(input_structure).stem))
        scale_iter = tqdm(
            scale_values,
            desc=f"Scales for structure {i}",
            unit="scale",
            leave=False,
        )
        for j, scale_factor in enumerate(scale_iter):
            trial = base_atoms.copy()
            #  if trial.cell is not None and trial.cell.rank > 0:
            trial = _scale_atoms_distance(trial, float(scale_factor))
            scale_range = (cfg.scale_min, cfg.scale_max)
            for atom_index, atom_position in enumerate(trial.positions):
                trial.positions[atom_index] = _displaced_atomic_position(
                    atom_position=np.array(atom_position),
                    scale_range=scale_range,
                    max_atom_displacement=cfg.max_atom_displacement,
                    displacement_attempts=cfg.displacement_attempts,
                )
            trial.info["label"] = f"{label}_{i}_{j:03d}"
            generated.append(trial)

    write(output_structures, generated, format="extxyz")
    return [str(output_path)], len(generated)


def non_equilibrium_node(state: PipelineState) -> PipelineState:
    cfg = NonEquilibriumConfig(
        n_structures=state.get("n_structures", 30),
        scale_min=state.get("scale_min", 0.97),
        scale_max=state.get("scale_max", 1.09),
        max_atom_displacement=state.get("max_atom_displacement", 0.16),
        displacement_attempts=state.get("displacement_attempts", 200),
    )
    input_structure = state["input_structure"]
    output_structures = state.get("output_structures", "outputs/non_eq_geometries.extxyz")
    files, generated_count = generate_non_equilibrium_geometries(input_structure, output_structures, cfg)

    return {
        "generated_count": generated_count,
        "generated_files": files,
        "notes": f"Generated non-equilibrium geometries from {input_structure} with scale range [{cfg.scale_min}, {cfg.scale_max}].",
    }
