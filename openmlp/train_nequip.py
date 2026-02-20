from pathlib import Path
import shlex
import subprocess
from typing import Any, Dict, Optional, Tuple

from ase import Atoms
from ase.io import read
import yaml

from openmlp.state import PipelineState


def _load_structures_count(extxyz_path: Path) -> int:
    atoms_data = read(str(extxyz_path), index=":")
    if isinstance(atoms_data, Atoms):
        return 1
    return len(list(atoms_data))


def _compute_split_sizes(
    total_count: int,
    train_n_train: Optional[int],
    train_n_val: Optional[int],
    train_val_ratio: float,
) -> Tuple[int, int]:
    if total_count < 2:
        raise ValueError("Need at least 2 structures for train/val split.")

    if train_n_train is not None and train_n_val is not None:
        n_train = int(train_n_train)
        n_val = int(train_n_val)
    elif train_n_val is not None:
        n_val = int(train_n_val)
        n_train = total_count - n_val
    elif train_n_train is not None:
        n_train = int(train_n_train)
        n_val = total_count - n_train
    else:
        n_val = max(1, int(round(total_count * train_val_ratio)))
        n_train = total_count - n_val

    if n_train < 1 or n_val < 1:
        raise ValueError(f"Invalid split sizes: n_train={n_train}, n_val={n_val}, total={total_count}")
    if n_train + n_val > total_count:
        raise ValueError(
            f"n_train + n_val exceeds dataset size: {n_train} + {n_val} > {total_count}"
        )
    return n_train, n_val


def _prepare_nequip_config(
    template_path: Path,
    output_config_path: Path,
    dataset_extxyz_path: Path,
    workdir: Path,
    n_train: int,
    n_val: int,
    train_root: Optional[str],
) -> Dict[str, Any]:
    with template_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"NequIP config template is not a mapping: {template_path}")

    config["dataset_file_name"] = str(dataset_extxyz_path)
    config["n_train"] = int(n_train)
    config["n_val"] = int(n_val)
    config["root"] = str(Path(train_root).resolve()) if train_root else str((workdir / "results").resolve())

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with output_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config


def train_nequip_node(state: PipelineState) -> PipelineState:
    dataset_path_str = state.get("train_dataset_extxyz") or state.get("qm_output_extxyz")
    if not dataset_path_str:
        raise ValueError("train_dataset_extxyz is required or provide qm_output_extxyz from step 2.")
    dataset_path = Path(dataset_path_str).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    template_default = Path(__file__).resolve().parent / "train" / "full.yaml"
    template_path = Path(state.get("train_config_template", str(template_default))).resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"NequIP template config not found: {template_path}")

    workdir = Path(state.get("train_workdir", str(dataset_path.parent))).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    config_path = Path(
        state.get("train_config_path", str(workdir / "full.auto.yaml"))
    ).resolve()

    total_count = _load_structures_count(dataset_path)
    n_train, n_val = _compute_split_sizes(
        total_count=total_count,
        train_n_train=state.get("train_n_train"),
        train_n_val=state.get("train_n_val"),
        train_val_ratio=float(state.get("train_val_ratio", 0.1)),
    )
    _prepare_nequip_config(
        template_path=template_path,
        output_config_path=config_path,
        dataset_extxyz_path=dataset_path,
        workdir=workdir,
        n_train=n_train,
        n_val=n_val,
        train_root=state.get("train_root"),
    )

    run_training = bool(state.get("train_run", True))
    nequip_command = state.get("nequip_command", "nequip-train")
    log_path = workdir / "nequip_train.log"
    if run_training:
        cmd_parts = shlex.split(nequip_command) + [str(config_path)]
        completed = subprocess.run(
            cmd_parts,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            check=False,
        )
        log_path.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""), encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                "NequIP training failed.\n"
                f"Command: {' '.join(cmd_parts)}\n"
                f"See log: {log_path}"
            )

    note = (
        f"Prepared NequIP config with dataset={dataset_path.name}, "
        f"n_train={n_train}, n_val={n_val}."
    )
    if run_training:
        note += " Training completed."
    else:
        note += " Training skipped (train_run=false)."

    return {
        "train_config_path": str(config_path),
        "train_dataset_extxyz": str(dataset_path),
        "train_n_train": n_train,
        "train_n_val": n_val,
        "train_workdir": str(workdir),
        "train_log_path": str(log_path),
        "notes": note,
    }
