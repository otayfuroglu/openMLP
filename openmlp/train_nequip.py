from pathlib import Path
import shlex
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from ase import Atoms
from ase.io import read
import yaml

from openmlp.state import PipelineState


class _TupleSafeLoader(yaml.SafeLoader):
    pass


def _construct_python_tuple(loader: yaml.Loader, node: yaml.Node) -> tuple:
    return tuple(loader.construct_sequence(node))


_TupleSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    _construct_python_tuple,
)


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
    seed: int,
    run_name_suffix: str,
) -> Dict[str, Any]:
    with template_path.open("r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=_TupleSafeLoader)
    if not isinstance(config, dict):
        raise ValueError(f"NequIP config template is not a mapping: {template_path}")

    config["dataset_file_name"] = str(dataset_extxyz_path)
    config["n_train"] = int(n_train)
    config["n_val"] = int(n_val)
    config["seed"] = int(seed)
    config["root"] = str(Path(train_root).resolve()) if train_root else str((workdir / "results").resolve())
    if "run_name" in config and isinstance(config["run_name"], str):
        config["run_name"] = f"{config['run_name']}_{run_name_suffix}"
    else:
        config["run_name"] = run_name_suffix

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with output_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config


def _resolve_model_seeds(state: PipelineState, base_seed: int) -> List[int]:
    model_seeds = state.get("train_model_seeds")
    if model_seeds:
        return [int(seed) for seed in model_seeds]
    num_models = int(state.get("train_num_models", 2))
    if num_models < 1:
        raise ValueError("train_num_models must be >= 1.")
    return [base_seed + index for index in range(num_models)]


def _resolve_command_parts(command: str, bin_dir: Optional[str], executable_name: str) -> List[str]:
    parts = shlex.split(command)
    if not parts:
        raise ValueError(f"Command is empty for {executable_name}.")
    if bin_dir and parts[0] == executable_name:
        parts[0] = str((Path(bin_dir).resolve() / executable_name))
    return parts


def _resolve_train_dir(root_dir: Path, run_name: str) -> Path:
    exact_dir = root_dir / run_name
    if exact_dir.exists():
        return exact_dir
    candidates = sorted(
        root_dir.glob(f"{run_name}*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Could not find NequIP train directory for run_name={run_name} under {root_dir}")
    return candidates[0]


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

    config_path = Path(state.get("train_config_path", str(workdir / "full.auto.yaml"))).resolve()

    total_count = _load_structures_count(dataset_path)
    n_train, n_val = _compute_split_sizes(
        total_count=total_count,
        train_n_train=state.get("train_n_train"),
        train_n_val=state.get("train_n_val"),
        train_val_ratio=float(state.get("train_val_ratio", 0.1)),
    )
    with template_path.open("r", encoding="utf-8") as handle:
        template_config = yaml.load(handle, Loader=_TupleSafeLoader)
    if not isinstance(template_config, dict):
        raise ValueError(f"NequIP config template is not a mapping: {template_path}")
    base_seed = int(template_config.get("seed", 123))
    model_seeds = _resolve_model_seeds(state, base_seed)

    run_training = bool(state.get("train_run", True))
    deploy_run = bool(state.get("deploy_run", run_training))
    nequip_bin_dir = state.get("nequip_bin_dir")
    train_command_parts_base = _resolve_command_parts(
        state.get("nequip_command", "nequip-train"),
        nequip_bin_dir,
        "nequip-train",
    )
    deploy_command_parts_base = _resolve_command_parts(
        state.get("nequip_deploy_command", "nequip-deploy"),
        nequip_bin_dir,
        "nequip-deploy",
    )
    model_config_paths: List[str] = []
    model_log_paths: List[str] = []
    model_cmd_parts: List[List[str]] = []
    model_root_dirs: List[str] = []
    model_run_names: List[str] = []
    model_run_dirs: List[str] = []
    deployed_model_paths: List[str] = []
    base_stem = config_path.stem
    base_suffix = config_path.suffix or ".yaml"
    for model_index, seed in enumerate(model_seeds, start=1):
        seed_suffix = f"seed{seed}"
        model_config_path = config_path.with_name(f"{base_stem}.{seed_suffix}{base_suffix}")
        prepared_config = _prepare_nequip_config(
            template_path=template_path,
            output_config_path=model_config_path,
            dataset_extxyz_path=dataset_path,
            workdir=workdir,
            n_train=n_train,
            n_val=n_val,
            train_root=state.get("train_root"),
            seed=seed,
            run_name_suffix=f"m{model_index}_{seed_suffix}",
        )
        model_config_paths.append(str(model_config_path))
        model_cmd_parts.append(train_command_parts_base + [str(model_config_path)])
        model_root_dirs.append(str(Path(prepared_config["root"]).resolve()))
        model_run_names.append(str(prepared_config["run_name"]))

        model_log_path = workdir / f"nequip_train_{seed_suffix}.log"
        model_log_paths.append(str(model_log_path))
    if run_training:
        processes: List[subprocess.Popen] = []
        for cmd_parts in model_cmd_parts:
            processes.append(
                subprocess.Popen(
                    cmd_parts,
                    cwd=str(workdir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )

        failures: List[Tuple[List[str], Path, int]] = []
        for cmd_parts, model_log_path, process in zip(model_cmd_parts, model_log_paths, processes):
            stdout_text, stderr_text = process.communicate()
            Path(model_log_path).write_text(
                (stdout_text or "") + "\n" + (stderr_text or ""),
                encoding="utf-8",
            )
            if process.returncode != 0:
                failures.append((cmd_parts, Path(model_log_path), int(process.returncode)))

        if failures:
            cmd_parts, model_log_path, return_code = failures[0]
            raise RuntimeError(
                "NequIP training failed.\n"
                f"Command: {' '.join(cmd_parts)}\n"
                f"Return code: {return_code}\n"
                f"See log: {model_log_path}"
            )

    for model_root_dir, model_run_name in zip(model_root_dirs, model_run_names):
        model_run_dir = _resolve_train_dir(Path(model_root_dir), model_run_name)
        model_run_dirs.append(str(model_run_dir))

    if deploy_run:
        for seed, model_run_dir in zip(model_seeds, model_run_dirs):
            deploy_out_path = workdir / f"deployed_model_seed{seed}.pth"
            deploy_log_path = workdir / f"nequip_deploy_seed{seed}.log"
            deploy_cmd_parts = deploy_command_parts_base + [
                "build",
                "--train-dir",
                model_run_dir,
                str(deploy_out_path),
            ]
            completed = subprocess.run(
                deploy_cmd_parts,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                check=False,
            )
            deploy_log_path.write_text(
                (completed.stdout or "") + "\n" + (completed.stderr or ""),
                encoding="utf-8",
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "NequIP deploy failed.\n"
                    f"Command: {' '.join(deploy_cmd_parts)}\n"
                    f"See log: {deploy_log_path}"
                )
            deployed_model_paths.append(str(deploy_out_path))

    note = (
        f"Prepared {len(model_seeds)} NequIP configs with dataset={dataset_path.name}, "
        f"n_train={n_train}, n_val={n_val}, seeds={model_seeds}."
    )
    if run_training:
        note += " Ensemble training completed in parallel."
    else:
        note += " Training skipped (train_run=false)."
    if deploy_run:
        note += " Deployment completed."
    else:
        note += " Deployment skipped (deploy_run=false)."

    return {
        "train_config_path": model_config_paths[0],
        "train_dataset_extxyz": str(dataset_path),
        "train_n_train": n_train,
        "train_n_val": n_val,
        "train_num_models": len(model_seeds),
        "train_model_seeds": model_seeds,
        "train_model_config_paths": model_config_paths,
        "train_workdir": str(workdir),
        "train_log_path": model_log_paths[0],
        "train_model_log_paths": model_log_paths,
        "train_model_run_dirs": model_run_dirs,
        "deployed_model_paths": deployed_model_paths,
        "notes": note,
    }
