from pathlib import Path
import shutil
import subprocess

from openmlp.state import PipelineState


def qm_calculation_node(state: PipelineState) -> PipelineState:
    qm_input_extxyz = state.get("qm_input_extxyz") or state.get("output_structures")
    if not qm_input_extxyz:
        raise ValueError("qm_input_extxyz is required or provide output_structures from step 1.")

    qm_input_path = Path(qm_input_extxyz).resolve()
    if not qm_input_path.exists():
        raise FileNotFoundError(f"QM input extxyz not found: {qm_input_path}")

    calc_type = state.get("qm_calc_type", "sp")
    calculator_type = state.get("qm_calculator_type", "orca")
    n_core = int(state.get("qm_n_core", 24))
    orca_path = state.get("qm_orca_path")
    if calculator_type.lower() == "orca" and not orca_path:
        raise ValueError("qm_orca_path is required when qm_calculator_type is 'orca'.")
    if calculator_type.lower() != "orca":
        orca_path = orca_path or "unused"

    workdir = Path(state.get("qm_workdir", str(qm_input_path.parent))).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    runner_script = Path(__file__).resolve().parent / "qm_calc" / "runCalculateGeomWithQM.py"
    if not runner_script.exists():
        raise FileNotFoundError(f"QM runner script not found: {runner_script}")

    staged_input_path = workdir / qm_input_path.name
    if qm_input_path.resolve() != staged_input_path.resolve():
        shutil.copy2(str(qm_input_path), str(staged_input_path))

    command = [
        "python",
        str(runner_script),
        "-in_extxyz",
        str(staged_input_path),
        "-orca_path",
        str(orca_path),
        "-calc_type",
        calc_type,
        "-calculator_type",
        calculator_type,
        "-n_core",
        str(n_core),
    ]

    completed = subprocess.run(
        command,
        cwd=str(workdir),
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "QM calculation step failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    input_name = staged_input_path.name
    qm_output_extxyz = workdir / f"{calc_type}_{input_name}"
    qm_output_csv = workdir / f"{calc_type}_{input_name.replace('.extxyz', '.csv')}"

    return {
        "qm_output_extxyz": str(qm_output_extxyz),
        "qm_output_csv": str(qm_output_csv),
        "notes": f"QM calculation finished for {input_name} in {workdir}.",
    }
