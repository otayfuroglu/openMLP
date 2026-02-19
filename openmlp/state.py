from typing import List, TypedDict


class PipelineState(TypedDict, total=False):
    input_structure: str
    output_structures: str
    n_structures: int
    scale_min: float
    scale_max: float
    max_atom_displacement: float
    displacement_attempts: int
    generated_count: int
    generated_files: List[str]
    qm_input_extxyz: str
    qm_orca_path: str
    qm_calc_type: str
    qm_calculator_type: str
    qm_n_core: int
    qm_workdir: str
    qm_output_extxyz: str
    qm_output_csv: str
    notes: str
