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
    train_config_template: str
    train_config_path: str
    train_dataset_extxyz: str
    train_workdir: str
    train_n_train: int
    train_n_val: int
    train_val_ratio: float
    train_num_models: int
    train_model_seeds: List[int]
    train_root: str
    train_run: bool
    nequip_command: str
    train_log_path: str
    train_model_config_paths: List[str]
    train_model_log_paths: List[str]
    notes: str
