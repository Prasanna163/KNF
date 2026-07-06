from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class RunOptions:
    charge: int = 0
    spin: int = 1
    water: bool = False
    force: bool = False
    clean: bool = False
    debug: bool = False
    processing: Literal["auto", "single", "multi"] = "auto"
    multi: bool = False
    single: bool = False
    workers: int | None = None
    output_dir: str | None = None
    batches: int | None = None
    universal_kuid: bool = False
    merge_master_csv: str | None = None
    merge_new_csv: str | None = None
    merge_output_dir: str | None = None
    overwrite_master_csv: bool = False
    ram_per_job: float = 50.0
    refresh_autoconfig: bool = False
    quiet_config: bool = False
    full_files: bool = False
    enable_stop_key: bool = False
    interactive_quadrant_plot: bool = False
    atlas_bundle: bool = False
    gpu: bool = False
    cpu: bool = False
    multiwfn: bool = False
    nci_backend: Literal["torch", "multiwfn"] = "torch"
    nci_grid_spacing: float = 0.2
    nci_grid_padding: float = 3.0
    nci_device: Literal["cpu", "cuda"] = "cpu"
    nci_dtype: Literal["float32", "float64"] = "float32"
    nci_batch_size: int = 250000
    nci_eig_batch_size: int = 200000
    nci_rho_floor: float = 1e-12
    nci_apply_primitive_norm: bool = False
    scdi_var_min: float | None = None
    scdi_var_max: float | None = None
    wbo_mode: Literal["native", "xtb"] = "native"
    preopt: Literal["uff", "geoinit"] = "geoinit"
    xtb_engine: Literal["xtb", "xtbx", "auto"] = "xtbx"
    xtb_gpu_atoms: int = 350
    refresh_first_run: bool = False
    multiwfn_path: str | None = None
    knf: bool = False
    project_root: str | None = None


@dataclass(frozen=True)
class SingleFileResult:
    input_file: str
    success: bool
    error: str | None
    elapsed_seconds: float
    output_root: str | None = None
    knf: dict | None = None
    kuid_summary: dict | None = None


@dataclass(frozen=True)
class BatchFileRecord:
    input_file: str
    status: Literal["success", "failed", "stopped"]
    elapsed_seconds: float
    error: str | None = None
    knf: dict | None = None


@dataclass(frozen=True)
class BatchResult:
    input_directory: str
    results_root: str
    records: list[BatchFileRecord]
    mode: str
    workers: int
    total_time_seconds: float
    aggregate_json_path: str | None = None
    aggregate_csv_path: str | None = None
    batch_delta_json_path: str | None = None
    batch_delta_txt_path: str | None = None
    quadrant_payload: dict | None = None
    skipped_existing: int = 0
    stopped_count: int = 0
    failures: list[tuple[str, str]] | None = None
