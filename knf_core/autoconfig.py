import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import psutil


AUTO_CONFIG_FILENAME = ".knf_autoconfig.json"
DEFAULT_RAM_PER_JOB_MB = 50.0
RAM_HEADROOM_FRACTION = 0.75
CPU_DIVISOR = 2.5
MAX_WORKERS = 32


@dataclass
class MultiConfig:
    workers: int
    omp_num_threads: int
    physical_cores: int
    logical_threads: int
    available_ram_mb: float
    ram_per_job_mb: float
    source: str
    limiting_factor: str

    @property
    def total_threads(self) -> int:
        return self.workers * self.omp_num_threads

    @property
    def utilization_pct(self) -> float:
        if self.logical_threads <= 0:
            return 0.0
        used = min(self.total_threads, self.logical_threads)
        return (used / self.logical_threads) * 100.0

    @property
    def env_overrides(self) -> dict:
        omp = str(self.omp_num_threads)
        return {
            "OMP_NUM_THREADS": omp,
            "MKL_NUM_THREADS": omp,
            "OPENBLAS_NUM_THREADS": omp,
            "VECLIB_MAXIMUM_THREADS": omp,
            "NUMEXPR_NUM_THREADS": omp,
        }


def _detect_machine_signature() -> dict:
    vm = psutil.virtual_memory()
    return {
        "physical_cores": psutil.cpu_count(logical=False) or 1,
        "logical_threads": psutil.cpu_count(logical=True) or 1,
        "total_ram_mb": int(vm.total / (1024 * 1024)),
    }


def _config_path(project_root: Optional[str] = None) -> Path:
    if project_root:
        return Path(project_root) / AUTO_CONFIG_FILENAME
    return Path.cwd() / AUTO_CONFIG_FILENAME


def _load_cached_config(project_root: Optional[str] = None) -> Optional[dict]:
    path = _config_path(project_root)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_cached_config(payload: dict, project_root: Optional[str] = None) -> None:
    path = _config_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _find_best_workers_from_benchmark(project_root: Optional[str] = None) -> Optional[int]:
    root = Path(project_root) if project_root else Path.cwd()
    bench_root = root / "benchmarks"
    if not bench_root.exists():
        return None

    metrics_files = sorted(bench_root.glob("profile_*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for metrics_file in metrics_files:
        try:
            data = json.loads(metrics_file.read_text(encoding="utf-8"))
            batch = data.get("batch_workers_monitor", {})
            candidates = []
            for key, val in batch.items():
                wall = float(val.get("wall_time_sec", 0.0))
                if wall > 0:
                    candidates.append((wall, int(key)))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                return candidates[0][1]
        except Exception:
            continue
    return None


def _workers_by_cpu(physical_cores: int) -> int:
    return max(1, min(MAX_WORKERS, math.floor(physical_cores / CPU_DIVISOR)))


def _workers_by_ram(available_ram_mb: float, ram_per_job_mb: float) -> int:
    budget = available_ram_mb * RAM_HEADROOM_FRACTION
    if ram_per_job_mb <= 0:
        return MAX_WORKERS
    return max(1, min(MAX_WORKERS, int(budget / ram_per_job_mb)))


def _build_config(
    preferred_workers: int,
    n_jobs: int,
    machine: dict,
    ram_per_job_mb: float,
    source: str,
) -> MultiConfig:
    available_ram_mb = psutil.virtual_memory().available / (1024 * 1024)
    physical = int(machine["physical_cores"])
    logical = int(machine["logical_threads"])

    cpu_limit = _workers_by_cpu(physical)
    ram_limit = _workers_by_ram(available_ram_mb, ram_per_job_mb)
    job_limit = n_jobs if n_jobs > 0 else preferred_workers

    workers = max(1, min(preferred_workers, cpu_limit, ram_limit, job_limit, MAX_WORKERS))

    if workers == job_limit and job_limit < min(preferred_workers, cpu_limit, ram_limit):
        limiting = "job_count"
    elif workers == ram_limit and ram_limit < min(preferred_workers, cpu_limit):
        limiting = "ram"
    elif workers == cpu_limit and cpu_limit < preferred_workers:
        limiting = "cpu"
    else:
        limiting = "preferred"

    omp = max(1, math.floor(logical / workers))
    return MultiConfig(
        workers=workers,
        omp_num_threads=omp,
        physical_cores=physical,
        logical_threads=logical,
        available_ram_mb=available_ram_mb,
        ram_per_job_mb=ram_per_job_mb,
        source=source,
        limiting_factor=limiting,
    )


def resolve_multi_config(
    n_jobs: int,
    ram_per_job_mb: float = DEFAULT_RAM_PER_JOB_MB,
    project_root: Optional[str] = None,
    force_refresh: bool = False,
) -> MultiConfig:
    machine = _detect_machine_signature()
    cache = None if force_refresh else _load_cached_config(project_root)

    if cache and cache.get("machine_signature") == machine:
        preferred_workers = int(cache.get("preferred_workers", 1))
        return _build_config(
            preferred_workers=preferred_workers,
            n_jobs=n_jobs,
            machine=machine,
            ram_per_job_mb=ram_per_job_mb,
            source="cached",
        )

    benchmark_workers = _find_best_workers_from_benchmark(project_root)
    if benchmark_workers:
        preferred_workers = benchmark_workers
        source = "benchmark"
    else:
        preferred_workers = _workers_by_cpu(machine["physical_cores"])
        source = "heuristic"

    _save_cached_config(
        {
            "machine_signature": machine,
            "preferred_workers": preferred_workers,
            "source": source,
        },
        project_root=project_root,
    )

    return _build_config(
        preferred_workers=preferred_workers,
        n_jobs=n_jobs,
        machine=machine,
        ram_per_job_mb=ram_per_job_mb,
        source=f"{source}:saved",
    )


def apply_env(config: MultiConfig) -> dict:
    env = os.environ.copy()
    env.update(config.env_overrides)
    return env


def apply_env_inplace(config: MultiConfig) -> None:
    os.environ.update(config.env_overrides)


def print_config(config: MultiConfig) -> None:
    print("")
    print("KNF-Core Auto Multi Configuration")
    print("---------------------------------")
    print(f"Source:          {config.source}")
    print(f"CPU:             {config.physical_cores} physical / {config.logical_threads} logical")
    print(f"Available RAM:   {config.available_ram_mb:.0f} MB")
    print(f"Workers:         {config.workers} (limit: {config.limiting_factor})")
    print(f"OMP threads:     {config.omp_num_threads} per process")
    print(f"Total threads:   {config.total_threads} ({config.utilization_pct:.0f}% of logical)")
    print(f"RAM per job est: {config.ram_per_job_mb:.0f} MB")
    print(f"Config file:     {_config_path()}")
    print("")


def get_cached_payload(project_root: Optional[str] = None) -> Optional[dict]:
    return _load_cached_config(project_root)
