import gc
import time
from typing import Dict, Optional

import torch

from .engine import NCIConfig, run_nci_engine
from .export import write_nci_grid_npz, write_nci_grid_text
from .grid import build_grid
from .molden import parse_molden


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def is_cuda_oom_error(exc: BaseException) -> bool:
    cuda_module = getattr(torch, "cuda", None)
    oom_type = getattr(cuda_module, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True

    msg = str(exc).strip().lower()
    if "out of memory" not in msg:
        return False
    return any(token in msg for token in ("cuda", "cublas", "cudnn", "nvrtc", "gpu"))


def release_cuda_memory(device: Optional[torch.device] = None) -> None:
    if not torch.cuda.is_available():
        return
    try:
        if device is None:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(device)
    except Exception:
        # Best-effort sync before allocator cleanup.
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def run_nci_torch(
    molden_path: str,
    output_path: str,
    output_text_path: Optional[str] = None,
    spacing_angstrom: float = 0.2,
    padding_angstrom: float = 3.0,
    device: str = "auto",
    dtype: str = "float32",
    batch_size: int = 250000,
    eig_batch_size: int = 200000,
    rho_floor: float = 1e-12,
    output_units: str = "bohr",
    apply_primitive_normalization: bool = False,
    cpu_threads: Optional[int] = None,
) -> Dict[str, object]:
    t0 = time.perf_counter()
    resolved_device = None
    wavefunction = None
    grid = None
    fields = None
    cpu_threads_configured = None
    device_requested = (device or "auto").strip().lower()

    if device_requested == "cpu" and cpu_threads:
        target_threads = max(1, int(cpu_threads))
        try:
            torch.set_num_threads(target_threads)
            cpu_threads_configured = int(torch.get_num_threads())
        except Exception:
            cpu_threads_configured = None

    try:
        t_parse0 = time.perf_counter()
        wavefunction = parse_molden(
            molden_path,
            apply_primitive_normalization=apply_primitive_normalization,
        )
        t_parse1 = time.perf_counter()

        t_grid0 = time.perf_counter()
        grid = build_grid(
            atoms_bohr=wavefunction.atoms_bohr,
            spacing_angstrom=spacing_angstrom,
            padding_angstrom=padding_angstrom,
        )
        t_grid1 = time.perf_counter()

        t_compute0 = time.perf_counter()
        fields, resolved_device = run_nci_engine(
            wavefunction=wavefunction,
            grid=grid,
            config=NCIConfig(
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                eig_batch_size=eig_batch_size,
                rho_floor=rho_floor,
            ),
        )
        _sync_if_cuda(resolved_device)
        t_compute1 = time.perf_counter()

        t_export0 = time.perf_counter()
        write_nci_grid_npz(
            output_path=output_path,
            grid=grid,
            sign_lambda2_rho=fields.sign_lambda2_rho,
            rdg=fields.rdg,
            output_units=output_units,
        )
        if output_text_path:
            write_nci_grid_text(
                output_path=output_text_path,
                grid=grid,
                sign_lambda2_rho=fields.sign_lambda2_rho,
                rdg=fields.rdg,
                output_units=output_units,
            )
        _sync_if_cuda(resolved_device)
        t_export1 = time.perf_counter()

        elapsed = time.perf_counter() - t0
        gpu_meta = {
            "cuda_available": bool(torch.cuda.is_available()),
            "is_cuda_run": bool(resolved_device.type == "cuda"),
            "device_requested": device,
            "device_resolved": str(resolved_device),
        }
        if resolved_device.type == "cuda":
            gpu_meta["cuda_device_name"] = torch.cuda.get_device_name(resolved_device)
            gpu_meta["cuda_device_capability"] = list(torch.cuda.get_device_capability(resolved_device))

        return {
            "device": str(resolved_device),
            "elapsed_seconds": elapsed,
            "timings_seconds": {
                "parse_molden": float(t_parse1 - t_parse0),
                "build_grid": float(t_grid1 - t_grid0),
                "compute_fields": float(t_compute1 - t_compute0),
                "export_grid": float(t_export1 - t_export0),
            },
            "gpu": gpu_meta,
            "n_atoms": int(wavefunction.atoms_bohr.shape[0]),
            "n_basis": int(len(wavefunction.basis_functions)),
            "grid_shape": grid.shape,
            "n_grid_points": int(grid.n_points),
            "apply_primitive_normalization": bool(apply_primitive_normalization),
            "eig_batch_size": int(eig_batch_size),
            "cpu_threads_configured": cpu_threads_configured,
            "output_binary_path": output_path,
            "output_text_path": output_text_path,
        }
    finally:
        del fields
        del grid
        del wavefunction
        gc.collect()
        if (resolved_device is not None and resolved_device.type == "cuda") or (
            resolved_device is None
            and (device_requested == "auto" or device_requested.startswith("cuda"))
            and torch.cuda.is_available()
        ):
            release_cuda_memory(resolved_device)
