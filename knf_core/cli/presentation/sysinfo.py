"""Best-effort host + run-configuration facts for the dashboard specs panels.

Everything here is defensive: a missing dependency or an odd platform degrades
to a sensible placeholder rather than raising, because this is decoration around
the actual calculation.
"""

from __future__ import annotations

import os
import platform
import sys

from .formatting import display_path


def _human_bytes(num: float) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            precision = 0 if unit in {"B", "KB"} else 1
            return f"{value:.{precision}f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def os_label() -> str:
    system = platform.system()
    if system == "Windows":
        release = platform.release()
        try:  # Windows 11 still reports release "10"; disambiguate by build.
            build = int(platform.version().split(".")[-1])
            if build >= 22000:
                release = "11"
        except Exception:
            pass
        return f"Windows {release}"
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0] or ''}".strip()
    return f"{system} {platform.release()}".strip()


def cpu_label() -> str:
    brand = ""
    try:
        if os.name == "nt":
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                brand = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except Exception:
        brand = ""
    if not brand:
        brand = platform.processor() or platform.machine() or "CPU"

    cores = ""
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
        if physical and logical:
            cores = f"  ({physical}C / {logical}T)"
    except Exception:
        try:
            logical = os.cpu_count()
            if logical:
                cores = f"  ({logical}T)"
        except Exception:
            cores = ""
    return f"{brand}{cores}"


def memory_label() -> str:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return f"{_human_bytes(vm.total)}  ({_human_bytes(vm.available)} free)"
    except Exception:
        return "n/a"


def gpu_label(args=None) -> str:
    device = str(getattr(args, "nci_device", "cpu") or "cpu").lower()
    want_gpu = bool(getattr(args, "gpu", False)) or device == "cuda"
    torch = sys.modules.get("torch")
    if torch is None and want_gpu:
        try:
            import torch  # noqa: PLC0415 - lazy; heavy import only when GPU is in play
        except Exception:
            torch = None
    if torch is not None:
        try:
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                return f"{name}  (CUDA)"
        except Exception:
            pass
    return "CPU only"


def system_rows(args=None) -> list[tuple[str, str]]:
    return [
        ("OS", os_label()),
        ("CPU", cpu_label()),
        ("Memory", memory_label()),
        ("GPU", gpu_label(args)),
        ("Python", platform.python_version()),
    ]


def _mode_label(args, mode: str | None) -> str:
    resolved = mode or getattr(args, "processing", None) or "auto"
    return str(resolved)


def run_config_rows(
    args,
    *,
    mode: str | None = None,
    workers: int | None = None,
    results_root: str | None = None,
) -> list[tuple[str, str]]:
    backend = str(getattr(args, "nci_backend", "torch") or "torch")
    device = str(getattr(args, "nci_device", "cpu") or "cpu")
    backend_label = f"{backend} ({device})" if backend == "torch" else backend

    water = bool(getattr(args, "water", False))
    solvent = "alpb water" if water else "cosmo water"

    worker_value = workers if workers is not None else getattr(args, "workers", None)
    workers_label = str(worker_value) if worker_value else "auto"

    rows = [
        ("Mode", _mode_label(args, mode)),
        ("NCI backend", backend_label),
        ("xTB engine", str(getattr(args, "xtb_engine", "xtbx") or "xtbx")),
        ("Pre-opt", str(getattr(args, "preopt", "geoinit") or "geoinit")),
        ("Workers", workers_label),
        ("Charge / Spin", f"{getattr(args, 'charge', 0)} / {getattr(args, 'spin', 1)}"),
        ("Solvent", solvent),
    ]
    if getattr(args, "sp", False):
        rows.append(("SP-only", "yes"))
    root = results_root or getattr(args, "output_dir", None)
    if root:
        rows.append(("Output", display_path(str(root))))
    return rows
