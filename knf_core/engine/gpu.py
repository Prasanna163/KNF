import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone

from .. import utils
from .timeutil import _parse_iso_z, _utc_now_iso_z

GPU_SETUP_STATE_FILE = os.path.join(os.path.expanduser("~"), ".knf_gpu_setup_state.json")
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
GPU_RUNTIME_CACHE_MAX_AGE_SECONDS = 12 * 60 * 60

# Guards the GPU setup state file's load-modify-save cycle and the CUDA
# install prompt so concurrent callers (e.g. api.py's job worker threads)
# can't race each other into a double `pip install` or a torn state file.
_GPU_STATE_LOCK = threading.Lock()


def _gpu_state_key() -> str:
    try:
        return os.path.normcase(os.path.abspath(sys.executable))
    except Exception:
        return str(sys.executable)


def _load_gpu_setup_state() -> dict:
    try:
        if not os.path.exists(GPU_SETUP_STATE_FILE):
            return {}
        with open(GPU_SETUP_STATE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_gpu_setup_state(state: dict) -> None:
    try:
        parent = os.path.dirname(GPU_SETUP_STATE_FILE)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = f"{GPU_SETUP_STATE_FILE}.tmp-{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, GPU_SETUP_STATE_FILE)
    except Exception as e:
        logging.debug("Could not persist GPU setup state: %s", e)


def _nvidia_gpu_probe() -> dict:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"has_gpu": False, "reason": "nvidia-smi not found in PATH."}

    try:
        proc = subprocess.run(
            [smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as e:
        return {"has_gpu": False, "reason": f"nvidia-smi check failed: {e}"}

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if proc.returncode != 0:
        msg = stderr or stdout or f"exit code {proc.returncode}"
        return {"has_gpu": False, "reason": f"nvidia-smi returned an error: {msg}"}

    if not stdout or "no devices were found" in stdout.lower():
        return {"has_gpu": False, "reason": "No CUDA-capable NVIDIA GPU detected."}

    names = [line.strip() for line in stdout.splitlines() if line.strip()]
    return {
        "has_gpu": bool(names),
        "reason": "" if names else "No CUDA-capable NVIDIA GPU detected.",
        "gpu_names": names,
    }


def _probe_torch_cuda_runtime() -> dict:
    probe_code = (
        "import json\n"
        "out = {}\n"
        "try:\n"
        "    import torch\n"
        "    out['torch_import_ok'] = True\n"
        "    out['torch_version'] = getattr(torch, '__version__', '')\n"
        "    out['torch_cuda_version'] = getattr(getattr(torch, 'version', None), 'cuda', None)\n"
        "    out['cuda_available'] = bool(torch.cuda.is_available())\n"
        "    out['cuda_device_count'] = int(torch.cuda.device_count()) if out['cuda_available'] else 0\n"
        "    out['cuda_device_name'] = torch.cuda.get_device_name(0) if out['cuda_available'] else ''\n"
        "except Exception as e:\n"
        "    out = {\n"
        "        'torch_import_ok': False,\n"
        "        'torch_version': '',\n"
        "        'torch_cuda_version': None,\n"
        "        'cuda_available': False,\n"
        "        'cuda_device_count': 0,\n"
        "        'cuda_device_name': '',\n"
        "        'error': str(e),\n"
        "    }\n"
        "print(json.dumps(out))\n"
    )

    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as e:
        return {
            "torch_import_ok": False,
            "torch_version": "",
            "torch_cuda_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": "",
            "error": str(e),
        }

    if proc.returncode != 0:
        return {
            "torch_import_ok": False,
            "torch_version": "",
            "torch_cuda_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": "",
            "error": (proc.stderr or proc.stdout or f"probe exit code {proc.returncode}").strip(),
        }

    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception:
        payload = {
            "torch_import_ok": False,
            "torch_version": "",
            "torch_cuda_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": "",
            "error": "Unable to parse torch probe output.",
        }
    if not isinstance(payload, dict):
        payload = {
            "torch_import_ok": False,
            "torch_version": "",
            "torch_cuda_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": "",
            "error": "Unexpected torch probe payload type.",
        }
    return payload


def _prompt_yes_no(question: str, default: str = "n"):
    default = (default or "n").strip().lower()
    if default not in {"y", "n"}:
        default = "n"

    if not sys.stdin or not sys.stdin.isatty():
        return None

    suffix = "[Y/n]" if default == "y" else "[y/N]"
    while True:
        answer = input(f"{question} {suffix}: ").strip().lower()
        if not answer:
            return default == "y"
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter 'y' or 'n'.")


def _install_cuda_torch() -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "torch",
        "torchvision",
        "torchaudio",
        "--index-url",
        PYTORCH_CUDA_INDEX_URL,
    ]
    try:
        proc = subprocess.run(cmd, check=False)
    except Exception as e:
        return False, str(e)
    if proc.returncode != 0:
        return False, f"pip install exited with code {proc.returncode}"
    return True, ""


def ensure_cuda_runtime_for_gpu_mode(allow_prompt: bool = True) -> None:
    with _GPU_STATE_LOCK:
        nvidia = _nvidia_gpu_probe()
        has_gpu = bool(nvidia.get("has_gpu"))
        if not has_gpu:
            raise RuntimeError(
                "GPU mode requested, but no CUDA-capable NVIDIA GPU was detected. "
                f"Details: {nvidia.get('reason', 'unknown')}"
            )

        state = _load_gpu_setup_state()
        py_key = _gpu_state_key()
        per_python = state.get("by_python")
        if not isinstance(per_python, dict):
            per_python = {}
        entry = per_python.get(py_key)
        if not isinstance(entry, dict):
            entry = {}

        first_gpu_check = not bool(entry.get("gpu_checked"))
        now_utc = datetime.now(timezone.utc)
        cached_checked_at = _parse_iso_z(entry.get("last_checked_at"))
        cache_age_ok = bool(
            cached_checked_at
            and (now_utc - cached_checked_at).total_seconds() <= GPU_RUNTIME_CACHE_MAX_AGE_SECONDS
        )
        cached_names = entry.get("gpu_names") if isinstance(entry.get("gpu_names"), list) else []
        gpu_names_now = nvidia.get("gpu_names", [])
        gpu_identity_same = bool(cached_names) and cached_names == gpu_names_now
        can_reuse_cached = bool(
            cache_age_ok
            and gpu_identity_same
            and entry.get("torch_cuda_version")
            and entry.get("cuda_available") is True
        )

        if can_reuse_cached:
            torch_info = {
                "torch_version": entry.get("torch_version", ""),
                "torch_cuda_version": entry.get("torch_cuda_version"),
                "cuda_available": bool(entry.get("cuda_available")),
                "cuda_device_name": (gpu_names_now[0] if gpu_names_now else ""),
            }
        else:
            torch_info = _probe_torch_cuda_runtime()

        torch_has_cuda_build = bool(torch_info.get("torch_cuda_version"))
        cuda_available = bool(torch_info.get("cuda_available"))

        entry.update(
            {
                "gpu_checked": True,
                "last_checked_at": _utc_now_iso_z(),
                "gpu_names": nvidia.get("gpu_names", []),
                "torch_version": torch_info.get("torch_version", ""),
                "torch_cuda_version": torch_info.get("torch_cuda_version"),
                "cuda_available": cuda_available,
            }
        )

        if not torch_has_cuda_build:
            can_prompt = bool(allow_prompt and first_gpu_check)
            install_choice = None

            if can_prompt:
                print(
                    "\nGPU detected, but the current PyTorch build does not include CUDA support.\n"
                    f"Python: {sys.executable}\n"
                    "A CUDA-enabled PyTorch build is required for --gpu."
                )
                yn = _prompt_yes_no("Install CUDA-enabled PyTorch now?", default="n")
                if yn is None:
                    print(
                        "Cannot prompt for installation in this session (non-interactive stdin). "
                        "Please install CUDA-enabled PyTorch manually."
                    )
                    install_choice = "no_prompt_available"
                elif yn:
                    install_choice = "yes"
                    print("Installing CUDA-enabled PyTorch...")
                    ok, err = _install_cuda_torch()
                    if not ok:
                        entry["install_attempt"] = "failed"
                        entry["install_error"] = err
                        per_python[py_key] = entry
                        state["by_python"] = per_python
                        _save_gpu_setup_state(state)
                        raise RuntimeError(
                            "CUDA PyTorch installation failed. "
                            f"Reason: {err}. "
                            f"Try manually: {sys.executable} -m pip install --upgrade "
                            f"torch torchvision torchaudio --index-url {PYTORCH_CUDA_INDEX_URL}"
                        )
                    torch_info = _probe_torch_cuda_runtime()
                    torch_has_cuda_build = bool(torch_info.get("torch_cuda_version"))
                    cuda_available = bool(torch_info.get("cuda_available"))
                    entry["install_attempt"] = "succeeded"
                else:
                    install_choice = "no"

            if install_choice:
                entry["first_prompt_choice"] = install_choice

        per_python[py_key] = entry
        state["by_python"] = per_python
        _save_gpu_setup_state(state)

        if not bool(torch_info.get("torch_cuda_version")):
            raise RuntimeError(
                "GPU mode requested and NVIDIA GPU detected, but PyTorch CUDA build is not available. "
                f"Install with: {sys.executable} -m pip install --upgrade torch torchvision torchaudio "
                f"--index-url {PYTORCH_CUDA_INDEX_URL}"
            )

        if not bool(torch_info.get("cuda_available")):
            raise RuntimeError(
                "GPU mode requested and CUDA PyTorch appears installed, but torch.cuda.is_available() is False. "
                "Please verify NVIDIA driver/CUDA runtime compatibility."
            )


def _is_torch_available() -> tuple[bool, str]:
    try:
        import torch  # type: ignore  # noqa: F401
        return True, ""
    except Exception as e:
        return False, str(e)


def resolve_cpu_backend_when_torch_missing(
    args,
    *,
    is_torch_available=None,
    ensure_cuda_runtime=None,
) -> None:
    is_torch_available = is_torch_available or _is_torch_available
    ensure_cuda_runtime = ensure_cuda_runtime or ensure_cuda_runtime_for_gpu_mode

    backend = (getattr(args, "nci_backend", "torch") or "torch").strip().lower()
    device = (getattr(args, "nci_device", "cpu") or "cpu").strip().lower()
    if backend != "torch":
        return

    torch_ok, torch_err = is_torch_available()
    if torch_ok:
        return

    if device == "cuda" or bool(getattr(args, "gpu", False)):
        ensure_cuda_runtime(allow_prompt=True)
        torch_ok, torch_err = is_torch_available()
        if torch_ok:
            return
        raise RuntimeError(
            "GPU mode requires PyTorch with CUDA support, but PyTorch is still "
            "not available after setup. "
            f"Import error: {torch_err}"
        )

    # CPU mode: if torch is missing, auto-fallback to Multiwfn CPU backend.
    utils.ensure_multiwfn_in_path(explicit_path=getattr(args, "multiwfn_path", None))
    if shutil.which("Multiwfn") or shutil.which("Multiwfn.exe"):
        args.nci_backend = "multiwfn"
        args.nci_device = "cpu"
        logging.warning(
            "PyTorch not available; falling back to Multiwfn CPU backend. "
            "Install torch to use the torch backend."
        )
        return

    raise RuntimeError(
        "PyTorch is not available for torch CPU backend and Multiwfn was not found for CPU fallback. "
        "Install torch (CPU) or install Multiwfn and run with --cpu/--multiwfn."
    )
