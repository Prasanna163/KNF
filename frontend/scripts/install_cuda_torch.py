"""Install an optional, isolated CUDA PyTorch layer for NCIForge."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


TORCH_VERSION = "2.11.0"
CUDA_VARIANT = "cu128"
CUDA_INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_VARIANT}"


def _runtime_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("LOCALAPPDATA is not defined.")
    return Path(local_app_data) / "NCIForge" / "runtime"


def _probe_nvidia() -> list[str]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        raise RuntimeError("No NVIDIA GPU detected: nvidia-smi.exe was not found.")
    proc = subprocess.run(
        [smi, "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or not names:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
        raise RuntimeError(f"No usable NVIDIA GPU detected: {detail}")
    return names


def _validate(site_packages: Path) -> dict:
    code = (
        "import json, torch\n"
        "value = torch.ones(1024, device='cuda').sum().item()\n"
        "print(json.dumps({"
        "'torch_version': torch.__version__, "
        "'torch_cuda_version': torch.version.cuda, "
        "'cuda_available': torch.cuda.is_available(), "
        "'device_name': torch.cuda.get_device_name(0), "
        "'tensor_check': value"
        "}))\n"
    )
    env = {
        **os.environ,
        "NCIFORGE_CUDA_SITE_PACKAGES": str(site_packages),
        "PYTHONNOUSERSITE": "1",
        "PYTHONUTF8": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "CUDA validation failed.")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    if not payload.get("cuda_available") or payload.get("tensor_check") != 1024.0:
        raise RuntimeError(f"CUDA validation returned an unexpected result: {payload}")
    return payload


def install() -> dict:
    gpu_names = _probe_nvidia()
    root = _runtime_root()
    target = root / "cuda-site-packages"
    staging = root / f".cuda-staging-{uuid.uuid4().hex}"
    backup = root / f".cuda-backup-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)

    print(f"NVIDIA GPU detected: {', '.join(gpu_names)}", flush=True)
    print(
        f"Installing CUDA PyTorch {TORCH_VERSION} ({CUDA_VARIANT}). "
        "This is a large download and can take several minutes.",
        flush=True,
    )
    try:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-compile",
            "--target",
            str(staging),
            f"torch=={TORCH_VERSION}",
            "--index-url",
            CUDA_INDEX_URL,
        ]
        subprocess.run(command, check=True)
        verification = _validate(staging)

        if target.exists():
            target.replace(backup)
        staging.replace(target)
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)

        manifest = {
            "schema": 1,
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "gpu_names": gpu_names,
            "variant": CUDA_VARIANT,
            "index_url": CUDA_INDEX_URL,
            **verification,
        }
        (root / "cuda-runtime.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2), flush=True)
        print("CUDA PyTorch installation completed successfully.", flush=True)
        return manifest
    except Exception:
        if not target.exists() and backup.exists():
            backup.replace(target)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def check() -> dict:
    target = _runtime_root() / "cuda-site-packages"
    if not target.exists():
        raise RuntimeError("The optional CUDA PyTorch layer is not installed.")
    result = _validate(target)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("install", "check"))
    args = parser.parse_args()
    try:
        if args.action == "install":
            install()
        else:
            check()
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
