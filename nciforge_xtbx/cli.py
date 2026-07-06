from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


class XtbxUnavailable(RuntimeError):
    """Raised when the bundled xtbx launcher cannot find its runtime."""


CUDA_RUNTIME_DLLS = (
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cusolver64_11.dll",
)


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def config_path() -> Path:
    override = os.environ.get("NCIFORGE_XTBX_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".knf" / "xtbx_runtime.json"


def subprocess_invocation() -> list[str]:
    """Return a stable argv prefix for NCIForge subprocess calls."""
    return [sys.executable, str(package_dir() / "_runner.py")]


def _is_runtime(root: Path) -> bool:
    return (
        (root / "bin" / "xtb-cpu.exe").exists()
        and (root / "bin" / "xtb.exe").exists()
        and (root / "params" / "param_gfn2-xtb.txt").exists()
    )


def _has_cuda_redistributables(root: Path) -> bool:
    lib = root / "lib"
    return all((lib / name).exists() for name in CUDA_RUNTIME_DLLS)


def _is_full_gpu_runtime(root: Path) -> bool:
    return _is_runtime(root) and _has_cuda_redistributables(root)


def _wants_gpu(args: Iterable[str]) -> bool:
    return any(arg in {"--gpu", "--gpu-batch"} for arg in args)


def _env_runtime() -> Path | None:
    for name in ("NCIFORGE_XTBX_RUNTIME", "XTB_GPU_PKG"):
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return None


def _load_config() -> dict:
    path = config_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_config(payload: dict) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _configured_runtime() -> Path | None:
    payload = _load_config()
    value = payload.get("gpu_runtime")
    if isinstance(value, str) and value.strip():
        return Path(value).expanduser()
    return None


def _runtime_candidates(args: Iterable[str]) -> list[Path]:
    bundled = package_dir() / "runtime" / "xtb-win-release"
    dev_full = Path(r"E:\Prasanna\xTB\xtb\xtb-win-release")
    candidates: list[Path] = []

    env_runtime = _env_runtime()
    if env_runtime is not None:
        candidates.append(env_runtime)

    configured = _configured_runtime()
    if _wants_gpu(args):
        if configured is not None:
            candidates.append(configured)
        candidates.extend([dev_full, bundled])
    else:
        candidates.extend([bundled])
        if configured is not None:
            candidates.append(configured)
        candidates.append(dev_full)

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def resolve_runtime(args: Iterable[str] = ()) -> Path:
    wants_gpu = _wants_gpu(args)
    for candidate in _runtime_candidates(args):
        if _is_runtime(candidate):
            if wants_gpu and not _has_cuda_redistributables(candidate):
                continue
            return candidate

    if wants_gpu:
        raise XtbxUnavailable(
            "explicit GPU execution needs a full xtb-win-release runtime with "
            "the NVIDIA CUDA redistributable DLLs. Set NCIFORGE_XTBX_RUNTIME to "
            "that folder. Checked: "
            + ", ".join(str(p) for p in _runtime_candidates(args))
        )

    for candidate in _runtime_candidates(args):
        if _is_runtime(candidate):
            return candidate

    raise XtbxUnavailable(
        "xtbx runtime not found. Expected bin/xtb-cpu.exe, bin/xtb.exe, and "
        "params/ under one of: "
        + ", ".join(str(p) for p in _runtime_candidates(args))
        + ". Set NCIFORGE_XTBX_RUNTIME to a full xtb-win-release folder."
    )


def _full_runtime_problem(path: Path) -> str:
    if not _is_runtime(path):
        return (
            "missing bin/xtb-cpu.exe, bin/xtb.exe, or "
            "params/param_gfn2-xtb.txt"
        )
    missing = [name for name in CUDA_RUNTIME_DLLS if not (path / "lib" / name).exists()]
    if missing:
        return "missing CUDA DLLs: " + ", ".join(missing)
    return ""


def _prompt_yes_no(question: str, default: str = "n") -> bool | None:
    default = (default or "n").strip().lower()
    if default not in {"y", "n"}:
        default = "n"
    if not sys.stdin or not sys.stdin.isatty():
        return None

    suffix = "[Y/n]" if default == "y" else "[y/N]"
    while True:
        try:
            answer = input(f"{question} {suffix}: ").strip().lower()
        except EOFError:
            return None
        if not answer:
            return default == "y"
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter 'y' or 'n'.")


def _save_gpu_runtime(path: Path) -> None:
    payload = _load_config()
    payload["gpu_runtime"] = str(path.resolve())
    payload["configured_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["note"] = (
        "Full xtbx GPU runtime configured by NCIForge. This folder must contain "
        "the large NVIDIA CUDA redistributable DLLs."
    )
    _save_config(payload)


def _detected_full_runtime() -> Path | None:
    candidates = []
    env_runtime = _env_runtime()
    configured = _configured_runtime()
    if env_runtime is not None:
        candidates.append(env_runtime)
    if configured is not None:
        candidates.append(configured)
    candidates.append(Path(r"E:\Prasanna\xTB\xtb\xtb-win-release"))

    for candidate in candidates:
        if _is_full_gpu_runtime(candidate):
            return candidate
    return None


def setup_gpu_runtime(runtime_path: str | None = None, *, interactive: bool = True) -> int:
    """Configure the full xtbx GPU runtime used by explicit ``--gpu`` runs."""
    print("NCIForge xtbx GPU runtime setup")
    print("-------------------------------")
    print(
        "The NCIForge package ships a compact xtbx runtime for normal CPU use. "
        "Explicit GPU execution also needs the large NVIDIA CUDA redistributable "
        "DLLs from a full xtb-win-release folder."
    )

    path: Path | None = Path(runtime_path).expanduser() if runtime_path else None
    if path is None:
        detected = _detected_full_runtime()
        if detected is not None:
            if interactive:
                use_detected = _prompt_yes_no(
                    f"Use detected full runtime at {detected}?", default="y"
                )
                if use_detected:
                    path = detected
            else:
                path = detected

    while path is None and interactive and sys.stdin and sys.stdin.isatty():
        try:
            raw = input(
                "Path to full xtb-win-release folder "
                "(blank to cancel): "
            ).strip().strip('"')
        except EOFError:
            raw = ""
        if not raw:
            print("xtbx GPU setup cancelled.")
            return 1
        path = Path(raw).expanduser()

    if path is None:
        print(
            "No full runtime path was provided. Run again with:\n"
            "  xtbx --setup-gpu C:\\path\\to\\xtb-win-release"
        )
        return 1

    problem = _full_runtime_problem(path)
    if problem:
        print(f"Cannot use {path}: {problem}", file=sys.stderr)
        return 1

    _save_gpu_runtime(path)
    print(f"Saved xtbx GPU runtime: {path.resolve()}")
    print(f"Config file: {config_path()}")
    return 0


def resolve_bash() -> str:
    env_bash = os.environ.get("XTBX_BASH")
    candidates = [
        Path(env_bash) if env_bash else None,
        Path(r"C:\msys64\usr\bin\bash.exe"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)

    found = shutil.which("bash")
    if found:
        return found

    raise XtbxUnavailable(
        "MSYS2 bash was not found. Install MSYS2 or set XTBX_BASH to bash.exe."
    )


def windows_to_msys_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name != "nt":
        return resolved
    drive = resolved[0].lower()
    rest = resolved[2:].replace("\\", "/")
    return f"/{drive}{rest}"


def _normalize_arg(arg: str) -> str:
    if os.name == "nt":
        return arg.replace("\\", "/")
    return arg


def is_available() -> bool:
    try:
        resolve_bash()
        resolve_runtime(())
        return True
    except Exception:
        return False


def gpu_runtime_available() -> bool:
    try:
        resolve_runtime(["--gpu"])
        return True
    except Exception:
        return False


def _consume_gpu_runtime_arg(args: list[str]) -> list[str]:
    if "--gpu-runtime" not in args:
        return args
    idx = args.index("--gpu-runtime")
    if idx == len(args) - 1:
        raise XtbxUnavailable("--gpu-runtime requires a path argument.")
    os.environ["NCIFORGE_XTBX_RUNTIME"] = args[idx + 1]
    return args[:idx] + args[idx + 2:]


def _setup_path_from_args(args: list[str]) -> tuple[bool, str | None]:
    for flag in ("--setup-gpu", "setup-gpu"):
        if flag in args:
            idx = args.index(flag)
            setup_path = None
            if idx + 1 < len(args) and not args[idx + 1].startswith("-"):
                setup_path = args[idx + 1]
            return True, setup_path
    return False, None


def _resolve_runtime_or_prompt(args: list[str]) -> Path:
    try:
        return resolve_runtime(args)
    except XtbxUnavailable as exc:
        if not _wants_gpu(args):
            raise
        if sys.stdin and sys.stdin.isatty():
            print(f"xtbx: {exc}", file=sys.stderr)
            yn = _prompt_yes_no("Configure the full xtbx GPU runtime now?", default="n")
            if yn:
                if setup_gpu_runtime(interactive=True) == 0:
                    return resolve_runtime(args)
        raise XtbxUnavailable(
            "full xtbx GPU runtime is not configured. Run "
            "`xtbx --setup-gpu <path-to-full-xtb-win-release>` or pass "
            "`--gpu-runtime <path>` for one command."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    setup_requested, setup_path = _setup_path_from_args(args)
    if setup_requested:
        return setup_gpu_runtime(setup_path, interactive=True)

    try:
        args = _consume_gpu_runtime_arg(args)
        bash = resolve_bash()
        runtime = _resolve_runtime_or_prompt(args)
    except XtbxUnavailable as exc:
        print(f"xtbx: {exc}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["XTB_GPU_PKG"] = windows_to_msys_path(runtime)

    script = package_dir() / "xtbx_run.sh"
    cmd = [bash, str(script), *(_normalize_arg(arg) for arg in args)]
    return subprocess.run(cmd, env=env).returncode
