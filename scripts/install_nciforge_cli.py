#!/usr/bin/env python3
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _run(cmd, check=True):
    print(f"[run] {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def _ask_choice(prompt, options, default):
    option_set = {o.lower() for o in options}
    default = default.lower()
    while True:
        raw = input(f"{prompt} ({'/'.join(options)}) [{default}]: ").strip().lower()
        val = raw or default
        if val in option_set:
            return val
        print(f"Please choose one of: {', '.join(options)}")


def _ask_yes_no(prompt, default=True):
    default_s = "y" if default else "n"
    while True:
        raw = input(f"{prompt} [y/n] [{default_s}]: ").strip().lower()
        val = raw or default_s
        if val in {"y", "yes"}:
            return True
        if val in {"n", "no"}:
            return False
        print("Please answer y or n.")


def _venv_python(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _choose_tool_installer():
    if shutil.which("conda"):
        return "conda"
    if shutil.which("mamba"):
        return "mamba"
    if os.name == "nt" and shutil.which("winget"):
        return "winget"
    if platform.system().lower() == "darwin" and shutil.which("brew"):
        return "brew"
    if platform.system().lower() == "linux":
        if shutil.which("apt-get"):
            return "apt"
        if shutil.which("dnf"):
            return "dnf"
    return None


def _install_external_tools(installer):
    print(f"\nSetting up external tools via: {installer}")
    try:
        if installer in {"conda", "mamba"}:
            _run([installer, "install", "-y", "-c", "conda-forge", "xtb", "openbabel"], check=False)
        elif installer == "winget":
            _run(["winget", "install", "--id", "OpenBabel.OpenBabel", "-e"], check=False)
            # xTB winget package name may differ by region; this is best-effort.
            _run(["winget", "install", "--id", "GrimmeLab.xTB", "-e"], check=False)
        elif installer == "brew":
            _run(["brew", "install", "xtb", "open-babel"], check=False)
        elif installer == "apt":
            _run(["sudo", "apt-get", "update"], check=False)
            _run(["sudo", "apt-get", "install", "-y", "xtb", "openbabel"], check=False)
        elif installer == "dnf":
            _run(["sudo", "dnf", "install", "-y", "xtb", "openbabel"], check=False)
    except Exception as exc:
        print(f"External dependency setup encountered an issue: {exc}")

    _embed_external_tool_paths()
    missing = [name for name in ("xtb", "obabel") if shutil.which(name) is None]
    if missing:
        print(f"WARNING: Missing required tools after setup: {', '.join(missing)}")
        print("Please install them manually and ensure they are available in PATH.")
    else:
        print("External tool check passed: xtb and obabel found.")


def _windows_persist_user_path(path_dir: str) -> bool:
    if os.name != "nt":
        return False
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Environment",
            0,
            winreg.KEY_READ | winreg.KEY_WRITE,
        ) as key:
            try:
                raw, _ = winreg.QueryValueEx(key, "Path")
                current = raw if isinstance(raw, str) else ""
            except FileNotFoundError:
                current = ""
            parts = [p for p in current.split(";") if p]
            norm = lambda p: os.path.normcase(os.path.normpath(p))
            merged = []
            seen = set()
            for item in [path_dir] + parts:
                n = norm(item)
                if n in seen:
                    continue
                seen.add(n)
                merged.append(item)
            winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, ";".join(merged))
        return True
    except Exception:
        return False


def _find_tool_executable(tool: str) -> Optional[str]:
    current = shutil.which(tool)
    if current:
        return current
    names = [tool]
    if os.name == "nt":
        names.insert(0, f"{tool}.exe")

    candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.extend(
            [
                Path(conda_prefix) / "Library" / "bin",
                Path(conda_prefix) / "Scripts",
                Path(conda_prefix) / "bin",
            ]
        )
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_root = Path(conda_exe).resolve().parent.parent
        candidates.extend(
            [
                conda_root / "Library" / "bin",
                conda_root / "Scripts",
                conda_root / "bin",
            ]
        )
    if os.name == "nt" and tool == "obabel":
        pf = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        candidates.extend([pf / "OpenBabel-3.1.1", pf / "Open Babel 3.1.1", pf / "OpenBabel"])

    seen = set()
    for base in candidates:
        key = os.path.normcase(os.path.normpath(str(base)))
        if key in seen:
            continue
        seen.add(key)
        for name in names:
            exe = base / name
            if exe.exists() and exe.is_file():
                return str(exe)
    return None


def _embed_external_tool_paths() -> None:
    for tool in ("xtb", "obabel"):
        exe = _find_tool_executable(tool)
        if not exe:
            continue
        tool_dir = str(Path(exe).parent)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if not any(os.path.normcase(os.path.normpath(p or "")) == os.path.normcase(os.path.normpath(tool_dir)) for p in path_parts):
            os.environ["PATH"] = tool_dir + os.pathsep + os.environ.get("PATH", "")
        _windows_persist_user_path(tool_dir)


def _install_pytorch(python_exe: str, mode: str):
    if mode == "skip":
        return
    if mode == "cpu":
        _run(
            [
                python_exe,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
            check=False,
        )
        return
    if mode == "gpu":
        _run(
            [
                python_exe,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ],
            check=False,
        )


def main():
    print("NCIForge Interactive Installer")
    print("------------------------------")

    scope = _ask_choice("Install scope", ["local", "global"], "local")
    torch_mode = _ask_choice("PyTorch mode", ["cpu", "gpu", "skip"], "skip")
    setup_external = _ask_yes_no("Set up xtb/obabel and other external tools now?", True)

    repo_root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable

    if scope == "local":
        default_venv = repo_root / ".venv-nciforge"
        raw = input(f"Virtual environment path [{default_venv}]: ").strip()
        venv_path = Path(raw) if raw else default_venv
        _run([python_exe, "-m", "venv", str(venv_path)])
        python_exe = str(_venv_python(venv_path))
        _run([python_exe, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        _run([python_exe, "-m", "pip", "install", "-e", str(repo_root)])
    else:
        _run([python_exe, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        _run([python_exe, "-m", "pip", "install", "--user", "-e", str(repo_root)])

    _install_pytorch(python_exe, torch_mode)

    if setup_external:
        installer = _choose_tool_installer()
        if installer is None:
            print("No supported external package manager found for auto setup.")
            print("Please install xtb and obabel manually.")
        else:
            _install_external_tools(installer)

    print("\nVerifying CLI...")
    _run([python_exe, "-m", "knf_core.main", "--help"], check=False)
    print("\nSetup complete. You can now run:")
    if scope == "local":
        print(f"  {python_exe} -m knf_core.main <file-or-folder> [flags]")
    else:
        print("  nciforge <file-or-folder> [flags]")


if __name__ == "__main__":
    main()
