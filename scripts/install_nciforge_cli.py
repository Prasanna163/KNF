#!/usr/bin/env python3
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


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

    missing = [name for name in ("xtb", "obabel") if shutil.which(name) is None]
    if missing:
        print(f"WARNING: Missing required tools after setup: {', '.join(missing)}")
        print("Please install them manually and ensure they are available in PATH.")
    else:
        print("External tool check passed: xtb and obabel found.")


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


def _install_gxtb_related(python_exe: str):
    # Best-effort helper packages commonly used around differentiable/graph xTB workflows.
    _run([python_exe, "-m", "pip", "install", "--upgrade", "gxtb", "dxtb"], check=False)


def main():
    print("NCIForge Interactive Installer")
    print("------------------------------")

    scope = _ask_choice("Install scope", ["local", "global"], "local")
    torch_mode = _ask_choice("PyTorch mode", ["cpu", "gpu", "skip"], "cpu")
    setup_external = _ask_yes_no("Set up xtb/obabel and other external tools now?", True)
    setup_gxtb = _ask_yes_no("Also install g-xtb helper Python packages (gxtb/dxtb)?", True)

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

    if setup_gxtb:
        _install_gxtb_related(python_exe)

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
