import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from . import autoconfig, utils


STATE_DIRNAME = ".knf"
STATE_FILENAME = "first_run_state.json"
DEFAULT_RAM_PER_JOB_MB = 50.0


def _state_dir() -> Path:
    return Path.home() / STATE_DIRNAME


def _state_path() -> Path:
    return _state_dir() / STATE_FILENAME


def _load_state() -> Dict:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(payload: Dict) -> None:
    folder = _state_dir()
    folder.mkdir(parents=True, exist_ok=True)
    _state_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _tool_exists(candidates: List[str]) -> bool:
    return any(shutil.which(name) for name in candidates)


def _run_cmd(cmd: List[str]) -> bool:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def _check_tools(
    explicit_multiwfn_path: Optional[str] = None,
    require_multiwfn: bool = True,
) -> Dict[str, bool]:
    utils.ensure_multiwfn_in_path(explicit_path=explicit_multiwfn_path)
    status = {
        "xtb": _tool_exists(["xtb"]),
        "obabel": _tool_exists(["obabel"]),
    }
    if require_multiwfn:
        status["multiwfn"] = _tool_exists(["Multiwfn", "Multiwfn.exe"])
    return status


def _try_install_with_conda(status: Dict[str, bool]) -> None:
    if shutil.which("conda") is None:
        return

    packages = []
    if not status.get("xtb"):
        packages.append("xtb")
    if not status.get("obabel"):
        packages.append("openbabel")

    if not packages:
        return

    _run_cmd(["conda", "install", "-y", "-c", "conda-forge"] + packages)


def _try_install_openbabel_with_winget(status: Dict[str, bool]) -> None:
    if status.get("obabel"):
        return
    if os.name != "nt":
        return
    if shutil.which("winget") is None:
        return
    _run_cmd(
        [
            "winget",
            "install",
            "--id",
            "OpenBabel.OpenBabel",
            "-e",
            "--accept-package-agreements",
            "--accept-source-agreements",
        ]
    )


def _print_missing_help(status: Dict[str, bool]) -> None:
    missing = [name for name, ok in status.items() if not ok]
    if not missing:
        return

    print("")
    print("KNF-Core first-run setup: unresolved external dependencies")
    print("---------------------------------------------------------")
    for name in missing:
        if name == "xtb":
            print("Missing: xtb")
            print("Install (conda): conda install -c conda-forge xtb")
        elif name == "obabel":
            print("Missing: obabel (Open Babel)")
            print("Install (conda): conda install -c conda-forge openbabel")
            if os.name == "nt":
                print("Install (winget): winget install --id OpenBabel.OpenBabel -e")
        elif name == "multiwfn":
            print("Missing: Multiwfn")
            print("Install manually, then provide path with:")
            print("  knf <input> --multiwfn-path \"<path-to-Multiwfn.exe-or-folder>\"")
            print("Or set env var:")
            print("  KNF_MULTIWFN_PATH=<path-to-Multiwfn.exe-or-folder>")
    print("")


def _run_one_time_autoconfig(force: bool = False) -> autoconfig.MultiConfig:
    cache_root = str(_state_dir())
    n_jobs = max(2, (os.cpu_count() or 2) * 2)
    cfg = autoconfig.resolve_multi_config(
        n_jobs=n_jobs,
        ram_per_job_mb=DEFAULT_RAM_PER_JOB_MB,
        project_root=cache_root,
        force_refresh=force,
    )
    autoconfig.apply_env_inplace(cfg)
    return cfg


def _print_autoconfig_hint(cfg: autoconfig.MultiConfig) -> None:
    print("")
    print("KNF-Core multiprocessing recommendation")
    print("--------------------------------------")
    print(f"Suggested workers: {cfg.workers}")
    print(f"OMP threads/job:   {cfg.omp_num_threads}")
    print(f"RAM estimate/job:  {cfg.ram_per_job_mb:.0f} MB")
    print("Suggested CLI:     --processing multi")
    print("")


def ensure_first_run_setup(
    force: bool = False,
    multiwfn_path: Optional[str] = None,
    require_multiwfn: bool = True,
) -> bool:
    resolved_multiwfn = None
    if multiwfn_path:
        resolved_multiwfn = utils.register_multiwfn_path(multiwfn_path, persist=True)
        if not resolved_multiwfn:
            print(f"WARNING: Invalid --multiwfn-path: {multiwfn_path}")

    state = _load_state()
    if state.get("completed") and not force:
        return True

    print("")
    print("Running KNF-Core first-time setup...")

    status = _check_tools(
        explicit_multiwfn_path=resolved_multiwfn,
        require_multiwfn=require_multiwfn,
    )
    if not all(status.values()):
        _try_install_with_conda(status)
        status = _check_tools(
            explicit_multiwfn_path=resolved_multiwfn,
            require_multiwfn=require_multiwfn,
        )

    if not all(status.values()):
        _try_install_openbabel_with_winget(status)
        status = _check_tools(
            explicit_multiwfn_path=resolved_multiwfn,
            require_multiwfn=require_multiwfn,
        )

    cfg = _run_one_time_autoconfig(force=force)
    _print_autoconfig_hint(cfg)

    if not all(status.values()):
        _print_missing_help(status)
        return False

    _save_state(
        {
            "completed": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "dependency_status": status,
            "multi_config": {
                "workers": cfg.workers,
                "omp_num_threads": cfg.omp_num_threads,
                "source": cfg.source,
                "ram_per_job_mb": cfg.ram_per_job_mb,
            },
        }
    )

    print("KNF-Core first-time setup complete.")
    print("")
    return True
