from __future__ import annotations

import os
import sys

from rich.console import Console

from . import first_run, utils
from .cli import app as cli_app
from .cli import commands as cli_commands
from .cli import interactive as cli_interactive
from .cli.dependency_report import print_missing_tools_warning
from .engine.constants import CLI_NAME, CLI_SUBTITLE, CLI_TITLE, CLI_VERSION, STOP_KEY, VALID_INPUT_EXTS
from .engine.dependencies import (
    _DEPENDENCY_CHECK_CACHE,
    probe_missing_dependencies,
    xtbx_available as _xtbx_available,
)
from .engine.discovery import resolve_results_root
from .engine.gpu import (
    _is_torch_available,
    ensure_cuda_runtime_for_gpu_mode as _ensure_cuda_runtime_for_gpu_mode,
    resolve_cpu_backend_when_torch_missing as _engine_resolve_cpu_backend_when_torch_missing,
)
from .engine.processing import (
    _best_effort_release_memory,
    _build_pipeline,
    _is_oom_error,
    _is_transient_file_error,
    process_file,
    process_file_post_nci,
    process_file_pre_nci,
)


def _supports_unicode_terminal(stream=None) -> bool:
    stream = stream or sys.stdout
    encoding = getattr(stream, "encoding", "") or ""
    if "utf" in encoding.lower():
        return True
    if os.name == "nt" and getattr(stream, "isatty", lambda: False)():
        return True
    if os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM"):
        return True
    return False


def _clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _ensure_utf8_stdout() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _show_startup_splash() -> None:
    print()
    print(f"{CLI_TITLE}")
    print(CLI_SUBTITLE)
    print()


def _live_repaint_supported(console: Console) -> bool:
    """Return True only when Rich Live cursor controls are safe to emit."""
    if os.environ.get("NCIFORGE_FORCE_LIVE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if os.environ.get("NCIFORGE_NO_LIVE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if not bool(getattr(sys.stdout, "isatty", lambda: False)()):
        return False
    if not bool(getattr(console, "is_terminal", False)):
        return False
    if bool(getattr(console, "is_dumb_terminal", False)):
        return False
    if os.name == "nt" and bool(getattr(console, "legacy_windows", False)):
        return False
    return True


def _resolve_cpu_backend_when_torch_missing(args) -> None:
    """Compat shim whose dependency hooks remain patchable through this module."""
    return _engine_resolve_cpu_backend_when_torch_missing(
        args,
        is_torch_available=_is_torch_available,
        ensure_cuda_runtime=_ensure_cuda_runtime_for_gpu_mode,
    )


def check_dependencies(multiwfn_path: str = None, nci_backend: str = "torch", xtb_engine: str = "xtbx"):
    """Compat shim for legacy call sites: probes then prints, discarding the list."""
    missing = probe_missing_dependencies(
        multiwfn_path=multiwfn_path,
        nci_backend=nci_backend,
        xtb_engine=xtb_engine,
    )
    if missing:
        print_missing_tools_warning(missing)


run_single_file = cli_commands.run_single_file
run_batch_directory = cli_commands.run_batch_directory
run_universal_kuid = cli_commands.run_universal_kuid
run_batch_directory_batched = cli_commands.run_batch_directory_batched


def main():
    _ensure_utf8_stdout()
    _clear_terminal()
    _show_startup_splash()
    if len(sys.argv) == 1:
        return cli_interactive.run_interactive()
    return cli_app.main(sys.argv[1:], prog_name=os.path.basename(sys.argv[0]) or CLI_NAME.lower())


if __name__ == "__main__":
    main()
