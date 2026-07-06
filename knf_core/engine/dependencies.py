import shutil
import threading

from .. import utils

_DEPENDENCY_CHECK_CACHE: dict = {}
_DEPENDENCY_CHECK_LOCK = threading.Lock()


def xtbx_available() -> bool:
    try:
        from nciforge_xtbx.cli import is_available

        return bool(is_available())
    except Exception:
        return shutil.which('xtbx') is not None


def probe_missing_dependencies(
    multiwfn_path: str = None, nci_backend: str = "torch", xtb_engine: str = "xtbx"
) -> list:
    """Returns the list of required external tools not found in PATH or
    registered fallback locations. Pure/print-free: callers decide how (or
    whether) to surface the result."""
    engine = (xtb_engine or "xtbx").strip().lower()
    cache_key = (
        (multiwfn_path or "").strip(),
        (nci_backend or "torch").strip().lower(),
        engine,
    )
    with _DEPENDENCY_CHECK_LOCK:
        cached = _DEPENDENCY_CHECK_CACHE.get(cache_key)
        if cached is not None:
            return list(cached)

    missing = []

    utils.ensure_external_tools_in_path(persist=False)
    if not utils.resolve_external_tool_command('obabel'):
        missing.append('obabel (Open Babel)')

    if engine in ("xtb", "auto") and not utils.resolve_external_tool_command('xtb'):
        missing.append('xtb (Extended Tight Binding)')

    # The unified xtbx front-end is required when selected or when auto may invoke it.
    if engine in ("xtbx", "auto") and not xtbx_available():
        missing.append(
            'xtbx (native xTB front-end; required for --xtb-engine xtbx/auto)'
        )

    backend = (nci_backend or "torch").strip().lower()
    if backend == "multiwfn":
        # Avoid expensive Multiwfn auto-discovery for torch/gpu runs.
        utils.ensure_multiwfn_in_path(explicit_path=multiwfn_path)
        if not shutil.which('Multiwfn') and not shutil.which('Multiwfn.exe'):
            missing.append('Multiwfn')

    with _DEPENDENCY_CHECK_LOCK:
        _DEPENDENCY_CHECK_CACHE[cache_key] = list(missing)

    return missing
