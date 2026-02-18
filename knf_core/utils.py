import os
import shutil
import logging
import subprocess
import unicodedata
from pathlib import Path
from typing import Optional

TOOL_CONFIG_DIR = ".knf"
TOOL_CONFIG_FILE = "tool_paths.json"
_MOJIBAKE_PROBE_CHARS = {
    "\u00c2",  # Â
    "\u00c3",  # Ã
    "\u00ce",  # Î
    "\u00cf",  # Ï
    "\u00d0",  # Ð
    "\u00d1",  # Ñ
    "\u00e2",  # â
}


def setup_logging(debug: bool = False):
    """Configures logging to console."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def detect_file_type(filepath: str) -> str:
    """Detects file type based on extension."""
    ext = Path(filepath).suffix.lower()
    if ext == '.mol':
        return 'mol'
    elif ext == '.xyz':
        return 'xyz'
    elif ext == '.molden':
        return 'molden'
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def ensure_directory(path: str):
    """Creates directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_copy(src: str, dst: str):
    """Copies a file from src to dst."""
    shutil.copy2(src, dst)


def run_subprocess(cmd: list, cwd: str = None, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Runs a subprocess command."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            check=True,
            errors='replace'
        )
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {' '.join(cmd)}")
        if capture_output:
            logging.error(f"STDOUT: {e.stdout}")
            logging.error(f"STDERR: {e.stderr}")
        raise e


def _tool_config_path() -> Path:
    return Path.home() / TOOL_CONFIG_DIR / TOOL_CONFIG_FILE


def _load_tool_config() -> dict:
    path = _tool_config_path()
    if not path.exists():
        return {}
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_tool_config(payload: dict) -> None:
    path = _tool_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_multiwfn_executable(candidate_path: str) -> Optional[str]:
    """Resolves a candidate executable or directory into a concrete Multiwfn executable path."""
    if not candidate_path:
        return None

    candidate = Path(candidate_path).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()

    if candidate.is_file():
        if candidate.name.lower() in {"multiwfn", "multiwfn.exe"}:
            return str(candidate)
        return None

    if candidate.is_dir():
        for name in ("Multiwfn.exe", "Multiwfn"):
            exe = candidate / name
            if exe.exists() and exe.is_file():
                return str(exe)
    return None


def get_registered_multiwfn_path() -> Optional[str]:
    config = _load_tool_config()
    path = config.get("multiwfn_path")
    return resolve_multiwfn_executable(path) if path else None


def register_multiwfn_path(candidate_path: str, persist: bool = True) -> Optional[str]:
    """Registers a user-provided Multiwfn path and injects it into PATH for current process."""
    exe = resolve_multiwfn_executable(candidate_path)
    if not exe:
        return None

    exe_dir = str(Path(exe).parent)
    if exe_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = exe_dir + os.pathsep + os.environ.get("PATH", "")

    if persist:
        config = _load_tool_config()
        config["multiwfn_path"] = exe
        _save_tool_config(config)
    return exe


def _common_multiwfn_candidates() -> list[str]:
    candidates = []
    env_path = os.environ.get("KNF_MULTIWFN_PATH")
    if env_path:
        candidates.append(env_path)

    registered = get_registered_multiwfn_path()
    if registered:
        candidates.append(registered)

    candidates.extend(
        [
            r"E:\Prasanna\Multiwfn (cosmo)\Multiwfn_3.8_dev_bin_Win64\Multiwfn.exe",
            str(Path.home() / "Multiwfn.exe"),
            str(Path.home() / "Downloads" / "Multiwfn.exe"),
            str(Path.home() / "Desktop" / "Multiwfn.exe"),
        ]
    )
    return candidates


def _scan_for_multiwfn_roots() -> Optional[str]:
    """Lightweight scan for Multiwfn executable in likely parent directories."""
    roots = [Path.home(), Path.cwd()]
    drive = os.environ.get("HOMEDRIVE")
    if drive:
        roots.append(Path(drive + "\\"))

    visited = set()
    for root in roots:
        try:
            root_resolved = root.resolve()
        except Exception:
            continue
        if root_resolved in visited or not root_resolved.exists():
            continue
        visited.add(root_resolved)

        queue = [root_resolved]
        while queue:
            current = queue.pop(0)
            try:
                if current.name and "multiwfn" in current.name.lower():
                    resolved = resolve_multiwfn_executable(str(current))
                    if resolved:
                        return resolved

                depth = len(current.relative_to(root_resolved).parts)
                if depth >= 3:
                    continue
                for child in current.iterdir():
                    if child.is_dir():
                        queue.append(child)
                    elif child.is_file() and child.name.lower() in {"multiwfn.exe", "multiwfn"}:
                        return str(child.resolve())
            except Exception:
                continue
    return None


def find_multiwfn(explicit_path: Optional[str] = None) -> Optional[str]:
    """Searches for Multiwfn executable. Returns absolute path if found, else None."""
    if explicit_path:
        resolved = resolve_multiwfn_executable(explicit_path)
        if resolved:
            return resolved

    for candidate in _common_multiwfn_candidates():
        resolved = resolve_multiwfn_executable(candidate)
        if resolved:
            return resolved

    return _scan_for_multiwfn_roots()


def ensure_multiwfn_in_path(explicit_path: Optional[str] = None):
    """Attempts to find Multiwfn and add it to PATH."""
    if shutil.which('Multiwfn') or shutil.which('Multiwfn.exe'):
        return

    exe_path = find_multiwfn(explicit_path=explicit_path)
    if exe_path:
        register_multiwfn_path(exe_path, persist=False)
        logging.debug(f"Auto-detected Multiwfn at {exe_path}")


def _repair_mojibake(text: str) -> str:
    """Attempts to repair common UTF-8->Latin-1/CP1252 mojibake artifacts."""
    if not text:
        return text
    if text.isascii():
        return text
    if not any(ch in text for ch in _MOJIBAKE_PROBE_CHARS):
        return text
    current = text
    for _ in range(2):
        changed = False
        for enc in ("latin-1", "cp1252"):
            try:
                fixed = current.encode(enc, errors="strict").decode("utf-8", errors="strict")
            except Exception:
                continue
            if fixed and fixed != current:
                current = fixed
                changed = True
                break
        if not changed:
            break
    return current


def normalize_name_for_matching(name: str) -> str:
    """Normalizes names for robust matching across Unicode/encoding artifacts."""
    if name is None:
        return ""
    value = _repair_mojibake(name)
    value = unicodedata.normalize("NFKC", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) not in {"Cc", "Cf", "Cs"})
    return value.strip()


def normalized_extension(name: str) -> str:
    """Returns a normalized lowercase extension from a possibly artifacted filename."""
    normalized = normalize_name_for_matching(name)
    ext = os.path.splitext(normalized)[1].lower()
    if ext and all(ch.isascii() and (ch.isalnum() or ch == ".") for ch in ext):
        return ext

    repaired = _repair_mojibake(name)
    repaired_ext = os.path.splitext(repaired)[1].lower()
    if repaired_ext and all(ch.isascii() and (ch.isalnum() or ch == ".") for ch in repaired_ext):
        return repaired_ext

    ascii_tail = "".join(ch for ch in ext if ch.isascii() and (ch.isalnum() or ch == "."))
    if ascii_tail and ascii_tail.startswith(".") and len(ascii_tail) > 1:
        return ascii_tail

    return ext


def resolve_artifacted_path(path: str) -> str:
    """
    Resolves a possibly mojibake/corrupted user path to an existing sibling entry.
    Returns the original absolute path when no match is found.
    """
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        return abs_path

    parent = os.path.dirname(abs_path)
    target_name = os.path.basename(abs_path)
    if not parent or not os.path.isdir(parent):
        return abs_path

    target_keys = {
        normalize_name_for_matching(target_name).casefold(),
        unicodedata.normalize("NFKC", target_name).strip().casefold(),
    }
    target_keys.discard("")
    if not target_keys:
        return abs_path

    for candidate in os.listdir(parent):
        candidate_keys = {
            normalize_name_for_matching(candidate).casefold(),
            unicodedata.normalize("NFKC", candidate).strip().casefold(),
        }
        if target_keys & candidate_keys:
            return os.path.join(parent, candidate)

    return abs_path
