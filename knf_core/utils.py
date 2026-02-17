import os
import shutil
import logging
import subprocess
import unicodedata
from pathlib import Path

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

def find_multiwfn() -> str:
    """
    Searches for Multiwfn executable.
    Returns absolute path if found, or None.
    """
    candidates = [
        r'E:\Prasanna\Multiwfn (cosmo)\Multiwfn_3.8_dev_bin_Win64\Multiwfn.exe',
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
            
    return None

def ensure_multiwfn_in_path():
    """Attempts to find Multiwfn and add it to PATH."""
    if shutil.which('Multiwfn') or shutil.which('Multiwfn.exe'):
        return

    exe_path = find_multiwfn()
    if exe_path:
        directory = os.path.dirname(exe_path)
        logging.debug(f"Auto-detected Multiwfn at {exe_path}")
        logging.debug(f"Adding {directory} to temporary PATH.")
        os.environ['PATH'] = directory + os.pathsep + os.environ['PATH']


def _repair_mojibake(text: str) -> str:
    """Attempts to repair common UTF-8->Latin-1/CP1252 mojibake artifacts."""
    if not text:
        return text
    probe = ("\u00c3" in text) or ("\u00c2" in text) or ("\u00e2" in text)
    if not probe:
        return text
    for enc in ("latin-1", "cp1252"):
        try:
            fixed = text.encode(enc, errors="strict").decode("utf-8", errors="strict")
            if fixed and fixed != text:
                return fixed
        except Exception:
            continue
    return text


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

    target_key = normalize_name_for_matching(target_name).casefold()
    if not target_key:
        return abs_path

    for candidate in os.listdir(parent):
        if normalize_name_for_matching(candidate).casefold() == target_key:
            return os.path.join(parent, candidate)

    return abs_path

