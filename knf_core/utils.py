import os
import shutil
import logging
import subprocess
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
        logging.info(f"Auto-detected Multiwfn at {exe_path}")
        logging.info(f"Adding {directory} to temporary PATH.")
        os.environ['PATH'] = directory + os.pathsep + os.environ['PATH']
