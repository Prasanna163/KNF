import numpy as np
import logging
import os


def parse_grid_file(filepath: str):
    """
    Parses Multiwfn grid data file.
    Expected format: X Y Z sign(lambda2)rho RDG
    """
    data = []
    with open(filepath, 'r') as f:
        # Skip header if any? Multiwfn exported text files usually have a header or just data.
        # We'll assume just data or comments starting with #
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            try:
                parts = line.split()
                if len(parts) >= 5:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    sl2rho = float(parts[3])
                    rdg = float(parts[4])
                    data.append([x, y, z, sl2rho, rdg])
            except ValueError:
                continue
    return np.array(data)

def _axis_step(axis: np.ndarray) -> float:
    if axis.size < 2:
        return 1.0
    return float(abs(axis[1] - axis[0]))


def compute_delta_v(data: np.ndarray) -> float:
    """Estimates volume element deltaV from text-grid rows."""
    if len(data) < 2:
        return 1.0

    xs = np.unique(data[:, 0])
    ys = np.unique(data[:, 1])
    zs = np.unique(data[:, 2])
    return _axis_step(xs) * _axis_step(ys) * _axis_step(zs)


def _load_grid_payload(grid_path: str) -> tuple[np.ndarray, float]:
    ext = os.path.splitext(grid_path)[1].lower()

    if ext == ".npz":
        with np.load(grid_path) as payload:
            required = {"x", "y", "z", "sign_lambda2_rho", "rdg"}
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(f"Incomplete NCI NPZ payload, missing keys: {sorted(missing)}")
            x = np.asarray(payload["x"], dtype=np.float64)
            y = np.asarray(payload["y"], dtype=np.float64)
            z = np.asarray(payload["z"], dtype=np.float64)
            sl2rho = np.asarray(payload["sign_lambda2_rho"], dtype=np.float64).reshape(-1)
            delta_v = _axis_step(x) * _axis_step(y) * _axis_step(z)
        return sl2rho, float(delta_v)

    data = parse_grid_file(grid_path)
    if len(data) == 0:
        return np.array([], dtype=np.float64), 1.0
    return np.asarray(data[:, 3], dtype=np.float64), float(compute_delta_v(data))

def compute_snci(grid_path: str) -> float:
    """
    Computes SNCI from grid file.
    SNCI = sum( -sign(lambda2)*rho * deltaV ) for lambda2 < 0
    sign(lambda2)*rho is the 4th column.
    If lambda2 < 0, then sign(lambda2)*rho < 0.
    So we filter for points where column 4 < 0.
    """
    if not os.path.exists(grid_path):
        logging.warning(f"Grid file not found: {grid_path}")
        return 0.0

    sl2rho, delta_v = _load_grid_payload(grid_path)
    if sl2rho.size == 0:
        return 0.0

    attractive = sl2rho[sl2rho < 0.0]
    if attractive.size == 0:
        return 0.0

    snci = np.sum(-attractive * delta_v)
    return float(snci)

def compute_nci_statistics(grid_path: str) -> dict:
    """Computes f6-f9 statistics for attractive points."""
    stats = {
        'f6': 0, 'f7': 0.0, 'f8': 0.0, 'f9': 0.0
    }
    
    if not os.path.exists(grid_path):
        return stats

    sl2rho, _ = _load_grid_payload(grid_path)
    if sl2rho.size == 0:
        return stats

    attractive = sl2rho[sl2rho < 0.0]
    if attractive.size == 0:
        return stats

    stats['f6'] = int(attractive.size)
    stats['f7'] = float(np.mean(attractive))
    stats['f8'] = float(np.std(attractive))
    from scipy.stats import skew
    stats['f9'] = float(skew(attractive))

    return stats
