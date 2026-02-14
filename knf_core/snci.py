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

def compute_delta_v(data: np.ndarray) -> float:
    """Estimates volume element deltaV from grid data."""
    if len(data) < 2:
        return 1.0 # Fallback
        
    # Assuming regular grid, find unique sorted coordinates
    xs = sorted(list(set(data[:, 0])))
    ys = sorted(list(set(data[:, 1])))
    zs = sorted(list(set(data[:, 2])))
    
    dx = xs[1] - xs[0] if len(xs) > 1 else 1.0
    dy = ys[1] - ys[0] if len(ys) > 1 else 1.0
    dz = zs[1] - zs[0] if len(zs) > 1 else 1.0
    
    return dx * dy * dz

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
        
    data = parse_grid_file(grid_path)
    if len(data) == 0:
        return 0.0
        
    # Filter: sign(lambda2)rho < 0
    # Column 3 (0-indexed) is sign(lambda2)rho
    # Wait, col 3 is index 3.
    
    attractive_points = data[data[:, 3] < 0]
    
    if len(attractive_points) == 0:
        return 0.0
        
    delta_v = compute_delta_v(data)
    
    # SNCI = sum( -1 * (sign(lambda2)rho) * deltaV )
    # Since term is negative, -term is positive.
    # It sums the magnitude of attractive density.
    
    sl2rho = attractive_points[:, 3]
    snci = np.sum( -sl2rho * delta_v )
    
    return float(snci)

def compute_nci_statistics(grid_path: str) -> dict:
    """Computes f6-f9 statistics for attractive points."""
    stats = {
        'f6': 0, 'f7': 0.0, 'f8': 0.0, 'f9': 0.0
    }
    
    if not os.path.exists(grid_path):
        return stats
        
    data = parse_grid_file(grid_path)
    if len(data) == 0:
        return stats
        
    # Attractive points only
    attractive = data[data[:, 3] < 0]
    if len(attractive) == 0:
        return stats
        
    xi = attractive[:, 3] # sign(lambda2)rho values
    
    # f6 = count
    stats['f6'] = len(xi)
    
    # f7 = mean(xi)
    stats['f7'] = float(np.mean(xi))
    
    # f8 = std(xi)
    stats['f8'] = float(np.std(xi))
    
    # f9 = skewness(xi)
    # manual skewness or scipy
    from scipy.stats import skew
    stats['f9'] = float(skew(xi))
    
    return stats
