import numpy as np
import torch

from .molden import ANGSTROM_TO_BOHR
from .types import GridSpec


def build_grid(
    atoms_bohr: np.ndarray,
    spacing_angstrom: float = 0.2,
    padding_angstrom: float = 3.0,
) -> GridSpec:
    if spacing_angstrom <= 0:
        raise ValueError("Grid spacing must be > 0.")
    if padding_angstrom < 0:
        raise ValueError("Grid padding must be >= 0.")

    spacing_bohr = float(spacing_angstrom * ANGSTROM_TO_BOHR)
    padding_bohr = float(padding_angstrom * ANGSTROM_TO_BOHR)

    mins = atoms_bohr.min(axis=0) - padding_bohr
    maxs = atoms_bohr.max(axis=0) + padding_bohr

    x = np.arange(mins[0], maxs[0] + 0.5 * spacing_bohr, spacing_bohr, dtype=np.float64)
    y = np.arange(mins[1], maxs[1] + 0.5 * spacing_bohr, spacing_bohr, dtype=np.float64)
    z = np.arange(mins[2], maxs[2] + 0.5 * spacing_bohr, spacing_bohr, dtype=np.float64)

    return GridSpec(x_bohr=x, y_bohr=y, z_bohr=z, spacing_bohr=spacing_bohr)


def flatten_grid_points(spec: GridSpec, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = torch.as_tensor(spec.x_bohr, device=device, dtype=dtype)
    y = torch.as_tensor(spec.y_bohr, device=device, dtype=dtype)
    z = torch.as_tensor(spec.z_bohr, device=device, dtype=dtype)
    gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
    return torch.stack((gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)), dim=1)
