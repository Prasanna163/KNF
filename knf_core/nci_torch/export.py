import numpy as np
import torch

from .molden import ANGSTROM_TO_BOHR
from .types import GridSpec


BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR


def write_nci_grid_npz(
    output_path: str,
    grid: GridSpec,
    sign_lambda2_rho: torch.Tensor,
    rdg: torch.Tensor,
    output_units: str = "bohr",
) -> None:
    unit = output_units.strip().lower()
    if unit not in {"bohr", "angstrom", "angs"}:
        raise ValueError("output_units must be 'bohr' or 'angstrom'.")

    x = grid.x_bohr
    y = grid.y_bohr
    z = grid.z_bohr
    if unit in {"angstrom", "angs"}:
        x = x * BOHR_TO_ANGSTROM
        y = y * BOHR_TO_ANGSTROM
        z = z * BOHR_TO_ANGSTROM

    sl2 = sign_lambda2_rho.detach().cpu().numpy()
    rdg_np = rdg.detach().cpu().numpy()

    np.savez(
        output_path,
        x=np.asarray(x),
        y=np.asarray(y),
        z=np.asarray(z),
        sign_lambda2_rho=np.asarray(sl2),
        rdg=np.asarray(rdg_np),
        output_units=np.asarray([unit]),
    )


def write_nci_grid_text(
    output_path: str,
    grid: GridSpec,
    sign_lambda2_rho: torch.Tensor,
    rdg: torch.Tensor,
    output_units: str = "bohr",
) -> None:
    unit = output_units.strip().lower()
    if unit not in {"bohr", "angstrom", "angs"}:
        raise ValueError("output_units must be 'bohr' or 'angstrom'.")

    x = grid.x_bohr
    y = grid.y_bohr
    z = grid.z_bohr
    if unit in {"angstrom", "angs"}:
        x = x * BOHR_TO_ANGSTROM
        y = y * BOHR_TO_ANGSTROM
        z = z * BOHR_TO_ANGSTROM

    sl2 = sign_lambda2_rho.detach().cpu().numpy()
    rdg_np = rdg.detach().cpu().numpy()
    nx, ny, nz = grid.shape

    yy, zz = np.meshgrid(y, z, indexing="ij")
    yz_coords = np.column_stack((yy.reshape(-1), zz.reshape(-1)))

    with open(output_path, "w", encoding="utf-8") as f:
        for ix in range(nx):
            x_col = np.full((ny * nz,), x[ix], dtype=np.float64)
            sl2_plane = sl2[ix, :, :].reshape(-1)
            rdg_plane = rdg_np[ix, :, :].reshape(-1)
            rows = np.column_stack((x_col, yz_coords[:, 0], yz_coords[:, 1], sl2_plane, rdg_plane))
            np.savetxt(f, rows, fmt="%.8f %.8f %.8f %.10e %.10e")
