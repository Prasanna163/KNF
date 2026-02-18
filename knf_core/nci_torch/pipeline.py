import time
from typing import Dict

from .engine import NCIConfig, run_nci_engine
from .export import write_nci_grid_text
from .grid import build_grid
from .molden import parse_molden


def run_nci_torch(
    molden_path: str,
    output_path: str,
    spacing_angstrom: float = 0.2,
    padding_angstrom: float = 3.0,
    device: str = "auto",
    dtype: str = "float32",
    batch_size: int = 250000,
    rho_floor: float = 1e-12,
    output_units: str = "bohr",
    apply_primitive_normalization: bool = False,
) -> Dict[str, object]:
    t0 = time.perf_counter()
    wavefunction = parse_molden(
        molden_path,
        apply_primitive_normalization=apply_primitive_normalization,
    )
    grid = build_grid(
        atoms_bohr=wavefunction.atoms_bohr,
        spacing_angstrom=spacing_angstrom,
        padding_angstrom=padding_angstrom,
    )

    fields, resolved_device = run_nci_engine(
        wavefunction=wavefunction,
        grid=grid,
        config=NCIConfig(
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            rho_floor=rho_floor,
        ),
    )

    write_nci_grid_text(
        output_path=output_path,
        grid=grid,
        sign_lambda2_rho=fields.sign_lambda2_rho,
        rdg=fields.rdg,
        output_units=output_units,
    )

    elapsed = time.perf_counter() - t0
    return {
        "device": str(resolved_device),
        "elapsed_seconds": elapsed,
        "n_atoms": int(wavefunction.atoms_bohr.shape[0]),
        "n_basis": int(len(wavefunction.basis_functions)),
        "grid_shape": grid.shape,
        "n_grid_points": int(grid.n_points),
        "apply_primitive_normalization": bool(apply_primitive_normalization),
    }
