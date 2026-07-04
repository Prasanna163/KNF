"""RDKit UFF runner for GeoInit benchmarking.

Integrates RDKit's Universal Force Field to serve as a baseline comparison.
"""

from __future__ import annotations

import time
import numpy as np

from geoinit.core.topology import Topology
from geoinit.core.io_xyz import write_xyz


def run_uff_opt(
    symbols: list[str],
    coords: np.ndarray,
    topology: Topology,
    out_xyz_path: str,
) -> float:
    """Run RDKit UFF optimization and save coordinates to out_xyz_path.

    Parameters
    ----------
    symbols : list[str]
        Atomic element symbols.
    coords : np.ndarray, shape (N, 3)
        Input distorted coordinates in Å.
    topology : Topology
        Molecular topology.
    out_xyz_path : str
        Path where the optimized UFF coordinates will be written as XYZ.

    Returns
    -------
    float
        The wall-time duration in seconds spent running UFF optimization.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    coords = np.asarray(coords, dtype=np.float64)

    # 1. Build RDKit Mol object
    rw_mol = Chem.RWMol()
    conf = Chem.Conformer(len(symbols))

    for i, sym in enumerate(symbols):
        atom = Chem.Atom(sym)
        rw_mol.AddAtom(atom)
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))

    # Add single bonds for inferred connectivity
    for u, v in topology.bonds:
        rw_mol.AddBond(int(u), int(v), Chem.BondType.SINGLE)

    rw_mol.AddConformer(conf)

    # 2. Update properties and sanitize to initialize valences & hybridizations
    rw_mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(
        rw_mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
    )

    # 3. Perform UFF optimization
    t0 = time.perf_counter()
    AllChem.UFFOptimizeMolecule(rw_mol, maxIters=500)
    t1 = time.perf_counter()

    # 4. Retrieve optimized coordinates
    opt_conf = rw_mol.GetConformer()
    opt_coords = np.zeros_like(coords)
    for i in range(len(symbols)):
        pos = opt_conf.GetAtomPosition(i)
        opt_coords[i] = [pos.x, pos.y, pos.z]

    # 5. Write XYZ file
    write_xyz(out_xyz_path, symbols, opt_coords, comment="UFF optimized")

    return t1 - t0
