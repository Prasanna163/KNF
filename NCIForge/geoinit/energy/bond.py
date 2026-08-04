"""Bond stretch energy for the GeoInit functional.

Energy expression
-----------------
Φ_bond = Σ_{(i,j) ∈ bonds} ((r_ij − r⁰_ij) / σ_ij)²

where
    r⁰_ij = r_i^cov + r_j^cov   (ideal bond length from covalent radii)
    σ_ij   = tolerance in Å (default 0.05 Å)

A smaller σ produces a steeper well, enforcing tighter bond‑length adherence.
"""

from __future__ import annotations

import numpy as np

from geoinit.core.geometry import distance
from geoinit.core.params import get_covalent_radius


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bond_energy(
    symbols: list[str],
    coords: np.ndarray,
    bonds: list[Bond] | list[tuple[int, int]],
    sigma: float = 0.05,
) -> float:
    """Compute the total bond stretch energy.

    Parameters
    ----------
    symbols : list[str]
        Atomic symbols, length *N*.
    coords : np.ndarray, shape (N, 3)
        Cartesian coordinates in Å.
    bonds : list
        Bond representation list.
    sigma : float, optional
        Bond‑length tolerance in Å (default 0.05).

    Returns
    -------
    float
        Total bond stretch energy.
    """
    if not bonds:
        return 0.0

    energy = 0.0
    for b in bonds:
        if hasattr(b, "r0"):
            i, j = b.i, b.j
            r0 = b.r0
            k = b.k
        else:
            i, j = b
            r0 = get_covalent_radius(symbols[i]) + get_covalent_radius(symbols[j])
            k = 1.0 / (sigma * sigma)

        r_ij = distance(coords, i, j)
        deviation = r_ij - r0
        energy += k * deviation * deviation
    return energy


def bond_energy_decomposed(
    symbols: list[str],
    coords: np.ndarray,
    bonds: list[Bond] | list[tuple[int, int]],
    sigma: float = 0.05,
) -> list[dict]:
    """Return a per‑bond energy breakdown."""
    results: list[dict] = []
    for b in bonds:
        if hasattr(b, "r0"):
            i, j = b.i, b.j
            r0 = b.r0
            k = b.k
        else:
            i, j = b
            r0 = get_covalent_radius(symbols[i]) + get_covalent_radius(symbols[j])
            k = 1.0 / (sigma * sigma)

        r_ij = distance(coords, i, j)
        dev = r_ij - r0
        e = k * dev * dev
        results.append(
            {
                "bond": (i, j),
                "symbols": (symbols[i], symbols[j]),
                "r_ij": r_ij,
                "r0": r0,
                "deviation": dev,
                "energy": e,
            }
        )
    return results


def bond_gradient(
    symbols: list[str],
    coords: np.ndarray,
    bonds: list[Bond] | list[tuple[int, int]],
    sigma: float = 0.05,
) -> np.ndarray:
    """Compute the analytical gradient of the bond stretch energy."""
    n_atoms = len(symbols)
    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    if not bonds:
        return grad

    for b in bonds:
        if hasattr(b, "r0"):
            i, j = b.i, b.j
            r0 = b.r0
            k = b.k
        else:
            i, j = b
            r0 = get_covalent_radius(symbols[i]) + get_covalent_radius(symbols[j])
            k = 1.0 / (sigma * sigma)

        diff = coords[i] - coords[j]
        r_ij = float(np.linalg.norm(diff))
        if r_ij < 1e-12:
            continue
        factor = 2.0 * k * (r_ij - r0) / r_ij
        g_i = factor * diff
        grad[i] += g_i
        grad[j] -= g_i

    return grad
