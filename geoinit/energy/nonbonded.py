"""Non‑bonded energy terms for the GeoInit functional.

Three contributions are provided:

1. **Clash** (soft exponential repulsion)::

       Φ_clash = Σ_{i<j, nb} exp[−α (r_ij − s_ij)]

   where  s_ij = 0.75 × (r_i^vdw + r_j^vdw)  and  α ≈ 3.0.

2. **Dispersion** (damped attractive −C₆/r⁶)::

       Φ_disp = −Σ_{i<j, nb} C6_ij / (r_ij⁶ + δ) × f_damp(r_ij)

   Becke‑Johnson damping:  f_damp = r_ij⁶ / (r_ij⁶ + s_ij⁶).
   C6_ij = √(C6_i × C6_j),  δ = 1.0 for numerical stability.

3. **Soft Coulomb** (optional)::

       Φ_coul = Σ_{i<j} q_i q_j / √(r_ij² + ε²)

   ε = 0.1 Å for smoothing.
"""

from __future__ import annotations

import numpy as np

from geoinit.core.geometry import distance
from geoinit.core.params import (
    DEFAULT_WEIGHTS,
    get_c6_pair,
    get_vdw_radius,
)


# ---------------------------------------------------------------------------
# Clash energy
# ---------------------------------------------------------------------------

def clash_energy(
    symbols: list[str],
    coords: np.ndarray,
    nonbonded_pairs: list[tuple[int, int]],
    alpha: float = 3.0,
    clash_mode: str = "compact",
    k: float = 100.0,
) -> float:
    """Compute soft exponential or compact clash repulsion.

    Parameters
    ----------
    symbols : list[str]
        Atomic symbols, length *N*.
    coords : np.ndarray, shape (N, 3)
        Cartesian coordinates in Å.
    nonbonded_pairs : list[tuple[int, int]]
        Non‑bonded atom index pairs.
    alpha : float, optional
        Steepness of the exponential wall (default 3.0 Å⁻¹).
    clash_mode : str, optional
        Clash mode: 'compact' or 'exp' (default 'compact').
    k : float, optional
        Compact clash force constant (default 100.0).

    Returns
    -------
    float
        Total clash energy.
    """
    if not nonbonded_pairs:
        return 0.0

    energy = 0.0
    for i, j in nonbonded_pairs:
        r_ij = distance(coords, i, j)
        s_ij = 0.75 * (get_vdw_radius(symbols[i]) + get_vdw_radius(symbols[j]))
        if clash_mode == "compact":
            if r_ij < s_ij:
                term = 1.0 - r_ij / s_ij
                energy += k * (term ** 4)
        else:
            energy += np.exp(-alpha * (r_ij - s_ij))
    return energy


# ---------------------------------------------------------------------------
# Dispersion energy
# ---------------------------------------------------------------------------

def dispersion_energy(
    symbols: list[str],
    coords: np.ndarray,
    nonbonded_pairs: list[tuple[int, int]],
    delta: float = 1.0,
) -> float:
    """Compute damped C₆ dispersion attraction.

    Uses Becke‑Johnson style damping so the potential is finite at *r = 0*.

    Parameters
    ----------
    symbols : list[str]
        Atomic symbols, length *N*.
    coords : np.ndarray, shape (N, 3)
        Cartesian coordinates in Å.
    nonbonded_pairs : list[tuple[int, int]]
        Non‑bonded atom index pairs.
    delta : float, optional
        Regularisation constant added to *r⁶* in the denominator
        (default 1.0).

    Returns
    -------
    float
        Total dispersion energy (negative = attractive).
    """
    if not nonbonded_pairs:
        return 0.0

    energy = 0.0
    for i, j in nonbonded_pairs:
        r_ij = distance(coords, i, j)
        c6_ij = get_c6_pair(symbols[i], symbols[j])
        s_ij = 0.75 * (get_vdw_radius(symbols[i]) + get_vdw_radius(symbols[j]))

        r6 = r_ij ** 6
        s6 = s_ij ** 6
        f_damp = r6 / (r6 + s6)
        energy -= c6_ij / (r6 + delta) * f_damp
    return energy


# ---------------------------------------------------------------------------
# Soft Coulomb energy
# ---------------------------------------------------------------------------

def coulomb_energy(
    symbols: list[str],
    coords: np.ndarray,
    nonbonded_pairs: list[tuple[int, int]],
    charges: np.ndarray | None = None,
    epsilon: float = 0.1,
) -> float:
    """Compute soft Coulomb interaction energy.

    Parameters
    ----------
    symbols : list[str]
        Atomic symbols, length *N*.  (Used only for consistency; charges
        are the primary input.)
    coords : np.ndarray, shape (N, 3)
        Cartesian coordinates in Å.
    nonbonded_pairs : list[tuple[int, int]]
        Non‑bonded atom index pairs.
    charges : np.ndarray or None, shape (N,)
        Per‑atom partial charges.  If *None*, all charges are zero and the
        function returns 0.0 immediately.
    epsilon : float, optional
        Smoothing parameter in Å (default 0.1).

    Returns
    -------
    float
        Total soft Coulomb energy.
    """
    if charges is None or not nonbonded_pairs:
        return 0.0

    energy = 0.0
    eps2 = epsilon * epsilon
    for i, j in nonbonded_pairs:
        r_ij = distance(coords, i, j)
        energy += charges[i] * charges[j] / np.sqrt(r_ij * r_ij + eps2)
    return energy


# ---------------------------------------------------------------------------
# Combined non‑bonded energy
# ---------------------------------------------------------------------------

def nonbonded_energy(
    symbols: list[str],
    coords: np.ndarray,
    nonbonded_pairs: list[tuple[int, int]],
    charges: np.ndarray | None = None,
    weights: dict | None = None,
) -> float:
    """Combined non‑bonded energy.

    .. math::

       Φ_{nb} = w_{clash} Φ_{clash} + w_{disp} Φ_{disp} + w_{coul} Φ_{coul}

    Parameters
    ----------
    symbols : list[str]
        Atomic symbols, length *N*.
    coords : np.ndarray, shape (N, 3)
        Cartesian coordinates in Å.
    nonbonded_pairs : list[tuple[int, int]]
        Non‑bonded atom index pairs.
    charges : np.ndarray or None, shape (N,)
        Per‑atom partial charges (optional).
    weights : dict or None
        Override weights.  Relevant keys: ``'clash'``, ``'disp'``,
        ``'coul'``.  Missing keys fall back to ``DEFAULT_WEIGHTS``.

    Returns
    -------
    float
        Weighted sum of clash, dispersion, and Coulomb energies.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        w.update(weights)

    e_clash = clash_energy(symbols, coords, nonbonded_pairs)
    e_disp = dispersion_energy(symbols, coords, nonbonded_pairs)
    e_coul = coulomb_energy(symbols, coords, nonbonded_pairs, charges=charges)

    return (
        w.get("clash", 1.0) * e_clash
        + w.get("disp", 0.1) * e_disp
        + w.get("coul", 0.0) * e_coul
    )


def clash_gradient(
    symbols: list[str],
    coords: np.ndarray,
    nonbonded_pairs: list[tuple[int, int]],
    alpha: float = 3.0,
    clash_mode: str = "compact",
    k: float = 100.0,
) -> np.ndarray:
    """Compute the analytical gradient of the clash repulsion."""
    n_atoms = len(symbols)
    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    if not nonbonded_pairs:
        return grad

    for i, j in nonbonded_pairs:
        diff = coords[i] - coords[j]
        r_ij = float(np.linalg.norm(diff))
        if r_ij < 1e-12:
            continue
        s_ij = 0.75 * (get_vdw_radius(symbols[i]) + get_vdw_radius(symbols[j]))

        if clash_mode == "compact":
            if r_ij < s_ij:
                term = 1.0 - r_ij / s_ij
                # dE/dr = -4 * k / s * (1 - r/s)^3
                dedr = -4.0 * k / s_ij * (term ** 3)
                factor = dedr / r_ij
                g_i = factor * diff
                grad[i] += g_i
                grad[j] -= g_i
        else:
            val = np.exp(-alpha * (r_ij - s_ij))
            factor = -alpha * val / r_ij
            g_i = factor * diff
            grad[i] += g_i
            grad[j] -= g_i

    return grad


def dispersion_gradient(
    symbols: list[str],
    coords: np.ndarray,
    nonbonded_pairs: list[tuple[int, int]],
    delta: float = 1.0,
) -> np.ndarray:
    """Compute the analytical gradient of the damped C₆ dispersion attraction.

    Parameters
    ----------
    symbols : list[str]
        Atomic symbols, length *N*.
    coords : np.ndarray, shape (N, 3)
        Cartesian coordinates in Å.
    nonbonded_pairs : list[tuple[int, int]]
        Non‑bonded atom index pairs.
    delta : float, optional
        Regularisation constant (default 1.0).

    Returns
    -------
    np.ndarray, shape (N, 3)
        Gradient matrix of the dispersion energy.
    """
    n_atoms = len(symbols)
    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    if not nonbonded_pairs:
        return grad

    for i, j in nonbonded_pairs:
        diff = coords[i] - coords[j]
        r_ij = float(np.linalg.norm(diff))
        if r_ij < 1e-12:
            continue
        c6_ij = get_c6_pair(symbols[i], symbols[j])
        s_ij = 0.75 * (get_vdw_radius(symbols[i]) + get_vdw_radius(symbols[j]))

        r6 = r_ij ** 6
        s6 = s_ij ** 6

        denom = ((r6 + delta) * (r6 + s6)) ** 2
        num = delta * s6 - r6 * r6
        factor = -6.0 * c6_ij * (r_ij**4) * num / denom

        g_i = factor * diff
        grad[i] += g_i
        grad[j] -= g_i

    return grad


def coulomb_gradient(
    symbols: list[str],
    coords: np.ndarray,
    nonbonded_pairs: list[tuple[int, int]],
    charges: np.ndarray | None = None,
    epsilon: float = 0.1,
) -> np.ndarray:
    """Compute the analytical gradient of the soft Coulomb interaction energy.

    Parameters
    ----------
    symbols : list[str]
        Atomic symbols, length *N*.
    coords : np.ndarray, shape (N, 3)
        Cartesian coordinates in Å.
    nonbonded_pairs : list[tuple[int, int]]
        Non‑bonded atom index pairs.
    charges : np.ndarray or None, shape (N,)
        Per‑atom partial charges.
    epsilon : float, optional
        Smoothing parameter in Å (default 0.1).

    Returns
    -------
    np.ndarray, shape (N, 3)
        Gradient matrix of the Coulomb energy.
    """
    n_atoms = len(symbols)
    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    if charges is None or not nonbonded_pairs:
        return grad

    eps2 = epsilon * epsilon
    for i, j in nonbonded_pairs:
        diff = coords[i] - coords[j]
        r_ij = float(np.linalg.norm(diff))

        dist_term = r_ij * r_ij + eps2
        factor = -charges[i] * charges[j] / (dist_term * np.sqrt(dist_term))
        g_i = factor * diff
        grad[i] += g_i
        grad[j] -= g_i

    return grad


def nonbonded_gradient(
    symbols: list[str],
    coords: np.ndarray,
    nonbonded_pairs: list[tuple[int, int]],
    charges: np.ndarray | None = None,
    weights: dict | None = None,
) -> np.ndarray:
    """Combined non‑bonded energy gradient.

    Parameters
    ----------
    symbols : list[str]
        Atomic symbols, length *N*.
    coords : np.ndarray, shape (N, 3)
        Cartesian coordinates in Å.
    nonbonded_pairs : list[tuple[int, int]]
        Non‑bonded atom index pairs.
    charges : np.ndarray or None, shape (N,)
        Per‑atom partial charges.
    weights : dict or None
        Override weights.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Combined gradient.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        w.update(weights)

    g_clash = clash_gradient(symbols, coords, nonbonded_pairs)
    g_disp = dispersion_gradient(symbols, coords, nonbonded_pairs)
    g_coul = coulomb_gradient(symbols, coords, nonbonded_pairs, charges=charges)

    return (
        w.get("clash", 1.0) * g_clash
        + w.get("disp", 0.1) * g_disp
        + w.get("coul", 0.0) * g_coul
    )
