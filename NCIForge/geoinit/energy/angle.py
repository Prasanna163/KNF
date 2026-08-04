"""Angle bend energy for the GeoInit functional.

Energy expression
-----------------
Φ_angle = Σ_{(i,j,k) ∈ angles} (cos θ_ijk − cos θ⁰_ijk)²

where *j* is the central atom and θ⁰ is determined by the coordination
number of *j*:

* coord 2 → 180° (linear)
* coord 3 → 120° (trigonal planar)
* coord 4 → 109.47° (tetrahedral)

Element‑specific overrides (applied via ``get_ideal_angle``):

* O with coord 2 → 104.5°
* N with coord 3 → 107°
* S with coord 2 → 92°
"""

from __future__ import annotations

import numpy as np

from geoinit.core.geometry import angle as compute_angle
from geoinit.core.params import get_ideal_angle


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def angle_energy(
    symbols: list[str],
    coords: np.ndarray,
    angles: list[tuple[int, int, int]],
    coordination_or_targets: list[int] | dict[tuple[int, int, int], float],
) -> float:
    """Compute the total angle bend energy.

    Parameters
    ----------
    symbols : list[str]
        Atomic symbols, length *N*.
    coords : np.ndarray, shape (N, 3)
        Cartesian coordinates in Å.
    angles : list[tuple[int, int, int]]
        Triples ``(i, j, k)`` where *j* is the central atom.
    coordination_or_targets : list or dict
        Coordination list or target angles dict.

    Returns
    -------
    float
        Total angle bend energy.
    """
    if not angles:
        return 0.0

    energy = 0.0
    for i, j, k in angles:
        theta = compute_angle(coords, i, j, k)
        if isinstance(coordination_or_targets, dict):
            theta0 = coordination_or_targets[(i, j, k)]
        else:
            theta0 = get_ideal_angle(symbols[j], coordination_or_targets[j])
        delta_cos = np.cos(theta) - np.cos(theta0)
        energy += delta_cos * delta_cos
    return energy


def angle_energy_decomposed(
    symbols: list[str],
    coords: np.ndarray,
    angles: list[tuple[int, int, int]],
    coordination_or_targets: list[int] | dict[tuple[int, int, int], float],
) -> list[dict]:
    """Return a per‑angle energy breakdown."""
    results: list[dict] = []
    for i, j, k in angles:
        theta = compute_angle(coords, i, j, k)
        if isinstance(coordination_or_targets, dict):
            theta0 = coordination_or_targets[(i, j, k)]
            coord_j = -1
        else:
            theta0 = get_ideal_angle(symbols[j], coordination_or_targets[j])
            coord_j = coordination_or_targets[j]

        delta_cos = np.cos(theta) - np.cos(theta0)
        e = delta_cos * delta_cos
        results.append(
            {
                "angle": (i, j, k),
                "central_sym": symbols[j],
                "coord_j": coord_j,
                "theta_deg": np.degrees(theta),
                "theta0_deg": np.degrees(theta0),
                "delta_cos": delta_cos,
                "energy": e,
            }
        )
    return results


def angle_gradient(
    symbols: list[str],
    coords: np.ndarray,
    angles: list[tuple[int, int, int]],
    coordination_or_targets: list[int] | dict[tuple[int, int, int], float],
) -> np.ndarray:
    """Compute the analytical gradient of the angle bend energy."""
    n_atoms = len(symbols)
    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    if not angles:
        return grad

    for i, j, k in angles:
        v1 = coords[i] - coords[j]
        v2 = coords[k] - coords[j]

        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))

        if norm1 < 1e-12 or norm2 < 1e-12:
            continue

        u1 = v1 / norm1
        u2 = v2 / norm2

        cos_theta = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
        if isinstance(coordination_or_targets, dict):
            theta0 = coordination_or_targets[(i, j, k)]
        else:
            theta0 = get_ideal_angle(symbols[j], coordination_or_targets[j])

        cos_theta0 = np.cos(theta0)

        factor = 2.0 * (cos_theta - cos_theta0)

        g_i = (u2 - cos_theta * u1) / norm1
        g_k = (u1 - cos_theta * u2) / norm2

        grad[i] += factor * g_i
        grad[k] += factor * g_k
        grad[j] -= factor * (g_i + g_k)

    return grad
