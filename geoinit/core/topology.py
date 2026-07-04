"""
Topology inference from Cartesian coordinates and covalent radii.

This module detects bonds, valence angles, coordination numbers, and
non-bonded pair lists by comparing inter-atomic distances against
scaled sums of covalent radii.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from geoinit.core.geometry import distance_matrix
from geoinit.core.params import get_covalent_radius


# ===================================================================
# Free functions
# ===================================================================

def infer_bonds(
    symbols: list[str],
    coords: np.ndarray,
    scale: float = 1.25,
) -> list[tuple[int, int]]:
    """Infer covalent bonds from geometry using covalent radii.

    A bond is assigned between atoms *i* and *j* when

    .. math::

        r_{ij} < \\text{scale} \\times (r_i^{\\text{cov}} + r_j^{\\text{cov}})

    Parameters
    ----------
    symbols : list[str]
        Element symbols, length *N*.
    coords : np.ndarray
        Cartesian coordinates, shape ``(N, 3)``.
    scale : float, optional
        Multiplicative tolerance factor (default 1.25).

    Returns
    -------
    list[tuple[int, int]]
        Sorted list of ``(i, j)`` bond pairs with ``i < j``.

    Examples
    --------
    >>> import numpy as np
    >>> syms = ["O", "H", "H"]
    >>> xyz = np.array([[0.0, 0.0, 0.0],
    ...                 [0.96, 0.0, 0.0],
    ...                 [-0.24, 0.93, 0.0]])
    >>> infer_bonds(syms, xyz)
    [(0, 1), (0, 2)]
    """
    n = len(symbols)
    dmat = distance_matrix(coords)
    bonds: list[tuple[int, int]] = []

    for i in range(n):
        ri = get_covalent_radius(symbols[i])
        for j in range(i + 1, n):
            rj = get_covalent_radius(symbols[j])
            if dmat[i, j] < scale * (ri + rj):
                bonds.append((i, j))

    return bonds


def infer_angles(
    bonds: list[tuple[int, int]],
    n_atoms: int,
) -> list[tuple[int, int, int]]:
    """Generate valence-angle triplets from a bond list.

    An angle ``(i, j, k)`` is created for every pair of bonds that share
    atom *j* as a common vertex, with ``i < k``.

    Parameters
    ----------
    bonds : list[tuple[int, int]]
        Bond list from :func:`infer_bonds`.
    n_atoms : int
        Total number of atoms (used to size the adjacency structure).

    Returns
    -------
    list[tuple[int, int, int]]
        Angle triplets ``(i, j, k)`` where *j* is the central atom and
        ``i < k``.
    """
    # Build adjacency list
    neighbours: dict[int, list[int]] = defaultdict(list)
    for i, j in bonds:
        neighbours[i].append(j)
        neighbours[j].append(i)

    angles: list[tuple[int, int, int]] = []
    for center in range(n_atoms):
        nbrs = sorted(neighbours[center])
        for a, b in combinations(nbrs, 2):
            # a < b guaranteed by sorted + combinations
            angles.append((a, center, b))

    return angles


def get_coordination(
    bonds: list[tuple[int, int]],
    n_atoms: int,
) -> list[int]:
    """Return coordination number (number of bonded neighbours) per atom.

    Parameters
    ----------
    bonds : list[tuple[int, int]]
        Bond list.
    n_atoms : int
        Total number of atoms.

    Returns
    -------
    list[int]
        Coordination numbers indexed by atom index.
    """
    coord = [0] * n_atoms
    for i, j in bonds:
        coord[i] += 1
        coord[j] += 1
    return coord


def get_nonbonded_pairs(
    n_atoms: int,
    bonds: list[tuple[int, int]],
    exclude_13: bool = True,
) -> list[tuple[int, int]]:
    """Return non-bonded atom pairs suitable for clash / dispersion terms.

    1-2 pairs (directly bonded) are always excluded.  1-3 pairs (atoms
    sharing a common bonded neighbour) are excluded when *exclude_13* is
    ``True`` (the default), which avoids double-counting with the angle
    term.

    Parameters
    ----------
    n_atoms : int
        Total number of atoms.
    bonds : list[tuple[int, int]]
        Bond list.
    exclude_13 : bool, optional
        Whether to also exclude 1-3 (angle) pairs (default ``True``).

    Returns
    -------
    list[tuple[int, int]]
        Sorted list of ``(i, j)`` non-bonded pairs with ``i < j``.
    """
    excluded: set[tuple[int, int]] = set()

    # 1-2 exclusions
    for i, j in bonds:
        pair = (min(i, j), max(i, j))
        excluded.add(pair)

    # 1-3 exclusions
    if exclude_13:
        neighbours: dict[int, list[int]] = defaultdict(list)
        for i, j in bonds:
            neighbours[i].append(j)
            neighbours[j].append(i)

        for center in range(n_atoms):
            nbrs = neighbours[center]
            for a, b in combinations(nbrs, 2):
                pair = (min(a, b), max(a, b))
                excluded.add(pair)

    # All pairs minus excluded
    nb_pairs: list[tuple[int, int]] = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if (i, j) not in excluded:
                nb_pairs.append((i, j))

    return nb_pairs


# ===================================================================
# Topology container
# ===================================================================

class Topology:
    """Container for the inferred molecular topology.

    On construction the bond list, angle list, per-atom coordination
    numbers, and non-bonded pair list are computed from Cartesian
    coordinates and covalent radii.

    Parameters
    ----------
    symbols : list[str]
        Element symbols, length *N*.
    coords : np.ndarray
        Cartesian coordinates, shape ``(N, 3)``.
    scale : float, optional
        Covalent-radius scaling factor passed to :func:`infer_bonds`
        (default 1.25).

    Attributes
    ----------
    bonds : list[tuple[int, int]]
        Detected covalent bonds.
    angles : list[tuple[int, int, int]]
        Detected valence angles.
    coordination : list[int]
        Coordination number per atom.
    nonbonded_pairs : list[tuple[int, int]]
        Non-bonded atom pairs (1-2 and 1-3 excluded).

    Examples
    --------
    >>> import numpy as np
    >>> syms = ["O", "H", "H"]
    >>> xyz = np.array([[0.0, 0.0, 0.0],
    ...                 [0.96, 0.0, 0.0],
    ...                 [-0.24, 0.93, 0.0]])
    >>> topo = Topology(syms, xyz)
    >>> topo.bonds
    [(0, 1), (0, 2)]
    >>> topo.coordination
    [2, 1, 1]
    """

    def __init__(
        self,
        symbols: list[str],
        coords: np.ndarray,
        scale: float = 1.25,
        sigma: float = 0.05,
    ) -> None:
        from geoinit.core.bond_rules import (
            assign_bond_orders,
            assign_reference_lengths,
            assign_angle_targets,
        )
        raw_bonds = infer_bonds(symbols, coords, scale)
        bonds_meta = assign_bond_orders(symbols, coords, raw_bonds)
        self.bonds = assign_reference_lengths(symbols, bonds_meta, sigma)

        self.angles: list[tuple[int, int, int]] = infer_angles(
            self.bonds, len(symbols)
        )
        self.coordination: list[int] = get_coordination(self.bonds, len(symbols))
        self.angle_targets: dict[tuple[int, int, int], float] = assign_angle_targets(symbols, self)

        self.nonbonded_pairs: list[tuple[int, int]] = get_nonbonded_pairs(
            len(symbols), self.bonds
        )
        self.reference_coords = coords.copy()

    def __repr__(self) -> str:
        return (
            f"Topology(bonds={len(self.bonds)}, "
            f"angles={len(self.angles)}, "
            f"nonbonded={len(self.nonbonded_pairs)})"
        )
