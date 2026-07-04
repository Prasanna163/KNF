"""Post‑relaxation geometry quality guards.

Provides two utilities:

1. :func:`check_geometry` — analyse bond‑length accuracy and steric
   clashes for a single geometry.
2. :func:`compare_geometries` — compute RMSD, maximum displacement, and
   energy change between two geometries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geoinit.core.geometry import distance
from geoinit.core.params import get_covalent_radius, get_vdw_radius
from geoinit.core.topology import Topology


# ---------------------------------------------------------------------------
# Geometry report
# ---------------------------------------------------------------------------

@dataclass
class GeometryReport:
    """Report on geometry quality.

    Attributes
    ----------
    max_bond_error : float
        Maximum absolute deviation |r − r_ref| across all bonds (Å).
    mean_bond_error : float
        Mean absolute deviation |r − r_ref| across all bonds (Å).
    max_clash_ratio : float
        Maximum clash ratio  C = 0.75 (r_i^vdw + r_j^vdw) / r_ij  among
        nonbonded pairs.  Values > 1.0 indicate overlap.
    n_clashes : int
        Number of nonbonded pairs with *C* > 1.0.
    is_safe : bool
        ``True`` if ``max_clash_ratio < 1.2``.
    bond_details : list[dict]
        Per‑bond information: indices, symbols, actual/reference length,
        and absolute error.
    """

    max_bond_error: float = 0.0
    mean_bond_error: float = 0.0
    max_clash_ratio: float = 0.0
    n_clashes: int = 0
    is_safe: bool = True
    bond_details: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geometry checker
# ---------------------------------------------------------------------------

def check_geometry(
    symbols: list[str],
    coords: np.ndarray,
    topology: Topology | None = None,
    scale: float = 1.25,
) -> GeometryReport:
    """Analyse geometry quality: bond accuracy and steric clashes.

    Parameters
    ----------
    symbols : list[str]
        Atomic symbols, length *N*.
    coords : np.ndarray, shape (N, 3)
        Cartesian coordinates in Å.
    topology : Topology or None
        Pre‑built topology.  If *None*, one is inferred from *coords*.
    scale : float
        Covalent‑radii scaling factor for bond inference (used only when
        *topology* is ``None``).

    Returns
    -------
    GeometryReport
        Comprehensive geometry quality metrics.
    """
    coords = np.asarray(coords, dtype=np.float64)

    if topology is None:
        topology = Topology(symbols, coords, scale=scale)

    # --- Bond accuracy ---------------------------------------------------
    bond_details: list[dict] = []
    bond_errors: list[float] = []

    for b in topology.bonds:
        if hasattr(b, "r0"):
            i, j = b.i, b.j
            r_ref = b.r0
        else:
            i, j = b
            r_ref = get_covalent_radius(symbols[i]) + get_covalent_radius(symbols[j])
        r_ij = distance(coords, i, j)
        error = abs(r_ij - r_ref)
        bond_errors.append(error)
        bond_details.append(
            {
                "bond": (i, j),
                "symbols": (symbols[i], symbols[j]),
                "r_ij": r_ij,
                "r_ref": r_ref,
                "error": error,
            }
        )

    max_bond_error = max(bond_errors) if bond_errors else 0.0
    mean_bond_error = float(np.mean(bond_errors)) if bond_errors else 0.0

    # --- Clash detection --------------------------------------------------
    max_clash_ratio = 0.0
    n_clashes = 0

    for i, j in topology.nonbonded_pairs:
        r_ij = distance(coords, i, j)
        s_ij = 0.75 * (get_vdw_radius(symbols[i]) + get_vdw_radius(symbols[j]))
        # Clash ratio: how much the vdw shield overlaps relative to distance
        if r_ij > 0.0:
            ratio = s_ij / r_ij
        else:
            ratio = float("inf")
        if ratio > max_clash_ratio:
            max_clash_ratio = ratio
        if ratio > 1.0:
            n_clashes += 1

    is_safe = max_clash_ratio < 1.2

    return GeometryReport(
        max_bond_error=max_bond_error,
        mean_bond_error=mean_bond_error,
        max_clash_ratio=max_clash_ratio,
        n_clashes=n_clashes,
        is_safe=is_safe,
        bond_details=bond_details,
    )


# ---------------------------------------------------------------------------
# Geometry comparison
# ---------------------------------------------------------------------------

def compare_geometries(
    symbols: list[str],
    coords_before: np.ndarray,
    coords_after: np.ndarray,
) -> dict:
    """Compare two geometries of the same molecule.

    Parameters
    ----------
    symbols : list[str]
        Atomic symbols, length *N*.
    coords_before : np.ndarray, shape (N, 3)
        Reference (e.g. initial) coordinates.
    coords_after : np.ndarray, shape (N, 3)
        Comparison (e.g. relaxed) coordinates.

    Returns
    -------
    dict
        * ``'rmsd'``              – root‑mean‑square displacement (Å)
        * ``'max_displacement'``  – maximum per‑atom displacement (Å)
        * ``'per_atom_disp'``     – (N,) array of per‑atom displacements
        * ``'report_before'``     – :class:`GeometryReport` for *before*
        * ``'report_after'``      – :class:`GeometryReport` for *after*
    """
    coords_before = np.asarray(coords_before, dtype=np.float64)
    coords_after = np.asarray(coords_after, dtype=np.float64)

    diff = coords_after - coords_before
    per_atom_disp = np.linalg.norm(diff, axis=1)
    rmsd = float(np.sqrt(np.mean(per_atom_disp ** 2)))
    max_disp = float(np.max(per_atom_disp))

    report_before = check_geometry(symbols, coords_before)
    report_after = check_geometry(symbols, coords_after)

    return {
        "rmsd": rmsd,
        "max_displacement": max_disp,
        "per_atom_disp": per_atom_disp,
        "report_before": report_before,
        "report_after": report_after,
    }


def check_aromatic_planarity(coords: np.ndarray, ring: list[int]) -> float:
    """Compute the maximum deviation of ring atoms from their best-fit plane using SVD."""
    pts = coords[ring]
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    try:
        _, _, Vt = np.linalg.svd(centered)
        normal = Vt[-1, :]
        dists = np.abs(np.dot(centered, normal))
        return float(np.max(dists))
    except Exception:
        return 0.0


def check_damage(
    symbols: list[str],
    coords_raw: np.ndarray,
    coords_relaxed: np.ndarray,
    topology: Topology,
) -> bool:
    """Check whether the relaxation functional damaged chemically important bonds or structure.

    Returns True if damaged, False otherwise.
    """
    coords_raw = np.asarray(coords_raw, dtype=np.float64)
    coords_relaxed = np.asarray(coords_relaxed, dtype=np.float64)

    # 1. Multiple-bond check
    for b in topology.bonds:
        order = getattr(b, "order", 1.0)
        label = getattr(b, "label", "single")
        r0 = getattr(b, "r0", None)
        if r0 is None:
            continue

        r_relaxed = distance(coords_relaxed, b.i, b.j)

        # General linear fragment check
        if label in ("CO2 C=O", "nitrile C#N", "alkyne C#C") and r_relaxed > 1.25:
            return True

        # Multiple bond error > 0.03 A
        if order > 1.0:
            if abs(r_relaxed - r0) > 0.03:
                return True

    # 2. Aromatic planarity check
    from geoinit.core.bond_rules import detect_rigid_subgraphs
    subgraphs = detect_rigid_subgraphs(symbols, topology)
    for g in subgraphs:
        if len(g) in (5, 6):
            bonds_in_g = sum(1 for b in topology.bonds if b.i in g and b.j in g)
            if bonds_in_g >= len(g):
                dev = check_aromatic_planarity(coords_relaxed, g)
                if dev > 0.15:
                    return True

    # 3. COM distance change for complexes
    from collections import defaultdict
    adj = defaultdict(list)
    for b in topology.bonds:
        adj[b.i].append(b.j)
        adj[b.j].append(b.i)

    visited = set()
    fragments = []
    n_atoms = len(symbols)
    for i in range(n_atoms):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for nbr in adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            fragments.append(sorted(comp))

    if len(fragments) >= 2:
        for idx_a in range(len(fragments)):
            for idx_b in range(idx_a + 1, len(fragments)):
                frag_a = fragments[idx_a]
                frag_b = fragments[idx_b]

                com_a_raw = np.mean(coords_raw[frag_a], axis=0)
                com_b_raw = np.mean(coords_raw[frag_b], axis=0)
                dist_raw = float(np.linalg.norm(com_a_raw - com_b_raw))

                com_a_relaxed = np.mean(coords_relaxed[frag_a], axis=0)
                com_b_relaxed = np.mean(coords_relaxed[frag_b], axis=0)
                dist_relaxed = float(np.linalg.norm(com_a_relaxed - com_b_relaxed))

                if dist_relaxed - dist_raw > 2.0:
                    return True

    return False


def accept_geoinit(
    symbols: list[str],
    coords_raw: np.ndarray,
    coords_relaxed: np.ndarray,
    topology: Topology,
) -> tuple[bool, str]:
    """Evaluate whether to accept the GeoInit relaxed geometry or fall back to raw.

    Parameters
    ----------
    symbols : list[str]
        Atomic element symbols.
    coords_raw : np.ndarray, shape (N, 3)
        Initial (raw distorted) coordinates.
    coords_relaxed : np.ndarray, shape (N, 3)
        Relaxed coordinates from GeoInit.
    topology : Topology
        Molecular topology.

    Returns
    -------
    tuple[bool, str]
        * (True, "") if accepted.
        * (False, fallback_reason) if rejected.
    """
    coords_raw = np.asarray(coords_raw, dtype=np.float64)
    coords_relaxed = np.asarray(coords_relaxed, dtype=np.float64)

    # 1. Damage check (linear, multiple bond distortion, planarity)
    for b in topology.bonds:
        order = getattr(b, "order", 1.0)
        label = getattr(b, "label", "single")
        r0 = getattr(b, "r0", None)
        if r0 is None:
            continue

        r_relaxed = distance(coords_relaxed, b.i, b.j)

        # General linear fragment check
        if label in ("CO2 C=O", "nitrile C#N", "alkyne C#C") and r_relaxed > 1.25:
            return False, "linear_fragment_damage"

        # Multiple bond error > 0.03 A
        if order > 1.0:
            if abs(r_relaxed - r0) > 0.03:
                return False, "multiple_bond_damage"

    # 2. Aromatic planarity check
    from geoinit.core.bond_rules import detect_rigid_subgraphs
    subgraphs = detect_rigid_subgraphs(symbols, topology)
    for g in subgraphs:
        if len(g) in (5, 6):
            bonds_in_g = sum(1 for b in topology.bonds if b.i in g and b.j in g)
            if bonds_in_g >= len(g):
                dev = check_aromatic_planarity(coords_relaxed, g)
                if dev > 0.15:
                    return False, "aromatic_planarity_damage"

    # 3. Fragment connectivity and drift check
    from collections import defaultdict
    adj = defaultdict(list)
    for b in topology.bonds:
        adj[b.i].append(b.j)
        adj[b.j].append(b.i)

    visited = set()
    fragments = []
    n_atoms = len(symbols)
    for i in range(n_atoms):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for nbr in adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            fragments.append(sorted(comp))

    is_complex = len(fragments) >= 2

    if is_complex:
        # Check fragment drift
        for idx_a in range(len(fragments)):
            for idx_b in range(idx_a + 1, len(fragments)):
                frag_a = fragments[idx_a]
                frag_b = fragments[idx_b]

                com_a_raw = np.mean(coords_raw[frag_a], axis=0)
                com_b_raw = np.mean(coords_raw[frag_b], axis=0)
                dist_raw = float(np.linalg.norm(com_a_raw - com_b_raw))

                com_a_relaxed = np.mean(coords_relaxed[frag_a], axis=0)
                com_b_relaxed = np.mean(coords_relaxed[frag_b], axis=0)
                dist_relaxed = float(np.linalg.norm(com_a_relaxed - com_b_relaxed))

                if dist_relaxed - dist_raw > 2.0:
                    return False, "fragment_drift"

        # Check complex-specific placement guards
        frag_A = fragments[0]
        frag_B = fragments[1]

        # Detect contact pairs to exclude from clashes/minimum distance
        from geoinit.optimize.complex import detect_contact_pairs
        contact_pairs = detect_contact_pairs(symbols, topology, fragments)
        contact_indices = {tuple(sorted(cp["pair"])) for cp in contact_pairs if "pair" in cp}

        # Minimum interfragment distance (excluding contact pairs)
        min_dist = float("inf")
        for i in frag_A:
            for j in frag_B:
                if tuple(sorted((i, j))) in contact_indices:
                    continue
                d = distance(coords_relaxed, i, j)
                if d < min_dist:
                    min_dist = d
        if min_dist < 1.75:
            return False, "too_short_interfragment_distance"

        # Inter-fragment clashes
        inter_clashes = 0
        set_A = set(frag_A)
        set_B = set(frag_B)
        for i, j in topology.nonbonded_pairs:
            if (i in set_A and j in set_B) or (i in set_B and j in set_A):
                if tuple(sorted((i, j))) in contact_indices:
                    continue
                r_ij = distance(coords_relaxed, i, j)
                s_ij = 0.75 * (get_vdw_radius(symbols[i]) + get_vdw_radius(symbols[j]))
                if r_ij > 0.0 and (s_ij / r_ij) > 1.0:
                    inter_clashes += 1

        if inter_clashes > 5:
            return False, "too_many_interfragment_clashes"

        return True, ""

    else:
        # Single molecule check
        report = check_geometry(symbols, coords_relaxed, topology=topology)
        safe_success = bool(report.is_safe and report.max_bond_error < 0.01 and report.n_clashes == 0)
        if not safe_success:
            return False, "unsafe_single_molecule"
        return True, ""


def should_skip_geoinit(
    symbols: list[str],
    coords: np.ndarray,
    topology: Topology | None = None,
) -> tuple[bool, str]:
    """Evaluate raw geometry features to decide if we should skip GeoInit relaxation."""
    coords = np.asarray(coords, dtype=np.float64)
    N = len(symbols)

    # 1. Size guard: N <= 3 is too small to warrant overhead
    if N <= 3:
        return True, "too_small"

    if topology is None:
        topology = Topology(symbols, coords)

    # Inbuilt fragment detection
    from collections import defaultdict
    adj = defaultdict(list)
    for b in topology.bonds:
        adj[b.i].append(b.j)
        adj[b.j].append(b.i)

    visited = set()
    fragments = []
    for i in range(N):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for nbr in adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            fragments.append(sorted(comp))

    is_complex = len(fragments) >= 2
    report_raw = check_geometry(symbols, coords, topology=topology)

    if not is_complex:
        if report_raw.max_bond_error < 0.015 and report_raw.n_clashes == 0:
            return True, "already_safe_low_benefit"
        if report_raw.max_bond_error > 0.45 or report_raw.n_clashes > 15:
            return True, "extreme_distortion_likely_unsafe"
    else:
        frag_A = fragments[0]
        frag_B = fragments[1]

        # Detect contact pairs to exclude from clash checks
        from geoinit.optimize.complex import detect_contact_pairs
        contact_pairs = detect_contact_pairs(symbols, topology, fragments)
        contact_indices = {tuple(sorted(cp["pair"])) for cp in contact_pairs if "pair" in cp}

        # Raw min interfragment distance (excluding contact pairs)
        min_dist_raw = float("inf")
        for i in frag_A:
            for j in frag_B:
                if tuple(sorted((i, j))) in contact_indices:
                    continue
                d = distance(coords, i, j)
                if d < min_dist_raw:
                    min_dist_raw = d

        # Raw interfragment clash count (excluding contact pairs)
        inter_clashes_raw = 0
        set_A = set(frag_A)
        set_B = set(frag_B)
        for i, j in topology.nonbonded_pairs:
            if (i in set_A and j in set_B) or (i in set_B and j in set_A):
                if tuple(sorted((i, j))) in contact_indices:
                    continue
                r_ij = distance(coords, i, j)
                s_ij = 0.75 * (get_vdw_radius(symbols[i]) + get_vdw_radius(symbols[j]))
                if r_ij > 0.0 and (s_ij / r_ij) > 1.0:
                    inter_clashes_raw += 1

        if inter_clashes_raw == 0 and min_dist_raw > 2.2:
            return True, "complex_already_safe_low_benefit"
        if min_dist_raw < 1.1 or inter_clashes_raw > 15:
            return True, "complex_extreme_clash_likely_unsafe"

    return False, ""
