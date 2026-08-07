"""Fragment-based complex optimization for GeoInit V0.2.

This module implements relax_complex, which optimizes guest translation/rotation
while keeping fragments internally rigid, followed by a light all-atom cleanup.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

from geoinit.core.geometry import distance
from geoinit.core.topology import Topology
from geoinit.energy.nonbonded import clash_energy, dispersion_energy, coulomb_energy
from geoinit.optimize.relax import RelaxResult


def rotation_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Compute 3D rotation matrix from Euler angles (in radians)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    Rx = np.array([[1, 0, 0],
                   [0, ca, -sa],
                   [0, sa, ca]])
    Ry = np.array([[cb, 0, sb],
                   [0, 1, 0],
                   [-sb, 0, cb]])
    Rz = np.array([[cg, -sg, 0],
                   [sg, cg, 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


def get_inter_pairs(
    nonbonded_pairs: list[tuple[int, int]],
    fragments: list[list[int]],
) -> list[tuple[int, int]]:
    """Filter non-bonded pairs to keep only inter-fragment pairs."""
    # Build a lookup for which fragment each atom belongs to
    atom_frag = {}
    for f_idx, frag in enumerate(fragments):
        for atom in frag:
            atom_frag[atom] = f_idx

    inter_pairs = []
    for i, j in nonbonded_pairs:
        if i in atom_frag and j in atom_frag:
            if atom_frag[i] != atom_frag[j]:
                inter_pairs.append((i, j))
    return inter_pairs


def detect_contact_pairs(
    symbols: list[str],
    topology: Topology,
    fragments: list[list[int]],
) -> list[dict]:
    """Detect likely interacting contact pairs (H-bonds, halogen, polar, and pi-contacts)."""
    # 1. Map atoms to fragment index
    atom_frag = {}
    for f_idx, frag in enumerate(fragments):
        for atom in frag:
            atom_frag[atom] = f_idx

    # Build coordination adjacency
    neighbors = defaultdict(list)
    for b in topology.bonds:
        neighbors[b.i].append(b.j)
        neighbors[b.j].append(b.i)

    # Find H-bond donors (H attached to O, N, S, F)
    donors = []
    for idx, sym in enumerate(symbols):
        if sym == "H":
            for nbr in neighbors[idx]:
                if symbols[nbr] in ("O", "N", "S", "F"):
                    donors.append(idx)
                    break

    # Find aromatic rings
    from geoinit.core.bond_rules import detect_rigid_subgraphs
    subgraphs = detect_rigid_subgraphs(symbols, topology)
    rings = []
    for g in subgraphs:
        if len(g) in (5, 6):
            bonds_in_g = sum(1 for b in topology.bonds if b.i in g and b.j in g)
            if bonds_in_g >= len(g):
                rings.append(g)

    # Group rings by fragment index
    frag_rings = defaultdict(list)
    for r in rings:
        f_idx = atom_frag[r[0]]
        frag_rings[f_idx].append(r)

    contact_pairs = []

    # Check all inter-fragment pairs
    n_frags = len(fragments)
    for f_a in range(n_frags):
        for f_b in range(f_a + 1, n_frags):
            # 1. Centroid-based contacts
            rings_a = frag_rings[f_a]
            rings_b = frag_rings[f_b]
            for r_a in rings_a:
                for r_b in rings_b:
                    contact_pairs.append({
                        "centroid_a": r_a,
                        "centroid_b": r_b,
                        "min_r": 3.2,
                        "max_r": 4.0,
                        "k": 5.0,
                        "label": "pi-pi"
                    })
                # Ring A to polar atom or donor H in B
                for idx_b in fragments[f_b]:
                    sym_b = symbols[idx_b]
                    if sym_b in ("O", "N", "S", "F") or idx_b in donors:
                        contact_pairs.append({
                            "centroid_a": r_a,
                            "atom_b": idx_b,
                            "min_r": 3.0,
                            "max_r": 3.8,
                            "k": 5.0,
                            "label": "pi-polar"
                        })
            for r_b in rings_b:
                # Ring B to polar atom or donor H in A
                for idx_a in fragments[f_a]:
                    sym_a = symbols[idx_a]
                    if sym_a in ("O", "N", "S", "F") or idx_a in donors:
                        contact_pairs.append({
                            "centroid_a": r_b,
                            "atom_b": idx_a,
                            "min_r": 3.0,
                            "max_r": 3.8,
                            "k": 5.0,
                            "label": "pi-polar"
                        })

            # 2. Atom-atom contacts
            for idx_a in fragments[f_a]:
                sym_a = symbols[idx_a]
                for idx_b in fragments[f_b]:
                    sym_b = symbols[idx_b]

                    # H-bonds: donor H in A to acceptor in B
                    if idx_a in donors and sym_b in ("O", "N", "S", "F"):
                        contact_pairs.append({
                            "pair": (idx_a, idx_b),
                            "min_r": 1.7,
                            "max_r": 2.3,
                            "k": 10.0,
                            "label": "H-bond"
                        })
                    elif idx_b in donors and sym_a in ("O", "N", "S", "F"):
                        contact_pairs.append({
                            "pair": (idx_a, idx_b),
                            "min_r": 1.7,
                            "max_r": 2.3,
                            "k": 10.0,
                            "label": "H-bond"
                        })

                    # Halogen contacts: Cl/Br/I to acceptor O/N/S
                    elif sym_a in ("Cl", "Br", "I") and sym_b in ("O", "N", "S"):
                        contact_pairs.append({
                            "pair": (idx_a, idx_b),
                            "min_r": 2.8,
                            "max_r": 3.6,
                            "k": 5.0,
                            "label": "halogen"
                        })
                    elif sym_b in ("Cl", "Br", "I") and sym_a in ("O", "N", "S"):
                        contact_pairs.append({
                            "pair": (idx_a, idx_b),
                            "min_r": 2.8,
                            "max_r": 3.6,
                            "k": 5.0,
                            "label": "halogen"
                        })

                    # Polar contacts: O/N/S to O/N/S
                    elif sym_a in ("O", "N", "S") and sym_b in ("O", "N", "S"):
                        contact_pairs.append({
                            "pair": (idx_a, idx_b),
                            "min_r": 2.5,
                            "max_r": 3.5,
                            "k": 5.0,
                            "label": "polar"
                        })

    return contact_pairs


from collections import defaultdict

def relax_complex(
    symbols: list[str],
    coords: np.ndarray,
    fragments: list[list[int]],
    weights: dict | None = None,
    charges: np.ndarray | None = None,
    sigma: float = 0.05,
    maxiter: int = 500,
    clash_mode: str = "compact",
    k_anchor: float = 0.1,
    topology_scale: float = 1.25,
    mode: str = "fast",
    t_start: float | None = None,
    prep_time_budget: float | None = None,
    engine: str = "scalar",
    profile: str = "v1",
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Relax a molecular complex by first optimizing rigid fragment placement

    followed by a light all-atom relaxation with anchor coordinates.
    """
    if t_start is None:
        t_start = time.perf_counter()
    if prep_time_budget is None:
        # Default safety budget
        prep_time_budget = 0.20

    coords = np.asarray(coords, dtype=np.float64).copy()
    raw_coords = coords.copy()

    # Build topology once from raw coordinates
    topo = Topology(symbols, coords, scale=topology_scale, sigma=sigma)
    inter_pairs = get_inter_pairs(topo.nonbonded_pairs, fragments)
    contact_pairs = detect_contact_pairs(symbols, topo, fragments)
    contact_indices = {tuple(sorted(cp["pair"])) for cp in contact_pairs if "pair" in cp}

    # We optimize fragments 1, 2, ... relative to fragment 0 (host).
    # Each guest fragment p has 6 variables: tx, ty, tz, alpha, beta, gamma.
    n_guests = len(fragments) - 1
    if n_guests < 1:
        # Fallback to standard full coordinate minimization if only 1 fragment
        return coords, coords

    # Store guest centers of mass in raw coords
    guest_coms = {}
    for p in range(1, len(fragments)):
        guest_idx = fragments[p]
        guest_coms[p] = np.mean(raw_coords[guest_idx], axis=0)

    # Initial guess: 0 translation and rotation for all guests
    x0 = np.zeros(6 * n_guests)

    def reconstruct_coords(params):
        curr_coords = raw_coords.copy()
        for p in range(1, len(fragments)):
            offset = 6 * (p - 1)
            tx, ty, tz, alpha, beta, gamma = params[offset : offset + 6]
            guest_idx = fragments[p]
            com = guest_coms[p]

            # Center, rotate, translate, and shift back
            R = rotation_matrix(alpha, beta, gamma)
            guest_pts = raw_coords[guest_idx]
            rotated = (guest_pts - com) @ R.T
            curr_coords[guest_idx] = rotated + com + np.array([tx, ty, tz])
        return curr_coords

    def objective(params):
        curr_coords = reconstruct_coords(params)

        # 1. Non-bonded inter-fragment terms
        e_clash = clash_energy(symbols, curr_coords, inter_pairs, clash_mode=clash_mode)
        e_disp = dispersion_energy(symbols, curr_coords, inter_pairs)
        e_coul = coulomb_energy(symbols, curr_coords, inter_pairs, charges=charges)

        # Apply term weights
        w = weights or {}
        e_nb = (
            w.get("clash", 1.0) * e_clash
            + w.get("disp", 0.1) * e_disp
            + w.get("coul", 0.0) * e_coul
        )

        # 2. Contact window terms
        e_contact = 0.0
        for cp in contact_pairs:
            if "centroid_a" in cp and "centroid_b" in cp:
                com_a = np.mean(curr_coords[cp["centroid_a"]], axis=0)
                com_b = np.mean(curr_coords[cp["centroid_b"]], axis=0)
                r = float(np.linalg.norm(com_a - com_b))
            elif "centroid_a" in cp and "atom_b" in cp:
                com_a = np.mean(curr_coords[cp["centroid_a"]], axis=0)
                pt_b = curr_coords[cp["atom_b"]]
                r = float(np.linalg.norm(com_a - pt_b))
            else:
                idx_i, idx_j = cp["pair"]
                r = distance(curr_coords, idx_i, idx_j)

            min_r = cp["min_r"]
            max_r = cp["max_r"]
            k = cp["k"]
            if r < min_r:
                e_contact += k * ((r - min_r) ** 2)
            elif r > max_r:
                e_contact += k * ((r - max_r) ** 2)

        # 2b. Short-range penalty if minimum interfragment distance < 1.8 A (excluding contact pairs)
        e_short_pen = 0.0
        k_pen = 100.0
        for idx_a in range(len(fragments)):
            for idx_b in range(idx_a + 1, len(fragments)):
                frag_a = fragments[idx_a]
                frag_b = fragments[idx_b]
                for i in frag_a:
                    for j in frag_b:
                        if tuple(sorted((i, j))) in contact_indices:
                            continue
                        d = distance(curr_coords, i, j)
                        if d < 1.8:
                            e_short_pen += k_pen * ((d - 1.8) ** 2)

        # 3. Anchor term to prevent too much drift
        e_anchor = 0.0
        for p in range(1, len(fragments)):
            offset = 6 * (p - 1)
            tx, ty, tz = params[offset : offset + 3]
            e_anchor += k_anchor * (tx**2 + ty**2 + tz**2)

        return e_nb + e_contact + e_short_pen + e_anchor

    class TimeoutException(Exception):
        pass

    def rigid_callback(xk) -> None:
        if time.perf_counter() - t_start > prep_time_budget:
            raise TimeoutException()

    # --- Phase 1: Rigid Guest Placement ---
    phase1_maxiter = 50 if mode == "fast" else maxiter
    try:
        res = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            callback=rigid_callback,
            options={"maxiter": phase1_maxiter}
        )
        rigid_coords = reconstruct_coords(res.x)
    except TimeoutException:
        return coords, coords

    # --- Phase 2: Light All-Atom Cleanup ---
    if time.perf_counter() - t_start > prep_time_budget:
        return rigid_coords, rigid_coords

    phase2_maxiter = 15 if mode == "fast" else 50

    from geoinit.energy.functional import GeoInitFunctional

    functional = GeoInitFunctional(
        symbols,
        rigid_coords,
        weights=weights,
        charges=charges,
        sigma=sigma,
        topology_scale=topology_scale,
        clash_mode=clash_mode,
        anchor_coords=rigid_coords,
        k_anchor=0.5,
        use_sparse=(mode == "fast"),
        engine=engine,
        profile=profile,
        backend=backend,
    )
    # Override with raw topology to prevent incorrect bonds due to compression
    functional.topology = topo

    def cleanup_callback(xk) -> None:
        if time.perf_counter() - t_start > prep_time_budget:
            raise TimeoutException()

    all_atom_x0 = rigid_coords.ravel().copy()
    try:
        cleanup_res = minimize(
            fun=functional.energy_flat,
            x0=all_atom_x0,
            jac=functional.gradient_flat,
            method="L-BFGS-B",
            callback=cleanup_callback,
            options={"maxiter": phase2_maxiter, "disp": False}
        )
        final_coords = cleanup_res.x.reshape(len(symbols), 3)
    except TimeoutException:
        final_coords = rigid_coords

    return final_coords, rigid_coords
