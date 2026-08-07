"""Bond rules and chemistry assignments for GeoInit V0.2.

This module provides rule-based assignment of bond orders, reference bond lengths,
and ideal valence angles based on local coordination geometry and element symbols.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from geoinit.core.params import get_covalent_radius


@dataclass
class Bond:
    """Dataclass representing a covalent bond with chemical metadata.

    Behaves like a 2-tuple of (i, j) for backward compatibility.
    """
    i: int
    j: int
    order: float
    r0: float
    k: float
    label: str

    def __iter__(self):
        yield self.i
        yield self.j

    def __getitem__(self, index):
        if index == 0:
            return self.i
        elif index == 1:
            return self.j
        raise IndexError("Bond index out of range")

    def __len__(self) -> int:
        return 2

    def __repr__(self) -> str:
        return f"Bond({self.i}-{self.j}, order={self.order}, r0={self.r0:.3f}, label='{self.label}')"


# Standard reference bond lengths (Å) for single bonds
STANDARD_BONDS: dict[tuple[str, str], float] = {
    ("C", "H"): 1.09,
    ("H", "O"): 0.96,
    ("C", "C"): 1.54,
    ("C", "O"): 1.43,
    ("C", "N"): 1.47,
    ("H", "S"): 1.34,
    ("C", "S"): 1.82,
}


def find_cycles(neighbors: dict[int, list[int]], n_atoms: int) -> list[list[int]]:
    """Find simple cycles of length 5 and 6 in the connectivity graph."""
    cycles = []

    def dfs(node, start, path, visited):
        if len(path) > 6:
            return
        for nbr in neighbors[node]:
            if nbr == start and len(path) >= 5:
                cycle_sorted = sorted(path)
                if cycle_sorted not in [sorted(c) for c in cycles]:
                    cycles.append(list(path))
            elif nbr not in visited:
                visited.add(nbr)
                dfs(nbr, start, path + [nbr], visited)
                visited.remove(nbr)

    for i in range(n_atoms):
        dfs(i, i, [i], {i})
    return cycles


def assign_bond_orders(
    symbols: list[str],
    coords: np.ndarray,
    bonds: list[tuple[int, int]],
) -> list[Bond]:
    """Assign bond orders and labels based on chemistry rules."""
    neighbors = defaultdict(list)
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)

    degrees = {i: len(neighbors[i]) for i in range(len(symbols))}
    cycles = find_cycles(neighbors, len(symbols))

    # Aromatic ring detection
    aromatic_bonds = set()
    for cycle in cycles:
        cycle_symbols = [symbols[idx] for idx in cycle]
        if len(cycle) == 6 and all(s in ("C", "N") for s in cycle_symbols):
            for i in range(6):
                u, v = cycle[i], cycle[(i+1)%6]
                pair = (min(u, v), max(u, v))
                if symbols[u] == "C" and symbols[v] == "C":
                    aromatic_bonds.add((pair[0], pair[1], "aromatic C-C"))
                elif symbols[u] == "N" and symbols[v] == "N":
                    aromatic_bonds.add((pair[0], pair[1], "aromatic N-N"))
                else:
                    aromatic_bonds.add((pair[0], pair[1], "aromatic C-N"))
        elif len(cycle) == 5 and cycle_symbols.count("C") >= 3 and (cycle_symbols.count("S") == 1 or cycle_symbols.count("O") == 1 or cycle_symbols.count("N") == 1):
            for i in range(5):
                u, v = cycle[i], cycle[(i+1)%5]
                pair = (min(u, v), max(u, v))
                sym_u, sym_v = symbols[u], symbols[v]
                if sym_u == "C" and sym_v == "C":
                    aromatic_bonds.add((pair[0], pair[1], "aromatic C-C"))
                elif sym_u == "S" or sym_v == "S":
                    aromatic_bonds.add((pair[0], pair[1], "aromatic C-S"))
                elif sym_u == "O" or sym_v == "O":
                    aromatic_bonds.add((pair[0], pair[1], "aromatic C-O"))
                elif sym_u == "N" or sym_v == "N":
                    aromatic_bonds.add((pair[0], pair[1], "aromatic C-N"))

    aromatic_lookup = { (u, v): label for u, v, label in aromatic_bonds }

    bond_objs = []
    for u, v in bonds:
        i, j = min(u, v), max(u, v)
        sym_i, sym_j = symbols[i], symbols[j]
        order = 1.0
        label = "single"

        # 1. CO2 rule: linear O=C=O
        is_co2 = False
        for c_idx in (i, j):
            if symbols[c_idx] == "C" and degrees[c_idx] == 2:
                nbrs = neighbors[c_idx]
                if len(nbrs) == 2 and symbols[nbrs[0]] == "O" and symbols[nbrs[1]] == "O":
                    is_co2 = True

        if is_co2 and ((sym_i == "C" and sym_j == "O") or (sym_i == "O" and sym_j == "C")):
            order = 2.0
            label = "CO2 C=O"

        # 2. Nitrile rule: C#N
        elif (sym_i == "C" and sym_j == "N") or (sym_i == "N" and sym_j == "C"):
            c_idx = i if sym_i == "C" else j
            n_idx = j if sym_i == "C" else i
            if degrees[c_idx] == 2 and degrees[n_idx] == 1:
                order = 3.0
                label = "nitrile C#N"
            else:
                has_carbonyl = False
                for nbr in neighbors[c_idx]:
                    if symbols[nbr] == "O" and degrees[nbr] == 1:
                        has_carbonyl = True
                if has_carbonyl:
                    order = 1.5
                    label = "amide C-N"

        # 3. Alkyne rule: C#C
        elif sym_i == "C" and sym_j == "C" and degrees[i] == 2 and degrees[j] == 2:
            order = 3.0
            label = "alkyne C#C"

        # 4. Carbonyl rule: C=O double bond
        elif (sym_i == "C" and sym_j == "O") or (sym_i == "O" and sym_j == "C"):
            c_idx = i if sym_i == "C" else j
            o_idx = j if sym_i == "C" else i
            if degrees[o_idx] == 1 and degrees[c_idx] >= 2:
                order = 2.0
                label = "carbonyl C=O"

        # 5. Aromatic ring override
        elif (i, j) in aromatic_lookup:
            order = 1.5
            label = aromatic_lookup[(i, j)]

        bond_objs.append(Bond(i=i, j=j, order=order, r0=0.0, k=1.0, label=label))

    return bond_objs


def assign_reference_lengths(
    symbols: list[str],
    bonds: list[Bond],
    sigma: float = 0.05,
) -> list[Bond]:
    """Calculate and assign the ideal reference length r0 and force constant k."""
    for b in bonds:
        sym_i, sym_j = symbols[b.i], symbols[b.j]

        # General bond order scaling model
        ri_cov = get_covalent_radius(sym_i)
        rj_cov = get_covalent_radius(sym_j)
        sum_cov = ri_cov + rj_cov

        if abs(b.order - 1.0) < 0.01:
            s_bo = 1.00
        elif abs(b.order - 1.5) < 0.01:
            s_bo = 0.92
        elif abs(b.order - 2.0) < 0.01:
            s_bo = 0.87
        elif abs(b.order - 3.0) < 0.01:
            s_bo = 0.80
        else:
            s_bo = 1.00

        r0 = s_bo * sum_cov

        # High-precision overrides
        if b.label == "CO2 C=O":
            r0 = 1.16
        elif b.label == "carbonyl C=O":
            r0 = 1.21
        elif b.label == "amide C-N":
            r0 = 1.35
        elif b.label == "aromatic C-C":
            r0 = 1.39
        elif b.label == "aromatic C-S":
            r0 = 1.72
        else:
            pair = tuple(sorted([sym_i, sym_j]))
            if pair in STANDARD_BONDS and abs(b.order - 1.0) < 0.01:
                r0 = STANDARD_BONDS[pair]

        b.r0 = r0
        b.k = 1.0 / (sigma * sigma)

    return bonds


def detect_rigid_subgraphs(symbols: list[str], topology: any) -> list[list[int]]:
    """Detect chemically stiff subgraphs where internal geometry should be preserved.

    Returns a list of lists of atom indices representing these subgraphs.
    """
    neighbors = defaultdict(list)
    for b in topology.bonds:
        neighbors[b.i].append(b.j)
        neighbors[b.j].append(b.i)

    n_atoms = len(symbols)
    degrees = {i: len(neighbors[i]) for i in range(n_atoms)}

    subgraphs: list[set[int]] = []

    # 1. Find aromatic cycles
    cycles = find_cycles(neighbors, n_atoms)
    for cycle in cycles:
        cycle_symbols = [symbols[idx] for idx in cycle]
        is_aromatic = False
        if len(cycle) == 6 and all(s in ("C", "N") for s in cycle_symbols):
            is_aromatic = True
        elif len(cycle) == 5 and cycle_symbols.count("C") >= 3 and (cycle_symbols.count("S") == 1 or cycle_symbols.count("O") == 1 or cycle_symbols.count("N") == 1):
            is_aromatic = True

        if is_aromatic:
            subgraphs.append(set(cycle))

    # Map bonds by label
    bonds_by_label = defaultdict(list)
    for b in topology.bonds:
        bonds_by_label[b.label].append(b)

    # 2. Carbonyl-centered groups (C=O and immediate heavy neighbors)
    for b in bonds_by_label["carbonyl C=O"]:
        c_idx = b.i if symbols[b.i] == "C" else b.j
        o_idx = b.j if symbols[b.i] == "C" else b.i
        group = {c_idx, o_idx}
        for nbr in neighbors[c_idx]:
            if symbols[nbr] != "H":
                group.add(nbr)
        subgraphs.append(group)

    # 3. Amide groups (O=C-N and immediate heavy neighbors of C and N)
    for b in bonds_by_label["amide C-N"]:
        c_idx = b.i if symbols[b.i] == "C" else b.j
        n_idx = b.j if symbols[b.i] == "C" else b.i
        o_idx = None
        for nbr in neighbors[c_idx]:
            if symbols[nbr] == "O" and degrees[nbr] == 1:
                o_idx = nbr
                break
        group = {c_idx, n_idx}
        if o_idx is not None:
            group.add(o_idx)
        for nbr in neighbors[c_idx]:
            if symbols[nbr] != "H":
                group.add(nbr)
        for nbr in neighbors[n_idx]:
            if symbols[nbr] != "H":
                group.add(nbr)
        subgraphs.append(group)

    # 4. Linear multiple-bond chains
    # CO2 (O=C=O)
    co2_groups = defaultdict(set)
    for b in bonds_by_label["CO2 C=O"]:
        c_idx = b.i if symbols[b.i] == "C" else b.j
        o_idx = b.j if symbols[b.i] == "C" else b.i
        co2_groups[c_idx].add(o_idx)
        co2_groups[c_idx].add(c_idx)
    for c_idx, group in co2_groups.items():
        if len(group) == 3:
            subgraphs.append(group)

    # Nitrile (C-C#N)
    for b in bonds_by_label["nitrile C#N"]:
        c_idx = b.i if symbols[b.i] == "C" else b.j
        n_idx = b.j if symbols[b.i] == "C" else b.i
        group = {c_idx, n_idx}
        for nbr in neighbors[c_idx]:
            if nbr != n_idx:
                group.add(nbr)
        subgraphs.append(group)

    # Alkyne (C-C#C-C)
    for b in bonds_by_label["alkyne C#C"]:
        c1, c2 = b.i, b.j
        group = {c1, c2}
        for nbr in neighbors[c1]:
            if nbr != c2:
                group.add(nbr)
        for nbr in neighbors[c2]:
            if nbr != c1:
                group.add(nbr)
        subgraphs.append(group)

    # Remove duplicates and return sorted list of sorted subgraphs
    unique_subgraphs = []
    seen = set()
    for g in subgraphs:
        g_sorted = tuple(sorted(g))
        if g_sorted not in seen:
            seen.add(g_sorted)
            unique_subgraphs.append(list(g_sorted))

    return unique_subgraphs


def assign_angle_targets(
    symbols: list[str],
    topology: Topology,
) -> dict[tuple[int, int, int], float]:
    """Assign ideal target angles in radians for all topology angles."""
    from geoinit.core.params import get_ideal_angle
    targets = {}

    degrees = topology.coordination

    for i, j, k in topology.angles:
        is_co2_central = False
        if symbols[j] == "C" and degrees[j] == 2:
            nbrs = []
            for b in topology.bonds:
                if b.i == j:
                    nbrs.append(b.j)
                elif b.j == j:
                    nbrs.append(b.i)
            if len(nbrs) == 2 and symbols[nbrs[0]] == "O" and symbols[nbrs[1]] == "O":
                is_co2_central = True

        if is_co2_central:
            targets[(i, j, k)] = np.pi  # 180 degrees in radians
        else:
            targets[(i, j, k)] = get_ideal_angle(symbols[j], degrees[j])

    return targets
