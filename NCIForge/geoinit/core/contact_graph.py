"""Contact graph utilities for GeoInit complexes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geoinit.core.classes import ChemicalClasses, detect_chemical_classes
from geoinit.core.geometry import distance
from geoinit.core.topology import Topology


@dataclass(frozen=True)
class ContactEdge:
    endpoint_a: tuple[int, ...]
    endpoint_b: tuple[int, ...]
    contact_type: str
    distance: float
    min_r: float
    max_r: float
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)

    def key(self) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
        a = tuple(sorted(self.endpoint_a))
        b = tuple(sorted(self.endpoint_b))
        if a > b:
            a, b = b, a
        return (self.contact_type, a, b)


@dataclass
class ContactGraph:
    edges: list[ContactEdge] = field(default_factory=list)

    def keys(self) -> set[tuple[str, tuple[int, ...], tuple[int, ...]]]:
        return {edge.key() for edge in self.edges}

    def by_type(self) -> dict[str, list[ContactEdge]]:
        out: dict[str, list[ContactEdge]] = {}
        for edge in self.edges:
            out.setdefault(edge.contact_type, []).append(edge)
        return out


def _centroid(coords: np.ndarray, atoms: tuple[int, ...]) -> np.ndarray:
    return np.mean(coords[list(atoms)], axis=0)


def _endpoint_distance(coords: np.ndarray, a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if len(a) == 1 and len(b) == 1:
        return distance(coords, a[0], b[0])
    return float(np.linalg.norm(_centroid(coords, a) - _centroid(coords, b)))


def _atom_fragment_map(fragments: list[list[int]]) -> dict[int, int]:
    atom_frag: dict[int, int] = {}
    for frag_idx, fragment in enumerate(fragments):
        for atom in fragment:
            atom_frag[atom] = frag_idx
    return atom_frag


def build_contact_graph(
    symbols: list[str],
    coords: np.ndarray,
    fragments: list[list[int]] | None = None,
    classes: ChemicalClasses | None = None,
    topology: Topology | None = None,
) -> ContactGraph:
    """Build an intermolecular graph of likely noncovalent contacts."""
    coords = np.asarray(coords, dtype=np.float64)
    topology = topology or Topology(symbols, coords)
    classes = classes or detect_chemical_classes(symbols, topology)
    fragments = fragments or classes.fragments
    atom_frag = _atom_fragment_map(fragments)

    edges: list[ContactEdge] = []

    def add_edge(a, b, contact_type: str, min_r: float, max_r: float, weight: float) -> None:
        if not a or not b:
            return
        a_tuple = tuple(sorted(a))
        b_tuple = tuple(sorted(b))
        if atom_frag.get(a_tuple[0]) == atom_frag.get(b_tuple[0]):
            return
        d = _endpoint_distance(coords, a_tuple, b_tuple)
        if d <= max_r + 0.75:
            edges.append(ContactEdge(a_tuple, b_tuple, contact_type, d, min_r, max_r, weight))

    donor_set = set(classes.donors)
    acceptor_set = set(classes.acceptors)
    polar_set = set(classes.polar_atoms)
    halogen_set = set(classes.halogen_atoms)

    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            if atom_frag.get(i) == atom_frag.get(j):
                continue
            if i in donor_set and j in acceptor_set:
                add_edge((i,), (j,), "H_BOND", 1.7, 2.3, 10.0)
            elif j in donor_set and i in acceptor_set:
                add_edge((i,), (j,), "H_BOND", 1.7, 2.3, 10.0)
            elif (i in halogen_set and j in acceptor_set) or (j in halogen_set and i in acceptor_set):
                add_edge((i,), (j,), "HALOGEN", 2.8, 3.6, 5.0)
            elif i in polar_set and j in polar_set:
                add_edge((i,), (j,), "POLAR", 2.5, 3.5, 5.0)

    for idx_a, pi_a in enumerate(classes.pi_systems):
        for pi_b in classes.pi_systems[idx_a + 1 :]:
            add_edge(pi_a.atoms, pi_b.atoms, "PI_PI", 3.2, 4.0, 5.0)
        for atom in sorted(polar_set | donor_set):
            add_edge(pi_a.atoms, (atom,), "PI_POLAR", 3.0, 3.8, 5.0)

    # Deduplicate while preserving first measured distance.
    deduped: dict[tuple[str, tuple[int, ...], tuple[int, ...]], ContactEdge] = {}
    for edge in edges:
        deduped.setdefault(edge.key(), edge)
    return ContactGraph(list(deduped.values()))


def contact_graph_similarity(raw_graph: ContactGraph, candidate_graph: ContactGraph) -> float:
    """Jaccard similarity between two contact edge-key sets."""
    raw_keys = raw_graph.keys()
    candidate_keys = candidate_graph.keys()
    if not raw_keys and not candidate_keys:
        return 1.0
    union = raw_keys | candidate_keys
    if not union:
        return 1.0
    return len(raw_keys & candidate_keys) / len(union)


def contact_window_score(raw_graph: ContactGraph, coords: np.ndarray) -> float:
    """Score how far raw contact edges are outside their target windows."""
    coords = np.asarray(coords, dtype=np.float64)
    score = 0.0
    for edge in raw_graph.edges:
        d = _endpoint_distance(coords, edge.endpoint_a, edge.endpoint_b)
        if edge.min_r <= d <= edge.max_r:
            continue
        if d < edge.min_r:
            score += edge.weight * (edge.min_r - d) ** 2
        else:
            score += edge.weight * (d - edge.max_r) ** 2
    return float(score)


__all__ = [
    "ContactEdge",
    "ContactGraph",
    "build_contact_graph",
    "contact_graph_similarity",
    "contact_window_score",
]
