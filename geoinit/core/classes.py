"""Rule-based chemical class perception for geoinit.

This module centralizes the lightweight chemistry perception that was previously
spread across guards, rigid-fragment handling, and complex contact detection.
It intentionally stays deterministic and topology-based; learned calibration is
a later phase.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geoinit.core.topology import Topology


POLAR_ELEMENTS = {"O", "N", "S", "F"}
HALOGEN_ELEMENTS = {"Cl", "Br", "I"}


@dataclass(frozen=True)
class ChemicalFeature:
    """A perceived chemical class instance."""

    kind: str
    atoms: tuple[int, ...]
    label: str
    metadata: dict = field(default_factory=dict)

    def contains(self, atom: int) -> bool:
        return atom in self.atoms


@dataclass
class ChemicalClasses:
    """Container for rule-based chemical perception results."""

    multiple_bond_groups: list[ChemicalFeature] = field(default_factory=list)
    linear_fragments: list[ChemicalFeature] = field(default_factory=list)
    aromatic_rings: list[ChemicalFeature] = field(default_factory=list)
    carbonyl_groups: list[ChemicalFeature] = field(default_factory=list)
    amide_groups: list[ChemicalFeature] = field(default_factory=list)
    rigid_subgraphs: list[ChemicalFeature] = field(default_factory=list)
    donors: list[int] = field(default_factory=list)
    acceptors: list[int] = field(default_factory=list)
    polar_atoms: list[int] = field(default_factory=list)
    halogen_atoms: list[int] = field(default_factory=list)
    pi_systems: list[ChemicalFeature] = field(default_factory=list)
    fragments: list[list[int]] = field(default_factory=list)

    def all_features(self) -> list[ChemicalFeature]:
        return (
            self.multiple_bond_groups
            + self.linear_fragments
            + self.aromatic_rings
            + self.carbonyl_groups
            + self.amide_groups
            + self.rigid_subgraphs
            + self.pi_systems
        )

    def features_by_kind(self, kind: str) -> list[ChemicalFeature]:
        return [feature for feature in self.all_features() if feature.kind == kind]

    def has_kind(self, kind: str) -> bool:
        return bool(self.features_by_kind(kind))


def build_adjacency(topology: "Topology", n_atoms: int) -> dict[int, list[int]]:
    """Return sorted covalent adjacency lists from a topology."""
    adjacency: dict[int, list[int]] = {idx: [] for idx in range(n_atoms)}
    for bond in topology.bonds:
        adjacency[bond.i].append(bond.j)
        adjacency[bond.j].append(bond.i)
    return {idx: sorted(neighbors) for idx, neighbors in adjacency.items()}


def find_fragments(topology: "Topology", n_atoms: int) -> list[list[int]]:
    """Find connected components in the covalent graph."""
    adjacency = build_adjacency(topology, n_atoms)
    visited: set[int] = set()
    fragments: list[list[int]] = []

    for start in range(n_atoms):
        if start in visited:
            continue
        queue = [start]
        visited.add(start)
        fragment: list[int] = []
        while queue:
            atom = queue.pop(0)
            fragment.append(atom)
            for neighbor in adjacency[atom]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        fragments.append(sorted(fragment))
    return fragments


def _feature(kind: str, atoms: set[int] | list[int] | tuple[int, ...], label: str, **metadata) -> ChemicalFeature:
    return ChemicalFeature(
        kind=kind,
        atoms=tuple(sorted(atoms)),
        label=label,
        metadata=metadata,
    )


def _dedupe_features(features: list[ChemicalFeature]) -> list[ChemicalFeature]:
    seen: set[tuple[str, tuple[int, ...], str]] = set()
    out: list[ChemicalFeature] = []
    for feature in features:
        key = (feature.kind, feature.atoms, feature.label)
        if key in seen:
            continue
        seen.add(key)
        out.append(feature)
    return out


def _ring_like_subgraphs(symbols: list[str], topology: "Topology", subgraphs: list[list[int]]) -> list[ChemicalFeature]:
    rings: list[ChemicalFeature] = []
    for subgraph in subgraphs:
        if len(subgraph) not in (5, 6):
            continue
        bonds_in_graph = sum(
            1
            for bond in topology.bonds
            if bond.i in subgraph and bond.j in subgraph
        )
        if bonds_in_graph < len(subgraph):
            continue
        labels = sorted({getattr(bond, "label", "") for bond in topology.bonds if bond.i in subgraph and bond.j in subgraph})
        ring_symbols = tuple(symbols[idx] for idx in subgraph)
        rings.append(
            _feature(
                "aromatic_ring",
                subgraph,
                "aromatic_ring",
                size=len(subgraph),
                symbols=ring_symbols,
                bond_labels=labels,
            )
        )
    return rings


def detect_chemical_classes(
    symbols: list[str],
    topology: "Topology",
) -> ChemicalClasses:
    """Detect reusable chemistry classes from symbols and topology."""
    from geoinit.core.bond_rules import detect_rigid_subgraphs

    n_atoms = len(symbols)
    adjacency = build_adjacency(topology, n_atoms)
    degrees = {idx: len(adjacency[idx]) for idx in range(n_atoms)}
    bonds_by_label: dict[str, list] = defaultdict(list)
    for bond in topology.bonds:
        bonds_by_label[getattr(bond, "label", "single")].append(bond)

    multiple_bond_groups: list[ChemicalFeature] = []
    linear_fragments: list[ChemicalFeature] = []
    carbonyl_groups: list[ChemicalFeature] = []
    amide_groups: list[ChemicalFeature] = []

    for bond in topology.bonds:
        label = getattr(bond, "label", "single")
        order = float(getattr(bond, "order", 1.0))
        if order > 1.0:
            multiple_bond_groups.append(
                _feature(
                    "multiple_bond",
                    {bond.i, bond.j},
                    label,
                    order=order,
                    bond=(bond.i, bond.j),
                )
            )

        if label in {"CO2 C=O", "nitrile C#N", "alkyne C#C"}:
            group = {bond.i, bond.j}
            for atom in (bond.i, bond.j):
                if symbols[atom] == "C":
                    group.update(adjacency[atom])
            linear_fragments.append(
                _feature(
                    "linear_fragment",
                    group,
                    label,
                    bond=(bond.i, bond.j),
                )
            )

        if label in {"carbonyl C=O", "CO2 C=O"}:
            c_idx = bond.i if symbols[bond.i] == "C" else bond.j
            o_idx = bond.j if c_idx == bond.i else bond.i
            group = {c_idx, o_idx}
            for neighbor in adjacency[c_idx]:
                if symbols[neighbor] != "H":
                    group.add(neighbor)
            carbonyl_groups.append(
                _feature(
                    "carbonyl",
                    group,
                    label,
                    carbon=c_idx,
                    oxygen=o_idx,
                    bond=(bond.i, bond.j),
                )
            )

        if label == "amide C-N":
            c_idx = bond.i if symbols[bond.i] == "C" else bond.j
            n_idx = bond.j if c_idx == bond.i else bond.i
            group = {c_idx, n_idx}
            oxygen = None
            for neighbor in adjacency[c_idx]:
                if symbols[neighbor] == "O" and degrees[neighbor] == 1:
                    oxygen = neighbor
                if symbols[neighbor] != "H":
                    group.add(neighbor)
            for neighbor in adjacency[n_idx]:
                if symbols[neighbor] != "H":
                    group.add(neighbor)
            amide_groups.append(
                _feature(
                    "amide",
                    group,
                    label,
                    carbon=c_idx,
                    nitrogen=n_idx,
                    oxygen=oxygen,
                    bond=(bond.i, bond.j),
                )
            )

    rigid_raw = detect_rigid_subgraphs(symbols, topology)
    rigid_subgraphs = [
        _feature("rigid_subgraph", subgraph, "rigid_subgraph", size=len(subgraph))
        for subgraph in rigid_raw
    ]
    aromatic_rings = _ring_like_subgraphs(symbols, topology, rigid_raw)
    pi_systems = [
        _feature("pi_system", ring.atoms, "aromatic_pi_system", size=ring.metadata["size"])
        for ring in aromatic_rings
    ]

    donors: list[int] = []
    for idx, symbol in enumerate(symbols):
        if symbol != "H":
            continue
        if any(symbols[neighbor] in POLAR_ELEMENTS for neighbor in adjacency[idx]):
            donors.append(idx)

    acceptors = [
        idx
        for idx, symbol in enumerate(symbols)
        if symbol in POLAR_ELEMENTS and symbol != "H"
    ]
    polar_atoms = [
        idx
        for idx, symbol in enumerate(symbols)
        if symbol in POLAR_ELEMENTS
    ]
    halogen_atoms = [
        idx
        for idx, symbol in enumerate(symbols)
        if symbol in HALOGEN_ELEMENTS
    ]

    return ChemicalClasses(
        multiple_bond_groups=_dedupe_features(multiple_bond_groups),
        linear_fragments=_dedupe_features(linear_fragments),
        aromatic_rings=_dedupe_features(aromatic_rings),
        carbonyl_groups=_dedupe_features(carbonyl_groups),
        amide_groups=_dedupe_features(amide_groups),
        rigid_subgraphs=_dedupe_features(rigid_subgraphs),
        donors=sorted(set(donors)),
        acceptors=sorted(set(acceptors)),
        polar_atoms=sorted(set(polar_atoms)),
        halogen_atoms=sorted(set(halogen_atoms)),
        pi_systems=_dedupe_features(pi_systems),
        fragments=find_fragments(topology, n_atoms),
    )


__all__ = [
    "ChemicalClasses",
    "ChemicalFeature",
    "HALOGEN_ELEMENTS",
    "POLAR_ELEMENTS",
    "build_adjacency",
    "detect_chemical_classes",
    "find_fragments",
]
