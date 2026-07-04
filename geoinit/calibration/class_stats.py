"""Class descriptor collection and robust statistics for geoinit."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from geoinit.core.classes import ChemicalClasses, detect_chemical_classes
from geoinit.core.geometry import angle as compute_angle
from geoinit.core.geometry import distance
from geoinit.core.topology import Topology


@dataclass
class ClassStats:
    class_label: str
    count: int
    median: float
    mad: float
    p05: float
    p95: float

    def contains(self, value: float, margin: float = 0.0) -> bool:
        return (self.p05 - margin) <= value <= (self.p95 + margin)


def collect_class_descriptors(
    symbols: list[str],
    coords: np.ndarray,
    topology: Topology | None = None,
    classes: ChemicalClasses | None = None,
) -> dict[str, list[float]]:
    """Collect simple scalar descriptors keyed by chemical class label."""
    coords = np.asarray(coords, dtype=np.float64)
    topology = topology or Topology(symbols, coords)
    classes = classes or detect_chemical_classes(symbols, topology)
    descriptors: dict[str, list[float]] = {}

    def add(label: str, value: float) -> None:
        descriptors.setdefault(label, []).append(float(value))

    for bond in topology.bonds:
        add(f"bond:{getattr(bond, 'label', 'single')}", distance(coords, bond.i, bond.j))

    for i, j, k in topology.angles:
        add(f"angle:{symbols[j]}:coord{topology.coordination[j]}", compute_angle(coords, i, j, k))

    for feature in classes.rigid_subgraphs:
        atoms = list(feature.atoms)
        for idx_a in range(len(atoms)):
            for idx_b in range(idx_a + 1, len(atoms)):
                add("rigid_pair:distance", distance(coords, atoms[idx_a], atoms[idx_b]))

    for feature in classes.carbonyl_groups:
        carbon = feature.metadata.get("carbon")
        oxygen = feature.metadata.get("oxygen")
        if carbon is not None and oxygen is not None:
            add("class:carbonyl:C=O", distance(coords, carbon, oxygen))

    for feature in classes.amide_groups:
        carbon = feature.metadata.get("carbon")
        nitrogen = feature.metadata.get("nitrogen")
        if carbon is not None and nitrogen is not None:
            add("class:amide:C-N", distance(coords, carbon, nitrogen))

    return descriptors


def fit_class_stats(descriptor_sets: list[dict[str, list[float]]]) -> dict[str, ClassStats]:
    """Fit robust percentile stats from descriptor dictionaries."""
    merged: dict[str, list[float]] = {}
    for descriptors in descriptor_sets:
        for label, values in descriptors.items():
            merged.setdefault(label, []).extend(values)

    stats: dict[str, ClassStats] = {}
    for label, values in merged.items():
        arr = np.asarray(values, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        stats[label] = ClassStats(
            class_label=label,
            count=int(arr.size),
            median=median,
            mad=mad,
            p05=float(np.percentile(arr, 5)),
            p95=float(np.percentile(arr, 95)),
        )
    return stats


def save_class_stats(stats: dict[str, ClassStats], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {label: asdict(stat) for label, stat in stats.items()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_class_stats(path: str | Path) -> dict[str, ClassStats]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {label: ClassStats(**data) for label, data in payload.items()}


__all__ = [
    "ClassStats",
    "collect_class_descriptors",
    "fit_class_stats",
    "load_class_stats",
    "save_class_stats",
]
