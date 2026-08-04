"""Constraint primitives for closest-feasible geometry projection.

The classes in this module represent chemistry checks as normalized residuals.
They are intentionally independent of the optimizer so candidate generation,
guards, calibration, and future projection code can all score the same geometry
in the same units.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING
from typing import Iterable

import numpy as np

from geoinit.core.geometry import angle as compute_angle
from geoinit.core.geometry import distance
from geoinit.core.params import get_vdw_radius

if TYPE_CHECKING:
    from geoinit.core.topology import Topology


def _as_coords(coords: np.ndarray) -> np.ndarray:
    return np.asarray(coords, dtype=np.float64)


@dataclass(frozen=True)
class Constraint:
    """Base normalized geometry constraint."""

    kind: str
    atoms: tuple[int, ...]
    target: float
    sigma: float
    priority: int = 1
    class_label: str = "generic"
    weight: float = 1.0

    def value(self, coords: np.ndarray) -> float:
        raise NotImplementedError

    def residual(self, coords: np.ndarray) -> float:
        if self.sigma <= 0.0:
            raise ValueError("Constraint sigma must be positive.")
        return (self.value(coords) - self.target) / self.sigma

    def score(self, coords: np.ndarray) -> float:
        residual = self.residual(coords)
        return float(self.weight * residual * residual)

    def summary(self, coords: np.ndarray) -> dict:
        value = self.value(coords)
        residual = self.residual(coords)
        return {
            "kind": self.kind,
            "atoms": self.atoms,
            "target": self.target,
            "sigma": self.sigma,
            "priority": self.priority,
            "class_label": self.class_label,
            "weight": self.weight,
            "value": value,
            "residual": residual,
            "score": float(self.weight * residual * residual),
        }


@dataclass(frozen=True)
class BondLengthConstraint(Constraint):
    """Normalized bond-length constraint."""

    def __init__(
        self,
        i: int,
        j: int,
        target: float,
        sigma: float = 0.05,
        priority: int = 1,
        class_label: str = "bond",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            kind="bond_length",
            atoms=(i, j),
            target=target,
            sigma=sigma,
            priority=priority,
            class_label=class_label,
            weight=weight,
        )

    def value(self, coords: np.ndarray) -> float:
        i, j = self.atoms
        return distance(_as_coords(coords), i, j)


@dataclass(frozen=True)
class AngleConstraint(Constraint):
    """Normalized valence-angle constraint in radians."""

    def __init__(
        self,
        i: int,
        j: int,
        k: int,
        target: float,
        sigma: float = float(np.deg2rad(5.0)),
        priority: int = 2,
        class_label: str = "angle",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            kind="angle",
            atoms=(i, j, k),
            target=target,
            sigma=sigma,
            priority=priority,
            class_label=class_label,
            weight=weight,
        )

    def value(self, coords: np.ndarray) -> float:
        i, j, k = self.atoms
        return compute_angle(_as_coords(coords), i, j, k)


@dataclass(frozen=True)
class RigidPairDistanceConstraint(Constraint):
    """Preserve a pairwise distance inside a rigid subgraph."""

    def __init__(
        self,
        i: int,
        j: int,
        target: float,
        sigma: float = 0.03,
        priority: int = 1,
        class_label: str = "rigid_pair",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            kind="rigid_pair_distance",
            atoms=(i, j),
            target=target,
            sigma=sigma,
            priority=priority,
            class_label=class_label,
            weight=weight,
        )

    def value(self, coords: np.ndarray) -> float:
        i, j = self.atoms
        return distance(_as_coords(coords), i, j)


@dataclass(frozen=True)
class PlaneConstraint(Constraint):
    """Constrain one atom to a plane defined by three reference atoms."""

    def __init__(
        self,
        atom: int,
        plane_atoms: tuple[int, int, int],
        target: float = 0.0,
        sigma: float = 0.05,
        priority: int = 1,
        class_label: str = "plane",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            kind="plane",
            atoms=(atom, *plane_atoms),
            target=target,
            sigma=sigma,
            priority=priority,
            class_label=class_label,
            weight=weight,
        )

    def value(self, coords: np.ndarray) -> float:
        coords = _as_coords(coords)
        atom, a, b, c = self.atoms
        p0 = coords[a]
        v1 = coords[b] - p0
        v2 = coords[c] - p0
        normal = np.cross(v1, v2)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-12:
            return 0.0
        normal = normal / norm
        return float(abs(np.dot(coords[atom] - p0, normal)))


class WindowConstraint(Constraint):
    """Base constraint for values that must remain inside a window."""

    min_value: float
    max_value: float

    def __init__(
        self,
        kind: str,
        atoms: tuple[int, ...],
        min_value: float,
        max_value: float,
        sigma: float,
        priority: int,
        class_label: str,
        weight: float = 1.0,
    ) -> None:
        midpoint = 0.5 * (min_value + max_value) if np.isfinite(max_value) else min_value
        object.__setattr__(self, "min_value", min_value)
        object.__setattr__(self, "max_value", max_value)
        Constraint.__init__(
            self,
            kind=kind,
            atoms=atoms,
            target=midpoint,
            sigma=sigma,
            priority=priority,
            class_label=class_label,
            weight=weight,
        )

    def residual(self, coords: np.ndarray) -> float:
        if self.sigma <= 0.0:
            raise ValueError("Constraint sigma must be positive.")
        value = self.value(coords)
        if self.min_value <= value <= self.max_value:
            return 0.0
        if value < self.min_value:
            return (value - self.min_value) / self.sigma
        return (value - self.max_value) / self.sigma

    def summary(self, coords: np.ndarray) -> dict:
        out = super().summary(coords)
        out["min_value"] = self.min_value
        out["max_value"] = self.max_value
        return out


class ContactWindowConstraint(WindowConstraint):
    """Keep an atom-atom contact inside a target distance window."""

    def __init__(
        self,
        i: int,
        j: int,
        min_value: float,
        max_value: float,
        sigma: float = 0.10,
        priority: int = 4,
        class_label: str = "contact",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            kind="contact_window",
            atoms=(i, j),
            min_value=min_value,
            max_value=max_value,
            sigma=sigma,
            priority=priority,
            class_label=class_label,
            weight=weight,
        )

    def value(self, coords: np.ndarray) -> float:
        i, j = self.atoms
        return distance(_as_coords(coords), i, j)


class ClashConstraint(WindowConstraint):
    """Penalize nonbonded pairs closer than a minimum allowed distance."""

    def __init__(
        self,
        i: int,
        j: int,
        min_value: float | None = None,
        sigma: float = 0.10,
        priority: int = 2,
        class_label: str = "clash",
        weight: float = 1.0,
        symbols: tuple[str, str] | None = None,
        vdw_scale: float = 0.75,
    ) -> None:
        if min_value is None:
            if symbols is None:
                raise ValueError("ClashConstraint needs min_value or symbols.")
            min_value = vdw_scale * (get_vdw_radius(symbols[0]) + get_vdw_radius(symbols[1]))
        super().__init__(
            kind="clash",
            atoms=(i, j),
            min_value=min_value,
            max_value=float("inf"),
            sigma=sigma,
            priority=priority,
            class_label=class_label,
            weight=weight,
        )

    def value(self, coords: np.ndarray) -> float:
        i, j = self.atoms
        return distance(_as_coords(coords), i, j)

    def residual(self, coords: np.ndarray) -> float:
        if self.sigma <= 0.0:
            raise ValueError("Constraint sigma must be positive.")
        value = self.value(coords)
        if value >= self.min_value:
            return 0.0
        return (self.min_value - value) / self.sigma


def constraint_score(
    constraints: Iterable[Constraint],
    coords: np.ndarray,
    max_priority: int | None = None,
) -> float:
    """Return the weighted sum of squared normalized residuals."""
    total = 0.0
    for constraint in constraints:
        if max_priority is not None and constraint.priority > max_priority:
            continue
        total += constraint.score(coords)
    return float(total)


def constraint_summary(
    constraints: Iterable[Constraint],
    coords: np.ndarray,
) -> dict:
    """Return aggregate and per-constraint scoring details."""
    rows = [constraint.summary(coords) for constraint in constraints]
    by_kind: dict[str, float] = {}
    by_priority: dict[int, float] = {}
    for row in rows:
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0.0) + row["score"]
        by_priority[row["priority"]] = by_priority.get(row["priority"], 0.0) + row["score"]

    max_abs_residual = max((abs(row["residual"]) for row in rows), default=0.0)
    total_score = sum(row["score"] for row in rows)
    return {
        "count": len(rows),
        "total_score": float(total_score),
        "mean_score": float(total_score / len(rows)) if rows else 0.0,
        "max_abs_residual": float(max_abs_residual),
        "by_kind": by_kind,
        "by_priority": by_priority,
        "constraints": rows,
    }


def hard_constraint_failed(
    summary: dict,
    max_priority: int = 1,
    max_abs_residual: float = 3.0,
) -> bool:
    """Return True when a high-priority constraint exceeds a residual limit."""
    for row in summary.get("constraints", []):
        if row["priority"] <= max_priority and abs(row["residual"]) > max_abs_residual:
            return True
    return False


def build_bond_constraints(
    symbols: list[str],
    topology: "Topology",
    sigma: float = 0.05,
    priority: int = 1,
    weight: float = 1.0,
) -> list[BondLengthConstraint]:
    """Build bond-length constraints from topology bond metadata."""
    constraints: list[BondLengthConstraint] = []
    for bond in topology.bonds:
        i, j = bond.i, bond.j
        target = getattr(bond, "r0", None)
        if target is None:
            from geoinit.core.params import get_covalent_radius

            target = get_covalent_radius(symbols[i]) + get_covalent_radius(symbols[j])
        constraints.append(
            BondLengthConstraint(
                i,
                j,
                target=float(target),
                sigma=sigma,
                priority=priority,
                class_label=getattr(bond, "label", "bond"),
                weight=weight,
            )
        )
    return constraints


def build_angle_constraints(
    symbols: list[str],
    topology: "Topology",
    sigma: float = float(np.deg2rad(5.0)),
    priority: int = 2,
    weight: float = 1.0,
) -> list[AngleConstraint]:
    """Build valence-angle constraints from topology angle targets."""
    constraints: list[AngleConstraint] = []
    for i, j, k in topology.angles:
        target = topology.angle_targets.get((i, j, k))
        if target is None:
            from geoinit.core.params import get_ideal_angle

            target = get_ideal_angle(symbols[j], topology.coordination[j])
        constraints.append(
            AngleConstraint(
                i,
                j,
                k,
                target=float(target),
                sigma=sigma,
                priority=priority,
                class_label=f"{symbols[j]}_coord{topology.coordination[j]}_angle",
                weight=weight,
            )
        )
    return constraints


def build_rigid_pair_constraints(
    symbols: list[str],
    topology: "Topology",
    reference_coords: np.ndarray | None = None,
    sigma: float = 0.03,
    priority: int = 1,
    weight: float = 1.0,
) -> list[RigidPairDistanceConstraint]:
    """Build pair-distance constraints for chemically rigid subgraphs."""
    from geoinit.core.bond_rules import detect_rigid_subgraphs

    if reference_coords is None:
        reference_coords = getattr(topology, "reference_coords", None)
    if reference_coords is None:
        raise ValueError("Rigid pair constraints require reference coordinates.")

    reference_coords = _as_coords(reference_coords)
    constraints: list[RigidPairDistanceConstraint] = []
    seen: set[tuple[int, int]] = set()
    for subgraph in detect_rigid_subgraphs(symbols, topology):
        for i, j in combinations(sorted(subgraph), 2):
            pair = (min(i, j), max(i, j))
            if pair in seen:
                continue
            seen.add(pair)
            constraints.append(
                RigidPairDistanceConstraint(
                    pair[0],
                    pair[1],
                    target=distance(reference_coords, pair[0], pair[1]),
                    sigma=sigma,
                    priority=priority,
                    class_label="rigid_subgraph",
                    weight=weight,
                )
            )
    return constraints


def build_clash_constraints(
    symbols: list[str],
    topology: "Topology",
    sigma: float = 0.10,
    priority: int = 2,
    weight: float = 1.0,
    vdw_scale: float = 0.75,
    max_reference_distance: float | None = None,
    reference_coords: np.ndarray | None = None,
) -> list[ClashConstraint]:
    """Build lower-bound nonbonded distance constraints from topology pairs."""
    if max_reference_distance is not None:
        if reference_coords is None:
            reference_coords = getattr(topology, "reference_coords", None)
        if reference_coords is None:
            raise ValueError("max_reference_distance requires reference coordinates.")
        reference_coords = _as_coords(reference_coords)

    constraints: list[ClashConstraint] = []
    for i, j in topology.nonbonded_pairs:
        if max_reference_distance is not None:
            if distance(reference_coords, i, j) > max_reference_distance:
                continue
        constraints.append(
            ClashConstraint(
                i,
                j,
                sigma=sigma,
                priority=priority,
                class_label="nonbonded_clash",
                weight=weight,
                symbols=(symbols[i], symbols[j]),
                vdw_scale=vdw_scale,
            )
        )
    return constraints


def build_constraints(
    symbols: list[str],
    coords: np.ndarray,
    topology: "Topology | None" = None,
    include_bonds: bool = True,
    include_angles: bool = True,
    include_rigid_pairs: bool = True,
    include_clashes: bool = False,
    bond_sigma: float = 0.05,
    angle_sigma: float = float(np.deg2rad(5.0)),
    rigid_sigma: float = 0.03,
    clash_sigma: float = 0.10,
    clash_max_reference_distance: float | None = None,
) -> list[Constraint]:
    """Build the default Phase 1 constraint set for a molecule.

    Clash constraints are opt-in because dense nonbonded lower bounds can be a
    large set. Projection/selection policies should enable them when needed.
    """
    if topology is None:
        from geoinit.core.topology import Topology

        topology = Topology(symbols, coords)

    constraints: list[Constraint] = []
    if include_bonds:
        constraints.extend(build_bond_constraints(symbols, topology, sigma=bond_sigma))
    if include_angles:
        constraints.extend(build_angle_constraints(symbols, topology, sigma=angle_sigma))
    if include_rigid_pairs:
        constraints.extend(
            build_rigid_pair_constraints(
                symbols,
                topology,
                reference_coords=coords,
                sigma=rigid_sigma,
            )
        )
    if include_clashes:
        constraints.extend(
            build_clash_constraints(
                symbols,
                topology,
                sigma=clash_sigma,
                max_reference_distance=clash_max_reference_distance,
                reference_coords=coords,
            )
        )
    return constraints


__all__ = [
    "AngleConstraint",
    "BondLengthConstraint",
    "ClashConstraint",
    "Constraint",
    "ContactWindowConstraint",
    "PlaneConstraint",
    "RigidPairDistanceConstraint",
    "WindowConstraint",
    "build_angle_constraints",
    "build_bond_constraints",
    "build_clash_constraints",
    "build_constraints",
    "build_rigid_pair_constraints",
    "constraint_score",
    "constraint_summary",
    "hard_constraint_failed",
]
