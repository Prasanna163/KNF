"""Candidate portfolio generation for geoinit."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from geoinit.core.classes import detect_chemical_classes
from geoinit.core.constraints import build_constraints, constraint_summary
from geoinit.core.contact_graph import build_contact_graph, contact_graph_similarity, contact_window_score
from geoinit.core.topology import Topology
from geoinit.optimize.guards import check_geometry
from geoinit.optimize.project import ProjectionPolicy, ProjectionResult, project_to_feasible_geometry
from geoinit.optimize.relax import project_bonds


@dataclass
class Candidate:
    name: str
    coords: np.ndarray
    prep_time: float
    source: str
    projection_result: ProjectionResult | None = None
    chemical_score: float = 0.0
    contact_score: float = 0.0
    clash_score: float = 0.0
    raw_movement_score: float = 0.0
    accepted: bool = True
    rejection_reason: str = ""
    estimated_xtb_steps: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def chem_score(self) -> float:
        """Compatibility alias for the V0.9 planning schema."""
        return self.chemical_score

    @chem_score.setter
    def chem_score(self, value: float) -> None:
        self.chemical_score = value

    @property
    def class_score(self) -> float:
        """Class-level risk proxy used by V0.9 reporting."""
        return float(self.metadata.get("class_score", max(self.chemical_score, self.contact_score)))

    @class_score.setter
    def class_score(self, value: float) -> None:
        self.metadata["class_score"] = float(value)

    @property
    def predicted_xtb_steps(self) -> float | None:
        """Compatibility alias for the V0.9 planning schema."""
        return self.estimated_xtb_steps

    @predicted_xtb_steps.setter
    def predicted_xtb_steps(self, value: float | None) -> None:
        self.estimated_xtb_steps = value

    def to_report_row(self, case_name: str = "", selected: bool = False) -> dict:
        """Return a stable CSV-friendly candidate row."""
        return {
            "case": case_name,
            "selected": selected,
            "candidate_name": self.name,
            "candidate": self.name,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "prep_time": self.prep_time,
            "chem_score": self.chemical_score,
            "chemical_score": self.chemical_score,
            "contact_score": self.contact_score,
            "class_score": self.class_score,
            "clash_score": self.clash_score,
            "raw_movement_score": self.raw_movement_score,
            "predicted_xtb_steps": self.estimated_xtb_steps,
            "estimated_xtb_steps": self.estimated_xtb_steps,
            "estimated_total_cost": self.metadata.get("estimated_total_cost"),
            "source": self.source,
        }


def _movement(coords: np.ndarray, raw_coords: np.ndarray) -> float:
    disp = coords - raw_coords
    per_atom = np.linalg.norm(disp, axis=1)
    return float(np.sqrt(np.mean(per_atom * per_atom)))


def _score_candidate(
    candidate: Candidate,
    symbols: list[str],
    raw_coords: np.ndarray,
    topology: Topology,
    raw_contact_graph=None,
) -> Candidate:
    constraints = build_constraints(symbols, raw_coords, topology=topology, include_clashes=True, clash_max_reference_distance=6.0)
    summary = constraint_summary(constraints, candidate.coords)
    report = check_geometry(symbols, candidate.coords, topology=topology)

    candidate.chemical_score = float(summary["mean_score"])
    candidate.clash_score = float(report.n_clashes)
    candidate.raw_movement_score = _movement(candidate.coords, raw_coords)
    candidate.metadata["constraint_summary"] = summary
    candidate.metadata["max_bond_error"] = report.max_bond_error
    candidate.metadata["max_clash_ratio"] = report.max_clash_ratio

    if raw_contact_graph is not None:
        classes = detect_chemical_classes(symbols, topology)
        graph = build_contact_graph(symbols, candidate.coords, classes=classes, topology=topology)
        candidate.contact_score = 1.0 - contact_graph_similarity(raw_contact_graph, graph)
        candidate.metadata["contact_window_score"] = contact_window_score(raw_contact_graph, candidate.coords)
        candidate.metadata["contact_edge_count"] = len(graph.edges)
    candidate.class_score = max(
        candidate.chemical_score / 25.0,
        candidate.contact_score,
        min(1.0, candidate.clash_score / 5.0),
        candidate.raw_movement_score / 1.5,
    )
    return candidate


def _atom_fragment_map(fragments: list[list[int]]) -> dict[int, int]:
    atom_frag: dict[int, int] = {}
    for frag_idx, fragment in enumerate(fragments):
        for atom in fragment:
            atom_frag[atom] = frag_idx
    return atom_frag


def _endpoint_center(coords: np.ndarray, endpoint: tuple[int, ...]) -> np.ndarray:
    return np.mean(coords[list(endpoint)], axis=0)


def _translate_fragment(coords: np.ndarray, fragment: list[int], shift: np.ndarray) -> None:
    if fragment:
        coords[fragment] = coords[fragment] + shift


def _candidate_from_contact_edges(
    name: str,
    symbols: list[str],
    coords: np.ndarray,
    fragments: list[list[int]],
    raw_contact_graph,
    contact_types: set[str],
    max_translation: float = 0.35,
) -> Candidate:
    """Build a conservative rigid-fragment translation candidate from contact windows."""
    t0 = time.perf_counter()
    candidate_coords = np.asarray(coords, dtype=np.float64).copy()
    atom_frag = _atom_fragment_map(fragments)
    usable_edges = [edge for edge in raw_contact_graph.edges if edge.contact_type in contact_types]

    if not usable_edges:
        return Candidate(
            name=name,
            coords=candidate_coords,
            prep_time=time.perf_counter() - t0,
            source="contact_projection",
            accepted=False,
            rejection_reason="no_matching_contacts",
            metadata={"contact_types": sorted(contact_types)},
        )

    moved_edges = 0
    total_shift = 0.0
    for edge in usable_edges:
        frag_a = atom_frag.get(edge.endpoint_a[0])
        frag_b = atom_frag.get(edge.endpoint_b[0])
        if frag_a is None or frag_b is None or frag_a == frag_b:
            continue

        center_a = _endpoint_center(candidate_coords, edge.endpoint_a)
        center_b = _endpoint_center(candidate_coords, edge.endpoint_b)
        vector = center_b - center_a
        dist = float(np.linalg.norm(vector))
        if dist < 1e-8:
            continue
        target = 0.5 * (edge.min_r + edge.max_r)
        delta = target - dist
        if abs(delta) < 0.03:
            continue

        # Move the non-anchor fragment when possible. Fragment 0 acts as the
        # reference host, matching the existing complex optimizer convention.
        movable_frag = frag_b if frag_a == 0 else frag_a
        direction = vector / dist
        if movable_frag == frag_a:
            direction = -direction
        shift_mag = float(np.clip(delta, -max_translation, max_translation))
        shift = shift_mag * direction
        _translate_fragment(candidate_coords, fragments[movable_frag], shift)
        moved_edges += 1
        total_shift += abs(shift_mag)

    if moved_edges == 0:
        accepted = False
        reason = "contacts_already_in_window"
    else:
        accepted = True
        reason = ""

    return Candidate(
        name=name,
        coords=candidate_coords,
        prep_time=time.perf_counter() - t0,
        source="contact_projection",
        accepted=accepted,
        rejection_reason=reason,
        metadata={
            "contact_types": sorted(contact_types),
            "moved_edges": moved_edges,
            "total_contact_shift": total_shift,
        },
    )


def generate_hbond_window_candidate(
    symbols: list[str],
    coords: np.ndarray,
    fragments: list[list[int]],
    raw_contact_graph,
) -> Candidate:
    return _candidate_from_contact_edges(
        "hbond_contact_projection",
        symbols,
        coords,
        fragments,
        raw_contact_graph,
        {"H_BOND"},
        max_translation=0.30,
    )


def generate_polar_contact_candidate(
    symbols: list[str],
    coords: np.ndarray,
    fragments: list[list[int]],
    raw_contact_graph,
) -> Candidate:
    return _candidate_from_contact_edges(
        "polar_contact_projection",
        symbols,
        coords,
        fragments,
        raw_contact_graph,
        {"POLAR", "HALOGEN"},
        max_translation=0.25,
    )


def generate_pi_contact_candidate(
    symbols: list[str],
    coords: np.ndarray,
    fragments: list[list[int]],
    raw_contact_graph,
) -> Candidate:
    return _candidate_from_contact_edges(
        "pi_contact_projection",
        symbols,
        coords,
        fragments,
        raw_contact_graph,
        {"PI_PI", "PI_POLAR"},
        max_translation=0.25,
    )


def generate_anchor_cleanup_candidate(
    symbols: list[str],
    coords: np.ndarray,
    topology: Topology,
    seed: Candidate | None = None,
) -> Candidate:
    """Apply a shallow covalent cleanup to a complex candidate."""
    t0 = time.perf_counter()
    start_coords = seed.coords if seed is not None else coords
    try:
        cleaned = project_bonds(symbols, start_coords, topology, max_iters=8, tolerance=0.002)
        movement = _movement(cleaned, np.asarray(coords, dtype=np.float64))
        return Candidate(
            name="fragment_anchor_cleanup",
            coords=cleaned,
            prep_time=time.perf_counter() - t0,
            source="project_bonds_after_contact",
            accepted=True,
            metadata={
                "seed_candidate": seed.name if seed is not None else "raw",
                "cleanup_movement": movement,
            },
        )
    except Exception as exc:
        return Candidate(
            name="fragment_anchor_cleanup",
            coords=np.asarray(coords, dtype=np.float64).copy(),
            prep_time=time.perf_counter() - t0,
            source="project_bonds_after_contact",
            accepted=False,
            rejection_reason=f"candidate_error:{exc}",
            metadata={"seed_candidate": seed.name if seed is not None else "raw"},
        )


def generate_candidates(
    symbols: list[str],
    coords: np.ndarray,
    topology: Topology | None = None,
    projection_policy: ProjectionPolicy | None = None,
    include_projection_candidates: bool = False,
    include_light_cleanup: bool = True,
    include_complex_candidates: bool = False,
    include_v1_relax: bool = False,
    include_v2_relax: bool = False,
    v2_engine: str = "auto",
) -> list[Candidate]:
    """Generate a small portfolio of cheap warm-start candidates."""
    coords = np.asarray(coords, dtype=np.float64)
    topology = topology or Topology(symbols, coords)
    classes = detect_chemical_classes(symbols, topology)
    is_complex = len(classes.fragments) >= 2
    raw_contact_graph = build_contact_graph(symbols, coords, classes=classes, topology=topology) if is_complex else None

    candidates: list[Candidate] = [
        Candidate(
            name="raw",
            coords=coords.copy(),
            prep_time=0.0,
            source="raw",
            accepted=True,
        )
    ]

    t0 = time.perf_counter()
    try:
        bond_coords = project_bonds(symbols, coords, topology)
        candidates.append(
            Candidate(
                name="bond_projection",
                coords=bond_coords,
                prep_time=time.perf_counter() - t0,
                source="project_bonds",
            )
        )
    except Exception as exc:
        candidates.append(
            Candidate(
                name="bond_projection",
                coords=coords.copy(),
                prep_time=time.perf_counter() - t0,
                source="project_bonds",
                accepted=False,
                rejection_reason=f"candidate_error:{exc}",
            )
        )

    if include_v1_relax or include_v2_relax:
        t0 = time.perf_counter()
        try:
            from geoinit.optimize.relax import relax as _relax

            v1_res = _relax(
                symbols, coords, topology=topology, mode="fast",
                engine=v2_engine, profile="v2",
            )
            candidates.append(
                Candidate(
                    name="v1_relax",
                    coords=np.asarray(v1_res.final_coords, dtype=np.float64),
                    prep_time=time.perf_counter() - t0,
                    source="relax_profile_v2",
                    accepted=True,
                    metadata={"relax_message": v1_res.message},
                )
            )
        except Exception as exc:
            candidates.append(
                Candidate(
                    name="v1_relax",
                    coords=coords.copy(),
                    prep_time=time.perf_counter() - t0,
                    source="relax_profile_v2",
                    accepted=False,
                    rejection_reason=f"candidate_error:{exc}",
                )
            )

    policy = projection_policy or ProjectionPolicy()
    if include_projection_candidates:
        try:
            constraints = build_constraints(
                symbols,
                coords,
                topology=topology,
                include_bonds=True,
                include_angles=True,
                include_rigid_pairs=True,
                include_clashes=False,
            )
            projection = project_to_feasible_geometry(symbols, coords, constraints=constraints, topology=topology, policy=policy)
            candidates.append(
                Candidate(
                    name="rigid_projection",
                    coords=projection.final_coords,
                    prep_time=projection.wall_time,
                    source="project_to_feasible_geometry",
                    projection_result=projection,
                    accepted=projection.success or projection.rolled_back,
                    rejection_reason="" if projection.success else projection.message,
                )
            )
        except Exception as exc:
            candidates.append(
                Candidate(
                    name="rigid_projection",
                    coords=coords.copy(),
                    prep_time=0.0,
                    source="project_to_feasible_geometry",
                    accepted=False,
                    rejection_reason=f"candidate_error:{exc}",
                )
            )

    complex_seed: Candidate | None = None
    if is_complex and include_complex_candidates:
        t0 = time.perf_counter()
        try:
            from geoinit.optimize.complex import relax_complex

            complex_coords, rigid_coords = relax_complex(
                symbols,
                coords,
                classes.fragments,
                maxiter=100,
                mode="fast",
                t_start=t0,
                prep_time_budget=0.25,
            )
            complex_seed = Candidate(
                name="rigid_fragment_SE3",
                coords=rigid_coords if rigid_coords is not None else complex_coords,
                prep_time=time.perf_counter() - t0,
                source="relax_complex",
            )
            candidates.append(complex_seed)
        except Exception as exc:
            candidates.append(
                Candidate(
                    name="rigid_fragment_SE3",
                    coords=coords.copy(),
                    prep_time=time.perf_counter() - t0,
                    source="relax_complex",
                    accepted=False,
                    rejection_reason=f"candidate_error:{exc}",
                )
            )

        if raw_contact_graph is not None:
            contact_candidates = [
                generate_hbond_window_candidate(symbols, coords, classes.fragments, raw_contact_graph),
                generate_polar_contact_candidate(symbols, coords, classes.fragments, raw_contact_graph),
                generate_pi_contact_candidate(symbols, coords, classes.fragments, raw_contact_graph),
            ]
            candidates.extend(contact_candidates)
            accepted_contact = next((cand for cand in contact_candidates if cand.accepted), None)
            candidates.append(
                generate_anchor_cleanup_candidate(
                    symbols,
                    coords,
                    topology,
                    seed=accepted_contact or complex_seed,
                )
            )

    if include_light_cleanup and include_projection_candidates:
        try:
            constraints = build_constraints(
                symbols,
                coords,
                topology=topology,
                include_bonds=True,
                include_angles=True,
                include_rigid_pairs=True,
                include_clashes=True,
                clash_max_reference_distance=5.0,
            )
            cleanup_policy = ProjectionPolicy(raw_anchor_weight=2.0, maxiter=25, prep_time_budget=0.25)
            cleanup = project_to_feasible_geometry(symbols, coords, constraints=constraints, topology=topology, policy=cleanup_policy)
            candidates.append(
                Candidate(
                    name="light_cleanup",
                    coords=cleanup.final_coords,
                    prep_time=cleanup.wall_time,
                    source="project_to_feasible_geometry",
                    projection_result=cleanup,
                    accepted=cleanup.success or cleanup.rolled_back,
                    rejection_reason="" if cleanup.success else cleanup.message,
                )
            )
        except Exception as exc:
            candidates.append(
                Candidate(
                    name="light_cleanup",
                    coords=coords.copy(),
                    prep_time=0.0,
                    source="project_to_feasible_geometry",
                    accepted=False,
                    rejection_reason=f"candidate_error:{exc}",
                )
            )

    return [
        _score_candidate(candidate, symbols, coords, topology, raw_contact_graph=raw_contact_graph)
        for candidate in candidates
    ]


__all__ = [
    "Candidate",
    "generate_anchor_cleanup_candidate",
    "generate_candidates",
    "generate_hbond_window_candidate",
    "generate_pi_contact_candidate",
    "generate_polar_contact_candidate",
]
