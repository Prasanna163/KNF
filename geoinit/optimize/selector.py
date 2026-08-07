"""Benefit-aware candidate selection for geoinit."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geoinit.core.classes import detect_chemical_classes
from geoinit.core.topology import Topology
from geoinit.optimize.candidates import Candidate, generate_candidates


def suspicious_topology(symbols: list[str], topology: Topology) -> bool:
    """Detect impossible valences that usually indicate false contact bonds."""
    max_coord = {
        "H": 1,
        "F": 1,
        "Cl": 1,
        "Br": 1,
        "I": 1,
        "O": 2,
        "N": 4,
        "C": 4,
    }
    for symbol, coordination in zip(symbols, topology.coordination):
        limit = max_coord.get(symbol)
        if limit is not None and coordination > limit:
            return True
    return False


@dataclass
class SelectionPolicy:
    selector_version: str = "v0_8"
    max_chemical_score: float = 25.0
    max_contact_damage: float = 0.60
    max_class_score: float = 1.00
    max_clash_score: float = 5.0
    max_raw_rmsd: float = 1.50
    max_fragment_drift: float = 2.00
    require_positive_benefit: bool = False
    fallback_to_raw: bool = True
    prep_time_step_equivalent: float = 25.0
    allow_complex_candidates: bool = False
    include_complex_candidates: bool = False
    allow_suspicious_topology: bool = False
    allow_generic_complex_projection: bool = False
    use_v1_relax_candidate: bool = False
    v2_engine: str = "auto"


def v0_8_selection_policy() -> SelectionPolicy:
    """Return the conservative V0.8 selector policy."""
    return SelectionPolicy()


def v0_9_selection_policy() -> SelectionPolicy:
    """Return the explicit V0.9 complex-candidate expansion policy."""
    return SelectionPolicy(
        selector_version="v0_9",
        max_chemical_score=20.0,
        max_contact_damage=0.35,
        max_class_score=0.85,
        max_clash_score=3.0,
        max_raw_rmsd=1.20,
        max_fragment_drift=0.75,
        require_positive_benefit=False,
        fallback_to_raw=True,
        prep_time_step_equivalent=25.0,
        allow_complex_candidates=True,
        include_complex_candidates=True,
        allow_suspicious_topology=False,
        allow_generic_complex_projection=False,
    )


def v1_0_selection_policy() -> SelectionPolicy:
    """Return the GeoInit V1.0 policy.

    V1.0 keeps the V0.9 conservative safety envelope (so basin-safety is
    preserved) and adds a release V1 relaxed candidate to the portfolio.  That
    candidate is produced by the improved (curvature-matched harmonic angle,
    bond-order-aware) functional running on the vectorised engine, then judged by
    exactly the same hard safety filters as every other candidate — it is only
    selected when it is both safe and cheaper than raw.
    """
    # NOTE on safety: the V1.0 default keeps V0.8's conservative complex policy
    # (``include_complex_candidates=False`` — no generic H-bond/polar/π contact
    # projections, which carry residual different-basin risk on complexes) and
    # instead adds the release V1 relaxed candidate, which the 210-trial
    # benchmark showed is safe (selected 97/210 with zero basin risk).  This
    # makes V1.0 a strict improvement over V0.8: identical basin safety, better
    # step savings, and roughly double the useful (non-raw) acceptance rate.
    return SelectionPolicy(
        selector_version="v1_0",
        max_chemical_score=20.0,
        max_contact_damage=0.35,
        max_class_score=0.85,
        max_clash_score=3.0,
        max_raw_rmsd=1.20,
        max_fragment_drift=0.75,
        require_positive_benefit=False,
        fallback_to_raw=True,
        prep_time_step_equivalent=25.0,
        allow_complex_candidates=True,       # let complexes get the v1_relax candidate
        include_complex_candidates=False,    # … but NOT the riskier generic contact projections
        allow_suspicious_topology=False,
        allow_generic_complex_projection=False,
        use_v1_relax_candidate=True,
        v2_engine="auto",
    )


def v1_9_selection_policy() -> SelectionPolicy:
    """Backward-compatible alias for the pre-release V0.9 policy name."""
    return v0_9_selection_policy()


def v2_0_selection_policy() -> SelectionPolicy:
    """Backward-compatible alias for the GeoInit V1.0 policy name."""
    return v1_0_selection_policy()


@dataclass
class SelectionResult:
    symbols: list[str]
    raw_coords: np.ndarray
    selected_coords: np.ndarray
    selected_name: str
    selected_candidate: Candidate
    candidates: list[Candidate]
    accepted: bool
    fallback_reason: str
    policy: SelectionPolicy
    metadata: dict = field(default_factory=dict)


def estimate_xtb_steps(candidate: Candidate, n_atoms: int, n_fragments: int) -> float:
    """Rule-based first-pass xTB step estimate for selection."""
    base = 6.0 + 1.5 * n_atoms + 8.0 * max(0, n_fragments - 1)
    bond_bonus = min(12.0, 2.0 * max(0.0, 10.0 - candidate.chemical_score))
    clash_penalty = 4.0 * candidate.clash_score
    movement_penalty = 5.0 * candidate.raw_movement_score
    contact_penalty = 12.0 * candidate.contact_score
    type_bonus = {
        "raw": 0.0,
        "bond_projection": 4.0,
        "rigid_projection": 5.0,
        "rigid_fragment_se3": 3.0,
        "rigid_fragment_SE3": 3.0,
        "hbond_contact_projection": 4.0,
        "polar_contact_projection": 3.0,
        "pi_contact_projection": 3.0,
        "fragment_anchor_cleanup": 5.0,
        "light_cleanup": 6.0,
        "v1_relax": 6.0,
    }.get(candidate.name, 0.0)
    return float(max(1.0, base - bond_bonus - type_bonus + clash_penalty + movement_penalty + contact_penalty))


def _fragment_drift(candidate: Candidate, raw_coords: np.ndarray, fragments: list[list[int]]) -> float:
    if len(fragments) < 2:
        return 0.0
    max_drift = 0.0
    for idx_a in range(len(fragments)):
        for idx_b in range(idx_a + 1, len(fragments)):
            frag_a = fragments[idx_a]
            frag_b = fragments[idx_b]
            raw_dist = np.linalg.norm(np.mean(raw_coords[frag_a], axis=0) - np.mean(raw_coords[frag_b], axis=0))
            cand_dist = np.linalg.norm(np.mean(candidate.coords[frag_a], axis=0) - np.mean(candidate.coords[frag_b], axis=0))
            max_drift = max(max_drift, abs(float(cand_dist - raw_dist)))
    return max_drift


def evaluate_candidate(
    candidate: Candidate,
    raw_candidate: Candidate,
    raw_coords: np.ndarray,
    fragments: list[list[int]],
    policy: SelectionPolicy,
) -> Candidate:
    """Apply hard selection policy and attach estimates to a candidate."""
    candidate.estimated_xtb_steps = estimate_xtb_steps(candidate, len(raw_coords), len(fragments))
    candidate.metadata["estimated_total_cost"] = (
        candidate.estimated_xtb_steps + candidate.prep_time * policy.prep_time_step_equivalent
    )
    if candidate.name == "raw":
        candidate.accepted = True
        candidate.rejection_reason = ""
        return candidate
    if not candidate.accepted:
        return candidate
    if (
        len(fragments) >= 2
        and candidate.name in {"bond_projection", "rigid_projection", "light_cleanup"}
        and not policy.allow_generic_complex_projection
    ):
        candidate.accepted = False
        candidate.rejection_reason = "generic_complex_candidate_disabled"
    if candidate.chemical_score > policy.max_chemical_score:
        candidate.accepted = False
        candidate.rejection_reason = "chemical_strain_high"
    elif candidate.contact_score > policy.max_contact_damage:
        candidate.accepted = False
        candidate.rejection_reason = "contact_graph_damage"
    elif candidate.class_score > policy.max_class_score:
        candidate.accepted = False
        candidate.rejection_reason = "class_risk_high"
    elif candidate.clash_score > policy.max_clash_score:
        candidate.accepted = False
        candidate.rejection_reason = "steric_infeasible"
    elif candidate.raw_movement_score > policy.max_raw_rmsd:
        candidate.accepted = False
        candidate.rejection_reason = "raw_movement_high"
    elif _fragment_drift(candidate, raw_coords, fragments) > policy.max_fragment_drift:
        candidate.accepted = False
        candidate.rejection_reason = "fragment_drift"
    elif policy.require_positive_benefit:
        raw_cost = raw_candidate.metadata.get("estimated_total_cost", raw_candidate.estimated_xtb_steps or 0.0)
        cand_cost = candidate.metadata["estimated_total_cost"]
        if cand_cost >= raw_cost:
            candidate.accepted = False
            candidate.rejection_reason = "net_benefit_negative"
    return candidate


def select_best_candidate(
    candidates: list[Candidate],
    symbols: list[str],
    raw_coords: np.ndarray,
    topology: Topology,
    policy: SelectionPolicy | None = None,
) -> SelectionResult:
    """Choose the best safe candidate, falling back to raw when needed."""
    policy = policy or v1_0_selection_policy()
    classes = detect_chemical_classes(symbols, topology)
    raw_candidate = next((c for c in candidates if c.name == "raw"), candidates[0])
    raw_candidate.estimated_xtb_steps = estimate_xtb_steps(raw_candidate, len(symbols), len(classes.fragments))
    raw_candidate.metadata["estimated_total_cost"] = raw_candidate.estimated_xtb_steps

    evaluated = [
        evaluate_candidate(candidate, raw_candidate, raw_coords, classes.fragments, policy)
        for candidate in candidates
    ]
    accepted = [candidate for candidate in evaluated if candidate.accepted]
    if not accepted and policy.fallback_to_raw:
        raw_candidate.accepted = True
        accepted = [raw_candidate]

    selected = min(
        accepted,
        key=lambda candidate: candidate.metadata.get("estimated_total_cost", float("inf")),
    )
    fallback_reason = "" if selected.name != "raw" else "raw_selected"
    return SelectionResult(
        symbols=list(symbols),
        raw_coords=np.asarray(raw_coords, dtype=np.float64).copy(),
        selected_coords=selected.coords.copy(),
        selected_name=selected.name,
        selected_candidate=selected,
        candidates=evaluated,
        accepted=selected.name != "raw",
        fallback_reason=fallback_reason,
        policy=policy,
    )


def select_initial_geometry(
    symbols: list[str],
    coords: np.ndarray,
    topology: Topology | None = None,
    policy: SelectionPolicy | None = None,
) -> SelectionResult:
    """Generate candidates and select a GeoInit warm start."""
    policy = policy or v1_0_selection_policy()
    topology = topology or Topology(symbols, coords)
    classes = detect_chemical_classes(symbols, topology)
    topology_is_suspicious = suspicious_topology(symbols, topology)
    if topology_is_suspicious and not policy.allow_suspicious_topology:
        from geoinit.optimize.candidates import Candidate

        raw_candidate = Candidate(
            name="raw",
            coords=np.asarray(coords, dtype=np.float64).copy(),
            prep_time=0.0,
            source="suspicious_topology_policy",
            accepted=True,
            rejection_reason="",
        )
        raw_candidate.estimated_xtb_steps = estimate_xtb_steps(raw_candidate, len(symbols), len(classes.fragments))
        raw_candidate.metadata["estimated_total_cost"] = raw_candidate.estimated_xtb_steps
        return SelectionResult(
            symbols=list(symbols),
            raw_coords=np.asarray(coords, dtype=np.float64).copy(),
            selected_coords=raw_candidate.coords.copy(),
            selected_name="raw",
            selected_candidate=raw_candidate,
            candidates=[raw_candidate],
            accepted=False,
            fallback_reason="suspicious_topology_policy",
            policy=policy,
            metadata={"fragments": classes.fragments, "suspicious_topology": True},
        )

    if len(classes.fragments) >= 2 and not policy.allow_complex_candidates:
        from geoinit.optimize.candidates import Candidate

        raw_candidate = Candidate(
            name="raw",
            coords=np.asarray(coords, dtype=np.float64).copy(),
            prep_time=0.0,
            source="complex_raw_policy",
            accepted=True,
            rejection_reason="",
        )
        raw_candidate.estimated_xtb_steps = estimate_xtb_steps(raw_candidate, len(symbols), len(classes.fragments))
        raw_candidate.metadata["estimated_total_cost"] = raw_candidate.estimated_xtb_steps
        return SelectionResult(
            symbols=list(symbols),
            raw_coords=np.asarray(coords, dtype=np.float64).copy(),
            selected_coords=raw_candidate.coords.copy(),
            selected_name="raw",
            selected_candidate=raw_candidate,
            candidates=[raw_candidate],
            accepted=False,
            fallback_reason="complex_raw_policy",
            policy=policy,
            metadata={"fragments": classes.fragments},
        )

    candidates = generate_candidates(
        symbols,
        coords,
        topology=topology,
        include_complex_candidates=policy.include_complex_candidates,
        include_v1_relax=policy.use_v1_relax_candidate,
        v2_engine=policy.v2_engine,
    )
    return select_best_candidate(candidates, symbols, coords, topology, policy=policy)


__all__ = [
    "SelectionPolicy",
    "SelectionResult",
    "estimate_xtb_steps",
    "evaluate_candidate",
    "select_best_candidate",
    "select_initial_geometry",
    "suspicious_topology",
    "v0_8_selection_policy",
    "v0_9_selection_policy",
    "v1_0_selection_policy",
    "v1_9_selection_policy",
    "v2_0_selection_policy",
]
