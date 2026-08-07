"""Closest-feasible projection for geoinit.

Projection minimizes movement from the raw geometry while reducing normalized
chemical constraint violations. This is the V0.8 replacement surface for
"minimize one pseudo-energy" workflows, but it is not wired into legacy relax
behavior by default.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from geoinit.core.constraints import (
    Constraint,
    build_constraints,
    constraint_summary,
    hard_constraint_failed,
)
from geoinit.core.topology import Topology


@dataclass
class ProjectionPolicy:
    raw_anchor_weight: float = 1.0
    huber_delta: float = 3.0
    maxiter: int = 50
    max_hard_residual: float = 3.0
    hard_priority: int = 1
    rollback_on_guard_failure: bool = True
    prep_time_budget: float | None = None


@dataclass
class ProjectionResult:
    symbols: list[str]
    initial_coords: np.ndarray
    final_coords: np.ndarray
    constraints: list[Constraint]
    initial_summary: dict
    final_summary: dict
    success: bool
    n_steps: int
    message: str
    wall_time: float
    raw_rmsd: float
    max_displacement: float
    objective_initial: float
    objective_final: float
    rolled_back: bool = False
    metadata: dict = field(default_factory=dict)


def huber_loss(residual: float, delta: float) -> float:
    """Huber-style robust loss for a normalized residual."""
    abs_res = abs(float(residual))
    if abs_res <= delta:
        return 0.5 * abs_res * abs_res
    return delta * (abs_res - 0.5 * delta)


def projection_objective(
    coords: np.ndarray,
    raw_coords: np.ndarray,
    constraints: list[Constraint],
    policy: ProjectionPolicy,
) -> float:
    """Evaluate raw-anchor plus robust constraint penalties."""
    raw_term = policy.raw_anchor_weight * float(np.sum((coords - raw_coords) ** 2))
    constraint_term = 0.0
    for constraint in constraints:
        constraint_term += constraint.weight * huber_loss(constraint.residual(coords), policy.huber_delta)
    return float(raw_term + constraint_term)


def _movement(coords: np.ndarray, raw_coords: np.ndarray) -> tuple[float, float]:
    disp = coords - raw_coords
    per_atom = np.linalg.norm(disp, axis=1)
    return float(np.sqrt(np.mean(per_atom * per_atom))), float(np.max(per_atom))


def project_to_feasible_geometry(
    symbols: list[str],
    raw_coords: np.ndarray,
    constraints: list[Constraint] | None = None,
    topology: Topology | None = None,
    policy: ProjectionPolicy | None = None,
) -> ProjectionResult:
    """Project raw coordinates toward the nearest chemically feasible geometry."""
    t0 = time.perf_counter()
    policy = policy or ProjectionPolicy()
    raw_coords = np.asarray(raw_coords, dtype=np.float64)
    topology = topology or Topology(symbols, raw_coords)
    constraints = constraints or build_constraints(symbols, raw_coords, topology=topology)

    initial_summary = constraint_summary(constraints, raw_coords)
    objective_initial = projection_objective(raw_coords, raw_coords, constraints, policy)

    if policy.prep_time_budget is not None and policy.prep_time_budget <= 0.0:
        rmsd, max_disp = _movement(raw_coords, raw_coords)
        return ProjectionResult(
            symbols=list(symbols),
            initial_coords=raw_coords.copy(),
            final_coords=raw_coords.copy(),
            constraints=constraints,
            initial_summary=initial_summary,
            final_summary=initial_summary,
            success=False,
            n_steps=0,
            message="projection_timeout",
            wall_time=time.perf_counter() - t0,
            raw_rmsd=rmsd,
            max_displacement=max_disp,
            objective_initial=objective_initial,
            objective_final=objective_initial,
        )

    def objective_flat(x: np.ndarray) -> float:
        if policy.prep_time_budget is not None and time.perf_counter() - t0 > policy.prep_time_budget:
            return float("inf")
        coords = x.reshape(raw_coords.shape)
        return projection_objective(coords, raw_coords, constraints, policy)

    result = minimize(
        objective_flat,
        raw_coords.ravel().copy(),
        method="L-BFGS-B",
        options={"maxiter": policy.maxiter, "disp": False},
    )

    candidate_coords = result.x.reshape(raw_coords.shape)
    final_summary = constraint_summary(constraints, candidate_coords)
    objective_final = projection_objective(candidate_coords, raw_coords, constraints, policy)
    rolled_back = False

    guard_failed = hard_constraint_failed(
        final_summary,
        max_priority=policy.hard_priority,
        max_abs_residual=policy.max_hard_residual,
    )
    if guard_failed and policy.rollback_on_guard_failure:
        candidate_coords = raw_coords.copy()
        final_summary = initial_summary
        objective_final = objective_initial
        rolled_back = True

    rmsd, max_disp = _movement(candidate_coords, raw_coords)
    success = bool(result.success and not guard_failed)
    if rolled_back:
        message = "rolled_back_guard_failure"
    else:
        message = str(result.message)

    return ProjectionResult(
        symbols=list(symbols),
        initial_coords=raw_coords.copy(),
        final_coords=candidate_coords,
        constraints=constraints,
        initial_summary=initial_summary,
        final_summary=final_summary,
        success=success,
        n_steps=int(getattr(result, "nit", 0)),
        message=message,
        wall_time=time.perf_counter() - t0,
        raw_rmsd=rmsd,
        max_displacement=max_disp,
        objective_initial=objective_initial,
        objective_final=objective_final,
        rolled_back=rolled_back,
    )


__all__ = [
    "ProjectionPolicy",
    "ProjectionResult",
    "huber_loss",
    "project_to_feasible_geometry",
    "projection_objective",
]
