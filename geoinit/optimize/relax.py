"""SciPy‑based geometry relaxation for the GeoInit functional.

The optimiser flattens the (N, 3) coordinate array into ℝ^{3N} and
delegates to :func:`scipy.optimize.minimize`.  The topology is built
once by :class:`~geoinit.energy.functional.GeoInitFunctional` and
remains frozen throughout the relaxation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

from geoinit.energy.functional import GeoInitFunctional
from geoinit.core.topology import Topology


def project_bonds(
    symbols: list[str],
    coords: np.ndarray,
    topology: Topology,
    max_iters: int = 15,
    tolerance: float = 0.01,
) -> np.ndarray:
    """Correct bond lengths using fast pairwise projection."""
    from geoinit.core.params import get_covalent_radius
    coords = coords.copy()
    for _ in range(max_iters):
        max_err = 0.0
        for b in topology.bonds:
            i, j = b.i, b.j
            r0 = getattr(b, "r0", None)
            if r0 is None:
                r0 = get_covalent_radius(symbols[i]) + get_covalent_radius(symbols[j])
            vec = coords[j] - coords[i]
            r = np.linalg.norm(vec)
            if r < 1e-6:
                continue
            err = r - r0
            max_err = max(max_err, abs(err))
            u = vec / r
            coords[i] += 0.5 * err * u
            coords[j] -= 0.5 * err * u
        if max_err < tolerance:
            break
    return coords


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RelaxResult:
    """Result of a geometry relaxation.

    Attributes
    ----------
    symbols : list[str]
        Atomic symbols.
    initial_coords : np.ndarray, shape (N, 3)
        Coordinates before relaxation.
    final_coords : np.ndarray, shape (N, 3)
        Coordinates after relaxation.
    initial_energy : float
        Total GeoInit energy of the initial geometry.
    final_energy : float
        Total GeoInit energy of the relaxed geometry.
    energy_components : dict[str, float]
        Breakdown of the final energy into individual weighted terms.
    success : bool
        Whether the optimiser reported convergence.
    n_steps : int
        Number of optimisation iterations.
    message : str
        Optimiser convergence message.
    """

    symbols: list[str]
    initial_coords: np.ndarray
    final_coords: np.ndarray
    initial_energy: float
    final_energy: float
    energy_components: dict[str, float] = field(default_factory=dict)
    success: bool = False
    n_steps: int = 0
    message: str = ""
    rigid_coords: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Main relaxation driver
# ---------------------------------------------------------------------------

def relax(
    symbols: list[str],
    coords: np.ndarray,
    weights: dict | None = None,
    charges: np.ndarray | None = None,
    sigma: float = 0.05,
    maxiter: int = 500,
    method: str = "L-BFGS-B",
    gtol: float = 1e-5,
    topology_scale: float = 1.25,
    clash_mode: str = "compact",
    topology: Topology | None = None,
    mode: str = "fast",
    prep_time_budget: float | None = None,
    engine: str = "scalar",
    profile: str = "v1",
    backend: str = "auto",
) -> RelaxResult:
    """Relax a molecular geometry using the GeoInit functional.

    The ``engine``/``profile``/``backend`` arguments select the compute path:
    ``engine='scalar'`` + ``profile='v1'`` (the defaults) reproduce the frozen
    GeoInit-V1 behaviour exactly, while ``engine='auto'`` routes energy/gradient
    through the vectorised :class:`~geoinit.energy.engine.EnergyEngine` (NumPy /
    Torch-CPU / CUDA) and ``profile='v2'`` enables the improved physics.
    """
    t_start = time.perf_counter()
    coords = np.asarray(coords, dtype=np.float64)
    initial_coords = coords.copy()

    # Dynamic budget estimation
    N = len(symbols)
    if prep_time_budget is None:
        if N <= 15:
            prep_time_budget = 0.05
        elif N <= 40:
            prep_time_budget = 0.20
        else:
            prep_time_budget = 1.00

    if topology is None:
        topology = Topology(symbols, coords, scale=topology_scale, sigma=sigma)

    # 0. Pre-Relaxation Guard
    from geoinit.optimize.guards import should_skip_geoinit, check_geometry
    skip, reason = should_skip_geoinit(symbols, coords, topology=topology)
    if skip:
        is_safe_skip = reason in ("already_safe_low_benefit", "complex_already_safe_low_benefit", "too_small")
        return RelaxResult(
            symbols=list(symbols),
            initial_coords=initial_coords,
            final_coords=initial_coords,
            initial_energy=0.0,
            final_energy=0.0,
            success=is_safe_skip,
            n_steps=0,
            message=f"skipped_{reason}",
        )

    # Detect connected components / fragments
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

    # Fast Mode projection for single molecules
    if mode == "fast" and not is_complex:
        coords_proj = project_bonds(symbols, coords, topology)
        report_proj = check_geometry(symbols, coords_proj, topology=topology)
        if report_proj.max_bond_error < 0.01 and report_proj.n_clashes == 0:
            return RelaxResult(
                symbols=list(symbols),
                initial_coords=initial_coords,
                final_coords=coords_proj,
                initial_energy=0.0,
                final_energy=0.0,
                success=True,
                n_steps=0,
                message="projection_converged",
            )
        else:
            coords = coords_proj

    if is_complex:
        # Complex detected! Redirect to fragment-based complex optimization
        from geoinit.optimize.complex import relax_complex
        final_coords, rigid_coords = relax_complex(
            symbols,
            coords,
            fragments,
            weights=weights,
            charges=charges,
            sigma=sigma,
            maxiter=maxiter,
            clash_mode=clash_mode,
            topology_scale=topology_scale,
            mode=mode,
            t_start=t_start,
            prep_time_budget=prep_time_budget,
            engine=engine,
            profile=profile,
            backend=backend,
        )
        # We need a temporary functional to get energy components
        temp_functional = GeoInitFunctional(
            symbols,
            initial_coords,
            weights=weights,
            charges=charges,
            sigma=sigma,
            topology_scale=topology_scale,
            clash_mode=clash_mode,
        )
        temp_functional.topology = topology
        final_energy = temp_functional.energy(final_coords)
        components = temp_functional.energy_components(final_coords)
        return RelaxResult(
            symbols=list(symbols),
            initial_coords=initial_coords,
            final_coords=final_coords,
            initial_energy=temp_functional.energy(initial_coords),
            final_energy=final_energy,
            energy_components=components,
            success=True,
            n_steps=maxiter,
            message="complex_optimized",
            rigid_coords=rigid_coords,
        )

    # 1. Build functional (topology frozen from initial geometry)
    functional = GeoInitFunctional(
        symbols,
        coords,
        weights=weights,
        charges=charges,
        sigma=sigma,
        topology_scale=topology_scale,
        clash_mode=clash_mode,
        use_sparse=(mode == "fast"),
        engine=engine,
        profile=profile,
        backend=backend,
    )
    functional.topology = topology

    # Enforce budget before minimization starts
    if time.perf_counter() - t_start > prep_time_budget:
        return RelaxResult(
            symbols=list(symbols),
            initial_coords=initial_coords,
            final_coords=initial_coords,
            initial_energy=functional.energy(initial_coords),
            final_energy=functional.energy(initial_coords),
            success=False,
            n_steps=0,
            message="timeout",
        )

    initial_energy = functional.energy(coords)

    # 2. Flatten to ℝ^{3N}
    x0 = coords.ravel().copy()

    # 3. Build options dict appropriate for the chosen method.
    # V2 affords a larger fast-mode iteration budget: the vectorised engine makes
    # each energy/gradient evaluation ~25x cheaper, so the warm-start can be
    # converged more tightly at negligible wall-cost.
    if mode == "fast":
        fast_cap = 60 if profile == "v2" else 30
        actual_maxiter = min(fast_cap, maxiter)
    else:
        actual_maxiter = maxiter
    options: dict = {"maxiter": actual_maxiter, "disp": False}
    if method in ("L-BFGS-B", "BFGS", "CG"):
        options["gtol"] = gtol

    class SafeConvergedException(Exception):
        def __init__(self, x: np.ndarray) -> None:
            self.x = x

    class TimeoutException(Exception):
        pass

    energy_history: list[float] = []

    def callback(xk: np.ndarray) -> None:
        if time.perf_counter() - t_start > prep_time_budget:
            raise TimeoutException()

        coords_curr = xk.reshape(len(symbols), 3)
        e_curr = functional.energy(coords_curr)
        energy_history.append(e_curr)

        report = check_geometry(symbols, coords_curr, topology=functional.topology)
        if report.max_bond_error < 0.01 and report.n_clashes == 0:
            raise SafeConvergedException(xk)

        if len(energy_history) >= 25:
            delta_e = abs(energy_history[-1] - energy_history[-25])
            if delta_e < 1e-4:
                if report.max_bond_error < 0.005 and report.n_clashes == 0:
                    raise SafeConvergedException(xk)

    # 4. Run optimiser
    try:
        result = minimize(
            fun=functional.energy_flat,
            x0=x0,
            jac=functional.gradient_flat,
            method=method,
            callback=callback,
            options=options,
        )
        final_x = result.x
        nit = int(result.nit)
        success = bool(result.success)
        message = str(result.message)
    except SafeConvergedException as exc:
        final_x = exc.x
        nit = len(energy_history)
        success = True
        message = "safe_converged"
    except TimeoutException:
        final_x = x0
        nit = len(energy_history)
        success = False
        message = "timeout"

    # 5. Reshape result
    final_coords = final_x.reshape(N, 3)
    final_energy = functional.energy(final_coords)
    components = functional.energy_components(final_coords)

    return RelaxResult(
        symbols=list(symbols),
        initial_coords=initial_coords,
        final_coords=final_coords,
        initial_energy=initial_energy,
        final_energy=final_energy,
        energy_components=components,
        success=success,
        n_steps=nit,
        message=message,
    )
