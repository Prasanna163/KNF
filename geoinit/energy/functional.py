"""GeoInit‑V1 combined energy functional.

The total GeoInit energy is the weighted sum of all terms::

    Φ_GeoInit = w_b Φ_bond + w_a Φ_angle + w_c Φ_clash + w_d Φ_disp + w_q Φ_coul

**Critical design choice** — the molecular topology (bonds, angles,
nonbonded pair list) is inferred *once* from the initial geometry and then
frozen.  During optimisation only coordinates change; the connectivity
graph remains constant.  This avoids discontinuities that would arise if
bonds appeared or disappeared between steps.
"""

from __future__ import annotations

import numpy as np

from geoinit.core.params import DEFAULT_WEIGHTS
from geoinit.core.topology import Topology
from geoinit.energy.angle import angle_energy, angle_gradient
from geoinit.energy.bond import bond_energy, bond_gradient
from geoinit.energy.nonbonded import (
    clash_energy,
    clash_gradient,
    coulomb_energy,
    coulomb_gradient,
    dispersion_energy,
    dispersion_gradient,
)


class GeoInitFunctional:
    """The GeoInit‑V1 functional combining all energy terms.

    Parameters
    ----------
    symbols : list[str]
        Atomic symbols, length *N*.
    coords : np.ndarray, shape (N, 3)
        **Initial** Cartesian coordinates in Å used to build the topology.
    weights : dict or None
        Override default term weights.  Keys: ``'bond'``, ``'angle'``,
        ``'clash'``, ``'disp'``, ``'coul'``.
    charges : np.ndarray or None, shape (N,)
        Per‑atom partial charges for the Coulomb term.
    sigma : float
        Bond‑length tolerance in Å (passed to :func:`bond_energy`).
    topology_scale : float
        Scaling factor for covalent‑radii sum used by
        :func:`~geoinit.core.topology.infer_bonds`.

    Attributes
    ----------
    symbols : list[str]
    n_atoms : int
    topology : Topology
        Frozen molecular topology built from the initial geometry.
    weights : dict[str, float]
    charges : np.ndarray or None
    sigma : float
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        symbols: list[str],
        coords: np.ndarray,
        weights: dict | None = None,
        charges: np.ndarray | None = None,
        sigma: float = 0.05,
        topology_scale: float = 1.25,
        clash_mode: str = "compact",
        anchor_coords: np.ndarray | None = None,
        k_anchor: float = 0.0,
        use_sparse: bool = False,
        engine: str = "scalar",
        profile: str = "v1",
        backend: str = "auto",
        dtype=None,
    ) -> None:
        self.symbols = list(symbols)
        self.n_atoms = len(symbols)
        self.sigma = sigma
        self.charges = np.asarray(charges, dtype=np.float64) if charges is not None else None
        self.clash_mode = clash_mode
        self.anchor_coords = anchor_coords.copy() if anchor_coords is not None else None
        self.k_anchor = k_anchor
        self._use_sparse = use_sparse
        self._initial_coords_for_sparse = coords.copy()

        # Compute backend selection.  ``engine='scalar'`` keeps the legacy
        # per-pair Python path (byte-identical to GeoInit-V1, default).  Any
        # other value, or ``profile='v2'``, routes energy/gradient through the
        # vectorised :class:`~geoinit.energy.engine.EnergyEngine`.
        self.engine_mode = engine
        self.profile = profile
        self.backend = backend
        self.dtype = dtype
        self._engine_obj = None

        # Merge user weights with defaults
        self.weights: dict[str, float] = dict(DEFAULT_WEIGHTS)
        if weights is not None:
            self.weights.update(weights)

        # Build topology ONCE from the initial geometry and freeze it
        self.topology = Topology(symbols, coords, scale=topology_scale, sigma=self.sigma)

    @property
    def topology(self) -> Topology:
        return self._topology

    @topology.setter
    def topology(self, topo: Topology) -> None:
        self._topology = topo
        # Any cached vectorised engine is bound to the previous topology's
        # frozen index tables; drop it so it is rebuilt lazily on next use.
        self._engine_obj = None

        # 1. Detect rigid subgraphs
        from geoinit.core.bond_rules import detect_rigid_subgraphs
        self.rigid_subgraphs = detect_rigid_subgraphs(self.symbols, topo)

        # 2. Extract ideal pairwise distances for all pairs in each subgraph
        self.rigid_pairs = []
        for subgraph in self.rigid_subgraphs:
            n_sg = len(subgraph)
            for idx_a in range(n_sg):
                for idx_b in range(idx_a + 1, n_sg):
                    i = subgraph[idx_a]
                    j = subgraph[idx_b]
                    diff = topo.reference_coords[i] - topo.reference_coords[j]
                    d0 = float(np.linalg.norm(diff))
                    self.rigid_pairs.append((i, j, d0))

        # 3. Filter nonbonded pairs
        if hasattr(self, "_use_sparse"):
            if self._use_sparse:
                from geoinit.core.geometry import distance
                self.nonbonded_pairs = [
                    (i, j) for (i, j) in topo.nonbonded_pairs
                    if distance(self._initial_coords_for_sparse, i, j) < 6.0
                ]
            else:
                self.nonbonded_pairs = topo.nonbonded_pairs

    # ------------------------------------------------------------------ #
    #  Vectorised engine (opt-in; default keeps the scalar path)
    # ------------------------------------------------------------------ #

    @property
    def uses_engine(self) -> bool:
        """Whether energy/gradient route through the vectorised engine."""
        return self.engine_mode != "scalar" or self.profile == "v2"

    def _get_engine(self):
        """Lazily build (and cache) the vectorised :class:`EnergyEngine`."""
        if self._engine_obj is None:
            from geoinit.energy.engine import EnergyEngine, build_tables

            tables = build_tables(
                symbols=self.symbols,
                topology=self.topology,
                nonbonded_pairs=self.nonbonded_pairs,
                rigid_pairs=self.rigid_pairs,
                weights=self.weights,
                sigma=self.sigma,
                charges=self.charges,
                anchor_coords=self.anchor_coords,
                k_anchor=self.k_anchor,
                clash_mode=self.clash_mode,
                profile=self.profile,
            )
            backend = "numpy" if self.engine_mode == "scalar" else self.engine_mode
            self._engine_obj = EnergyEngine(tables, backend=backend, dtype=self.dtype)
        return self._engine_obj

    @property
    def backend_name(self) -> str:
        """Human-readable name of the active compute backend (engine paths only)."""
        return self._get_engine().backend_name if self.uses_engine else "scalar:cpu"

    # ------------------------------------------------------------------ #
    #  Energy evaluation
    # ------------------------------------------------------------------ #

    def energy(self, coords: np.ndarray) -> float:
        """Compute total GeoInit energy for the given coordinates.

        Parameters
        ----------
        coords : np.ndarray, shape (N, 3)
            Current Cartesian coordinates in Å.

        Returns
        -------
        float
            Total weighted energy.
        """
        if self.uses_engine:
            return float(self._get_engine().energy(coords))
        components = self.energy_components(coords)
        return sum(components.values())

    def energy_flat(self, x: np.ndarray) -> float:
        """Energy evaluated from a flattened coordinate vector.

        Parameters
        ----------
        x : np.ndarray, shape (3N,)
            Flattened Cartesian coordinates.

        Returns
        -------
        float
            Total weighted energy.
        """
        coords = x.reshape(self.n_atoms, 3)
        return self.energy(coords)

    def rigid_energy(self, coords: np.ndarray) -> float:
        """Compute the rigid subgraph preservation energy using pairwise restraints."""
        if not hasattr(self, "rigid_pairs") or not self.rigid_pairs:
            return 0.0

        energy = 0.0
        k = 1.0 / (self.sigma * self.sigma)
        for i, j, d0 in self.rigid_pairs:
            diff = coords[i] - coords[j]
            d = float(np.linalg.norm(diff))
            dev = d - d0
            energy += k * dev * dev
        return energy

    def rigid_gradient(self, coords: np.ndarray) -> np.ndarray:
        """Compute the analytical gradient of the rigid subgraph preservation energy."""
        grad = np.zeros_like(coords, dtype=np.float64)
        if not hasattr(self, "rigid_pairs") or not self.rigid_pairs:
            return grad

        k = 1.0 / (self.sigma * self.sigma)
        for i, j, d0 in self.rigid_pairs:
            diff = coords[i] - coords[j]
            d = float(np.linalg.norm(diff))
            if d < 1e-12:
                continue
            factor = 2.0 * k * (d - d0) / d
            g_i = factor * diff
            grad[i] += g_i
            grad[j] -= g_i

        return grad

    def energy_components(self, coords: np.ndarray) -> dict[str, float]:
        """Return individual *weighted* energy‑term values."""
        w = self.weights
        topo = self.topology
        sym = self.symbols

        e_bond = bond_energy(sym, coords, topo.bonds, sigma=self.sigma)
        e_angle = angle_energy(sym, coords, topo.angles, topo.angle_targets)
        e_clash = clash_energy(sym, coords, self.nonbonded_pairs, clash_mode=self.clash_mode)
        e_disp = dispersion_energy(sym, coords, self.nonbonded_pairs)
        e_coul = coulomb_energy(
            sym, coords, self.nonbonded_pairs, charges=self.charges
        )
        e_rigid = self.rigid_energy(coords)

        e_anchor = 0.0
        if self.k_anchor > 0.0 and self.anchor_coords is not None:
            e_anchor = self.k_anchor * np.sum((coords - self.anchor_coords) ** 2)

        return {
            "bond": w.get("bond", 10.0) * e_bond,
            "angle": w.get("angle", 5.0) * e_angle,
            "clash": w.get("clash", 1.0) * e_clash,
            "disp": w.get("disp", 0.1) * e_disp,
            "coul": w.get("coul", 0.0) * e_coul,
            "rigid": w.get("rigid", 10.0) * e_rigid,
            "anchor": e_anchor,
        }

    # ------------------------------------------------------------------ #
    #  Gradient (numerical, central differences)
    # ------------------------------------------------------------------ #

    def numerical_gradient(
        self,
        coords: np.ndarray,
        h: float = 1e-5,
    ) -> np.ndarray:
        """Compute the energy gradient via central finite differences.

        Parameters
        ----------
        coords : np.ndarray, shape (N, 3)
            Current Cartesian coordinates in Å.
        h : float, optional
            Step size for finite differences (default 1 × 10⁻⁵ Å).

        Returns
        -------
        np.ndarray, shape (N, 3)
            Gradient ∂Φ/∂x for each Cartesian component.
        """
        grad = np.zeros_like(coords, dtype=np.float64)
        coords_work = coords.astype(np.float64).copy()
        inv_2h = 1.0 / (2.0 * h)

        for i in range(self.n_atoms):
            for a in range(3):
                coords_work[i, a] += h
                e_plus = self.energy(coords_work)
                coords_work[i, a] -= 2.0 * h
                e_minus = self.energy(coords_work)
                coords_work[i, a] += h  # restore
                grad[i, a] = (e_plus - e_minus) * inv_2h

        return grad

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        """Compute the total analytical gradient of the GeoInit functional.

        Parameters
        ----------
        coords : np.ndarray, shape (N, 3)
            Current Cartesian coordinates in Å.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Gradient matrix.
        """
        if self.uses_engine:
            return self._get_engine().gradient(coords)

        w = self.weights
        topo = self.topology
        sym = self.symbols

        g_bond = bond_gradient(sym, coords, topo.bonds, sigma=self.sigma)
        g_angle = angle_gradient(sym, coords, topo.angles, topo.angle_targets)
        g_clash = clash_gradient(sym, coords, self.nonbonded_pairs, clash_mode=self.clash_mode)
        g_disp = dispersion_gradient(sym, coords, self.nonbonded_pairs)
        g_coul = coulomb_gradient(
            sym, coords, self.nonbonded_pairs, charges=self.charges
        )
        g_rigid = self.rigid_gradient(coords)

        g_anchor = np.zeros_like(coords)
        if self.k_anchor > 0.0 and self.anchor_coords is not None:
            g_anchor = 2.0 * self.k_anchor * (coords - self.anchor_coords)

        return (
            w.get("bond", 10.0) * g_bond
            + w.get("angle", 5.0) * g_angle
            + w.get("clash", 1.0) * g_clash
            + w.get("disp", 0.1) * g_disp
            + w.get("coul", 0.0) * g_coul
            + w.get("rigid", 10.0) * g_rigid
            + g_anchor
        )

    def gradient_flat(self, x: np.ndarray) -> np.ndarray:
        """Gradient evaluated from a flattened coordinate vector.

        Designed for direct use as the ``jac`` argument of
        :func:`scipy.optimize.minimize`.

        Parameters
        ----------
        x : np.ndarray, shape (3N,)
            Flattened Cartesian coordinates.

        Returns
        -------
        np.ndarray, shape (3N,)
            Flattened gradient vector.
        """
        coords = x.reshape(self.n_atoms, 3)
        return self.gradient(coords).ravel()
