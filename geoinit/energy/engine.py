"""Vectorised, backend-agnostic energy/gradient engine for GeoInit.

This module replaces the per-pair Python ``for``-loops in
:mod:`geoinit.energy.bond`, :mod:`~geoinit.energy.angle`, and
:mod:`~geoinit.energy.nonbonded` with a single set of array kernels that run
unchanged on NumPy (CPU, the always-available fallback) or PyTorch (CPU
multi-thread / CUDA GPU).

Two physics profiles are supported through the *same* kernels:

* ``"v1"`` — reproduces the legacy GeoInit-V1 functional **bit-for-bit** (to
  floating-point round-off).  This lets the frozen V0.8 selector run on the new
  engine with no change in results.
* ``"v2"`` — improved molecular-mechanics forms (harmonic-in-angle bending with
  a Urey–Bradley-style 1-3 term, damped dispersion without the dimensionful
  regulariser) that produce better warm-starts.  Opt-in only.

Correctness strategy
---------------------
The energy is written once (``_total_energy``) and used both for the NumPy
analytic gradient (``_grad_numpy``) and the Torch autograd path.  The test suite
cross-checks NumPy-analytic ↔ scalar ↔ finite-difference ↔ Torch-autograd, so a
mistake in any single path is caught.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geoinit.compute.backends import Backend, NumpyBackend, select_backend
from geoinit.core.params import get_c6_pair, get_vdw_radius

# Angle / dispersion / clash mode tags
ANGLE_COS = "cos"           # legacy:  (cosθ − cosθ0)²
ANGLE_HARMONIC = "harmonic"  # v2:      kθ (θ − θ0)²
DISP_V1 = "v1"               # legacy:  −C6/(r⁶+δ)·r⁶/(r⁶+s⁶)
DISP_BJ = "bj"               # v2:      −C6/(r⁶ + s⁶)        (no δ regulariser)
CLASH_COMPACT = "compact"
CLASH_EXP = "exp"

_EPS = 1e-12


def _bond_order_k_factor(order: float) -> float:
    """Conservative bond-order → stiffness multiplier for the V2 profile.

    Grounded in spectroscopic force constants (mdyn/Å): a double bond is roughly
    2× a single, a triple ~3.5×, aromatic ~1.4×.  Capped conservatively to avoid
    ill-conditioning the optimiser while still keeping multiple bonds rigid.
    """
    if order >= 2.5:
        return 2.0
    if order >= 1.75:
        return 1.6
    if order >= 1.25:
        return 1.3
    return 1.0


# --------------------------------------------------------------------------- #
#  Interaction tables (built once, frozen with the topology)
# --------------------------------------------------------------------------- #

@dataclass
class InteractionTables:
    """Flat per-interaction parameter arrays consumed by the kernels.

    All index arrays are ``int64``; all parameter arrays are ``float64``.  The
    arrays are stored as NumPy on the host and uploaded to the backend lazily.
    """

    n_atoms: int

    # bonds (also reused for rigid pairwise restraints)
    bi: np.ndarray
    bj: np.ndarray
    b_r0: np.ndarray
    b_k: np.ndarray

    # angles
    ai: np.ndarray
    aj: np.ndarray
    ak: np.ndarray
    a_theta0: np.ndarray      # radians
    a_cos0: np.ndarray        # cos(theta0), precomputed
    a_k: np.ndarray           # per-angle stiffness (1.0 for v1)

    # nonbonded
    ni: np.ndarray
    nj: np.ndarray
    nb_s: np.ndarray          # 0.75 * (vdw_i + vdw_j)
    nb_c6: np.ndarray
    nb_qq: np.ndarray | None  # q_i * q_j  (None ⇒ no Coulomb)

    # rigid pairwise restraints
    ri: np.ndarray
    rj: np.ndarray
    r_d0: np.ndarray
    r_k: np.ndarray

    # urey-bradley 1-3 restraints (v2 only; empty otherwise)
    ui: np.ndarray
    uj: np.ndarray
    u_d0: np.ndarray
    u_k: np.ndarray

    # anchor (full-array harmonic tether)
    anchor: np.ndarray | None
    k_anchor: float

    # weights
    w_bond: float
    w_angle: float
    w_clash: float
    w_disp: float
    w_coul: float
    w_rigid: float

    # modes / constants
    angle_mode: str = ANGLE_COS
    disp_mode: str = DISP_V1
    clash_mode: str = CLASH_COMPACT
    clash_k: float = 100.0
    clash_alpha: float = 3.0
    disp_delta: float = 1.0
    coul_eps: float = 0.1

    @property
    def work_size(self) -> int:
        """Total interaction count — used to size-gate GPU offload."""
        return (
            len(self.bi) + len(self.ai) + len(self.ni)
            + len(self.ri) + len(self.ui)
        )


def _as_int(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.int64)


def _as_float(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


def build_tables(
    symbols: list[str],
    topology,
    nonbonded_pairs: list[tuple[int, int]],
    rigid_pairs: list[tuple[int, int, float]],
    weights: dict,
    sigma: float,
    charges: np.ndarray | None = None,
    anchor_coords: np.ndarray | None = None,
    k_anchor: float = 0.0,
    clash_mode: str = CLASH_COMPACT,
    profile: str = "v1",
    urey_bradley_pairs: list[tuple[int, int, float, float]] | None = None,
    angle_k_overrides: dict | None = None,
) -> InteractionTables:
    """Assemble :class:`InteractionTables` from a frozen topology.

    Parameters mirror :class:`geoinit.energy.functional.GeoInitFunctional`.  The
    ``profile`` selects the legacy (``"v1"``) or improved (``"v2"``) physics
    forms; ``"v1"`` reproduces the existing functional exactly.
    """
    n = len(symbols)
    is_v2 = profile == "v2"

    # -- bonds ------------------------------------------------------------- #
    # V2 scales the harmonic force constant by bond order (multiple bonds are
    # physically much stiffer), which keeps double/triple/aromatic bonds closer
    # to their reference length during relaxation and so reduces the chance of a
    # multiple-bond-damage guard rejection.
    bi, bj, b_r0, b_k = [], [], [], []
    for b in topology.bonds:
        i, j = b.i, b.j
        r0 = getattr(b, "r0", None)
        k = getattr(b, "k", None)
        if r0 is None or k is None:
            from geoinit.core.params import get_covalent_radius

            r0 = get_covalent_radius(symbols[i]) + get_covalent_radius(symbols[j])
            k = 1.0 / (sigma * sigma)
        if is_v2:
            k = k * _bond_order_k_factor(getattr(b, "order", 1.0))
        bi.append(i); bj.append(j); b_r0.append(r0); b_k.append(k)

    # -- angles ------------------------------------------------------------ #
    # V2 uses a harmonic-in-angle bend, k_θ (θ − θ0)², with the per-angle
    # stiffness set to the *curvature-matched* value k_θ = sin²(θ0).  This makes
    # the local curvature at the minimum identical to the legacy
    # (cosθ − cosθ0)² form (whose curvature is 2 sin²θ0), so near equilibrium V2
    # reproduces V1, but far from equilibrium the harmonic form gives a
    # monotonic restoring force instead of the cosine form's vanishing/periodic
    # behaviour — a strictly better global basin without shifting the minimum.
    ai, aj, ak, a_theta0, a_k = [], [], [], [], []
    targets = topology.angle_targets
    angle_k_overrides = angle_k_overrides or {}
    for (i, j, k) in topology.angles:
        theta0 = targets[(i, j, k)]
        ai.append(i); aj.append(j); ak.append(k)
        a_theta0.append(theta0)
        if (i, j, k) in angle_k_overrides:
            a_k.append(float(angle_k_overrides[(i, j, k)]))
        elif is_v2:
            a_k.append(max(float(np.sin(theta0) ** 2), 0.10))
        else:
            a_k.append(1.0)

    # -- nonbonded --------------------------------------------------------- #
    ni, nj, nb_s, nb_c6 = [], [], [], []
    for (i, j) in nonbonded_pairs:
        ni.append(i); nj.append(j)
        nb_s.append(0.75 * (get_vdw_radius(symbols[i]) + get_vdw_radius(symbols[j])))
        nb_c6.append(get_c6_pair(symbols[i], symbols[j]))
    if charges is not None and len(nonbonded_pairs) > 0:
        ch = np.asarray(charges, dtype=np.float64)
        nb_qq = np.array([ch[i] * ch[j] for (i, j) in nonbonded_pairs], dtype=np.float64)
    else:
        nb_qq = None

    # -- rigid pairwise restraints ----------------------------------------- #
    ri, rj, r_d0 = [], [], []
    for (i, j, d0) in rigid_pairs:
        ri.append(i); rj.append(j); r_d0.append(d0)
    r_k = np.full(len(ri), 1.0 / (sigma * sigma), dtype=np.float64)

    # -- urey-bradley 1-3 (v2) --------------------------------------------- #
    ui, uj, u_d0, u_k = [], [], [], []
    if urey_bradley_pairs:
        for (i, j, d0, k) in urey_bradley_pairs:
            ui.append(i); uj.append(j); u_d0.append(d0); u_k.append(k)

    a_theta0_arr = _as_float(a_theta0)
    return InteractionTables(
        n_atoms=n,
        bi=_as_int(bi), bj=_as_int(bj), b_r0=_as_float(b_r0), b_k=_as_float(b_k),
        ai=_as_int(ai), aj=_as_int(aj), ak=_as_int(ak),
        a_theta0=a_theta0_arr, a_cos0=np.cos(a_theta0_arr), a_k=_as_float(a_k),
        ni=_as_int(ni), nj=_as_int(nj), nb_s=_as_float(nb_s), nb_c6=_as_float(nb_c6),
        nb_qq=nb_qq,
        ri=_as_int(ri), rj=_as_int(rj), r_d0=_as_float(r_d0), r_k=r_k,
        ui=_as_int(ui), uj=_as_int(uj), u_d0=_as_float(u_d0), u_k=_as_float(u_k),
        anchor=(np.asarray(anchor_coords, dtype=np.float64).copy()
                if anchor_coords is not None else None),
        k_anchor=float(k_anchor),
        w_bond=float(weights.get("bond", 10.0)),
        w_angle=float(weights.get("angle", 5.0)),
        w_clash=float(weights.get("clash", 1.0)),
        w_disp=float(weights.get("disp", 0.1)),
        w_coul=float(weights.get("coul", 0.0)),
        w_rigid=float(weights.get("rigid", 10.0)),
        angle_mode=ANGLE_HARMONIC if profile == "v2" else ANGLE_COS,
        disp_mode=DISP_BJ if profile == "v2" else DISP_V1,
        clash_mode=clash_mode,
    )


# --------------------------------------------------------------------------- #
#  Backend-side parameter cache (uploads tables to device once)
# --------------------------------------------------------------------------- #

class _DeviceTables:
    """Holds backend-resident copies of the index/parameter arrays."""

    def __init__(self, T: InteractionTables, B: Backend) -> None:
        self.T = T
        self.B = B
        f = lambda a: B.asarray(a) if a is not None and len(a) else B.asarray(np.zeros(0))
        idx = getattr(B, "as_index", None)
        ix = (lambda a: idx(a)) if idx is not None else (lambda a: B.asarray(a))

        self.bi, self.bj = ix(T.bi), ix(T.bj)
        self.b_r0, self.b_k = f(T.b_r0), f(T.b_k)
        self.ai, self.aj, self.ak = ix(T.ai), ix(T.aj), ix(T.ak)
        self.a_cos0, self.a_theta0, self.a_k = f(T.a_cos0), f(T.a_theta0), f(T.a_k)
        self.ni, self.nj = ix(T.ni), ix(T.nj)
        self.nb_s, self.nb_c6 = f(T.nb_s), f(T.nb_c6)
        self.nb_qq = f(T.nb_qq) if T.nb_qq is not None else None
        self.ri, self.rj = ix(T.ri), ix(T.rj)
        self.r_d0, self.r_k = f(T.r_d0), f(T.r_k)
        self.ui, self.uj = ix(T.ui), ix(T.uj)
        self.u_d0, self.u_k = f(T.u_d0), f(T.u_k)
        self.anchor = B.asarray(T.anchor) if T.anchor is not None else None


# --------------------------------------------------------------------------- #
#  Backend-generic energy kernel (used for NumPy energy AND Torch autograd)
# --------------------------------------------------------------------------- #

def _acos_clamped(B: Backend, cos_t):
    """Numerically safe arccos for both backends."""
    c = B.clip(cos_t, -1.0 + 1e-12, 1.0 - 1e-12)
    xp = np  # numpy path
    try:
        import torch  # type: ignore

        if "torch" in type(c).__module__:
            return torch.arccos(c)
    except Exception:
        pass
    return xp.arccos(c)


def _total_energy(X, D: _DeviceTables, B: Backend):
    """Total weighted GeoInit energy as a backend scalar (differentiable on Torch)."""
    T = D.T
    E = B.asarray(0.0)

    # --- bonds --------------------------------------------------------- #
    if len(T.bi):
        d = X[D.bj] - X[D.bi]
        r = B.norm_rows(d)
        E = E + T.w_bond * B.sum(D.b_k * (r - D.b_r0) ** 2)

    # --- angles -------------------------------------------------------- #
    if len(T.ai):
        v1 = X[D.ai] - X[D.aj]
        v2 = X[D.ak] - X[D.aj]
        n1 = B.norm_rows(v1)
        n2 = B.norm_rows(v2)
        cos_t = B.clip(B.sum(v1 * v2, axis=-1) / (n1 * n2), -1.0, 1.0)
        if T.angle_mode == ANGLE_HARMONIC:
            theta = _acos_clamped(B, cos_t)
            E = E + T.w_angle * B.sum(D.a_k * (theta - D.a_theta0) ** 2)
        else:
            E = E + T.w_angle * B.sum(D.a_k * (cos_t - D.a_cos0) ** 2)

    # --- nonbonded (clash + dispersion + coulomb share the pair list) -- #
    if len(T.ni):
        d = X[D.ni] - X[D.nj]
        r = B.norm_rows(d)
        # clash
        if T.clash_mode == CLASH_EXP:
            E = E + T.w_clash * B.sum(_exp(B, -T.clash_alpha * (r - D.nb_s)))
        else:
            term = B.clip(1.0 - r / D.nb_s, 0.0, None)
            E = E + T.w_clash * T.clash_k * B.sum(term ** 4)
        # dispersion
        r6 = r ** 6
        s6 = D.nb_s ** 6
        if T.disp_mode == DISP_BJ:
            E = E - T.w_disp * B.sum(D.nb_c6 / (r6 + s6))
        else:
            E = E - T.w_disp * B.sum(D.nb_c6 / (r6 + T.disp_delta) * (r6 / (r6 + s6)))
        # coulomb
        if D.nb_qq is not None and T.w_coul != 0.0:
            E = E + T.w_coul * B.sum(D.nb_qq / B.sqrt(r * r + T.coul_eps ** 2))

    # --- rigid pairwise restraints ------------------------------------ #
    if len(T.ri):
        d = X[D.rj] - X[D.ri]
        r = B.norm_rows(d)
        E = E + T.w_rigid * B.sum(D.r_k * (r - D.r_d0) ** 2)

    # --- urey-bradley 1-3 (v2) ---------------------------------------- #
    if len(T.ui):
        d = X[D.uj] - X[D.ui]
        r = B.norm_rows(d)
        E = E + B.sum(D.u_k * (r - D.u_d0) ** 2)

    # --- anchor -------------------------------------------------------- #
    if D.anchor is not None and T.k_anchor > 0.0:
        E = E + T.k_anchor * B.sum((X - D.anchor) ** 2)

    return E


def _exp(B: Backend, x):
    try:
        import torch  # type: ignore

        if "torch" in type(x).__module__:
            return torch.exp(x)
    except Exception:
        pass
    return np.exp(x)


# --------------------------------------------------------------------------- #
#  NumPy analytic gradient (the validated fallback path)
# --------------------------------------------------------------------------- #

def _grad_numpy(X: np.ndarray, T: InteractionTables) -> tuple[float, np.ndarray]:
    """Return ``(energy, gradient)`` using vectorised analytic derivatives."""
    X = np.asarray(X, dtype=np.float64)
    grad = np.zeros((T.n_atoms, 3), dtype=np.float64)
    E = 0.0

    # --- bonds --------------------------------------------------------- #
    if len(T.bi):
        d = X[T.bi] - X[T.bj]                       # (M,3)  i − j  (legacy sign)
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        safe = r > _EPS
        dev = r - T.b_r0
        E += T.w_bond * float(np.sum(T.b_k * dev * dev))
        fac = np.where(safe, 2.0 * T.b_k * dev / np.where(safe, r, 1.0), 0.0)
        gi = (T.w_bond * fac)[:, None] * d
        np.add.at(grad, T.bi, gi)
        np.add.at(grad, T.bj, -gi)

    # --- angles -------------------------------------------------------- #
    if len(T.ai):
        v1 = X[T.ai] - X[T.aj]
        v2 = X[T.ak] - X[T.aj]
        n1 = np.sqrt(np.einsum("ij,ij->i", v1, v1))
        n2 = np.sqrt(np.einsum("ij,ij->i", v2, v2))
        ok = (n1 > _EPS) & (n2 > _EPS)
        n1s = np.where(ok, n1, 1.0)
        n2s = np.where(ok, n2, 1.0)
        u1 = v1 / n1s[:, None]
        u2 = v2 / n2s[:, None]
        cos_t = np.clip(np.einsum("ij,ij->i", u1, u2), -1.0, 1.0)
        g_i = (u2 - cos_t[:, None] * u1) / n1s[:, None]   # ∂cosθ/∂x_i
        g_k = (u1 - cos_t[:, None] * u2) / n2s[:, None]   # ∂cosθ/∂x_k
        if T.angle_mode == ANGLE_HARMONIC:
            theta = np.arccos(np.clip(cos_t, -1.0 + 1e-12, 1.0 - 1e-12))
            dev = theta - T.a_theta0
            E += T.w_angle * float(np.sum(T.a_k * dev * dev))
            sin_t = np.sqrt(np.clip(1.0 - cos_t * cos_t, 1e-24, None))
            # dE/dcosθ = 2 k dev · dθ/dcosθ = 2 k dev · (−1/sinθ)
            dEdcos = -2.0 * T.a_k * dev / sin_t
        else:
            dev = cos_t - T.a_cos0
            E += T.w_angle * float(np.sum(T.a_k * dev * dev))
            dEdcos = 2.0 * T.a_k * dev
        scale = np.where(ok, T.w_angle * dEdcos, 0.0)
        ci = scale[:, None] * g_i
        ck = scale[:, None] * g_k
        np.add.at(grad, T.ai, ci)
        np.add.at(grad, T.ak, ck)
        np.add.at(grad, T.aj, -(ci + ck))

    # --- nonbonded ----------------------------------------------------- #
    if len(T.ni):
        d = X[T.ni] - X[T.nj]                       # i − j
        r2 = np.einsum("ij,ij->i", d, d)
        r = np.sqrt(r2)
        safe = r > _EPS
        rs = np.where(safe, r, 1.0)

        # clash
        if T.clash_mode == CLASH_EXP:
            val = np.exp(-T.clash_alpha * (r - T.nb_s))
            E += T.w_clash * float(np.sum(val))
            fac = -T.clash_alpha * val / rs
        else:
            term = np.clip(1.0 - r / T.nb_s, 0.0, None)
            E += T.w_clash * T.clash_k * float(np.sum(term ** 4))
            fac = -4.0 * T.clash_k / T.nb_s * (term ** 3) / rs
        g = (T.w_clash * np.where(safe, fac, 0.0))[:, None] * d
        np.add.at(grad, T.ni, g)
        np.add.at(grad, T.nj, -g)

        # dispersion
        r6 = r2 ** 3
        s6 = T.nb_s ** 6
        if T.disp_mode == DISP_BJ:
            denom = r6 + s6
            E += -T.w_disp * float(np.sum(T.nb_c6 / denom))
            # E = −c6/(r⁶+s⁶);  dE/dr = c6·6r⁵/(r⁶+s⁶)²;  /r → 6 c6 r⁴/denom²
            fac = 6.0 * T.nb_c6 * (r2 ** 2) / (denom * denom)
        else:
            denom = (r6 + T.disp_delta) * (r6 + s6)
            E += -T.w_disp * float(np.sum(T.nb_c6 / (r6 + T.disp_delta) * (r6 / (r6 + s6))))
            num = T.disp_delta * s6 - r6 * r6
            fac = -6.0 * T.nb_c6 * (r2 ** 2) * num / (denom * denom)
        g = (T.w_disp * np.where(safe, fac, 0.0))[:, None] * d
        np.add.at(grad, T.ni, g)
        np.add.at(grad, T.nj, -g)

        # coulomb
        if T.nb_qq is not None and T.w_coul != 0.0:
            dist_term = r2 + T.coul_eps ** 2
            E += T.w_coul * float(np.sum(T.nb_qq / np.sqrt(dist_term)))
            fac = -T.nb_qq / (dist_term * np.sqrt(dist_term))
            g = (T.w_coul * fac)[:, None] * d
            np.add.at(grad, T.ni, g)
            np.add.at(grad, T.nj, -g)

    # --- rigid --------------------------------------------------------- #
    if len(T.ri):
        d = X[T.ri] - X[T.rj]
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        safe = r > _EPS
        dev = r - T.r_d0
        E += T.w_rigid * float(np.sum(T.r_k * dev * dev))
        fac = np.where(safe, 2.0 * T.r_k * dev / np.where(safe, r, 1.0), 0.0)
        g = (T.w_rigid * fac)[:, None] * d
        np.add.at(grad, T.ri, g)
        np.add.at(grad, T.rj, -g)

    # --- urey-bradley -------------------------------------------------- #
    if len(T.ui):
        d = X[T.ui] - X[T.uj]
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        safe = r > _EPS
        dev = r - T.u_d0
        E += float(np.sum(T.u_k * dev * dev))
        fac = np.where(safe, 2.0 * T.u_k * dev / np.where(safe, r, 1.0), 0.0)
        g = fac[:, None] * d
        np.add.at(grad, T.ui, g)
        np.add.at(grad, T.uj, -g)

    # --- anchor -------------------------------------------------------- #
    if T.anchor is not None and T.k_anchor > 0.0:
        diff = X - T.anchor
        E += T.k_anchor * float(np.sum(diff * diff))
        grad += 2.0 * T.k_anchor * diff

    return E, grad


# --------------------------------------------------------------------------- #
#  Public engine object
# --------------------------------------------------------------------------- #

class EnergyEngine:
    """Evaluate the GeoInit energy/gradient on a chosen backend.

    Parameters
    ----------
    tables : InteractionTables
        Pre-built interaction tables (see :func:`build_tables`).
    backend : str
        ``'auto'`` (size-aware GPU/CPU), ``'numpy'``, ``'torch'``/``'torch-cpu'``,
        or ``'cuda'``/``'gpu'``.  Unavailable backends degrade to NumPy.
    dtype :
        Optional float dtype override for the Torch backend (e.g. ``float32`` for
        faster GPU at reduced precision).  NumPy always uses ``float64``.
    """

    def __init__(self, tables: InteractionTables, backend: str = "auto", dtype=None) -> None:
        self.tables = tables
        self.B: Backend = select_backend(tables.work_size, prefer=backend, dtype=dtype)
        self._device_tables = _DeviceTables(tables, self.B)

    # -- introspection ------------------------------------------------------ #
    @property
    def backend_name(self) -> str:
        return f"{self.B.name}:{self.B.device}"

    # -- energy ------------------------------------------------------------- #
    def energy(self, coords: np.ndarray) -> float:
        if isinstance(self.B, NumpyBackend):
            E, _ = _grad_numpy(coords, self.tables)
            return float(E)
        X = self.B.asarray(coords)
        return self.B.item(_total_energy(X, self._device_tables, self.B))

    # -- gradient ----------------------------------------------------------- #
    def gradient(self, coords: np.ndarray) -> np.ndarray:
        return self.energy_and_grad(coords)[1]

    def energy_and_grad(self, coords: np.ndarray) -> tuple[float, np.ndarray]:
        """Return ``(energy, gradient)`` as Python ``float`` / NumPy ``(N,3)``."""
        if isinstance(self.B, NumpyBackend):
            E, g = _grad_numpy(coords, self.tables)
            return float(E), g
        return self._torch_energy_and_grad(coords)

    def _torch_energy_and_grad(self, coords: np.ndarray) -> tuple[float, np.ndarray]:
        import torch  # type: ignore

        X = self.B.asarray(coords)
        X = X.detach().clone().requires_grad_(True)
        E = _total_energy(X, self._device_tables, self.B)
        (g,) = torch.autograd.grad(E, X)
        return float(E.detach().to("cpu").item()), self.B.to_numpy(g)

    # -- flat helpers for scipy.optimize ----------------------------------- #
    def energy_flat(self, x: np.ndarray) -> float:
        return self.energy(x.reshape(self.tables.n_atoms, 3))

    def grad_flat(self, x: np.ndarray) -> np.ndarray:
        return self.energy_and_grad(x.reshape(self.tables.n_atoms, 3))[1].ravel()
