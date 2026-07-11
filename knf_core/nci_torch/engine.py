from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .grid import flatten_grid_points
from .types import GridSpec, NCIFields, PreparedBasisFunction, Wavefunction


RDG_PREFAC = float(1.0 / (2.0 * (3.0 * torch.pi**2) ** (1.0 / 3.0)))


@dataclass
class NCIConfig:
    device: str = "auto"
    dtype: str = "float32"
    batch_size: int = 250000
    rho_floor: float = 1e-12
    eig_batch_size: int = 200000


def _resolve_device(device: str) -> torch.device:
    normalized = (device or "auto").strip().lower()
    if normalized in {"auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but no CUDA-capable GPU is available.")
    return torch.device(normalized)


def _resolve_dtype(dtype: str) -> torch.dtype:
    normalized = (dtype or "float32").strip().lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float64", "fp64"}:
        return torch.float64
    raise ValueError(f"Unsupported dtype '{dtype}'. Use float32 or float64.")


def _prepare_basis(
    wavefunction: Wavefunction,
    device: torch.device,
    dtype: torch.dtype,
) -> List[PreparedBasisFunction]:
    prepared: List[PreparedBasisFunction] = []
    centers = torch.as_tensor(wavefunction.atoms_bohr, device=device, dtype=dtype)
    for bf in wavefunction.basis_functions:
        prepared.append(
            PreparedBasisFunction(
                center=centers[bf.center_index],
                powers=bf.powers,
                exponents=torch.as_tensor(bf.exponents, device=device, dtype=dtype),
                coefficients=torch.as_tensor(
                    bf.normalized_coefficients, device=device, dtype=dtype
                ),
            )
        )
    return prepared


@dataclass
class _BasisGroup:
    """All basis functions sharing one (lx, ly, lz) power triple.

    Grouping lets one basis function's worth of tensor ops cover every basis
    function in the group at once (see ``_group_prepared_basis``).
    """

    powers: Tuple[int, int, int]
    indices: List[int]
    centers: torch.Tensor       # (m, 3)
    exponents: torch.Tensor     # (m, max_primitives), zero-padded
    coefficients: torch.Tensor  # (m, max_primitives), zero-padded


def _group_prepared_basis(
    prepared_basis: List[PreparedBasisFunction],
) -> List[_BasisGroup]:
    """Bucket basis functions by angular-momentum powers, once per molecule.

    A molden basis expands each contracted shell into one basis function per
    cartesian power triple (see ``molden._cartesian_powers``): an s shell
    contributes 1, p contributes 3, d contributes 6, and so on. Across a whole
    molecule there are only a handful of distinct triples (grouping is keyed
    on the triple alone, not the shell, so e.g. every "(1, 0, 0)" p_x function
    on every atom lands in one group) even when there are hundreds of basis
    functions, because typical GFN-xTB basis sets stay at s/p/d. Padding
    exponents/coefficients with zero is exact: exp(-r^2 * 0) * 0 == 0, so
    padded primitive slots contribute nothing regardless of r.
    """
    buckets: dict[Tuple[int, int, int], List[int]] = {}
    for idx, bf in enumerate(prepared_basis):
        buckets.setdefault(bf.powers, []).append(idx)

    groups: List[_BasisGroup] = []
    for powers, indices in buckets.items():
        group_basis = [prepared_basis[i] for i in indices]
        device = group_basis[0].center.device
        dtype = group_basis[0].center.dtype
        max_prim = max(int(bf.exponents.shape[0]) for bf in group_basis)

        centers = torch.stack([bf.center for bf in group_basis], dim=0)
        exponents = torch.zeros((len(group_basis), max_prim), device=device, dtype=dtype)
        coefficients = torch.zeros((len(group_basis), max_prim), device=device, dtype=dtype)
        for row, bf in enumerate(group_basis):
            n_prim = bf.exponents.shape[0]
            exponents[row, :n_prim] = bf.exponents
            coefficients[row, :n_prim] = bf.coefficients

        groups.append(
            _BasisGroup(
                powers=powers,
                indices=indices,
                centers=centers,
                exponents=exponents,
                coefficients=coefficients,
            )
        )
    return groups


def _evaluate_basis_chunk(
    points: torch.Tensor, basis_groups: List[_BasisGroup], n_basis: int
) -> torch.Tensor:
    out = torch.empty((points.shape[0], n_basis), device=points.device, dtype=points.dtype)

    for group in basis_groups:
        lx, ly, lz = group.powers
        # (n_points, m_in_group) via broadcasting instead of one (n_points,)
        # column per basis function.
        dx = points[:, 0:1] - group.centers[None, :, 0]
        dy = points[:, 1:2] - group.centers[None, :, 1]
        dz = points[:, 2:3] - group.centers[None, :, 2]
        r2 = dx * dx + dy * dy + dz * dz

        poly = torch.ones_like(r2)
        if lx:
            poly = poly * (dx**lx)
        if ly:
            poly = poly * (dy**ly)
        if lz:
            poly = poly * (dz**lz)

        # (n_points, m_in_group, max_primitives) -> summed over primitives.
        prim = (
            torch.exp(-r2[:, :, None] * group.exponents[None, :, :])
            * group.coefficients[None, :, :]
        )
        out[:, group.indices] = poly * prim.sum(dim=2)

    return out


def compute_density(
    wavefunction: Wavefunction,
    grid: GridSpec,
    config: NCIConfig,
) -> Tuple[torch.Tensor, torch.device]:
    device = _resolve_device(config.device)
    dtype = _resolve_dtype(config.dtype)

    points = flatten_grid_points(grid, device=device, dtype=dtype)
    prepared_basis = _prepare_basis(wavefunction, device=device, dtype=dtype)
    basis_groups = _group_prepared_basis(prepared_basis)
    n_basis = len(prepared_basis)

    coeff = torch.as_tensor(wavefunction.mo_coefficients, device=device, dtype=dtype)
    occ = torch.as_tensor(wavefunction.occupations, device=device, dtype=dtype)

    rho = torch.empty(points.shape[0], device=device, dtype=dtype)
    batch_size = max(1, int(config.batch_size))
    for start in range(0, points.shape[0], batch_size):
        end = min(points.shape[0], start + batch_size)
        basis_values = _evaluate_basis_chunk(points[start:end], basis_groups, n_basis)
        psi = basis_values @ coeff
        rho[start:end] = torch.sum((psi * psi) * occ[None, :], dim=1)

    rho = torch.clamp(torch.nan_to_num(rho, nan=0.0, posinf=1e20, neginf=0.0), min=0.0)
    nx, ny, nz = grid.shape
    return rho.reshape(nx, ny, nz), device


def _compute_lambda2_batched(
    hessian: torch.Tensor,
    eig_batch_size: int,
) -> torch.Tensor:
    hflat = hessian.reshape(-1, 3, 3)
    n = hflat.shape[0]
    out = torch.empty((n,), device=hessian.device, dtype=hessian.dtype)
    batch = max(1, int(eig_batch_size))

    for start in range(0, n, batch):
        end = min(n, start + batch)
        chunk = hflat[start:end]
        chunk = torch.nan_to_num(chunk, nan=0.0, posinf=1e20, neginf=-1e20)
        chunk = 0.5 * (chunk + chunk.transpose(-1, -2))
        try:
            eigvals = torch.linalg.eigvalsh(chunk)
        except RuntimeError as err:
            # CUDA batched eig can fail on pathological slices; fallback keeps run alive.
            if chunk.device.type == "cuda":
                eigvals = torch.linalg.eigvalsh(chunk.cpu()).to(chunk.device)
            else:
                raise err
        out[start:end] = torch.nan_to_num(eigvals[:, 1], nan=0.0, posinf=0.0, neginf=0.0)

    return out.reshape(hessian.shape[:-2])


def _first_derivative(u: torch.Tensor, h: float, dim: int) -> torch.Tensor:
    out = torch.empty_like(u)

    interior = [slice(None)] * 3
    upper = [slice(None)] * 3
    lower = [slice(None)] * 3
    interior[dim] = slice(1, -1)
    upper[dim] = slice(2, None)
    lower[dim] = slice(None, -2)
    out[tuple(interior)] = (u[tuple(upper)] - u[tuple(lower)]) / (2.0 * h)

    edge0 = [slice(None)] * 3
    edge1 = [slice(None)] * 3
    edge0[dim] = 0
    edge1[dim] = 1
    out[tuple(edge0)] = (u[tuple(edge1)] - u[tuple(edge0)]) / h

    edge_last = [slice(None)] * 3
    edge_prev = [slice(None)] * 3
    edge_last[dim] = -1
    edge_prev[dim] = -2
    out[tuple(edge_last)] = (u[tuple(edge_last)] - u[tuple(edge_prev)]) / h
    return out


def _second_derivative(u: torch.Tensor, h: float, dim: int) -> torch.Tensor:
    out = torch.zeros_like(u)
    interior = [slice(None)] * 3
    upper = [slice(None)] * 3
    center = [slice(None)] * 3
    lower = [slice(None)] * 3
    interior[dim] = slice(1, -1)
    upper[dim] = slice(2, None)
    center[dim] = slice(1, -1)
    lower[dim] = slice(None, -2)
    out[tuple(interior)] = (
        u[tuple(upper)] - 2.0 * u[tuple(center)] + u[tuple(lower)]
    ) / (h * h)

    edge0 = [slice(None)] * 3
    edge1 = [slice(None)] * 3
    edge0[dim] = 0
    edge1[dim] = 1
    out[tuple(edge0)] = out[tuple(edge1)]

    edge_last = [slice(None)] * 3
    edge_prev = [slice(None)] * 3
    edge_last[dim] = -1
    edge_prev[dim] = -2
    out[tuple(edge_last)] = out[tuple(edge_prev)]
    return out


def _cross_second_derivative(
    u: torch.Tensor,
    h_a: float,
    h_b: float,
    dim_a: int,
    dim_b: int,
) -> torch.Tensor:
    p_ap_b = torch.roll(torch.roll(u, shifts=-1, dims=dim_a), shifts=-1, dims=dim_b)
    p_am_b = torch.roll(torch.roll(u, shifts=-1, dims=dim_a), shifts=1, dims=dim_b)
    m_ap_b = torch.roll(torch.roll(u, shifts=1, dims=dim_a), shifts=-1, dims=dim_b)
    m_am_b = torch.roll(torch.roll(u, shifts=1, dims=dim_a), shifts=1, dims=dim_b)
    out = (p_ap_b - p_am_b - m_ap_b + m_am_b) / (4.0 * h_a * h_b)

    edge0 = [slice(None)] * 3
    edge_last = [slice(None)] * 3
    edge0[dim_a] = 0
    edge_last[dim_a] = -1
    out[tuple(edge0)] = 0.0
    out[tuple(edge_last)] = 0.0
    edge0 = [slice(None)] * 3
    edge_last = [slice(None)] * 3
    edge0[dim_b] = 0
    edge_last[dim_b] = -1
    out[tuple(edge0)] = 0.0
    out[tuple(edge_last)] = 0.0
    return out


def compute_nci_fields(
    rho: torch.Tensor,
    spacing_bohr: float,
    rho_floor: float = 1e-12,
    eig_batch_size: int = 200000,
) -> NCIFields:
    rho = torch.clamp(torch.nan_to_num(rho, nan=0.0, posinf=1e20, neginf=0.0), min=0.0)
    gx = _first_derivative(rho, spacing_bohr, dim=0)
    gy = _first_derivative(rho, spacing_bohr, dim=1)
    gz = _first_derivative(rho, spacing_bohr, dim=2)
    grad_mag = torch.sqrt(gx * gx + gy * gy + gz * gz)

    hxx = _second_derivative(rho, spacing_bohr, dim=0)
    hyy = _second_derivative(rho, spacing_bohr, dim=1)
    hzz = _second_derivative(rho, spacing_bohr, dim=2)
    hxy = _cross_second_derivative(rho, spacing_bohr, spacing_bohr, dim_a=0, dim_b=1)
    hxz = _cross_second_derivative(rho, spacing_bohr, spacing_bohr, dim_a=0, dim_b=2)
    hyz = _cross_second_derivative(rho, spacing_bohr, spacing_bohr, dim_a=1, dim_b=2)

    hessian = torch.empty((*rho.shape, 3, 3), device=rho.device, dtype=rho.dtype)
    hessian[..., 0, 0] = hxx
    hessian[..., 1, 1] = hyy
    hessian[..., 2, 2] = hzz
    hessian[..., 0, 1] = hxy
    hessian[..., 1, 0] = hxy
    hessian[..., 0, 2] = hxz
    hessian[..., 2, 0] = hxz
    hessian[..., 1, 2] = hyz
    hessian[..., 2, 1] = hyz

    lambda2 = _compute_lambda2_batched(hessian, eig_batch_size=eig_batch_size)

    safe_rho = torch.clamp(torch.abs(rho), min=rho_floor)
    rdg = rho.new_tensor(RDG_PREFAC) * grad_mag / safe_rho.pow(4.0 / 3.0)
    rdg = torch.where(torch.abs(rho) < rho_floor, torch.zeros_like(rdg), rdg)
    sign_lambda2_rho = torch.sign(lambda2) * rho

    return NCIFields(rho=rho, rdg=rdg, sign_lambda2_rho=sign_lambda2_rho)


def run_nci_engine(
    wavefunction: Wavefunction,
    grid: GridSpec,
    config: Optional[NCIConfig] = None,
) -> Tuple[NCIFields, torch.device]:
    cfg = config or NCIConfig()
    rho, device = compute_density(wavefunction=wavefunction, grid=grid, config=cfg)
    fields = compute_nci_fields(
        rho=rho,
        spacing_bohr=grid.spacing_bohr,
        rho_floor=cfg.rho_floor,
        eig_batch_size=cfg.eig_batch_size,
    )
    return fields, device
