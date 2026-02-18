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


def _evaluate_basis_chunk(
    points: torch.Tensor, prepared_basis: List[PreparedBasisFunction]
) -> torch.Tensor:
    n_points = points.shape[0]
    n_basis = len(prepared_basis)
    out = torch.empty((n_points, n_basis), device=points.device, dtype=points.dtype)

    for col, bf in enumerate(prepared_basis):
        dx = points[:, 0] - bf.center[0]
        dy = points[:, 1] - bf.center[1]
        dz = points[:, 2] - bf.center[2]
        r2 = dx * dx + dy * dy + dz * dz

        lx, ly, lz = bf.powers
        poly = torch.ones_like(r2)
        if lx:
            poly = poly * (dx**lx)
        if ly:
            poly = poly * (dy**ly)
        if lz:
            poly = poly * (dz**lz)

        # Primitive contraction stays vectorized across all points.
        prim = torch.exp(-r2[:, None] * bf.exponents[None, :]) * bf.coefficients[None, :]
        out[:, col] = poly * prim.sum(dim=1)

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

    coeff = torch.as_tensor(wavefunction.mo_coefficients, device=device, dtype=dtype)
    occ = torch.as_tensor(wavefunction.occupations, device=device, dtype=dtype)

    rho = torch.empty(points.shape[0], device=device, dtype=dtype)
    batch_size = max(1, int(config.batch_size))
    for start in range(0, points.shape[0], batch_size):
        end = min(points.shape[0], start + batch_size)
        basis_values = _evaluate_basis_chunk(points[start:end], prepared_basis)
        psi = basis_values @ coeff
        rho[start:end] = torch.sum((psi * psi) * occ[None, :], dim=1)

    nx, ny, nz = grid.shape
    return rho.reshape(nx, ny, nz), device


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
) -> NCIFields:
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

    eigvals = torch.linalg.eigvalsh(hessian.reshape(-1, 3, 3))
    lambda2 = eigvals[:, 1].reshape(rho.shape)

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
    fields = compute_nci_fields(rho=rho, spacing_bohr=grid.spacing_bohr, rho_floor=cfg.rho_floor)
    return fields, device
