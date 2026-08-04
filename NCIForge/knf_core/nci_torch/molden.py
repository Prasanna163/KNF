import math
from typing import Dict, List, Tuple

import numpy as np

from .types import BasisFunction, ContractedShell, Wavefunction


ANGSTROM_TO_BOHR = 1.8897259886
SHELL_TO_L = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}


def _parse_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _double_factorial(n: int) -> float:
    if n <= 0:
        return 1.0
    out = 1.0
    k = n
    while k > 1:
        out *= k
        k -= 2
    return out


def _primitive_cartesian_norm(alpha: float, lx: int, ly: int, lz: int) -> float:
    lsum = lx + ly + lz
    pref = (2.0 * alpha / math.pi) ** 0.75
    num = (4.0 * alpha) ** (0.5 * lsum)
    den = math.sqrt(
        _double_factorial(2 * lx - 1)
        * _double_factorial(2 * ly - 1)
        * _double_factorial(2 * lz - 1)
    )
    return pref * num / den


def _cartesian_powers(l: int) -> List[Tuple[int, int, int]]:
    if l == 0:
        return [(0, 0, 0)]
    if l == 1:
        return [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if l == 2:
        return [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    if l == 3:
        return [
            (3, 0, 0),
            (0, 3, 0),
            (0, 0, 3),
            (2, 1, 0),
            (2, 0, 1),
            (1, 2, 0),
            (0, 2, 1),
            (1, 0, 2),
            (0, 1, 2),
            (1, 1, 1),
        ]
    if l == 4:
        return [
            (4, 0, 0),
            (0, 4, 0),
            (0, 0, 4),
            (3, 1, 0),
            (3, 0, 1),
            (1, 3, 0),
            (0, 3, 1),
            (1, 0, 3),
            (0, 1, 3),
            (2, 2, 0),
            (2, 0, 2),
            (0, 2, 2),
            (2, 1, 1),
            (1, 2, 1),
            (1, 1, 2),
        ]

    powers = []
    for lx in range(l, -1, -1):
        for ly in range(l - lx, -1, -1):
            lz = l - lx - ly
            powers.append((lx, ly, lz))
    return powers


def _section_indices(lines: List[str]) -> Dict[str, int]:
    out = {}
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("[") and "]" in stripped:
            name = stripped[1 : stripped.index("]")].strip().lower()
            out[name] = idx
    return out


def _extract_section(lines: List[str], section_name: str) -> Tuple[List[str], str]:
    indices = _section_indices(lines)
    if section_name not in indices:
        raise ValueError(f"Missing [{section_name}] section in Molden file.")

    ordered = sorted((idx, name) for name, idx in indices.items())
    start = indices[section_name] + 1
    stop = len(lines)
    raw_header = lines[indices[section_name]].strip()
    for idx, _ in ordered:
        if idx > indices[section_name]:
            stop = idx
            break
    return lines[start:stop], raw_header


def _parse_atoms(atom_lines: List[str], atom_header: str) -> Tuple[np.ndarray, List[str], str]:
    symbols = []
    coords = []

    units = "au"
    header_lower = atom_header.lower()
    if "angs" in header_lower:
        units = "angs"
    elif "au" in header_lower:
        units = "au"

    for raw in atom_lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        symbols.append(parts[0])
        x = _parse_float(parts[3])
        y = _parse_float(parts[4])
        z = _parse_float(parts[5])
        coords.append([x, y, z])

    if not coords:
        raise ValueError("No atomic coordinates found in [Atoms] section.")

    atoms = np.asarray(coords, dtype=np.float64)
    if units == "angs":
        atoms = atoms * ANGSTROM_TO_BOHR
    return atoms, symbols, units


def _parse_gto(gto_lines: List[str]) -> List[ContractedShell]:
    shells: List[ContractedShell] = []
    i = 0
    current_center = None
    while i < len(gto_lines):
        line = gto_lines[i].strip()
        if not line:
            i += 1
            continue

        parts = line.split()
        token0 = parts[0].lower()

        if parts and parts[0].isdigit():
            current_center = int(parts[0]) - 1
            i += 1
            continue

        if token0 in SHELL_TO_L:
            if current_center is None:
                raise ValueError("Encountered shell before center index in [GTO] block.")

            shell_type = token0
            if len(parts) < 2:
                raise ValueError(f"Invalid shell declaration line: '{line}'")
            n_prim = int(parts[1])
            scale = _parse_float(parts[2]) if len(parts) >= 3 else 1.0

            exponents = []
            coefficients = []
            i += 1
            prim_read = 0
            while prim_read < n_prim:
                if i >= len(gto_lines):
                    raise ValueError("Unexpected end of [GTO] while reading primitives.")
                prim_line = gto_lines[i].strip()
                if not prim_line:
                    i += 1
                    continue
                prim_parts = prim_line.split()
                if len(prim_parts) < 2:
                    raise ValueError(f"Invalid primitive line: '{prim_line}'")
                exponents.append(_parse_float(prim_parts[0]))
                coefficients.append(_parse_float(prim_parts[1]) * scale)
                i += 1
                prim_read += 1

            shells.append(
                ContractedShell(
                    center_index=current_center,
                    shell_type=shell_type,
                    angular_momentum=SHELL_TO_L[shell_type],
                    exponents=np.asarray(exponents, dtype=np.float64),
                    coefficients=np.asarray(coefficients, dtype=np.float64),
                )
            )
            continue

        i += 1

    if not shells:
        raise ValueError("No basis functions parsed from [GTO] section.")
    return shells


def _parse_mo(mo_lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    mo_coeffs: List[np.ndarray] = []
    occupations: List[float] = []

    i = 0
    while i < len(mo_lines):
        line = mo_lines[i].strip()
        if not line.startswith("Sym="):
            i += 1
            continue

        coeff_values: List[float] = []
        occ = 0.0
        i += 1

        while i < len(mo_lines):
            row = mo_lines[i].strip()
            if row.startswith("Sym="):
                break
            if not row:
                i += 1
                continue
            if row.startswith("Occup="):
                occ = _parse_float(row.split("=", 1)[1].strip())
                i += 1
                continue
            if "=" in row:
                i += 1
                continue

            parts = row.split()
            if len(parts) >= 2 and parts[0].lstrip("+-").isdigit():
                coeff_values.append(_parse_float(parts[1]))
            i += 1

        mo_coeffs.append(np.asarray(coeff_values, dtype=np.float64))
        occupations.append(float(occ))

    if not mo_coeffs:
        raise ValueError("No MO coefficient blocks found in [MO] section.")

    n_basis = len(mo_coeffs[0])
    for coeff in mo_coeffs:
        if len(coeff) != n_basis:
            raise ValueError("Inconsistent MO coefficient lengths in [MO] section.")

    coeff_matrix = np.stack(mo_coeffs, axis=1)
    return coeff_matrix, np.asarray(occupations, dtype=np.float64)


def _count_cartesian_functions(shells: List[ContractedShell]) -> int:
    return sum((s.angular_momentum + 1) * (s.angular_momentum + 2) // 2 for s in shells)


def _count_spherical_functions(shells: List[ContractedShell]) -> int:
    return sum(2 * s.angular_momentum + 1 for s in shells)


def _expand_basis_functions(
    shells: List[ContractedShell],
    n_basis_mo: int,
    apply_primitive_normalization: bool,
) -> List[BasisFunction]:
    n_cart = _count_cartesian_functions(shells)
    n_sph = _count_spherical_functions(shells)

    if n_basis_mo != n_cart:
        if n_basis_mo == n_sph:
            raise NotImplementedError(
                "Molden file appears to use spherical d/f/g shells. "
                "Current experimental backend supports cartesian basis expansion only."
            )
        raise ValueError(
            f"Basis size mismatch: MO has {n_basis_mo} functions, "
            f"while cartesian expansion predicts {n_cart}."
        )

    basis_functions: List[BasisFunction] = []
    for shell in shells:
        for powers in _cartesian_powers(shell.angular_momentum):
            lx, ly, lz = powers
            if apply_primitive_normalization:
                norms = np.asarray(
                    [
                        _primitive_cartesian_norm(alpha, lx, ly, lz)
                        for alpha in shell.exponents.tolist()
                    ],
                    dtype=np.float64,
                )
                coeffs = shell.coefficients * norms
            else:
                coeffs = shell.coefficients.copy()
            basis_functions.append(
                BasisFunction(
                    center_index=shell.center_index,
                    powers=powers,
                    exponents=shell.exponents.copy(),
                    normalized_coefficients=coeffs,
                )
            )

    if len(basis_functions) != n_basis_mo:
        raise ValueError(
            f"Expanded basis count {len(basis_functions)} does not match MO size {n_basis_mo}."
        )

    return basis_functions


def parse_molden(
    molden_path: str,
    apply_primitive_normalization: bool = False,
) -> Wavefunction:
    with open(molden_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    atom_lines, atom_header = _extract_section(lines, "atoms")
    gto_lines, _ = _extract_section(lines, "gto")
    mo_lines, _ = _extract_section(lines, "mo")

    atoms_bohr, atom_symbols, source_units = _parse_atoms(atom_lines, atom_header)
    shells = _parse_gto(gto_lines)
    mo_coefficients, occupations = _parse_mo(mo_lines)
    basis_functions = _expand_basis_functions(
        shells,
        n_basis_mo=mo_coefficients.shape[0],
        apply_primitive_normalization=apply_primitive_normalization,
    )

    return Wavefunction(
        atoms_bohr=atoms_bohr,
        atom_symbols=atom_symbols,
        shells=shells,
        basis_functions=basis_functions,
        mo_coefficients=mo_coefficients,
        occupations=occupations,
        source_units=source_units,
    )
