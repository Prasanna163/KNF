from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class ContractedShell:
    center_index: int
    shell_type: str
    angular_momentum: int
    exponents: np.ndarray
    coefficients: np.ndarray


@dataclass
class BasisFunction:
    center_index: int
    powers: Tuple[int, int, int]
    exponents: np.ndarray
    normalized_coefficients: np.ndarray


@dataclass
class Wavefunction:
    atoms_bohr: np.ndarray
    atom_symbols: List[str]
    shells: List[ContractedShell]
    basis_functions: List[BasisFunction]
    mo_coefficients: np.ndarray
    occupations: np.ndarray
    source_units: str


@dataclass
class GridSpec:
    x_bohr: np.ndarray
    y_bohr: np.ndarray
    z_bohr: np.ndarray
    spacing_bohr: float

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (len(self.x_bohr), len(self.y_bohr), len(self.z_bohr))

    @property
    def n_points(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz


@dataclass
class PreparedBasisFunction:
    center: torch.Tensor
    powers: Tuple[int, int, int]
    exponents: torch.Tensor
    coefficients: torch.Tensor


@dataclass
class NCIFields:
    rho: torch.Tensor
    rdg: torch.Tensor
    sign_lambda2_rho: torch.Tensor
