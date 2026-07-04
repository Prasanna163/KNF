"""
Lightweight :class:`Molecule` container.

Wraps a list of element symbols and a ``(N, 3)`` coordinate array with
convenience methods for XYZ I/O and copying.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from geoinit.core.io_xyz import read_xyz, write_xyz


@dataclass
class Molecule:
    """Immutable-style container for a molecular geometry.

    Parameters
    ----------
    symbols : list[str]
        Element symbols, one per atom.
    coords : np.ndarray
        Cartesian coordinates with shape ``(N, 3)`` and dtype ``float64``.

    Examples
    --------
    >>> mol = Molecule.from_xyz("examples/water_bad.xyz")
    >>> mol.n_atoms
    3
    >>> relaxed = mol.copy()
    >>> relaxed.coords[0, 0] += 0.1  # does not mutate *mol*
    """

    symbols: list[str]
    coords: np.ndarray  # (N, 3), float64

    # --- derived properties -----------------------------------------------

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return len(self.symbols)

    # --- constructors / serialisation -------------------------------------

    @classmethod
    def from_xyz(cls, path: str) -> Molecule:
        """Create a :class:`Molecule` from an XYZ file.

        Parameters
        ----------
        path : str
            Path to the XYZ file.

        Returns
        -------
        Molecule
        """
        symbols, coords = read_xyz(path)
        return cls(symbols=symbols, coords=coords)

    def to_xyz(self, path: str, comment: str = "") -> None:
        """Write the molecule to an XYZ file.

        Parameters
        ----------
        path : str
            Destination file path.
        comment : str, optional
            Comment line (second line of the XYZ file).
        """
        write_xyz(path, self.symbols, self.coords, comment=comment)

    # --- utilities --------------------------------------------------------

    def copy(self) -> Molecule:
        """Return a deep copy so that coordinate mutations are independent.

        Returns
        -------
        Molecule
            A new :class:`Molecule` with copied symbols and coordinates.
        """
        return Molecule(
            symbols=list(self.symbols),
            coords=self.coords.copy(),
        )

    # --- dunder -----------------------------------------------------------

    def __repr__(self) -> str:
        formula_counts: dict[str, int] = {}
        for s in self.symbols:
            formula_counts[s] = formula_counts.get(s, 0) + 1
        formula = "".join(
            f"{el}{cnt}" if cnt > 1 else el
            for el, cnt in sorted(formula_counts.items())
        )
        return f"Molecule({formula}, n_atoms={self.n_atoms})"
