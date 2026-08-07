"""
XYZ file reader and writer.

The standard XYZ format is::

    <number_of_atoms>
    <comment line>
    <symbol>  <x>  <y>  <z>
    ...

Coordinates are assumed to be in ångströms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_xyz(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Read an XYZ file and return atomic symbols and Cartesian coordinates.

    Parameters
    ----------
    path : str or Path
        Path to the XYZ file.

    Returns
    -------
    symbols : list[str]
        Element symbols, length *N*.
    coords : np.ndarray
        Cartesian coordinates with shape ``(N, 3)`` and dtype ``float64``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file is malformed (wrong atom count, unparsable lines, etc.).

    Examples
    --------
    >>> symbols, coords = read_xyz("examples/water_bad.xyz")
    >>> len(symbols)
    3
    >>> coords.shape
    (3, 3)
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"XYZ file not found: {path}")

    with path.open("r") as fh:
        lines = fh.readlines()

    if len(lines) < 2:
        raise ValueError(f"XYZ file too short (need ≥ 2 header lines): {path}")

    # --- first line: number of atoms ------------------------------------
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        raise ValueError(
            f"First line of XYZ must be an integer (number of atoms), "
            f"got: '{lines[0].strip()}'"
        ) from None

    # --- second line: comment (ignored) ---------------------------------
    # lines[1] is the comment — skip it.

    # --- atom lines -----------------------------------------------------
    atom_lines = lines[2:]
    if len(atom_lines) < n_atoms:
        raise ValueError(
            f"Expected {n_atoms} atom lines but found {len(atom_lines)} in {path}"
        )

    symbols: list[str] = []
    coords = np.empty((n_atoms, 3), dtype=np.float64)

    for idx in range(n_atoms):
        parts = atom_lines[idx].split()
        if len(parts) < 4:
            raise ValueError(
                f"Line {idx + 3} in {path} does not have 4 columns: "
                f"'{atom_lines[idx].strip()}'"
            )
        symbols.append(parts[0])
        try:
            coords[idx, 0] = float(parts[1])
            coords[idx, 1] = float(parts[2])
            coords[idx, 2] = float(parts[3])
        except ValueError:
            raise ValueError(
                f"Cannot parse coordinates on line {idx + 3} of {path}: "
                f"'{atom_lines[idx].strip()}'"
            ) from None

    return symbols, coords


def write_xyz(
    path: str | Path,
    symbols: list[str],
    coords: np.ndarray,
    comment: str = "",
) -> None:
    """Write atomic symbols and coordinates to an XYZ file.

    Parameters
    ----------
    path : str or Path
        Destination file path.  Parent directories are created if needed.
    symbols : list[str]
        Element symbols, length *N*.
    coords : np.ndarray
        Cartesian coordinates, shape ``(N, 3)``.
    comment : str, optional
        Comment written on the second line of the XYZ file.

    Raises
    ------
    ValueError
        If ``len(symbols)`` does not match ``coords.shape[0]``.
    """
    n_atoms = len(symbols)
    if coords.shape != (n_atoms, 3):
        raise ValueError(
            f"Shape mismatch: {n_atoms} symbols but coords has shape {coords.shape}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="\n") as fh:
        fh.write(f"{n_atoms}\n")
        fh.write(f"{comment}\n")
        for i in range(n_atoms):
            sym = symbols[i]
            x, y, z = coords[i]
            fh.write(f"{sym:<4s} {x:14.8f} {y:14.8f} {z:14.8f}\n")
