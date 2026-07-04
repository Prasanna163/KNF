"""
Atomic and molecular parameter tables for GeoInit.

Contains covalent radii, van der Waals radii, dispersion C₆ coefficients,
atomic masses, default energy weights, ideal bond angles, and helper
look-up functions used throughout the package.

All radii are in ångströms (Å).  C₆ coefficients are in eV·Å⁶.
Angles are stored in degrees but the ``get_ideal_angle`` helper returns
radians for direct use in NumPy trigonometry.

Sources
-------
* Covalent radii — Cordero *et al.*, Dalton Trans. (2008).
* van der Waals radii — Bondi, J. Phys. Chem. (1964), augmented with
  Alvarez, Dalton Trans. (2013) for heavier elements.
* C₆ — rough estimates inspired by Grimme DFT-D3 reference data.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Covalent radii (Å)
# ---------------------------------------------------------------------------
COVALENT_RADII: dict[str, float] = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "F": 0.57,
    "Cl": 1.02,
    "Br": 1.20,
    "P": 1.07,
    "Si": 1.11,
    "B": 0.84,
}

# ---------------------------------------------------------------------------
# van der Waals radii (Å)
# ---------------------------------------------------------------------------
VDW_RADII: dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "F": 1.47,
    "Cl": 1.75,
    "Br": 1.85,
    "P": 1.80,
    "Si": 2.10,
    "B": 1.92,
}

# ---------------------------------------------------------------------------
# Crude C₆ dispersion coefficients (eV·Å⁶)
# ---------------------------------------------------------------------------
C6_PARAMS: dict[str, float] = {
    "H": 1.5,
    "C": 7.0,
    "N": 5.5,
    "O": 4.5,
    "S": 15.0,
    "F": 3.5,
    "Cl": 12.0,
    "Br": 18.0,
    "P": 14.0,
    "Si": 18.0,
    "B": 8.0,
}

# ---------------------------------------------------------------------------
# Atomic masses (amu)
# ---------------------------------------------------------------------------
ATOMIC_MASSES: dict[str, float] = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
    "F": 18.998,
    "Cl": 35.45,
    "Br": 79.904,
    "P": 30.974,
    "Si": 28.085,
    "B": 10.81,
}

# ---------------------------------------------------------------------------
# Default energy-term weights for the GeoInit-V1 functional
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS: dict[str, float] = {
    "bond": 10.0,
    "angle": 5.0,
    "clash": 1.0,
    "disp": 0.1,
    "coul": 0.0,
    "rigid": 10.0,
}

# ---------------------------------------------------------------------------
# Ideal angles by coordination number (degrees)
# ---------------------------------------------------------------------------
IDEAL_ANGLES: dict[int, float] = {
    1: 180.0,   # terminal — not really used
    2: 180.0,   # linear
    3: 120.0,   # trigonal planar
    4: 109.47,  # tetrahedral
    5: 90.0,    # trigonal bipyramidal (approximate)
    6: 90.0,    # octahedral
}

# ---------------------------------------------------------------------------
# Special angle overrides keyed by (central_element, coordination)
# ---------------------------------------------------------------------------
ANGLE_OVERRIDES: dict[tuple[str, int], float] = {
    ("O", 2): 104.5,   # water-like: O with 2 bonds
    ("N", 3): 107.0,   # pyramidal nitrogen (like NH₃)
    ("S", 2): 92.0,    # sulfur with 2 bonds
}


# ===================================================================
# Helper look-up functions
# ===================================================================

def get_covalent_radius(symbol: str) -> float:
    """Return the covalent radius for *symbol* (Å).

    Parameters
    ----------
    symbol : str
        Chemical element symbol, e.g. ``"C"``, ``"Cl"``.

    Returns
    -------
    float
        Covalent radius in ångströms.

    Raises
    ------
    KeyError
        If *symbol* is not in the parameter table.
    """
    try:
        return COVALENT_RADII[symbol]
    except KeyError:
        raise KeyError(
            f"Covalent radius not available for element '{symbol}'. "
            f"Known elements: {sorted(COVALENT_RADII)}"
        ) from None


def get_vdw_radius(symbol: str) -> float:
    """Return the van der Waals radius for *symbol* (Å).

    Parameters
    ----------
    symbol : str
        Chemical element symbol.

    Returns
    -------
    float
        vdW radius in ångströms.

    Raises
    ------
    KeyError
        If *symbol* is not in the parameter table.
    """
    try:
        return VDW_RADII[symbol]
    except KeyError:
        raise KeyError(
            f"vdW radius not available for element '{symbol}'. "
            f"Known elements: {sorted(VDW_RADII)}"
        ) from None


def get_c6(symbol: str) -> float:
    """Return the homonuclear C₆ dispersion coefficient for *symbol* (eV·Å⁶).

    Parameters
    ----------
    symbol : str
        Chemical element symbol.

    Returns
    -------
    float
        C₆ coefficient in eV·Å⁶.

    Raises
    ------
    KeyError
        If *symbol* is not in the parameter table.
    """
    try:
        return C6_PARAMS[symbol]
    except KeyError:
        raise KeyError(
            f"C6 coefficient not available for element '{symbol}'. "
            f"Known elements: {sorted(C6_PARAMS)}"
        ) from None


def get_c6_pair(sym_i: str, sym_j: str) -> float:
    """Return the heteronuclear C₆ coefficient via geometric-mean mixing.

    .. math::

        C_6^{ij} = \\sqrt{C_6^{ii} \\cdot C_6^{jj}}

    Parameters
    ----------
    sym_i, sym_j : str
        Chemical element symbols.

    Returns
    -------
    float
        Combined C₆ coefficient in eV·Å⁶.
    """
    return math.sqrt(get_c6(sym_i) * get_c6(sym_j))


def get_ideal_angle(central_symbol: str, coordination: int) -> float:
    """Return the ideal valence angle at *central_symbol* in **radians**.

    First checks :data:`ANGLE_OVERRIDES` for a ``(central_symbol, coordination)``
    entry; otherwise falls back to :data:`IDEAL_ANGLES` keyed by *coordination*.

    Parameters
    ----------
    central_symbol : str
        Element symbol of the central (vertex) atom.
    coordination : int
        Number of bonded neighbours around the central atom.

    Returns
    -------
    float
        Ideal angle in radians.

    Raises
    ------
    KeyError
        If neither the override table nor the default table contains
        an entry for the requested coordination number.
    """
    key = (central_symbol, coordination)
    if key in ANGLE_OVERRIDES:
        return math.radians(ANGLE_OVERRIDES[key])

    if coordination in IDEAL_ANGLES:
        return math.radians(IDEAL_ANGLES[coordination])

    raise KeyError(
        f"No ideal angle defined for element '{central_symbol}' "
        f"with coordination {coordination}."
    )
