"""
GeoInit — Physics-inspired geometry initializer for molecular optimisation.

Provides warm-start coordinates that satisfy basic physical constraints
(bond lengths, valence angles, steric clashes, dispersion contacts) before
handing off to a full quantum-chemistry or force-field optimiser.
"""

__version__ = "1.0.0"

from geoinit.core import atoms, geometry, io_xyz, params, topology  # noqa: F401
from geoinit.optimize.selector import (  # noqa: F401
    select_initial_geometry,
    v0_8_selection_policy,
    v0_9_selection_policy,
    v1_0_selection_policy,
)
