"""Optimisation subpackage for GeoInit."""

from geoinit.optimize.selector import (
    SelectionPolicy,
    SelectionResult,
    select_initial_geometry,
    v0_8_selection_policy,
    v0_9_selection_policy,
    v1_0_selection_policy,
)

__all__ = [
    "SelectionPolicy",
    "SelectionResult",
    "select_initial_geometry",
    "v0_8_selection_policy",
    "v0_9_selection_policy",
    "v1_0_selection_policy",
]
