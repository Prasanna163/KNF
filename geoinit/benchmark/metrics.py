"""Benchmark quality metrics for GeoInit geometry relaxation.

This module computes per-molecule quality indicators (bond errors, clash
counts, RMSD, energy reduction, …) and formats them into a human-readable
ASCII table for terminal display.

All public functions are pure — they accept arrays and dataclass instances
and return plain dicts or strings — so they can be used both by the CLI
and by downstream analysis scripts.
"""

from __future__ import annotations

import numpy as np

from geoinit.optimize.guards import check_geometry, compare_geometries

# Type alias for the RelaxResult dataclass — imported lazily to avoid
# circular imports at module level.
if False:  # TYPE_CHECKING
    from geoinit.optimize.relax import RelaxResult  # noqa: F401


# ── Per-molecule metrics ───────────────────────────────────────────────────────

def compute_metrics(
    symbols: list[str],
    coords_before: np.ndarray,
    coords_after: np.ndarray,
    relax_result: "RelaxResult",
) -> dict:
    """Compute all benchmark metrics for a single molecule.

    Parameters
    ----------
    symbols : list[str]
        Atomic element symbols of length *N*.
    coords_before : np.ndarray, shape (N, 3)
        Cartesian coordinates **before** relaxation (the distorted input).
    coords_after : np.ndarray, shape (N, 3)
        Cartesian coordinates **after** relaxation.
    relax_result : RelaxResult
        The result object returned by :func:`geoinit.optimize.relax.relax`.

    Returns
    -------
    dict
        A flat dictionary with the following keys:

        * ``initial_energy``, ``final_energy``, ``energy_reduction``
        * ``max_bond_error_before``, ``max_bond_error_after``
        * ``mean_bond_error_before``, ``mean_bond_error_after``
        * ``max_clash_before``, ``max_clash_after``
        * ``n_clashes_before``, ``n_clashes_after``
        * ``optimizer_success``, ``optimizer_steps``
        * ``rmsd``, ``max_displacement``
        * ``is_safe_before``, ``is_safe_after``
    """
    report_before = check_geometry(symbols, coords_before)
    report_after = check_geometry(symbols, coords_after)

    # Displacement analysis
    disp = coords_after - coords_before  # (N, 3)
    per_atom_disp = np.linalg.norm(disp, axis=1)  # (N,)
    rmsd = float(np.sqrt(np.mean(per_atom_disp ** 2)))
    max_disp = float(np.max(per_atom_disp))

    # Try compare_geometries for additional cross-checks (it may add
    # more info in future versions), but don't fail if the API changes.
    try:
        _comparison = compare_geometries(symbols, coords_before, coords_after)
    except Exception:
        _comparison = {}

    return {
        # Energy
        "initial_energy": relax_result.initial_energy,
        "final_energy": relax_result.final_energy,
        "energy_reduction": relax_result.initial_energy - relax_result.final_energy,
        # Bond errors
        "max_bond_error_before": report_before.max_bond_error,
        "max_bond_error_after": report_after.max_bond_error,
        "mean_bond_error_before": report_before.mean_bond_error,
        "mean_bond_error_after": report_after.mean_bond_error,
        # Clashes
        "max_clash_before": report_before.max_clash_ratio,
        "max_clash_after": report_after.max_clash_ratio,
        "n_clashes_before": report_before.n_clashes,
        "n_clashes_after": report_after.n_clashes,
        # Optimiser
        "optimizer_success": relax_result.success,
        "optimizer_steps": relax_result.n_steps,
        # Displacement
        "rmsd": rmsd,
        "max_displacement": max_disp,
        # Safety
        "is_safe_before": report_before.is_safe,
        "is_safe_after": report_after.is_safe,
        "safe_success": bool(report_after.is_safe and report_after.max_bond_error < 0.01 and report_after.n_clashes == 0),
    }


# ── ASCII table formatter ─────────────────────────────────────────────────────

# The columns shown in the summary table.  Each tuple is
# (column header, dict key, format spec, column width).
_TABLE_COLUMNS: list[tuple[str, str, str, int]] = [
    ("Name",       "_name",                "s",   18),
    ("E_init",     "initial_energy",       ".3f", 10),
    ("E_final",    "final_energy",         ".3f", 10),
    ("ΔE",         "energy_reduction",     ".3f",  8),
    ("BondErr→",   "max_bond_error_after", ".4f",  9),
    ("Clash→",     "max_clash_after",      ".4f",  8),
    ("#Clash→",    "n_clashes_after",      "d",    7),
    ("RMSD",       "rmsd",                 ".4f",  8),
    ("Safe?",      "is_safe_after",        "s",    6),
    ("Steps",      "optimizer_steps",      "d",    6),
    ("OK?",        "optimizer_success",    "s",    4),
    ("SafeOK?",    "safe_success",         "s",    7),
]


def format_metrics_table(
    all_metrics: list[dict],
    names: list[str],
) -> str:
    """Format metrics as a pretty ASCII table for terminal display.

    Parameters
    ----------
    all_metrics : list[dict]
        One metrics dict per molecule (as returned by :func:`compute_metrics`).
    names : list[str]
        Human-readable names for each molecule (same length as *all_metrics*).

    Returns
    -------
    str
        A multi-line string ready for ``print()``.
    """
    if not all_metrics:
        return "  (no results)\n"

    # ── Build header ────────────────────────────────────────────────────
    hdr_parts: list[str] = []
    sep_parts: list[str] = []
    for header, _key, _fmt, w in _TABLE_COLUMNS:
        hdr_parts.append(f"{header:>{w}s}")
        sep_parts.append("─" * w)

    header_line = " │ ".join(hdr_parts)
    sep_line = "─┼─".join(sep_parts)

    lines: list[str] = [f"  {header_line}", f"  {sep_line}"]

    # ── Build rows ──────────────────────────────────────────────────────
    for name, m in zip(names, all_metrics):
        row_parts: list[str] = []
        for _header, key, fmt, w in _TABLE_COLUMNS:
            if key == "_name":
                # Truncate long names
                display = name if len(name) <= w else name[: w - 1] + "…"
                row_parts.append(f"{display:>{w}s}")
            else:
                raw = m.get(key, "")
                if isinstance(raw, bool):
                    display = "✓" if raw else "✗"
                    row_parts.append(f"{display:>{w}s}")
                else:
                    row_parts.append(f"{raw:>{w}{fmt}}")
        lines.append("  " + " │ ".join(row_parts))

    return "\n".join(lines)
