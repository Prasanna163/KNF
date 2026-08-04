"""Benchmark suite runner for GeoInit.

Defines named suites of example XYZ files, runs relaxation on each,
collects per-molecule metrics, and returns a :class:`pandas.DataFrame`
that can be saved to CSV or pretty-printed.

Usage from the CLI::

    geoinit benchmark --suite small --out results.csv

Usage from Python::

    from geoinit.benchmark.suite import run_suite, save_results, print_summary

    df = run_suite("small")
    save_results(df, "results.csv")
    print_summary(df)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geoinit.benchmark.metrics import compute_metrics, format_metrics_table


# ── Suite definitions ──────────────────────────────────────────────────────────
# Paths are relative to the project root (the directory that contains the
# ``geoinit`` package *and* the ``examples`` folder).

SUITES: dict[str, list[str]] = {
    "small": [
        "examples/water_bad.xyz",
        "examples/co2_bad.xyz",
        "examples/methanol_bad.xyz",
        "examples/dmf_bad.xyz",
        "examples/eg_bad.xyz",
        "examples/thiophene_bad.xyz",
    ],
    "medium": [
        # Singles
        "examples/water_bad.xyz",
        "examples/co2_bad.xyz",
        "examples/methanol_bad.xyz",
        "examples/dmf_bad.xyz",
        "examples/eg_bad.xyz",
        "examples/thiophene_bad.xyz",
        "examples/acetonitrile_bad.xyz",
        "examples/acetone_bad.xyz",
        "examples/formamide_bad.xyz",
        "examples/benzene_bad.xyz",
        "examples/pyridine_bad.xyz",
        # Complexes
        "examples/complexes/water_dimer_bad.xyz",
        "examples/complexes/co2_dmf_bad.xyz",
        "examples/complexes/co2_eg_bad.xyz",
        "examples/complexes/methanol_water_bad.xyz",
        "examples/complexes/acetone_water_bad.xyz",
        "examples/complexes/dmf_water_bad.xyz",
        "examples/complexes/benzene_water_bad.xyz",
        "examples/complexes/thiophene_water_bad.xyz",
        "examples/complexes/benzene_co2_bad.xyz",
        "examples/complexes/acetonitrile_water_bad.xyz",
    ],
}


# ── Helpers ────────────────────────────────────────────────────────────────────

# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_project_root() -> Path:
    """Resolve the GeoInit project root directory.

    Strategy (in order):
    1. Walk up from *this* file until we find a directory containing
       ``pyproject.toml``.
    2. Fall back to the current working directory.
    """
    here = Path(__file__).resolve().parent
    for ancestor in [here] + list(here.parents):
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    return Path.cwd()


def _pretty_name(rel_path: str) -> str:
    """Derive a short human-readable name from a relative file path.

    ``examples/water_bad.xyz`` → ``water_bad``
    ``examples/complexes/co2_dmf_bad.xyz`` → ``co2_dmf_bad``
    """
    return Path(rel_path).stem


def find_fragments(bonds: list[tuple[int, int]], n_atoms: int) -> list[list[int]]:
    """Find connected fragments (components) based on bonds."""
    adj = {i: [] for i in range(n_atoms)}
    for u, v in bonds:
        adj[u].append(v)
        adj[v].append(u)

    visited = set()
    fragments = []
    for i in range(n_atoms):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for nbr in adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            fragments.append(sorted(comp))
    return fragments


def kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Compute structural RMSD between two geometries using Kabsch alignment."""
    if len(coords_a) <= 1:
        return 0.0
    # Shift to centroids
    c_a = coords_a - np.mean(coords_a, axis=0)
    c_b = coords_b - np.mean(coords_b, axis=0)
    # Covariance matrix
    H = c_a.T @ c_b
    try:
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        F = np.eye(3)
        if d < 0:
            F[2, 2] = -1.0
        R = Vt.T @ F @ U.T
        c_a_rot = c_a @ R.T
        diff = c_a_rot - c_b
        return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
    except Exception:
        # Fallback to displacement if SVD fails
        diff = c_a - c_b
        return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


# ── Public API ─────────────────────────────────────────────────────────────────

def run_suite(
    suite_name: str = "small",
    output_dir: str = "outputs",
    weights: dict[str, float] | None = None,
    sigma: float = 0.05,
    maxiter: int = 500,
    clash_mode: str = "compact",
) -> pd.DataFrame:
    """Run a benchmark suite and return results as a DataFrame.

    For each molecule in the suite:

    1. Read the input XYZ.
    2. Compute pre-relaxation metrics.
    3. Run :func:`~geoinit.optimize.relax.relax`.
    4. Compute post-relaxation & fragment metrics.
    5. Save the optimised geometry to *output_dir*.
    """
    from geoinit.core.atoms import Molecule
    from geoinit.optimize.relax import relax
    from geoinit.core.topology import Topology
    from geoinit.core.geometry import angle as compute_angle
    from geoinit.core.params import get_ideal_angle, get_vdw_radius
    from geoinit.core.geometry import distance
    from scipy.spatial.distance import cdist

    if suite_name not in SUITES:
        print(
            f"  ✗ Unknown suite '{suite_name}'. "
            f"Available: {', '.join(sorted(SUITES))}",
            file=sys.stderr,
        )
        return pd.DataFrame()

    root = _find_project_root()
    file_list = SUITES[suite_name]

    # Ensure output directory exists
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    n_total = len(file_list)

    for idx, rel_path in enumerate(file_list, 1):
        name = _pretty_name(rel_path)
        abs_path = root / rel_path

        # Progress indicator
        print(f"  [{idx}/{n_total}] {name} … ", end="", flush=True)

        if not abs_path.is_file():
            print("SKIP (file not found)")
            continue

        mol = Molecule.from_xyz(str(abs_path))
        n_atoms = len(mol.symbols)

        # ── Relax ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        result = relax(
            mol.symbols,
            mol.coords,
            weights=weights,
            sigma=sigma,
            maxiter=maxiter,
            clash_mode=clash_mode,
        )
        wall = time.perf_counter() - t0

        # ── Metrics ─────────────────────────────────────────────────────
        m = compute_metrics(
            mol.symbols,
            result.initial_coords,
            result.final_coords,
            result,
        )
        m["name"] = name
        m["wall_time"] = round(wall, 4)

        # Connected component fragment search
        topo = Topology(mol.symbols, result.initial_coords)
        topo_for_frags = Topology(mol.symbols, result.final_coords, scale=1.15)
        frags = find_fragments(topo_for_frags.bonds, n_atoms)
        is_complex = len(frags) >= 2
        m["is_complex"] = is_complex

        if not is_complex:
            # Single molecule metrics
            angles_dev = []
            for u, v, w_idx in topo.angles:
                theta = compute_angle(result.final_coords, u, v, w_idx)
                theta0 = get_ideal_angle(mol.symbols[v], topo.coordination[v])
                angles_dev.append(abs(np.degrees(theta) - np.degrees(theta0)))
            angle_error = float(np.mean(angles_dev)) if angles_dev else 0.0

            m["bond_error"] = m["max_bond_error_after"]
            m["angle_error"] = angle_error
            m["clashes"] = m["n_clashes_after"]
        else:
            # Complex metrics (assuming first two fragments are A and B)
            frag_A = frags[0]
            frag_B = frags[1]

            m["fragment_A_internal_RMSD"] = kabsch_rmsd(mol.coords[frag_A], result.final_coords[frag_A])
            m["fragment_B_internal_RMSD"] = kabsch_rmsd(mol.coords[frag_B], result.final_coords[frag_B])

            # Center of mass distance change
            com_before_A = np.mean(mol.coords[frag_A], axis=0)
            com_before_B = np.mean(mol.coords[frag_B], axis=0)
            com_after_A = np.mean(result.final_coords[frag_A], axis=0)
            com_after_B = np.mean(result.final_coords[frag_B], axis=0)
            dist_before = float(np.linalg.norm(com_before_A - com_before_B))
            dist_after = float(np.linalg.norm(com_after_A - com_after_B))
            m["COM_distance_change"] = dist_after - dist_before

            # Minimum inter-fragment distance
            dists = cdist(result.final_coords[frag_A], result.final_coords[frag_B])
            m["minimum_interfragment_distance"] = float(np.min(dists))

            # Inter-fragment clashes
            inter_clashes = 0
            set_A = set(frag_A)
            set_B = set(frag_B)
            for u, v in topo.nonbonded_pairs:
                if (u in set_A and v in set_B) or (u in set_B and v in set_A):
                    r_uv = distance(result.final_coords, u, v)
                    s_uv = 0.75 * (get_vdw_radius(mol.symbols[u]) + get_vdw_radius(mol.symbols[v]))
                    if r_uv > 0.0 and (s_uv / r_uv) > 1.0:
                        inter_clashes += 1
            m["interfragment_clash_count"] = inter_clashes

        all_rows.append(m)

        status = "✓" if result.success else "✗"
        if result.message == "safe_converged":
            status = "✓ (safe)"
        safe = "safe" if m["is_safe_after"] else "unsafe"
        print(
            f"{status}  E {m['initial_energy']:.3f} → {m['final_energy']:.3f}  "
            f"({m['optimizer_steps']} steps, {wall:.2f}s, {safe})"
        )

        # ── Save relaxed geometry ───────────────────────────────────────
        try:
            relaxed_mol = Molecule(symbols=mol.symbols, coords=result.final_coords)
            out_xyz = out_dir / f"{name}_relaxed.xyz"
            comment = (
                f"GeoInit relaxed | E={result.final_energy:.6f} | "
                f"steps={result.n_steps} | msg={result.message}"
            )
            relaxed_mol.to_xyz(str(out_xyz), comment=comment)
        except Exception as exc:  # pragma: no cover
            print(f"    ⚠ could not write output: {exc}", file=sys.stderr)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Save split outputs
    df_mols = df[~df["is_complex"]].copy()
    df_complexes = df[df["is_complex"]].copy()

    mol_cols = ["name", "bond_error", "angle_error", "clashes", "rmsd", "optimizer_steps", "safe_success"]
    complex_cols = ["name", "fragment_A_internal_RMSD", "fragment_B_internal_RMSD", "COM_distance_change",
                    "minimum_interfragment_distance", "interfragment_clash_count", "optimizer_steps", "safe_success"]

    mol_cols = [c for c in mol_cols if c in df_mols.columns]
    complex_cols = [c for c in complex_cols if c in df_complexes.columns]

    if not df_mols.empty:
        df_mols[mol_cols].to_csv(out_dir / "small_molecules.csv", index=False)
    if not df_complexes.empty:
        df_complexes[complex_cols].to_csv(out_dir / "complexes.csv", index=False)
    df.to_csv(out_dir / "combined.csv", index=False)

    # Re-order so 'name' is the first column
    cols = ["name"] + [c for c in df.columns if c != "name"]
    return df[cols]


def save_results(df: pd.DataFrame, output_path: str) -> None:
    """Save benchmark results to CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out), index=False, float_format="%.6f")


def print_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary of benchmark results to the terminal."""
    if df.empty:
        print("  (no benchmark results to summarise)")
        return

    df_mols = df[~df["is_complex"]].copy()
    df_complexes = df[df["is_complex"]].copy()

    if not df_mols.empty:
        print("\n  ┌────────────────── Small Molecules ───────────────────┐")
        names_mols = df_mols["name"].tolist()
        metrics_mols = df_mols.drop(columns=["name", "wall_time"], errors="ignore").to_dict("records")
        print(format_metrics_table(metrics_mols, names_mols))
        print("  └──────────────────────────────────────────────────────┘")

    if not df_complexes.empty:
        print("\n  ┌──────────────────── Complexes ───────────────────────┐")
        names_comps = df_complexes["name"].tolist()
        metrics_comps = df_complexes.drop(columns=["name", "wall_time"], errors="ignore").to_dict("records")
        print(format_metrics_table(metrics_comps, names_comps))
        print("  └──────────────────────────────────────────────────────┘")

    # ── Aggregate statistics ────────────────────────────────────────────
    n_total = len(df)
    n_success = int(df["optimizer_success"].sum())
    n_safe = int(df["is_safe_after"].sum())
    n_safe_success = int(df["safe_success"].sum())
    mean_steps = df["optimizer_steps"].mean()
    mean_time = df["wall_time"].mean() if "wall_time" in df.columns else 0.0
    mean_rmsd = df["rmsd"].mean()
    mean_reduction = df["energy_reduction"].mean()

    print()
    print("  ── Aggregate ──────────────────────────────────────────────")
    print(f"  Molecules          : {n_total}")
    print(f"  Converged (optim)  : {n_success}/{n_total}")
    print(f"  Safe after relax   : {n_safe}/{n_total}")
    print(f"  Safe success       : {n_safe_success}/{n_total}")
    print(f"  Mean steps         : {mean_steps:.1f}")
    print(f"  Mean wall time     : {mean_time:.3f} s")
    print(f"  Mean RMSD          : {mean_rmsd:.4f} Å")
    print(f"  Mean ΔE            : {mean_reduction:.4f}")
