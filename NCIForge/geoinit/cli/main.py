"""
GeoInit CLI entry point.

Provides the ``geoinit`` command with subcommands:

* ``relax``     – relax a distorted molecular geometry
* ``report``    – print a diagnostic geometry report
* ``benchmark`` – run the built-in benchmark suite

Usage::

    geoinit relax examples/water_bad.xyz --out outputs/water_geoinit.xyz
    geoinit report examples/water_bad.xyz
    geoinit benchmark --suite small --out outputs/geoinit_v1_summary.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time


# ── Box-drawing helpers ────────────────────────────────────────────────────

def _box_header(title: str, width: int = 60) -> str:
    """Return a Unicode box header line."""
    inner = f" {title} "
    pad = width - 2 - len(inner)
    left = pad // 2
    right = pad - left
    return f"┌{'─' * left}{inner}{'─' * right}┐"


def _box_line(label: str, value: str, width: int = 60) -> str:
    """Return a box content line with label and value."""
    content = f"  {label:<28s} {value}"
    pad = width - 2 - len(content)
    if pad < 0:
        pad = 0
    return f"│{content}{' ' * pad}│"


def _box_footer(width: int = 60) -> str:
    """Return a Unicode box footer line."""
    return f"└{'─' * (width - 2)}┘"


def _box_separator(width: int = 60) -> str:
    """Return a Unicode box separator."""
    return f"├{'─' * (width - 2)}┤"


# ── Parser ─────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the GeoInit CLI."""
    parser = argparse.ArgumentParser(
        prog="geoinit",
        description=(
            "GeoInit - physics-inspired geometry warm-start engine. "
            "Generates chemically sane initial geometries using lightweight "
            "physics-informed functionals before xTB/DFT optimization."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    sub = parser.add_subparsers(dest="command", help="Available subcommands")

    # ── relax ──────────────────────────────────────────────────────────
    p_relax = sub.add_parser(
        "relax",
        help="Relax a distorted geometry using the GeoInit functional.",
    )
    p_relax.add_argument("input", help="Path to input XYZ file.")
    p_relax.add_argument(
        "--out", "-o", default=None,
        help="Path to write relaxed XYZ (default: <input>_geoinit.xyz).",
    )
    p_relax.add_argument(
        "--maxiter", type=int, default=500,
        help="Maximum optimizer iterations (default: 500).",
    )
    p_relax.add_argument(
        "--sigma", type=float, default=0.05,
        help="Bond-length tolerance σ in Å (default: 0.05).",
    )
    p_relax.add_argument(
        "--weights-bond", type=float, default=None,
        help="Weight for bond term (default: 10.0).",
    )
    p_relax.add_argument(
        "--weights-angle", type=float, default=None,
        help="Weight for angle term (default: 5.0).",
    )
    p_relax.add_argument(
        "--weights-clash", type=float, default=None,
        help="Weight for clash term (default: 1.0).",
    )
    p_relax.add_argument(
        "--weights-disp", type=float, default=None,
        help="Weight for dispersion term (default: 0.1).",
    )
    p_relax.add_argument(
        "--clash-mode", choices=["exp", "compact"], default="compact",
        help="Clash potential mode (default: compact).",
    )
    p_relax.add_argument(
        "--mode", choices=["legacy", "select", "select-v0.9", "select-v1.0"], default="legacy",
        help="Initializer mode: legacy relaxation, V1.0 select, or V0.9 "
             "pre-release complex-candidate select (default: legacy).",
    )

    # ── report ─────────────────────────────────────────────────────────
    p_report = sub.add_parser(
        "report",
        help="Print a diagnostic geometry report.",
    )
    p_report.add_argument("input", help="Path to input XYZ file.")

    # ── benchmark ──────────────────────────────────────────────────────
    p_bench = sub.add_parser(
        "benchmark",
        help="Run the built-in benchmark suite.",
    )
    p_bench.add_argument(
        "--suite", choices=["small", "medium"], default="small",
        help="Benchmark suite to run (default: small).",
    )
    p_bench.add_argument(
        "--out", "-o", default=None,
        help="Path to save results CSV (default: outputs/geoinit_v1_0_geometric_summary.csv).",
    )
    p_bench.add_argument(
        "--maxiter", type=int, default=500,
        help="Maximum optimizer iterations per molecule (default: 500).",
    )
    p_bench.add_argument(
        "--sigma", type=float, default=0.05,
        help="Bond-length tolerance σ in Å (default: 0.05).",
    )
    p_bench.add_argument(
        "--clash-mode", choices=["exp", "compact"], default="compact",
        help="Clash potential mode (default: compact).",
    )

    # ── benchmark-xtb ──────────────────────────────────────────────────
    p_bench_xtb = sub.add_parser(
        "benchmark-xtb",
        help="Run the downstream xTB optimization benchmark comparing raw vs GeoInit.",
    )
    p_bench_xtb.add_argument(
        "--suite", choices=["small", "medium"], default="medium",
        help="Benchmark suite to run (default: medium).",
    )
    p_bench_xtb.add_argument(
        "--out", "-o", default=None,
        help="Path to save results CSV (default: preliminary_results_v1_0/geoinit_v1_0_casewise_results.csv).",
    )
    p_bench_xtb.add_argument(
        "--charge", type=int, default=0,
        help="Molecular charge for xTB (default: 0).",
    )
    p_bench_xtb.add_argument(
        "--uhf", type=int, default=0,
        help="Unpaired electrons for xTB (default: 0).",
    )
    p_bench_xtb.add_argument(
        "--xtb-method", default="gfn2",
        help="Hamiltonian for xTB (default: gfn2).",
    )
    p_bench_xtb.add_argument(
        "--clash-mode", choices=["exp", "compact"], default="compact",
        help="Clash potential mode (default: compact).",
    )
    p_bench_xtb.add_argument(
        "--distortions", type=int, default=10,
        help="Number of random distortions per system (default: 10).",
    )
    p_bench_xtb.add_argument(
        "--baselines", nargs="+", default=["raw", "uff", "geoinit_guarded"],
        help="Baselines to run (default: raw uff geoinit_guarded).",
    )
    p_bench_xtb.add_argument(
        "--select-version", choices=["v0_8", "v0_9", "v1_0"], default="v1_0",
        help="GeoInit policy version when geoinit_select is included (default: v1_0).",
    )
    p_bench_xtb.add_argument(
        "--workers", "-j", type=int, default=1,
        help="Parallel worker processes for trials (default: 1 = serial). "
             "Use -1 for all CPU cores. xTB dominates wall-time, so this scales nearly linearly.",
    )
    p_bench_xtb.add_argument(
        "--engine", choices=["scalar", "auto", "numpy", "torch-cpu", "cuda"], default="scalar",
        help="Compute backend for GeoInit prep (default: scalar = legacy path). "
             "'auto' uses the vectorised engine with size-aware GPU/CPU selection.",
    )
    p_bench_xtb.add_argument(
        "--profile", choices=["v1", "v2"], default="v1",
        help="Physics profile for GeoInit prep (default: v1 = frozen functional; v2 = improved physics).",
    )
    p_bench_xtb.add_argument(
        "--xtb-exe", default="xtb",
        help="xTB launcher for downstream optimisation: 'xtb' (stock CPU, default) "
             "or 'xtbx' (GPU-accelerated WSL front-end). GeoInit prep is identical "
             "either way, so this isolates the engine's effect. NOTE: xtbx has a large "
             "fixed WSL/CUDA launch cost per call and only pays off on large systems "
             "(>=350 atoms) or high-throughput batches — not this small-molecule suite.",
    )
    p_bench_xtb.add_argument(
        "--xtb-timeout", type=float, default=120.0,
        help="Per-call xTB subprocess timeout in seconds (default: 120). "
             "Raise it for the slower xtbx launcher.",
    )

    # ── relax-complex ──────────────────────────────────────────────────
    p_relax_complex = sub.add_parser(
        "relax-complex",
        help="Relax a distorted non-covalent complex using host and guest fragments.",
    )
    p_relax_complex.add_argument("host", help="Path to host XYZ file.")
    p_relax_complex.add_argument("guest", help="Path to guest XYZ file.")
    p_relax_complex.add_argument(
        "--out", "-o", default=None,
        help="Path to write relaxed complex XYZ (default: outputs/complex_relaxed.xyz).",
    )
    p_relax_complex.add_argument(
        "--maxiter", type=int, default=500,
        help="Maximum optimizer iterations for rigid placement (default: 500).",
    )
    p_relax_complex.add_argument(
        "--sigma", type=float, default=0.05,
        help="Bond-length tolerance σ in Å (default: 0.05).",
    )
    p_relax_complex.add_argument(
        "--clash-mode", choices=["exp", "compact"], default="compact",
        help="Clash potential mode (default: compact).",
    )

    return parser


# ── Command handlers ───────────────────────────────────────────────────────

def _cmd_relax(args: argparse.Namespace) -> None:
    """Handle the ``relax`` subcommand."""
    from geoinit.core.io_xyz import read_xyz, write_xyz
    from geoinit.optimize.relax import relax
    from geoinit.optimize.guards import check_geometry

    # Read input
    symbols, coords = read_xyz(args.input)
    n_atoms = len(symbols)

    # Build custom weights if any overrides specified
    weights = {}
    if args.weights_bond is not None:
        weights["bond"] = args.weights_bond
    if args.weights_angle is not None:
        weights["angle"] = args.weights_angle
    if args.weights_clash is not None:
        weights["clash"] = args.weights_clash
    if args.weights_disp is not None:
        weights["disp"] = args.weights_disp
    if not weights:
        weights = None

    # Pre-relaxation check
    from geoinit.core.topology import Topology
    topo = Topology(symbols, coords, scale=1.25, sigma=args.sigma)
    report_before = check_geometry(symbols, coords, topology=topo)

    # Relax
    print(_box_header("GeoInit Relax"))
    print(_box_line("Input", args.input))
    print(_box_line("Atoms", str(n_atoms)))
    print(_box_line("σ", f"{args.sigma:.3f} Å"))
    print(_box_line("Max iterations", str(args.maxiter)))
    print(_box_line("Mode", args.mode))
    print(_box_separator())

    if args.mode in {"select", "select-v0.9", "select-v1.0"}:
        from geoinit.optimize.selector import (
            select_initial_geometry,
            v0_9_selection_policy,
            v1_0_selection_policy,
        )

        t0 = time.perf_counter()
        if args.mode in {"select", "select-v1.0"}:
            policy = v1_0_selection_policy()
        elif args.mode == "select-v0.9":
            policy = v0_9_selection_policy()
        else:
            policy = None
        selection = select_initial_geometry(symbols, coords, topology=topo, policy=policy)
        wall = time.perf_counter() - t0
        final_coords_write = selection.selected_coords
        report_after = check_geometry(symbols, final_coords_write, topology=topo)

        print(_box_line("Status", "GeoInit complete"))
        print(_box_line("Selector version", selection.policy.selector_version))
        print(_box_line("Selected", selection.selected_name))
        print(_box_line("Accepted", "yes" if selection.accepted else "raw fallback"))
        print(_box_line("Candidates", str(len(selection.candidates))))
        print(_box_line("Wall time", f"{wall:.3f} s"))
        print(_box_separator())
        print(_box_line("Max bond error (before)", f"{report_before.max_bond_error:.4f} Ã…"))
        print(_box_line("Max bond error (after)", f"{report_after.max_bond_error:.4f} Ã…"))
        print(_box_line("Max clash ratio (before)", f"{report_before.max_clash_ratio:.4f}"))
        print(_box_line("Max clash ratio (after)", f"{report_after.max_clash_ratio:.4f}"))
        print(_box_line("Clashes (before)", str(report_before.n_clashes)))
        print(_box_line("Clashes (after)", str(report_after.n_clashes)))
        print(_box_line("Safe", "yes" if report_after.is_safe else "NO"))
        print(_box_footer())

        if args.out is None:
            base = os.path.splitext(os.path.basename(args.input))[0]
            out_dir = "outputs"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{base}_geoinit_select.xyz")
        else:
            out_path = args.out
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        comment = (
            f"GeoInit {selection.policy.selector_version} | "
            f"selected={selection.selected_name} | accepted={selection.accepted}"
        )
        write_xyz(out_path, symbols, final_coords_write, comment=comment)
        print(f"\n  Output written to: {out_path}")
        return

    t0 = time.perf_counter()
    result = relax(
        symbols, coords,
        weights=weights,
        sigma=args.sigma,
        maxiter=args.maxiter,
        clash_mode=args.clash_mode,
        topology=topo,
    )
    wall = time.perf_counter() - t0

    # Damage guard check
    from geoinit.optimize.guards import check_damage
    is_damaged = check_damage(symbols, coords, result.final_coords, topo)

    final_coords_write = result.final_coords
    status_msg = "✓ converged" if result.success else "✗ not converged"

    if is_damaged:
        print("\n[WARNING] Damage guard triggered: relaxation distorted multiple bonds or separated fragments.")
        if result.rigid_coords is not None:
            print("Falling back to rigid-body optimized coordinates to preserve structure.")
            final_coords_write = result.rigid_coords
            status_msg += " (fallback: rigid-placement)"
        else:
            print("Falling back to raw initial coordinates.")
            final_coords_write = coords
            status_msg += " (fallback: raw)"

    # Post-relaxation check
    report_after = check_geometry(symbols, final_coords_write)

    # Results
    print(_box_line("Status", status_msg))
    print(_box_line("Steps", str(result.n_steps)))
    print(_box_line("Wall time", f"{wall:.3f} s"))
    print(_box_separator())
    print(_box_line("Energy (initial)", f"{result.initial_energy:.6f}"))
    print(_box_line("Energy (final)", f"{result.final_energy:.6f}"))
    print(_box_line("ΔE", f"{result.final_energy - result.initial_energy:.6f}"))
    print(_box_separator())

    # Energy components
    for term, val in result.energy_components.items():
        print(_box_line(f"  {term}", f"{val:.6f}"))

    print(_box_separator())
    print(_box_line("Max bond error (before)", f"{report_before.max_bond_error:.4f} Å"))
    print(_box_line("Max bond error (after)", f"{report_after.max_bond_error:.4f} Å"))
    print(_box_line("Max clash ratio (before)", f"{report_before.max_clash_ratio:.4f}"))
    print(_box_line("Max clash ratio (after)", f"{report_after.max_clash_ratio:.4f}"))
    print(_box_line("Clashes (before)", str(report_before.n_clashes)))
    print(_box_line("Clashes (after)", str(report_after.n_clashes)))
    print(_box_line("Safe", "✓ yes" if report_after.is_safe else "✗ NO"))
    print(_box_footer())

    # Write output
    if args.out is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_dir = "outputs"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}_geoinit.xyz")
    else:
        out_path = args.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    comment = (
        f"GeoInit-V0.2 relaxed | E={result.final_energy:.6f} | "
        f"steps={result.n_steps} | {'converged' if result.success else 'not_converged'}"
    )
    write_xyz(out_path, symbols, final_coords_write, comment=comment)
    print(f"\n  Output written to: {out_path}")


def _cmd_report(args: argparse.Namespace) -> None:
    """Handle the ``report`` subcommand."""
    from geoinit.core.io_xyz import read_xyz
    from geoinit.core.topology import Topology
    from geoinit.core.geometry import distance, angle
    from geoinit.energy.functional import GeoInitFunctional
    from geoinit.optimize.guards import check_geometry

    import numpy as np

    symbols, coords = read_xyz(args.input)
    n_atoms = len(symbols)
    topo = Topology(symbols, coords)
    report = check_geometry(symbols, coords, topology=topo)

    # Energy breakdown
    func = GeoInitFunctional(symbols, coords)
    components = func.energy_components(coords)
    total_e = func.energy(coords)

    print(_box_header("GeoInit Report"))
    print(_box_line("File", args.input))
    print(_box_line("Atoms", str(n_atoms)))
    print(_box_line("Formula", _formula(symbols)))
    print(_box_separator())

    # Topology
    print(_box_line("Bonds detected", str(len(topo.bonds))))
    print(_box_line("Angles detected", str(len(topo.angles))))
    print(_box_line("Nonbonded pairs", str(len(topo.nonbonded_pairs))))
    print(_box_separator())

    # Energy
    print(_box_line("Total energy", f"{total_e:.6f}"))
    for term, val in components.items():
        print(_box_line(f"  {term}", f"{val:.6f}"))
    print(_box_separator())

    # Quality
    print(_box_line("Max bond error", f"{report.max_bond_error:.4f} Å"))
    print(_box_line("Mean bond error", f"{report.mean_bond_error:.4f} Å"))
    print(_box_line("Max clash ratio", f"{report.max_clash_ratio:.4f}"))
    print(_box_line("Clashes (ratio > 1.0)", str(report.n_clashes)))
    print(_box_line("Safe geometry", "✓ yes" if report.is_safe else "✗ NO"))
    print(_box_footer())

    # Bond detail table
    if report.bond_details:
        print("\n  Bond Details:")
        print(f"  {'Bond':<12s} {'r (Å)':>8s} {'r_ref (Å)':>10s} {'Error (Å)':>10s}")
        print(f"  {'─' * 42}")
        for bd in report.bond_details:
            i, j = bd["bond"]
            si, sj = bd["symbols"]
            bond_str = f"{si}{i}-{sj}{j}"
            print(
                f"  {bond_str:<12s} "
                f"{bd['r_ij']:8.4f} {bd['r_ref']:10.4f} {bd['error']:10.4f}"
            )

    # Angle detail (top 5 most distorted)
    if topo.angles:
        from geoinit.energy.angle import angle_energy_decomposed

        angle_info = angle_energy_decomposed(symbols, coords, topo.angles, topo.coordination)
        angle_info.sort(key=lambda x: x["energy"], reverse=True)

        print(f"\n  Top Angle Deviations:")
        print(f"  {'Angle':<16s} {'θ (°)':>8s} {'θ₀ (°)':>8s} {'Energy':>10s}")
        print(f"  {'─' * 44}")
        for ai in angle_info[:min(5, len(angle_info))]:
            i, j, k = ai["angle"]
            label = f"{symbols[i]}{i}-{symbols[j]}{j}-{symbols[k]}{k}"
            print(
                f"  {label:<16s} "
                f"{ai['theta_deg']:8.2f} {ai['theta0_deg']:8.2f} {ai['energy']:10.6f}"
            )


def _cmd_benchmark(args: argparse.Namespace) -> None:
    """Handle the ``benchmark`` subcommand."""
    from geoinit.benchmark.suite import run_suite, save_results, print_summary

    print(_box_header("GeoInit Benchmark"))
    print(_box_line("Suite", args.suite))
    print(_box_line("Max iterations", str(args.maxiter)))
    print(_box_line("σ", f"{args.sigma:.3f} Å"))
    print(_box_footer())
    print()

    df = run_suite(
        suite_name=args.suite,
        sigma=args.sigma,
        maxiter=args.maxiter,
        clash_mode=args.clash_mode,
    )

    if df.empty:
        print("  No results produced.")
        return

    print()
    print_summary(df)

    # Save CSV
    out_path = args.out or f"outputs/geoinit_v1_0_geometric_summary.csv"
    save_results(df, out_path)
    print(f"\n  Results saved to: {out_path}")


def _cmd_benchmark_xtb(args: argparse.Namespace) -> None:
    """Handle the ``benchmark-xtb`` subcommand."""
    from geoinit.benchmark.xtb_runner import run_hardened_benchmark
    run_hardened_benchmark(
        suite_name=args.suite,
        charge=args.charge,
        uhf=args.uhf,
        xtb_method=args.xtb_method,
        distortions=args.distortions,
        out_path=args.out,
        clash_mode=args.clash_mode,
        baselines=args.baselines,
        select_version=args.select_version,
        workers=args.workers,
        engine=args.engine,
        profile=args.profile,
        xtb_cmd=args.xtb_exe,
        xtb_timeout=args.xtb_timeout,
    )



def _cmd_relax_complex(args: argparse.Namespace) -> None:
    """Handle the ``relax-complex`` subcommand."""
    from geoinit.core.io_xyz import read_xyz, write_xyz
    from geoinit.optimize.relax import relax
    from geoinit.optimize.guards import check_geometry
    import numpy as np

    host_syms, host_coords = read_xyz(args.host)
    guest_syms, guest_coords = read_xyz(args.guest)

    symbols = host_syms + guest_syms
    coords = np.vstack([host_coords, guest_coords])
    n_atoms = len(symbols)

    print(_box_header("GeoInit Relax Complex"))
    print(_box_line("Host Input", args.host))
    print(_box_line("Guest Input", args.guest))
    print(_box_line("Total Atoms", str(n_atoms)))
    print(_box_line("σ", f"{args.sigma:.3f} Å"))
    print(_box_line("Max iterations", str(args.maxiter)))
    print(_box_line("Clash Mode", args.clash_mode))
    print(_box_separator())

    # Pre-relaxation check
    report_before = check_geometry(symbols, coords)

    t0 = time.perf_counter()
    result = relax(
        symbols, coords,
        sigma=args.sigma,
        maxiter=args.maxiter,
        clash_mode=args.clash_mode,
    )
    wall = time.perf_counter() - t0

    # Damage guard check
    from geoinit.optimize.guards import check_damage
    from geoinit.core.topology import Topology
    topo = Topology(symbols, coords, scale=1.25, sigma=args.sigma)
    is_damaged = check_damage(symbols, coords, result.final_coords, topo)

    final_coords_write = result.final_coords
    status_msg = "✓ optimized"
    if is_damaged:
        print("\n[WARNING] Damage guard triggered: relaxation distorted multiple bonds or separated fragments.")
        if result.rigid_coords is not None:
            print("Falling back to rigid-body optimized coordinates to preserve structure.")
            final_coords_write = result.rigid_coords
            status_msg += " (fallback: rigid-placement)"
        else:
            print("Falling back to raw initial coordinates.")
            final_coords_write = coords
            status_msg += " (fallback: raw)"

    # Post-relaxation check
    report_after = check_geometry(symbols, final_coords_write)

    print(_box_line("Status", status_msg))
    print(_box_line("Wall time", f"{wall:.3f} s"))
    print(_box_separator())
    print(_box_line("Energy (initial)", f"{result.initial_energy:.6f}"))
    print(_box_line("Energy (final)", f"{result.final_energy:.6f}"))
    print(_box_line("ΔE", f"{result.final_energy - result.initial_energy:.6f}"))
    print(_box_separator())
    print(_box_line("Max bond error (before)", f"{report_before.max_bond_error:.4f} Å"))
    print(_box_line("Max bond error (after)", f"{report_after.max_bond_error:.4f} Å"))
    print(_box_line("Max clash ratio (before)", f"{report_before.max_clash_ratio:.4f}"))
    print(_box_line("Max clash ratio (after)", f"{report_after.max_clash_ratio:.4f}"))
    print(_box_line("Clashes (before)", str(report_before.n_clashes)))
    print(_box_line("Clashes (after)", str(report_after.n_clashes)))
    print(_box_line("Safe", "✓ yes" if report_after.is_safe else "✗ NO"))
    print(_box_footer())

    out_path = args.out or "outputs/complex_relaxed.xyz"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_xyz(out_path, symbols, final_coords_write, comment="GeoInit V0.2 complex relaxed")
    print(f"\n  Complex output written to: {out_path}")


# ── Utility ────────────────────────────────────────────────────────────────

def _formula(symbols: list[str]) -> str:
    """Return a molecular formula string like C2H6O."""
    from collections import Counter
    counts = Counter(symbols)
    # Standard chemical formula ordering: C, H, then alphabetical
    parts = []
    for el in ["C", "H"]:
        if el in counts:
            n = counts.pop(el)
            parts.append(f"{el}{n}" if n > 1 else el)
    for el in sorted(counts):
        n = counts[el]
        parts.append(f"{el}{n}" if n > 1 else el)
    return "".join(parts)


# ── Entry point ────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """Entry point for the GeoInit CLI."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "relax":
        _cmd_relax(args)
    elif args.command == "relax-complex":
        _cmd_relax_complex(args)
    elif args.command == "report":
        _cmd_report(args)
    elif args.command == "benchmark":
        _cmd_benchmark(args)
    elif args.command == "benchmark-xtb":
        _cmd_benchmark_xtb(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
