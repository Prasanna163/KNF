"""xTB runner for GeoInit benchmarking.

This module shells out to xTB to optimize geometries and parses results
to compare optimization steps, energy convergence, and wall times.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
import numpy as np


def check_xtb_available(xtb_cmd: str = "xtb") -> bool:
    """Check whether the requested xTB launcher is on ``PATH``.

    Parameters
    ----------
    xtb_cmd : str
        Name (or path) of the launcher to look for, e.g. ``"xtb"`` for the
        stock CPU build or ``"xtbx"`` for the GPU-accelerated front-end.

    Returns
    -------
    bool
        *True* if the launcher (resolving ``.exe``/``.cmd``/``.bat`` on
        Windows) is found, *False* otherwise.
    """
    return shutil.which(xtb_cmd) is not None


def _xtb_invocation(xtb_cmd: str) -> list[str] | None:
    """Resolve ``xtb_cmd`` to an argv prefix suitable for :func:`subprocess.run`.

    ``shutil.which`` finds the launcher on ``PATH`` (including ``.cmd``/``.bat``
    shims such as the ``xtbx`` GPU front-end, which wraps a WSL call). Windows'
    ``CreateProcess`` cannot execute a batch file directly, so those are invoked
    through ``cmd /c``. A resolved ``.exe`` (or any POSIX binary) is returned
    as-is. Returns *None* when the launcher is not found.
    """
    resolved = shutil.which(xtb_cmd)
    if resolved is None:
        return None
    if os.name == "nt" and resolved.lower().endswith((".cmd", ".bat")):
        return ["cmd", "/c", resolved]
    return [resolved]


def run_xtb_opt(
    xyz_path: str,
    output_dir: str,
    charge: int = 0,
    uhf: int = 0,
    method: str = "gfn2",
    xtb_cmd: str = "xtb",
    timeout: float = 120.0,
) -> dict | None:
    """Run an xTB geometry optimisation.

    Parameters
    ----------
    xyz_path : str
        Path to the input ``.xyz`` file.
    output_dir : str
        Directory where xTB output files will be written.
    charge : int
        Molecular charge (default 0).
    uhf : int
        Number of unpaired electrons (default 0).
    method : str
        xTB Hamiltonian: ``"gfn0"``, ``"gfn1"``, ``"gfn2"`` (default),
        or ``"gfnff"``.
    xtb_cmd : str
        Launcher to invoke: ``"xtb"`` (stock CPU build, default) or
        ``"xtbx"`` (GPU-accelerated front-end). Both write the same
        ``xtbopt.xyz``/``xtbopt.log`` scratch files, so parsing is identical.
    timeout : float
        Per-call subprocess timeout in seconds (default 120). The ``xtbx``
        WSL/CUDA launcher has a large fixed startup cost, so give it a
        larger budget when benchmarking that engine.

    Returns
    -------
    dict or None
        Dict containing keys:
        - success : bool
        - steps : int
        - walltime_s : float
        - energy_hartree : float or None
        - gradient_norm : float or None
        - final_xyz : str or None
        - failure_reason : str
    """
    invocation = _xtb_invocation(xtb_cmd)
    if invocation is None:
        print(f"Warning: '{xtb_cmd}' not found on PATH. xTB benchmarking unavailable.")
        return None

    xyz_path_obj = Path(xyz_path)
    if not xyz_path_obj.is_file():
        raise FileNotFoundError(f"Input XYZ file not found: {xyz_path}")

    # Create unique scratch folder inside output_dir to prevent collision.  The
    # PID + uuid suffix makes this safe under process-parallel benchmarking, where
    # many xTB runs execute concurrently and would otherwise clobber each other's
    # xtbopt.xyz / xtbrestart scratch files.
    out_dir_path = Path(output_dir)
    scratch_dir = out_dir_path / f"scratch_xtb_{xyz_path_obj.stem}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Copy input XYZ to scratch
    input_xyz = scratch_dir / "input.xyz"
    shutil.copy(xyz_path, input_xyz)

    # Build command. The launcher argv prefix (resolved above) is followed by
    # the identical xTB arguments regardless of engine, so ``xtb`` and ``xtbx``
    # produce the same scratch files (xtbopt.xyz / xtbopt.log) to parse.
    cmd = invocation + [
        "input.xyz",
        "--opt",
        "--chrg", str(charge),
        "--uhf", str(uhf),
        f"--{method}"
    ]

    t0 = time.perf_counter()
    try:
        # Run xTB (with a timeout to prevent hanging)
        res = subprocess.run(
            cmd,
            cwd=str(scratch_dir),
            capture_output=True,
            timeout=timeout
        )
        walltime = time.perf_counter() - t0
        stdout = res.stdout.decode("utf-8", errors="replace")
        stderr = res.stderr.decode("utf-8", errors="replace")
        exit_code = res.returncode
    except subprocess.TimeoutExpired:
        shutil.rmtree(scratch_dir, ignore_errors=True)
        return {
            "success": False,
            "steps": 0,
            "walltime_s": float(timeout),
            "energy_hartree": None,
            "gradient_norm": None,
            "final_xyz": None,
            "failure_reason": "timeout"
        }

    # Parse stdout and logs
    success = ("normal termination of xtb" in stdout or "normal termination of xtb" in stderr) and exit_code == 0

    # Parse final energy & gradient norm
    energy = None
    gnorm = None
    for line in stdout.splitlines():
        if "TOTAL ENERGY" in line:
            parts = line.split()
            for part in reversed(parts):
                clean = part.strip("|: ")
                try:
                    energy = float(clean)
                    break
                except ValueError:
                    pass
        if "GRADIENT NORM" in line:
            parts = line.split()
            for part in reversed(parts):
                clean = part.strip("|: ")
                try:
                    gnorm = float(clean)
                    break
                except ValueError:
                    pass

    # Parse steps from xtbopt.log
    steps = 0
    log_file = scratch_dir / "xtbopt.log"
    if log_file.is_file():
        try:
            content = log_file.read_text(encoding="utf-8")
            steps = max(0, content.count("energy:") - 1)
        except Exception:
            pass

    # Check optimization convergence criteria satisfied
    opt_converged = False
    for line in stdout.splitlines():
        if "GEOMETRY OPTIMIZATION CONVERGED" in line.upper():
            opt_converged = True
            break

    # Save final optimized geometry
    final_xyz_dest = None
    opt_xyz = scratch_dir / "xtbopt.xyz"
    if opt_xyz.is_file():
        final_xyz_dest = out_dir_path / f"{xyz_path_obj.stem}_xtb_opt.xyz"
        shutil.copy(opt_xyz, final_xyz_dest)

    # Save full xtb stdout log
    xtb_log_path = out_dir_path / f"{xyz_path_obj.stem}_xtb.log"
    xtb_log_path.write_text(stdout + "\n" + stderr, encoding="utf-8")

    # Clean up scratch folder
    shutil.rmtree(scratch_dir, ignore_errors=True)

    # Deduce failure reason
    failure_reason = ""
    if not success:
        failure_reason = "execution error"
    elif not opt_converged:
        failure_reason = "not converged"

    return {
        "success": success and opt_converged,
        "steps": steps,
        "walltime_s": round(walltime, 4),
        "energy_hartree": energy,
        "gradient_norm": gnorm,
        "final_xyz": str(final_xyz_dest) if final_xyz_dest else None,
        "failure_reason": failure_reason
    }


def _execute_trial(task: dict) -> dict:
    """Run every baseline for a single (molecule, distortion) trial.

    This is the unit of work for the benchmark.  It is intentionally pure and
    self-contained — it rebuilds the (deterministic) topology from the reference
    coordinates and writes/cleans up only files unique to its own
    ``(molecule, distortion_idx)`` — so it can be dispatched across processes
    with no shared state.  Returns a dict with ``trial_row`` (or ``None`` if the
    raw xTB run failed), ``selection_rows`` and a ``log`` line.
    """
    import time as _time
    from pathlib import Path as _Path

    import numpy as _np

    from geoinit.core.atoms import Molecule
    from geoinit.core.topology import Topology
    from geoinit.benchmark.uff_runner import run_uff_opt
    from geoinit.optimize.relax import relax
    from geoinit.optimize.guards import accept_geoinit, check_damage
    from geoinit.optimize.selector import select_initial_geometry, v0_9_selection_policy, v1_0_selection_policy
    from geoinit.benchmark.select_report import selection_result_rows

    name = task["name"]
    d_idx = task["d_idx"]
    symbols = task["symbols"]
    ref_coords = _np.asarray(task["ref_coords"], dtype=_np.float64)
    dist_coords = _np.asarray(task["dist_coords"], dtype=_np.float64)
    chemical_class = task["chemical_class"]
    baselines = task["baselines"]
    select_version = task["select_version"]
    clash_mode = task["clash_mode"]
    charge, uhf, xtb_method = task["charge"], task["uhf"], task["xtb_method"]
    out_dir = _Path(task["out_dir"])
    engine = task.get("engine", "scalar")
    profile = task.get("profile", "v1")
    backend = task.get("backend", "auto")
    xtb_cmd = task.get("xtb_cmd", "xtb")
    xtb_timeout = task.get("xtb_timeout", 120.0)

    # Deterministic topology rebuild (identical to the sequential driver).
    topo = Topology(symbols, ref_coords, scale=1.25, sigma=0.05)

    dist_xyz_path = out_dir / f"{name}_trial_{d_idx}_distorted.xyz"
    Molecule(symbols=symbols, coords=dist_coords).to_xyz(str(dist_xyz_path))

    # ── BASELINE 1: RAW ──
    raw_res = run_xtb_opt(str(dist_xyz_path), str(out_dir), charge, uhf, xtb_method,
                          xtb_cmd=xtb_cmd, timeout=xtb_timeout)
    if not raw_res:
        if dist_xyz_path.is_file():
            dist_xyz_path.unlink()
        return {"trial_row": None, "selection_rows": [],
                "log": f"    Trial {d_idx}: FAIL (Raw xTB execution error)"}

    # ── BASELINE 2: UFF ──
    uff_xyz_path = out_dir / f"{name}_trial_{d_idx}_uff.xyz"
    try:
        uff_prep_time = run_uff_opt(symbols, dist_coords, topo, str(uff_xyz_path))
        uff_res = run_xtb_opt(str(uff_xyz_path), str(out_dir), charge, uhf, xtb_method,
                              xtb_cmd=xtb_cmd, timeout=xtb_timeout)
    except Exception as exc:
        uff_prep_time = 0.0
        uff_res = {"success": False, "steps": 0, "walltime_s": 0.0,
                   "energy_hartree": None, "failure_reason": f"UFF setup error: {exc}"}

    # ── BASELINE 3: GeoInit Guarded ──
    t0 = _time.perf_counter()
    result = relax(symbols, dist_coords, sigma=0.05, maxiter=500, clash_mode=clash_mode,
                   topology=topo, mode="fast", engine=engine, profile=profile, backend=backend)
    geo_prep_time = _time.perf_counter() - t0

    is_damaged = check_damage(symbols, dist_coords, result.final_coords, topo)
    final_relaxed = result.rigid_coords if (is_damaged and result.rigid_coords is not None) else result.final_coords
    accepted, reason = accept_geoinit(symbols, dist_coords, final_relaxed, topo)

    if accepted:
        geo_xyz_path = out_dir / f"{name}_trial_{d_idx}_geoinit.xyz"
        Molecule(symbols=symbols, coords=final_relaxed).to_xyz(str(geo_xyz_path))
        geo_res = run_xtb_opt(str(geo_xyz_path), str(out_dir), charge, uhf, xtb_method,
                              xtb_cmd=xtb_cmd, timeout=xtb_timeout)
        if geo_res:
            guarded_success = geo_res["success"]; guarded_steps = geo_res["steps"]
            guarded_xtb_time = geo_res["walltime_s"]; guarded_energy = geo_res["energy_hartree"]
            guarded_failure_reason = geo_res["failure_reason"]
        else:
            guarded_success = False; guarded_steps = 0; guarded_xtb_time = 0.0
            guarded_energy = None; guarded_failure_reason = "execution error"
        if geo_xyz_path.is_file():
            geo_xyz_path.unlink()
    else:
        guarded_success = raw_res["success"]; guarded_steps = raw_res["steps"]
        guarded_xtb_time = raw_res["walltime_s"]; guarded_energy = raw_res["energy_hartree"]
        guarded_failure_reason = raw_res["failure_reason"]

    guarded_total_time = geo_prep_time + (guarded_xtb_time if accepted else raw_res["walltime_s"])

    # ── OPTIONAL BASELINE 4: GeoInit ──
    select_prep_time = 0.0
    select_success = raw_res["success"]; select_steps = raw_res["steps"]
    select_xtb_time = raw_res["walltime_s"]; select_energy = raw_res["energy_hartree"]
    select_failure_reason = raw_res["failure_reason"]
    select_accepted = False; select_candidate = "raw"; selection = None
    if "geoinit_select" in baselines:
        t0 = _time.perf_counter()
        if select_version in {"v0_9", "v0_9"}:
            policy = v0_9_selection_policy()
        elif select_version in {"v1_0", "v1_0"}:
            policy = v1_0_selection_policy()
        else:
            policy = None
        selection = select_initial_geometry(symbols, dist_coords, topology=topo, policy=policy)
        select_prep_time = _time.perf_counter() - t0
        select_accepted = selection.accepted
        select_candidate = selection.selected_name
        if selection.accepted:
            select_xyz_path = out_dir / f"{name}_trial_{d_idx}_geoinit_select.xyz"
            Molecule(symbols=symbols, coords=selection.selected_coords).to_xyz(str(select_xyz_path))
            select_res = run_xtb_opt(str(select_xyz_path), str(out_dir), charge, uhf, xtb_method,
                                     xtb_cmd=xtb_cmd, timeout=xtb_timeout)
            if select_res:
                select_success = select_res["success"]; select_steps = select_res["steps"]
                select_xtb_time = select_res["walltime_s"]; select_energy = select_res["energy_hartree"]
                select_failure_reason = select_res["failure_reason"]
            else:
                select_success = False; select_steps = 0; select_xtb_time = 0.0
                select_energy = None; select_failure_reason = "execution error"
            if select_xyz_path.is_file():
                select_xyz_path.unlink()

    select_total_time = select_prep_time + (select_xtb_time if select_accepted else raw_res["walltime_s"])

    # ── Energy Gaps ──
    def _gap(energy):
        if energy is not None and raw_res["energy_hartree"] is not None:
            gap = energy - raw_res["energy_hartree"]
            kcal = gap * 627.509
            return gap, kcal, bool(abs(kcal) < 0.1)
        return None, None, False

    uff_gap_Eh, uff_gap_kcal, uff_same_min = _gap(uff_res["energy_hartree"])
    guard_gap_Eh, guard_gap_kcal, guard_same_min = _gap(guarded_energy)
    select_gap_Eh, select_gap_kcal, select_same_min = _gap(select_energy)

    # ── Net Benefit ──
    net_time_saving_s = raw_res["walltime_s"] - guarded_total_time
    net_time_saving_pct = (net_time_saving_s / raw_res["walltime_s"] * 100) if raw_res["walltime_s"] > 0 else 0.0
    if not guarded_success or not guard_same_min:
        net_benefit_cat = "different_basin_risk"
    elif net_time_saving_s > 0.0:
        net_benefit_cat = "net_win"
    else:
        net_benefit_cat = "net_loss"

    trial_row = {
        "molecule": name, "distortion_idx": d_idx,
        "raw_success": raw_res["success"], "raw_steps": raw_res["steps"],
        "raw_xtb_time": raw_res["walltime_s"], "raw_total_time": raw_res["walltime_s"],
        "raw_energy": raw_res["energy_hartree"], "raw_failure_reason": raw_res["failure_reason"],
        "uff_success": uff_res["success"], "uff_steps": uff_res["steps"],
        "uff_prep_time": round(uff_prep_time, 4), "uff_xtb_time": uff_res["walltime_s"],
        "uff_total_time": round(uff_prep_time + uff_res["walltime_s"], 4),
        "uff_energy": uff_res["energy_hartree"], "uff_failure_reason": uff_res["failure_reason"],
        "uff_energy_gap": uff_gap_Eh, "uff_energy_gap_kcal": uff_gap_kcal, "uff_same_min": uff_same_min,
        "geoinit_success": guarded_success, "geoinit_steps": guarded_steps,
        "geoinit_prep_time": round(geo_prep_time, 4), "geoinit_xtb_time": guarded_xtb_time,
        "geoinit_total_time": round(guarded_total_time, 4), "geoinit_energy": guarded_energy,
        "geoinit_failure_reason": guarded_failure_reason, "geoinit_accepted": accepted,
        "geoinit_fallback": not accepted, "geoinit_fallback_reason": reason,
        "geoinit_energy_gap": guard_gap_Eh, "geoinit_energy_gap_kcal": guard_gap_kcal,
        "geoinit_same_min": guard_same_min,
        "geoinit_select_success": select_success, "geoinit_select_steps": select_steps,
        "geoinit_select_prep_time": round(select_prep_time, 4), "geoinit_select_xtb_time": select_xtb_time,
        "geoinit_select_total_time": round(select_total_time, 4), "geoinit_select_energy": select_energy,
        "geoinit_select_failure_reason": select_failure_reason, "geoinit_select_accepted": select_accepted,
        "geoinit_select_candidate": select_candidate,
        "geoinit_select_fallback_reason": selection.fallback_reason if selection is not None else "not_run",
        "geoinit_select_version": select_version if "geoinit_select" in baselines else "",
        "chemical_class": chemical_class,
        "geoinit_select_energy_gap": select_gap_Eh, "geoinit_select_energy_gap_kcal": select_gap_kcal,
        "geoinit_select_same_min": select_same_min,
        "net_time_saving_s": round(net_time_saving_s, 4),
        "net_time_saving_pct": round(net_time_saving_pct, 2),
        "net_benefit_category": net_benefit_cat,
    }

    selection_rows = []
    if selection is not None:
        case_name = f"{name}_trial_{d_idx}"
        for row in selection_result_rows(selection, case_name=case_name):
            row["molecule"] = name
            row["distortion_idx"] = d_idx
            row["chemical_class"] = chemical_class
            row["selected_same_min"] = select_same_min if row["selected"] else ""
            row["selected_net_time_saving_s"] = raw_res["walltime_s"] - select_total_time if row["selected"] else ""
            selection_rows.append(row)

    # Cleanup all files for this trial.
    for path in [
        dist_xyz_path, uff_xyz_path,
        out_dir / f"{name}_trial_{d_idx}_distorted_xtb_opt.xyz",
        out_dir / f"{name}_trial_{d_idx}_distorted_xtb.log",
        out_dir / f"{name}_trial_{d_idx}_uff_xtb_opt.xyz",
        out_dir / f"{name}_trial_{d_idx}_uff_xtb.log",
        out_dir / f"{name}_trial_{d_idx}_geoinit_xtb_opt.xyz",
        out_dir / f"{name}_trial_{d_idx}_geoinit_xtb.log",
        out_dir / f"{name}_trial_{d_idx}_geoinit_select_xtb_opt.xyz",
        out_dir / f"{name}_trial_{d_idx}_geoinit_select_xtb.log",
    ]:
        if path.is_file():
            path.unlink()

    log = (f"    Trial {d_idx:2d}: Steps Raw={raw_res['steps']:3d} | "
           f"UFF={uff_res['steps']:3d} | Guarded={guarded_steps:3d} "
           f"({'accept' if accepted else 'fallback'})")
    return {"trial_row": trial_row, "selection_rows": selection_rows, "log": log}


def run_hardened_benchmark(
    suite_name: str = "medium",
    charge: int = 0,
    uhf: int = 0,
    xtb_method: str = "gfn2",
    distortions: int = 10,
    out_path: str | None = None,
    clash_mode: str = "compact",
    baselines: list[str] | None = None,
    select_version: str = "v1_0",
    workers: int = 1,
    engine: str = "scalar",
    profile: str = "v1",
    backend: str = "auto",
    xtb_cmd: str = "xtb",
    xtb_timeout: float = 120.0,
) -> None:
    """Run the hardened benchmark comparing Raw vs UFF vs GeoInit Guarded/Select.

    Generates random distortions for each system and runs repeated GFN2-xTB
    calculations.

    Parameters
    ----------
    xtb_cmd : str
        xTB launcher used for every downstream optimisation: ``"xtb"`` (stock
        CPU build, default) or ``"xtbx"`` (GPU-accelerated WSL front-end). The
        GeoInit prep is identical either way, so this isolates the engine's
        effect on the downstream optimisation.
    xtb_timeout : float
        Per-call subprocess timeout in seconds (default 120).
    """
    import hashlib
    import pandas as pd
    from geoinit.core.atoms import Molecule
    from geoinit.core.topology import Topology
    from geoinit.benchmark.distortion import generate_distorted_coords
    from geoinit.benchmark.uff_runner import run_uff_opt
    from geoinit.optimize.relax import relax
    from geoinit.optimize.guards import accept_geoinit, check_damage, check_geometry
    from geoinit.core.classes import detect_chemical_classes
    from geoinit.optimize.selector import select_initial_geometry
    from geoinit.benchmark.select_report import candidate_classwise_summary, selection_result_rows
    from geoinit.benchmark.suite import SUITES, _pretty_name, _find_project_root

    if not check_xtb_available(xtb_cmd):
        raise RuntimeError(f"Error: xTB launcher '{xtb_cmd}' is not available on PATH.")

    # The summary table uses Unicode box-drawing characters; ensure stdout can
    # encode them on consoles that default to a legacy codepage (e.g. cp1252 on
    # Windows) so the run does not abort with a UnicodeEncodeError at the end.
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    root = _find_project_root()
    if suite_name not in SUITES:
        raise ValueError(f"Unknown suite '{suite_name}'")

    baselines = baselines or ["raw", "uff", "geoinit_guarded"]
    version_aliases = {
        "v0_8": "v0_8",
        "v0_9": "v0_9",
        "v1_0": "v1_0",
    }
    select_version = version_aliases.get(select_version, select_version)
    _select_prefix = {
        "v0_8": "geoinit_v0_8",
        "v0_9": "geoinit_v0_9",
        "v1_0": "geoinit_v1_0",
    }
    _select_dir = {
        "v0_8": "benchmark_results_v0_8",
        "v0_9": "benchmark_results_v0_9",
        "v1_0": "preliminary_results_v1_0",
    }
    if "geoinit_select" in baselines:
        output_prefix = _select_prefix.get(select_version, "geoinit_v1_0")
    else:
        output_prefix = "geoinit_v0_7"
    file_list = SUITES[suite_name]
    if out_path is not None:
        casewise_csv = Path(out_path)
        out_dir = casewise_csv.parent if str(casewise_csv.parent) else Path(".")
    elif "geoinit_select" in baselines:
        out_dir = Path(_select_dir.get(select_version, "preliminary_results_v1_0"))
        casewise_csv = out_dir / f"{output_prefix}_casewise_results.csv"
    else:
        out_dir = Path("outputs")
        casewise_csv = out_dir / f"{output_prefix}_casewise_results.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_label = {"v1_0": "V1.0", "v0_9": "V0.9", "v0_8": "V0.8"}.get(select_version, "V0.7")
    extra = []
    if profile == "v2":
        extra.append("physics=v2")
    if engine != "scalar":
        extra.append(f"engine={engine}")
    if xtb_cmd != "xtb":
        extra.append(f"xtb={xtb_cmd}")
    if workers and int(workers) != 1:
        extra.append(f"workers={workers}")
    suffix = (" [" + ", ".join(extra) + "]") if extra else ""
    print(f"Starting GeoInit {bench_label} Hardened Benchmark "
          f"({distortions} distortions per molecule){suffix}")
    print(f"Baselines: {', '.join(baselines)}\n")

    all_trials = []
    select_candidate_rows: list[dict] = []
    tasks: list[dict] = []
    n_mols = len(file_list)

    def chemical_class_label(classes) -> str:
        if len(classes.fragments) >= 2:
            return "complex"
        if classes.has_kind("amide"):
            return "amide"
        if classes.has_kind("carbonyl"):
            return "carbonyl"
        if classes.has_kind("aromatic_ring"):
            return "aromatic"
        if classes.has_kind("linear_fragment"):
            return "linear_fragment"
        if classes.has_kind("multiple_bond"):
            return "multiple_bond"
        if classes.polar_atoms:
            return "polar_single"
        return "single_molecule"

    for m_idx, rel_path in enumerate(file_list, 1):
        name = _pretty_name(rel_path)
        abs_path = root / rel_path

        if not abs_path.is_file():
            print(f"  [{m_idx}/{n_mols}] {name} … SKIP (file not found)")
            continue

        mol = Molecule.from_xyz(str(abs_path))
        topo = Topology(mol.symbols, mol.coords, scale=1.25, sigma=0.05)
        classes = detect_chemical_classes(mol.symbols, topo)
        chemical_class = chemical_class_label(classes)

        # Draw reproducible seed based on molecule name hash
        h = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % (10**8)
        rng = np.random.default_rng(seed=h)

        print(f"  [{m_idx}/{n_mols}] {name} (formula: {''.join(mol.symbols)})")

        # Pre-generate every distorted geometry *in RNG order* so the work list
        # is identical regardless of how many worker processes execute it.  Each
        # trial is fully independent (own files, own xTB scratch dirs), so the
        # numerical results are byte-for-byte the same whether run serially or
        # across processes — only the wall-clock changes.
        for d_idx in range(distortions):
            dist_coords = generate_distorted_coords(mol.symbols, mol.coords, topo, rng)
            tasks.append({
                "name": name, "d_idx": d_idx,
                "symbols": list(mol.symbols), "ref_coords": np.asarray(mol.coords, dtype=np.float64),
                "dist_coords": dist_coords, "chemical_class": chemical_class,
                "baselines": baselines, "select_version": select_version,
                "clash_mode": clash_mode, "charge": charge, "uhf": uhf,
                "xtb_method": xtb_method, "out_dir": str(out_dir),
                "engine": engine, "profile": profile, "backend": backend,
                "xtb_cmd": xtb_cmd, "xtb_timeout": xtb_timeout,
            })

    # ── Dispatch trials: process-parallel when requested, else serial fallback ──
    # workers == 1 → serial (exact legacy path).  workers > 1 → that many worker
    # processes.  workers < 0 → joblib convention "all cores" (e.g. -1 = every
    # core, -2 = all but one), resolved against os.cpu_count() for the banner.
    n_trials = len(tasks)
    workers = int(workers)
    if workers < 0:
        resolved = max(1, (os.cpu_count() or 1) + 1 + workers)
    else:
        resolved = max(1, workers)
    parallel = resolved > 1 and n_trials > 1
    if parallel:
        try:
            from joblib import Parallel, delayed
            print(f"\nRunning {n_trials} trials across {resolved} worker processes …")
            results = Parallel(n_jobs=workers, backend="loky")(
                delayed(_execute_trial)(t) for t in tasks
            )
        except Exception as exc:
            print(f"  Parallel dispatch unavailable ({exc}); falling back to serial.")
            results = [_execute_trial(t) for t in tasks]
    else:
        print(f"\nRunning {n_trials} trials serially …")
        results = [_execute_trial(t) for t in tasks]

    for res in results:
        if res is None or res.get("trial_row") is None:
            if res is not None:
                print(res["log"])
            continue
        all_trials.append(res["trial_row"])
        select_candidate_rows.extend(res["selection_rows"])
        print(res["log"])


    if not all_trials:
        print("Error: No trials completed successfully.")
        return

    # ── Write Casewise Results ──
    df_trials = pd.DataFrame(all_trials)
    df_trials.to_csv(casewise_csv, index=False)
    print(f"\nDetailed trials results saved to: {casewise_csv}")

    # ── Generate summaries ──
    # 1. Molecule-level steps and total pipeline times
    summary_rows = []
    for m in sorted(df_trials["molecule"].unique()):
        sub = df_trials[df_trials["molecule"] == m]

        summary_rows.append({
            "molecule": m,
            "raw_steps_mean": sub["raw_steps"].mean(),
            "raw_steps_std": sub["raw_steps"].std(),
            "raw_total_time_mean": sub["raw_total_time"].mean(),
            "raw_total_time_std": sub["raw_total_time"].std(),
            "raw_success_rate": sub["raw_success"].mean() * 100,

            "uff_steps_mean": sub["uff_steps"].mean(),
            "uff_steps_std": sub["uff_steps"].std(),
            "uff_total_time_mean": sub["uff_total_time"].mean(),
            "uff_total_time_std": sub["uff_total_time"].std(),
            "uff_success_rate": sub["uff_success"].mean() * 100,
            "uff_same_min_rate": sub["uff_same_min"].mean() * 100,

            "guarded_steps_mean": sub["geoinit_steps"].mean(),
            "guarded_steps_std": sub["geoinit_steps"].std(),
            "guarded_total_time_mean": sub["geoinit_total_time"].mean(),
            "guarded_total_time_std": sub["geoinit_total_time"].std(),
            "guarded_success_rate": sub["geoinit_success"].mean() * 100,
            "guarded_same_min_rate": sub["geoinit_same_min"].mean() * 100,
            "guarded_accept_rate": sub["geoinit_accepted"].mean() * 100,

            "select_steps_mean": sub["geoinit_select_steps"].mean(),
            "select_steps_std": sub["geoinit_select_steps"].std(),
            "select_total_time_mean": sub["geoinit_select_total_time"].mean(),
            "select_total_time_std": sub["geoinit_select_total_time"].std(),
            "select_success_rate": sub["geoinit_select_success"].mean() * 100,
            "select_same_min_rate": sub["geoinit_select_same_min"].mean() * 100,
            "select_accept_rate": sub["geoinit_select_accepted"].mean() * 100,
        })
    df_summary = pd.DataFrame(summary_rows)
    summary_csv = out_dir / f"{output_prefix}_benchmark_summary.csv"
    df_summary.to_csv(summary_csv, index=False, float_format="%.4f")
    print(f"Molecule-level summary saved to: {summary_csv}")

    # 2. Guard Decisions & fallbacks
    reasons = [
        "unsafe_single_molecule",
        "too_short_interfragment_distance",
        "too_many_interfragment_clashes",
        "multiple_bond_damage",
        "linear_fragment_damage",
        "aromatic_planarity_damage",
        "fragment_drift",
    ]
    reasons_counts = {r: int((df_trials["geoinit_fallback_reason"] == r).sum()) for r in reasons}

    # Subset steps & times
    subset_data = []
    # Accepted subset
    sub_acc = df_trials[df_trials["geoinit_accepted"]]
    if not sub_acc.empty:
        subset_data.append({
            "subset": "Accepted",
            "count": len(sub_acc),
            "mean_steps": sub_acc["geoinit_steps"].mean(),
            "mean_time": sub_acc["geoinit_total_time"].mean(),
            "success_rate": sub_acc["geoinit_success"].mean() * 100,
        })
    # Fallback subset
    sub_fall = df_trials[df_trials["geoinit_fallback"]]
    if not sub_fall.empty:
        subset_data.append({
            "subset": "Fallback",
            "count": len(sub_fall),
            "mean_steps": sub_fall["geoinit_steps"].mean(),
            "mean_time": sub_fall["geoinit_total_time"].mean(),
            "success_rate": sub_fall["geoinit_success"].mean() * 100,
        })
    # Combined subset
    subset_data.append({
        "subset": "Combined",
        "count": len(df_trials),
        "mean_steps": df_trials["geoinit_steps"].mean(),
        "mean_time": df_trials["geoinit_total_time"].mean(),
        "success_rate": df_trials["geoinit_success"].mean() * 100,
    })

    select_subset_data = []
    sub_select_acc = df_trials[df_trials["geoinit_select_accepted"]]
    if not sub_select_acc.empty:
        select_subset_data.append({
            "subset": "SelectAccepted",
            "count": len(sub_select_acc),
            "mean_steps": sub_select_acc["geoinit_select_steps"].mean(),
            "mean_time": sub_select_acc["geoinit_select_total_time"].mean(),
            "success_rate": sub_select_acc["geoinit_select_success"].mean() * 100,
        })
    sub_select_raw = df_trials[~df_trials["geoinit_select_accepted"]]
    if not sub_select_raw.empty:
        select_subset_data.append({
            "subset": "SelectRawFallback",
            "count": len(sub_select_raw),
            "mean_steps": sub_select_raw["geoinit_select_steps"].mean(),
            "mean_time": sub_select_raw["geoinit_select_total_time"].mean(),
            "success_rate": sub_select_raw["geoinit_select_success"].mean() * 100,
        })
    select_subset_data.append({
        "subset": "SelectCombined",
        "count": len(df_trials),
        "mean_steps": df_trials["geoinit_select_steps"].mean(),
        "mean_time": df_trials["geoinit_select_total_time"].mean(),
        "success_rate": df_trials["geoinit_select_success"].mean() * 100,
    })

    df_subsets = pd.DataFrame(subset_data)
    df_reasons = pd.DataFrame(list(reasons_counts.items()), columns=["fallback_reason", "count"])

    # Write combined guard summary file
    guard_csv = out_dir / f"{output_prefix}_guard_summary.csv"
    with open(guard_csv, "w", encoding="utf-8") as f:
        f.write("# Guard decisions and subsets summary\n")
        df_subsets.to_csv(f, index=False)
        f.write("\n# Fallback reasons breakdown\n")
        df_reasons.to_csv(f, index=False)
        f.write("\n# GeoInit decisions and subsets summary\n")
        pd.DataFrame(select_subset_data).to_csv(f, index=False)
        f.write("\n# GeoInit selected candidate breakdown\n")
        df_trials["geoinit_select_candidate"].value_counts().rename_axis("candidate").reset_index(name="count").to_csv(f, index=False)
    print(f"Guard summary saved to: {guard_csv}")

    # 3. Energy Gap classifications
    def classify_energy_gap(val_kcal):
        if val_kcal is None:
            return "unknown"
        abs_val = abs(val_kcal)
        if abs_val < 0.1:
            return "< 0.1 kcal/mol (same minimum)"
        elif abs_val < 0.5:
            return "0.1 to 0.5 kcal/mol (probably acceptable)"
        elif abs_val < 1.0:
            return "0.5 to 1.0 kcal/mol (inspect)"
        else:
            return "> 1.0 kcal/mol (different basin risk)"

    categories = [
        "< 0.1 kcal/mol (same minimum)",
        "0.1 to 0.5 kcal/mol (probably acceptable)",
        "0.5 to 1.0 kcal/mol (inspect)",
        "> 1.0 kcal/mol (different basin risk)"
    ]

    gap_rows = []
    total_runs = len(df_trials)
    for cat in categories:
        u_count = sum(df_trials["uff_energy_gap_kcal"].apply(classify_energy_gap) == cat)
        g_count = sum(df_trials["geoinit_energy_gap_kcal"].apply(classify_energy_gap) == cat)
        s_count = sum(df_trials["geoinit_select_energy_gap_kcal"].apply(classify_energy_gap) == cat)
        gap_rows.append({
            "category": cat,
            "guarded_count": g_count,
            "guarded_percentage": (g_count / total_runs) * 100,
            "select_count": s_count,
            "select_percentage": (s_count / total_runs) * 100,
            "uff_count": u_count,
            "uff_percentage": (u_count / total_runs) * 100,
        })
    df_gap = pd.DataFrame(gap_rows)
    gap_csv = out_dir / f"{output_prefix}_energy_gap.csv"
    df_gap.to_csv(gap_csv, index=False, float_format="%.2f")
    print(f"Energy gap categorization saved to: {gap_csv}")

    # 4. Net Benefit Summary
    benefit_cats = ["net_win", "net_loss", "different_basin_risk"]
    benefit_rows = []
    for cat in benefit_cats:
        count = int((df_trials["net_benefit_category"] == cat).sum())
        benefit_rows.append({
            "category": cat,
            "count": count,
            "percentage": (count / total_runs) * 100
        })
    df_benefit = pd.DataFrame(benefit_rows)
    benefit_csv = out_dir / f"{output_prefix}_net_benefit_summary.csv"
    df_benefit.to_csv(benefit_csv, index=False, float_format="%.2f")
    print(f"Net benefit summary saved to: {benefit_csv}")

    select_categories = []
    for ok, same, raw_time, select_time in zip(
        df_trials["geoinit_select_success"],
        df_trials["geoinit_select_same_min"],
        df_trials["raw_total_time"],
        df_trials["geoinit_select_total_time"],
    ):
        if not ok or not same:
            select_categories.append("different_basin_risk")
        elif raw_time - select_time > 0.0:
            select_categories.append("net_win")
        else:
            select_categories.append("net_loss")
    df_trials["geoinit_select_net_benefit_category"] = select_categories
    select_benefit_rows = []
    for cat in benefit_cats:
        count = int((df_trials["geoinit_select_net_benefit_category"] == cat).sum())
        select_benefit_rows.append({
            "category": cat,
            "count": count,
            "percentage": (count / total_runs) * 100,
        })
    select_benefit_csv = out_dir / f"{output_prefix}_select_net_benefit_summary.csv"
    pd.DataFrame(select_benefit_rows).to_csv(select_benefit_csv, index=False, float_format="%.2f")
    if select_candidate_rows:
        candidate_rows_csv = out_dir / f"{output_prefix}_candidate_rows.csv"
        pd.DataFrame(select_candidate_rows).to_csv(candidate_rows_csv, index=False, float_format="%.6f")

        candidate_class_csv = out_dir / f"{output_prefix}_candidate_class_summary.csv"
        candidate_classwise_summary(df_trials.to_dict("records")).to_csv(candidate_class_csv, index=False)
        print(f"GeoInit candidate rows saved to: {candidate_rows_csv}")
        print(f"GeoInit candidate class summary saved to: {candidate_class_csv}")
    df_trials.to_csv(casewise_csv, index=False)
    print(f"GeoInit net benefit summary saved to: {select_benefit_csv}")

    # ── Print beautiful ASCII aggregate table ──
    raw_tot_steps = int(df_trials["raw_steps"].sum())
    uff_tot_steps = int(df_trials["uff_steps"].sum())
    guard_tot_steps = int(df_trials["geoinit_steps"].sum())

    raw_tot_time = float(df_trials["raw_total_time"].sum())
    uff_tot_time = float(df_trials["uff_total_time"].sum())
    guard_tot_time = float(df_trials["geoinit_total_time"].sum())

    raw_success_count = int(df_trials["raw_success"].sum())
    uff_success_count = int(df_trials["uff_success"].sum())
    guard_success_count = int(df_trials["geoinit_success"].sum())
    total_cases = len(df_trials)

    uff_step_pct = (raw_tot_steps - uff_tot_steps) / raw_tot_steps * 100
    guard_step_pct = (raw_tot_steps - guard_tot_steps) / raw_tot_steps * 100

    uff_time_pct = (raw_tot_time - uff_tot_time) / raw_tot_time * 100
    guard_time_pct = (raw_tot_time - guard_tot_time) / raw_tot_time * 100

    print("\n  ┌───────────────────────────────────────────────────────────────────────────────────┐")
    print(f"  │ xTB Hardened Benchmark Summary Comparison ({total_cases:3d} total trials)             │")
    print("  ├──────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┬───────┤")
    print("  │ Mode                 │ Success     │ Total Steps │ Step Saving │ Total Time  │ Savings│")
    print("  ├──────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┼───────┤")
    guard_label = "GeoInit V1.0 Guarded" if profile == "v2" else "GeoInit V0.7 Guarded"
    print(f"  │ Raw                  │ {raw_success_count:3d}/{total_cases:3d} ({raw_success_count/total_cases*100:4.1f}%)│ {raw_tot_steps:11d} │    baseline │ {raw_tot_time:10.2f}s │    0.0% │")
    print(f"  │ UFF                  │ {uff_success_count:3d}/{total_cases:3d} ({uff_success_count/total_cases*100:4.1f}%)│ {uff_tot_steps:11d} │      {uff_step_pct:+.1f}% │ {uff_tot_time:10.2f}s │  {uff_time_pct:+.1f}% │")
    print(f"  │ {guard_label} │ {guard_success_count:3d}/{total_cases:3d} ({guard_success_count/total_cases*100:4.1f}%)│ {guard_tot_steps:11d} │      {guard_step_pct:+.1f}% │ {guard_tot_time:10.2f}s │  {guard_time_pct:+.1f}% │")
    if "geoinit_select" in baselines:
        sel_steps = int(df_trials["geoinit_select_steps"].sum())
        sel_time = float(df_trials["geoinit_select_total_time"].sum())
        sel_succ = int(df_trials["geoinit_select_success"].sum())
        sel_step_pct = (raw_tot_steps - sel_steps) / raw_tot_steps * 100 if raw_tot_steps else 0.0
        sel_time_pct = (raw_tot_time - sel_time) / raw_tot_time * 100 if raw_tot_time else 0.0
        sel_label = {"v1_0": "GeoInit V1.0 ", "v0_9": "GeoInit V0.9 "}.get(
            select_version, "GeoInit V0.8 ")
        print(f"  │ {sel_label} │ {sel_succ:3d}/{total_cases:3d} ({sel_succ/total_cases*100:4.1f}%)│ {sel_steps:11d} │      {sel_step_pct:+.1f}% │ {sel_time:10.2f}s │  {sel_time_pct:+.1f}% │")
    print("  └──────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┴───────┘")

    # Basin-safety headline (the metric that matters most): same-minimum rates.
    guard_same = int(df_trials["geoinit_same_min"].sum())
    uff_same = int(df_trials["uff_same_min"].sum())
    print("\n  Same-minimum (basin-safety) rates vs raw xTB:")
    print(f"    UFF                 : {uff_same:3d}/{total_cases:3d} ({uff_same/total_cases*100:5.1f}%)   "
          f"different-basin risk: {total_cases - uff_same}/{total_cases}")
    print(f"    {guard_label}: {guard_same:3d}/{total_cases:3d} ({guard_same/total_cases*100:5.1f}%)   "
          f"different-basin risk: {total_cases - guard_same}/{total_cases}")
    if "geoinit_select" in baselines:
        sel_same = int(df_trials["geoinit_select_same_min"].sum())
        sel_acc = int(df_trials["geoinit_select_accepted"].sum())
        print(f"    {sel_label}: {sel_same:3d}/{total_cases:3d} ({sel_same/total_cases*100:5.1f}%)   "
              f"different-basin risk: {total_cases - sel_same}/{total_cases}   "
              f"accepted non-raw: {sel_acc}/{total_cases}")
