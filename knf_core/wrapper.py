import os
import shutil
import subprocess
import logging

from . import utils


def _solvent_args(use_water: bool) -> list[str]:
    if use_water:
        return ['--alpb', 'water']
    return ['--cosmo', 'water']


def _xtb_invocation(xtb_cmd: str = "xtb") -> list[str]:
    """Resolve an xTB launcher name into an argv prefix for subprocess.

    Accepts either the stock CPU build (``xtb``) or the GPU-accelerated
    front-end (``xtbx``). ``shutil.which`` locates the launcher on PATH
    (including ``.cmd``/``.bat`` shims like ``xtbx.cmd``, which wraps a WSL
    call); for the stock ``xtb`` we also fall back to KNF-CORE's own tool
    discovery. Windows' ``CreateProcess`` cannot execute a batch file
    directly, so ``.cmd``/``.bat`` launchers are invoked through ``cmd /c``.

    Raises
    ------
    FileNotFoundError
        If the requested launcher cannot be resolved on PATH.
    """
    resolved = shutil.which(xtb_cmd)
    if resolved is None and xtb_cmd == "xtb":
        # Reuse KNF-CORE's conda/registered-path discovery for the stock build.
        resolved = utils.resolve_external_tool_command("xtb")
    if resolved is None:
        raise FileNotFoundError(
            f"xTB launcher '{xtb_cmd}' was not found on PATH. "
            "Install it (or, for 'xtbx', ensure the WSL GPU build is available) "
            "or select a different --xtb-engine."
        )
    if os.name == "nt" and resolved.lower().endswith((".cmd", ".bat")):
        return ["cmd", "/c", resolved]
    return [resolved]


def run_uff_preopt(filepath: str, max_iters: int = 200) -> str:
    """
    Runs a UFF (Universal Force Field) pre-optimisation on the input
    geometry using RDKit.  Overwrites *filepath* in-place with the
    relaxed coordinates so that the downstream xTB optimiser starts
    from a better initial geometry.

    Returns the (unchanged) filepath for chaining convenience.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds

    ext = os.path.splitext(filepath)[1].lower()

    # ---- load molecule ------------------------------------------------
    if ext == '.xyz':
        mol = Chem.MolFromXYZFile(filepath)
        if mol is not None:
            try:
                rdDetermineBonds.DetermineConnectivity(mol)
                rdDetermineBonds.DetermineBondOrders(mol)
            except Exception as e:
                logging.warning("UFF pre-opt: bond perception failed for "
                                ".xyz input (%s). Skipping UFF step.", e)
                return filepath
    elif ext == '.mol' or ext == '.mol2':
        mol = Chem.MolFromMolFile(filepath, removeHs=False)
    elif ext == '.sdf':
        suppl = Chem.SDMolSupplier(filepath, removeHs=False)
        mol = suppl[0] if len(suppl) > 0 else None
    else:
        logging.warning("UFF pre-opt: unsupported extension '%s'. "
                        "Skipping UFF step.", ext)
        return filepath

    if mol is None:
        logging.warning("UFF pre-opt: RDKit failed to load %s. "
                        "Skipping UFF step.", filepath)
        return filepath

    # ---- add hydrogens if missing & embed if no 3-D coords -----------
    try:
        mol = Chem.AddHs(mol, addCoords=True)
    except Exception:
        pass  # keep original if AddHs fails

    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

    # ---- UFF relaxation -----------------------------------------------
    try:
        ff_result = AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
        if ff_result == 0:
            logging.info("UFF pre-optimisation converged in ≤%d iterations.",
                         max_iters)
        elif ff_result == 1:
            logging.info("UFF pre-optimisation hit iteration cap (%d). "
                         "Using best geometry so far.", max_iters)
        else:
            logging.warning("UFF setup/optimisation returned code %s. "
                            "Skipping UFF step.", ff_result)
            return filepath
    except Exception as e:
        logging.warning("UFF pre-opt failed (%s). Skipping UFF step.", e)
        return filepath

    # ---- write relaxed geometry back as XYZ ---------------------------
    try:
        xyz_block = Chem.MolToXYZBlock(mol)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(xyz_block)
        logging.info("UFF-relaxed geometry written to %s", filepath)
    except Exception as e:
        logging.warning("UFF pre-opt: failed to write relaxed geometry "
                        "(%s). Original file preserved.", e)

    return filepath


def run_geoinit_preopt(filepath: str, sigma: float = 0.05, maxiter: int = 500) -> str:
    """Basin-safe geometry warm-start (drop-in replacement for run_uff_preopt).

    Uses GeoInit's physics functional to relax the input geometry, then applies
    GeoInit's own guards: if the relaxation damaged bonds / drifted a fragment
    it falls back to the rigid placement, and if that is still unsafe it keeps
    the raw geometry untouched. This preserves the xTB basin the downstream
    optimiser lands in, which is the invariant a descriptor pipeline needs.

    Overwrites *filepath* in-place (like run_uff_preopt) and returns it.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext != ".xyz":
        # GeoInit consumes .xyz only; the pipeline always hands us input.xyz,
        # but stay defensive and skip cleanly for anything else.
        logging.warning(
            "GeoInit pre-opt: unsupported extension '%s'. Skipping GeoInit step.", ext
        )
        return filepath

    try:
        from geoinit.core.io_xyz import read_xyz, write_xyz
        from geoinit.core.topology import Topology
        from geoinit.optimize.relax import relax
        from geoinit.optimize.guards import accept_geoinit, check_damage
    except ImportError as e:
        raise RuntimeError(
            "GeoInit pre-optimisation was requested (--preopt geoinit) but the "
            "bundled 'geoinit' package is not importable in this environment "
            f"({e}). Reinstall NCIForge or use --preopt uff."
        ) from e

    try:
        symbols, coords = read_xyz(filepath)
    except Exception as e:
        logging.warning(
            "GeoInit pre-opt: failed to read %s (%s). Skipping GeoInit step.",
            filepath,
            e,
        )
        return filepath

    try:
        topo = Topology(symbols, coords, scale=1.25, sigma=sigma)
        result = relax(
            symbols, coords, sigma=sigma, maxiter=maxiter, topology=topo, mode="fast"
        )

        # GeoInit guards: prefer the relaxed coords, fall back to the rigid
        # placement if bonds/fragments were damaged, then to raw if still unsafe.
        damaged = check_damage(symbols, coords, result.final_coords, topo)
        if damaged and result.rigid_coords is not None:
            candidate = result.rigid_coords
        else:
            candidate = result.final_coords

        accepted, reason = accept_geoinit(symbols, coords, candidate, topo)
        if accepted:
            final = candidate
            logging.info("GeoInit pre-opt accepted (%s).", result.message)
        else:
            final = coords  # raw fallback preserves the xTB basin
            logging.info(
                "GeoInit pre-opt rejected (%s); preserving raw geometry.", reason
            )

        write_xyz(filepath, symbols, final, comment="GeoInit warm-start")
    except Exception as e:
        logging.warning(
            "GeoInit pre-opt failed (%s). Original geometry preserved.", e
        )

    return filepath


def run_preopt(filepath: str, engine: str = "geoinit") -> str:
    """Dispatch pre-optimisation to the selected engine (default GeoInit)."""
    engine = (engine or "geoinit").strip().lower()
    if engine == "geoinit":
        return run_geoinit_preopt(filepath)
    if engine in ("", "uff"):
        return run_uff_preopt(filepath)
    raise ValueError(f"Unsupported preopt engine '{engine}'. Use 'uff' or 'geoinit'.")


def run_subprocess(cmd: list, cwd: str = None) -> subprocess.CompletedProcess:
    """Runs a subprocess command."""
    try:
        # On Windows, shell=True can help with finding executables if they are not direct binaries
        # But for list args, it's safer to rely on PATH.
        # If exit code 128 persists, we might check if 'xtb' is actually found.
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            errors='replace'
        )
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {' '.join(cmd)}")
        logging.error(f"STDOUT: {e.stdout}")
        logging.error(f"STDERR: {e.stderr}")
        raise e

def run_xtb_opt(
    filepath: str,
    charge: int = 0,
    uhf: int = 0,
    use_water: bool = False,
    xtb_cmd: str = "xtb",
) -> str:
    cwd = os.path.dirname(os.path.abspath(filepath))
    filename = os.path.basename(filepath)

    cmd = _xtb_invocation(xtb_cmd) + [
        filename,
        '--opt',
        '--cycles',
        '50',
    ]
    cmd.extend(_solvent_args(use_water))
    cmd.extend(['--charge', str(charge), '--uhf', str(uhf)])

    logging.info(f"Wrapper Executing xTB Opt: {cmd} in {cwd}")
    xtb_opt_log = os.path.join(cwd, 'xtb_opt.log')
    with open(xtb_opt_log, 'w', encoding='utf-8', errors='replace') as log:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            errors='replace',
            check=False,
        )

    output = os.path.join(cwd, 'xtbopt.xyz')
    if result.returncode != 0:
        if os.path.exists(output):
            logging.warning(
                "xTB optimization exited with code %s, but xtbopt.xyz exists. "
                "Proceeding to NCI pipeline using the latest available geometry.",
                result.returncode,
            )
            return output
        raise subprocess.CalledProcessError(result.returncode, cmd)

    if not os.path.exists(output):
        raise FileNotFoundError(f"xTB opt failed: {output}")
    return output

def run_xtb_sp(
    filepath: str,
    charge: int = 0,
    uhf: int = 0,
    use_water: bool = False,
    xtb_cmd: str = "xtb",
):
    cwd = os.path.dirname(os.path.abspath(filepath))
    filename = os.path.basename(filepath)

    cmd = _xtb_invocation(xtb_cmd) + [filename, '--esp', '--molden', '--hess', '--wbo']
    cmd.extend(_solvent_args(use_water))
    cmd.extend(['--charge', str(charge), '--uhf', str(uhf)])
    
    logging.info(f"Wrapper Executing xTB SP: {cmd} in {cwd}")
    
    with open(os.path.join(cwd, 'xtb.log'), 'w') as log:
        # Using subprocess.run directly to redirect stdout/stderr to file
        try:
             subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as e:
             logging.error(f"Command failed with {e.returncode}. Check xtb.log in {cwd}")
             raise e
