import os
import queue
import re
import shutil
import subprocess
import threading
import time
import logging
from typing import Callable

from . import utils


XtbProgressCallback = Callable[[dict], None]

_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_CYCLE_WORD_RE = re.compile(r"\b(?:cycle|iter(?:ation)?)\s*[:=]?\s*(\d+)\b", re.IGNORECASE)
_CYCLE_ROW_RE = re.compile(r"^\s*(\d{1,5})(?=\s+[-+]?(?:\d|\.))")
_ENERGY_RE = re.compile(rf"\b(?:energy|total\s+energy|E)\s*[:=]?\s*({_FLOAT_RE})", re.IGNORECASE)
_GRADIENT_RE = re.compile(rf"\b(?:grad(?:ient)?|gnorm)\s*[:=]?\s*({_FLOAT_RE})", re.IGNORECASE)
_FLOAT_TOKEN_RE = re.compile(_FLOAT_RE)


def _solvent_args(use_water: bool) -> list[str]:
    if use_water:
        return ['--alpb', 'water']
    return ['--cosmo', 'water']


def _xtb_invocation(xtb_cmd: str = "xtb") -> list[str]:
    """Resolve an xTB launcher name into an argv prefix for subprocess.

    Accepts either the stock CPU build (``xtb``) or the unified accelerated
    front-end (``xtbx``). ``shutil.which`` locates the launcher on PATH
    (including ``.cmd``/``.bat`` shims like ``xtbx.cmd``); for the stock
    ``xtb`` we also fall back to KNF-CORE's own tool discovery. Windows'
    ``CreateProcess`` cannot execute a batch file
    directly, so ``.cmd``/``.bat`` launchers are invoked through ``cmd /c``.

    Raises
    ------
    FileNotFoundError
        If the requested launcher cannot be resolved on PATH.
    """
    if (xtb_cmd or "").strip().lower() == "xtbx":
        try:
            from nciforge_xtbx.cli import subprocess_invocation

            return subprocess_invocation()
        except Exception as e:
            logging.warning(
                "Bundled xtbx launcher unavailable (%s); falling back to PATH.", e
            )

    resolved = shutil.which(xtb_cmd)
    if resolved is None and xtb_cmd == "xtb":
        # Reuse KNF-CORE's conda/registered-path discovery for the stock build.
        resolved = utils.resolve_external_tool_command("xtb")
    if resolved is None:
        raise FileNotFoundError(
            f"xTB launcher '{xtb_cmd}' was not found on PATH. "
            "Install it (or, for 'xtbx', ensure the native launcher is available) "
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


def _parse_xtb_progress_line(line: str) -> dict:
    payload: dict = {}
    stripped = line.strip()
    if not stripped:
        return payload

    cycle_row_match = _CYCLE_ROW_RE.match(stripped)
    cycle_match = _CYCLE_WORD_RE.search(stripped) or cycle_row_match
    if cycle_match:
        try:
            payload["cycle"] = int(cycle_match.group(1))
        except ValueError:
            pass

    energy_match = _ENERGY_RE.search(stripped)
    if energy_match:
        payload["energy"] = energy_match.group(1)
    elif cycle_row_match:
        row_tail = stripped[cycle_row_match.end() :]
        row_floats = _FLOAT_TOKEN_RE.findall(row_tail)
        if row_floats:
            payload["energy"] = row_floats[0]

    gradient_match = _GRADIENT_RE.search(stripped)
    if gradient_match:
        payload["gradient"] = gradient_match.group(1)

    return payload


def _format_xtb_progress_message(payload: dict) -> str:
    parts = [str(payload.get("label") or "xTB")]
    if payload.get("cycle") is not None:
        parts.append(f"cycle {payload['cycle']}")
    if payload.get("energy"):
        parts.append(f"Energy {payload['energy']} Eh")
    if payload.get("gradient"):
        parts.append(f"Grad {payload['gradient']}")
    if len(parts) == 1 and payload.get("last_line"):
        last_line = str(payload["last_line"]).strip()
        if last_line:
            parts.append(last_line[:96])
    return " | ".join(parts)


def _run_xtb_streaming(
    cmd: list,
    *,
    cwd: str,
    log_path: str,
    progress_callback: XtbProgressCallback | None,
    stage: str,
    launcher: str,
    use_gpu: bool,
) -> int:
    started_at = time.perf_counter()
    label = f"xTB {stage.upper()} [{launcher}{' gpu' if use_gpu else ''}]"
    line_queue: queue.Queue[str | None] = queue.Queue()

    def emit(event: str, **payload) -> None:
        if progress_callback is None:
            return
        elapsed = time.perf_counter() - started_at
        body = {
            "event": event,
            "stage": stage,
            "label": label,
            "launcher": launcher,
            "use_gpu": bool(use_gpu),
            "elapsed_seconds": elapsed,
            "log_path": log_path,
        }
        body.update(payload)
        body["message"] = _format_xtb_progress_message(body)
        progress_callback(body)

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )

    def reader() -> None:
        assert process.stdout is not None
        try:
            for output_line in process.stdout:
                line_queue.put(output_line)
        finally:
            line_queue.put(None)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    emit("started")
    last_payload: dict = {}
    last_heartbeat = time.perf_counter()
    stream_closed = False
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        while True:
            try:
                line = line_queue.get(timeout=0.2)
            except queue.Empty:
                if process.poll() is not None and stream_closed:
                    break
                now = time.perf_counter()
                if progress_callback is not None and now - last_heartbeat >= 2.0:
                    emit("heartbeat", **last_payload)
                    last_heartbeat = now
                continue

            if line is None:
                stream_closed = True
                if process.poll() is not None:
                    break
                continue

            log.write(line)
            log.flush()
            parsed = _parse_xtb_progress_line(line)
            parsed["last_line"] = line.strip()
            last_payload.update(parsed)
            emit("output", **last_payload)
            last_heartbeat = time.perf_counter()

    reader_thread.join(timeout=1.0)
    return_code = process.wait()
    emit("finished", return_code=return_code, **last_payload)
    return return_code

def run_xtb_opt(
    filepath: str,
    charge: int = 0,
    uhf: int = 0,
    use_water: bool = False,
    xtb_cmd: str = "xtb",
    force_gpu: bool = False,
    progress_callback: XtbProgressCallback | None = None,
) -> str:
    cwd = os.path.dirname(os.path.abspath(filepath))
    filename = os.path.basename(filepath)

    cmd = _xtb_invocation(xtb_cmd) + [
        filename,
        '--opt',
        '--cycles',
        '50',
    ]
    if force_gpu and (xtb_cmd or "").strip().lower() == "xtbx":
        cmd.append('--gpu')
    cmd.extend(_solvent_args(use_water))
    cmd.extend(['--charge', str(charge), '--uhf', str(uhf)])

    logging.info(f"Wrapper Executing xTB Opt: {cmd} in {cwd}")
    xtb_opt_log = os.path.join(cwd, 'xtb_opt.log')
    if progress_callback is None:
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
        return_code = result.returncode
    else:
        return_code = _run_xtb_streaming(
            cmd,
            cwd=cwd,
            log_path=xtb_opt_log,
            progress_callback=progress_callback,
            stage="opt",
            launcher=(xtb_cmd or "xtb").strip().lower(),
            use_gpu=bool(force_gpu and (xtb_cmd or "").strip().lower() == "xtbx"),
        )

    output = os.path.join(cwd, 'xtbopt.xyz')
    if return_code != 0:
        if os.path.exists(output):
            logging.warning(
                "xTB optimization exited with code %s, but xtbopt.xyz exists. "
                "Proceeding to NCI pipeline using the latest available geometry.",
                return_code,
            )
            return output
        raise subprocess.CalledProcessError(return_code, cmd)

    if not os.path.exists(output):
        raise FileNotFoundError(f"xTB opt failed: {output}")
    return output

def run_xtb_sp(
    filepath: str,
    charge: int = 0,
    uhf: int = 0,
    use_water: bool = False,
    xtb_cmd: str = "xtb",
    force_gpu: bool = False,
    include_hess: bool = True,
    include_esp: bool = False,
    progress_callback: XtbProgressCallback | None = None,
):
    cwd = os.path.dirname(os.path.abspath(filepath))
    filename = os.path.basename(filepath)

    cmd = _xtb_invocation(xtb_cmd) + [filename]
    if include_esp:
        cmd.append('--esp')
    cmd.append('--molden')
    if include_hess:
        cmd.append('--hess')
    cmd.append('--wbo')
    if force_gpu and (xtb_cmd or "").strip().lower() == "xtbx":
        cmd.append('--gpu')
    cmd.extend(_solvent_args(use_water))
    cmd.extend(['--charge', str(charge), '--uhf', str(uhf)])
    
    logging.info(f"Wrapper Executing xTB SP: {cmd} in {cwd}")

    xtb_log = os.path.join(cwd, 'xtb.log')
    if progress_callback is None:
        with open(xtb_log, 'w', encoding='utf-8', errors='replace') as log:
            try:
                subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Command failed with {e.returncode}. Check xtb.log in {cwd}")
                raise e
    else:
        return_code = _run_xtb_streaming(
            cmd,
            cwd=cwd,
            log_path=xtb_log,
            progress_callback=progress_callback,
            stage="sp",
            launcher=(xtb_cmd or "xtb").strip().lower(),
            use_gpu=bool(force_gpu and (xtb_cmd or "").strip().lower() == "xtbx"),
        )
        if return_code != 0:
            logging.error(f"Command failed with {return_code}. Check xtb.log in {cwd}")
            raise subprocess.CalledProcessError(return_code, cmd)
