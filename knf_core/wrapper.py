import os
import subprocess
import logging


def _solvent_args(use_water: bool) -> list[str]:
    if use_water:
        return ['--alpb', 'water']
    return ['--cosmo', 'water']


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

def run_xtb_opt(filepath: str, charge: int = 0, uhf: int = 0, use_water: bool = False) -> str:
    cwd = os.path.dirname(os.path.abspath(filepath))
    filename = os.path.basename(filepath)

    cmd = [
        'xtb',
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

def run_xtb_sp(filepath: str, charge: int = 0, uhf: int = 0, use_water: bool = False):
    cwd = os.path.dirname(os.path.abspath(filepath))
    filename = os.path.basename(filepath)

    cmd = ['xtb', filename, '--esp', '--molden', '--hess', '--wbo']
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
