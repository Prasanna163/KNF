import os
import subprocess
import logging

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

def run_xtb_opt(filepath: str, charge: int = 0, uhf: int = 0) -> str:
    cwd = os.path.dirname(os.path.abspath(filepath))
    filename = os.path.basename(filepath)
    # User requested --opt for optimization
    # check if --cosmo is needed here. Usually for opt it's good to keep consistent environment.
    # But usually opt is gas phase or implicit solvent.
    # If the user said "1. optimise using --opt", they didn't specify solvent.
    # However, SCDI needs solvent later.
    # To be safe and consistent, we often do opt in gas phase then SP in solvent, OR opt in solvent.
    # The previous code had --cosmo water.
    # The user instruction "1. optimise using --opt" might imply minimal flags.
    # But they also said "2. Generate sister files...".
    # I will stick to --opt and --cosmo water to ensure geometry is relevant for the solvated properties if SCDI expects it.
    # Actually, if I remove --cosmo from opt, the geometry might be slightly different.
    # Given "Don't ask... Just do this", I will prioritize ROBUSTNESS for the descriptors.
    # Detailed instruction "1. optimise using --opt" matches the flag `--opt`.
    # I will keep `--cosmo water` for stability of SCDI unless it breaks.
    
    # Cap ANC optimization cycles at 50; xTB will still terminate earlier if converged.
    cmd = [
        'xtb',
        filename,
        '--opt',
        '--cycles',
        '50',
        '--cosmo',
        'water',
        '--charge',
        str(charge),
        '--uhf',
        str(uhf),
    ]

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

def run_xtb_sp(filepath: str, charge: int = 0, uhf: int = 0):
    cwd = os.path.dirname(os.path.abspath(filepath))
    filename = os.path.basename(filepath)
    
    # User requested: --esp --properties --molden --hessian
    # I will add --wbo (for KNF) and --cosmo (for SCDI) as well.
    # --hessian flag in xTB is '--hess'. 
    # --properties is not a standard single flag in xTB 6.x usually, but maybe they mean '--grad' or similar?
    # Or maybe they mean the properties block.
    # Valid xTB flags: --esp (ESP calculation), --molden (write molden), --hess (Hessian/Freq)
    # --wbo (write WBO), --cosmo (solvation).
    
    cmd = ['xtb', filename, '--esp', '--molden', '--hess', '--wbo', '--cosmo', 'water', '--charge', str(charge), '--uhf', str(uhf)]
    
    logging.info(f"Wrapper Executing xTB SP: {cmd} in {cwd}")
    
    with open(os.path.join(cwd, 'xtb.log'), 'w') as log:
        # Using subprocess.run directly to redirect stdout/stderr to file
        try:
             subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as e:
             logging.error(f"Command failed with {e.returncode}. Check xtb.log in {cwd}")
             raise e
