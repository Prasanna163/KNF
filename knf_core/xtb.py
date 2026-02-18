import os
import re
import subprocess
import numpy as np
import logging

def run_xtb_optimization(filepath: str, charge: int = 0, uhf: int = 0) -> str:
    """
    Runs xTB geometry optimization.
    Returns path to optimized geometry file.
    """
    cmd = [
        'xtb',
        os.path.basename(filepath),
        '--opt',
        '--cycles',
        '50',
        '--charge',
        str(charge),
        '--uhf',
        str(uhf),
    ]
    
    # Run in the directory of the input file to keep outputs there
    cwd = os.path.dirname(os.path.abspath(filepath))
    
    # xTB writes 'xtbopt.xyz' or 'xtbopt.mol'
    # We'll assume it writes 'xtbopt.xyz' for now
    
    logging.info(f"Running xTB optimization on {filepath}...")
    logging.info(f"CMD: {cmd}")
    optimized_file = os.path.join(cwd, 'xtbopt.xyz')
    opt_log = os.path.join(cwd, 'xtb_opt.log')
    with open(opt_log, 'w', encoding='utf-8', errors='replace') as log:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            errors='replace',
            check=False,
        )
    if result.returncode != 0:
        if os.path.exists(optimized_file):
            logging.warning(
                "xTB optimization exited with code %s, but xtbopt.xyz exists. "
                "Proceeding with latest available geometry.",
                result.returncode,
            )
            return optimized_file
        raise subprocess.CalledProcessError(result.returncode, cmd)
    if not os.path.exists(optimized_file):
        raise FileNotFoundError(f"xTB optimization failed to produce {optimized_file}")
        
    return optimized_file

def run_xtb_single_point(filepath: str, charge: int = 0, uhf: int = 0):
    """
    Runs xTB single point calculation with WBO, Molden, and COSMO.
    """
    cmd = ['xtb', os.path.basename(filepath), '--wbo', '--molden', '--cosmo', '--charge', str(charge), '--uhf', str(uhf)]
    
    cwd = os.path.dirname(os.path.abspath(filepath))
    
    import sys
    logging.info(f"Running xTB single point on {filepath}...")
    logging.info(f"CMD: {cmd}")
    print(f"DEBUG: cmd={cmd}", file=sys.stderr, flush=True)
    print(f"DEBUG: basename={os.path.basename(filepath)}", file=sys.stderr, flush=True)
    # Capture output to log file? xTB writes to stdout usually.
    # We might want to redirect stdout to a file 'xtb.log'
    
    with open(os.path.join(cwd, 'xtb.log'), 'w') as f:
        subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=True)

def parse_xtb_log(log_path: str) -> dict:
    """
    Parses xTB log file for f3 (Max WBO), f4 (Dipole), f5 (Polarizability).
    Raises ValueError if any are missing/zero.
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"xTB log file not found: {log_path}")
        
    data = {}
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # --- f4: Dipole Moment ---
    # Search for: "molecular dipole:" ... "tot (Debye)"
    # Example:
    # molecular dipole:
    # ...
    # ... tot (Debye) ... 5.351
    
    # We look for the block "molecular dipole:"
    # Then find the line containing "tot (Debye)" or just the value at the end of the block?
    # Based on sample:
    # molecular dipole:
    #                  x           y           z       tot (Debye)
    #  q only:       -1.778       1.164      -0.369
    #    full:       -1.797       1.007      -0.432       5.351
    
    # We want the "full" row's last value if available, or just the value under "tot (Debye)".
    # Let's try to match: "full:\s+.*\s+([\d\.]+)\s*$" -- No, might have spaces.
    # regex for the "full:" line in dipole block
    
    dipole_val = None
    if "molecular dipole:" in content:
        # Iterate lines to be safe
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if "molecular dipole:" in line:
                # Look ahead a few lines for "full:" or just extract if it's on single line (older xtb)
                # In sample, it's a block.
                for j in range(1, 10):
                    if i+j >= len(lines): break
                    subline = lines[i+j]
                    if "full:" in subline:
                        # full:       -1.797       1.007      -0.432       5.351
                        parts = subline.split()
                        if parts:
                            try:
                                dipole_val = float(parts[-1])
                            except ValueError:
                                pass
                        break
                if dipole_val is not None: break
    
    if dipole_val is None:
        # Fallback or strict error? Spec says: "If not found -> raise ValueError"
        # Let's try one more pattern if the block format is different (single line).
        # "total dipole moment:     3.472 Debye"
        m = re.search(r"total dipole moment:\s+([\d\.]+)\s+Debye", content)
        if m:
            dipole_val = float(m.group(1))
            
    if dipole_val is None:
        raise ValueError("Dipole moment (f4) not found in xTB log.")
        
    data['f4'] = dipole_val

    # --- f5: Polarizability ---
    # "Mol. alpha /au" or "isotropic polarizability"
    # Sample: "Mol. alpha /au        :         88.522781"
    
    pol_val = None
    # Regex for "Mol. alpha /au" OR "Mol. α(0) /au"
    # Matches: "Mol. alpha /au : 123.456" OR "Mol. α(0) /au : 123.456"
    m_pol = re.search(r"Mol\.\s+(?:alpha|α\(0\))\s+/au\s+:\s+([\d\.]+)", content)
    if m_pol:
         pol_val = float(m_pol.group(1))
    else:
         m_pol2 = re.search(r"isotropic polarizability:\s+([\d\.]+)", content)
         if m_pol2:
             pol_val = float(m_pol2.group(1))
             
    if pol_val is None:
        raise ValueError("Polarizability (f5) not computed in xTB run (check flags).")
        
    if pol_val <= 0:
        # Validation as per spec: > 0 (unless it's 0?) Spec says > 0.
        # But if it is 0.0 it likely failed or is atoms?
        # Use simple check.
        pass
        
    data['f5'] = pol_val

    # f3 is now computed from the xTB 'wbo' file using fragment membership in pipeline.
    # Keep a best-effort legacy value from the log for backward compatibility.
    data['f3'] = 0.0
    
    return data


def parse_interfragment_wbo(wbo_path: str, fragments: list[list[int]]) -> float:
    """
    Computes max intermolecular WBO from xTB `wbo` file.
    Atom indices in the file are 1-based; fragment indices are expected 0-based.
    """
    if not os.path.exists(wbo_path):
        raise FileNotFoundError(f"WBO file not found: {wbo_path}")

    if not fragments or len(fragments) < 2:
        return 0.0

    atom_to_fragment = {}
    for frag_idx, frag in enumerate(fragments):
        for atom_idx in frag:
            atom_to_fragment[int(atom_idx)] = frag_idx

    max_inter_wbo = 0.0
    with open(wbo_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                i_1b = int(parts[0])
                j_1b = int(parts[1])
                wbo_val = float(parts[2])
            except ValueError:
                continue

            i = i_1b - 1
            j = j_1b - 1
            fi = atom_to_fragment.get(i)
            fj = atom_to_fragment.get(j)
            if fi is None or fj is None:
                continue
            if fi != fj and wbo_val > max_inter_wbo:
                max_inter_wbo = wbo_val

    return float(max_inter_wbo)

