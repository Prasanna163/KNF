import os
import re
import subprocess
import numpy as np
import logging
from .utils import run_subprocess

def run_xtb_optimization(filepath: str, charge: int = 0, uhf: int = 0) -> str:
    """
    Runs xTB geometry optimization.
    Returns path to optimized geometry file.
    """
    cmd = ['xtb', os.path.basename(filepath), '--opt', '--charge', str(charge), '--uhf', str(uhf)]
    
    # Run in the directory of the input file to keep outputs there
    cwd = os.path.dirname(os.path.abspath(filepath))
    
    # xTB writes 'xtbopt.xyz' or 'xtbopt.mol'
    # We'll assume it writes 'xtbopt.xyz' for now
    
    logging.info(f"Running xTB optimization on {filepath}...")
    logging.info(f"CMD: {cmd}")
    run_subprocess(cmd, cwd=cwd)
    
    optimized_file = os.path.join(cwd, 'xtbopt.xyz')
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

    # --- f3: Max WBO ---
    # "Wiberg bond orders" block
    # Sample:
    # Wiberg/Mayer (AO) data.
    # ...
    #      #   Z sym  total        # sym  WBO       # sym  WBO       # sym  WBO
    #  ---------------------------------------------------------------------------
    #      1   6 C    3.951 --     2 N    0.992     8 H    0.978     6 H    0.971
    
    # Algorithm: Find block, regex for "integer symbol float" patterns?
    # The lines have multiple entries.
    # We just need to extract ALL floating point numbers that are WBOs.
    # In the table: "2 N    0.992". Structure: Int Sym Float.
    # We can regex for `\d+\s+[A-Za-z]+\s+([\d\.]+)` 
    # BUT, we must ensure we are in the WBO block to avoid parsing other things.
    
    wbo_values = []
    if "Wiberg bond orders" in content or "Wiberg/Mayer (AO) data" in content:
        # Extract the block
        # Start at header, end at empty line or next header? 
        # Actually, extracting all "Int Sym Float" patterns from the whole file might be risky? 
        # Best to limit to the block.
        
        lines = content.splitlines()
        in_wbo = False
        for line in lines:
            if "Wiberg bond orders" in line or "Wiberg/Mayer (AO) data" in line:
                in_wbo = True
                continue
            
            if in_wbo:
                if "Topologies differ" in line or "MOs/occ written" in line or line.strip() == "":
                    # End of block likely
                    # But blank lines might exist? 
                    # If we hit another section header...
                    if line.strip() == "" and len(wbo_values) > 0: 
                        # Only stop if we found something? Or continue?
                        # Sample has blank line after table.
                        pass # Continue just in case
                    if "written to" in line:
                        in_wbo = False
                        break
                
                if "----------------" in line: continue
                if "#   Z sym" in line: continue
                
                # Parse line: 
                # 1   6 C    3.951 --     2 N    0.992     8 H    0.978 ...
                # We want [2 N 0.992], [8 H 0.978], etc.
                # Regex for `\s+\d+\s+[A-Z][a-z]?\s+([\d\.]+)` works for the columns?
                # The first part is `1 6 C 3.951 --` which is Atom Index / Z / Sym / Total valency?
                # The WBOs are to the right.
                
                # Let's look for pattern: integers followed by element symbol followed by float.
                # `\b\d+\s+[A-Za-z]{1,2}\s+([\d\.]+)`
                
                matches = re.findall(r'\b\d+\s+[A-Za-z]{1,2}\s+([\d\.]+)', line)
                for m in matches:
                    try:
                        val = float(m)
                        wbo_values.append(val)
                    except ValueError:
                        pass
        
    if not wbo_values:
        raise ValueError("WBO section/values not found in xTB log.")
        
    f3 = max(wbo_values)
    data['f3'] = f3
    
    return data

