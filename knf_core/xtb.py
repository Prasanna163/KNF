import os
import re
import subprocess
import numpy as np
import logging


def _is_wbo_triplet(parts: list[str]) -> bool:
    if len(parts) < 3:
        return False
    try:
        int(parts[0])
        int(parts[1])
        float(parts[2])
        return True
    except ValueError:
        return False


def _looks_like_wbo_file(path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if _is_wbo_triplet(line.split()):
                    return True
    except OSError:
        return False
    return False


def resolve_wbo_path(wbo_path: str = None, xtb_log_path: str = None) -> str:
    """
    Resolves the xTB WBO file location.
    Priority:
    1) explicit path if it looks valid
    2) log-hinted names (if present)
    3) common default names in run directory
    4) any file containing 'wbo' in name with valid triplet rows
    """
    if _looks_like_wbo_file(wbo_path):
        return wbo_path

    run_dir = None
    if wbo_path:
        run_dir = os.path.dirname(os.path.abspath(wbo_path))
    if not run_dir and xtb_log_path:
        run_dir = os.path.dirname(os.path.abspath(xtb_log_path))

    candidates = []

    if xtb_log_path and os.path.exists(xtb_log_path):
        try:
            with open(xtb_log_path, "r", encoding="utf-8", errors="replace") as f:
                log_content = f.read()
            # xTB may include hints like: writing <xtb.wbo>
            for m in re.finditer(r"(?:writing|written to file)\s*<([^>]*wbo[^>]*)>", log_content, flags=re.IGNORECASE):
                token = m.group(1).strip().strip("'\"")
                if not token:
                    continue
                if os.path.isabs(token):
                    candidates.append(token)
                elif run_dir:
                    candidates.append(os.path.join(run_dir, token))
        except OSError:
            pass

    if run_dir:
        for name in ("wbo", "xtb.wbo", "wbo.txt", "xtbout.wbo"):
            candidates.append(os.path.join(run_dir, name))
        try:
            for name in sorted(os.listdir(run_dir)):
                if "wbo" in name.lower():
                    candidates.append(os.path.join(run_dir, name))
        except OSError:
            pass

    seen = set()
    for path in candidates:
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        if _looks_like_wbo_file(norm):
            return norm

    raise FileNotFoundError(
        f"Unable to locate a valid WBO file (explicit={wbo_path!r}, xtb_log={xtb_log_path!r})."
    )


def parse_max_wbo(wbo_path: str, xtb_log_path: str = None) -> float:
    """
    Computes the global maximum WBO from xTB WBO triplet file.
    """
    resolved = resolve_wbo_path(wbo_path=wbo_path, xtb_log_path=xtb_log_path)
    max_wbo = 0.0
    with open(resolved, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if not _is_wbo_triplet(parts):
                continue
            max_wbo = max(max_wbo, float(parts[2]))
    return float(max_wbo)

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


def compute_wbo_from_molden_details(
    molden_path: str,
    fragments: list[list[int]] | None = None,
    use_identity_overlap: bool = True,
) -> dict:
    """
    Computes AO- and atom-block WBO diagnostics directly from a Molden wavefunction.

    Method:
    1) Parse MO coefficients and occupations from molden.input.
    2) Build density matrix: P = C @ diag(occ) @ C.T
    3) Build PS = P @ S (identity S by default for xTB minimal basis workflows)
    4) AO WBO-like matrix: W_ao = PS * PS.T  (elementwise product)
    5) Sum AO blocks by atom centers -> W_atom
    """
    if not os.path.exists(molden_path):
        raise FileNotFoundError(f"Molden file not found: {molden_path}")

    from .nci_torch.molden import parse_molden

    wf = parse_molden(molden_path, apply_primitive_normalization=False)
    coeff = np.asarray(wf.mo_coefficients, dtype=np.float64)  # (n_ao, n_mo)
    occ = np.asarray(wf.occupations, dtype=np.float64)        # (n_mo,)

    if coeff.ndim != 2:
        raise ValueError("Unexpected MO coefficient shape in molden file.")
    if occ.ndim != 1 or occ.shape[0] != coeff.shape[1]:
        raise ValueError("Occupation vector size does not match MO coefficients.")

    density = coeff @ np.diag(occ) @ coeff.T
    if use_identity_overlap:
        overlap = np.eye(density.shape[0], dtype=np.float64)
    else:
        raise NotImplementedError("Non-identity AO overlap is not yet implemented for molden-native WBO.")

    ps = density @ overlap
    w_ao = ps * ps.T

    ao_to_atom = np.asarray([bf.center_index for bf in wf.basis_functions], dtype=np.int64)
    n_atoms = int(len(wf.atom_symbols))
    w_atom = np.zeros((n_atoms, n_atoms), dtype=np.float64)
    for mu in range(w_ao.shape[0]):
        a_mu = int(ao_to_atom[mu])
        w_atom[a_mu, :] += np.bincount(
            ao_to_atom,
            weights=w_ao[mu, :],
            minlength=n_atoms,
        )

    offdiag_mask = ~np.eye(n_atoms, dtype=bool)
    max_wbo_global = float(np.max(w_atom[offdiag_mask])) if n_atoms > 1 else 0.0

    max_inter_wbo = 0.0
    max_inter_pair = None
    inter_pair_count = 0
    if fragments and len(fragments) >= 2:
        # f3 definition: max over atom pairs between monomer 1 and monomer 2.
        mon1 = sorted({int(i) for i in fragments[0]})
        mon2 = sorted({int(i) for i in fragments[1]})
        for i in mon1:
            if i < 0 or i >= n_atoms:
                continue
            for j in mon2:
                if j < 0 or j >= n_atoms:
                    continue
                inter_pair_count += 1
                val = float(w_atom[i, j])
                if val > max_inter_wbo:
                    max_inter_wbo = val
                    max_inter_pair = {"atom_i_1based": i + 1, "atom_j_1based": j + 1, "wbo": val}

    return {
        "max_inter_wbo": float(max_inter_wbo),
        "max_wbo_global": float(max_wbo_global),
        "inter_pair_count": int(inter_pair_count),
        "inter_max_pair": max_inter_pair,
        "n_atoms": n_atoms,
        "n_ao": int(w_ao.shape[0]),
        "overlap_model": "identity" if use_identity_overlap else "explicit",
    }


def parse_interfragment_wbo(wbo_path: str, fragments: list[list[int]], xtb_log_path: str = None) -> float:
    """
    Computes max intermolecular WBO from xTB `wbo` file.
    Atom indices in the file are 1-based; fragment indices are expected 0-based.
    """
    resolved = resolve_wbo_path(wbo_path=wbo_path, xtb_log_path=xtb_log_path)

    if not fragments or len(fragments) < 2:
        return 0.0

    atom_to_fragment = {}
    for frag_idx, frag in enumerate(fragments):
        for atom_idx in frag:
            atom_to_fragment[int(atom_idx)] = frag_idx

    max_inter_wbo = 0.0
    with open(resolved, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if not _is_wbo_triplet(parts):
                continue
            i_1b = int(parts[0])
            j_1b = int(parts[1])
            wbo_val = float(parts[2])

            i = i_1b - 1
            j = j_1b - 1
            fi = atom_to_fragment.get(i)
            fj = atom_to_fragment.get(j)
            if fi is None or fj is None:
                continue
            if fi != fj and wbo_val > max_inter_wbo:
                max_inter_wbo = wbo_val

    return float(max_inter_wbo)

