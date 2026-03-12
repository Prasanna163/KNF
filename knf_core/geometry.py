import logging
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from scipy.spatial.distance import euclidean

# Periodic table masses (simplified for common elements, can be expanded)
# RDKit has this built-in, so we'll use RDKit's GetMass()

def load_molecule(filepath: str) -> Chem.Mol:
    """Loads a molecule from a file."""
    ext = filepath.split('.')[-1].lower()
    if ext == 'mol':
        mol = Chem.MolFromMolFile(filepath, removeHs=False)
    elif ext == 'xyz':
        mol = Chem.MolFromXYZFile(filepath)
        # Attempt to perceive bonds for XYZ to allow fragment detection
        if mol:
            try:
                from rdkit.Chem import rdDetermineBonds
                rdDetermineBonds.DetermineConnectivity(mol)
                rdDetermineBonds.DetermineBondOrders(mol)
            except ImportError:
                logging.warning("rdDetermineBonds not available. Fragment detection may fail for XY input.")
            except Exception as e:
                logging.warning(f"Bond perception failed: {e}")
    else:
        # Try generic loader from RDKit if extension is not explicit
        # But for 'sdf', MolFromMolFile usually works? No, MolFromMolFile is for .mol
        # SDMolSupplier is for .sdf
        if ext == 'sdf':
             suppl = Chem.SDMolSupplier(filepath, removeHs=False)
             mol = suppl[0] if len(suppl) > 0 else None
        else:
             raise ValueError(f"Unsupported file format for geometry analysis: {ext}")
    
    if mol is None:
        raise ValueError(f"Failed to load molecule from {filepath}")
    return mol

def detect_fragments(mol: Chem.Mol) -> list[list[int]]:
    """
    Detects independent fragments in the molecule.
    Returns a list of lists, where each inner list contains atom indices of a fragment.
    """
    frags = Chem.GetMolFrags(mol, asMols=False)
    # frags is a tuple of tuples of indices
    return [list(frag) for frag in frags]

def compute_center_of_mass(mol: Chem.Mol, atom_indices: list[int]) -> np.ndarray:
    """Computes the Center of Mass (COM) for a given set of atoms."""
    conf = mol.GetConformer()
    masses = []
    coords = []
    
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        mass = atom.GetMass()
        pos = conf.GetAtomPosition(idx)
        masses.append(mass)
        coords.append(np.array([pos.x, pos.y, pos.z]))
        
    masses = np.array(masses)
    coords = np.array(coords)
    
    total_mass = np.sum(masses)
    if total_mass == 0:
        return np.zeros(3)
        
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    return com

def compute_fragment_distance(mol: Chem.Mol, frag1_indices: list[int], frag2_indices: list[int]) -> float:
    """Computes the Euclidean distance between the COMs of two fragments."""
    com1 = compute_center_of_mass(mol, frag1_indices)
    com2 = compute_center_of_mass(mol, frag2_indices)
    return float(euclidean(com1, com2))

def detect_hb_angle(mol: Chem.Mol, frag1_indices: list[int], frag2_indices: list[int]) -> float:
    """
    Detects Hydrogen Bond (D-H...A) between two fragments and returns the angle.
    Returns NaN if no HB is detected.
    
    Criteria (simplified):
    - H attached to O, N, F in one fragment.
    - Acceptor (O, N, F) in the other fragment.
    - Distance H...A < 3.5 Angstrom (loose cutoff for detection).
    - If multiple are found, returns the one with the shortest H...A distance.
    """
    conf = mol.GetConformer()
    
    # Identify Potential Donors (D-H) and Acceptors (A)
    # Donors: H attached to O, N, F
    # Acceptors: O, N, F
    
    donors = [] # (D_idx, H_idx)
    acceptors = [] # A_idx
    
    electronegative = [7, 8, 9] # N, O, F atomic numbers
    
    # Classify atoms in frag1 and frag2
    # We need to check H-bonds in both directions: Frag1->Frag2 and Frag2->Frag1
    
    all_indices = frag1_indices + frag2_indices
    frag1_set = set(frag1_indices)
    frag2_set = set(frag2_indices)
    
    possible_hbs = [] # (dist, angle)
    
    for idx in all_indices:
        atom = mol.GetAtomWithIdx(idx)
        an = atom.GetAtomicNum()
        
        # Check if Acceptor
        if an in electronegative:
            acceptors.append(idx)
            
        # Check if Donor (H attached to electronegative)
        if an == 1: # Hydrogen
            neighbors = atom.GetNeighbors()
            if len(neighbors) == 1:
                neighbor = neighbors[0]
                if neighbor.GetAtomicNum() in electronegative:
                    donors.append((neighbor.GetIdx(), idx))

    # Check interactions
    for d_idx, h_idx in donors:
        h_pos = np.array(conf.GetAtomPosition(h_idx))
        
        # Determine which fragment H belongs to
        h_in_frag1 = h_idx in frag1_set
        target_acceptors = [a for a in acceptors if (a in frag2_set if h_in_frag1 else a in frag1_set)]
        
        for a_idx in target_acceptors:
            a_pos = np.array(conf.GetAtomPosition(a_idx))
            dist = np.linalg.norm(h_pos - a_pos)
            
            if dist < 3.5: # Loose distance cutoff
                # Compute Angle D-H...A
                # vector HD
                # vector HA
                # Actually, standard definition is angle DHA. 
                # Ideal is 180. Let's compute angle at H? Or angle at D?
                # Usually it's the angle D-H...A. 180 is linear.
                
                # RDKit GetAngleRad returns angle between three atoms
                angle_deg = rdMolTransforms.GetAngleDeg(conf, d_idx, h_idx, a_idx)
                possible_hbs.append((dist, angle_deg))
                
    if not possible_hbs:
        return float("nan")
        
    # Sort by distance, take the shortest one
    possible_hbs.sort(key=lambda x: x[0])
    best_hb = possible_hbs[0]
    
    return best_hb[1]


def enumerate_hbond_triplets(
    mol: Chem.Mol,
    fragments: list[list[int]],
    ha_cutoff: float = 3.5,
) -> list[dict]:
    """
    Enumerates all cross-fragment donor-H-acceptor triplets.

    Triplet definition:
    - donor hydrogen: H attached to N/O/F
    - acceptor: N/O/F
    - D-H...A angle computed at H
    - H...A distance <= ha_cutoff
    """
    if not fragments or len(fragments) < 2:
        return []

    conf = mol.GetConformer()
    electronegative = {7, 8, 9}  # N, O, F

    atom_to_fragment = {}
    for frag_idx, frag in enumerate(fragments):
        for atom_idx in frag:
            atom_to_fragment[int(atom_idx)] = int(frag_idx)

    all_indices = sorted(atom_to_fragment.keys())
    acceptors_by_fragment = {idx: [] for idx in range(len(fragments))}
    donors = []  # (donor_idx, hydrogen_idx, donor_fragment_idx)

    for idx in all_indices:
        atom = mol.GetAtomWithIdx(idx)
        an = atom.GetAtomicNum()
        frag_idx = atom_to_fragment.get(idx)
        if frag_idx is None:
            continue

        if an in electronegative:
            acceptors_by_fragment[frag_idx].append(idx)

        if an != 1:
            continue
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            continue
        donor = neighbors[0]
        if donor.GetAtomicNum() not in electronegative:
            continue
        donors.append((donor.GetIdx(), idx, frag_idx))

    triplets = []
    for d_idx, h_idx, donor_frag in donors:
        h_pos = np.array(conf.GetAtomPosition(h_idx))
        for acceptor_frag, acceptor_indices in acceptors_by_fragment.items():
            if acceptor_frag == donor_frag:
                continue
            for a_idx in acceptor_indices:
                a_pos = np.array(conf.GetAtomPosition(a_idx))
                ha_distance = float(np.linalg.norm(h_pos - a_pos))
                if ha_distance > float(ha_cutoff):
                    continue

                angle_deg = float(rdMolTransforms.GetAngleDeg(conf, d_idx, h_idx, a_idx))
                triplets.append(
                    {
                        "d_idx": int(d_idx),
                        "h_idx": int(h_idx),
                        "a_idx": int(a_idx),
                        "donor_fragment": int(donor_frag),
                        "acceptor_fragment": int(acceptor_frag),
                        "ha_distance": ha_distance,
                        "angle_deg": angle_deg,
                    }
                )
    return triplets


def _pair_key(i: int, j: int) -> tuple[int, int]:
    ii = int(i)
    jj = int(j)
    return (ii, jj) if ii <= jj else (jj, ii)


def compute_weighted_hbond_angle(
    mol: Chem.Mol,
    fragments: list[list[int]],
    wbo_by_pair: dict | None = None,
    nci_strength_by_triplet=None,
    ha_cutoff: float = 3.5,
) -> dict:
    """
    Computes f2 using weighted donor-H-acceptor triplets:

        f2 = sum_j(w_j * theta_j) / sum_j(w_j)

    Weight model:
    - inverse H...A distance
    - optional interfragment WBO for donor/acceptor atom pair
    - optional NCI local strength callback

    Sentinel policy:
    - If no meaningful triplets exist, returns f2=NaN and f2_defined=0.
    """
    triplets = enumerate_hbond_triplets(mol=mol, fragments=fragments, ha_cutoff=ha_cutoff)
    if not triplets:
        return {
            "f2": float("nan"),
            "f2_defined": 0,
            "triplet_count": 0,
            "weight_sum": 0.0,
            "undefined_reason": "no_candidate_triplets",
            "weight_model": "inv_ha_distance*(1+wbo_da)*(1+nci_local)",
            "top_triplets": [],
        }

    weighted_angle_sum = 0.0
    weight_sum = 0.0
    scored_triplets = []
    for triplet in triplets:
        ha_distance = max(float(triplet["ha_distance"]), 1e-6)
        distance_weight = 1.0 / ha_distance

        wbo_weight = 0.0
        if isinstance(wbo_by_pair, dict):
            key = _pair_key(triplet["d_idx"], triplet["a_idx"])
            try:
                val = float(wbo_by_pair.get(key, 0.0))
                if math.isfinite(val) and val > 0.0:
                    wbo_weight = val
            except (TypeError, ValueError):
                wbo_weight = 0.0

        nci_weight = 0.0
        if callable(nci_strength_by_triplet):
            try:
                val = float(nci_strength_by_triplet(triplet))
                if math.isfinite(val) and val > 0.0:
                    nci_weight = val
            except Exception:
                nci_weight = 0.0

        # Keep a non-zero base from geometry while letting WBO/NCI reweight each triplet.
        total_weight = distance_weight * (1.0 + wbo_weight) * (1.0 + nci_weight)
        if total_weight <= 0.0 or not math.isfinite(total_weight):
            continue

        weighted_angle_sum += total_weight * float(triplet["angle_deg"])
        weight_sum += total_weight

        scored_triplet = dict(triplet)
        scored_triplet["weight"] = float(total_weight)
        scored_triplet["distance_weight"] = float(distance_weight)
        scored_triplet["wbo_weight"] = float(wbo_weight)
        scored_triplet["nci_weight"] = float(nci_weight)
        scored_triplets.append(scored_triplet)

    if weight_sum <= 1e-12:
        return {
            "f2": float("nan"),
            "f2_defined": 0,
            "triplet_count": len(triplets),
            "weight_sum": float(weight_sum),
            "undefined_reason": "no_positive_weights",
            "weight_model": "inv_ha_distance*(1+wbo_da)*(1+nci_local)",
            "top_triplets": [],
        }

    f2 = weighted_angle_sum / weight_sum
    scored_triplets.sort(key=lambda item: item["weight"], reverse=True)
    return {
        "f2": float(f2),
        "f2_defined": 1,
        "triplet_count": len(triplets),
        "weight_sum": float(weight_sum),
        "undefined_reason": None,
        "weight_model": "inv_ha_distance*(1+wbo_da)*(1+nci_local)",
        "top_triplets": scored_triplets[:5],
    }


def write_xyz(mol: Chem.Mol, filepath: str):
    """Writes current 3D coordinates to an XYZ file."""
    xyz_block = Chem.MolToXYZBlock(mol)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(xyz_block)


def promote_hbond_interaction(
    mol: Chem.Mol,
    frag1_indices: list[int],
    frag2_indices: list[int],
    target_ha_distance: float = 1.95,
) -> dict:
    """
    Attempts to enforce an intermolecular D-H...A contact between two fragments.

    Strategy:
    - Find donor hydrogens (H bound to N/O/F) on one fragment and acceptors (N/O/F) on the other.
    - Select the shortest existing H...A candidate.
    - Translate the acceptor fragment so A is placed along the D-H extension at target H...A distance.

    Returns diagnostic dict:
      {"applied": bool, "reason": str, "d_idx": int|None, "h_idx": int|None, "a_idx": int|None}
    """
    conf = mol.GetConformer()
    frag1 = set(frag1_indices)
    frag2 = set(frag2_indices)
    electronegative = {7, 8, 9}  # N, O, F

    donors = []      # (donor_idx, h_idx, donor_fragment_id)
    acceptors_f1 = []
    acceptors_f2 = []

    for idx in frag1 | frag2:
        atom = mol.GetAtomWithIdx(idx)
        an = atom.GetAtomicNum()
        if an in electronegative:
            if idx in frag1:
                acceptors_f1.append(idx)
            else:
                acceptors_f2.append(idx)
        if an != 1:
            continue
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            continue
        donor = neighbors[0]
        if donor.GetAtomicNum() not in electronegative:
            continue
        donors.append((donor.GetIdx(), idx, 1 if idx in frag1 else 2))

    candidates = []  # (ha_dist, d_idx, h_idx, a_idx, move_fragment)
    for d_idx, h_idx, donor_frag in donors:
        h = np.array(conf.GetAtomPosition(h_idx))
        if donor_frag == 1:
            target_acceptors = acceptors_f2
            move_fragment = frag2_indices
        else:
            target_acceptors = acceptors_f1
            move_fragment = frag1_indices
        for a_idx in target_acceptors:
            a = np.array(conf.GetAtomPosition(a_idx))
            ha_dist = float(np.linalg.norm(h - a))
            candidates.append((ha_dist, d_idx, h_idx, a_idx, move_fragment))

    if not candidates:
        return {"applied": False, "reason": "no_donor_acceptor_pair", "d_idx": None, "h_idx": None, "a_idx": None}

    candidates.sort(key=lambda x: x[0])
    _, d_idx, h_idx, a_idx, move_fragment = candidates[0]

    d = np.array(conf.GetAtomPosition(d_idx))
    h = np.array(conf.GetAtomPosition(h_idx))
    a = np.array(conf.GetAtomPosition(a_idx))

    dh = h - d
    dh_norm = np.linalg.norm(dh)
    if dh_norm < 1e-8:
        return {"applied": False, "reason": "degenerate_dh_vector", "d_idx": d_idx, "h_idx": h_idx, "a_idx": a_idx}

    # Place acceptor on D-H extension to favor near-linear D-H...A geometry.
    direction = dh / dh_norm
    target_a = h + direction * float(target_ha_distance)
    shift = target_a - a

    for idx in move_fragment:
        p = np.array(conf.GetAtomPosition(idx))
        new_p = p + shift
        conf.SetAtomPosition(idx, (float(new_p[0]), float(new_p[1]), float(new_p[2])))

    return {"applied": True, "reason": "ok", "d_idx": d_idx, "h_idx": h_idx, "a_idx": a_idx}
