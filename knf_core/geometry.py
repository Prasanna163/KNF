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
    Returns 180.0 if no HB is detected.
    
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
        return 180.0
        
    # Sort by distance, take the shortest one
    possible_hbs.sort(key=lambda x: x[0])
    best_hb = possible_hbs[0]
    
    return best_hb[1]
