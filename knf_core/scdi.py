import os
import logging
import numpy as np

def compute_scdi(cosmo_path: str) -> float:
    """
    Computes SCDI variance from COSMO file.
    SCDI = sum(area * (charge - mean_charge)^2) / sum(area)
    Wait, formula in plan:
    mu_A = sum(a_i q_i) / sum(a_i)
    SCDI = sum(a_i (q_i - mu_A)^2) / sum(a_i)
    """
    if not os.path.exists(cosmo_path):
        logging.warning(f"COSMO file not found: {cosmo_path}")
        return 0.0
        
    areas = []
    charges = []
    
    # Parsing COSMO file
    # Format depends on software. xTB writes standard COSMO format?
    # Usually segment data starts after some header.
    # Lines look like: 
    #   n   x   y   z   area   charge ...
    # Or
    #  segment information
    #  ...
    
    # We need to withstand different formats or standard .cosmo from xTB.
    # xTB .cosmo format:
    # $segment_information
    # atom  n   x   y   z   area   charge   pot
    # 1     1   ...
    
    with open(cosmo_path, 'r') as f:
        in_segment = False
        for line in f:
            if '$segment_information' in line:
                in_segment = True
                continue
            if line.startswith('$') and in_segment:
                in_segment = False
                continue
                
            if in_segment:
                parts = line.split()
                # Expected: atom n x y z area charge pot
                # Check column count. Usually area is col 6 (index 5), charge is col 7 (index 6)
                # But sometimes columns vary.
                # Let's try to parse floats.
                try:
                    # xTB COSCO format:
                    #   #   atom   n   x   y   z   area   charge   pot
                    # No, actually:
                    #  1  1  1.23 ...
                    if len(parts) >= 8:
                        # Correct mapping based on xtb.cosmo file:
                        # Column 5 (index 5) is CHARGE
                        # Column 6 (index 6) is AREA
                        
                        charge = float(parts[5])
                        area = float(parts[6])
                        
                        areas.append(area)
                        charges.append(charge)

                except ValueError:
                    continue

    if not areas:
        return 0.0
        
    areas = np.array(areas)
    charges = np.array(charges)
    
    total_area = np.sum(areas)
    if total_area == 0:
        return 0.0
        
    # Mean charge density? No, q_i is charge.
    # Plan says: mu_A = sum(a_i q_i) / sum(a_i)
    # This is area-weighted mean charge.
    
    mu_A = np.sum(areas * charges) / total_area
    
    # Variance
    # SCDI = sum(a_i (q_i - mu_A)^2) / sum(a_i)
    
    scdi = 1-np.sum(areas * (charges - mu_A)**2) / total_area
    
    # Ensure non-negative variance despite potential floating point noise
    return max(0.0, float(scdi))

