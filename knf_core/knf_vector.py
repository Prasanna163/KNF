from dataclasses import dataclass, asdict
import json
from typing import Optional

@dataclass
class KNFResult:
    SNCI: float
    SCDI: Optional[float]
    SCDI_variance: float
    KNF_vector: list[float]
    metadata: dict

def assemble_knf_vector(
    f1: float, f2: float, 
    f3: float, f4: float, f5: float,
    f6: int, f7: float, f8: float, f9: float
) -> list[float]:
    """
    Assembles the 9D KNF vector.
    Order: [f1, f2, f3, f4, f5, f6, f7, f8, f9]
    """
    return [f1, f2, f3, f4, f5, float(f6), f7, f8, f9]

def write_output_txt(filepath: str, result: KNFResult):
    """Writes human-readable output.txt."""
    with open(filepath, 'w') as f:
        f.write("KNF-Core Analysis Results\n")
        f.write("=========================\n\n")
        f.write(f"SNCI_raw:       {result.SNCI:.6f}\n")
        if result.SCDI is None:
            f.write("SCDI:           n/a (set fixed var_min/var_max for normalization)\n")
        else:
            f.write(f"SCDI:           {result.SCDI:.6f}\n")
        f.write(f"SCDI_variance:  {result.SCDI_variance:.6f}\n\n")
        
        vec = result.KNF_vector
        f.write("KNF Vector Components:\n")
        f.write(f"f1 (COM Dist):  {vec[0]:.4f} A\n")
        f.write(f"f2 (HB Angle):  {vec[1]:.2f} deg\n")
        f.write(f"f3 (Max Inter WBO):   {vec[2]:.4f}\n")
        f.write(f"f4 (Dipole):    {vec[3]:.4f} D\n")
        f.write(f"f5 (Pol):       {vec[4]:.4f} au\n")
        f.write(f"f6 (NCI Count): {vec[5]:.0f}\n")
        f.write(f"f7 (NCI Mean):  {vec[6]:.6f}\n")
        f.write(f"f8 (NCI Std):   {vec[7]:.6f}\n")
        f.write(f"f9 (NCI Skew):  {vec[8]:.6f}\n")

def write_knf_json(filepath: str, result: KNFResult):
    """Writes machine-readable knf.json."""
    with open(filepath, 'w') as f:
        json.dump(asdict(result), f, indent=4)
