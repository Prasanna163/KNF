from dataclasses import dataclass, asdict
import json
from typing import Optional
import os

@dataclass
class KNFResult:
    SNCI: float
    SCDI: Optional[float]
    SCDI_variance: float
    KNF_vector: list[float]
    metadata: dict


METRIC_SPECS = [
    ("SNCI", "SNCI_raw", ""),
    ("SCDI", "SCDI", ""),
    ("SCDI_variance", "SCDI_variance", ""),
    ("f1", "f1 (COM Dist)", "A"),
    ("f2", "f2 (HB Angle)", "deg"),
    ("f3", "f3 (Max Inter WBO)", ""),
    ("f4", "f4 (Dipole)", "D"),
    ("f5", "f5 (Pol)", "au"),
    ("f6", "f6 (NCI Count)", ""),
    ("f7", "f7 (NCI Mean)", ""),
    ("f8", "f8 (NCI Std)", ""),
    ("f9", "f9 (NCI Skew)", ""),
]

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
        
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        f2_defined = metadata.get("f2_defined")

        vec = result.KNF_vector
        f.write("KNF Vector Components:\n")
        f.write(f"f1 (COM Dist):  {vec[0]:.4f} A\n")
        if f2_defined == 0:
            reason = metadata.get("f2_undefined_reason", "undefined")
            f.write(f"f2 (HB Angle):  n/a ({reason})\n")
        else:
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


def _metric_value_map(result_dict: dict) -> dict:
    vector = result_dict.get("KNF_vector") or []
    return {
        "SNCI": result_dict.get("SNCI"),
        "SCDI": result_dict.get("SCDI"),
        "SCDI_variance": result_dict.get("SCDI_variance"),
        "f1": vector[0] if len(vector) > 0 else None,
        "f2": vector[1] if len(vector) > 1 else None,
        "f3": vector[2] if len(vector) > 2 else None,
        "f4": vector[3] if len(vector) > 3 else None,
        "f5": vector[4] if len(vector) > 4 else None,
        "f6": vector[5] if len(vector) > 5 else None,
        "f7": vector[6] if len(vector) > 6 else None,
        "f8": vector[7] if len(vector) > 7 else None,
        "f9": vector[8] if len(vector) > 8 else None,
    }


def _numeric_delta(current, reference):
    if current is None or reference is None:
        return None
    try:
        return float(current) - float(reference)
    except (TypeError, ValueError):
        return None


def _format_metric_value(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def write_water_delta_outputs(
    delta_txt_path: str,
    delta_json_path: str,
    water_result: KNFResult,
    reference_json_path: str,
    water_json_path: str,
):
    water_dict = asdict(water_result)
    water_metrics = _metric_value_map(water_dict)
    reference_found = os.path.exists(reference_json_path)
    reference_dict = None
    reference_metrics = {}

    if reference_found:
        with open(reference_json_path, "r", encoding="utf-8") as f:
            reference_dict = json.load(f)
        reference_metrics = _metric_value_map(reference_dict)

    metrics_payload = {}
    for key, label, unit in METRIC_SPECS:
        water_value = water_metrics.get(key)
        reference_value = reference_metrics.get(key) if reference_found else None
        metrics_payload[key] = {
            "label": label,
            "unit": unit or None,
            "reference": reference_value,
            "water": water_value,
            "delta": _numeric_delta(water_value, reference_value),
        }

    payload = {
        "comparison": "water_minus_reference",
        "reference_found": reference_found,
        "reference_file": reference_json_path,
        "water_file": water_json_path,
        "metrics": metrics_payload,
    }

    with open(delta_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    with open(delta_txt_path, "w", encoding="utf-8") as f:
        f.write("KNF-Core Water Delta Results\n")
        f.write("============================\n\n")
        f.write("Comparison: water - reference\n")
        f.write(f"Reference file: {reference_json_path}\n")
        f.write(f"Water file:     {water_json_path}\n\n")

        if not reference_found:
            f.write("Reference knf.json not found. Delta metrics are unavailable.\n\n")

        f.write(f"{'Metric':<22} {'Reference':>14} {'Water':>14} {'Delta':>14}\n")
        f.write(f"{'-' * 22} {'-' * 14} {'-' * 14} {'-' * 14}\n")
        for key, label, unit in METRIC_SPECS:
            metric = metrics_payload[key]
            f.write(
                f"{label:<22} "
                f"{_format_metric_value(metric['reference']):>14} "
                f"{_format_metric_value(metric['water']):>14} "
                f"{_format_metric_value(metric['delta']):>14}"
            )
            if unit:
                f.write(f"  {unit}")
            f.write("\n")
