import json
import os

from .. import knf_vector
from .constants import CLI_NAME

BATCH_METRIC_SPECS = list(knf_vector.METRIC_SPECS) + [
    ("SNCI_Norm", "SNCI_Norm", ""),
    ("SCDI_Norm", "SCDI_Norm", ""),
]


def _metric_value_map_from_batch_entry(entry: dict) -> dict:
    knf_data = entry.get("knf") or {}
    metadata = knf_data.get("metadata") if isinstance(knf_data, dict) else None
    vector = knf_data.get("KNF_vector") or []
    return {
        "SNCI": knf_data.get("SNCI"),
        "SCDI": knf_data.get("SCDI"),
        "SCDI_variance": knf_data.get("SCDI_variance"),
        "f1": vector[0] if len(vector) > 0 else None,
        "f2": vector[1] if len(vector) > 1 else None,
        "f3": vector[2] if len(vector) > 2 else None,
        "f4": vector[3] if len(vector) > 3 else None,
        "f5": vector[4] if len(vector) > 4 else None,
        "f6": vector[5] if len(vector) > 5 else None,
        "f7": vector[6] if len(vector) > 6 else None,
        "f8": vector[7] if len(vector) > 7 else None,
        "f9": vector[8] if len(vector) > 8 else None,
        "f2_defined": (metadata or {}).get("f2_defined"),
        "SNCI_Norm": entry.get("SNCI_Norm"),
        "SCDI_Norm": entry.get("SCDI_Norm"),
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


def write_batch_water_delta_outputs(
    delta_txt_path: str,
    delta_json_path: str,
    reference_aggregate_path: str,
    water_aggregate_path: str,
    water_payload: dict,
):
    reference_found = os.path.exists(reference_aggregate_path)
    reference_payload = None
    if reference_found:
        with open(reference_aggregate_path, "r", encoding="utf-8") as f:
            reference_payload = json.load(f)

    water_records = {
        entry.get("input_file"): entry
        for entry in (water_payload.get("records") or [])
        if entry.get("input_file")
    }
    reference_records = {
        entry.get("input_file"): entry
        for entry in ((reference_payload or {}).get("records") or [])
        if entry.get("input_file")
    }

    summary_keys = [
        "total_files",
        "successful_files",
        "failed_files",
        "stopped_files",
        "total_time_seconds",
    ]
    water_summary = (water_payload.get("summary") or {}).copy()
    reference_summary = ((reference_payload or {}).get("summary") or {}).copy()
    summary_delta = {
        key: _numeric_delta(water_summary.get(key), reference_summary.get(key))
        for key in summary_keys
    }

    file_deltas = []
    for input_file in sorted(set(reference_records) | set(water_records)):
        water_entry = water_records.get(input_file, {})
        reference_entry = reference_records.get(input_file, {})
        water_metrics = _metric_value_map_from_batch_entry(water_entry)
        reference_metrics = _metric_value_map_from_batch_entry(reference_entry)
        metrics_payload = {}
        for key, label, unit in BATCH_METRIC_SPECS:
            water_value = water_metrics.get(key)
            reference_value = reference_metrics.get(key)
            metrics_payload[key] = {
                "label": label,
                "unit": unit or None,
                "reference": reference_value,
                "water": water_value,
                "delta": _numeric_delta(water_value, reference_value),
            }

        file_deltas.append(
            {
                "input_file": input_file,
                "input_file_name": water_entry.get("input_file_name") or reference_entry.get("input_file_name"),
                "reference_status": reference_entry.get("status"),
                "water_status": water_entry.get("status"),
                "reference_result_dir": reference_entry.get("result_dir"),
                "water_result_dir": water_entry.get("result_dir"),
                "metrics": metrics_payload,
            }
        )

    payload = {
        "comparison": "water_minus_reference",
        "reference_found": reference_found,
        "reference_file": reference_aggregate_path,
        "water_file": water_aggregate_path,
        "summary": {
            "reference": reference_summary,
            "water": water_summary,
            "delta": summary_delta,
        },
        "files": file_deltas,
    }

    with open(delta_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(delta_txt_path, "w", encoding="utf-8") as f:
        f.write(f"{CLI_NAME} Batch Water Delta Results\n")
        f.write("=================================\n\n")
        f.write("Comparison: water - reference\n")
        f.write(f"Reference batch file: {reference_aggregate_path}\n")
        f.write(f"Water batch file:     {water_aggregate_path}\n\n")

        if not reference_found:
            f.write("Reference batch_knf.json not found. Batch delta metrics are unavailable.\n")
            return

        f.write("Summary:\n")
        for key in summary_keys:
            f.write(
                f"  {key}: "
                f"reference={_format_metric_value(reference_summary.get(key))} "
                f"water={_format_metric_value(water_summary.get(key))} "
                f"delta={_format_metric_value(summary_delta.get(key))}\n"
            )
        f.write("\n")

        for file_delta in file_deltas:
            name = file_delta.get("input_file_name") or file_delta.get("input_file") or "unknown"
            f.write(f"{name}\n")
            f.write(f"{'-' * len(name)}\n")
            f.write(
                f"Status: reference={file_delta.get('reference_status', 'n/a')} "
                f"water={file_delta.get('water_status', 'n/a')}\n"
            )
            f.write(f"{'Metric':<22} {'Reference':>14} {'Water':>14} {'Delta':>14}\n")
            f.write(f"{'-' * 22} {'-' * 14} {'-' * 14} {'-' * 14}\n")
            for key, label, unit in BATCH_METRIC_SPECS:
                metric = file_delta["metrics"][key]
                f.write(
                    f"{label:<22} "
                    f"{_format_metric_value(metric['reference']):>14} "
                    f"{_format_metric_value(metric['water']):>14} "
                    f"{_format_metric_value(metric['delta']):>14}"
                )
                if unit:
                    f.write(f"  {unit}")
                f.write("\n")
            f.write("\n")
