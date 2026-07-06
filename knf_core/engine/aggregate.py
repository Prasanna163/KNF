import csv
import json
import os
from datetime import datetime, timezone

from .discovery import _batch_primary_csv_path, _cleanup_redundant_batch_aliases
from .kuid_ops import (
    _apply_kuid_prefix_fields,
    _compute_kuid_intensive_payload,
    _compute_kuid_payload,
    _write_kuid_index_outputs,
    _write_kuid_intensive_distribution_outputs,
    _write_kuid_reverse_index_outputs,
)
from .naming import _final_output_name
from .quadrants import _compute_norm_and_quadrants
from .water_delta import write_batch_water_delta_outputs


def write_batch_aggregate_json(
    directory: str,
    results_root: str,
    records: list[dict],
    mode: str,
    workers: int,
    total_time: float,
    water: bool = False,
    interactive_quadrant_plot: bool = False,
):
    """Writes combined JSON and CSV payloads for batch outputs."""
    aggregate_path = os.path.join(results_root, _final_output_name("batch_knf.json", water))
    aggregate_csv_path = _batch_primary_csv_path(results_root, water=water)
    delta_json_path = None
    delta_txt_path = None
    os.makedirs(results_root, exist_ok=True)

    enriched_records = []
    knf_results = []
    success_count = 0
    failure_count = 0
    stopped_count = 0

    for record in records:
        input_file = os.path.abspath(record["input_file"])
        stem = os.path.splitext(os.path.basename(input_file))[0]
        result_dir = os.path.join(results_root, stem)
        knf_path = os.path.join(result_dir, _final_output_name("knf.json", water))

        entry = {
            "input_file": input_file,
            "input_file_name": os.path.basename(input_file),
            "result_dir": result_dir,
            "status": record["status"],
            "elapsed_seconds": round(float(record.get("elapsed_seconds", 0.0)), 4),
            "error": record.get("error"),
            "knf": None,
        }

        if record["status"] == "success":
            knf_data = record.get("knf") if isinstance(record.get("knf"), dict) else None
            if knf_data is None and os.path.exists(knf_path):
                try:
                    with open(knf_path, "r", encoding="utf-8") as f:
                        knf_data = json.load(f)
                except Exception as e:
                    entry["status"] = "failed"
                    entry["error"] = f"Failed to read {_final_output_name('knf.json', water)}: {e}"
                    failure_count += 1
            if knf_data is not None and entry["status"] == "success":
                entry["knf"] = knf_data
                knf_results.append(
                    {
                        "input_file": input_file,
                        "input_file_name": os.path.basename(input_file),
                        "result_dir": result_dir,
                        "knf": knf_data,
                    }
                )
                success_count += 1
            elif entry["status"] == "success":
                entry["status"] = "failed"
                entry["error"] = (
                    f"Missing KNF payload (record.knf or {_final_output_name('knf.json', water)} output)."
                )
                failure_count += 1
        elif record["status"] == "stopped":
            stopped_count += 1
        else:
            failure_count += 1

        enriched_records.append(entry)

    successful_records = [
        entry for entry in enriched_records if entry.get("status") == "success"
    ]

    quadrant_payload = _compute_norm_and_quadrants(
        enriched_records=successful_records,
        results_root=results_root,
        water=water,
        interactive_plot=interactive_quadrant_plot,
    )
    kuid_payload = _compute_kuid_payload(
        enriched_records=successful_records,
        results_root=results_root,
        water=water,
    )
    kuid_intensive_payload = _compute_kuid_intensive_payload(
        enriched_records=successful_records,
        results_root=results_root,
        water=water,
    )

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": os.path.abspath(directory),
        "results_root": os.path.abspath(results_root),
        "mode": mode,
        "workers": workers,
        "summary": {
            "total_files": len(records),
            "successful_files": success_count,
            "failed_files": failure_count,
            "stopped_files": stopped_count,
            "total_time_seconds": round(float(total_time), 4),
        },
        "normalization_and_quadrants": quadrant_payload,
        "kuid": kuid_payload,
        "kuid_intensive": kuid_intensive_payload,
        "records": successful_records,
        "knf_results": knf_results,
    }

    csv_fields = (
        ["File"]
        + [f"f{i}" for i in range(1, 10)]
        + [
            "f2_defined",
            "KUID_raw",
            "KUID",
            "KUID_Cluster",
            "KUID_Intensive_raw",
            "KUID_Intensive",
            "KUID_Intensive_Cluster",
            "KUID_prefix2",
            "KUID_prefix4",
            "KUID_prefix6",
            "SNCI",
            "SCDI_variance",
            "SNCI_Norm",
            "SCDI_Norm",
        ]
    )
    csv_rows = []
    with open(aggregate_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for entry in successful_records:
            knf_data = entry.get("knf") or {}
            knf_vector = knf_data.get("KNF_vector") or []
            metadata = knf_data.get("metadata") if isinstance(knf_data, dict) else None
            row = {
                "File": entry.get("input_file_name", ""),
                "f2_defined": (metadata or {}).get("f2_defined", ""),
                "KUID_raw": entry.get("KUID_raw", ""),
                "KUID": entry.get("KUID", ""),
                "KUID_Cluster": entry.get("KUID_Cluster", ""),
                "KUID_Intensive_raw": entry.get("KUID_Intensive_raw", ""),
                "KUID_Intensive": entry.get("KUID_Intensive", ""),
                "KUID_Intensive_Cluster": entry.get("KUID_Intensive_Cluster", ""),
                "SNCI": knf_data.get("SNCI", ""),
                "SCDI_variance": knf_data.get("SCDI_variance", ""),
                "SNCI_Norm": entry.get("SNCI_Norm", ""),
                "SCDI_Norm": entry.get("SCDI_Norm", ""),
            }
            _apply_kuid_prefix_fields(row)
            for idx in range(9):
                row[f"f{idx + 1}"] = knf_vector[idx] if idx < len(knf_vector) else ""
            writer.writerow(row)
            csv_rows.append(row)

    kuid_index_outputs = _write_kuid_index_outputs(csv_rows, results_root, water=water)
    kuid_reverse_index_outputs = _write_kuid_reverse_index_outputs(csv_rows, results_root, water=water)
    kuid_intensive_distribution_outputs = _write_kuid_intensive_distribution_outputs(
        csv_rows, results_root, water=water
    )
    if isinstance(payload.get("kuid"), dict) and payload["kuid"].get("enabled"):
        payload["kuid"].update(kuid_index_outputs)
        payload["kuid"].update(kuid_reverse_index_outputs)
    if isinstance(payload.get("kuid_intensive"), dict) and payload["kuid_intensive"].get("enabled"):
        payload["kuid_intensive"].update(kuid_intensive_distribution_outputs)
    payload["kuid_reverse_index"] = kuid_reverse_index_outputs
    payload["kuid_intensive_distribution"] = kuid_intensive_distribution_outputs

    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _cleanup_redundant_batch_aliases(
        results_root=results_root,
        primary_csv_path=aggregate_csv_path,
        primary_json_path=aggregate_path,
        water=water,
    )

    if water:
        delta_json_path = os.path.join(results_root, _final_output_name("batch_delta.json", water))
        delta_txt_path = os.path.join(results_root, _final_output_name("batch_delta.txt", water))
        write_batch_water_delta_outputs(
            delta_txt_path=delta_txt_path,
            delta_json_path=delta_json_path,
            reference_aggregate_path=os.path.join(results_root, "batch_knf.json"),
            water_aggregate_path=aggregate_path,
            water_payload=payload,
        )

    return aggregate_path, aggregate_csv_path, quadrant_payload, delta_json_path, delta_txt_path
