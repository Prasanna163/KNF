import csv
import json
import logging
import os
import shutil
from copy import deepcopy
from datetime import datetime, timezone

from .discovery import (
    _BATCH_LEGACY_CSV_NAMES,
    _BATCH_PRIMARY_CSV_NAME,
    _batch_primary_csv_path,
    _cleanup_redundant_batch_aliases,
)
from .kuid_ops import (
    _apply_kuid_prefix_fields,
    _compute_kuid_intensive_payload,
    _compute_kuid_payload,
    _extract_kuid_vector_from_values,
    _safe_float,
    _write_kuid_index_outputs,
    _write_kuid_intensive_distribution_outputs,
    _write_kuid_reverse_index_outputs,
)
from .naming import _final_output_name
from .quadrants import _compute_norm_and_quadrants


def _safe_source_label(seed: str, used: set[str]) -> str:
    raw = (seed or "").strip()
    if not raw:
        raw = "source"
    label = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in raw).strip("_")
    if not label:
        label = "source"
    candidate = label
    suffix = 2
    while candidate in used:
        candidate = f"{label}_{suffix:02d}"
        suffix += 1
    used.add(candidate)
    return candidate


def _load_source_records_from_batch_json(source_batch: str, json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    records = []
    for entry in (payload.get("records") or []):
        if not isinstance(entry, dict):
            continue
        item = deepcopy(entry)
        item["source_batch"] = source_batch
        item["result_dir"] = ""
        if not item.get("input_file_name"):
            input_file = item.get("input_file")
            if isinstance(input_file, str) and input_file.strip():
                item["input_file_name"] = os.path.basename(input_file)
        records.append(item)

    knf_results = []
    for item in (payload.get("knf_results") or []):
        if not isinstance(item, dict):
            continue
        out = deepcopy(item)
        out["source_batch"] = source_batch
        out["result_dir"] = ""
        knf_results.append(out)

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "records": records,
        "knf_results": knf_results,
        "summary": {
            "total_files": int(summary.get("total_files") or len(records)),
            "successful_files": int(summary.get("successful_files") or sum(1 for r in records if r.get("status") == "success")),
            "failed_files": int(summary.get("failed_files") or sum(1 for r in records if r.get("status") == "failed")),
            "stopped_files": int(summary.get("stopped_files") or sum(1 for r in records if r.get("status") == "stopped")),
            "total_time_seconds": float(summary.get("total_time_seconds") or 0.0),
        },
        "source_path": json_path,
        "source_type": "json",
    }


def _build_entry_from_csv_row(source_batch: str, csv_path: str, row: dict) -> tuple[dict, dict]:
    file_name = str(row.get("File") or "").strip()
    source_dir = os.path.dirname(csv_path)
    if file_name:
        input_file = os.path.abspath(os.path.join(source_dir, os.path.basename(file_name)))
        input_file_name = os.path.basename(file_name)
    else:
        input_file = os.path.abspath(csv_path)
        input_file_name = ""

    raw_vector = [row.get(f"f{i}") for i in range(1, 10)]
    vector_values = [_safe_float(value) for value in raw_vector]
    parsed_kuid_vector, _ = _extract_kuid_vector_from_values(raw_vector)
    status = "success" if parsed_kuid_vector is not None else "failed"

    metadata = {}
    f2_defined_raw = str(row.get("f2_defined") or "").strip()
    if f2_defined_raw:
        f2_defined_val = _safe_float(f2_defined_raw)
        if f2_defined_val is not None:
            metadata["f2_defined"] = int(f2_defined_val)
        else:
            metadata["f2_defined"] = f2_defined_raw
    f3_protocol = str(row.get("f3_protocol") or "").strip()
    if f3_protocol:
        if f3_protocol == "xtb":
            metadata["wbo_mode"] = "xtb"
        elif f3_protocol == "native":
            metadata["wbo_mode"] = "native"
        else:
            metadata["f3_definition"] = f3_protocol

    knf_payload = {
        "SNCI": _safe_float(row.get("SNCI")),
        "SCDI": _safe_float(row.get("SCDI")),
        "SCDI_variance": _safe_float(row.get("SCDI_variance")),
        "KNF_vector": vector_values,
        "metadata": metadata,
    }

    entry = {
        "input_file": input_file,
        "input_file_name": input_file_name,
        "result_dir": "",
        "status": status,
        "elapsed_seconds": 0.0,
        "error": None if status == "success" else "Missing valid KNF feature values in source CSV row.",
        "knf": knf_payload,
        "source_batch": source_batch,
        "SNCI_Norm": _safe_float(row.get("SNCI_Norm")),
        "SCDI_Norm": _safe_float(row.get("SCDI_Norm")),
    }

    knf_result = {
        "input_file": input_file,
        "input_file_name": input_file_name,
        "result_dir": "",
        "knf": deepcopy(knf_payload),
        "source_batch": source_batch,
    }
    return entry, knf_result


def _load_source_records_from_batch_csv(source_batch: str, csv_path: str) -> dict:
    records = []
    knf_results = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            entry, knf_result = _build_entry_from_csv_row(source_batch, csv_path, row)
            records.append(entry)
            if entry.get("status") == "success":
                knf_results.append(knf_result)

    return {
        "records": records,
        "knf_results": knf_results,
        "summary": {
            "total_files": len(records),
            "successful_files": sum(1 for r in records if r.get("status") == "success"),
            "failed_files": sum(1 for r in records if r.get("status") == "failed"),
            "stopped_files": sum(1 for r in records if r.get("status") == "stopped"),
            "total_time_seconds": 0.0,
        },
        "source_path": csv_path,
        "source_type": "csv",
    }


def _load_source_records(source_batch: str, source_path: str, source_type: str) -> dict:
    if source_type == "json":
        return _load_source_records_from_batch_json(source_batch, source_path)
    return _load_source_records_from_batch_csv(source_batch, source_path)


def _write_combined_batch_outputs(
    source_directory: str,
    output_root: str,
    source_summaries: list[dict],
    combined_records: list[dict],
    combined_knf_results: list[dict],
    total_time_seconds: float,
    water: bool = False,
    mode: str = "combined_from_existing_batches",
) -> dict:
    os.makedirs(output_root, exist_ok=True)
    successful_records = [
        record for record in combined_records if record.get("status") == "success"
    ]
    successful_knf_results = [
        result for result in combined_knf_results if isinstance(result, dict)
    ]

    quadrant_payload = _compute_norm_and_quadrants(
        enriched_records=successful_records,
        results_root=output_root,
        water=water,
        interactive_plot=False,
    )
    kuid_payload = _compute_kuid_payload(
        enriched_records=successful_records,
        results_root=output_root,
        water=water,
    )
    kuid_intensive_payload = _compute_kuid_intensive_payload(
        enriched_records=successful_records,
        results_root=output_root,
        water=water,
    )

    summary = {
        "total_files": len(combined_records),
        "successful_files": sum(1 for r in combined_records if r.get("status") == "success"),
        "failed_files": sum(1 for r in combined_records if r.get("status") == "failed"),
        "stopped_files": sum(1 for r in combined_records if r.get("status") == "stopped"),
        "total_time_seconds": round(float(total_time_seconds), 4),
    }

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": os.path.abspath(source_directory),
        "results_root": os.path.abspath(output_root),
        "mode": mode,
        "workers": None,
        "source_batches": source_summaries,
        "summary": summary,
        "normalization_and_quadrants": quadrant_payload,
        "kuid": kuid_payload,
        "kuid_intensive": kuid_intensive_payload,
        "records": successful_records,
        "knf_results": successful_knf_results,
    }

    aggregate_json_path = os.path.join(output_root, _final_output_name("batch_knf.json", water))
    aggregate_csv_path = _batch_primary_csv_path(output_root, water=water)

    csv_fields = [
        "source_batch",
        "File",
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "f2_defined",
        "f3_protocol",
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
    csv_rows = []
    with open(aggregate_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for entry in successful_records:
            knf_data = entry.get("knf") or {}
            knf_vector = knf_data.get("KNF_vector") or []
            metadata = knf_data.get("metadata") if isinstance(knf_data, dict) else None
            row = {
                "source_batch": entry.get("source_batch", ""),
                "File": entry.get("input_file_name", ""),
                "f2_defined": (metadata or {}).get("f2_defined", ""),
                "f3_protocol": (metadata or {}).get(
                    "f3_definition",
                    (metadata or {}).get("wbo_mode", ""),
                ),
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

    kuid_index_outputs = _write_kuid_index_outputs(csv_rows, output_root, water=water)
    kuid_reverse_index_outputs = _write_kuid_reverse_index_outputs(csv_rows, output_root, water=water)
    kuid_intensive_distribution_outputs = _write_kuid_intensive_distribution_outputs(
        csv_rows, output_root, water=water
    )

    if isinstance(payload.get("kuid"), dict) and payload["kuid"].get("enabled"):
        payload["kuid"].update(kuid_index_outputs)
        payload["kuid"].update(kuid_reverse_index_outputs)
    if isinstance(payload.get("kuid_intensive"), dict) and payload["kuid_intensive"].get("enabled"):
        payload["kuid_intensive"].update(kuid_intensive_distribution_outputs)
    payload["kuid_reverse_index"] = kuid_reverse_index_outputs
    payload["kuid_intensive_distribution"] = kuid_intensive_distribution_outputs

    with open(aggregate_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    removed_aliases = _cleanup_redundant_batch_aliases(
        results_root=output_root,
        primary_csv_path=aggregate_csv_path,
        primary_json_path=aggregate_json_path,
        water=water,
    )

    kuid_calibration = os.path.join(output_root, _final_output_name("kuid_calibration.json", water))
    kuid_int_calibration = os.path.join(output_root, _final_output_name("kuid_intensive_calibration.json", water))
    kuid_calibration_unified = os.path.join(
        output_root, _final_output_name("kuid_calibration_unified.json", water)
    )
    kuid_int_calibration_unified = os.path.join(
        output_root, _final_output_name("kuid_intensive_calibration_unified.json", water)
    )
    if os.path.exists(kuid_calibration):
        try:
            shutil.copyfile(kuid_calibration, kuid_calibration_unified)
        except Exception as e:
            import logging

            logging.warning("Could not write KUID unified calibration alias %s: %s", kuid_calibration_unified, e)
    if os.path.exists(kuid_int_calibration):
        try:
            shutil.copyfile(kuid_int_calibration, kuid_int_calibration_unified)
        except Exception as e:
            import logging

            logging.warning(
                "Could not write KUID-Intensive unified calibration alias %s: %s",
                kuid_int_calibration_unified,
                e,
            )

    return {
        "output_root": output_root,
        "batch_json": aggregate_json_path,
        "batch_csv": aggregate_csv_path,
        "removed_aliases": removed_aliases,
    }


def _combine_batch_sources(
    source_directory: str,
    source_specs: list[dict],
    output_root: str,
    water: bool = False,
    mode: str = "combined_from_existing_batches",
) -> dict:
    combined_records = []
    combined_knf_results = []
    source_summaries = []
    total_time_seconds = 0.0

    for spec in source_specs:
        source_batch = spec.get("source_batch")
        source_path = spec.get("path")
        source_type = spec.get("type")
        if not source_batch or not source_path or not source_type:
            continue
        loaded = _load_source_records(source_batch, source_path, source_type)
        combined_records.extend(loaded.get("records") or [])
        combined_knf_results.extend(loaded.get("knf_results") or [])
        summary = loaded.get("summary") or {}
        total_time_seconds += float(summary.get("total_time_seconds") or 0.0)
        source_summaries.append(
            {
                "source_batch": source_batch,
                "source_path": loaded.get("source_path") or source_path,
                "source_type": loaded.get("source_type") or source_type,
                "total_files": int(summary.get("total_files") or 0),
                "successful_files": int(summary.get("successful_files") or 0),
                "failed_files": int(summary.get("failed_files") or 0),
                "stopped_files": int(summary.get("stopped_files") or 0),
                "total_time_seconds": float(summary.get("total_time_seconds") or 0.0),
            }
        )

    if not combined_records:
        raise ValueError("No records were loaded from batch sources for combined KUID recomputation.")

    return _write_combined_batch_outputs(
        source_directory=source_directory,
        output_root=output_root,
        source_summaries=source_summaries,
        combined_records=combined_records,
        combined_knf_results=combined_knf_results,
        total_time_seconds=total_time_seconds,
        water=water,
        mode=mode,
    )


def _discover_universal_batch_sources(directory: str, water: bool = False) -> list[dict]:
    json_name = _final_output_name("batch_knf.json", water)
    csv_names = [
        _final_output_name(_BATCH_PRIMARY_CSV_NAME, water),
        _final_output_name(_BATCH_LEGACY_CSV_NAMES[0], water),
    ]
    used_labels = set()
    specs = []

    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d.lower() not in {"combined results", "combined_results", "combined-results"}]
        file_set = set(files)
        if json_name in file_set:
            label = _safe_source_label(os.path.basename(root), used_labels)
            specs.append(
                {
                    "source_batch": label,
                    "path": os.path.join(root, json_name),
                    "type": "json",
                }
            )
            continue
        csv_name = next((name for name in csv_names if name in file_set), None)
        if csv_name:
            label = _safe_source_label(os.path.basename(root), used_labels)
            specs.append(
                {
                    "source_batch": label,
                    "path": os.path.join(root, csv_name),
                    "type": "csv",
                }
            )

    specs.sort(key=lambda item: item["source_batch"])
    return specs


# NOTE: This function was originally grouped with the KUID helpers in
# main.py (contiguous with _persist_entry_outputs_with_kuid etc.), but it is
# placed here rather than in engine/kuid_ops.py because its whole job is to
# build source_specs and call _combine_batch_sources (defined in this same
# module); keeping it in kuid_ops.py would create a circular import, since
# this module already depends on kuid_ops.py for the KUID compute helpers.
def _merge_master_and_batch_csv(master_csv: str, new_csv: str, args) -> dict:
    water_mode = bool(getattr(args, "water", False))
    master_csv = os.path.abspath(master_csv)
    new_csv = os.path.abspath(new_csv)
    if not os.path.exists(master_csv):
        raise FileNotFoundError(f"Master CSV not found: {master_csv}")
    if not os.path.exists(new_csv):
        raise FileNotFoundError(f"New CSV not found: {new_csv}")

    output_root = getattr(args, "merge_output_dir", None)
    if output_root:
        output_root = os.path.abspath(output_root)
    else:
        output_root = os.path.join(os.path.dirname(master_csv), "Combined Results")
    os.makedirs(output_root, exist_ok=True)

    source_specs = [
        {"source_batch": "master_batch", "path": master_csv, "type": "csv"},
        {"source_batch": "new_batch", "path": new_csv, "type": "csv"},
    ]
    result = _combine_batch_sources(
        source_directory=os.path.dirname(master_csv) or os.getcwd(),
        source_specs=source_specs,
        output_root=output_root,
        water=water_mode,
        mode="merge_master_and_new_csv",
    )

    if bool(getattr(args, "overwrite_master_csv", False)):
        shutil.copy2(result["batch_csv"], master_csv)
        result["master_csv_updated"] = master_csv

    return result
