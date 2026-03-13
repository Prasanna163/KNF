import argparse
import sys
import os
import shutil
import logging
import time
import json
import csv
import statistics
from copy import deepcopy
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError, CancelledError
from datetime import datetime, timezone
from .pipeline import KNFPipeline
from . import utils
from . import autoconfig
from . import first_run
from . import knf_vector, kuid, kuid_index, kuid_intensive
import psutil
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.table import Table

CLI_TITLE = "KNF-Core v1.0.5"
DISPLAY_NAME_LIMIT = 40
STOP_KEY = "q"
VALID_INPUT_EXTS = {".xyz", ".sdf", ".mol", ".pdb", ".mol2"}


def _final_output_name(filename: str, water: bool) -> str:
    if not water:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}_water{ext}"


_BATCH_PRIMARY_CSV_NAME = "batch_knf_unified_kuid_intensive.csv"
_BATCH_LEGACY_CSV_NAMES = ("batch_knf.csv", "batch_knf_unified.csv")
_BATCH_LEGACY_JSON_NAMES = ("batch_knf_unified_kuid_intensive.json",)


def _batch_primary_csv_path(results_root: str, water: bool = False) -> str:
    return os.path.join(results_root, _final_output_name(_BATCH_PRIMARY_CSV_NAME, water))


def _batch_candidate_csv_paths(results_root: str, water: bool = False) -> list[str]:
    names = [_BATCH_PRIMARY_CSV_NAME, *_BATCH_LEGACY_CSV_NAMES]
    seen = set()
    paths = []
    for name in names:
        path = os.path.join(results_root, _final_output_name(name, water))
        norm = os.path.normcase(os.path.abspath(path))
        if norm in seen:
            continue
        seen.add(norm)
        paths.append(path)
    return paths


def _existing_batch_csv_path(results_root: str, water: bool = False) -> str:
    for path in _batch_candidate_csv_paths(results_root, water=water):
        if os.path.exists(path):
            return path
    return _batch_primary_csv_path(results_root, water=water)


def _cleanup_redundant_batch_aliases(
    results_root: str,
    primary_csv_path: str,
    primary_json_path: str = None,
    water: bool = False,
) -> list[str]:
    protected = {os.path.normcase(os.path.abspath(primary_csv_path))}
    if primary_json_path:
        protected.add(os.path.normcase(os.path.abspath(primary_json_path)))

    removed = []
    alias_names = [*_BATCH_LEGACY_CSV_NAMES, *_BATCH_LEGACY_JSON_NAMES]
    for name in alias_names:
        alias_path = os.path.join(results_root, _final_output_name(name, water))
        norm_alias = os.path.normcase(os.path.abspath(alias_path))
        if norm_alias in protected:
            continue
        if not os.path.exists(alias_path):
            continue
        try:
            os.remove(alias_path)
            removed.append(alias_path)
        except Exception as e:
            logging.warning("Could not remove redundant batch alias %s: %s", alias_path, e)

    return removed


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
        f.write("KNF-Core Batch Water Delta Results\n")
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

def check_dependencies(multiwfn_path: str = None, nci_backend: str = "torch"):
    """Checks if required external tools are available in PATH."""
    # Attempt to add Multiwfn to PATH if missing
    utils.ensure_multiwfn_in_path(explicit_path=multiwfn_path)

    missing = []
    
    if not shutil.which('obabel'):
        missing.append('obabel (Open Babel)')
        
    if not shutil.which('xtb'):
        missing.append('xtb (Extended Tight Binding)')
        
    backend = (nci_backend or "multiwfn").strip().lower()
    if backend == "multiwfn" and not shutil.which('Multiwfn') and not shutil.which('Multiwfn.exe'):
        missing.append('Multiwfn')
        
    if missing:
        print("WARNING: The following required tools were not found in your PATH:")
        for tool in missing:
            print(f"  - {tool}")
        print("Please resolve these dependencies for full functionality.")
        print("-" * 50)

def _build_pipeline(file_path: str, args, output_root: str = None) -> KNFPipeline:
    return KNFPipeline(
        input_file=file_path,
        charge=args.charge,
        spin=args.spin,
        water=args.water,
        force=args.force,
        clean=args.clean,
        debug=args.debug,
        output_root=output_root,
        keep_full_files=args.full_files,
        nci_backend=args.nci_backend,
        nci_grid_spacing=args.nci_grid_spacing,
        nci_grid_padding=args.nci_grid_padding,
        nci_device=args.nci_device,
        nci_dtype=args.nci_dtype,
        nci_batch_size=args.nci_batch_size,
        nci_eig_batch_size=args.nci_eig_batch_size,
        nci_rho_floor=args.nci_rho_floor,
        nci_apply_primitive_norm=args.nci_apply_primitive_norm,
        scdi_var_min=args.scdi_var_min,
        scdi_var_max=args.scdi_var_max,
        wbo_mode=getattr(args, "wbo_mode", "native"),
    )


def process_file(file_path: str, args, output_root: str = None):
    """Runs the pipeline for a single file and returns status."""
    start = time.perf_counter()
    try:
        pipeline = _build_pipeline(file_path, args, output_root=output_root)
        pipeline.run()
        return True, None, time.perf_counter() - start
    except Exception as e:
        if args.debug:
            logging.exception(f"Error processing {file_path}:")
        else:
            logging.error(f"Error processing {file_path}: {e}")
        return False, str(e), time.perf_counter() - start


def process_file_pre_nci(file_path: str, args, output_root: str = None):
    """Runs pre-NCI stages only (geometry + xTB) and returns pipeline context."""
    start = time.perf_counter()
    try:
        pipeline = _build_pipeline(file_path, args, output_root=output_root)
        context = pipeline.run_pre_nci_stage()
        return True, None, time.perf_counter() - start, pipeline, context
    except Exception as e:
        if args.debug:
            logging.exception(f"Pre-NCI error processing {file_path}:")
        else:
            logging.error(f"Pre-NCI error processing {file_path}: {e}")
        return False, str(e), time.perf_counter() - start, None, None


def process_file_post_nci(pipeline: KNFPipeline, context: dict, file_path: str):
    """Runs post-NCI stage (NCI + SNCI/SCDI + final output write)."""
    start = time.perf_counter()
    try:
        pipeline.run_post_nci_stage(context)
        return True, None, time.perf_counter() - start
    except Exception as e:
        if pipeline.debug:
            logging.exception(f"Post-NCI error processing {file_path}:")
        else:
            logging.error(f"Post-NCI error processing {file_path}: {e}")
        return False, str(e), time.perf_counter() - start


def _fmt_elapsed(seconds: float) -> str:
    seconds = int(max(0, round(seconds)))
    mm, ss = divmod(seconds, 60)
    hh, mm = divmod(mm, 60)
    if hh > 0:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"


def _display_name(file_path: str) -> str:
    stem = os.path.splitext(os.path.basename(file_path))[0]
    cleaned = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in stem)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_") or stem
    label = cleaned + os.path.splitext(os.path.basename(file_path))[1]
    if len(label) > DISPLAY_NAME_LIMIT:
        return label[: DISPLAY_NAME_LIMIT - 3] + "..."
    return label


def _active_tool_ram_mb() -> float:
    total = 0
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            name = (p.info.get("name") or "").lower()
            if "xtb" in name or "multiwfn" in name:
                mem = p.info.get("memory_info")
                total += int(getattr(mem, "rss", 0))
        except Exception:
            continue
    return total / (1024 * 1024)


def _self_ram_mb() -> float:
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _safe_float(value):
    try:
        if value is None:
            return None
        val = float(value)
        if val != val:  # NaN
            return None
        return val
    except (TypeError, ValueError):
        return None


def _normalize_minmax(values, invert: bool = False):
    finite = [v for v in values if v is not None]
    if not finite:
        return [None] * len(values)

    vmin = min(finite)
    vmax = max(finite)
    if abs(vmax - vmin) <= 1e-12:
        return [0.5 if v is not None else None for v in values]

    out = []
    for v in values:
        if v is None:
            out.append(None)
            continue
        normalized = (v - vmin) / (vmax - vmin)
        if invert:
            normalized = 1.0 - normalized
        out.append(max(0.0, min(1.0, float(normalized))))
    return out


def _extract_knf_vector(entry: dict):
    knf_data = entry.get("knf") or {}
    vector = knf_data.get("KNF_vector") or []
    if len(vector) < 9:
        return None
    values = [_safe_float(vector[idx]) for idx in range(9)]
    if any(v is None for v in values):
        return None
    return values


_KUID_NON_F2_REQUIRED_INDEXES = (0, 2, 3, 4, 5, 6, 7, 8)


def _extract_kuid_vector_from_values(values: list):
    if len(values) < 9:
        return None, False
    parsed = [_safe_float(values[idx]) for idx in range(9)]
    if any(parsed[idx] is None for idx in _KUID_NON_F2_REQUIRED_INDEXES):
        return None, False
    f2_surrogate_needed = parsed[1] is None
    return parsed, f2_surrogate_needed


def _extract_kuid_vector_from_entry(entry: dict):
    knf_data = entry.get("knf") or {}
    vector = knf_data.get("KNF_vector") or []
    return _extract_kuid_vector_from_values(vector)


def _extract_kuid_vector_from_csv_row(row: dict):
    values = [row.get(f"f{i}") for i in range(1, 10)]
    return _extract_kuid_vector_from_values(values)


def _kuid_vector_for_calibration(vector: list[float], f2_surrogate_needed: bool):
    out = list(vector)
    if f2_surrogate_needed:
        out[1] = 0.0
    return out


def _kuid_vector_for_encoding(vector: list[float], calibration: dict, f2_surrogate_needed: bool):
    out = list(vector)
    if f2_surrogate_needed:
        bounds = calibration.get("feature_bounds") or {}
        f2_bounds = bounds.get("f2") or {}
        f2_max = _safe_float(f2_bounds.get("max"))
        out[1] = 0.0 if f2_max is None else f2_max
    return out


_KUID_INTENSIVE_FEATURE_INDEX = (
    ("f3", 2),
    ("f4", 3),
    ("f7", 6),
    ("f8", 7),
    ("f9", 8),
)


def _extract_kuid_intensive_feature_map(entry: dict):
    knf_data = entry.get("knf") or {}
    vector = knf_data.get("KNF_vector") or []
    if len(vector) < 9:
        return None

    feature_map = {}
    for feature, idx in _KUID_INTENSIVE_FEATURE_INDEX:
        value = _safe_float(vector[idx])
        if value is None:
            return None
        feature_map[feature] = value
    return feature_map


def _build_knf_result_from_entry(entry: dict):
    knf_data = entry.get("knf") or {}
    vector = _extract_knf_vector(entry)
    if vector is None:
        return None

    snci_val = _safe_float(knf_data.get("SNCI"))
    if snci_val is None:
        snci_val = 0.0

    scdi_val = _safe_float(knf_data.get("SCDI"))
    scdi_var = _safe_float(knf_data.get("SCDI_variance"))
    if scdi_var is None:
        scdi_var = 0.0

    metadata = knf_data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    return knf_vector.KNFResult(
        SNCI=float(snci_val),
        SCDI=scdi_val,
        SCDI_variance=float(scdi_var),
        KNF_vector=[float(v) for v in vector],
        metadata=metadata,
    )


def _build_kuid_section(calibration: dict, encoded: dict) -> dict:
    return {
        "version": calibration.get("kuid_version"),
        "calibration_id": calibration.get("calibration_id"),
        "feature_order": calibration.get("feature_order"),
        "bins_per_feature": calibration.get("bins_per_feature"),
        "display_format": calibration.get("display_format"),
        "cluster_display_format": calibration.get("cluster_display_format"),
        "raw": encoded["raw"],
        "display": encoded["display"],
        "cluster_display": encoded.get("cluster_display", ""),
        "bins": encoded["bins"],
        "normalized": encoded["normalized"],
    }


def _build_kuid_intensive_section(calibration: dict, encoded: dict) -> dict:
    return {
        "version": calibration.get("kuid_intensive_version"),
        "calibration_id": calibration.get("calibration_id"),
        "feature_order": calibration.get("feature_order"),
        "bins_per_feature": calibration.get("bins_per_feature"),
        "display_format": calibration.get("display_format"),
        "cluster_display_format": calibration.get("cluster_display_format"),
        "raw": encoded["raw"],
        "display": encoded["display"],
        "cluster_display": encoded.get("cluster_display", ""),
        "bins": encoded["bins"],
        "normalized": encoded["normalized"],
    }


def _apply_kuid_prefix_fields(record: dict):
    raw = (record.get("KUID") or record.get("KUID_raw") or "").strip()
    record.update(kuid_index.kuid_prefix_fields(raw))


def _write_kuid_index_outputs(rows: list[dict], results_root: str, water: bool = False) -> dict:
    family_json_path = os.path.join(results_root, _final_output_name("kuid_family_stats.json", water))
    family_csv_path = os.path.join(results_root, _final_output_name("kuid_family_stats.csv", water))
    prefix_json_path = os.path.join(results_root, _final_output_name("kuid_prefix_index.json", water))

    family_stats = kuid_index.build_family_stats(rows, code_field="KUID")
    with open(family_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "code_field": "KUID",
                "family_count": len(family_stats),
                "families": family_stats,
            },
            f,
            indent=2,
        )

    family_fieldnames = [
        "kuid",
        "KUID_prefix2",
        "KUID_prefix4",
        "KUID_prefix6",
        "member_count",
        "example_files",
        "mean_SNCI",
        "mean_SCDI",
        "mean_SCDI_variance",
        "mean_SNCI_Norm",
        "mean_SCDI_Norm",
    ] + [f"mean_f{i}" for i in range(1, 10)]
    with open(family_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=family_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for family in family_stats:
            row = dict(family)
            row["example_files"] = "; ".join(family.get("example_files") or [])
            writer.writerow(row)

    prefix_index = kuid_index.build_prefix_index(rows, code_field="KUID")
    with open(prefix_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "code_field": "KUID",
                "index": prefix_index,
            },
            f,
            indent=2,
        )

    return {
        "family_stats_json": family_json_path,
        "family_stats_csv": family_csv_path,
        "prefix_index_json": prefix_json_path,
        "family_count": len(family_stats),
    }


def _write_kuid_reverse_index_outputs(rows: list[dict], results_root: str, water: bool = False) -> dict:
    reverse_json_path = os.path.join(results_root, _final_output_name("kuid_reverse_index.json", water))
    reverse_csv_path = os.path.join(results_root, _final_output_name("kuid_reverse_index.csv", water))

    reverse_index = {}
    missing_rows = 0
    for row in rows:
        code = (row.get("KUID_Cluster") or row.get("KUID") or row.get("KUID_raw") or "").strip()
        if not code:
            missing_rows += 1
            continue
        file_name = (row.get("File") or "").strip()
        source_batch = (row.get("source_batch") or "").strip()
        item = {"file": file_name}
        if source_batch:
            item["source_batch"] = source_batch
        reverse_index.setdefault(code, []).append(item)

    sorted_index = {}
    for code in sorted(reverse_index):
        sorted_index[code] = sorted(
            reverse_index[code],
            key=lambda item: ((item.get("source_batch") or ""), (item.get("file") or "")),
        )

    with open(reverse_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "code_field": "KUID_Cluster",
                "cluster_pattern": "f1f2f3-f4f5-f6f7-f8f9",
                "total_kuid_clusters": len(sorted_index),
                "missing_kuid_rows": missing_rows,
                "index": sorted_index,
            },
            f,
            indent=2,
        )

    with open(reverse_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["KUID_Cluster", "complex_count", "complexes"])
        writer.writeheader()
        for code, items in sorted_index.items():
            labels = []
            for item in items:
                source_batch = (item.get("source_batch") or "").strip()
                file_name = (item.get("file") or "").strip()
                if source_batch and file_name:
                    labels.append(f"{source_batch}::{file_name}")
                elif file_name:
                    labels.append(file_name)
                elif source_batch:
                    labels.append(source_batch)
            writer.writerow(
                {
                    "KUID_Cluster": code,
                    "complex_count": len(items),
                    "complexes": "; ".join(labels),
                }
            )

    return {
        "reverse_index_json": reverse_json_path,
        "reverse_index_csv": reverse_csv_path,
        "total_kuid_clusters": len(sorted_index),
        "missing_kuid_rows": missing_rows,
    }


def _write_kuid_intensive_distribution_outputs(
    rows: list[dict], results_root: str, water: bool = False
) -> dict:
    distribution_csv_path = os.path.join(
        results_root, _final_output_name("kuid_intensive_family_distribution.csv", water)
    )
    distribution_png_path = os.path.join(
        results_root, _final_output_name("kuid_intensive_family_distribution.png", water)
    )

    clusters = {}
    missing_rows = 0
    for row in rows:
        cluster = (row.get("KUID_Intensive_Cluster") or "").strip()
        if not cluster:
            missing_rows += 1
            continue
        clusters[cluster] = clusters.get(cluster, 0) + 1

    size_distribution = {}
    for member_count in clusters.values():
        size_distribution[member_count] = size_distribution.get(member_count, 0) + 1
    ordered_distribution = sorted(size_distribution.items(), key=lambda item: item[0])

    with open(distribution_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["family_size", "number_of_families"])
        writer.writeheader()
        for family_size, number_of_families in ordered_distribution:
            writer.writerow(
                {
                    "family_size": family_size,
                    "number_of_families": number_of_families,
                }
            )

    plot_path = None
    plot_error = None
    if ordered_distribution:
        try:
            import matplotlib.pyplot as plt

            x_values = [size for size, _ in ordered_distribution]
            y_values = [count for _, count in ordered_distribution]
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.bar(x_values, y_values, color="#2f6690")
            ax.set_xlabel("Family Size (members per KUID-Intensive cluster)")
            ax.set_ylabel("Number of Families")
            ax.set_title("KUID-Intensive Family Distribution")
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            fig.tight_layout()
            fig.savefig(distribution_png_path)
            plt.close(fig)
            plot_path = distribution_png_path
        except Exception as e:
            plot_error = str(e)

    return {
        "distribution_csv": distribution_csv_path,
        "distribution_png": plot_path,
        "plot_error": plot_error,
        "total_kuid_intensive_clusters": len(clusters),
        "missing_kuid_intensive_rows": missing_rows,
    }


def _ensure_kuid_csv_field_order(fieldnames: list[str]) -> list[str]:
    base_fields = [
        name
        for name in (fieldnames or [])
        if name
        not in {
            "KUID_raw",
            "KUID",
            "KUID_Cluster",
            "KUID_Intensive_raw",
            "KUID_Intensive",
            "KUID_Intensive_Cluster",
            "KUID_prefix2",
            "KUID_prefix4",
            "KUID_prefix6",
            "SCDI",
        }
    ]
    if not base_fields:
        base_fields = (
            ["File"]
            + [f"f{i}" for i in range(1, 10)]
            + ["SNCI", "SCDI_variance", "SNCI_Norm", "SCDI_Norm"]
        )
    if "f9" in base_fields:
        insert_idx = base_fields.index("f9") + 1
    else:
        insert_idx = len(base_fields)
    return (
        base_fields[:insert_idx]
        + [
            "KUID_raw",
            "KUID",
            "KUID_Cluster",
            "KUID_Intensive_raw",
            "KUID_Intensive",
            "KUID_Intensive_Cluster",
            "KUID_prefix2",
            "KUID_prefix4",
            "KUID_prefix6",
        ]
        + base_fields[insert_idx:]
    )


def _persist_entry_outputs_with_kuid(entry: dict, water: bool = False):
    result = _build_knf_result_from_entry(entry)
    if result is None:
        return

    result_dir = entry.get("result_dir")
    if not result_dir:
        return

    output_txt_path = os.path.join(result_dir, _final_output_name("output.txt", water))
    knf_json_path = os.path.join(result_dir, _final_output_name("knf.json", water))

    knf_vector.write_output_txt(output_txt_path, result)
    knf_vector.write_knf_json(knf_json_path, result)

    stale_summary_txt = os.path.join(result_dir, _final_output_name("summary.txt", water))
    if os.path.exists(stale_summary_txt):
        try:
            os.remove(stale_summary_txt)
        except Exception as e:
            logging.warning("Could not remove stale summary file %s: %s", stale_summary_txt, e)


def _run_kuid_for_single_result(
    file_path: str,
    results_root: str,
    water: bool = False,
) -> dict:
    """Backfills KUID metadata/outputs for a completed single-file run."""
    stem = os.path.splitext(os.path.basename(file_path))[0]
    result_dir = os.path.join(results_root, stem)
    knf_json_path = os.path.join(result_dir, _final_output_name("knf.json", water))
    calibration_path = os.path.join(results_root, _final_output_name("kuid_calibration.json", water))

    if not os.path.exists(knf_json_path):
        return {
            "ran": False,
            "updated": False,
            "reason": f"Missing {_final_output_name('knf.json', water)} output.",
            "knf_json": knf_json_path,
        }

    with open(knf_json_path, "r", encoding="utf-8") as f:
        knf_payload = json.load(f)

    if not isinstance(knf_payload, dict):
        return {
            "ran": True,
            "updated": False,
            "reason": "Invalid knf.json payload structure.",
            "knf_json": knf_json_path,
        }

    entry = {"knf": knf_payload, "result_dir": result_dir}
    vector, f2_surrogate_needed = _extract_kuid_vector_from_entry(entry)
    if vector is None:
        return {
            "ran": True,
            "updated": False,
            "reason": "No valid KNF_vector (f1..f9) available for KUID encoding.",
            "knf_json": knf_json_path,
        }

    calibration = None
    calibration_source = "new"
    if os.path.exists(calibration_path):
        try:
            with open(calibration_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                calibration = existing
                calibration_source = "existing"
        except Exception:
            calibration = None

    if calibration is None:
        calibration = kuid.build_calibration(
            [_kuid_vector_for_calibration(vector, f2_surrogate_needed)]
        )
        calibration_payload = dict(calibration)
        calibration_payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        with open(calibration_path, "w", encoding="utf-8") as f:
            json.dump(calibration_payload, f, indent=2)

    vector_for_encoding = _kuid_vector_for_encoding(
        vector, calibration, f2_surrogate_needed
    )
    try:
        encoded = kuid.encode_knf_vector(vector_for_encoding, calibration)
    except Exception:
        calibration = kuid.build_calibration(
            [_kuid_vector_for_calibration(vector, f2_surrogate_needed)]
        )
        calibration_payload = dict(calibration)
        calibration_payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        with open(calibration_path, "w", encoding="utf-8") as f:
            json.dump(calibration_payload, f, indent=2)
        calibration_source = "new"
        encoded = kuid.encode_knf_vector(
            _kuid_vector_for_encoding(vector, calibration, f2_surrogate_needed),
            calibration,
        )
    kuid_section = _build_kuid_section(calibration, encoded)

    metadata = knf_payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        knf_payload["metadata"] = metadata
    metadata["kuid"] = kuid_section
    knf_payload["kuid"] = kuid_section

    entry["KUID_raw"] = encoded["raw"]
    entry["KUID"] = encoded["raw"]
    entry["KUID_Cluster"] = encoded.get("cluster_display", "")
    _apply_kuid_prefix_fields(entry)

    _persist_entry_outputs_with_kuid(entry, water=water)

    return {
        "ran": True,
        "updated": True,
        "knf_json": knf_json_path,
        "calibration_file": calibration_path,
        "calibration_source": calibration_source,
        "kuid": encoded["raw"],
        "kuid_cluster": encoded.get("cluster_display", ""),
    }


def _run_kuid_only_from_existing_batch(
    directory: str,
    results_root: str,
    water: bool = False,
):
    existing_csv_path = _existing_batch_csv_path(results_root, water=water)
    aggregate_csv_path = _batch_primary_csv_path(results_root, water=water)
    aggregate_json_path = os.path.join(results_root, _final_output_name("batch_knf.json", water))
    calibration_path = os.path.join(results_root, _final_output_name("kuid_calibration.json", water))

    if not os.path.exists(existing_csv_path):
        return {
            "ran": False,
            "reason": (
                f"{_final_output_name(_BATCH_PRIMARY_CSV_NAME, water)} "
                f"or {_final_output_name(_BATCH_LEGACY_CSV_NAMES[0], water)} not found"
            ),
        }

    with open(existing_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        original_fieldnames = list(reader.fieldnames or [])

    parsed_rows = []
    calibration_vectors = []
    encodable_count = 0
    surrogate_rows = 0
    for row in rows:
        vec, f2_surrogate_needed = _extract_kuid_vector_from_csv_row(row)
        if vec is None:
            parsed_rows.append((row, None, False))
            continue
        parsed_rows.append((row, vec, f2_surrogate_needed))
        encodable_count += 1
        if f2_surrogate_needed:
            surrogate_rows += 1
        else:
            calibration_vectors.append(vec)

    if not calibration_vectors and encodable_count:
        calibration_vectors = [
            _kuid_vector_for_calibration(vec, f2_surrogate_needed)
            for _, vec, f2_surrogate_needed in parsed_rows
            if vec is not None
        ]

    if not calibration_vectors:
        return {
            "ran": True,
            "updated_rows": 0,
            "total_rows": len(rows),
            "batch_csv": aggregate_csv_path,
            "batch_json": aggregate_json_path if os.path.exists(aggregate_json_path) else None,
            "calibration_file": None,
            "reason": "No valid KNF rows available for KUID encoding.",
        }

    calibration = kuid.build_calibration(calibration_vectors)
    calibration_payload = dict(calibration)
    calibration_payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(calibration_path, "w", encoding="utf-8") as f:
        json.dump(calibration_payload, f, indent=2)

    updated_rows = []
    encoded_by_file = {}
    for row, vec, f2_surrogate_needed in parsed_rows:
        file_name = (row.get("File") or "").strip()
        if vec is None:
            row["KUID_raw"] = ""
            row["KUID"] = ""
            row["KUID_Cluster"] = ""
            _apply_kuid_prefix_fields(row)
            updated_rows.append(row)
            continue
        vector_for_encoding = _kuid_vector_for_encoding(
            vec, calibration, f2_surrogate_needed
        )
        encoded = kuid.encode_knf_vector(vector_for_encoding, calibration)
        row["KUID_raw"] = encoded["raw"]
        row["KUID"] = encoded["raw"]
        row["KUID_Cluster"] = encoded.get("cluster_display", "")
        _apply_kuid_prefix_fields(row)
        updated_rows.append(row)
        if file_name:
            encoded_by_file[file_name] = encoded

    output_fieldnames = _ensure_kuid_csv_field_order(original_fieldnames)
    with open(aggregate_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in updated_rows:
            writer.writerow(row)

    persist_errors = []
    json_updated = False
    kuid_index_outputs = None
    if os.path.exists(aggregate_json_path):
        try:
            with open(aggregate_json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            records = payload.get("records") or []
            for entry in records:
                input_file_name = (entry.get("input_file_name") or "").strip()
                encoded = encoded_by_file.get(input_file_name)
                if not encoded:
                    vector, f2_surrogate_needed = _extract_kuid_vector_from_entry(entry)
                    if vector is None:
                        continue
                    vector_for_encoding = _kuid_vector_for_encoding(
                        vector, calibration, f2_surrogate_needed
                    )
                    encoded = kuid.encode_knf_vector(vector_for_encoding, calibration)

                entry["KUID_raw"] = encoded["raw"]
                entry["KUID"] = encoded["raw"]
                entry["KUID_Cluster"] = encoded.get("cluster_display", "")
                _apply_kuid_prefix_fields(entry)
                kuid_section = _build_kuid_section(calibration, encoded)
                entry["kuid"] = kuid_section

                knf_data = entry.get("knf") or {}
                if isinstance(knf_data, dict):
                    metadata = knf_data.setdefault("metadata", {})
                    metadata["kuid"] = kuid_section
                    knf_data["kuid"] = kuid_section

                try:
                    _persist_entry_outputs_with_kuid(entry, water=water)
                except Exception as e:
                    persist_errors.append(
                        {
                            "file": input_file_name or entry.get("input_file") or "unknown",
                            "error": str(e),
                        }
                    )

            payload["kuid"] = {
                "enabled": True,
                "kuid_version": calibration.get("kuid_version"),
                "calibration_id": calibration.get("calibration_id"),
                "normalization": calibration.get("normalization"),
                "bins_per_feature": calibration.get("bins_per_feature"),
                "feature_order": calibration.get("feature_order"),
                "display_format": calibration.get("display_format"),
                "cluster_display_format": calibration.get("cluster_display_format"),
                "feature_bounds": calibration.get("feature_bounds"),
                "records_with_kuid": encodable_count,
                "records_without_kuid": len(rows) - encodable_count,
                "invalid_files": [
                    (row.get("File") or "unknown")
                    for row, vec, _ in parsed_rows
                    if vec is None
                ],
                "f2_surrogate_strategy": "f2=max_bound_when_undefined",
                "f2_surrogate_rows": surrogate_rows,
                "calibration_file": calibration_path,
                "persist_errors": persist_errors,
            }
            payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

            kuid_index_outputs = _write_kuid_index_outputs(updated_rows, results_root, water=water)
            payload["kuid"].update(kuid_index_outputs)
            with open(aggregate_json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            json_updated = True
        except Exception as e:
            persist_errors.append({"file": aggregate_json_path, "error": str(e)})

    if kuid_index_outputs is None:
        kuid_index_outputs = _write_kuid_index_outputs(updated_rows, results_root, water=water)

    _cleanup_redundant_batch_aliases(
        results_root=results_root,
        primary_csv_path=aggregate_csv_path,
        primary_json_path=aggregate_json_path if os.path.exists(aggregate_json_path) else None,
        water=water,
    )

    return {
        "ran": True,
        "updated_rows": encodable_count,
        "total_rows": len(rows),
        "batch_csv": aggregate_csv_path,
        "batch_json": aggregate_json_path if json_updated else None,
        "calibration_file": calibration_path,
        "kuid_index_outputs": kuid_index_outputs,
        "persist_errors": persist_errors,
    }


def _compute_kuid_payload(
    enriched_records: list[dict],
    results_root: str,
    water: bool = False,
):
    encodable_rows = []
    calibration_vectors = []
    invalid_files = []
    persist_errors = []
    surrogate_rows = 0

    for entry in enriched_records:
        if entry.get("status") != "success":
            continue
        vector, f2_surrogate_needed = _extract_kuid_vector_from_entry(entry)
        if vector is None:
            invalid_files.append(entry.get("input_file_name") or entry.get("input_file") or "unknown")
            continue
        encodable_rows.append((entry, vector, f2_surrogate_needed))
        if f2_surrogate_needed:
            surrogate_rows += 1
        else:
            calibration_vectors.append(vector)

    if not calibration_vectors and encodable_rows:
        calibration_vectors = [
            _kuid_vector_for_calibration(vector, f2_surrogate_needed)
            for _, vector, f2_surrogate_needed in encodable_rows
        ]

    if not encodable_rows:
        return {
            "enabled": False,
            "error": "No valid successful KNF rows were available for KUID encoding.",
            "records_with_kuid": 0,
            "records_without_kuid": len(invalid_files),
            "invalid_files": invalid_files,
            "calibration_file": None,
        }

    calibration = kuid.build_calibration(calibration_vectors)
    calibration_path = os.path.join(results_root, _final_output_name("kuid_calibration.json", water))
    calibration_payload = dict(calibration)
    calibration_payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(calibration_path, "w", encoding="utf-8") as f:
        json.dump(calibration_payload, f, indent=2)

    for entry, vector, f2_surrogate_needed in encodable_rows:
        vector_for_encoding = _kuid_vector_for_encoding(
            vector, calibration, f2_surrogate_needed
        )
        encoded = kuid.encode_knf_vector(vector_for_encoding, calibration)
        entry["KUID_raw"] = encoded["raw"]
        entry["KUID"] = encoded["raw"]
        entry["KUID_Cluster"] = encoded.get("cluster_display", "")
        _apply_kuid_prefix_fields(entry)

        knf_data = entry.get("knf") or {}
        if isinstance(knf_data, dict):
            kuid_section = _build_kuid_section(calibration, encoded)
            metadata = knf_data.setdefault("metadata", {})
            metadata["kuid"] = kuid_section
            knf_data["kuid"] = kuid_section
            entry["kuid"] = kuid_section

        try:
            _persist_entry_outputs_with_kuid(entry, water=water)
        except Exception as e:
            persist_errors.append(
                {
                    "file": entry.get("input_file_name") or entry.get("input_file") or "unknown",
                    "error": str(e),
                }
            )

    return {
        "enabled": True,
        "kuid_version": calibration.get("kuid_version"),
        "calibration_id": calibration.get("calibration_id"),
        "normalization": calibration.get("normalization"),
        "bins_per_feature": calibration.get("bins_per_feature"),
        "feature_order": calibration.get("feature_order"),
        "display_format": calibration.get("display_format"),
        "cluster_display_format": calibration.get("cluster_display_format"),
        "feature_bounds": calibration.get("feature_bounds"),
        "records_with_kuid": len(encodable_rows),
        "records_without_kuid": len(invalid_files),
        "invalid_files": invalid_files,
        "f2_surrogate_strategy": "f2=max_bound_when_undefined",
        "f2_surrogate_rows": surrogate_rows,
        "calibration_file": calibration_path,
        "persist_errors": persist_errors,
    }


def _compute_kuid_intensive_payload(
    enriched_records: list[dict],
    results_root: str,
    water: bool = False,
):
    valid_rows = []
    invalid_files = []
    persist_errors = []

    for entry in enriched_records:
        if entry.get("status") != "success":
            continue
        feature_map = _extract_kuid_intensive_feature_map(entry)
        if feature_map is None:
            invalid_files.append(entry.get("input_file_name") or entry.get("input_file") or "unknown")
            continue
        valid_rows.append((entry, feature_map))

    if not valid_rows:
        return {
            "enabled": False,
            "error": "No valid successful KNF rows were available for KUID-Intensive encoding.",
            "records_with_kuid_intensive": 0,
            "records_without_kuid_intensive": len(invalid_files),
            "invalid_files": invalid_files,
            "calibration_file": None,
        }

    calibration = kuid_intensive.build_calibration_from_feature_maps(
        [feature_map for _, feature_map in valid_rows]
    )
    calibration_path = os.path.join(
        results_root, _final_output_name("kuid_intensive_calibration.json", water)
    )
    calibration_payload = dict(calibration)
    calibration_payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(calibration_path, "w", encoding="utf-8") as f:
        json.dump(calibration_payload, f, indent=2)

    for entry, feature_map in valid_rows:
        encoded = kuid_intensive.encode_feature_map(feature_map, calibration)
        entry["KUID_Intensive_raw"] = encoded["raw"]
        entry["KUID_Intensive"] = encoded.get("display", "")
        entry["KUID_Intensive_Cluster"] = encoded.get("cluster_display", "")

        knf_data = entry.get("knf") or {}
        if isinstance(knf_data, dict):
            intensive_section = _build_kuid_intensive_section(calibration, encoded)
            metadata = knf_data.setdefault("metadata", {})
            metadata["kuid_intensive"] = intensive_section
            knf_data["kuid_intensive"] = intensive_section
            entry["kuid_intensive"] = intensive_section

        try:
            _persist_entry_outputs_with_kuid(entry, water=water)
        except Exception as e:
            persist_errors.append(
                {
                    "file": entry.get("input_file_name") or entry.get("input_file") or "unknown",
                    "error": str(e),
                }
            )

    return {
        "enabled": True,
        "kuid_intensive_version": calibration.get("kuid_intensive_version"),
        "calibration_id": calibration.get("calibration_id"),
        "normalization": calibration.get("normalization"),
        "bins_per_feature": calibration.get("bins_per_feature"),
        "feature_order": calibration.get("feature_order"),
        "display_format": calibration.get("display_format"),
        "cluster_display_format": calibration.get("cluster_display_format"),
        "feature_bounds": calibration.get("feature_bounds"),
        "records_with_kuid_intensive": len(valid_rows),
        "records_without_kuid_intensive": len(invalid_files),
        "invalid_files": invalid_files,
        "calibration_file": calibration_path,
        "persist_errors": persist_errors,
    }


def _classify_quadrant(x: float, y: float, mx: float, my: float) -> str:
    if x >= mx and y >= my:
        return "Q1"
    if x < mx and y >= my:
        return "Q2"
    if x < mx and y < my:
        return "Q3"
    return "Q4"


def _compute_norm_and_quadrants(
    enriched_records: list[dict],
    results_root: str,
    water: bool = False,
    interactive_plot: bool = False,
):
    normalized_rows = []
    for entry in enriched_records:
        if entry.get("status") != "success":
            continue
        knf_data = entry.get("knf") or {}
        snci_val = _safe_float(knf_data.get("SNCI"))
        scdi_val = _safe_float(knf_data.get("SCDI"))
        scdi_var = _safe_float(knf_data.get("SCDI_variance"))
        normalized_rows.append(
            {
                "entry": entry,
                "file_name": entry.get("input_file_name", ""),
                "snci": snci_val,
                "scdi": scdi_val,
                "scdi_variance": scdi_var,
                "snci_norm": None,
                "scdi_norm": None,
            }
        )

    if normalized_rows:
        snci_values = [row["snci"] for row in normalized_rows]
        snci_norm = _normalize_minmax(snci_values, invert=False)
        for row, norm_val in zip(normalized_rows, snci_norm):
            row["snci_norm"] = norm_val

        has_complete_scdi = all(row["scdi"] is not None for row in normalized_rows)
        if has_complete_scdi:
            for row in normalized_rows:
                row["scdi_norm"] = max(0.0, min(1.0, float(row["scdi"])))
            scdi_norm_source = "SCDI"
        else:
            variance_values = [row["scdi_variance"] for row in normalized_rows]
            scdi_norm = _normalize_minmax(variance_values, invert=True)
            for row, norm_val in zip(normalized_rows, scdi_norm):
                row["scdi_norm"] = norm_val
            scdi_norm_source = "SCDI_variance_inverse_minmax"

        for row in normalized_rows:
            row["entry"]["SNCI_Norm"] = row["snci_norm"]
            row["entry"]["SCDI_Norm"] = row["scdi_norm"]
    else:
        scdi_norm_source = None

    valid_plot_rows = [
        row
        for row in normalized_rows
        if row["snci_norm"] is not None and row["scdi_norm"] is not None
    ]
    if not valid_plot_rows:
        return {
            "SNCI_Norm_source": "minmax",
            "SCDI_Norm_source": scdi_norm_source,
            "median_SNCI_Norm": None,
            "median_SCDI_Norm": None,
            "quadrants": {},
            "quadrant_json": None,
            "quadrant_plot_png": None,
            "plot_error": "No successful normalized rows available.",
        }

    snci_norm_vals = [row["snci_norm"] for row in valid_plot_rows]
    scdi_norm_vals = [row["scdi_norm"] for row in valid_plot_rows]
    median_x = float(statistics.median(snci_norm_vals))
    median_y = float(statistics.median(scdi_norm_vals))

    quadrants = {
        "Q1": {"count": 0, "files": []},
        "Q2": {"count": 0, "files": []},
        "Q3": {"count": 0, "files": []},
        "Q4": {"count": 0, "files": []},
    }
    for row in valid_plot_rows:
        q = _classify_quadrant(row["snci_norm"], row["scdi_norm"], median_x, median_y)
        quadrants[q]["count"] += 1
        quadrants[q]["files"].append(row["file_name"])
        row["entry"]["quadrant"] = q

    quadrant_json_path = os.path.join(
        results_root,
        _final_output_name("snci_scdi_quadrants.json", water),
    )
    with open(quadrant_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "median_SNCI_Norm": median_x,
                "median_SCDI_Norm": median_y,
                "quadrants": quadrants,
            },
            f,
            indent=2,
        )

    plot_png_path = os.path.join(
        results_root,
        _final_output_name("snci_scdi_quadrants.png", water),
    )
    plot_error = None
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        ax.scatter(snci_norm_vals, scdi_norm_vals, s=45, alpha=0.75)
        ax.axvline(median_x, color="crimson", linestyle="--", linewidth=1.8, label=f"Median X = {median_x:.4f}")
        ax.axhline(median_y, color="darkgreen", linestyle="--", linewidth=1.8, label=f"Median Y = {median_y:.4f}")
        ax.set_xlabel("SNCI_Norm")
        ax.set_ylabel("SCDI_Norm")
        ax.set_title("SCDI_Norm vs SNCI_Norm with Median Quadrants")
        ax.grid(True, linestyle="--", alpha=0.4)

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_left = x_min + 0.08 * (x_max - x_min)
        x_right = x_min + 0.58 * (x_max - x_min)
        y_top = y_min + 0.94 * (y_max - y_min)
        y_bottom = y_min + 0.40 * (y_max - y_min)

        ax.text(x_right, y_top, f"Q1\n(n={quadrants['Q1']['count']})", fontsize=13, fontweight="bold")
        ax.text(x_left, y_top, f"Q2\n(n={quadrants['Q2']['count']})", fontsize=13, fontweight="bold")
        ax.text(x_left, y_bottom, f"Q3\n(n={quadrants['Q3']['count']})", fontsize=13, fontweight="bold")
        ax.text(x_right, y_bottom, f"Q4\n(n={quadrants['Q4']['count']})", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(plot_png_path)
        if interactive_plot:
            plt.show()
        plt.close(fig)
    except Exception as e:
        plot_error = str(e)
        plot_png_path = None

    return {
        "SNCI_Norm_source": "minmax",
        "SCDI_Norm_source": scdi_norm_source,
        "median_SNCI_Norm": median_x,
        "median_SCDI_Norm": median_y,
        "quadrants": quadrants,
        "quadrant_json": quadrant_json_path,
        "quadrant_plot_png": plot_png_path,
        "plot_error": plot_error,
    }


def _poll_stop_key(enable_stop_key: bool) -> bool:
    if not enable_stop_key:
        return False
    try:
        if os.name == "nt":
            import msvcrt
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch and ch.lower() == STOP_KEY:
                    return True
            return False
        import select
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if ready:
            ch = sys.stdin.read(1)
            return bool(ch and ch.lower() == STOP_KEY)
        return False
    except Exception:
        return False


def run_single_file(file_path: str, args):
    results_root = resolve_results_root(file_path, args.output_dir)
    console = Console()
    logical = psutil.cpu_count(logical=True) or (os.cpu_count() or 1)
    physical = psutil.cpu_count(logical=False) or max(1, logical // 2)

    peak_cpu = 0.0
    peak_ram = 0.0
    t0 = time.perf_counter()
    psutil.cpu_percent(interval=None)

    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
    task_id = progress.add_task("Single Job", total=1)

    def render(status_text: str, status_style: str):
        avg_cpu = psutil.cpu_percent(interval=None)
        ram_mb = max(_active_tool_ram_mb(), _self_ram_mb())
        nonlocal peak_cpu, peak_ram
        if avg_cpu >= 0:
            peak_cpu = max(peak_cpu, avg_cpu)
        peak_ram = max(peak_ram, ram_mb)

        header = Table.grid(padding=(0, 2))
        header.add_column(style="bold")
        header.add_column()
        header.add_row("KNF-Core", "v1.0.5")
        header.add_row("Detected", f"{physical}C / {logical}T")
        header.add_row("Mode", "single")
        header.add_row("File", _display_name(file_path))
        header.add_row("Output", results_root)
        header.add_row("Avg CPU", f"{avg_cpu:.1f}%")
        header.add_row("RAM", f"{ram_mb:.1f} MB")
        header.add_row("Status", f"[{status_style}]{status_text}[/{status_style}]")

        return Group(
            Panel(header, title="KNF-Core Single Run", border_style="cyan"),
            progress,
        )

    success = False
    error = None
    elapsed = 0.0
    kuid_summary = None
    with Live(render("running", "yellow"), console=console, refresh_per_second=5, transient=False) as live:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(process_file, file_path, args, results_root)
            while not future.done():
                live.update(render("running", "yellow"))
                time.sleep(0.4)

            success, error, elapsed = future.result()
            progress.advance(task_id, 1)
            live.update(render("completed" if success else "failed", "green" if success else "red"))

    if success:
        try:
            kuid_summary = _run_kuid_for_single_result(
                file_path=file_path,
                results_root=results_root,
                water=bool(getattr(args, "water", False)),
            )
        except Exception as e:
            kuid_summary = {"ran": True, "updated": False, "error": str(e)}

    total_time = elapsed if elapsed > 0 else (time.perf_counter() - t0)
    throughput = ((1 / total_time) * 3600) if (success and total_time > 0) else 0.0

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Total files", "1")
    summary.add_row("Success", "1" if success else "0")
    summary.add_row("Failed", "0" if success else "1")
    summary.add_row("Total time", _fmt_elapsed(total_time))
    summary.add_row("Molecule time", f"{elapsed:.1f}s" if elapsed > 0 else "n/a")
    summary.add_row("Throughput", f"{throughput:.1f} jobs/hour" if success else "n/a")
    summary.add_row("Peak CPU", f"{peak_cpu:.1f}%")
    summary.add_row("Peak RAM", f"{peak_ram:.1f} MB")
    if success and isinstance(kuid_summary, dict):
        if kuid_summary.get("updated"):
            summary.add_row("KUID", str(kuid_summary.get("kuid", "")))
            calibration_file = str(kuid_summary.get("calibration_file", ""))
            calibration_source = str(kuid_summary.get("calibration_source", "")).strip()
            if calibration_source:
                summary.add_row("KUID Calibration", f"{calibration_file} ({calibration_source})")
            else:
                summary.add_row("KUID Calibration", calibration_file)
        else:
            kuid_issue = kuid_summary.get("error") or kuid_summary.get("reason")
            if kuid_issue:
                summary.add_row("KUID", f"not updated ({kuid_issue})")
    console.print(Panel(summary, title="Run Completed", border_style="green" if success else "red"))

    if not success:
        fail_table = Table(title="Failure", expand=True)
        fail_table.add_column("File")
        fail_table.add_column("Error")
        fail_table.add_row(os.path.basename(file_path), str(error))
        console.print(fail_table)


def _discover_input_files(directory: str, valid_exts: set[str] = None) -> list[str]:
    extensions = valid_exts or VALID_INPUT_EXTS
    files = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if not os.path.isfile(full_path):
            continue
        ext = utils.normalized_extension(entry)
        if ext in extensions:
            files.append(full_path)
    files.sort()
    return files


def resolve_results_root(input_path: str, output_dir: str = None) -> str:
    """Resolves the top-level Results directory."""
    if output_dir:
        return os.path.abspath(output_dir)

    if os.path.isdir(input_path):
        return os.path.join(os.path.abspath(input_path), "Results")

    return os.path.join(os.path.dirname(os.path.abspath(input_path)), "Results")


def _normalize_batch_file_name(value) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().strip('"').strip("'")
    if not cleaned:
        return ""
    return os.path.normcase(os.path.basename(cleaned))


def _record_file_name(record: dict) -> str:
    if not isinstance(record, dict):
        return ""
    return _normalize_batch_file_name(
        record.get("input_file_name") or record.get("input_file") or ""
    )


def _dedupe_batch_records(records: list[dict]) -> list[dict]:
    deduped = {}
    order = []
    for record in records:
        if not isinstance(record, dict):
            continue
        input_file = str(record.get("input_file") or "").strip()
        input_file_name = str(record.get("input_file_name") or "").strip()
        if not input_file and input_file_name:
            input_file = os.path.abspath(input_file_name)
        if not input_file:
            continue

        key = _normalize_batch_file_name(input_file_name or input_file)
        if not key:
            key = os.path.normcase(os.path.abspath(input_file))

        elapsed = _safe_float(record.get("elapsed_seconds"))
        normalized = {
            "input_file": os.path.abspath(input_file),
            "status": str(record.get("status") or "failed"),
            "elapsed_seconds": float(elapsed) if elapsed is not None else 0.0,
            "error": record.get("error"),
        }
        if key not in deduped:
            order.append(key)
        deduped[key] = normalized

    return [deduped[key] for key in order]


def _load_existing_batch_records(
    directory: str,
    results_root: str,
    water: bool = False,
) -> dict:
    aggregate_csv_path = _existing_batch_csv_path(results_root, water=water)
    aggregate_json_path = os.path.join(results_root, _final_output_name("batch_knf.json", water))

    warnings = []
    records = []
    processed_names = set()
    source = None

    if os.path.exists(aggregate_json_path):
        try:
            with open(aggregate_json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            for entry in (payload.get("records") or []):
                if not isinstance(entry, dict):
                    continue
                input_file = str(entry.get("input_file") or "").strip()
                input_file_name = str(entry.get("input_file_name") or "").strip()
                if not input_file and input_file_name:
                    input_file = os.path.abspath(
                        os.path.join(directory, os.path.basename(input_file_name))
                    )
                if not input_file:
                    continue
                records.append(
                    {
                        "input_file": input_file,
                        "status": entry.get("status"),
                        "elapsed_seconds": entry.get("elapsed_seconds"),
                        "error": entry.get("error"),
                    }
                )
                normalized_name = _normalize_batch_file_name(input_file_name or input_file)
                if normalized_name:
                    processed_names.add(normalized_name)

            if records:
                source = "json"
        except Exception as e:
            warnings.append(
                f"Could not read existing {_final_output_name('batch_knf.json', water)}: {e}"
            )

    if not records and os.path.exists(aggregate_csv_path):
        try:
            with open(aggregate_csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    file_name = (row.get("File") or "").strip()
                    normalized_name = _normalize_batch_file_name(file_name)
                    if not normalized_name:
                        continue
                    processed_names.add(normalized_name)
                    records.append(
                        {
                            "input_file": os.path.abspath(
                                os.path.join(directory, os.path.basename(file_name))
                            ),
                            "status": "success",
                            "elapsed_seconds": 0.0,
                            "error": None,
                        }
                    )
            if records:
                source = "csv"
        except Exception as e:
            warnings.append(
                f"Could not read existing {os.path.basename(aggregate_csv_path)}: {e}"
            )

    records = _dedupe_batch_records(records)
    if not processed_names:
        for record in records:
            normalized_name = _record_file_name(record)
            if normalized_name:
                processed_names.add(normalized_name)

    return {
        "records": records,
        "processed_names": processed_names,
        "source": source,
        "csv_path": aggregate_csv_path if source == "csv" else None,
        "warnings": warnings,
    }


def _sum_elapsed_seconds(records: list[dict]) -> float:
    total = 0.0
    for record in records:
        if not isinstance(record, dict):
            continue
        elapsed = _safe_float(record.get("elapsed_seconds"))
        if elapsed is None:
            continue
        total += max(0.0, float(elapsed))
    return total

def _resolve_requested_batch_count(
    requested_batches: int,
    total_files: int,
    workers_hint: int = None,
) -> int:
    if total_files <= 0:
        return 0
    if requested_batches is None:
        return 1

    if int(requested_batches) > 0:
        return min(total_files, int(requested_batches))

    # Auto mode (--batches without an explicit number)
    if workers_hint and int(workers_hint) > 0:
        base = int(workers_hint)
    else:
        base = psutil.cpu_count(logical=False) or (os.cpu_count() or 1)
    return min(total_files, max(1, int(base)))


def _split_evenly(items: list[str], num_parts: int) -> list[list[str]]:
    if num_parts <= 0:
        return []
    if not items:
        return [[] for _ in range(num_parts)]
    q, r = divmod(len(items), num_parts)
    out = []
    start = 0
    for idx in range(num_parts):
        size = q + (1 if idx < r else 0)
        out.append(items[start:start + size])
        start += size
    return out


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

    quadrant_payload = _compute_norm_and_quadrants(
        enriched_records=combined_records,
        results_root=output_root,
        water=water,
        interactive_plot=False,
    )
    kuid_payload = _compute_kuid_payload(
        enriched_records=combined_records,
        results_root=output_root,
        water=water,
    )
    kuid_intensive_payload = _compute_kuid_intensive_payload(
        enriched_records=combined_records,
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
        "records": combined_records,
        "knf_results": combined_knf_results,
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
        for entry in combined_records:
            knf_data = entry.get("knf") or {}
            knf_vector = knf_data.get("KNF_vector") or []
            metadata = knf_data.get("metadata") if isinstance(knf_data, dict) else None
            row = {
                "source_batch": entry.get("source_batch", ""),
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
            logging.warning("Could not write KUID unified calibration alias %s: %s", kuid_calibration_unified, e)
    if os.path.exists(kuid_int_calibration):
        try:
            shutil.copyfile(kuid_int_calibration, kuid_int_calibration_unified)
        except Exception as e:
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


def run_universal_kuid(directory: str, args):
    water_mode = bool(getattr(args, "water", False))
    output_base = resolve_results_root(directory, args.output_dir)
    combined_output_root = os.path.join(output_base, "Combined Results")
    source_specs = _discover_universal_batch_sources(directory, water=water_mode)

    if not source_specs:
        print(
            f"No {_final_output_name('batch_knf.json', water_mode)} or "
            f"{_final_output_name(_BATCH_PRIMARY_CSV_NAME, water_mode)} "
            f"(legacy {_final_output_name(_BATCH_LEGACY_CSV_NAMES[0], water_mode)} also supported) "
            f"files found under {directory}."
        )
        return

    result = _combine_batch_sources(
        source_directory=directory,
        source_specs=source_specs,
        output_root=combined_output_root,
        water=water_mode,
        mode="universal_kuid_recompute",
    )

    print(f"Universal KUID sources: {len(source_specs)}")
    print(f"Combined results root: {result['output_root']}")
    print(f"Combined Batch JSON:  {result['batch_json']}")
    print(f"Combined Batch CSV:   {result['batch_csv']}")


def run_batch_directory_batched(directory: str, args):
    files = _discover_input_files(directory)
    if not files:
        print(f"No molecular files found in {directory}.")
        return

    batch_count = _resolve_requested_batch_count(
        requested_batches=getattr(args, "batches", None),
        total_files=len(files),
        workers_hint=getattr(args, "workers", None),
    )
    partitions = [part for part in _split_evenly(files, batch_count) if part]
    if not partitions:
        print(f"No molecular files found in {directory}.")
        return

    water_mode = bool(getattr(args, "water", False))
    output_base = resolve_results_root(directory, args.output_dir)
    batches_output_root = os.path.join(output_base, "Batches")
    combined_output_root = os.path.join(output_base, "Combined Results")
    os.makedirs(batches_output_root, exist_ok=True)

    print(f"Batching enabled: {len(files)} files across {len(partitions)} batch(es).")

    source_specs = []
    for idx, batch_files in enumerate(partitions, start=1):
        source_batch = f"batch_{idx:02d}"
        batch_results_root = os.path.join(batches_output_root, source_batch)
        os.makedirs(batch_results_root, exist_ok=True)

        print(f"\n[{source_batch}] processing {len(batch_files)} file(s) -> {batch_results_root}")
        run_batch_directory(
            directory=directory,
            args=args,
            file_paths=batch_files,
            results_root_override=batch_results_root,
        )

        batch_json = os.path.join(batch_results_root, _final_output_name("batch_knf.json", water_mode))
        batch_csv = _existing_batch_csv_path(batch_results_root, water=water_mode)
        if os.path.exists(batch_json):
            source_specs.append({"source_batch": source_batch, "path": batch_json, "type": "json"})
        elif os.path.exists(batch_csv):
            source_specs.append({"source_batch": source_batch, "path": batch_csv, "type": "csv"})
        else:
            logging.warning(
                "Batch source outputs not found for %s (expected %s or %s/%s).",
                source_batch,
                batch_json,
                os.path.join(batch_results_root, _final_output_name(_BATCH_PRIMARY_CSV_NAME, water_mode)),
                os.path.join(batch_results_root, _final_output_name(_BATCH_LEGACY_CSV_NAMES[0], water_mode)),
            )

    if not source_specs:
        print("No batch outputs were available to combine.")
        return

    result = _combine_batch_sources(
        source_directory=directory,
        source_specs=source_specs,
        output_root=combined_output_root,
        water=water_mode,
        mode="combined_from_internal_batches",
    )

    print(f"\nCombined results root: {result['output_root']}")
    print(f"Combined Batch JSON:  {result['batch_json']}")
    print(f"Combined Batch CSV:   {result['batch_csv']}")


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

        if record["status"] == "success" and os.path.exists(knf_path):
            try:
                with open(knf_path, "r", encoding="utf-8") as f:
                    knf_data = json.load(f)
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
            except Exception as e:
                entry["status"] = "failed"
                entry["error"] = f"Failed to read {_final_output_name('knf.json', water)}: {e}"
                failure_count += 1
        elif record["status"] == "success":
            entry["status"] = "failed"
            entry["error"] = f"Missing {_final_output_name('knf.json', water)} output."
            failure_count += 1
        elif record["status"] == "stopped":
            stopped_count += 1
        else:
            failure_count += 1

        enriched_records.append(entry)

    quadrant_payload = _compute_norm_and_quadrants(
        enriched_records=enriched_records,
        results_root=results_root,
        water=water,
        interactive_plot=interactive_quadrant_plot,
    )
    kuid_payload = _compute_kuid_payload(
        enriched_records=enriched_records,
        results_root=results_root,
        water=water,
    )
    kuid_intensive_payload = _compute_kuid_intensive_payload(
        enriched_records=enriched_records,
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
        "records": enriched_records,
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
        for entry in enriched_records:
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

def run_batch_directory(
    directory: str,
    args,
    file_paths: list[str] = None,
    results_root_override: str = None,
):
    """Runs the pipeline for all valid files in a directory using a queue."""
    results_root = results_root_override or resolve_results_root(directory, args.output_dir)
    water_mode = bool(getattr(args, "water", False))
    aggregate_csv_path = _batch_primary_csv_path(results_root, water=water_mode)
    aggregate_json_path = os.path.join(results_root, _final_output_name("batch_knf.json", water_mode))
    existing_resume_csv_path = _existing_batch_csv_path(results_root, water=water_mode)

    if file_paths is None:
        files = _discover_input_files(directory)
    else:
        files = []
        for file_path in file_paths:
            full_path = os.path.abspath(file_path)
            if not os.path.isfile(full_path):
                continue
            ext = utils.normalized_extension(os.path.basename(full_path))
            if ext in VALID_INPUT_EXTS:
                files.append(full_path)
        files.sort()
    
    if not files:
        print(f"No molecular files found in {directory}.")
        return

    existing_batch_records = []
    skipped_existing = 0
    has_resume_outputs = os.path.exists(aggregate_json_path) or os.path.exists(existing_resume_csv_path)
    if has_resume_outputs and not bool(getattr(args, "force", False)):
        resume_state = _load_existing_batch_records(
            directory=directory,
            results_root=results_root,
            water=water_mode,
        )
        existing_batch_records = resume_state.get("records") or []
        processed_names = resume_state.get("processed_names") or set()
        for warning in (resume_state.get("warnings") or []):
            logging.warning(warning)

        if processed_names:
            pending_files = [
                file_path
                for file_path in files
                if _normalize_batch_file_name(os.path.basename(file_path)) not in processed_names
            ]
            skipped_existing = len(files) - len(pending_files)
            files = pending_files
            if skipped_existing:
                source = resume_state.get("source")
                if source == "json":
                    source_name = _final_output_name("batch_knf.json", water_mode)
                else:
                    source_name = os.path.basename(
                        resume_state.get("csv_path") or existing_resume_csv_path
                    )
                print(
                    f"Resume mode: skipping {skipped_existing} file(s) already listed in {source_name}."
                )

        if not files:
            if existing_batch_records:
                print(
                    f"No new molecular files found in {directory}; refreshing aggregate outputs from existing batch records."
                )
                refresh_mode = args.processing.lower()
                if refresh_mode == "auto":
                    refresh_mode = "multi" if len(existing_batch_records) > 1 else "single"
                refresh_workers = max(1, int(getattr(args, "workers", 1) or 1))
                aggregate_total_time = _sum_elapsed_seconds(existing_batch_records)
                (
                    refreshed_json,
                    refreshed_csv,
                    _,
                    refreshed_delta_json,
                    refreshed_delta_txt,
                ) = write_batch_aggregate_json(
                    directory=directory,
                    results_root=results_root,
                    records=existing_batch_records,
                    mode=refresh_mode,
                    workers=refresh_workers,
                    total_time=aggregate_total_time,
                    water=water_mode,
                    interactive_quadrant_plot=False,
                )
                print(f"Batch JSON: {refreshed_json}")
                print(f"Batch CSV:  {refreshed_csv}")
                if refreshed_delta_json:
                    print(f"Batch Delta JSON: {refreshed_delta_json}")
                if refreshed_delta_txt:
                    print(f"Batch Delta TXT:  {refreshed_delta_txt}")
            else:
                print(
                    f"No new molecular files found in {directory}; all detected files are already listed in {aggregate_csv_path}."
                )
            return

    mode = args.processing.lower()
    if mode == "auto":
        mode = "multi" if len(files) > 1 else "single"
    use_gpu_overlap = (
        mode == "multi"
        and len(files) > 1
        and (args.nci_backend or "").strip().lower() == "torch"
        and (args.nci_device or "").strip().lower() == "cuda"
    )
    workers = 1
    failures = []
    batch_records = []
    succeeded = 0
    stopped_count = 0

    if mode == 'multi' and len(files) > 1:
        if args.workers is None:
            cfg = autoconfig.resolve_multi_config(
                n_jobs=len(files),
                ram_per_job_mb=args.ram_per_job,
                project_root=os.getcwd(),
                force_refresh=args.refresh_autoconfig
            )
            workers = cfg.workers
            autoconfig.apply_env_inplace(cfg)
        else:
            workers = max(1, args.workers)
            logical_threads = os.cpu_count() or 1
            omp = max(1, logical_threads // workers)
            os.environ.update(
                {
                    "OMP_NUM_THREADS": str(omp),
                    "MKL_NUM_THREADS": str(omp),
                    "OPENBLAS_NUM_THREADS": str(omp),
                    "VECLIB_MAXIMUM_THREADS": str(omp),
                    "NUMEXPR_NUM_THREADS": str(omp),
                }
            )
    else:
        workers = 1
        cfg = None

    logical = psutil.cpu_count(logical=True) or (os.cpu_count() or 1)
    physical = psutil.cpu_count(logical=False) or max(1, logical // 2)
    console = Console()
    enable_stop_key = bool(getattr(args, "enable_stop_key", False))
    interactive_quadrant_plot = bool(getattr(args, "interactive_quadrant_plot", False))
    stop_requested = False
    stop_notice_emitted = False

    if enable_stop_key:
        console.print(
            f"[yellow]Stop control enabled: press '{STOP_KEY}' to stop new jobs and finish safely.[/yellow]"
        )

    completed_rows = []
    recent_job_durations = []
    total = len(files)
    completed = 0
    peak_cpu = 0.0
    peak_ram = 0.0
    t0 = time.perf_counter()
    psutil.cpu_percent(interval=None)

    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )
    task_id = progress.add_task("Overall", total=total)

    def maybe_request_stop() -> bool:
        nonlocal stop_requested, stop_notice_emitted
        if not stop_requested and _poll_stop_key(enable_stop_key):
            stop_requested = True
        if stop_requested and not stop_notice_emitted:
            console.print("[yellow]Stop requested. Completing running tasks and finalizing partial outputs...[/yellow]")
            stop_notice_emitted = True
        return stop_requested

    def add_success(file_path: str, elapsed: float):
        nonlocal completed, succeeded
        completed += 1
        succeeded += 1
        if elapsed > 0:
            recent_job_durations.append(float(elapsed))
            if len(recent_job_durations) > 20:
                del recent_job_durations[:-20]
        progress.advance(task_id, 1)
        batch_records.append(
            {
                "input_file": file_path,
                "status": "success",
                "elapsed_seconds": elapsed,
                "error": None,
            }
        )
        completed_rows.append((_display_name(file_path), f"{elapsed:.1f}s", "[green]OK[/green]"))

    def add_failure(file_path: str, error: str, elapsed: float):
        nonlocal completed
        completed += 1
        if elapsed > 0:
            recent_job_durations.append(float(elapsed))
            if len(recent_job_durations) > 20:
                del recent_job_durations[:-20]
        progress.advance(task_id, 1)
        failures.append((file_path, error))
        batch_records.append(
            {
                "input_file": file_path,
                "status": "failed",
                "elapsed_seconds": elapsed,
                "error": str(error),
            }
        )
        completed_rows.append((_display_name(file_path), f"{elapsed:.1f}s", "[red]FAIL[/red]"))

    def add_stopped(file_path: str, elapsed: float = 0.0, reason: str = "Stopped by user before processing."):
        nonlocal completed, stopped_count
        completed += 1
        stopped_count += 1
        progress.advance(task_id, 1)
        batch_records.append(
            {
                "input_file": file_path,
                "status": "stopped",
                "elapsed_seconds": elapsed,
                "error": reason,
            }
        )
        elapsed_text = f"{elapsed:.1f}s" if elapsed > 0 else "-"
        completed_rows.append((_display_name(file_path), elapsed_text, "[yellow]STOP[/yellow]"))

    def render(active_workers: int):
        elapsed = time.perf_counter() - t0
        avg_cpu = psutil.cpu_percent(interval=None)
        ram_mb = _active_tool_ram_mb()
        nonlocal peak_cpu, peak_ram
        peak_cpu = max(peak_cpu, avg_cpu)
        peak_ram = max(peak_ram, ram_mb)
        rate = completed / max(elapsed, 1e-6)
        eta = (total - completed) / max(rate, 1e-6) if completed else 0.0
        processed_count = max(0, completed - stopped_count)
        compounds_per_hour = (processed_count / max(elapsed, 1e-6)) * 3600 if processed_count else 0.0
        avg_sec_per_compound = (elapsed / processed_count) if processed_count else None
        projected_total_runtime = (elapsed + eta) if completed else None

        recent_window = recent_job_durations[-8:]
        recent_avg_sec = (sum(recent_window) / len(recent_window)) if recent_window else None
        throughput_trend = "warming up"
        if (
            avg_sec_per_compound
            and recent_avg_sec
            and processed_count >= 3
            and len(recent_window) >= 3
        ):
            ratio = recent_avg_sec / max(avg_sec_per_compound, 1e-6)
            if ratio <= 0.95:
                throughput_trend = "faster than average"
            elif ratio >= 1.05:
                throughput_trend = "slower than average"
            else:
                throughput_trend = "near average"

        header = Table.grid(padding=(0, 2))
        header.add_column(style="bold")
        header.add_column()
        header.add_row("KNF-Core", "v1.0.5")
        header.add_row("Detected", f"{physical}C / {logical}T")
        if mode == "multi":
            if use_gpu_overlap:
                header.add_row("Mode", f"multi (cpu->gpu overlap: {workers} CPU + 1 GPU)")
            else:
                mode_text = "auto" if args.workers is None else "manual"
                header.add_row("Mode", f"multi ({mode_text} -> {workers} workers)")
        else:
            header.add_row("Mode", "single")
        header.add_row("Files", str(total))
        header.add_row("Completed", f"{completed}/{total}")
        header.add_row("Output", results_root)
        header.add_row("Active Workers", str(active_workers))
        header.add_row("Batch Runtime", _fmt_elapsed(elapsed))
        if projected_total_runtime is not None:
            header.add_row("Projected Total", _fmt_elapsed(projected_total_runtime))
        header.add_row("Avg CPU", f"{avg_cpu:.1f}%")
        header.add_row("RAM", f"{ram_mb:.1f} MB")
        if processed_count:
            header.add_row("Throughput", f"{compounds_per_hour:.1f} compounds/hour")
            header.add_row("Avg / Compound", f"{avg_sec_per_compound:.1f}s")
        else:
            header.add_row("Throughput", "n/a")
            header.add_row("Avg / Compound", "n/a")
        if recent_avg_sec is not None:
            header.add_row("Recent (8) / Compound", f"{recent_avg_sec:.1f}s ({throughput_trend})")
        header.add_row("ETA", _fmt_elapsed(eta))

        jobs = Table(title="Completed Jobs", expand=True)
        jobs.add_column("File", overflow="fold")
        jobs.add_column("Time", justify="right", width=8)
        jobs.add_column("Status", width=8)
        for row in completed_rows[-15:]:
            jobs.add_row(*row)
        if not completed_rows:
            jobs.add_row("-", "-", "running")

        return Group(
            Panel(header, title="KNF-Core Batch Summary", border_style="cyan"),
            progress,
            jobs,
        )

    if mode == 'single' or len(files) == 1:
        with Live(render(active_workers=1), console=console, refresh_per_second=5, transient=False) as live:
            queue = Queue()
            for path in files:
                queue.put(path)
            while not queue.empty():
                maybe_request_stop()
                if stop_requested:
                    while not queue.empty():
                        pending_file = queue.get()
                        add_stopped(pending_file)
                        queue.task_done()
                    live.update(render(active_workers=0))
                    break

                file_path = queue.get()
                success, error, elapsed = process_file(file_path, args, output_root=results_root)
                if success:
                    add_success(file_path, elapsed)
                else:
                    add_failure(file_path, error, elapsed)
                live.update(render(active_workers=0))
                queue.task_done()
    elif use_gpu_overlap:
        with Live(render(active_workers=workers + 1), console=console, refresh_per_second=5, transient=False) as live:
            with ThreadPoolExecutor(max_workers=workers) as cpu_executor, ThreadPoolExecutor(max_workers=1) as gpu_executor:
                pre_futures = {
                    cpu_executor.submit(process_file_pre_nci, file_path, args, results_root): file_path
                    for file_path in files
                }
                post_futures = {}
                pre_cancel_applied = False

                while pre_futures or post_futures:
                    maybe_request_stop()
                    if stop_requested and not pre_cancel_applied:
                        for future, file_path in list(pre_futures.items()):
                            if future.cancel():
                                pre_futures.pop(future, None)
                                add_stopped(file_path)
                        pre_cancel_applied = True

                    done_pre = []
                    if pre_futures:
                        try:
                            for future in as_completed(pre_futures, timeout=0.2):
                                done_pre.append(future)
                                if len(done_pre) >= workers:
                                    break
                        except TimeoutError:
                            pass

                    for future in done_pre:
                        file_path = pre_futures.pop(future)
                        if future.cancelled():
                            add_stopped(file_path)
                            continue
                        try:
                            success, error, pre_elapsed, pipeline, context = future.result()
                        except CancelledError:
                            add_stopped(file_path)
                            continue
                        except Exception as e:
                            success, error, pre_elapsed, pipeline, context = False, str(e), 0.0, None, None

                        if success and not stop_requested:
                            post_future = gpu_executor.submit(process_file_post_nci, pipeline, context, file_path)
                            post_futures[post_future] = (file_path, pre_elapsed)
                        elif success:
                            add_stopped(
                                file_path,
                                elapsed=pre_elapsed,
                                reason="Stopped by user after pre-NCI stage.",
                            )
                        else:
                            add_failure(file_path, error, pre_elapsed)

                    done_post = []
                    if post_futures:
                        try:
                            for future in as_completed(post_futures, timeout=0.2):
                                done_post.append(future)
                                break
                        except TimeoutError:
                            pass

                    for future in done_post:
                        file_path, pre_elapsed = post_futures.pop(future)
                        if future.cancelled():
                            add_stopped(file_path, elapsed=pre_elapsed)
                            continue
                        try:
                            success, error, post_elapsed = future.result()
                        except CancelledError:
                            add_stopped(file_path, elapsed=pre_elapsed)
                            continue
                        except Exception as e:
                            success, error, post_elapsed = False, str(e), 0.0

                        total_elapsed_file = pre_elapsed + post_elapsed
                        if success:
                            add_success(file_path, total_elapsed_file)
                        else:
                            add_failure(file_path, error, total_elapsed_file)

                    active_workers = min(workers, len(pre_futures)) + (1 if post_futures else 0)
                    live.update(render(active_workers=active_workers))
    else:
        with Live(render(active_workers=workers), console=console, refresh_per_second=5, transient=False) as live:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_file, file_path, args, results_root): file_path
                    for file_path in files
                }
                cancellation_applied = False

                while futures:
                    maybe_request_stop()
                    if stop_requested and not cancellation_applied:
                        for future, file_path in list(futures.items()):
                            if future.cancel():
                                futures.pop(future, None)
                                add_stopped(file_path)
                        cancellation_applied = True

                    done_futures = []
                    try:
                        for future in as_completed(futures, timeout=1):
                            done_futures.append(future)
                            if len(done_futures) >= workers:
                                break
                    except TimeoutError:
                        pass

                    for future in done_futures:
                        file_path = futures.pop(future)
                        if future.cancelled():
                            add_stopped(file_path)
                            continue
                        try:
                            success, error, elapsed = future.result()
                        except CancelledError:
                            add_stopped(file_path)
                            continue
                        except Exception as e:  # Defensive fallback for executor failures
                            success, error, elapsed = False, str(e), 0.0

                        if success:
                            add_success(file_path, elapsed)
                        else:
                            add_failure(file_path, error, elapsed)

                    active_workers = min(workers, len(futures))
                    live.update(render(active_workers=active_workers))

    total_time = time.perf_counter() - t0
    completed_non_stopped = max(0, total - stopped_count)
    throughput = (completed_non_stopped / total_time) * 3600 if total_time > 0 else 0.0
    avg_per_molecule = total_time / completed_non_stopped if completed_non_stopped else 0.0
    merged_records = _dedupe_batch_records(existing_batch_records + batch_records)
    aggregate_total_time = _sum_elapsed_seconds(merged_records)
    aggregate_json_path, aggregate_csv_path, quadrant_payload, batch_delta_json_path, batch_delta_txt_path = write_batch_aggregate_json(
        directory=directory,
        results_root=results_root,
        records=merged_records,
        mode=mode,
        workers=workers,
        total_time=aggregate_total_time,
        water=bool(getattr(args, "water", False)),
        interactive_quadrant_plot=(interactive_quadrant_plot or stop_requested),
    )

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Total files", str(total))
    summary.add_row("Success", str(succeeded))
    summary.add_row("Failed", str(len(failures)))
    summary.add_row("Stopped", str(stopped_count))
    if skipped_existing:
        summary.add_row("Skipped existing", str(skipped_existing))
    summary.add_row("Total time", _fmt_elapsed(total_time))
    summary.add_row("Avg per molecule", f"{avg_per_molecule:.1f}s" if completed_non_stopped else "n/a")
    summary.add_row("Throughput", f"{throughput:.1f} jobs/hour" if completed_non_stopped else "n/a")
    summary.add_row("Peak CPU", f"{peak_cpu:.1f}%")
    summary.add_row("Peak RAM", f"{peak_ram:.1f} MB")
    summary.add_row("Batch JSON", aggregate_json_path)
    summary.add_row("Batch CSV", aggregate_csv_path)
    reverse_json_path = os.path.join(results_root, _final_output_name("kuid_reverse_index.json", water_mode))
    reverse_csv_path = os.path.join(results_root, _final_output_name("kuid_reverse_index.csv", water_mode))
    intensive_dist_csv_path = os.path.join(
        results_root, _final_output_name("kuid_intensive_family_distribution.csv", water_mode)
    )
    intensive_dist_png_path = os.path.join(
        results_root, _final_output_name("kuid_intensive_family_distribution.png", water_mode)
    )
    intensive_calibration_path = os.path.join(
        results_root, _final_output_name("kuid_intensive_calibration.json", water_mode)
    )
    if os.path.exists(intensive_calibration_path):
        summary.add_row("KUID-Intensive Cal", intensive_calibration_path)
    if os.path.exists(reverse_json_path):
        summary.add_row("KUID Reverse JSON", reverse_json_path)
    if os.path.exists(reverse_csv_path):
        summary.add_row("KUID Reverse CSV", reverse_csv_path)
    if os.path.exists(intensive_dist_csv_path):
        summary.add_row("KUID-Intensive Dist CSV", intensive_dist_csv_path)
    if os.path.exists(intensive_dist_png_path):
        summary.add_row("KUID-Intensive Dist PNG", intensive_dist_png_path)
    if batch_delta_json_path:
        summary.add_row("Batch Delta JSON", batch_delta_json_path)
    if batch_delta_txt_path:
        summary.add_row("Batch Delta TXT", batch_delta_txt_path)
    if quadrant_payload.get("quadrant_plot_png"):
        summary.add_row("Quadrant Plot", quadrant_payload["quadrant_plot_png"])
    elif quadrant_payload.get("plot_error"):
        summary.add_row("Quadrant Plot", f"not generated ({quadrant_payload['plot_error']})")
    if quadrant_payload.get("quadrant_json"):
        summary.add_row("Quadrant JSON", quadrant_payload["quadrant_json"])
    panel_title = "Batch Stopped" if stop_requested else "Batch Completed"
    panel_color = "yellow" if stop_requested else "green"
    console.print(Panel(summary, title=panel_title, border_style=panel_color))

    if failures:
        fail_table = Table(title="Failed Files", expand=True)
        fail_table.add_column("File")
        fail_table.add_column("Error")
        for file_path, error in failures:
            fail_table.add_row(os.path.basename(file_path), str(error))
        console.print(fail_table)

def main():
    # If no arguments provided -> Interactive mode (simplified)
    if len(sys.argv) == 1:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        print("\n------------------------------------------------------------")
        print("      KNF-Core Interactive Mode")
        print("------------------------------------------------------------\n")
        
        while True:
            input_path = input("Enter path to input file or folder (or 'q' to quit): ").strip()
            if input_path.lower() == 'q':
                sys.exit(0)
                
            input_path = input_path.strip('"').strip("'")
            input_path = utils.resolve_artifacted_path(input_path)
            
            if not os.path.exists(input_path):
                print(f"Error: Path '{input_path}' not found.")
                continue
            break

        nci_mode = input(
            "Run mode [default/gpu/multiwfn] (default: default): "
        ).strip().lower()
        if nci_mode not in {"", "default", "gpu", "multiwfn"}:
            print(f"Unknown mode '{nci_mode}'. Using default.")
            nci_mode = "default"
        
        # Define defaults for interactive mode
        class Args:
            charge = 0
            spin = 1
            force = True # "Just do this" implies run it.
            clean = True # "Just do this" often implies fresh run.
            debug = True # Helpful for user to see what's happening.
            output_dir = None
            processing = 'auto'
            workers = None
            ram_per_job = 50.0
            refresh_autoconfig = False
            quiet_config = False
            full_files = False
            nci_backend = "torch"
            nci_grid_spacing = 0.2
            nci_grid_padding = 3.0
            nci_device = "cpu"
            nci_dtype = "float32"
            nci_batch_size = 250000
            nci_eig_batch_size = 200000
            nci_rho_floor = 1e-12
            nci_apply_primitive_norm = False
            scdi_var_min = None
            scdi_var_max = None
            wbo_mode = "native"
            enable_stop_key = True
            interactive_quadrant_plot = False
            
        args = Args()

        if nci_mode == "gpu":
            args.nci_backend = "torch"
            args.nci_device = "cuda"
        elif nci_mode == "multiwfn":
            args.nci_backend = "multiwfn"
            args.nci_device = "auto"

        first_ok = first_run.ensure_first_run_setup(
            require_multiwfn=(args.nci_backend == "multiwfn"),
        )
        check_dependencies(nci_backend=args.nci_backend)
        if not first_ok:
            print("First-time setup is incomplete. Please install missing tools and run again.")
            sys.exit(1)
        
        if os.path.isdir(input_path):
            mode = input("Processing mode [auto/single/multi] (default: auto): ").strip().lower()
            if mode in {'auto', 'single', 'multi'}:
                args.processing = mode
            run_batch_directory(input_path, args)
        else:
            run_single_file(input_path, args)
            
        print("\nDone.")
        return

    # Batch/CLI Mode
    # Supports: knf full <file/folder> [options]
    # Or just: knf <file/folder> [options] (as alias)
    
    # Remove 'full' if present to normalize
    if len(sys.argv) > 1 and sys.argv[1] == 'full':
        sys.argv.pop(1)
        
    parser = argparse.ArgumentParser(description="KNF-Core Execution")
    parser.add_argument('input_path', help="Path to input molecular file or directory")
    parser.add_argument('--charge', type=int, default=0, help="Total system charge")
    parser.add_argument('--spin', type=int, default=1, help="Total system spin multiplicity")
    parser.add_argument(
        '--water',
        action='store_true',
        help="Use xTB '--alpb water' for optimization and single-point instead of the default '--cosmo water'.",
    )
    parser.add_argument('--force', action='store_true', help="Force recomputation")
    parser.add_argument('--clean', action='store_true', help="Clean results")
    parser.add_argument('--debug', action='store_true', help="Debug logging")
    parser.add_argument(
        '--processing',
        '--processes',
        dest='processing',
        choices=['auto', 'single', 'multi'],
        default='auto',
        help="Processing mode: auto (default), single, multi"
    )
    parser.add_argument(
        '--multi',
        action='store_true',
        help="Shortcut for --processing multi"
    )
    parser.add_argument(
        '--single',
        action='store_true',
        help="Shortcut for --processing single"
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help="Optional override for worker threads. Default: auto-decide in multi mode"
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help="Custom output directory. Default: <input>/Results"
    )
    parser.add_argument(
        '--batches',
        nargs='?',
        type=int,
        const=0,
        default=None,
        help=(
            "Split directory inputs into evenly sized batches. "
            "Use '--batches N' to force N batches, or '--batches' for auto batch count."
        ),
    )
    parser.add_argument(
        '--universal-kuid',
        action='store_true',
        help=(
            "Recompute a universal KUID/KUID-Intensive calibration by combining existing "
            "batch_knf outputs discovered under the input directory."
        ),
    )
    parser.add_argument(
        '--ram-per-job',
        type=float,
        default=50.0,
        help="Estimated RAM in MB per concurrent job for auto-config"
    )
    parser.add_argument(
        '--refresh-autoconfig',
        action='store_true',
        help="Recompute and overwrite one-time auto-config cache"
    )
    parser.add_argument(
        '--quiet-config',
        action='store_true',
        help="Hide auto-configuration summary banner"
    )
    parser.add_argument(
        '--full-files',
        action='store_true',
        help=(
            "Keep all intermediate and large files. "
            "Default behavior is storage-efficient cleanup."
        ),
    )
    parser.add_argument(
        '--enable-stop-key',
        action='store_true',
        help=f"Enable graceful stop during batch runs by pressing '{STOP_KEY}'.",
    )
    parser.add_argument(
        '--interactive-quadrant-plot',
        action='store_true',
        help="Open an interactive SNCI_Norm vs SCDI_Norm quadrant plot window after batch aggregation.",
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help="Shortcut: use torch NCI backend on CUDA"
    )
    parser.add_argument(
        '--multiwfn',
        action='store_true',
        help="Use Multiwfn backend for NCI instead of default Torch backend"
    )
    parser.add_argument(
        '--nci-backend',
        choices=['multiwfn', 'torch'],
        default='torch',
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nci-grid-spacing',
        type=float,
        default=0.2,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nci-grid-padding',
        type=float,
        default=3.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nci-device',
        default='cpu',
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nci-dtype',
        choices=['float32', 'float64'],
        default='float32',
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nci-batch-size',
        type=int,
        default=250000,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nci-eig-batch-size',
        type=int,
        default=200000,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nci-rho-floor',
        type=float,
        default=1e-12,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nci-apply-primitive-norm',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--scdi-var-min',
        type=float,
        default=None,
        help="Fixed global Var_min for SCDI normalization.",
    )
    parser.add_argument(
        '--scdi-var-max',
        type=float,
        default=None,
        help="Fixed global Var_max for SCDI normalization.",
    )
    parser.add_argument(
        '--wbo-mode',
        choices=['native', 'xtb'],
        default='native',
        help="WBO computation mode: native (default, from molden.input) or xtb (from xTB wbo file).",
    )
    parser.add_argument(
        '--refresh-first-run',
        action='store_true',
        help="Re-run one-time first-run setup and overwrite its cached state"
    )
    parser.add_argument(
        '--multiwfn-path',
        default=None,
        help="Path to Multiwfn executable or folder (saved for future runs)"
    )
    
    args = parser.parse_args()
    args.input_path = utils.resolve_artifacted_path(args.input_path)

    if args.multi and args.single:
        parser.error("Use only one of --multi or --single.")
    if args.gpu and args.multiwfn:
        parser.error("Use only one of --gpu or --multiwfn.")
    if args.batches is not None and args.batches < 0:
        parser.error("--batches must be a positive integer, or provided without a value for auto mode.")
    if args.batches is not None and args.universal_kuid:
        parser.error("Use either --batches or --universal-kuid, not both in the same command.")

    if args.multi:
        args.processing = "multi"
    elif args.single:
        args.processing = "single"

    if args.multiwfn:
        args.nci_backend = "multiwfn"
        args.nci_device = "auto"
    elif args.gpu:
        args.nci_backend = "torch"
        args.nci_device = "cuda"
        args.nci_dtype = "float64"

    # Configure logging after CLI args are known.
    if args.debug:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

    if not args.universal_kuid:
        first_ok = first_run.ensure_first_run_setup(
            force=args.refresh_first_run,
            multiwfn_path=args.multiwfn_path,
            require_multiwfn=(args.nci_backend == "multiwfn"),
        )
        check_dependencies(multiwfn_path=args.multiwfn_path, nci_backend=args.nci_backend)
        if not first_ok:
            logging.error("First-time setup is incomplete. Install missing tools and retry.")
            sys.exit(1)
    
    # If user provided flags, use them.
    # Default behavior for 'knf <file>' without flags is now determined by argparse defaults.
    
    if os.path.isdir(args.input_path):
        if args.universal_kuid:
            run_universal_kuid(args.input_path, args)
        elif args.batches is not None:
            run_batch_directory_batched(args.input_path, args)
        else:
            run_batch_directory(args.input_path, args)
    else:
        if args.universal_kuid:
            parser.error("--universal-kuid requires a directory input path.")
        if args.batches is not None:
            parser.error("--batches requires a directory input path.")
        run_single_file(args.input_path, args)

if __name__ == "__main__":
    main()
