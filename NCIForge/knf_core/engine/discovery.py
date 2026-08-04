import csv
import json
import logging
import os
from pathlib import Path

import psutil

from .. import utils
from .constants import VALID_INPUT_EXTS
from .naming import _final_output_name

# NOTE: These batch-CSV-path/cleanup helpers and their backing constants were
# originally grouped with the "atlas" code in main.py (contiguous with
# ATLAS_* constants), but they are relocated here rather than into
# engine/atlas.py. Both engine/kuid_ops.py (_run_kuid_only_from_existing_batch)
# and this module's own _load_existing_batch_records need
# _existing_batch_csv_path, while engine/atlas.py needs resolve_results_root
# from this module. Keeping the path helpers in engine/atlas.py would create
# a real circular import (discovery <-> atlas, and kuid_ops <-> atlas); since
# these helpers are pure results-root/CSV-path resolution logic (no ATLAS_*
# schema knowledge), they belong here.
_BATCH_PRIMARY_CSV_NAME = "batch_knf_unified.csv"
_BATCH_LEGACY_CSV_NAMES = (
    "atlas_submission.csv",
    "batch_knf_unified_kuid_intensive.csv",
    "batch_knf.csv",
)
_BATCH_LEGACY_JSON_NAMES = ("batch_knf_unified_kuid_intensive.json",)


# NOTE: _safe_float is defined here (rather than in engine/kuid_ops.py, where
# most other KUID helpers live) because _dedupe_batch_records and
# _sum_elapsed_seconds in this module need it, and this module must not
# depend on engine/kuid_ops.py (kuid_ops.py's _run_kuid_only_from_existing_batch
# depends on this module's batch-CSV helpers, so the dependency has to run
# discovery -> kuid_ops, not the other way). engine/kuid_ops.py imports this
# same function from here.
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


def _has_reusable_compound_outputs(file_path: str, results_root: str, water: bool = False) -> bool:
    stem = Path(file_path).stem
    result_dir = os.path.join(results_root, stem)
    output_txt_path = os.path.join(result_dir, _final_output_name("output.txt", water))
    return os.path.exists(output_txt_path)


def _cleanup_compound_knf_json_outputs(results_root: str, water: bool = False) -> int:
    target_name = _final_output_name("knf.json", water)
    removed = 0
    for root, _, files in os.walk(results_root):
        if target_name not in files:
            continue
        path = os.path.join(root, target_name)
        try:
            os.remove(path)
            removed += 1
        except Exception as e:
            logging.warning("Could not remove %s: %s", path, e)
    return removed


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


def _cleanup_submission_auxiliary_outputs(results_root: str, water: bool = False) -> list[str]:
    """Remove non-submission artifacts to keep atlas-bundle workflows lean."""
    removable_names = [
        "batch_knf.json",
        "kuid_calibration.json",
        "kuid_calibration_unified.json",
        "kuid_family_stats.json",
        "kuid_family_stats.csv",
        "kuid_full_topology_bridge.json",
        "kuid_full_topology_bridge.csv",
        "kuid_instance_prefix_index.json",
        "kuid_intensive_calibration.json",
        "kuid_intensive_calibration_unified.json",
        "kuid_intensive_family_distribution.csv",
        "kuid_intensive_family_distribution.png",
        "kuid_prefix_index.json",
        "kuid_reverse_index.json",
        "kuid_reverse_index.csv",
        "kuid_topology_prefix_index.json",
        "kuid_topology_reverse_index.json",
        "kuid_topology_reverse_index.csv",
        "snci_scdi_quadrants.json",
        "snci_scdi_quadrants.png",
    ]

    removed = []
    for name in removable_names:
        path = os.path.join(results_root, _final_output_name(name, water))
        if not os.path.exists(path):
            continue
        try:
            os.remove(path)
            removed.append(path)
        except Exception as e:
            logging.warning("Could not remove auxiliary submission artifact %s: %s", path, e)
    return removed


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
