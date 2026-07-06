import csv
import json
import logging
import os
from datetime import datetime, timezone

from .. import kuid, kuid_index, kuid_intensive, knf_vector
from .discovery import (
    _BATCH_LEGACY_CSV_NAMES,
    _BATCH_PRIMARY_CSV_NAME,
    _batch_primary_csv_path,
    _cleanup_redundant_batch_aliases,
    _existing_batch_csv_path,
    _safe_float,
)
from .naming import _final_output_name

_KUID_NON_F2_REQUIRED_INDEXES = (0, 2, 3, 4, 5, 6, 7, 8)


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


def _kuid_intensive_prefix_fields(kuid_intensive_raw: str) -> dict:
    return kuid_index.kuid_intensive_progressive_prefix_fields(kuid_intensive_raw)


def _apply_kuid_prefix_fields(record: dict):
    intensive_raw = (record.get("KUID_Intensive_raw") or "").strip()
    if intensive_raw:
        record.update(_kuid_intensive_prefix_fields(intensive_raw))
        return
    raw = (record.get("KUID") or record.get("KUID_raw") or "").strip()
    record.update(kuid_index.kuid_prefix_fields(raw))


def _write_kuid_index_outputs(rows: list[dict], results_root: str, water: bool = False) -> dict:
    family_json_path = os.path.join(results_root, _final_output_name("kuid_family_stats.json", water))
    family_csv_path = os.path.join(results_root, _final_output_name("kuid_family_stats.csv", water))
    bridge_json_path = os.path.join(results_root, _final_output_name("kuid_full_topology_bridge.json", water))
    bridge_csv_path = os.path.join(results_root, _final_output_name("kuid_full_topology_bridge.csv", water))

    prefix_json_path = os.path.join(results_root, _final_output_name("kuid_prefix_index.json", water))
    topology_prefix_json_path = os.path.join(
        results_root, _final_output_name("kuid_topology_prefix_index.json", water)
    )
    instance_prefix_json_path = os.path.join(
        results_root, _final_output_name("kuid_instance_prefix_index.json", water)
    )

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

    topology_prefix_index = kuid_index.build_prefix_index(
        rows,
        code_field="KUID_Intensive_raw",
        use_row_prefix_fields=False,
        prefix_specs=(("prefix2", 1), ("prefix4", 2), ("prefix6", 3)),
        code_normalizer=kuid_index.normalize_prefix_token,
    )
    topology_prefix_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "index_type": "topology_passport",
        "code_field": "KUID_Intensive_raw",
        "prefix_semantics": {
            "prefix2": "f3",
            "prefix4": "f3+f4",
            "prefix6": "f3+f4+f7",
            "full_kuid_intensive": "f3+f4+f7+f8+f9",
        },
        "index": topology_prefix_index,
    }
    with open(topology_prefix_json_path, "w", encoding="utf-8") as f:
        json.dump(topology_prefix_payload, f, indent=2)

    # Backward-compatible file name: keep this as topology passport index.
    with open(prefix_json_path, "w", encoding="utf-8") as f:
        json.dump(topology_prefix_payload, f, indent=2)

    instance_prefix_index = kuid_index.build_prefix_index(
        rows,
        code_field="KUID",
        use_row_prefix_fields=False,
        prefix_specs=(("prefix2", 2), ("prefix4", 4), ("prefix6", 6)),
        code_normalizer=kuid_index.normalize_kuid_raw,
    )
    with open(instance_prefix_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "index_type": "instance_address",
                "code_field": "KUID",
                "prefix_semantics": {
                    "prefix2": "f1",
                    "prefix4": "f1+f2",
                    "prefix6": "f1+f2+f3",
                    "full_kuid": "f1+f2+f3+f4+f5+f6+f7+f8+f9",
                },
                "index": instance_prefix_index,
            },
            f,
            indent=2,
        )

    full_to_topology = {}
    for row in rows:
        full_code = kuid_index.normalize_kuid_raw(row.get("KUID") or row.get("KUID_raw"))
        topology_code = kuid_index.normalize_prefix_token(row.get("KUID_Intensive_raw"))
        if not full_code or not topology_code:
            continue
        entry = full_to_topology.setdefault(
            full_code,
            {
                "topology_passports": set(),
                "member_count": 0,
                "example_files": [],
            },
        )
        entry["topology_passports"].add(topology_code)
        entry["member_count"] += 1
        file_name = (
            row.get("File")
            or row.get("file")
            or row.get("input_file_name")
            or row.get("input_file")
            or ""
        )
        if file_name and len(entry["example_files"]) < 5 and file_name not in entry["example_files"]:
            entry["example_files"].append(file_name)

    bridge_rows = []
    for full_code in sorted(full_to_topology):
        item = full_to_topology[full_code]
        bridge_rows.append(
            {
                "kuid_full": full_code,
                "topology_passports": sorted(item["topology_passports"]),
                "topology_count": len(item["topology_passports"]),
                "member_count": item["member_count"],
                "example_files": item["example_files"],
            }
        )

    with open(bridge_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "kuid_full_role": "instance_address",
                "kuid_intensive_role": "topology_passport",
                "entries": bridge_rows,
            },
            f,
            indent=2,
        )

    with open(bridge_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "kuid_full",
                "topology_count",
                "topology_passports",
                "member_count",
                "example_files",
            ],
        )
        writer.writeheader()
        for item in bridge_rows:
            writer.writerow(
                {
                    "kuid_full": item["kuid_full"],
                    "topology_count": item["topology_count"],
                    "topology_passports": "; ".join(item["topology_passports"]),
                    "member_count": item["member_count"],
                    "example_files": "; ".join(item["example_files"]),
                }
            )

    return {
        "family_stats_json": family_json_path,
        "family_stats_csv": family_csv_path,
        "prefix_index_json": prefix_json_path,
        "topology_prefix_index_json": topology_prefix_json_path,
        "instance_prefix_index_json": instance_prefix_json_path,
        "full_topology_bridge_json": bridge_json_path,
        "full_topology_bridge_csv": bridge_csv_path,
        "family_count": len(family_stats),
        "bridge_entry_count": len(bridge_rows),
    }


def _write_kuid_reverse_index_outputs(rows: list[dict], results_root: str, water: bool = False) -> dict:
    reverse_json_path = os.path.join(results_root, _final_output_name("kuid_reverse_index.json", water))
    reverse_csv_path = os.path.join(results_root, _final_output_name("kuid_reverse_index.csv", water))
    topology_reverse_json_path = os.path.join(
        results_root, _final_output_name("kuid_topology_reverse_index.json", water)
    )
    topology_reverse_csv_path = os.path.join(
        results_root, _final_output_name("kuid_topology_reverse_index.csv", water)
    )

    def _build_reverse_index(code_fields: list[str]):
        reverse_index = {}
        missing_rows = 0
        for row in rows:
            code = ""
            for field in code_fields:
                code = (row.get(field) or "").strip()
                if code:
                    break
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
        return sorted_index, missing_rows

    instance_index, missing_instance = _build_reverse_index(["KUID_Cluster", "KUID", "KUID_raw"])
    topology_index, missing_topology = _build_reverse_index(
        ["KUID_Intensive_Cluster", "KUID_Intensive", "KUID_Intensive_raw"]
    )

    with open(reverse_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "index_type": "instance_address",
                "code_field": "KUID_Cluster",
                "cluster_pattern": "f1f2f3-f4f5-f6f7-f8f9",
                "total_kuid_clusters": len(instance_index),
                "missing_kuid_rows": missing_instance,
                "index": instance_index,
            },
            f,
            indent=2,
        )

    with open(reverse_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["KUID_Cluster", "complex_count", "complexes"])
        writer.writeheader()
        for code, items in instance_index.items():
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

    with open(topology_reverse_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "index_type": "topology_passport",
                "code_field": "KUID_Intensive_Cluster",
                "cluster_pattern": "f3f4f7-f8f9",
                "total_kuid_topology_clusters": len(topology_index),
                "missing_kuid_topology_rows": missing_topology,
                "index": topology_index,
            },
            f,
            indent=2,
        )

    with open(topology_reverse_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["KUID_Intensive_Cluster", "complex_count", "complexes"]
        )
        writer.writeheader()
        for code, items in topology_index.items():
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
                    "KUID_Intensive_Cluster": code,
                    "complex_count": len(items),
                    "complexes": "; ".join(labels),
                }
            )

    return {
        "reverse_index_json": reverse_json_path,
        "reverse_index_csv": reverse_csv_path,
        "topology_reverse_index_json": topology_reverse_json_path,
        "topology_reverse_index_csv": topology_reverse_csv_path,
        "total_kuid_clusters": len(instance_index),
        "total_kuid_topology_clusters": len(topology_index),
        "missing_kuid_rows": missing_instance,
        "missing_kuid_topology_rows": missing_topology,
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
            total_families = float(sum(y_values))

            # Build CCDF: P(Family Size >= x)
            ccdf_pairs_desc = []
            running_tail = 0
            for size, count in reversed(ordered_distribution):
                running_tail += int(count)
                ccdf_pairs_desc.append((size, running_tail / total_families))
            ccdf_pairs = list(reversed(ccdf_pairs_desc))
            ccdf_x = [size for size, _ in ccdf_pairs]
            ccdf_y = [prob for _, prob in ccdf_pairs]

            fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.6), dpi=170)

            ax_a.bar(
                x_values,
                y_values,
                width=0.9,
                color="#1f77b4",
                alpha=0.9,
                edgecolor="white",
                linewidth=0.25,
            )
            ax_a.set_yscale("log")
            ax_a.set_xlabel("Family Size")
            ax_a.set_ylabel("Number of Families (log)")
            ax_a.set_title("Panel A: Family Size Histogram")
            ax_a.grid(True, axis="y", linestyle="--", alpha=0.35)

            ax_b.step(
                ccdf_x,
                ccdf_y,
                where="post",
                color="#d94801",
                linewidth=2.0,
                label="CCDF",
            )
            ax_b.scatter(ccdf_x, ccdf_y, s=10, color="#d94801", alpha=0.75)
            ax_b.set_xscale("log")
            ax_b.set_yscale("log")
            ax_b.set_xlabel("Family Size")
            ax_b.set_ylabel("P(Size >= x)")
            ax_b.set_title("Panel B: CCDF of Family Size")
            ax_b.grid(True, which="both", linestyle="--", alpha=0.35)
            ax_b.legend(loc="upper right")

            fig.suptitle("KUID-Intensive Family Distribution", fontsize=14, y=0.99)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
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
    preferred_head = [
        "source_batch",
        "File",
        *[f"f{i}" for i in range(1, 10)],
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
    available = {str(name).strip() for name in (fieldnames or []) if str(name).strip()}
    filtered = [name for name in preferred_head if name in available]
    if filtered:
        return filtered
    return [name for name in preferred_head if name != "source_batch"]


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
        _apply_kuid_prefix_fields(entry)

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
