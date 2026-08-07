import csv
import hashlib
import json
import logging
import math
import os

from .. import kuid, kuid_index, kuid_intensive
from .constants import CLI_TITLE
from .discovery import _existing_batch_csv_path, resolve_results_root
from .kuid_ops import _compute_kuid_intensive_payload
from .naming import _final_output_name
from .timeutil import _utc_now_iso_z

ATLAS_REQUIRED_COLUMNS = [
    "molecule_name",
    "charge",
    "spin",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "SNCI",
    "SCDI",
    "SCDI_variance",
    "backend",
    "device",
    "xtb_version",
    "knf_core_version",
    "nci_grid_spacing",
    "nci_grid_padding",
    "water_mode",
    "KUID_raw",
    "KUID_Cluster",
    "KUID_Intensive_raw",
    "KUID_Intensive_Cluster",
    "instance_hash",
]
ATLAS_OPTIONAL_COLUMNS = ["source_batch"]
ATLAS_NUMERIC_FIELDS = [
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "SNCI",
    "SCDI_variance",
    "nci_grid_spacing",
    "nci_grid_padding",
]
ATLAS_OPTIONAL_NUMERIC_FIELDS = ["SCDI"]
ATLAS_INTEGER_FIELDS = ["charge", "spin"]
ATLAS_REQUIRED_STRING_FIELDS = [
    "molecule_name",
    "backend",
    "device",
    "xtb_version",
    "knf_core_version",
    "KUID_raw",
    "KUID_Cluster",
    "KUID_Intensive_raw",
    "KUID_Intensive_Cluster",
]
ATLAS_BUNDLE_DIRNAME = "submission_bundle"
ATLAS_BUNDLE_CSV_NAME = "atlas_submission.csv"
ATLAS_BUNDLE_MANIFEST_NAME = "manifest.json"
ATLAS_SCHEMA_VERSION = "2.0"
ATLAS_DEFAULT_XTB_VERSION = "unknown"
ATLAS_INSTANCE_HASH_FEATURE_PRECISION = 6
ATLAS_INSTANCE_HASH_GRID_PRECISION = 3
ATLAS_INSTANCE_HASH_BASIS = (
    "sha256(f1..f9,charge,spin,xtb_version,nci_grid_spacing,nci_grid_padding)"
)


def _hash_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _knf_version_from_title() -> str:
    marker = "v"
    idx = CLI_TITLE.rfind(marker)
    if idx >= 0 and idx + 1 < len(CLI_TITLE):
        return CLI_TITLE[idx + 1 :].strip()
    return CLI_TITLE.strip()


def _first_nonempty(*values):
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text != "":
            return text
    return ""


def _finite_float(value, field: str, row_number: int) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Row {row_number}: {field} must be numeric.")
    if not math.isfinite(out):
        raise ValueError(f"Row {row_number}: {field} must be finite (no NaN/inf).")
    return out


def _strict_int(value, field: str, row_number: int) -> int:
    out = _finite_float(value, field, row_number)
    rounded = int(round(out))
    if abs(out - rounded) > 1e-9:
        raise ValueError(f"Row {row_number}: {field} must be an integer value.")
    return rounded


def _coerce_bool_int(value, field: str, row_number: int) -> int:
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        num = int(round(float(value)))
        if num in (0, 1):
            return num
        raise ValueError(f"Row {row_number}: {field} must be boolean-like (0/1 or true/false).")

    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return 1
    if text in ("0", "false", "no", "n", "off", ""):
        return 0
    raise ValueError(f"Row {row_number}: {field} must be boolean-like (0/1 or true/false).")


def _safe_float_or_none(value):
    try:
        if value is None:
            return None
        out = float(value)
        if not math.isfinite(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _fallback_kuid_intensive_raw_from_features(source_row: dict) -> str:
    feature_map = {}
    for feature in ("f3", "f4", "f7", "f8", "f9"):
        value = _safe_float_or_none(source_row.get(feature))
        if value is None:
            return ""
        feature_map[feature] = value

    try:
        calibration = kuid_intensive.build_calibration_from_feature_maps([feature_map])
        encoded = kuid_intensive.encode_feature_map(feature_map, calibration)
        return str(encoded.get("raw") or "")
    except Exception:
        return ""


def _derive_kuid_cluster(raw_hex: str, provided_cluster: str) -> str:
    raw = kuid_index.normalize_kuid_raw(raw_hex)
    if len(raw) == 18:
        try:
            return kuid.format_kuid_cluster(raw)
        except Exception:
            pass
    return _first_nonempty(provided_cluster)


def _derive_kuid_intensive_cluster(raw_hex: str, provided_cluster: str) -> str:
    raw = kuid_index.normalize_prefix_token(raw_hex)
    if len(raw) == 5:
        try:
            return kuid_intensive.format_kuid_intensive_cluster(raw)
        except Exception:
            pass
    return _first_nonempty(provided_cluster)


def _compute_atlas_instance_hash(row: dict) -> str:
    payload = {
        "f1": round(float(row["f1"]), ATLAS_INSTANCE_HASH_FEATURE_PRECISION),
        "f2": round(float(row["f2"]), ATLAS_INSTANCE_HASH_FEATURE_PRECISION),
        "f3": round(float(row["f3"]), ATLAS_INSTANCE_HASH_FEATURE_PRECISION),
        "f4": round(float(row["f4"]), ATLAS_INSTANCE_HASH_FEATURE_PRECISION),
        "f5": round(float(row["f5"]), ATLAS_INSTANCE_HASH_FEATURE_PRECISION),
        "f6": round(float(row["f6"]), ATLAS_INSTANCE_HASH_FEATURE_PRECISION),
        "f7": round(float(row["f7"]), ATLAS_INSTANCE_HASH_FEATURE_PRECISION),
        "f8": round(float(row["f8"]), ATLAS_INSTANCE_HASH_FEATURE_PRECISION),
        "f9": round(float(row["f9"]), ATLAS_INSTANCE_HASH_FEATURE_PRECISION),
        "charge": int(row["charge"]),
        "spin": int(row["spin"]),
        "xtb": str(row["xtb_version"]).strip(),
        "spacing": round(float(row["nci_grid_spacing"]), ATLAS_INSTANCE_HASH_GRID_PRECISION),
        "padding": round(float(row["nci_grid_padding"]), ATLAS_INSTANCE_HASH_GRID_PRECISION),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:8]


def _map_to_atlas_row(source_row: dict, args) -> dict:
    kuid_raw = kuid_index.normalize_kuid_raw(
        _first_nonempty(
            source_row.get("KUID_raw"),
            source_row.get("KUID"),
            source_row.get("KUID_full"),
        )
    )
    kuid_int_raw = kuid_index.normalize_prefix_token(
        _first_nonempty(
            source_row.get("KUID_Intensive_raw"),
            source_row.get("KUID_Intensive"),
            source_row.get("KUIDINT"),
        )
    )
    if not kuid_int_raw:
        kuid_int_raw = kuid_index.normalize_prefix_token(
            _fallback_kuid_intensive_raw_from_features(source_row)
        )
    atlas_row = {
        "molecule_name": _first_nonempty(
            source_row.get("molecule_name"),
            source_row.get("source_name"),
            source_row.get("File"),
        ),
        "charge": _first_nonempty(source_row.get("charge"), getattr(args, "charge", 0)),
        "spin": _first_nonempty(source_row.get("spin"), getattr(args, "spin", 1)),
        "SNCI": _first_nonempty(source_row.get("SNCI"), source_row.get("snci")),
        "SCDI": _first_nonempty(source_row.get("SCDI"), source_row.get("scdi")),
        "SCDI_variance": _first_nonempty(source_row.get("SCDI_variance")),
        "backend": _first_nonempty(source_row.get("backend"), getattr(args, "nci_backend", "torch")),
        "device": _first_nonempty(source_row.get("device"), getattr(args, "nci_device", "cpu")),
        "xtb_version": _first_nonempty(source_row.get("xtb_version"), ATLAS_DEFAULT_XTB_VERSION),
        "knf_core_version": _first_nonempty(
            source_row.get("knf_core_version"),
            source_row.get("knf_version"),
            _knf_version_from_title(),
        ),
        "nci_grid_spacing": _first_nonempty(
            source_row.get("nci_grid_spacing"),
            source_row.get("grid_spacing"),
            getattr(args, "nci_grid_spacing", 0.2),
        ),
        "nci_grid_padding": _first_nonempty(
            source_row.get("nci_grid_padding"),
            source_row.get("grid_padding"),
            getattr(args, "nci_grid_padding", 3.0),
        ),
        "water_mode": _first_nonempty(
            source_row.get("water_mode"),
            source_row.get("xtb_water"),
            1 if bool(getattr(args, "water", False)) else 0,
        ),
        "KUID_raw": kuid_raw,
        "KUID_Cluster": _derive_kuid_cluster(kuid_raw, _first_nonempty(source_row.get("KUID_Cluster"))),
        "KUID_Intensive_raw": kuid_int_raw,
        "KUID_Intensive_Cluster": _derive_kuid_intensive_cluster(
            kuid_int_raw, _first_nonempty(source_row.get("KUID_Intensive_Cluster"))
        ),
        "instance_hash": _first_nonempty(source_row.get("instance_hash")),
        "source_batch": _first_nonempty(source_row.get("source_batch")),
    }
    for idx in range(1, 10):
        atlas_row[f"f{idx}"] = _first_nonempty(source_row.get(f"f{idx}"))
    return atlas_row


def _validate_atlas_rows(rows: list[dict]):
    if not rows:
        raise ValueError("Atlas bundle requires at least one valid row.")

    for row_idx, row in enumerate(rows, start=2):
        missing = [col for col in ATLAS_REQUIRED_COLUMNS if col not in row]
        if missing:
            raise ValueError(f"Row {row_idx}: missing required columns: {', '.join(missing)}")

        for field in ATLAS_REQUIRED_STRING_FIELDS:
            if not _first_nonempty(row.get(field)):
                raise ValueError(f"Row {row_idx}: {field} must be a non-empty string.")

        for field in ATLAS_NUMERIC_FIELDS:
            try:
                row[field] = _finite_float(row.get(field), field, row_idx)
            except ValueError:
                # f2 can be undefined for some systems; preserve submission flow with a stable surrogate.
                if field == "f2":
                    row[field] = 180.0
                    logging.warning(
                        "Row %s: f2 was non-finite; using surrogate 180.0 for atlas submission export.",
                        row_idx,
                    )
                    continue
                raise

        for field in ATLAS_OPTIONAL_NUMERIC_FIELDS:
            if _first_nonempty(row.get(field)):
                row[field] = _finite_float(row.get(field), field, row_idx)
            else:
                row[field] = ""

        for field in ATLAS_INTEGER_FIELDS:
            row[field] = _strict_int(row.get(field), field, row_idx)

        row["water_mode"] = _coerce_bool_int(row.get("water_mode"), "water_mode", row_idx)

        kuid_raw = kuid_index.normalize_kuid_raw(row.get("KUID_raw"))
        if len(kuid_raw) != 18:
            raise ValueError(f"Row {row_idx}: KUID_raw must contain 18 hex chars (f1..f9 bytes).")
        row["KUID_raw"] = kuid_raw
        row["KUID_Cluster"] = _derive_kuid_cluster(kuid_raw, row.get("KUID_Cluster"))

        kuid_int_raw = kuid_index.normalize_prefix_token(row.get("KUID_Intensive_raw"))
        if len(kuid_int_raw) != 5:
            raise ValueError(f"Row {row_idx}: KUID_Intensive_raw must contain 5 hex chars.")
        row["KUID_Intensive_raw"] = kuid_int_raw
        row["KUID_Intensive_Cluster"] = _derive_kuid_intensive_cluster(
            kuid_int_raw, row.get("KUID_Intensive_Cluster")
        )

        row["instance_hash"] = _compute_atlas_instance_hash(row)


def _write_atlas_bundle(rows: list[dict], results_root: str, args) -> dict:
    _validate_atlas_rows(rows)
    bundle_dir = os.path.join(results_root, ATLAS_BUNDLE_DIRNAME)
    os.makedirs(bundle_dir, exist_ok=True)

    csv_path = os.path.join(bundle_dir, ATLAS_BUNDLE_CSV_NAME)
    manifest_path = os.path.join(bundle_dir, ATLAS_BUNDLE_MANIFEST_NAME)
    fieldnames = [*ATLAS_REQUIRED_COLUMNS, *ATLAS_OPTIONAL_COLUMNS]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {name: row.get(name, "") for name in fieldnames}
            writer.writerow(out)

    manifest = {
        "submission_schema_version": ATLAS_SCHEMA_VERSION,
        "knf_core_version": _knf_version_from_title(),
        "kuid_intensive_mode": "physics_fixed_bounds_v1",
        "instance_hash_basis": ATLAS_INSTANCE_HASH_BASIS,
        "hash_precision": {
            "features_decimals": ATLAS_INSTANCE_HASH_FEATURE_PRECISION,
            "grid_decimals": ATLAS_INSTANCE_HASH_GRID_PRECISION,
        },
        "row_count": len(rows),
        "created_at": _utc_now_iso_z(),
        "csv_sha256": _hash_file_sha256(csv_path),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {"bundle_dir": bundle_dir, "csv_path": csv_path, "manifest_path": manifest_path}


def _build_atlas_rows_from_batch_csv(csv_path: str, args) -> list[dict]:
    rows = []
    skipped = []
    minimal_required = [*[f"f{i}" for i in range(1, 10)], "SNCI", "KUID_raw"]
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, source_row in enumerate(reader, start=2):
            mapped = _map_to_atlas_row(source_row, args)
            missing = [field for field in minimal_required if not _first_nonempty(mapped.get(field))]
            if missing:
                skipped.append(
                    {
                        "row": row_idx,
                        "molecule_name": _first_nonempty(mapped.get("molecule_name"), "<unknown>"),
                        "missing": missing,
                    }
                )
                continue
            rows.append(mapped)
    if skipped:
        preview = ", ".join(
            f"row {item['row']} ({item['molecule_name']})"
            for item in skipped[:5]
        )
        logging.warning(
            "Skipped %d row(s) while preparing atlas bundle from %s due to missing required fields. Examples: %s",
            len(skipped),
            csv_path,
            preview,
        )
    return rows


def _build_atlas_rows_from_single_result(file_path: str, args, results_root: str, water: bool) -> list[dict]:
    stem = os.path.splitext(os.path.basename(file_path))[0]
    result_dir = os.path.join(results_root, stem)
    knf_json_path = os.path.join(result_dir, _final_output_name("knf.json", water))
    if not os.path.exists(knf_json_path):
        raise FileNotFoundError(f"Missing KNF output for atlas bundle: {knf_json_path}")

    with open(knf_json_path, "r", encoding="utf-8") as f:
        knf_payload = json.load(f)

    entry = {
        "status": "success",
        "input_file": os.path.abspath(file_path),
        "input_file_name": os.path.basename(file_path),
        "result_dir": result_dir,
        "knf": knf_payload,
    }

    kuid_info = knf_payload.get("kuid") if isinstance(knf_payload, dict) else None
    if isinstance(kuid_info, dict):
        entry["KUID"] = _first_nonempty(kuid_info.get("display"), kuid_info.get("raw"))
        entry["KUID_raw"] = _first_nonempty(kuid_info.get("raw"))

    _compute_kuid_intensive_payload([entry], results_root=results_root, water=water)

    vector = knf_payload.get("KNF_vector") if isinstance(knf_payload, dict) else None
    vector = vector if isinstance(vector, list) else []
    metadata = knf_payload.get("metadata") if isinstance(knf_payload, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    nci_engine_meta = metadata.get("nci_engine_metadata")
    nci_engine_meta = nci_engine_meta if isinstance(nci_engine_meta, dict) else {}

    source_row = {
        "molecule_name": os.path.basename(file_path),
        "SNCI": knf_payload.get("SNCI"),
        "SCDI": knf_payload.get("SCDI"),
        "SCDI_variance": knf_payload.get("SCDI_variance"),
        "charge": metadata.get("charge", getattr(args, "charge", 0)),
        "spin": metadata.get("spin", getattr(args, "spin", 1)),
        "backend": metadata.get("nci_backend", getattr(args, "nci_backend", "torch")),
        "device": _first_nonempty(
            nci_engine_meta.get("device_resolved"),
            nci_engine_meta.get("device"),
            getattr(args, "nci_device", "cpu"),
        ),
        "xtb_version": metadata.get("xtb_version", ATLAS_DEFAULT_XTB_VERSION),
        "knf_core_version": _knf_version_from_title(),
        "nci_grid_spacing": getattr(args, "nci_grid_spacing", 0.2),
        "nci_grid_padding": getattr(args, "nci_grid_padding", 3.0),
        "water_mode": metadata.get("xtb_water", 1 if water else 0),
        "KUID": entry.get("KUID", ""),
        "KUID_raw": entry.get("KUID_raw", ""),
        "KUID_Cluster": entry.get("KUID_Cluster", ""),
        "KUID_Intensive": entry.get("KUID_Intensive", ""),
        "KUID_Intensive_raw": entry.get("KUID_Intensive_raw", ""),
        "KUID_Intensive_Cluster": entry.get("KUID_Intensive_Cluster", ""),
    }
    for idx in range(9):
        source_row[f"f{idx + 1}"] = vector[idx] if idx < len(vector) else ""

    return [_map_to_atlas_row(source_row, args)]


def maybe_write_atlas_bundle(args):
    if not bool(getattr(args, "atlas_bundle", False)):
        return None

    water_mode = bool(getattr(args, "water", False))
    input_path = args.input_path

    if os.path.isdir(input_path):
        if getattr(args, "batches", None) is not None or bool(getattr(args, "universal_kuid", False)):
            results_root = os.path.join(resolve_results_root(input_path, args.output_dir), "Combined Results")
        else:
            results_root = resolve_results_root(input_path, args.output_dir)
        source_csv_path = _existing_batch_csv_path(results_root, water=water_mode)
        if not os.path.exists(source_csv_path):
            raise FileNotFoundError(f"Could not find batch CSV for atlas bundle: {source_csv_path}")
        rows = _build_atlas_rows_from_batch_csv(source_csv_path, args)
    else:
        results_root = resolve_results_root(input_path, args.output_dir)
        rows = _build_atlas_rows_from_single_result(
            file_path=input_path,
            args=args,
            results_root=results_root,
            water=water_mode,
        )

    return _write_atlas_bundle(rows=rows, results_root=results_root, args=args)


def try_write_atlas_bundle_from_existing_outputs(args):
    """
    Upgrade path: if prior batch outputs already exist, create atlas bundle directly
    without re-running KNF computations.
    """
    if not bool(getattr(args, "atlas_bundle", False)):
        return None
    if not os.path.isdir(args.input_path):
        return None
    if bool(getattr(args, "force", False)):
        return None
    if getattr(args, "batches", None) is not None or bool(getattr(args, "universal_kuid", False)):
        return None

    water_mode = bool(getattr(args, "water", False))
    base_root = resolve_results_root(args.input_path, args.output_dir)
    candidate_roots = [base_root, os.path.join(base_root, "Combined Results")]
    last_error = None

    for root in candidate_roots:
        csv_path = _existing_batch_csv_path(root, water=water_mode)
        if not os.path.exists(csv_path):
            continue
        try:
            rows = _build_atlas_rows_from_batch_csv(csv_path, args)
            if not rows:
                logging.warning(
                    "Existing batch CSV found at %s but no valid rows were available for atlas bundle reuse; continuing with normal computation.",
                    csv_path,
                )
                continue
            bundle = _write_atlas_bundle(rows=rows, results_root=root, args=args)
            bundle["source_csv"] = csv_path
            bundle["source_results_root"] = root
            return bundle
        except ValueError as e:
            last_error = str(e)
            logging.warning(
                "Could not reuse existing outputs from %s for atlas bundle (%s). Falling back to normal computation.",
                csv_path,
                e,
            )
            continue

    if last_error:
        logging.info(
            "Atlas bundle reuse from existing outputs was skipped due to validation issues: %s",
            last_error,
        )

    return None
