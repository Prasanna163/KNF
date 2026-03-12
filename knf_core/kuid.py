import math
from datetime import datetime, timezone
from typing import Iterable

KUID_VERSION = "KUID-MVP-1.0"
NORMALIZATION = "minmax"
BINS_PER_FEATURE = 256
FEATURE_ORDER = [f"f{i}" for i in range(1, 10)]
DISPLAY_FORMAT = "XX-XX-XX-XX-XX-XX-XX-XX-XX"
CLUSTER_DISPLAY_FORMAT = "f1f2f3-f4f5-f6f7-f8f9"


def _safe_float(value):
    try:
        if value is None:
            return None
        val = float(value)
        if val != val:  # NaN guard
            return None
        return val
    except (TypeError, ValueError):
        return None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def format_kuid(raw_hex: str) -> str:
    expected_len = len(FEATURE_ORDER) * 2
    if len(raw_hex) != expected_len:
        raise ValueError(f"KUID raw hex must have length {expected_len}, got {len(raw_hex)}.")
    parts = [raw_hex[idx: idx + 2] for idx in range(0, expected_len, 2)]
    return "-".join(parts)


def format_kuid_cluster(raw_hex: str) -> str:
    expected_len = len(FEATURE_ORDER) * 2
    if len(raw_hex) != expected_len:
        raise ValueError(
            f"KUID raw hex must have length {expected_len}, got {len(raw_hex)}."
        )
    parts = [raw_hex[idx: idx + 2] for idx in range(0, expected_len, 2)]
    return f"{''.join(parts[0:3])}-{''.join(parts[3:5])}-{''.join(parts[5:7])}-{''.join(parts[7:9])}"


def build_calibration(knf_vectors: Iterable[Iterable[float]], calibration_id: str = None) -> dict:
    vectors = []
    for vector in knf_vectors:
        vals = [_safe_float(v) for v in vector]
        if len(vals) != 9 or any(v is None for v in vals):
            continue
        vectors.append(vals)

    if not vectors:
        raise ValueError("Cannot build KUID calibration: no valid KNF vectors provided.")

    feature_bounds = {}
    for idx, feature_name in enumerate(FEATURE_ORDER):
        values = [row[idx] for row in vectors]
        feature_bounds[feature_name] = {
            "min": float(min(values)),
            "max": float(max(values)),
        }

    calibration_value = calibration_id
    if not calibration_value:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        calibration_value = f"{KUID_VERSION}-{ts}"

    return {
        "kuid_version": KUID_VERSION,
        "calibration_id": calibration_value,
        "normalization": NORMALIZATION,
        "bins_per_feature": BINS_PER_FEATURE,
        "feature_order": list(FEATURE_ORDER),
        "display_format": DISPLAY_FORMAT,
        "cluster_display_format": CLUSTER_DISPLAY_FORMAT,
        "feature_bounds": feature_bounds,
    }


def encode_knf_vector(knf_vector: Iterable[float], calibration: dict) -> dict:
    values = [_safe_float(v) for v in knf_vector]
    if len(values) != 9 or any(v is None for v in values):
        raise ValueError("KNF vector must contain 9 numeric feature values.")

    bins_per_feature = int(calibration.get("bins_per_feature", BINS_PER_FEATURE))
    if bins_per_feature != 256:
        raise ValueError("KUID-MVP-1.0 expects bins_per_feature=256.")

    feature_order = calibration.get("feature_order") or FEATURE_ORDER
    if list(feature_order) != FEATURE_ORDER:
        raise ValueError("KUID-MVP-1.0 expects canonical feature order f1..f9.")

    feature_bounds = calibration.get("feature_bounds") or {}
    normalized_values = []
    bins = []
    hex_bytes = []

    for idx, feature_name in enumerate(FEATURE_ORDER):
        bounds = feature_bounds.get(feature_name) or {}
        fmin = _safe_float(bounds.get("min"))
        fmax = _safe_float(bounds.get("max"))
        if fmin is None or fmax is None:
            raise ValueError(f"Missing calibration bounds for {feature_name}.")

        if abs(fmax - fmin) <= 1e-12:
            normalized = 0.0
        else:
            normalized = (values[idx] - fmin) / (fmax - fmin)
            normalized = _clip01(normalized)
        bin_value = min(255, int(math.floor(256.0 * normalized)))
        hex_byte = f"{bin_value:02X}"

        normalized_values.append(float(normalized))
        bins.append(bin_value)
        hex_bytes.append(hex_byte)

    raw_hex = "".join(hex_bytes)
    return {
        "normalized": normalized_values,
        "bins": bins,
        "hex_bytes": hex_bytes,
        "raw": raw_hex,
        "display": format_kuid(raw_hex),
        "cluster_display": format_kuid_cluster(raw_hex),
    }
