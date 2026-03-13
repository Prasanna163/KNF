import math
from datetime import datetime, timezone
from typing import Iterable

KUID_INTENSIVE_VERSION = "KUID-Intensive-1.0"
NORMALIZATION = "minmax"
BINS_PER_FEATURE = 16
FEATURE_ORDER = ["f3", "f4", "f7", "f8", "f9"]
DISPLAY_FORMAT = "X-X-X-X-X"
CLUSTER_DISPLAY_FORMAT = "f3f4f7-f8f9"
HEX_ALPHABET = "0123456789ABCDEF"

_KNF_INDEX = {
    "f1": 0,
    "f2": 1,
    "f3": 2,
    "f4": 3,
    "f5": 4,
    "f6": 5,
    "f7": 6,
    "f8": 7,
    "f9": 8,
}


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


def format_kuid_intensive(raw_hex: str) -> str:
    expected_len = len(FEATURE_ORDER)
    if len(raw_hex) != expected_len:
        raise ValueError(
            f"KUID-Intensive raw hex must have length {expected_len}, got {len(raw_hex)}."
        )
    return "-".join(list(raw_hex))


def format_kuid_intensive_cluster(raw_hex: str) -> str:
    expected_len = len(FEATURE_ORDER)
    if len(raw_hex) != expected_len:
        raise ValueError(
            f"KUID-Intensive raw hex must have length {expected_len}, got {len(raw_hex)}."
        )
    return f"{raw_hex[:3]}-{raw_hex[3:]}"


def _extract_from_knf_vector(knf_vector: Iterable[float]):
    values = list(knf_vector)
    if len(values) != 9:
        return None

    selected = []
    for feature in FEATURE_ORDER:
        val = _safe_float(values[_KNF_INDEX[feature]])
        if val is None:
            return None
        selected.append(val)
    return selected


def _extract_from_feature_map(feature_map: dict):
    out = []
    for feature in FEATURE_ORDER:
        val = _safe_float(feature_map.get(feature))
        if val is None:
            return None
        out.append(val)
    return out


def build_calibration_from_knf_vectors(
    knf_vectors: Iterable[Iterable[float]],
    calibration_id: str = None,
) -> dict:
    vectors = []
    for vector in knf_vectors:
        selected = _extract_from_knf_vector(vector)
        if selected is None:
            continue
        vectors.append(selected)
    return _build_calibration_from_selected_vectors(vectors, calibration_id=calibration_id)


def build_calibration_from_feature_maps(
    feature_maps: Iterable[dict],
    calibration_id: str = None,
) -> dict:
    vectors = []
    for feature_map in feature_maps:
        selected = _extract_from_feature_map(feature_map)
        if selected is None:
            continue
        vectors.append(selected)
    return _build_calibration_from_selected_vectors(vectors, calibration_id=calibration_id)


def encode_knf_vector(knf_vector: Iterable[float], calibration: dict) -> dict:
    selected = _extract_from_knf_vector(knf_vector)
    if selected is None:
        raise ValueError("KNF vector must contain numeric f3,f4,f7,f8,f9 values.")
    return _encode_selected(selected, calibration)


def encode_feature_map(feature_map: dict, calibration: dict) -> dict:
    selected = _extract_from_feature_map(feature_map)
    if selected is None:
        raise ValueError("Feature map must contain numeric f3,f4,f7,f8,f9 values.")
    return _encode_selected(selected, calibration)


def _encode_selected(selected_values: list[float], calibration: dict) -> dict:
    bins_per_feature = int(calibration.get("bins_per_feature", BINS_PER_FEATURE))
    if bins_per_feature != 16:
        raise ValueError("KUID-Intensive expects bins_per_feature=16.")

    feature_order = calibration.get("feature_order") or FEATURE_ORDER
    if list(feature_order) != FEATURE_ORDER:
        raise ValueError("KUID-Intensive expects canonical feature order f3,f4,f7,f8,f9.")

    bounds = calibration.get("feature_bounds") or {}
    normalized = []
    bins = []
    hex_digits = []

    for idx, feature in enumerate(FEATURE_ORDER):
        fmin = _safe_float((bounds.get(feature) or {}).get("min"))
        fmax = _safe_float((bounds.get(feature) or {}).get("max"))
        if fmin is None or fmax is None:
            raise ValueError(f"Missing calibration bounds for {feature}.")

        if abs(fmax - fmin) <= 1e-12:
            x = 0.0
        else:
            x = _clip01((selected_values[idx] - fmin) / (fmax - fmin))

        b = min(15, int(math.floor(16.0 * x)))
        normalized.append(float(x))
        bins.append(b)
        hex_digits.append(HEX_ALPHABET[b])

    raw_hex = "".join(hex_digits)
    return {
        "features": {
            feature: selected_values[idx]
            for idx, feature in enumerate(FEATURE_ORDER)
        },
        "normalized": normalized,
        "bins": bins,
        "hex_digits": hex_digits,
        "raw": raw_hex,
        "display": format_kuid_intensive(raw_hex),
        "cluster_display": format_kuid_intensive_cluster(raw_hex),
    }


def _build_calibration_from_selected_vectors(
    selected_vectors: list[list[float]],
    calibration_id: str = None,
) -> dict:
    if not selected_vectors:
        raise ValueError("Cannot build KUID-Intensive calibration: no valid feature rows provided.")

    bounds = {}
    for idx, feature_name in enumerate(FEATURE_ORDER):
        vals = [row[idx] for row in selected_vectors]
        bounds[feature_name] = {
            "min": float(min(vals)),
            "max": float(max(vals)),
        }

    calibration_value = calibration_id
    if not calibration_value:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        calibration_value = f"{KUID_INTENSIVE_VERSION}-{ts}"

    return {
        "kuid_intensive_version": KUID_INTENSIVE_VERSION,
        "calibration_id": calibration_value,
        "normalization": NORMALIZATION,
        "bins_per_feature": BINS_PER_FEATURE,
        "feature_order": list(FEATURE_ORDER),
        "display_format": DISPLAY_FORMAT,
        "cluster_display_format": CLUSTER_DISPLAY_FORMAT,
        "feature_bounds": bounds,
    }
