from __future__ import annotations

import math
from collections import defaultdict

_HEX = set("0123456789ABCDEF")
_PREFIX_SPECS = (
    ("KUID_prefix2", 2),  # 1 byte
    ("KUID_prefix4", 4),  # 2 bytes
    ("KUID_prefix6", 6),  # 3 bytes
)


def normalize_kuid_raw(value) -> str:
    if value is None:
        return ""
    raw = "".join(ch for ch in str(value).upper() if ch in _HEX)
    # KUID is byte-addressable; keep full bytes only.
    if len(raw) % 2 != 0:
        raw = raw[:-1]
    return raw


def kuid_prefix_fields(kuid_raw: str) -> dict[str, str]:
    raw = normalize_kuid_raw(kuid_raw)
    out = {}
    for name, nchars in _PREFIX_SPECS:
        out[name] = raw[:nchars] if len(raw) >= nchars else ""
    return out


def byte_hamming_distance(kuid_a: str, kuid_b: str) -> int:
    a = normalize_kuid_raw(kuid_a)
    b = normalize_kuid_raw(kuid_b)
    nbytes_a = len(a) // 2
    nbytes_b = len(b) // 2
    overlap = min(nbytes_a, nbytes_b)
    dist = 0
    for idx in range(overlap):
        sa = a[2 * idx: 2 * idx + 2]
        sb = b[2 * idx: 2 * idx + 2]
        if sa != sb:
            dist += 1
    dist += abs(nbytes_a - nbytes_b)
    return dist


def nearest_neighbors(
    query_kuid: str,
    candidates: list[tuple[str, str]],
    top_k: int = 20,
) -> list[dict]:
    query = normalize_kuid_raw(query_kuid)
    scored = []
    for item_id, code in candidates:
        code_norm = normalize_kuid_raw(code)
        if not code_norm:
            continue
        scored.append(
            {
                "id": item_id,
                "kuid": code_norm,
                "distance": byte_hamming_distance(query, code_norm),
            }
        )
    scored.sort(key=lambda item: (item["distance"], item["id"]))
    return scored[: max(0, int(top_k))]


def _safe_float(value):
    try:
        if value is None:
            return None
        val = float(value)
        if not math.isfinite(val):
            return None
        return val
    except (TypeError, ValueError):
        return None


def build_prefix_index(rows: list[dict], code_field: str = "KUID") -> dict:
    buckets = {
        "prefix2": defaultdict(list),
        "prefix4": defaultdict(list),
        "prefix6": defaultdict(list),
    }
    for row in rows:
        code = row.get(code_field) or row.get("KUID_raw")
        raw = normalize_kuid_raw(code)
        if not raw:
            continue
        file_name = (
            row.get("File")
            or row.get("file")
            or row.get("input_file_name")
            or row.get("input_file")
            or ""
        )
        source_batch = row.get("source_batch", "")
        ref = {"file": file_name}
        if source_batch:
            ref["source_batch"] = source_batch

        if len(raw) >= 2:
            buckets["prefix2"][raw[:2]].append(ref)
        if len(raw) >= 4:
            buckets["prefix4"][raw[:4]].append(ref)
        if len(raw) >= 6:
            buckets["prefix6"][raw[:6]].append(ref)

    out = {}
    for key, mapping in buckets.items():
        out[key] = {prefix: mapping[prefix] for prefix in sorted(mapping)}
    return out


def build_family_stats(rows: list[dict], code_field: str = "KUID") -> list[dict]:
    numeric_fields = [
        "SNCI",
        "SCDI",
        "SCDI_variance",
        "SNCI_Norm",
        "SCDI_Norm",
    ] + [f"f{i}" for i in range(1, 10)]

    grouped = {}
    for row in rows:
        raw = normalize_kuid_raw(row.get(code_field) or row.get("KUID_raw"))
        if not raw:
            continue
        g = grouped.setdefault(
            raw,
            {
                "kuid": raw,
                "member_count": 0,
                "files": [],
                "sum": {field: 0.0 for field in numeric_fields},
                "n": {field: 0 for field in numeric_fields},
            },
        )
        g["member_count"] += 1
        file_name = (
            row.get("File")
            or row.get("file")
            or row.get("input_file_name")
            or row.get("input_file")
            or ""
        )
        if file_name:
            g["files"].append(file_name)

        for field in numeric_fields:
            val = _safe_float(row.get(field))
            if val is None:
                continue
            g["sum"][field] += float(val)
            g["n"][field] += 1

    out = []
    for raw, g in grouped.items():
        family = {
            "kuid": raw,
            "member_count": g["member_count"],
            "example_files": g["files"][:5],
        }
        family.update(kuid_prefix_fields(raw))
        for field in numeric_fields:
            count = g["n"][field]
            family[f"mean_{field}"] = (g["sum"][field] / count) if count > 0 else None
        out.append(family)

    out.sort(key=lambda item: (-item["member_count"], item["kuid"]))
    return out
