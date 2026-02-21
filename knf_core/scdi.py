import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SCDIMetrics:
    variance: float
    scdi: Optional[float]


def _parse_env_float(env_name: str) -> Optional[float]:
    value = os.getenv(env_name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        logging.warning("Ignoring invalid %s=%r (must be float).", env_name, value)
        return None


def _resolve_bounds(
    var_min: Optional[float],
    var_max: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    if var_min is None:
        var_min = _parse_env_float("KNF_SCDI_VAR_MIN")
    if var_max is None:
        var_max = _parse_env_float("KNF_SCDI_VAR_MAX")
    return var_min, var_max


def _extract_segment_rows(cosmo_path: str) -> list[list[float]]:
    rows: list[list[float]] = []
    with open(cosmo_path, "r", encoding="utf-8", errors="replace") as f:
        in_segment = False
        for raw_line in f:
            line = raw_line.strip()
            if not in_segment:
                if line == "$segment_information":
                    in_segment = True
                continue

            if line.startswith("$"):
                break
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                rows.append([float(token) for token in parts])
            except ValueError:
                continue
    return rows


def _select_columns(rows: list[list[float]]) -> tuple[int, int]:
    # Candidate mappings in (area_index, charge_index).
    candidates = [(6, 5), (5, 6), (7, 5), (5, 7), (6, 7), (7, 6)]
    best_score: Optional[tuple[float, float, float, float]] = None
    best_pair: Optional[tuple[int, int]] = None

    for area_idx, charge_idx in candidates:
        if any(len(row) <= max(area_idx, charge_idx) for row in rows):
            continue

        area_vals = np.array([row[area_idx] for row in rows], dtype=float)
        charge_vals = np.array([row[charge_idx] for row in rows], dtype=float)

        positive_ratio = float(np.mean(area_vals > 0.0))
        mean_area = float(np.mean(area_vals))
        mean_abs_charge = float(np.mean(np.abs(charge_vals)))

        # Valid area should be mostly positive; charge typically has smaller magnitude.
        if positive_ratio < 0.95:
            continue
        if mean_area <= 0.0:
            continue

        # Higher score is better.
        score = (positive_ratio, mean_area, -mean_abs_charge, -abs(area_idx - charge_idx))
        if best_score is None or score > best_score:
            best_score = score
            best_pair = (area_idx, charge_idx)

    if best_pair is not None:
        return best_pair

    # Fallback to xTB standard COSMO ordering.
    return 6, 5


def _compute_var_a(areas: np.ndarray, charges: np.ndarray) -> float:
    total_area = float(np.sum(areas))
    if total_area <= 0.0:
        return 0.0
    mu_a = float(np.sum(areas * charges) / total_area)
    var_a = float(np.sum(areas * (charges - mu_a) ** 2) / total_area)
    return max(0.0, var_a)


def _normalize_scdi(var_a: float, var_min: float, var_max: float) -> Optional[float]:
    if var_max <= var_min:
        logging.warning(
            "Invalid SCDI normalization bounds: var_min=%s var_max=%s",
            var_min,
            var_max,
        )
        return None

    scdi = 1.0 - ((var_a - var_min) / (var_max - var_min))
    # Keep normalized descriptor bounded.
    return float(np.clip(scdi, 0.0, 1.0))


def compute_scdi_metrics(
    cosmo_path: str,
    var_min: Optional[float] = None,
    var_max: Optional[float] = None,
) -> SCDIMetrics:
    """
    Computes:
      VarA(Q) = sum(a_i * (q_i - mu_A)^2) / sum(a_i)
      SCDI    = 1 - (VarA - var_min) / (var_max - var_min)  (if bounds provided)
    """
    if not os.path.exists(cosmo_path):
        logging.warning("COSMO file not found: %s", cosmo_path)
        return SCDIMetrics(variance=0.0, scdi=None)

    rows = _extract_segment_rows(cosmo_path)
    if not rows:
        return SCDIMetrics(variance=0.0, scdi=None)

    area_idx, charge_idx = _select_columns(rows)
    areas = np.array([row[area_idx] for row in rows], dtype=float)
    charges = np.array([row[charge_idx] for row in rows], dtype=float)

    # Keep only physically valid area weights.
    mask = np.isfinite(areas) & np.isfinite(charges) & (areas > 0.0)
    if not np.any(mask):
        return SCDIMetrics(variance=0.0, scdi=None)

    var_a = _compute_var_a(areas[mask], charges[mask])
    resolved_min, resolved_max = _resolve_bounds(var_min, var_max)

    scdi_value: Optional[float] = None
    if resolved_min is not None and resolved_max is not None:
        scdi_value = _normalize_scdi(var_a, resolved_min, resolved_max)
    elif resolved_min is not None or resolved_max is not None:
        logging.warning(
            "Only one SCDI bound provided; both var_min and var_max are required for normalization."
        )

    return SCDIMetrics(variance=var_a, scdi=scdi_value)


def compute_scdi(
    cosmo_path: str,
    var_min: Optional[float] = None,
    var_max: Optional[float] = None,
) -> float:
    """
    Backward-compatible helper.
    Returns normalized SCDI when bounds are available; otherwise returns raw VarA.
    """
    metrics = compute_scdi_metrics(cosmo_path, var_min=var_min, var_max=var_max)
    if metrics.scdi is not None:
        return metrics.scdi
    return metrics.variance

