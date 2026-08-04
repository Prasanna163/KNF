"""Reporting helpers for GeoInit candidate and benchmark outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from geoinit.optimize.selector import SelectionResult


def selection_result_rows(selection: SelectionResult, case_name: str = "") -> list[dict]:
    """Flatten a selection result into one row per candidate."""
    rows: list[dict] = []
    for candidate in selection.candidates:
        rows.append(candidate.to_report_row(case_name=case_name, selected=candidate.name == selection.selected_name))
    return rows


def summarize_selection_rows(rows: Iterable[dict]) -> dict:
    """Return lightweight aggregate counts for candidate selection rows."""
    rows = list(rows)
    candidate_counts = Counter(row["candidate"] for row in rows)
    selected_counts = Counter(row["candidate"] for row in rows if row.get("selected"))
    rejection_counts = Counter(row["rejection_reason"] for row in rows if row.get("rejection_reason"))
    return {
        "n_rows": len(rows),
        "candidate_counts": dict(candidate_counts),
        "selected_counts": dict(selected_counts),
        "rejection_counts": dict(rejection_counts),
    }


def classwise_summary(rows: Iterable[dict], class_key: str = "chemical_class") -> pd.DataFrame:
    """Build a class-wise table when benchmark rows include class labels."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(class_key, "unknown"))].append(row)

    out_rows: list[dict] = []
    for label, group in grouped.items():
        accepted = [row for row in group if row.get("accepted")]
        selected = [row for row in group if row.get("selected")]
        out_rows.append(
            {
                "chemical_class": label,
                "trials": len(group),
                "accepted": len(accepted),
                "selected": len(selected),
                "main_fallback": Counter(row.get("rejection_reason", "") for row in group).most_common(1)[0][0] if group else "",
            }
        )
    return pd.DataFrame(out_rows)


def candidate_classwise_summary(rows: Iterable[dict]) -> pd.DataFrame:
    """Summarize selected-candidate outcomes by candidate and chemical class."""
    df = pd.DataFrame(list(rows))
    columns = [
        "candidate_name",
        "chemical_class",
        "accepted_count",
        "same_min_count",
        "net_win_count",
        "fallback_count",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    if "candidate_name" not in df.columns and "geoinit_select_candidate" in df.columns:
        df["candidate_name"] = df["geoinit_select_candidate"]
    if "chemical_class" not in df.columns:
        df["chemical_class"] = "unknown"

    out_rows: list[dict] = []
    for (candidate_name, chemical_class), group in df.groupby(["candidate_name", "chemical_class"], dropna=False):
        accepted = group.get("geoinit_select_accepted", pd.Series([False] * len(group), index=group.index))
        same_min = group.get("geoinit_select_same_min", pd.Series([False] * len(group), index=group.index))
        benefit = group.get(
            "geoinit_select_net_benefit_category",
            pd.Series([""] * len(group), index=group.index),
        )
        out_rows.append(
            {
                "candidate_name": candidate_name,
                "chemical_class": chemical_class,
                "accepted_count": int(accepted.fillna(False).sum()),
                "same_min_count": int(same_min.fillna(False).sum()),
                "net_win_count": int((benefit == "net_win").sum()),
                "fallback_count": int((~accepted.fillna(False)).sum()),
            }
        )
    return pd.DataFrame(out_rows, columns=columns)


def write_selection_report(rows: Iterable[dict], out_path: str | Path) -> None:
    """Write candidate-level rows and a small summary CSV next to them."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    df.to_csv(out_path, index=False)
    summary = summarize_selection_rows(df.to_dict("records"))
    summary_rows = []
    for section, values in summary.items():
        if isinstance(values, dict):
            for key, value in values.items():
                summary_rows.append({"section": section, "key": key, "value": value})
        else:
            summary_rows.append({"section": "overall", "key": section, "value": values})
    pd.DataFrame(summary_rows).to_csv(out_path.with_name(out_path.stem + "_summary.csv"), index=False)


__all__ = [
    "candidate_classwise_summary",
    "classwise_summary",
    "selection_result_rows",
    "summarize_selection_rows",
    "write_selection_report",
]
