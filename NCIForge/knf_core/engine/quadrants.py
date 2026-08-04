import json
import os
import statistics

from .kuid_ops import _normalize_minmax, _safe_float
from .naming import _final_output_name


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
        from matplotlib.patches import Rectangle

        fig, ax = plt.subplots(figsize=(11, 7), dpi=170)
        quadrant_colors = {
            "Q1": "#1d4ed8",  # blue
            "Q2": "#ea580c",  # orange
            "Q3": "#15803d",  # green
            "Q4": "#be123c",  # red
        }
        quadrant_bg = {
            "Q1": "#dbeafe",
            "Q2": "#ffedd5",
            "Q3": "#dcfce7",
            "Q4": "#ffe4e6",
        }

        x_min = min(snci_norm_vals)
        x_max = max(snci_norm_vals)
        y_min = min(scdi_norm_vals)
        y_max = max(scdi_norm_vals)
        x_pad = max(0.03, (x_max - x_min) * 0.08) if x_max > x_min else 0.05
        y_pad = max(0.03, (y_max - y_min) * 0.08) if y_max > y_min else 0.05
        x_min = max(0.0, x_min - x_pad)
        x_max = min(1.0, x_max + x_pad)
        y_min = max(0.0, y_min - y_pad)
        y_max = min(1.0, y_max + y_pad)
        if x_max <= x_min:
            x_min, x_max = 0.0, 1.0
        if y_max <= y_min:
            y_min, y_max = 0.0, 1.0

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_facecolor("#f8fafc")

        quadrant_rectangles = {
            "Q2": (x_min, median_y, max(0.0, median_x - x_min), max(0.0, y_max - median_y)),
            "Q1": (median_x, median_y, max(0.0, x_max - median_x), max(0.0, y_max - median_y)),
            "Q3": (x_min, y_min, max(0.0, median_x - x_min), max(0.0, median_y - y_min)),
            "Q4": (median_x, y_min, max(0.0, x_max - median_x), max(0.0, median_y - y_min)),
        }
        for quadrant, (x0, y0, w, h) in quadrant_rectangles.items():
            if w > 0 and h > 0:
                ax.add_patch(
                    Rectangle(
                        (x0, y0),
                        w,
                        h,
                        facecolor=quadrant_bg[quadrant],
                        edgecolor="none",
                        alpha=0.35,
                        zorder=0,
                    )
                )

        for quadrant in ("Q1", "Q2", "Q3", "Q4"):
            q_rows = [row for row in valid_plot_rows if row["entry"].get("quadrant") == quadrant]
            if not q_rows:
                continue
            ax.scatter(
                [row["snci_norm"] for row in q_rows],
                [row["scdi_norm"] for row in q_rows],
                s=12,
                c=quadrant_colors[quadrant],
                alpha=0.85,
                edgecolors="white",
                linewidths=0.25,
                label=f"{quadrant} (n={quadrants[quadrant]['count']})",
                zorder=3,
            )

        ax.axvline(
            median_x,
            color="#334155",
            linestyle="--",
            linewidth=1.6,
            label=f"Median SNCI_Norm = {median_x:.4f}",
            zorder=4,
        )
        ax.axhline(
            median_y,
            color="#0f766e",
            linestyle="--",
            linewidth=1.6,
            label=f"Median SCDI_Norm = {median_y:.4f}",
            zorder=4,
        )
        ax.set_xlabel("SNCI_Norm")
        ax.set_ylabel("SCDI_Norm")
        ax.set_title("SNCI-SCDI Quadrant Map")
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.45, zorder=1)

        ax.legend(loc="lower right", frameon=True, framealpha=0.9)
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
