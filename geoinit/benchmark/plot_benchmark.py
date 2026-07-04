"""Plotting script for GeoInit benchmarking results.

Supports both V1.1/V0.2/V0.3 summary bar charts and V0.4 casewise boxplots
and statistical distribution charts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def generate_plots(csv_path: str | None = None) -> None:
    if csv_path is None:
        # Default to V0.7 if it exists, otherwise fall back to V0.6, V0.5, V0.4, etc.
        if Path("outputs/geoinit_v0_7_casewise_results.csv").is_file():
            csv_path = "outputs/geoinit_v0_7_casewise_results.csv"
        elif Path("outputs/geoinit_v0_6_casewise_results.csv").is_file():
            csv_path = "outputs/geoinit_v0_6_casewise_results.csv"
        elif Path("outputs/geoinit_v0_5_casewise_results.csv").is_file():
            csv_path = "outputs/geoinit_v0_5_casewise_results.csv"
        elif Path("outputs/geoinit_v0_4_casewise_results.csv").is_file():
            csv_path = "outputs/geoinit_v0_4_casewise_results.csv"
        elif Path("outputs/geoinit_v0_3_xtb_summary.csv").is_file():
            csv_path = "outputs/geoinit_v0_3_xtb_summary.csv"
        elif Path("outputs/geoinit_v0_2_xtb_summary.csv").is_file():
            csv_path = "outputs/geoinit_v0_2_xtb_summary.csv"
        else:
            csv_path = "outputs/geoinit_v0_1_xtb_summary.csv"

    csv_file = Path(csv_path)
    if not csv_file.is_file():
        print(f"Error: CSV file not found at {csv_path}")
        return

    out_dir = csv_file.parent

    print(f"Generating plots from: {csv_file}")
    df = pd.read_csv(str(csv_file))
    if df.empty:
        print("Error: CSV file is empty.")
        return

    # Sort data alphabetically by molecule
    df = df.sort_values(by="molecule")

    # Styling settings
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    # Detect version from file name
    base_name = csv_file.stem
    version_label = "V0.7"
    version_file_str = "v0_7"
    if "v0_6" in base_name:
        version_label = "V0.6"
        version_file_str = "v0_6"
    elif "v0_5" in base_name:
        version_label = "V0.5"
        version_file_str = "v0_5"
    elif "v0_4" in base_name:
        version_label = "V0.4"
        version_file_str = "v0_4"
    elif "v0_3" in base_name:
        version_label = "V0.3"
        version_file_str = "v0_3"
    elif "v0_2" in base_name:
        version_label = "V0.2"
        version_file_str = "v0_2"
    elif "v1_1" in base_name:
        version_label = "V1.1"
        version_file_str = "v1_1"

    # Detect if this is casewise data (contains 'distortion_idx')
    is_casewise = "distortion_idx" in df.columns

    if is_casewise:
        # ── V0.4 Hardened Benchmark Plotting ───────────────────────────
        molecules = sorted(df["molecule"].unique())
        x = np.arange(len(molecules))
        width = 0.25

        # 1. Steps Boxplot (Raw vs UFF vs GeoInit Guarded)
        fig, ax = plt.subplots(figsize=(12, 6))
        raw_data = [df[df["molecule"] == m]["raw_steps"].values for m in molecules]
        uff_data = [df[df["molecule"] == m]["uff_steps"].values for m in molecules]
        guard_data = [df[df["molecule"] == m]["geoinit_steps"].values for m in molecules]

        bp1 = ax.boxplot(raw_data, positions=x - width, widths=0.2,
                          patch_artist=True, boxprops=dict(facecolor='#e74c3c', color='black'),
                          medianprops=dict(color='black', linewidth=1.5), showfliers=False)
        bp2 = ax.boxplot(uff_data, positions=x, widths=0.2,
                          patch_artist=True, boxprops=dict(facecolor='#3498db', color='black'),
                          medianprops=dict(color='black', linewidth=1.5), showfliers=False)
        bp3 = ax.boxplot(guard_data, positions=x + width, widths=0.2,
                          patch_artist=True, boxprops=dict(facecolor='#2ecc71', color='black'),
                          medianprops=dict(color='black', linewidth=1.5), showfliers=False)

        ax.set_ylabel('xTB Optimization Steps', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(molecules, rotation=30, ha='right', fontsize=10)
        ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]],
                   ['Raw', 'UFF', f'GeoInit {version_label} Guarded'], frameon=True, facecolor='white', edgecolor='none')
        ax.set_title('xTB Steps Distribution across Distorted Configurations', fontsize=14, fontweight='bold', pad=15)
        fig.tight_layout()
        steps_out = out_dir / f"geoinit_{version_file_str}_xtb_steps_boxplot.png"
        plt.savefig(str(steps_out), dpi=300)
        plt.close()

        # 2. Total Pipeline Time Boxplot
        fig, ax = plt.subplots(figsize=(12, 6))
        raw_time = [df[df["molecule"] == m]["raw_total_time"].values for m in molecules]
        uff_time = [df[df["molecule"] == m]["uff_total_time"].values for m in molecules]
        guard_time = [df[df["molecule"] == m]["geoinit_total_time"].values for m in molecules]

        bp1 = ax.boxplot(raw_time, positions=x - width, widths=0.2,
                          patch_artist=True, boxprops=dict(facecolor='#e74c3c', color='black'),
                          medianprops=dict(color='black', linewidth=1.5), showfliers=False)
        bp2 = ax.boxplot(uff_time, positions=x, widths=0.2,
                          patch_artist=True, boxprops=dict(facecolor='#3498db', color='black'),
                          medianprops=dict(color='black', linewidth=1.5), showfliers=False)
        bp3 = ax.boxplot(guard_time, positions=x + width, widths=0.2,
                          patch_artist=True, boxprops=dict(facecolor='#2ecc71', color='black'),
                          medianprops=dict(color='black', linewidth=1.5), showfliers=False)

        ax.set_ylabel('Total Pipeline Wall Time (s)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(molecules, rotation=30, ha='right', fontsize=10)
        ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]],
                   ['Raw', 'UFF', f'GeoInit {version_label} Guarded'], frameon=True, facecolor='white', edgecolor='none')
        ax.set_title('Total Pipeline Wall Time Distribution (Prep + xTB)', fontsize=14, fontweight='bold', pad=15)
        fig.tight_layout()
        time_out = out_dir / f"geoinit_{version_file_str}_total_time_boxplot.png"
        plt.savefig(str(time_out), dpi=300)
        plt.close()

        # 3. Energy Gap Distribution Bar Chart
        categories = [
            "< 0.1 kcal/mol (same minimum)",
            "0.1 to 0.5 kcal/mol (probably acceptable)",
            "0.5 to 1.0 kcal/mol (inspect)",
            "> 1.0 kcal/mol (different basin risk)"
        ]
        def classify_energy_gap(val_kcal):
            if pd.isna(val_kcal):
                return "unknown"
            abs_val = abs(val_kcal)
            if abs_val < 0.1:
                return "< 0.1 kcal/mol (same minimum)"
            elif abs_val < 0.5:
                return "0.1 to 0.5 kcal/mol (probably acceptable)"
            elif abs_val < 1.0:
                return "0.5 to 1.0 kcal/mol (inspect)"
            else:
                return "> 1.0 kcal/mol (different basin risk)"

        df["uff_gap_cat"] = df["uff_energy_gap_kcal"].apply(classify_energy_gap)
        df["guard_gap_cat"] = df["geoinit_energy_gap_kcal"].apply(classify_energy_gap)

        uff_counts = [int((df["uff_gap_cat"] == cat).sum()) for cat in categories]
        guard_counts = [int((df["guard_gap_cat"] == cat).sum()) for cat in categories]

        fig, ax = plt.subplots(figsize=(10, 6))
        x_cats = np.arange(len(categories))
        width_cat = 0.35

        rects1 = ax.bar(x_cats - width_cat/2, uff_counts, width_cat, label='UFF', color='#3498db')
        rects2 = ax.bar(x_cats + width_cat/2, guard_counts, width_cat, label='GeoInit Guarded', color='#2ecc71')

        ax.set_ylabel('Trial Count', fontsize=12, fontweight='bold')
        ax.set_xticks(x_cats)
        ax.set_xticklabels([c.split(" (")[0] for c in categories], fontsize=10)
        ax.legend(frameon=True, facecolor='white', edgecolor='none')
        ax.set_title('Final Energy Gap Distribution (kcal/mol) relative to Raw xTB', fontsize=14, fontweight='bold', pad=15)

        # Add labels
        for rect in rects1:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f'{h}', xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        for rect in rects2:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f'{h}', xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

        fig.tight_layout()
        gap_out = out_dir / f"geoinit_{version_file_str}_energy_gap_hist.png"
        plt.savefig(str(gap_out), dpi=300)
        plt.close()

        # 4. Guard Decision Breakdown
        reasons = [
            "unsafe_single_molecule",
            "too_short_interfragment_distance",
            "too_many_interfragment_clashes",
            "multiple_bond_damage",
            "linear_fragment_damage",
            "aromatic_planarity_damage",
            "fragment_drift",
        ]
        accepted_count = int(df["geoinit_accepted"].sum())
        fallback_counts = {r: int((df["geoinit_fallback_reason"] == r).sum()) for r in reasons}

        labels_grd = ['Accepted'] + [f'Fallback: {r.replace("_", " ")}' for r in reasons]
        counts_grd = [accepted_count] + [fallback_counts[r] for r in reasons]

        # Filter out 0 counts to keep plot tidy
        filtered = [(l, c) for l, c in zip(labels_grd, counts_grd) if c > 0]
        if not filtered:
            filtered = [('No Guard Actions', 0)]
        labels_grd = [f[0] for f in filtered]
        counts_grd = [f[1] for f in filtered]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#2ecc71'] + ['#e74c3c'] * (len(labels_grd) - 1)
        bars = ax.barh(labels_grd, counts_grd, color=colors, edgecolor='none')
        ax.set_xlabel('Count of Trials', fontsize=12, fontweight='bold')
        ax.set_title('Guard Decisions & Fallback Reasons Breakdown', fontsize=14, fontweight='bold', pad=15)

        # Add labels to horizontal bars
        for bar in bars:
            val = bar.get_width()
            ax.annotate(f'{int(val)}', xy=(val, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points", ha='left', va='center', fontsize=9)

        fig.tight_layout()
        guard_out = out_dir / f"geoinit_{version_file_str}_guard_breakdown.png"
        plt.savefig(str(guard_out), dpi=300)
        plt.close()

        print("Boxplots and breakdowns generated successfully:")
        print(f"  - {steps_out}")
        print(f"  - {time_out}")
        print(f"  - {gap_out}")
        print(f"  - {guard_out}")

    else:
        # ── Older V1.1/V0.2/V0.3 Summary Plotting ──────────────────────
        labels = df["molecule"].tolist()
        raw_steps = df["raw_steps"].tolist()
        geo_steps = df["geoinit_steps"].tolist()
        raw_time = df["raw_walltime_s"].tolist()
        geo_time = df["geoinit_walltime_s"].tolist()

        has_guarded = "guarded_steps" in df.columns
        if has_guarded:
            guard_steps = df["guarded_steps"].tolist()
            guard_time = df["guarded_walltime_s"].tolist()

        x = np.arange(len(labels))
        base_name = csv_file.stem
        if base_name.endswith("_summary"):
            base_name = base_name[:-8]

        steps_out = out_dir / f"{base_name}_steps.png"

        walltime_prefix = base_name
        if walltime_prefix.endswith("_xtb"):
            walltime_prefix = walltime_prefix[:-4]
        time_out = out_dir / f"{walltime_prefix}_walltime.png"

        # Steps plot
        fig, ax = plt.subplots(figsize=(11, 6))
        if has_guarded:
            width = 0.25
            rects1 = ax.bar(x - width, raw_steps, width, label='Raw', color='#e74c3c')
            rects2 = ax.bar(x, geo_steps, width, label='GeoInit V0.2', color='#3498db')
            rects3 = ax.bar(x + width, guard_steps, width, label='GeoInit V0.3 Guarded', color='#2ecc71')
        else:
            width = 0.35
            rects1 = ax.bar(x - width/2, raw_steps, width, label='Raw', color='#e74c3c')
            rects2 = ax.bar(x + width/2, geo_steps, width, label='GeoInit', color='#2ecc71')

        ax.set_ylabel('xTB Optimization Steps', fontsize=12, fontweight='bold')
        ax.set_title(f'xTB Optimization Steps Comparison: Raw vs GeoInit ({base_name.replace("_", " ").title()})', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
        ax.legend(frameon=True, facecolor='white', edgecolor='none')

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f'{int(h)}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f'{int(h)}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        if has_guarded:
            for rect in rects3:
                h = rect.get_height()
                ax.annotate(f'{int(h)}', xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        fig.tight_layout()
        plt.savefig(str(steps_out), dpi=300)
        plt.close()

        # Wall time plot
        fig, ax = plt.subplots(figsize=(11, 6))
        if has_guarded:
            width = 0.25
            rects1 = ax.bar(x - width, raw_time, width, label='Raw', color='#e74c3c')
            rects2 = ax.bar(x, geo_time, width, label='GeoInit V0.2', color='#3498db')
            rects3 = ax.bar(x + width, guard_time, width, label='GeoInit V0.3 Guarded', color='#2ecc71')
        else:
            width = 0.35
            rects1 = ax.bar(x - width/2, raw_time, width, label='Raw', color='#e74c3c')
            rects2 = ax.bar(x + width/2, geo_time, width, label='GeoInit', color='#2ecc71')

        ax.set_ylabel('xTB Wall Time (s)', fontsize=12, fontweight='bold')
        ax.set_title(f'xTB Wall Time Comparison: Raw vs GeoInit ({base_name.replace("_", " ").title()})', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
        ax.legend(frameon=True, facecolor='white', edgecolor='none')

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        if has_guarded:
            for rect in rects3:
                h = rect.get_height()
                ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        fig.tight_layout()
        plt.savefig(str(time_out), dpi=300)
        plt.close()

        print("Plots generated successfully:")
        print(f"  - {steps_out}")
        print(f"  - {time_out}")


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    generate_plots(path_arg)
