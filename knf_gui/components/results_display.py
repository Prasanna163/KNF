import os
import json
from pathlib import Path

import plotly.express as px
import pandas as pd
import streamlit as st

from knf_gui.utils.export import json_bytes, zip_directory


def _read_text(path: str, max_chars: int = 100000):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read(max_chars)


def render_knf_results(results_dir: str):
    knf_json = os.path.join(results_dir, "knf.json")
    output_txt = os.path.join(results_dir, "output.txt")
    nci_grid = os.path.join(results_dir, "nci_grid.txt")

    if not os.path.exists(knf_json):
        st.info("No KNF results available yet.")
        return

    with open(knf_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    tabs = st.tabs(["KNF Vector", "Visualizations", "Full Report", "Downloads", "Re-run"])

    with tabs[0]:
        vector = payload.get("KNF_vector", [])
        labels = [f"f{i}" for i in range(1, len(vector) + 1)]
        df = pd.DataFrame({"Descriptor": labels, "Value": vector})
        fig = px.bar(df, x="Descriptor", y="Value", title="9D KNF Vector", color="Value")
        st.plotly_chart(fig, use_container_width=True)
        st.json(payload)

    with tabs[1]:
        st.markdown("NCI and descriptor visual outputs:")
        if os.path.exists(nci_grid):
            st.success(f"NCI grid found: `{nci_grid}`")
        else:
            st.info("NCI grid not found yet.")

    with tabs[2]:
        if os.path.exists(output_txt):
            st.code(_read_text(output_txt) or "", language="text")
        else:
            st.info("No output.txt available.")

    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download KNF JSON",
                data=json_bytes(payload),
                file_name="knf_results.json",
                mime="application/json",
            )
        with c2:
            st.download_button(
                "Download All Results (ZIP)",
                data=zip_directory(results_dir),
                file_name="knf_results.zip",
                mime="application/zip",
            )

    with tabs[4]:
        st.caption("Adjust settings in sidebar, then rerun from Quick Start.")


def render_results_browser(results_dir: str):
    st.subheader("Results Browser")
    st.caption(f"Directory: `{results_dir}`")
    if not os.path.exists(results_dir):
        st.info("No results directory found.")
        return
    files = sorted(Path(results_dir).glob("*"))
    if not files:
        st.info("No files available.")
        return
    selected = st.selectbox("Preview file", [f.name for f in files])
    full = os.path.join(results_dir, selected)
    if selected.lower().endswith(".json"):
        with open(full, "r", encoding="utf-8") as f:
            st.json(json.load(f))
    else:
        st.code(_read_text(full) or "", language="text")

