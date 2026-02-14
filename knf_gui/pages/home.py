import streamlit as st

from knf_gui.utils.formatters import format_seconds
from knf_gui.utils.state_manager import summary_metrics


def render(settings: dict):
    del settings
    st.header("KNF-CORE: Molecular Descriptor Engine")
    st.caption("Automated supramolecular stability analysis using xTB and Multiwfn")
    st.info(
        "Use this dashboard to choose your workflow. "
        "Start with Quick Analysis for full KNF output, use xTB Explorer for method learning, "
        "and use Custom Workflow for chained calculations."
    )

    m = summary_metrics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Analyses", m["total"])
    c2.metric("Success Rate", f"{m['success_rate']:.1f}%")
    c3.metric("Average Time", format_seconds(m["avg_time"]))
    c4.metric("Last Run", m["last_run"])

    st.markdown("### Getting Started")
    a, b, c = st.columns(3)
    with a:
        st.container(border=True).markdown(
            "**Quick Analysis**\n\nUpload -> Run -> Results in 3 clicks."
        )
    with b:
        st.container(border=True).markdown(
            "**xTB Explorer**\n\nLearn and run xTB operations interactively."
        )
    with c:
        st.container(border=True).markdown(
            "**Custom Workflow**\n\nBuild sequential calculation pipelines."
        )

    st.markdown("### Recommended Navigation")
    st.markdown("1. **Quick Start**: Upload and validate molecule.")
    st.markdown("2. **xTB Explorer**: Test/learn specific xTB operations.")
    st.markdown("3. **Results Viewer**: Inspect generated files and KNF outputs.")

    st.markdown("### Recent Activity")
    history = st.session_state.get("analysis_history", [])
    if not history:
        st.info("No analyses recorded yet.")
    else:
        st.dataframe(history[-5:], use_container_width=True)
