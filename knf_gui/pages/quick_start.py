import os
import streamlit as st

from knf_core import utils
from knf_gui.components.file_uploader import render_file_uploader
from knf_gui.components.molecule_viewer import render_molecule_viewer
from knf_gui.components.progress_tracker import run_pipeline_with_progress
from knf_gui.components.results_display import render_knf_results
from knf_gui.utils.state_manager import record_run


def render(settings: dict):
    st.header("Quick Analysis")
    st.caption("Upload -> Configure -> Run automated KNF pipeline.")
    st.info(
        "Context: This mode runs the full KNF pipeline (conversion, fragment handling, xTB, Multiwfn, "
        "SNCI/SCDI, and KNF vector generation) with minimal manual input."
    )
    with st.expander("What you should expect from this run"):
        st.markdown(
            "- Best for end-to-end descriptor generation.\n"
            "- Output folder includes `knf.json`, `output.txt`, geometry, and intermediate logs.\n"
            "- Runtime depends on molecule size and xTB/Multiwfn steps."
        )

    context = st.session_state.get("prepared_context")
    prepared = render_file_uploader()
    if prepared:
        st.session_state["prepared_context"] = prepared
        context = prepared

    if not context:
        st.info("Prepare a molecule to continue.")
        return

    st.markdown("### Step 2: Preview & Validate")
    left, right = st.columns([2, 1])
    with left:
        render_molecule_viewer(context["target_xyz"], height=420)
    with right:
        st.metric("Atoms", context["atom_count"])
        st.metric("Fragments", len(context["fragments"]))
        st.metric("Charge", settings["charge"])
        st.metric("Spin", settings["spin"])
        st.caption(f"Results directory: `{context['results_dir']}`")

    st.markdown("### Step 3: Run Analysis")
    if st.button("Start Automated Pipeline", type="primary", use_container_width=True):
        ok, elapsed = run_pipeline_with_progress(context, settings)
        record_run(context["base_name"], "Automated", "success" if ok else "failed", elapsed)

    render_knf_results(context["results_dir"])

    work_xyz = os.path.join(context["results_dir"], "input.xyz")
    if os.path.abspath(context["target_xyz"]) != os.path.abspath(work_xyz) and not os.path.exists(work_xyz):
        utils.safe_copy(context["target_xyz"], work_xyz)
