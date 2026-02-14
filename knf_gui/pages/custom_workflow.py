import os
import streamlit as st

from knf_core import utils
from knf_gui.components.file_uploader import render_file_uploader
from knf_gui.components.progress_tracker import run_xtb_with_progress


WORKFLOW_FLAGS = {
    "Geometry Optimization": ["--opt"],
    "Single Point Energy": [],
    "Frequency Calculation": ["--hess"],
    "Fukui Indices": ["--vfukui"],
    "Population Analysis": ["--pop"],
    "Bond Order Analysis": ["--wbo"],
}


def render(settings: dict):
    st.header("Custom Workflow Builder")
    st.caption("Chain multiple xTB operations and execute them sequentially.")

    context = st.session_state.get("prepared_context")
    prepared = render_file_uploader()
    if prepared:
        st.session_state["prepared_context"] = prepared
        context = prepared
    if not context:
        st.info("Prepare a molecule to continue.")
        return

    selected = st.multiselect("Select workflow steps (in order)", list(WORKFLOW_FLAGS.keys()))
    if not selected:
        st.info("Choose at least one workflow step.")
        return

    st.code(" -> ".join(selected), language="text")
    est = max(2, len(selected) * 2)
    st.caption(f"Estimated runtime: {est}-{est * 2} minutes")

    if st.button("Execute Workflow", type="primary"):
        work_xyz = os.path.join(context["results_dir"], "input.xyz")
        if os.path.abspath(context["target_xyz"]) != os.path.abspath(work_xyz):
            utils.safe_copy(context["target_xyz"], work_xyz)

        results = []
        for idx, step in enumerate(selected, start=1):
            st.markdown(f"**Step {idx}/{len(selected)}: {step}**")
            try:
                _, log_path = run_xtb_with_progress(
                    work_xyz,
                    settings["charge"],
                    settings["spin"],
                    WORKFLOW_FLAGS[step],
                    f"workflow_{idx}.log",
                )
                results.append({"step": step, "status": "success", "log": log_path})
            except Exception as e:
                results.append({"step": step, "status": "failed", "error": str(e)})
                st.error(f"Workflow stopped at '{step}': {e}")
                break
        st.session_state["workflow_results"] = results

    if st.session_state.get("workflow_results"):
        st.markdown("### Workflow Results")
        for item in st.session_state["workflow_results"]:
            if item["status"] == "success":
                st.success(f"{item['step']} completed. Log: {item['log']}")
            else:
                st.error(f"{item['step']} failed: {item['error']}")

