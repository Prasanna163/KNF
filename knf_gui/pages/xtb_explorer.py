import os
import streamlit as st

from knf_core import utils
from knf_gui.components.file_uploader import render_file_uploader
from knf_gui.components.molecule_viewer import render_molecule_viewer
from knf_gui.components.progress_tracker import run_nci_from_optimized, run_xtb_with_progress
from knf_gui.components.xtb_menu import render_xtb_menu
from knf_gui.utils.state_manager import record_run


def render(settings: dict):
    st.header("xTB Explorer")
    st.caption("Interactive mode for selecting and running xTB operations.")
    st.info(
        "Context: Use this tab when you want to run specific xTB calculations manually, inspect commands, "
        "and learn outputs before/without running the full KNF automation."
    )
    with st.expander("How to use xTB Explorer effectively"):
        st.markdown(
            "1. Prepare molecule.\n"
            "2. Choose xTB category and operation.\n"
            "3. Review command preview and run.\n"
            "4. If optimization succeeds, continue to NCI analysis optionally."
        )

    context = st.session_state.get("prepared_context")
    prepared = render_file_uploader()
    if prepared:
        st.session_state["prepared_context"] = prepared
        context = prepared
    if not context:
        st.info("Prepare a molecule to continue.")
        return

    work_xyz = os.path.join(context["results_dir"], "input.xyz")
    if os.path.abspath(context["target_xyz"]) != os.path.abspath(work_xyz):
        utils.safe_copy(context["target_xyz"], work_xyz)

    c1, c2 = st.columns([2, 1])
    with c1:
        render_molecule_viewer(context["target_xyz"], height=360)
    with c2:
        st.metric("Molecule", context["base_name"])
        st.metric("Fragments", len(context["fragments"]))
        st.metric("Charge", settings["charge"])
        st.metric("UHF", settings["spin"] - 1)

    option, flags = render_xtb_menu()
    command_preview = ["xtb", os.path.basename(work_xyz)] + flags + [
        "--charge",
        str(settings["charge"]),
        "--uhf",
        str(settings["spin"] - 1),
    ]
    st.code(" ".join(command_preview), language="bash")

    if st.button(f"Run: {option}", type="primary", use_container_width=True):
        try:
            log_name = "xtb_opt.log" if "--opt" in flags else "xtb_explorer.log"
            cmd, log_path = run_xtb_with_progress(
                work_xyz, settings["charge"], settings["spin"], flags, log_name
            )
            st.session_state["last_cmd"] = cmd
            st.session_state["last_xtb_log"] = log_path
            record_run(context["base_name"], "Explorer", "success", None)
            if "--opt" in flags:
                opt = os.path.join(context["results_dir"], "xtbopt.xyz")
                if os.path.exists(opt):
                    st.session_state["optimized_xyz"] = opt
            st.success("xTB command completed.")
        except Exception as e:
            record_run(context["base_name"], "Explorer", "failed", None)
            st.error(f"xTB command failed: {e}")

    log_path = st.session_state.get("last_xtb_log")
    if log_path and os.path.exists(log_path):
        with st.expander("View latest xTB log"):
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                st.code(f.read(50000), language="text")

    optimized_xyz = st.session_state.get("optimized_xyz")
    if optimized_xyz and os.path.exists(optimized_xyz):
        st.success(f"Optimization available: {optimized_xyz}")
        if st.button("Continue to NCI Analysis", type="primary"):
            try:
                nci_file = run_nci_from_optimized(
                    context["results_dir"], optimized_xyz, settings["charge"], settings["spin"]
                )
                st.success(f"NCI complete: {nci_file}")
            except Exception as e:
                st.error(f"NCI step failed: {e}")
