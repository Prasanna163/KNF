import streamlit as st
from knf_gui.components.results_display import render_results_browser


def render(settings: dict):
    del settings
    st.header("Results Viewer")
    context = st.session_state.get("prepared_context")
    if not context:
        st.info("Prepare a molecule first to browse results.")
        return
    render_results_browser(context["results_dir"])

