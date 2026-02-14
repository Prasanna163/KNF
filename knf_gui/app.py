import os
from pathlib import Path

import streamlit as st

from knf_gui.components.settings_panel import render_settings_panel
from knf_gui.pages import home, quick_start, xtb_explorer, custom_workflow, results_viewer, documentation
from knf_gui.utils.state_manager import clear_session, init_state


def _load_css():
    css_path = Path(__file__).resolve().parent / "assets" / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="KNF-CORE GUI", layout="wide")
    _load_css()
    init_state()

    settings = render_settings_panel()

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Go to",
            [
                "Home",
                "Quick Start",
                "xTB Explorer",
                "Custom Workflow",
                "Results Viewer",
                "Documentation",
            ],
            label_visibility="collapsed",
        )
        if st.button("Clear Session", use_container_width=True):
            clear_session()
            st.rerun()
        st.caption(f"Workspace: `{os.getcwd()}`")

    if page == "Home":
        home.render(settings)
    elif page == "Quick Start":
        quick_start.render(settings)
    elif page == "xTB Explorer":
        xtb_explorer.render(settings)
    elif page == "Custom Workflow":
        custom_workflow.render(settings)
    elif page == "Results Viewer":
        results_viewer.render(settings)
    else:
        documentation.render(settings)


if __name__ == "__main__":
    main()

