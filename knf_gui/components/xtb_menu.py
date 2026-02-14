import streamlit as st


XTB_OPTIONS = {
    "Basic Calculations": {
        "Single Point Energy": [],
        "Gradient Calculation": ["--grad"],
        "Geometry Optimization": ["--opt"],
    },
    "Electronic Properties": {
        "Ionization Potential (IP)": ["--vipea"],
        "Electron Affinity (EA)": ["--vipea"],
        "IP + EA Combined": ["--vipea"],
        "Electrophilicity Index": ["--vipea"],
        "Fukui Indices": ["--vfukui"],
    },
    "Advanced Analysis": {
        "Electrostatic Potential": ["--esp"],
        "Population Analysis": ["--pop"],
        "Bond Order Analysis": ["--wbo"],
        "Orbital Localization": ["--lmo"],
    },
    "Dynamics": {
        "Molecular Dynamics": ["--md"],
        "Conformer Search": ["--md"],
        "Frequency Calculation": ["--hess"],
    },
}


def render_xtb_menu():
    st.markdown("### xTB Explorer")
    category = st.selectbox("Category", list(XTB_OPTIONS.keys()))
    option = st.selectbox("Operation", list(XTB_OPTIONS[category].keys()))
    add_solvent = st.checkbox("Add implicit solvent (ALPB)")
    solvent = st.text_input("Solvent", "water", disabled=not add_solvent)
    flags = list(XTB_OPTIONS[category][option])
    if add_solvent and solvent.strip():
        flags += ["--alpb", solvent.strip()]
    return option, flags

