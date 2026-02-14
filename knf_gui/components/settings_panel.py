import shutil
import streamlit as st
from knf_core import utils


def render_settings_panel():
    utils.ensure_multiwfn_in_path()
    with st.sidebar:
        st.header("System")
        st.write(f"`obabel`: {'OK' if bool(shutil.which('obabel')) else 'Missing'}")
        st.write(f"`xtb`: {'OK' if bool(shutil.which('xtb')) else 'Missing'}")
        st.write(
            f"`Multiwfn`: {'OK' if bool(shutil.which('Multiwfn') or shutil.which('Multiwfn.exe')) else 'Missing'}"
        )
        st.divider()
        st.header("Calculation Settings")
        charge = st.number_input("Charge", value=0, step=1, format="%d")
        spin = st.selectbox("Spin Multiplicity", [1, 2, 3], index=0)
        solvent = st.selectbox("Solvent", ["Vacuum", "Water", "DMSO", "Acetone"], index=0)
        method = st.selectbox("xTB Method", ["GFN2-xTB (recommended)", "GFN1-xTB", "GFN-FF"], index=0)
        force = st.checkbox("Force recomputation", value=True)
        clean = st.checkbox("Clean run directory", value=False)
        debug = st.checkbox("Debug logging", value=True)
    return {
        "charge": int(charge),
        "spin": int(spin),
        "solvent": solvent,
        "method": method,
        "force": force,
        "clean": clean,
        "debug": debug,
    }

