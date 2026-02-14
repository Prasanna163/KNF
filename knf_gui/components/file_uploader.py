import os
import tempfile
from pathlib import Path

import streamlit as st

from knf_core import converter, geometry, utils
from knf_gui.utils.validators import validate_molecule_file


def _save_uploaded(uploaded_file):
    upload_dir = os.path.join(tempfile.gettempdir(), "knf_gui_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    target = os.path.join(upload_dir, uploaded_file.name)
    with open(target, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target


def _build_work_paths(input_path: str):
    abs_input = os.path.abspath(input_path)
    base_name = Path(abs_input).stem
    work_dir = os.path.join(os.path.dirname(abs_input), base_name)
    input_dir = os.path.join(work_dir, "input")
    results_dir = os.path.join(work_dir, "results")
    utils.ensure_directory(input_dir)
    utils.ensure_directory(results_dir)
    return abs_input, base_name, work_dir, input_dir, results_dir


def _prepare_context(input_path: str):
    abs_input, base_name, work_dir, input_dir, results_dir = _build_work_paths(input_path)
    target_xyz = converter.ensure_xyz(abs_input, input_dir)
    mol = geometry.load_molecule(target_xyz)
    fragments = geometry.detect_fragments(mol)
    return {
        "abs_input": abs_input,
        "base_name": base_name,
        "work_dir": work_dir,
        "input_dir": input_dir,
        "results_dir": results_dir,
        "target_xyz": target_xyz,
        "atom_count": mol.GetNumAtoms(),
        "fragments": fragments,
    }


def render_file_uploader():
    st.markdown("### Step 1: Upload Molecule")
    source = st.radio("Input source", ["Browse file", "Use local path", "Use example"], horizontal=True)
    selected_path = ""

    if source == "Browse file":
        uploaded = st.file_uploader("Choose molecule file", type=["xyz", "sdf", "mol", "pdb", "mol2"])
        if uploaded is not None:
            selected_path = _save_uploaded(uploaded)
            st.caption(f"Selected: `{uploaded.name}`")
    elif source == "Use local path":
        selected_path = st.text_input("Input path", placeholder="E:/path/to/molecule.xyz").strip()
    else:
        example_options = [
            "example.mol",
            "DES.mol",
        ]
        chosen = st.selectbox("Example files", example_options, index=0)
        selected_path = os.path.abspath(chosen)

    if st.button("Prepare Molecule", type="primary", use_container_width=True):
        valid, err, warnings = validate_molecule_file(selected_path)
        if not valid:
            st.error(err)
            return None
        for msg in warnings:
            st.warning(msg)
        try:
            return _prepare_context(selected_path)
        except Exception as e:
            st.error(f"Failed to prepare molecule: {e}")
            return None
    return None

