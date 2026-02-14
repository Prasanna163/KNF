import os

import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem

from knf_core import geometry


FRAGMENT_COLORS = [
    "#2E86AB",  # blue
    "#F18F01",  # orange
    "#A23B72",  # magenta
    "#06A77D",  # green
    "#8D6A9F",  # violet
    "#C73E1D",  # red
]


def _build_rdkit_3d_view(mol: Chem.Mol, height: int):
    mol_block = Chem.MolToMolBlock(mol)
    view = py3Dmol.view(width=1000, height=height)
    view.addModel(mol_block, "mol")
    view.setStyle({"stick": {"radius": 0.14}, "sphere": {"scale": 0.22}})

    fragments = geometry.detect_fragments(mol)
    if len(fragments) > 1:
        for frag_idx, atom_indices in enumerate(fragments):
            color = FRAGMENT_COLORS[frag_idx % len(FRAGMENT_COLORS)]
            for atom_idx in atom_indices:
                # py3Dmol atom serial is 1-based
                serial = atom_idx + 1
                view.addStyle(
                    {"serial": serial},
                    {"stick": {"color": color}, "sphere": {"color": color, "scale": 0.24}},
                )

    view.zoomTo()
    return view


def render_molecule_viewer(xyz_path: str, height: int = 420):
    if not xyz_path or not os.path.exists(xyz_path):
        st.info("No molecule geometry available for preview.")
        return
    try:
        mol = geometry.load_molecule(xyz_path)
        view = _build_rdkit_3d_view(mol, height)
        components.html(view._make_html(), height=height + 10, scrolling=False)
    except Exception as e:
        st.warning(f"3D preview unavailable: {e}")
