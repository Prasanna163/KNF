import streamlit as st


def render(settings: dict):
    del settings
    st.header("Documentation & Tutorials")
    st.info(
        "Context: This page explains concepts, expected outputs, and recommended usage patterns "
        "for KNF-CORE GUI workflows."
    )
    st.markdown(
        "Use this reference if you are unsure which tab to use, how to interpret descriptors, "
        "or how to troubleshoot common failures."
    )
    tabs = st.tabs(
        [
            "Getting Started",
            "xTB Fundamentals",
            "Descriptors",
            "Visualization Guide",
            "Advanced Workflows",
        ]
    )

    with tabs[0]:
        st.markdown(
            """
1. Go to **Quick Analysis**.
2. Upload a molecule (`xyz/sdf/mol/pdb/mol2`) or choose an example.
3. Set charge and multiplicity in the sidebar.
4. Run the automated pipeline and inspect KNF vector results.
"""
        )
        st.caption("Tip: Start with `example.mol` to verify your environment quickly.")

    with tabs[1]:
        st.markdown(
            """
- **Single Point**: Energy at fixed geometry.
- **Optimization**: Relax geometry to local minimum.
- **Fukui**: Reactivity indicators.
- **WBO/Population**: Bonding and charge information.
"""
        )

    with tabs[2]:
        st.markdown(
            """
KNF output contains:
- `SNCI`
- `SCDI_variance`
- `KNF_vector` with components `f1..f9`

Use the KNF Vector chart to compare descriptor magnitudes.
"""
        )
        st.caption("Interpretation note: compare vectors across molecules under consistent settings.")

    with tabs[3]:
        st.markdown(
            """
Molecule preview uses interactive 3D scatter:
- Rotate and zoom to inspect geometry.
- Hover over atoms for index and element.
"""
        )

    with tabs[4]:
        st.markdown(
            """
Custom Workflow Builder lets you run sequential xTB steps.
Use it for method exploration before standard KNF runs.
"""
        )
        st.caption("Recommended: validate a single step in xTB Explorer before building a longer chain.")
