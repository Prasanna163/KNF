import os
import json
import subprocess
import time
import tempfile
import shutil
import threading
from pathlib import Path

import streamlit as st

from knf_core.pipeline import KNFPipeline
from knf_core import converter, geometry, utils, wrapper, multiwfn


st.set_page_config(page_title="KNF-CORE GUI", layout="wide")


def build_work_paths(input_path: str):
    abs_input = os.path.abspath(input_path)
    base_name = Path(abs_input).stem
    work_dir = os.path.join(os.path.dirname(abs_input), base_name)
    input_dir = os.path.join(work_dir, "input")
    results_dir = os.path.join(work_dir, "results")
    utils.ensure_directory(input_dir)
    utils.ensure_directory(results_dir)
    return abs_input, base_name, work_dir, input_dir, results_dir


def prepare_input(input_path: str):
    abs_input, base_name, work_dir, input_dir, results_dir = build_work_paths(input_path)
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


def copy_xyz_to_results(target_xyz: str, results_dir: str):
    work_xyz = os.path.join(results_dir, "input.xyz")
    if os.path.abspath(target_xyz) != os.path.abspath(work_xyz):
        utils.safe_copy(target_xyz, work_xyz)
    return work_xyz


def run_xtb_with_log(filepath: str, charge: int, uhf: int, flags: list[str], log_name: str, progress_ui=None):
    cwd = os.path.dirname(os.path.abspath(filepath))
    filename = os.path.basename(filepath)
    cmd = ["xtb", filename] + flags + ["--charge", str(charge), "--uhf", str(uhf)]
    log_path = os.path.join(cwd, log_name)

    progress_bar = None
    status_box = None
    if progress_ui:
        progress_bar, status_box = progress_ui
        progress_bar.progress(1, text="Initializing xTB...")
        status_box.info("xTB started.")

    with open(log_path, "w") as log:
        process = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        started = time.time()
        while process.poll() is None:
            if progress_bar:
                elapsed = time.time() - started
                pct = min(95, int(5 + elapsed * 2))
                progress_bar.progress(pct, text=f"Running xTB... {pct}%")
            time.sleep(0.5)

        return_code = process.returncode
        if return_code != 0:
            if progress_bar:
                progress_bar.empty()
            if status_box:
                status_box.error("xTB failed. Check log for details.")
            raise subprocess.CalledProcessError(return_code, cmd)

    if progress_bar:
        progress_bar.progress(100, text="xTB run complete.")
    if status_box:
        status_box.success("xTB completed successfully.")

    return cmd, log_path


def run_nci_step(results_dir: str, optimized_xyz: str, charge: int, uhf: int):
    wrapper.run_xtb_sp(optimized_xyz, charge, uhf)
    molden_file = os.path.join(results_dir, "molden.input")
    if not os.path.exists(molden_file):
        raise FileNotFoundError(f"Missing molden file: {molden_file}")
    multiwfn.run_multiwfn(molden_file, results_dir)
    raw_output = os.path.join(results_dir, "output.txt")
    nci_grid = os.path.join(results_dir, "nci_grid.txt")
    if os.path.exists(raw_output):
        os.replace(raw_output, nci_grid)
    return nci_grid if os.path.exists(nci_grid) else None


def read_text_if_exists(path: str, max_chars: int = 100000):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read(max_chars)


def save_uploaded_input(uploaded_file):
    upload_dir = os.path.join(tempfile.gettempdir(), "knf_core_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    target = os.path.join(upload_dir, uploaded_file.name)
    with open(target, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target


def run_pipeline_with_progress(context: dict, charge: int, spin: int, force: bool, clean: bool, debug: bool):
    progress = st.progress(0, text="Starting KNF pipeline...")
    status = st.empty()
    status.info("Initializing pipeline...")

    result = {"error": None}

    def _runner():
        try:
            pipeline = KNFPipeline(
                input_file=context["abs_input"],
                charge=charge,
                spin=spin,
                force=force,
                clean=clean,
                debug=debug,
            )
            pipeline.run()
        except Exception as e:
            result["error"] = str(e)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()

    results_dir = context["results_dir"]
    milestones = [
        ("Input prepared", lambda: os.path.exists(os.path.join(results_dir, "input.xyz")), 15),
        ("xTB optimization completed", lambda: os.path.exists(os.path.join(results_dir, "xtbopt.xyz")), 40),
        ("xTB properties generated", lambda: os.path.exists(os.path.join(results_dir, "xtb.log")), 60),
        ("Multiwfn input ready", lambda: os.path.exists(os.path.join(results_dir, "molden.input")), 75),
        ("NCI grid generated", lambda: os.path.exists(os.path.join(results_dir, "nci_grid.txt")), 90),
        ("KNF output generated", lambda: os.path.exists(os.path.join(results_dir, "knf.json")), 100),
    ]

    last_pct = 5
    progress.progress(last_pct, text="Running pipeline...")

    while thread.is_alive():
        for label, check_fn, pct in milestones:
            if pct > last_pct and check_fn():
                last_pct = pct
                progress.progress(last_pct, text=f"{label} ({last_pct}%)")
                status.info(label)
        time.sleep(0.4)

    thread.join()

    if result["error"]:
        progress.empty()
        status.error(f"Pipeline failed: {result['error']}")
        return False

    progress.progress(100, text="Pipeline completed.")
    status.success("Pipeline completed successfully.")
    return True


def show_prepared_summary(context: dict):
    st.success("Input prepared successfully.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Atoms", context["atom_count"])
    c2.metric("Fragments", len(context["fragments"]))
    c3.metric("Input", os.path.basename(context["abs_input"]))
    st.caption(f"XYZ used: `{context['target_xyz']}`")
    st.caption(f"Results folder: `{context['results_dir']}`")


def main():
    st.title("KNF-CORE GUI")
    st.write("Streamlit interface for automated KNF generation and interactive xTB workflows.")

    utils.ensure_multiwfn_in_path()

    with st.sidebar:
        st.header("System")
        st.write(f"`obabel`: {'OK' if bool(shutil.which('obabel')) else 'Missing'}")
        st.write(f"`xtb`: {'OK' if bool(shutil.which('xtb')) else 'Missing'}")
        st.write(f"`Multiwfn`: {'OK' if bool(shutil.which('Multiwfn') or shutil.which('Multiwfn.exe')) else 'Missing'}")
        st.divider()
        charge = st.number_input("Charge", value=0, step=1, format="%d")
        spin = st.number_input("Multiplicity", min_value=1, value=1, step=1, format="%d")
        force = st.checkbox("Force recomputation", value=True)
        clean = st.checkbox("Clean run folder before execute", value=False)
        debug = st.checkbox("Debug logging", value=True)

    st.subheader("Step 1: Select Input File")
    source_mode = st.radio("Input source", ["Browse file", "Use local path"], horizontal=True)
    selected_input_path = ""

    if source_mode == "Browse file":
        uploaded = st.file_uploader(
            "Choose a molecule file",
            type=["xyz", "sdf", "mol", "pdb", "mol2"],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            selected_input_path = save_uploaded_input(uploaded)
            st.caption(f"Selected: `{uploaded.name}`")
            st.caption(f"Temporary copy: `{selected_input_path}`")
    else:
        selected_input_path = st.text_input("Input molecule path", placeholder="E:/path/to/complex.xyz")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        prepare_btn = st.button("Prepare Input", use_container_width=True)
    with col_b:
        clear_btn = st.button("Clear Session", use_container_width=True)

    if clear_btn:
        st.session_state.pop("prepared_context", None)
        st.session_state.pop("last_xtb_log", None)
        st.session_state.pop("last_cmd", None)
        st.session_state.pop("optimized_xyz", None)
        st.session_state.pop("last_error", None)
        st.rerun()

    if prepare_btn:
        try:
            if not selected_input_path or not os.path.exists(selected_input_path):
                st.error("Provide a valid existing file path.")
            else:
                st.session_state["prepared_context"] = prepare_input(selected_input_path)
        except Exception as e:
            st.session_state["last_error"] = str(e)

    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])

    context = st.session_state.get("prepared_context")
    if not context:
        st.info("Prepare an input file to continue.")
        return

    show_prepared_summary(context)

    tab1, tab2, tab3 = st.tabs(["Automated Pipeline", "xTB Explorer", "Results Browser"])

    with tab1:
        st.subheader("Automated Optimization + NCI Analysis")
        if st.button("Run Full KNF Pipeline", type="primary"):
            run_pipeline_with_progress(
                context=context,
                charge=int(charge),
                spin=int(spin),
                force=force,
                clean=clean,
                debug=debug,
            )

        knf_json = os.path.join(context["results_dir"], "knf.json")
        output_txt = os.path.join(context["results_dir"], "output.txt")
        if os.path.exists(knf_json):
            st.markdown("**knf.json**")
            try:
                with open(knf_json, "r", encoding="utf-8") as f:
                    st.json(json.load(f))
            except Exception:
                st.code(read_text_if_exists(knf_json) or "", language="json")
        if os.path.exists(output_txt):
            st.markdown("**output.txt**")
            st.code(read_text_if_exists(output_txt) or "", language="text")

    with tab2:
        st.subheader("Interactive xTB Explorer")
        work_xyz = copy_xyz_to_results(context["target_xyz"], context["results_dir"])
        uhf = int(spin) - 1

        xtb_options = {
            "Single Point Energy": [],
            "Gradient Calculation": ["--grad"],
            "Geometry Optimization": ["--opt"],
            "Ionization Potential (IP)": ["--vipea"],
            "Electron Affinity (EA)": ["--vipea"],
            "IP + EA Combined": ["--vipea"],
            "Electrophilicity Index": ["--vipea"],
            "Fukui Indices": ["--vfukui"],
            "Electrostatic Potential": ["--esp"],
            "Population Analysis": ["--pop"],
            "Bond Order Analysis": ["--wbo"],
            "Orbital Localization": ["--lmo"],
            "Molecular Dynamics": ["--md"],
            "Conformer Search": ["--md"],
            "Frequency Calculation": ["--hess"],
        }

        option = st.selectbox("xTB Operation", list(xtb_options.keys()))
        add_solvent = st.checkbox("Add implicit solvent (ALPB)")
        solvent = st.text_input("Solvent name", value="water", disabled=not add_solvent)

        flags = list(xtb_options[option])
        if add_solvent and solvent.strip():
            flags += ["--alpb", solvent.strip()]

        cmd_preview = ["xtb", os.path.basename(work_xyz)] + flags + ["--charge", str(int(charge)), "--uhf", str(uhf)]
        st.markdown("**Prepared command**")
        st.code(" ".join(cmd_preview), language="bash")

        if st.button("Run Selected xTB Command", use_container_width=True):
            try:
                progress = st.progress(0, text="Preparing xTB...")
                status = st.empty()
                log_name = "xtb_opt.log" if option == "Geometry Optimization" else "xtb_explorer.log"
                cmd, log_path = run_xtb_with_log(
                    work_xyz, int(charge), uhf, flags, log_name, progress_ui=(progress, status)
                )
                st.session_state["last_cmd"] = cmd
                st.session_state["last_xtb_log"] = log_path
                if option == "Geometry Optimization":
                    optimized_xyz = os.path.join(context["results_dir"], "xtbopt.xyz")
                    if os.path.exists(optimized_xyz):
                        st.session_state["optimized_xyz"] = optimized_xyz
                st.success("xTB command completed.")
            except Exception as e:
                st.error(f"xTB command failed: {e}")

        log_path = st.session_state.get("last_xtb_log")
        if log_path and os.path.exists(log_path):
            with st.expander("Show latest xTB log"):
                st.code(read_text_if_exists(log_path, max_chars=50000) or "", language="text")

        optimized_xyz = st.session_state.get("optimized_xyz")
        if optimized_xyz and os.path.exists(optimized_xyz):
            st.success(f"Optimization available: {optimized_xyz}")
            if st.button("Continue to NCI Analysis (Multiwfn)", type="primary"):
                try:
                    nci_file = run_nci_step(context["results_dir"], optimized_xyz, int(charge), uhf)
                    if nci_file:
                        st.success(f"NCI completed. Grid file: {nci_file}")
                    else:
                        st.warning("NCI command completed, but no nci_grid.txt was found.")
                except Exception as e:
                    st.error(f"NCI step failed: {e}")

    with tab3:
        st.subheader("Results Browser")
        results_dir = context["results_dir"]
        st.caption(f"Directory: `{results_dir}`")
        if not os.path.exists(results_dir):
            st.info("No results directory yet.")
        else:
            files = sorted(Path(results_dir).glob("*"))
            if not files:
                st.info("No files in results directory.")
            for f in files:
                st.write(f"- `{f.name}`")

            selected = st.selectbox("Preview file", [f.name for f in files], index=0 if files else None)
            if selected:
                full = os.path.join(results_dir, selected)
                if os.path.isfile(full):
                    lower = selected.lower()
                    if lower.endswith(".json"):
                        try:
                            with open(full, "r", encoding="utf-8") as fh:
                                st.json(json.load(fh))
                        except Exception:
                            st.code(read_text_if_exists(full) or "", language="json")
                    elif lower.endswith((".txt", ".log", ".xyz", ".inp", ".dat", ".cosmo", ".mol")):
                        st.code(read_text_if_exists(full) or "", language="text")
                    else:
                        st.info("Preview not supported for this extension.")


if __name__ == "__main__":
    main()
