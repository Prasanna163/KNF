import json
import os
import subprocess
import threading
import time

import streamlit as st

from knf_core.pipeline import KNFPipeline
from knf_core import wrapper, multiwfn


def run_pipeline_with_progress(context: dict, settings: dict):
    progress = st.progress(0, text="Starting KNF pipeline...")
    status = st.empty()
    status.info("Initializing pipeline...")
    result = {"error": None}

    def _runner():
        try:
            pipeline = KNFPipeline(
                input_file=context["abs_input"],
                charge=settings["charge"],
                spin=settings["spin"],
                force=settings["force"],
                clean=settings["clean"],
                debug=settings["debug"],
            )
            pipeline.run()
        except Exception as e:
            result["error"] = str(e)

    t0 = time.time()
    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()

    results_dir = context["results_dir"]
    milestones = [
        ("Geometry Prep", lambda: os.path.exists(os.path.join(results_dir, "input.xyz")), 15),
        ("xTB Optimization", lambda: os.path.exists(os.path.join(results_dir, "xtbopt.xyz")), 40),
        ("xTB Properties", lambda: os.path.exists(os.path.join(results_dir, "xtb.log")), 60),
        ("NCI Analysis", lambda: os.path.exists(os.path.join(results_dir, "nci_grid.txt")), 85),
        ("Finalize", lambda: os.path.exists(os.path.join(results_dir, "knf.json")), 100),
    ]
    current = 5
    progress.progress(current, text="Running...")

    while thread.is_alive():
        for label, check_fn, pct in milestones:
            if pct > current and check_fn():
                current = pct
                progress.progress(current, text=f"{label} ({pct}%)")
                status.info(f"Running: {label}")
        time.sleep(0.4)

    thread.join()
    elapsed = time.time() - t0
    if result["error"]:
        progress.empty()
        status.error(f"Pipeline failed: {result['error']}")
        return False, elapsed
    progress.progress(100, text="Pipeline completed.")
    status.success("Pipeline completed successfully.")
    return True, elapsed


def run_xtb_with_progress(work_xyz: str, charge: int, spin: int, flags: list[str], log_name: str):
    progress = st.progress(1, text="Starting xTB...")
    status = st.empty()
    cwd = os.path.dirname(os.path.abspath(work_xyz))
    filename = os.path.basename(work_xyz)
    cmd = ["xtb", filename] + flags + ["--charge", str(charge), "--uhf", str(spin - 1)]
    log_path = os.path.join(cwd, log_name)

    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        started = time.time()
        while proc.poll() is None:
            pct = min(95, int(5 + (time.time() - started) * 2))
            progress.progress(pct, text=f"Running xTB... {pct}%")
            time.sleep(0.5)
        if proc.returncode != 0:
            status.error("xTB failed. Check logs.")
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    progress.progress(100, text="xTB completed.")
    status.success("xTB completed.")
    return cmd, log_path


def run_nci_from_optimized(results_dir: str, optimized_xyz: str, charge: int, spin: int):
    wrapper.run_xtb_sp(optimized_xyz, charge, spin - 1)
    molden_file = os.path.join(results_dir, "molden.input")
    if not os.path.exists(molden_file):
        raise FileNotFoundError("Missing molden.input for Multiwfn.")
    multiwfn.run_multiwfn(molden_file, results_dir)
    raw_output = os.path.join(results_dir, "output.txt")
    nci_grid = os.path.join(results_dir, "nci_grid.txt")
    if os.path.exists(raw_output):
        os.replace(raw_output, nci_grid)
    return nci_grid


def load_knf_json(results_dir: str):
    path = os.path.join(results_dir, "knf.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

