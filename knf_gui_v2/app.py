import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from flask import Flask, Response, jsonify, render_template, request

from knf_core import utils

JOBS = {}
JOBS_LOCK = threading.RLock()
MAX_LOG_LINES = 5000
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _job_public_view(job: dict) -> dict:
    # Avoid deepcopy here: running jobs hold a live subprocess object containing
    # thread locks, which is not pickle/copy-safe.
    preview = job.get("result_preview") or {}
    rows = preview.get("rows") or []
    safe = {
        "id": job.get("id"),
        "status": job.get("status"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "returncode": job.get("returncode"),
        "error": job.get("error"),
        "pid": job.get("pid"),
        "payload": dict(job.get("payload") or {}),
        "command": list(job.get("command") or []),
        "logs": list(job.get("logs") or []),
        "artifacts": dict(job.get("artifacts") or {}),
        "result_preview": {
            "type": preview.get("type", "none"),
            "summary": dict(preview.get("summary") or {}),
            "rows": [dict(r) for r in rows],
            "artifacts": dict(preview.get("artifacts") or {}),
            "output_excerpt": preview.get("output_excerpt", ""),
        },
    }
    return safe


def _append_log(job: dict, line: str):
    logs = job.setdefault("logs", [])
    logs.append(line.rstrip("\n"))
    if len(logs) > MAX_LOG_LINES:
        overflow = len(logs) - MAX_LOG_LINES
        del logs[:overflow]


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _resolve_results_root(input_path: str, output_dir: Optional[str]) -> str:
    if output_dir:
        return os.path.abspath(output_dir)
    if os.path.isdir(input_path):
        return os.path.join(os.path.abspath(input_path), "Results")
    return os.path.join(os.path.dirname(os.path.abspath(input_path)), "Results")


def _derive_artifacts(input_path: str, output_dir: Optional[str]) -> dict:
    results_root = _resolve_results_root(input_path, output_dir)
    artifacts = {"results_root": results_root}
    if os.path.isdir(input_path):
        artifacts["batch_json"] = os.path.join(results_root, "batch_knf.json")
        artifacts["batch_csv"] = os.path.join(results_root, "batch_knf.csv")
    else:
        stem = Path(input_path).stem
        result_dir = os.path.join(results_root, stem)
        artifacts["result_dir"] = result_dir
        artifacts["knf_json"] = os.path.join(result_dir, "knf.json")
        artifacts["output_txt"] = os.path.join(result_dir, "output.txt")
    return artifacts


def _build_result_preview(artifacts: dict) -> dict:
    preview = {
        "type": "none",
        "summary": {},
        "rows": [],
        "artifacts": artifacts,
    }

    batch_json = artifacts.get("batch_json")
    if batch_json and os.path.exists(batch_json):
        with open(batch_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = []
        for entry in payload.get("knf_results", []):
            knf = entry.get("knf") or {}
            vec = knf.get("KNF_vector") or []
            rows.append(
                {
                    "File": entry.get("input_file_name", ""),
                    "f1": vec[0] if len(vec) > 0 else "",
                    "f2": vec[1] if len(vec) > 1 else "",
                    "f3": vec[2] if len(vec) > 2 else "",
                    "f4": vec[3] if len(vec) > 3 else "",
                    "f5": vec[4] if len(vec) > 4 else "",
                    "f6": vec[5] if len(vec) > 5 else "",
                    "f7": vec[6] if len(vec) > 6 else "",
                    "f8": vec[7] if len(vec) > 7 else "",
                    "f9": vec[8] if len(vec) > 8 else "",
                    "SNCI": knf.get("SNCI", ""),
                    "SCDI": knf.get("SCDI_variance", ""),
                }
            )
        preview["type"] = "batch"
        preview["summary"] = payload.get("summary", {})
        preview["rows"] = rows[:200]
        return preview

    knf_json = artifacts.get("knf_json")
    if knf_json and os.path.exists(knf_json):
        with open(knf_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        vec = payload.get("KNF_vector") or []
        preview["type"] = "single"
        preview["summary"] = {
            "total_files": 1,
            "successful_files": 1,
            "failed_files": 0,
            "total_time_seconds": "",
        }
        preview["rows"] = [
            {
                "File": Path(knf_json).parent.name,
                "f1": vec[0] if len(vec) > 0 else "",
                "f2": vec[1] if len(vec) > 1 else "",
                "f3": vec[2] if len(vec) > 2 else "",
                "f4": vec[3] if len(vec) > 3 else "",
                "f5": vec[4] if len(vec) > 4 else "",
                "f6": vec[5] if len(vec) > 5 else "",
                "f7": vec[6] if len(vec) > 6 else "",
                "f8": vec[7] if len(vec) > 7 else "",
                "f9": vec[8] if len(vec) > 8 else "",
                "SNCI": payload.get("SNCI", ""),
                "SCDI": payload.get("SCDI_variance", ""),
            }
        ]
        output_txt = artifacts.get("output_txt")
        if output_txt and os.path.exists(output_txt):
            with open(output_txt, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            preview["output_excerpt"] = "".join(lines[:80])
        return preview

    return preview


def _normalize_payload(raw: dict) -> dict:
    payload = dict(raw or {})
    input_path = (payload.get("input_path") or "").strip().strip('"').strip("'")
    input_path = utils.resolve_artifacted_path(input_path)
    payload["input_path"] = input_path
    payload["output_dir"] = (payload.get("output_dir") or "").strip() or None
    payload["processing"] = (payload.get("processing") or "single").lower()
    payload["nci_backend"] = (payload.get("nci_backend") or "torch").strip().lower()
    payload["nci_device"] = (payload.get("nci_device") or "cpu").strip().lower()
    payload["nci_dtype"] = (payload.get("nci_dtype") or "float32").strip().lower()
    payload["nci_grid_spacing"] = _as_float(payload.get("nci_grid_spacing", 0.2), 0.2)
    payload["nci_grid_padding"] = _as_float(payload.get("nci_grid_padding", 3.0), 3.0)
    payload["nci_batch_size"] = _as_int(payload.get("nci_batch_size", 250000), 250000)
    payload["nci_eig_batch_size"] = _as_int(payload.get("nci_eig_batch_size", 200000), 200000)
    payload["nci_rho_floor"] = _as_float(payload.get("nci_rho_floor", 1e-12), 1e-12)
    payload["charge"] = _as_int(payload.get("charge", 0), 0)
    payload["spin"] = _as_int(payload.get("spin", 1), 1)
    payload["ram_per_job"] = _as_float(payload.get("ram_per_job", 50.0), 50.0)
    workers = payload.get("workers")
    payload["workers"] = _as_int(workers, 1) if workers not in (None, "", 0, "0") else None
    payload["multiwfn_path"] = (payload.get("multiwfn_path") or "").strip() or None
    for flag in (
        "force",
        "clean",
        "debug",
        "refresh_autoconfig",
        "refresh_first_run",
        "quiet_config",
        "nci_apply_primitive_norm",
    ):
        payload[flag] = bool(payload.get(flag, False))
    payload["storage_efficient"] = bool(payload.get("storage_efficient", True))
    # New CLI uses --full-files; keep legacy GUI checkbox as inverse.
    payload["full_files"] = not bool(payload["storage_efficient"])
    return payload


def _build_command(payload: dict) -> list:
    cmd = [sys.executable, "-m", "knf_core.main", payload["input_path"]]
    cmd.extend(["--charge", str(payload["charge"])])
    cmd.extend(["--spin", str(payload["spin"])])
    cmd.extend(["--processing", payload["processing"]])
    cmd.extend(["--ram-per-job", str(payload["ram_per_job"])])
    cmd.extend(["--nci-backend", payload["nci_backend"]])
    cmd.extend(["--nci-device", payload["nci_device"]])
    cmd.extend(["--nci-dtype", payload["nci_dtype"]])
    cmd.extend(["--nci-grid-spacing", str(payload["nci_grid_spacing"])])
    cmd.extend(["--nci-grid-padding", str(payload["nci_grid_padding"])])
    cmd.extend(["--nci-batch-size", str(payload["nci_batch_size"])])
    cmd.extend(["--nci-eig-batch-size", str(payload["nci_eig_batch_size"])])
    cmd.extend(["--nci-rho-floor", str(payload["nci_rho_floor"])])
    if payload["workers"] is not None:
        cmd.extend(["--workers", str(payload["workers"])])
    if payload["output_dir"]:
        cmd.extend(["--output-dir", payload["output_dir"]])
    if payload["multiwfn_path"]:
        cmd.extend(["--multiwfn-path", payload["multiwfn_path"]])
    if payload["force"]:
        cmd.append("--force")
    if payload["clean"]:
        cmd.append("--clean")
    if payload["debug"]:
        cmd.append("--debug")
    if payload["refresh_autoconfig"]:
        cmd.append("--refresh-autoconfig")
    if payload["refresh_first_run"]:
        cmd.append("--refresh-first-run")
    if payload["quiet_config"]:
        cmd.append("--quiet-config")
    if payload["nci_apply_primitive_norm"]:
        cmd.append("--nci-apply-primitive-norm")
    if payload["full_files"]:
        cmd.append("--full-files")
    return cmd


def _run_job(job_id: str, payload: dict):
    with JOBS_LOCK:
        job = JOBS[job_id]
        job["status"] = "running"
        job["started_at"] = _iso_now()
        _append_log(job, f"$ {' '.join(job['command'])}")

    proc = None
    try:
        proc = subprocess.Popen(
            job["command"],
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        with JOBS_LOCK:
            job["_proc"] = proc
            job["pid"] = proc.pid

        if proc.stdout is not None:
            for line in proc.stdout:
                with JOBS_LOCK:
                    _append_log(job, line)

        rc = proc.wait()
        with JOBS_LOCK:
            job["returncode"] = rc
            job["finished_at"] = _iso_now()
            job["status"] = "succeeded" if rc == 0 else "failed"
            if rc != 0:
                job["error"] = f"Process exited with code {rc}"
            job["artifacts"] = _derive_artifacts(payload["input_path"], payload["output_dir"])
            try:
                job["result_preview"] = _build_result_preview(job["artifacts"])
            except Exception as e:
                job["result_preview"] = {"type": "none", "summary": {}, "rows": []}
                _append_log(job, f"[GUI] Result preview failed: {e}")
    except Exception as e:
        with JOBS_LOCK:
            job["status"] = "failed"
            job["finished_at"] = _iso_now()
            job["error"] = str(e)
            _append_log(job, f"[GUI] Runner error: {e}")
    finally:
        with JOBS_LOCK:
            if job.get("_proc") is not None:
                job["_proc"] = None


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.get("/")
    def index():
        return render_template("index.html", asset_version=int(time.time()))

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "time": _iso_now()})

    @app.get("/api/dependencies")
    def dependencies():
        utils.ensure_multiwfn_in_path()
        found = bool(utils.find_multiwfn()) or bool(
            shutil.which("Multiwfn") or shutil.which("Multiwfn.exe")
        )
        registered = utils.get_registered_multiwfn_path()
        return jsonify(
            {
                "multiwfn": {
                    "detected": found,
                    "registered_path": registered,
                }
            }
        )

    @app.post("/api/multiwfn-path")
    def set_multiwfn_path():
        payload = request.get_json(silent=True) or {}
        raw_path = (payload.get("path") or "").strip()
        if not raw_path:
            return jsonify({"error": "Path is required"}), 400
        resolved = utils.register_multiwfn_path(raw_path, persist=True)
        if not resolved:
            return jsonify({"error": "Invalid Multiwfn executable or directory path"}), 400
        return jsonify({"ok": True, "path": resolved})

    @app.get("/api/jobs")
    def list_jobs():
        with JOBS_LOCK:
            rows = [_job_public_view(j) for j in JOBS.values()]
        rows.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return jsonify({"jobs": rows})

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id: str):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return jsonify({"error": "Job not found"}), 404
            return jsonify(_job_public_view(job))

    @app.post("/api/jobs/<job_id>/cancel")
    def cancel_job(job_id: str):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return jsonify({"error": "Job not found"}), 404
            proc = job.get("_proc")
            if proc is None or job.get("status") != "running":
                return jsonify({"ok": False, "message": "Job is not running"}), 400
            proc.terminate()
            _append_log(job, "[GUI] Cancel requested.")
        return jsonify({"ok": True})

    @app.post("/api/run")
    def run():
        payload = _normalize_payload(request.get_json(silent=True) or {})
        if not payload.get("input_path"):
            return jsonify({"error": "input_path is required"}), 400
        if not os.path.exists(payload["input_path"]):
            return jsonify({"error": f"Input path not found: {payload['input_path']}"}), 400
        if payload["processing"] not in {"auto", "single", "multi"}:
            return jsonify({"error": "processing must be 'auto', 'single', or 'multi'"}), 400

        job_id = str(uuid.uuid4())
        cmd = _build_command(payload)
        job = {
            "id": job_id,
            "status": "queued",
            "created_at": _iso_now(),
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "error": None,
            "pid": None,
            "payload": payload,
            "command": cmd,
            "logs": [],
            "artifacts": {},
            "result_preview": {"type": "none", "summary": {}, "rows": []},
        }

        with JOBS_LOCK:
            JOBS[job_id] = job

        t = threading.Thread(target=_run_job, args=(job_id, payload), daemon=True)
        t.start()

        return jsonify({"ok": True, "job_id": job_id})

    @app.post("/api/dialog")
    def dialog():
        payload = request.get_json(silent=True) or {}
        mode = (payload.get("mode") or "").strip().lower()
        if mode not in {"file", "directory", "multiwfn_file", "multiwfn_directory"}:
            return jsonify({"error": "Unsupported dialog mode"}), 400

        try:
            path = _open_native_dialog(mode)
        except Exception as e:
            return jsonify({"error": f"Dialog failed: {e}"}), 500

        return jsonify({"ok": True, "path": path})

    @app.post("/api/upload")
    def upload_file():
        if "file" not in request.files:
            return jsonify({"error": "No file part in request"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "No file selected"}), 400
        allowed_ext = {".xyz", ".sdf", ".mol", ".pdb", ".mol2"}
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in allowed_ext:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        safe_name = f"{uuid.uuid4().hex[:8]}_{f.filename}"
        dest = os.path.join(UPLOAD_DIR, safe_name)
        f.save(dest)
        return jsonify({"ok": True, "path": os.path.abspath(dest), "filename": f.filename})

    @app.get("/api/molecule-data")
    def molecule_data():
        """Return molecule as SDF/molblock text with 3D coordinates for 3Dmol.js."""
        mol_path = request.args.get("path", "").strip()
        if not mol_path or not os.path.exists(mol_path):
            return jsonify({"error": "File not found"}), 404
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = _rdkit_mol_from_file(mol_path)
            if mol is None:
                return jsonify({"error": "Could not parse molecule"}), 400

            # Ensure 3D coordinates exist
            conf = mol.GetConformer(0) if mol.GetNumConformers() > 0 else None
            needs_3d = conf is None
            if conf is not None:
                # Check if coordinates are all zeros (2D-only)
                pts = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
                if all(abs(p.z) < 1e-6 for p in pts):
                    needs_3d = True

            if needs_3d:
                mol_h = Chem.AddHs(mol)
                result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
                if result == 0:
                    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
                    mol = mol_h
                else:
                    # Fallback: try with random coords
                    params = AllChem.ETKDGv3()
                    params.useRandomCoords = True
                    result = AllChem.EmbedMolecule(mol_h, params)
                    if result == 0:
                        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
                        mol = mol_h

            molblock = Chem.MolToMolBlock(mol)
            return Response(molblock, mimetype="text/plain")
        except Exception as e:
            return jsonify({"error": f"Molecule data failed: {e}"}), 500

    return app


def _rdkit_mol_from_file(path: str):
    """Parse a molecule file with RDKit, returning an RWMol or None."""
    from rdkit import Chem

    ext = os.path.splitext(path)[1].lower()
    mol = None

    if ext == ".xyz":
        mol = Chem.MolFromXYZFile(path)
        if mol is not None:
            try:
                from rdkit.Chem import rdDetermineBonds
                rdDetermineBonds.DetermineBonds(mol)
            except Exception:
                pass
    elif ext in (".sdf", ".mol"):
        mol = Chem.MolFromMolFile(path, removeHs=False)
    elif ext == ".pdb":
        mol = Chem.MolFromPDBFile(path, removeHs=False)
    elif ext == ".mol2":
        mol = Chem.MolFromMol2File(path, removeHs=False)

    return mol


def _open_native_dialog(mode: str) -> str:
    # Imported lazily to avoid GUI backend dependency unless explicitly used.
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    root.attributes("-topmost", True)

    try:
        if mode == "file":
            path = filedialog.askopenfilename(
                title="Select Input Molecule File",
                filetypes=[
                    ("Molecule Files", "*.xyz *.sdf *.mol *.pdb *.mol2"),
                    ("All Files", "*.*"),
                ],
            )
        elif mode == "multiwfn_file":
            path = filedialog.askopenfilename(
                title="Select Multiwfn Executable",
                filetypes=[
                    ("Executable", "*.exe"),
                    ("All Files", "*.*"),
                ],
            )
        else:
            path = filedialog.askdirectory(title="Select Folder", mustexist=True)
    finally:
        root.destroy()

    return str(path or "")


def main():
    host = os.environ.get("KNF_GUI_HOST", "127.0.0.1")
    port = int(os.environ.get("KNF_GUI_PORT", "8787"))
    app = create_app()
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
