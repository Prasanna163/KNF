from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import first_run, utils
from .engine import dependencies as engine_dependencies
from .engine import gpu as engine_gpu
from .engine.constants import CLI_VERSION
from .engine.discovery import resolve_results_root
from .engine.processing import process_file
from .engine.types import RunOptions as EngineRunOptions

API_TITLE = "NCIForge API"
API_VERSION = CLI_VERSION
JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_SUCCEEDED = "succeeded"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_CANCELLED = "cancelled"

_JOB_STATUSES = {
    JOB_STATUS_QUEUED,
    JOB_STATUS_RUNNING,
    JOB_STATUS_SUCCEEDED,
    JOB_STATUS_FAILED,
    JOB_STATUS_CANCELLED,
}

_JOB_STORE: Dict[str, Dict[str, Any]] = {}
_JOB_LOCK = threading.Lock()
_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, int(os.environ.get("NCIFORGE_API_WORKERS", "1"))),
    thread_name_prefix="nciforge-api",
)


class RunOptions(BaseModel):
    charge: int = 0
    spin: int = 1
    water: bool = False
    force: bool = False
    clean: bool = False
    debug: bool = False
    full_files: bool = False
    nci_backend: Literal["torch", "multiwfn"] = "torch"
    nci_grid_spacing: float = 0.2
    nci_grid_padding: float = 3.0
    nci_device: Literal["cpu", "cuda"] = "cpu"
    nci_dtype: str = "float32"
    nci_batch_size: int = 250000
    nci_eig_batch_size: int = 200000
    nci_rho_floor: float = 1e-12
    nci_apply_primitive_norm: bool = False
    scdi_var_min: Optional[float] = None
    scdi_var_max: Optional[float] = None
    wbo_mode: Literal["native", "xtb"] = "native"
    preopt: Literal["uff", "geoinit"] = "geoinit"
    xtb_engine: Literal["xtb", "xtbx", "auto"] = "xtbx"
    xtb_gpu_atoms: int = 350
    output_dir: Optional[str] = None
    multiwfn_path: Optional[str] = None


class PathRunRequest(RunOptions):
    input_path: str = Field(..., min_length=1)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return os.path.abspath(os.path.expanduser(path))


def _model_dump(model: BaseModel, **kwargs) -> Dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(**kwargs)
    return model.dict(**kwargs)


def _to_engine_options(options: RunOptions) -> EngineRunOptions:
    backend = (options.nci_backend or "torch").strip().lower()
    device = (options.nci_device or "cpu").strip().lower()
    if backend == "multiwfn":
        device = "cpu"

    return EngineRunOptions(
        charge=options.charge,
        spin=options.spin,
        water=bool(options.water),
        force=bool(options.force),
        clean=bool(options.clean),
        debug=bool(options.debug),
        full_files=bool(options.full_files),
        nci_backend=backend,
        nci_grid_spacing=options.nci_grid_spacing,
        nci_grid_padding=options.nci_grid_padding,
        nci_device=device,
        nci_dtype=options.nci_dtype,
        nci_batch_size=options.nci_batch_size,
        nci_eig_batch_size=options.nci_eig_batch_size,
        nci_rho_floor=options.nci_rho_floor,
        nci_apply_primitive_norm=bool(options.nci_apply_primitive_norm),
        scdi_var_min=options.scdi_var_min,
        scdi_var_max=options.scdi_var_max,
        wbo_mode=options.wbo_mode,
        preopt=options.preopt,
        xtb_engine=options.xtb_engine,
        xtb_gpu_atoms=options.xtb_gpu_atoms,
        output_dir=options.output_dir,
        multiwfn_path=options.multiwfn_path,
    )


def _preflight(runtime_args: EngineRunOptions) -> None:
    try:
        engine_gpu.resolve_cpu_backend_when_torch_missing(runtime_args)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        first_run.ensure_first_run_setup(
            force=False,
            multiwfn_path=getattr(runtime_args, "multiwfn_path", None),
            require_multiwfn=(runtime_args.nci_backend == "multiwfn"),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"First-run setup failed: {exc}",
        ) from exc

    missing_tools: List[str] = []
    utils.ensure_external_tools_in_path(persist=False)
    if not utils.resolve_external_tool_command("obabel"):
        missing_tools.append("obabel")
    xtb_engine = getattr(runtime_args, "xtb_engine", "xtbx")
    if xtb_engine in ("xtb", "auto") and not utils.resolve_external_tool_command("xtb"):
        missing_tools.append("xtb")
    if xtb_engine in ("xtbx", "auto") and not engine_dependencies.xtbx_available():
        missing_tools.append("xtbx")
    if runtime_args.nci_backend == "multiwfn":
        utils.ensure_multiwfn_in_path(explicit_path=getattr(runtime_args, "multiwfn_path", None))
        if not shutil.which("Multiwfn") and not shutil.which("Multiwfn.exe"):
            missing_tools.append("Multiwfn")

    if missing_tools:
        raise HTTPException(
            status_code=503,
            detail=f"Missing required external tools: {', '.join(missing_tools)}",
        )

    if runtime_args.nci_backend == "torch" and runtime_args.nci_device == "cuda":
        try:
            engine_gpu.ensure_cuda_runtime_for_gpu_mode(allow_prompt=False)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc


def _job_workspace(job_id: str) -> str:
    root = Path(tempfile.gettempdir()) / "nciforge-api" / job_id
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def _result_dir(output_root: str, input_path: str) -> str:
    return os.path.join(os.path.abspath(output_root), Path(input_path).stem)


def _parse_json_file(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        return {"error": f"Could not parse JSON: {exc}", "path": path}


def _list_artifacts(result_dir: str, job_id: str) -> List[Dict[str, Any]]:
    if not os.path.isdir(result_dir):
        return []

    artifacts: List[Dict[str, Any]] = []
    for entry in sorted(Path(result_dir).iterdir()):
        if not entry.is_file():
            continue
        artifacts.append(
            {
                "name": entry.name,
                "size_bytes": entry.stat().st_size,
                "download_url": f"/jobs/{job_id}/download/{entry.name}",
            }
        )
    return artifacts


def _preferred_json_path(result_dir: str, water: bool) -> str:
    candidates = ["knf.json", "knf_water.json"] if not water else ["knf_water.json", "knf.json"]
    for name in candidates:
        candidate = os.path.join(result_dir, name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(result_dir, candidates[0])


def _preferred_text_path(result_dir: str, water: bool) -> str:
    candidates = ["output.txt", "output_water.txt"] if not water else ["output_water.txt", "output.txt"]
    for name in candidates:
        candidate = os.path.join(result_dir, name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(result_dir, candidates[0])


def _job_summary(job: Dict[str, Any]) -> Dict[str, Any]:
    result_dir = job.get("result_dir") or ""
    water = bool(job.get("options", {}).get("water", False))
    knf_json_path = _preferred_json_path(result_dir, water)
    output_txt_path = _preferred_text_path(result_dir, water)
    return {
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "kind": job.get("kind"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "elapsed_seconds": job.get("elapsed_seconds"),
        "input_path": job.get("input_path"),
        "output_root": job.get("output_root"),
        "result_dir": result_dir,
        "managed_workspace": bool(job.get("managed_workspace", False)),
        "error": job.get("error"),
        "options": job.get("options", {}),
        "artifacts": _list_artifacts(result_dir, str(job.get("job_id", ""))) if result_dir else [],
        "knf_json_path": knf_json_path,
        "knf_json": _parse_json_file(knf_json_path),
        "output_txt_path": output_txt_path,
    }


def _store_job(job: Dict[str, Any]) -> None:
    with _JOB_LOCK:
        _JOB_STORE[job["job_id"]] = job


def _get_job(job_id: str) -> Dict[str, Any]:
    with _JOB_LOCK:
        job = _JOB_STORE.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job


def _update_job(job_id: str, **patch: Any) -> None:
    with _JOB_LOCK:
        job = _JOB_STORE.get(job_id)
        if job is None:
            return
        job.update(patch)


def _run_job(job_id: str, input_path: str, output_root: str, options: RunOptions) -> None:
    started_at = _utc_now()
    _update_job(job_id, status=JOB_STATUS_RUNNING, started_at=started_at)
    runtime_args = _to_engine_options(options)

    try:
        _preflight(runtime_args)
        success, error, elapsed = process_file(input_path, runtime_args, output_root=output_root)
        result_dir = _result_dir(output_root, input_path)
        summary = _job_summary(
            {
                "job_id": job_id,
                "status": JOB_STATUS_SUCCEEDED if success else JOB_STATUS_FAILED,
                "kind": "path" if _get_job(job_id).get("kind") == "path" else "upload",
                "created_at": _get_job(job_id).get("created_at"),
                "started_at": started_at,
                "finished_at": _utc_now(),
                "elapsed_seconds": elapsed,
                "input_path": input_path,
                "output_root": output_root,
                "result_dir": result_dir,
                "managed_workspace": _get_job(job_id).get("managed_workspace", False),
                "error": error,
                "options": _model_dump(options),
            }
        )
        if not success:
            raise RuntimeError(error or "The KNF pipeline failed.")
        _update_job(
            job_id,
            status=JOB_STATUS_SUCCEEDED,
            finished_at=_utc_now(),
            elapsed_seconds=elapsed,
            error=None,
            result=summary,
        )
    except Exception as exc:
        _update_job(
            job_id,
            status=JOB_STATUS_FAILED,
            finished_at=_utc_now(),
            error=str(exc),
        )


def _submit_job(
    *,
    kind: str,
    input_path: str,
    output_root: str,
    options: RunOptions,
    managed_workspace: bool,
) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex
    record = {
        "job_id": job_id,
        "kind": kind,
        "status": JOB_STATUS_QUEUED,
        "created_at": _utc_now(),
        "started_at": None,
        "finished_at": None,
        "elapsed_seconds": None,
        "input_path": input_path,
        "output_root": os.path.abspath(output_root),
        "result_dir": _result_dir(output_root, input_path),
        "managed_workspace": managed_workspace,
        "error": None,
        "result": None,
        "options": _model_dump(options),
        "workspace": os.path.dirname(output_root) if managed_workspace else None,
    }
    _store_job(record)
    _EXECUTOR.submit(_run_job, job_id, input_path, output_root, options)
    return _job_summary(record)


def _options_from_payload(payload: Dict[str, Any]) -> RunOptions:
    try:
        return RunOptions(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid run options: {exc}") from exc


def _resolve_path_job_output_root(input_path: str, output_dir: Optional[str]) -> str:
    if output_dir:
        return _normalize_path(output_dir) or output_dir
    return resolve_results_root(input_path, None)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    _EXECUTOR.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "status": "ready",
        "endpoints": [
            "/health",
            "/jobs",
            "/jobs/path",
            "/jobs/upload",
            "/jobs/{job_id}",
        ],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    with _JOB_LOCK:
        counts: Dict[str, int] = {status: 0 for status in _JOB_STATUSES}
        for job in _JOB_STORE.values():
            status = str(job.get("status") or "")
            if status in counts:
                counts[status] += 1

    return {
        "status": "ok",
        "service": API_TITLE,
        "version": API_VERSION,
        "job_counts": counts,
    }


@app.get("/jobs")
def list_jobs() -> Dict[str, Any]:
    with _JOB_LOCK:
        jobs = sorted(_JOB_STORE.values(), key=lambda item: item.get("created_at") or "", reverse=True)
    return {"jobs": [_job_summary(job) for job in jobs]}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    return _job_summary(_get_job(job_id))


@app.get("/jobs/{job_id}/download/{artifact_name}")
def download_artifact(job_id: str, artifact_name: str) -> FileResponse:
    job = _get_job(job_id)
    result_dir = job.get("result_dir")
    if not result_dir:
        raise HTTPException(status_code=404, detail="Job has no result directory yet")

    safe_name = Path(artifact_name).name
    file_path = os.path.join(result_dir, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(file_path, filename=safe_name)


@app.post("/jobs/path")
def submit_path_job(request: PathRunRequest) -> Dict[str, Any]:
    input_path = _normalize_path(request.input_path) or request.input_path
    if not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail=f"Input path not found: {input_path}")

    if os.path.isdir(input_path):
        raise HTTPException(
            status_code=400,
            detail="Directory inputs are not supported by the initial API wrapper. "
            "Use a single molecular file or add a batch-specific endpoint next.",
        )

    output_root = _resolve_path_job_output_root(input_path, request.output_dir)
    return _submit_job(
        kind="path",
        input_path=input_path,
        output_root=output_root,
        options=RunOptions(**_model_dump(request, exclude={"input_path"})),
        managed_workspace=False,
    )


@app.post("/jobs/upload")
async def submit_upload_job(
    file: UploadFile = File(...),
    options_json: str = Form("{}"),
) -> Dict[str, Any]:
    try:
        payload = json.loads(options_json or "{}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"options_json must be valid JSON: {exc}") from exc

    options = _options_from_payload(payload)
    job_id = uuid.uuid4().hex
    workspace = _job_workspace(job_id)
    input_dir = os.path.join(workspace, "input")
    os.makedirs(input_dir, exist_ok=True)

    safe_name = Path(file.filename or "input.mol").name
    input_path = os.path.join(input_dir, safe_name)
    data = await file.read()
    with open(input_path, "wb") as handle:
        handle.write(data)

    output_root = _normalize_path(options.output_dir) or os.path.join(workspace, "Results")
    record = {
        "job_id": job_id,
        "kind": "upload",
        "status": JOB_STATUS_QUEUED,
        "created_at": _utc_now(),
        "started_at": None,
        "finished_at": None,
        "elapsed_seconds": None,
        "input_path": input_path,
        "output_root": os.path.abspath(output_root),
        "result_dir": _result_dir(output_root, input_path),
        "managed_workspace": True,
        "error": None,
        "result": None,
        "options": _model_dump(options),
        "workspace": workspace,
    }
    _store_job(record)
    _EXECUTOR.submit(_run_job, job_id, input_path, output_root, options)
    return _job_summary(record)


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> Dict[str, Any]:
    job = _get_job(job_id)
    if not job.get("managed_workspace"):
        raise HTTPException(
            status_code=409,
            detail="Only uploaded jobs can be deleted safely from the API workspace.",
        )

    workspace = job.get("workspace")
    if workspace and os.path.isdir(workspace):
        shutil.rmtree(workspace, ignore_errors=True)

    with _JOB_LOCK:
        _JOB_STORE.pop(job_id, None)

    return {"job_id": job_id, "deleted": True}


def main() -> None:
    import argparse

    try:
        import uvicorn
    except Exception as exc:
        raise SystemExit(
            "FastAPI support requires the API extra. Install it with: "
            "pip install -e \".[api]\""
        ) from exc

    parser = argparse.ArgumentParser(description="Run the NCIForge FastAPI service")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address for the API server")
    parser.add_argument("--port", type=int, default=8000, help="Bind port for the API server")
    parser.add_argument("--reload", action="store_true", help="Enable Uvicorn auto-reload")
    parser.add_argument("--log-level", default="info", help="Uvicorn log level")
    args = parser.parse_args()

    uvicorn.run(
        "knf_core.api:app",
        host=args.host,
        port=args.port,
        reload=bool(args.reload),
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
