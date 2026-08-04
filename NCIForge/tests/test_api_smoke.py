from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from knf_core import api


def _write_xyz(path: Path) -> str:
    path.write_text(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.0 0.9\nH 0.8 0.0 -0.2\n",
        encoding="utf-8",
    )
    return str(path)


def setup_function():
    with api._JOB_LOCK:
        api._JOB_STORE.clear()


def test_api_health_smoke():
    client = TestClient(api.app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == api.API_TITLE
    assert payload["version"] == api.API_VERSION
    assert payload["job_counts"][api.JOB_STATUS_QUEUED] == 0


def test_api_gpu_preflight_never_prompts(monkeypatch):
    calls = []

    monkeypatch.setattr(
        api.engine_gpu,
        "ensure_cuda_runtime_for_gpu_mode",
        lambda allow_prompt=True: calls.append(allow_prompt),
    )
    monkeypatch.setattr(
        api.engine_gpu,
        "resolve_cpu_backend_when_torch_missing",
        lambda _args: pytest.fail("CUDA API preflight used the interactive resolver"),
    )
    monkeypatch.setattr(api.first_run, "ensure_first_run_setup", lambda **_kwargs: None)
    monkeypatch.setattr(api.utils, "ensure_external_tools_in_path", lambda **_kwargs: None)
    monkeypatch.setattr(api.utils, "resolve_external_tool_command", lambda _name: "available")

    api._preflight(api._to_engine_options(api.RunOptions(nci_device="cuda")))

    assert calls == [False]


def test_api_enables_scdi_for_desktop_jobs_by_default():
    options = api.RunOptions(batch_id="batch-a")
    runtime = api._to_engine_options(options)

    assert options.compute_scdi is True
    assert runtime.compute_scdi is True


def test_api_normalizes_scdi_variance_within_each_batch(tmp_path):
    jobs = []
    for job_id, snci_value, variance in (
        ("a", 10.0, 0.000002),
        ("b", 20.0, 0.000003),
        ("c", 30.0, 0.000006),
    ):
        result_dir = tmp_path / job_id
        result_dir.mkdir()
        (result_dir / "knf.json").write_text(
            json.dumps({
                "SNCI": snci_value,
                "SCDI": None,
                "SCDI_variance": variance,
                "KNF_vector": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            }),
            encoding="utf-8",
        )
        jobs.append({
            "job_id": job_id,
            "status": api.JOB_STATUS_SUCCEEDED,
            "kind": "upload",
            "created_at": "2026-01-01T00:00:00Z",
            "result_dir": str(result_dir),
            "options": {"batch_id": "batch-a"},
        })

    summaries = api._summaries_with_batch_normalization(jobs)
    by_id = {summary["job_id"]: summary for summary in summaries}

    assert by_id["a"]["knf_json"]["SCDI_Norm"] == pytest.approx(1.0)
    assert by_id["b"]["knf_json"]["SCDI_Norm"] == pytest.approx(0.75)
    assert by_id["c"]["knf_json"]["SCDI_Norm"] == pytest.approx(0.0)
    metadata = by_id["b"]["batch_normalization"]
    assert metadata["SCDI_method"] == "SCDI_variance_inverse_minmax"
    assert metadata["SCDI_variance_min"] == pytest.approx(0.000002)
    assert metadata["SCDI_variance_max"] == pytest.approx(0.000006)
    assert metadata["valid_SCDI_count"] == 3
    assert metadata["state"] == "final"


def test_api_equal_batch_variance_maps_to_midpoint(tmp_path):
    jobs = []
    for job_id in ("a", "b"):
        result_dir = tmp_path / job_id
        result_dir.mkdir()
        (result_dir / "knf.json").write_text(
            json.dumps({"SNCI": 1.0, "SCDI_variance": 0.000002}),
            encoding="utf-8",
        )
        jobs.append({
            "job_id": job_id,
            "status": api.JOB_STATUS_SUCCEEDED,
            "kind": "upload",
            "created_at": "2026-01-01T00:00:00Z",
            "result_dir": str(result_dir),
            "options": {"batch_id": "equal-batch"},
        })

    summaries = api._summaries_with_batch_normalization(jobs)
    assert all(summary["knf_json"]["SCDI_Norm"] == 0.5 for summary in summaries)


def test_api_gpu_preflight_reports_missing_runtime_without_stdin(monkeypatch):
    def fail_without_prompt(allow_prompt=True):
        assert allow_prompt is False
        raise RuntimeError("CUDA PyTorch runtime is unavailable")

    monkeypatch.setattr(
        api.engine_gpu,
        "ensure_cuda_runtime_for_gpu_mode",
        fail_without_prompt,
    )

    with pytest.raises(HTTPException) as exc_info:
        api._preflight(api._to_engine_options(api.RunOptions(nci_device="cuda")))

    assert exc_info.value.status_code == 400
    assert "CUDA PyTorch runtime is unavailable" in exc_info.value.detail


def test_api_path_job_happy_path(monkeypatch, tmp_path):
    input_path = _write_xyz(tmp_path / "mol.xyz")
    output_root = tmp_path / "Results"

    def fake_preflight(runtime_args):
        return None

    def fake_process_file(file_path, runtime_args, output_root=None):
        assert runtime_args.sp is True
        result_dir = Path(output_root) / Path(file_path).stem
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "knf.json").write_text(
            json.dumps({"KNF_vector": [1, 2, 3, 4, 5, 6, 7, 8, 9]}),
            encoding="utf-8",
        )
        (result_dir / "output.txt").write_text("ok", encoding="utf-8")
        return True, None, 0.01

    monkeypatch.setattr(api, "_preflight", fake_preflight)
    monkeypatch.setattr(api, "process_file", fake_process_file)

    client = TestClient(api.app)
    response = client.post(
        "/jobs/path",
        json={
            "input_path": input_path,
            "output_dir": str(output_root),
            "force": True,
            "sp": True,
        },
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]

    final_payload = None
    for _ in range(40):
        poll = client.get(f"/jobs/{job_id}")
        assert poll.status_code == 200
        final_payload = poll.json()
        if final_payload["status"] == api.JOB_STATUS_SUCCEEDED:
            break
        time.sleep(0.05)

    assert final_payload is not None
    assert final_payload["status"] == api.JOB_STATUS_SUCCEEDED
    assert final_payload["kind"] == "path"
    assert final_payload["managed_workspace"] is False
    assert final_payload["error"] is None
    assert final_payload["knf_json"]["KNF_vector"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert any(item["name"] == "knf.json" for item in final_payload["artifacts"])
