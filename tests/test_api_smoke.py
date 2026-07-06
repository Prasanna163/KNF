from __future__ import annotations

import json
import time
from pathlib import Path

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


def test_api_path_job_happy_path(monkeypatch, tmp_path):
    input_path = _write_xyz(tmp_path / "mol.xyz")
    output_root = tmp_path / "Results"

    def fake_preflight(runtime_args):
        return None

    def fake_process_file(file_path, runtime_args, output_root=None):
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
