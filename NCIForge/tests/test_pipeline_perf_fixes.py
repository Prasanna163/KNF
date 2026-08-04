"""Regression tests for three performance fixes:

1. ``nci_torch.pipeline.run_nci_torch`` must not call ``release_cuda_memory``
   (sync + empty_cache + ipc_collect) on the success path -- only on the
   exception path the CUDA-OOM adaptive fallback relies on.
2. ``snci.compute_snci_and_statistics`` must load the NCI grid payload once,
   not twice (the old ``compute_snci`` + ``compute_nci_statistics`` pair).
3. ``xtb.compute_wbo_from_molden_details`` / ``nci_torch.pipeline.run_nci_torch``
   must be able to share one parsed ``Wavefunction`` instead of each parsing
   the same molden file independently, and doing so must not change results.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from knf_core import snci
from knf_core import xtb as xtb_module
from knf_core.nci_torch import pipeline as nci_pipeline
from knf_core.nci_torch.molden import parse_molden

REPO_ROOT = Path(__file__).resolve().parent.parent
MOLDEN_PATH = REPO_ROOT / "molden.input"


# ---------------------------------------------------------------------------
# Fix 1: release_cuda_memory only on the exception path
# ---------------------------------------------------------------------------

def _fake_fields():
    z = torch.zeros(2, 2, 2)
    return SimpleNamespace(rho=z, rdg=z.clone(), sign_lambda2_rho=z.clone())


def test_release_cuda_memory_not_called_on_success(tmp_path, monkeypatch):
    release_calls = []
    monkeypatch.setattr(nci_pipeline, "release_cuda_memory", lambda *a, **k: release_calls.append(1))
    monkeypatch.setattr(nci_pipeline, "_sync_if_cuda", lambda device: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # This machine's torch build has no real CUDA runtime; stub the
    # device-introspection calls run_nci_torch makes for GPU metadata.
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "fake-gpu")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (0, 0))
    monkeypatch.setattr(
        nci_pipeline,
        "run_nci_engine",
        lambda **kwargs: (_fake_fields(), torch.device("cuda")),
    )

    metadata = nci_pipeline.run_nci_torch(
        molden_path=str(MOLDEN_PATH),
        output_path=str(tmp_path / "out.npz"),
        device="cuda",
    )

    assert metadata["device"] == "cuda"
    assert release_calls == [], "release_cuda_memory must not fire on the success path"


def test_release_cuda_memory_called_on_exception(tmp_path, monkeypatch):
    release_calls = []
    monkeypatch.setattr(nci_pipeline, "release_cuda_memory", lambda *a, **k: release_calls.append(1))
    monkeypatch.setattr(nci_pipeline, "_sync_if_cuda", lambda device: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _raise(**kwargs):
        raise RuntimeError("simulated CUDA OOM")

    monkeypatch.setattr(nci_pipeline, "run_nci_engine", _raise)

    with pytest.raises(RuntimeError):
        nci_pipeline.run_nci_torch(
            molden_path=str(MOLDEN_PATH),
            output_path=str(tmp_path / "out.npz"),
            device="cuda",
        )

    assert release_calls == [1], "release_cuda_memory must still fire when the run fails"


# ---------------------------------------------------------------------------
# Fix 2: single grid-payload load for SNCI + statistics
# ---------------------------------------------------------------------------

def test_compute_snci_and_statistics_loads_payload_once(tmp_path, monkeypatch):
    grid_path = tmp_path / "nci_grid.npz"
    sl2 = np.array([-1.0, -2.0, 3.0, -0.5], dtype=np.float32)
    np.savez(
        grid_path,
        x=np.array([0.0, 1.0]),
        y=np.array([0.0, 1.0]),
        z=np.array([0.0, 1.0]),
        sign_lambda2_rho=sl2,
        rdg=np.zeros_like(sl2),
        output_units=np.array(["bohr"]),
    )

    load_calls = []
    original_load = snci._load_grid_payload

    def counting_load(path):
        load_calls.append(path)
        return original_load(path)

    monkeypatch.setattr(snci, "_load_grid_payload", counting_load)

    snci_val, stats = snci.compute_snci_and_statistics(str(grid_path))

    assert len(load_calls) == 1, "compute_snci_and_statistics must load the grid payload exactly once"
    assert stats["f6"] == 3  # three attractive (negative) points
    assert snci_val == pytest.approx(float(np.sum(-sl2[sl2 < 0.0])))


def test_compute_snci_and_statistics_matches_old_separate_calls(tmp_path):
    grid_path = tmp_path / "nci_grid.npz"
    sl2 = np.array([-1.0, -2.0, 3.0, -0.5, 0.25], dtype=np.float32)
    np.savez(
        grid_path,
        x=np.array([0.0, 1.0]),
        y=np.array([0.0, 1.0]),
        z=np.array([0.0, 1.0]),
        sign_lambda2_rho=sl2,
        rdg=np.zeros_like(sl2),
        output_units=np.array(["bohr"]),
    )

    expected_snci = snci.compute_snci(str(grid_path))
    expected_stats = snci.compute_nci_statistics(str(grid_path))

    combined_snci, combined_stats = snci.compute_snci_and_statistics(str(grid_path))

    assert combined_snci == pytest.approx(expected_snci)
    assert combined_stats == expected_stats


# ---------------------------------------------------------------------------
# Fix 3: shared parsed wavefunction between WBO and NCI stages
# ---------------------------------------------------------------------------

def test_compute_wbo_returns_reusable_wavefunction():
    result = xtb_module.compute_wbo_from_molden_details(str(MOLDEN_PATH))
    wf = result["wavefunction"]
    assert wf is not None
    assert len(wf.basis_functions) > 0

    # A second call given that exact wavefunction must skip re-parsing and
    # produce identical WBO numbers.
    result_reused = xtb_module.compute_wbo_from_molden_details(str(MOLDEN_PATH), wavefunction=wf)
    assert result_reused["max_wbo_global"] == pytest.approx(result["max_wbo_global"])
    assert result_reused["wavefunction"] is wf


def test_run_nci_torch_reuse_matches_fresh_parse(tmp_path):
    wf = parse_molden(str(MOLDEN_PATH), apply_primitive_normalization=False)

    meta_fresh = nci_pipeline.run_nci_torch(
        molden_path=str(MOLDEN_PATH),
        output_path=str(tmp_path / "fresh.npz"),
        device="cpu",
        dtype="float32",
        batch_size=50000,
        eig_batch_size=50000,
    )
    meta_reused = nci_pipeline.run_nci_torch(
        molden_path=str(MOLDEN_PATH),
        output_path=str(tmp_path / "reused.npz"),
        device="cpu",
        dtype="float32",
        batch_size=50000,
        eig_batch_size=50000,
        wavefunction=wf,
    )

    assert meta_fresh["wavefunction_reused"] is False
    assert meta_reused["wavefunction_reused"] is True
    # Reused path must not pay any parse cost.
    assert meta_reused["timings_seconds"]["parse_molden"] < 0.001

    fresh_payload = np.load(tmp_path / "fresh.npz")
    reused_payload = np.load(tmp_path / "reused.npz")
    for key in ("x", "y", "z", "sign_lambda2_rho", "rdg"):
        assert np.array_equal(fresh_payload[key], reused_payload[key]), f"{key} differs between fresh and reused runs"
