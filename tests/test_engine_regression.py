"""First regression smoke test for the configurable pre-opt / xTB engine wiring.

Covers three layers, cheapest first:

1. Pure-unit tests (always run): the ``xtbx`` launcher resolver, the pre-opt
   dispatcher, the atom-count helper, and the ``auto`` engine size-gate.
2. GeoInit drop-in (using the bundled ``geoinit`` package):
   ``run_geoinit_preopt`` must be a safe in-place drop-in for ``run_uff_preopt``
   — same atom count, still-parseable .xyz, finite coordinates.
3. End-to-end pipeline regression (opt-in): set ``KNF_RUN_XTB_TESTS=1`` and have
   ``xtb`` on PATH. Runs the full pipeline on a water dimer and asserts a valid
   9-D KNF vector is produced for the default (geoinit+xtb) and for the legacy
   UFF warm-start path.

Run just the fast unit layer:
    pytest tests/test_engine_regression.py -q
Run the full end-to-end smoke:
    KNF_RUN_XTB_TESTS=1 pytest tests/test_engine_regression.py -q      # bash
    $env:KNF_RUN_XTB_TESTS=1; pytest tests/test_engine_regression.py -q  # PowerShell
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from knf_core import wrapper
from knf_core.pipeline import KNFPipeline


# A real (small) noncovalent system: the water dimer. Two fragments exercises
# GeoInit's complex-aware path and is a faithful stand-in for the CO2-capture
# dataset while staying sub-second under xTB.
WATER_DIMER_XYZ = """6
water dimer
O   -1.551007  -0.114520   0.000000
H   -1.934259   0.762503   0.000000
H   -0.599677   0.040712   0.000000
O    1.350625   0.111469   0.000000
H    1.680398  -0.373741  -0.758561
H    1.680398  -0.373741   0.758561
"""

_GEOINIT_AVAILABLE = True
try:  # pragma: no cover - import probe
    import geoinit  # noqa: F401
except Exception:
    _GEOINIT_AVAILABLE = False

_XTB_AVAILABLE = shutil.which("xtb") is not None
_RUN_XTB = os.environ.get("KNF_RUN_XTB_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}


def _write_xyz(path, contents=WATER_DIMER_XYZ):
    path.write_text(contents, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# 1. Launcher resolution
# ---------------------------------------------------------------------------

def test_xtb_invocation_missing_launcher_raises():
    with pytest.raises(FileNotFoundError):
        wrapper._xtb_invocation("definitely-not-a-real-xtb-launcher")


def test_xtb_invocation_wraps_cmd_shim(monkeypatch):
    """.cmd/.bat launchers (e.g. xtbx.cmd) must be invoked through ``cmd /c``."""
    monkeypatch.setattr(wrapper.os, "name", "nt")
    monkeypatch.setattr(wrapper.shutil, "which", lambda name: r"C:\tools\xtbx.cmd")
    assert wrapper._xtb_invocation("xtbx") == ["cmd", "/c", r"C:\tools\xtbx.cmd"]


def test_xtb_invocation_native_exe_passthrough(monkeypatch):
    monkeypatch.setattr(wrapper.os, "name", "nt")
    monkeypatch.setattr(wrapper.shutil, "which", lambda name: r"C:\xtb\bin\xtb.exe")
    assert wrapper._xtb_invocation("xtb") == [r"C:\xtb\bin\xtb.exe"]


@pytest.mark.skipif(not _XTB_AVAILABLE, reason="xtb not on PATH")
def test_xtb_invocation_resolves_real_xtb():
    argv = wrapper._xtb_invocation("xtb")
    assert isinstance(argv, list) and argv
    assert "xtb" in argv[-1].lower()


# ---------------------------------------------------------------------------
# 2. Pre-opt dispatch
# ---------------------------------------------------------------------------

def test_run_preopt_rejects_unknown_engine(tmp_path):
    xyz = _write_xyz(tmp_path / "m.xyz")
    with pytest.raises(ValueError):
        wrapper.run_preopt(xyz, engine="not-an-engine")


def test_geoinit_is_bundled_with_nciforge_checkout():
    import geoinit

    geoinit_path = Path(geoinit.__file__).resolve()
    repo_geoinit = Path(__file__).resolve().parents[1] / "geoinit"
    assert geoinit_path == (repo_geoinit / "__init__.py").resolve()


def test_default_preopt_engine_is_geoinit(tmp_path):
    xyz = _write_xyz(tmp_path / "m.xyz")
    pipe = KNFPipeline(input_file=xyz, output_root=str(tmp_path / "Results"))
    assert pipe.preopt_engine == "geoinit"


# ---------------------------------------------------------------------------
# 3. Atom count + auto engine size-gate
# ---------------------------------------------------------------------------

def test_atom_count_xyz(tmp_path):
    xyz = _write_xyz(tmp_path / "m.xyz")
    assert KNFPipeline._atom_count_xyz(xyz) == 6
    assert KNFPipeline._atom_count_xyz(str(tmp_path / "missing.xyz")) == 0


def _pipeline_for(tmp_path, xyz, **kwargs):
    return KNFPipeline(input_file=xyz, output_root=str(tmp_path / "Results"), **kwargs)


def test_resolve_xtb_cmd_explicit_engines(tmp_path):
    xyz = _write_xyz(tmp_path / "m.xyz")
    assert _pipeline_for(tmp_path, xyz, xtb_engine="xtb")._resolve_xtb_cmd(xyz) == "xtb"
    assert _pipeline_for(tmp_path, xyz, xtb_engine="xtbx")._resolve_xtb_cmd(xyz) == "xtbx"


def test_resolve_xtb_cmd_auto_gate(tmp_path):
    small = _write_xyz(tmp_path / "small.xyz")  # 6 atoms
    big_header = "400\nbig\n" + "H 0.0 0.0 0.0\n" * 400
    big = _write_xyz(tmp_path / "big.xyz", big_header)

    pipe = _pipeline_for(tmp_path, small, xtb_engine="auto", xtb_gpu_atom_cutoff=350)
    assert pipe._resolve_xtb_cmd(small) == "xtb"     # below cutoff -> native
    assert pipe._resolve_xtb_cmd(big) == "xtbx"      # at/above cutoff -> GPU front-end


# ---------------------------------------------------------------------------
# 4. GeoInit drop-in safety
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _GEOINIT_AVAILABLE, reason="geoinit package not installed")
def test_geoinit_preopt_is_safe_dropin(tmp_path):
    xyz = _write_xyz(tmp_path / "dimer.xyz")
    returned = wrapper.run_geoinit_preopt(xyz)
    assert returned == xyz

    # File must remain a well-formed .xyz with the same atom count and finite coords.
    lines = [ln for ln in open(xyz, encoding="utf-8").read().splitlines() if ln.strip()]
    assert int(lines[0].strip()) == 6
    atom_lines = lines[2:8]
    assert len(atom_lines) == 6
    for ln in atom_lines:
        parts = ln.split()
        assert len(parts) >= 4
        for c in parts[1:4]:
            assert abs(float(c)) < 1e6  # finite, no NaN/inf


# ---------------------------------------------------------------------------
# 5. End-to-end pipeline regression (opt-in)
# ---------------------------------------------------------------------------

def _assert_valid_knf_json(results_dir):
    import json
    knf_path = os.path.join(results_dir, "dimer", "knf.json")
    assert os.path.exists(knf_path), f"missing {knf_path}"
    with open(knf_path, encoding="utf-8") as f:
        payload = json.load(f)
    vector = payload.get("KNF_vector")
    assert isinstance(vector, list) and len(vector) == 9
    import math
    assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector)


@pytest.mark.skipif(not (_RUN_XTB and _XTB_AVAILABLE), reason="set KNF_RUN_XTB_TESTS=1 and have xtb on PATH")
def test_pipeline_end_to_end_uff_xtb(tmp_path):
    xyz = _write_xyz(tmp_path / "dimer.xyz")
    results = str(tmp_path / "Results")
    KNFPipeline(
        input_file=xyz,
        output_root=results,
        force=True,
        preopt_engine="uff",
        xtb_engine="xtb",
        nci_backend="torch",
        nci_device="cpu",
    ).run()
    _assert_valid_knf_json(results)


@pytest.mark.skipif(
    not (_RUN_XTB and _XTB_AVAILABLE and _GEOINIT_AVAILABLE),
    reason="set KNF_RUN_XTB_TESTS=1 and have xtb + geoinit available",
)
def test_pipeline_end_to_end_geoinit_xtb(tmp_path):
    xyz = _write_xyz(tmp_path / "dimer.xyz")
    results = str(tmp_path / "Results")
    KNFPipeline(
        input_file=xyz,
        output_root=results,
        force=True,
        preopt_engine="geoinit",
        xtb_engine="xtb",
        nci_backend="torch",
        nci_device="cpu",
    ).run()
    _assert_valid_knf_json(results)
