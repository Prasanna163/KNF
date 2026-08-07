"""First regression smoke test for the configurable pre-opt / xTB engine wiring.

Covers three layers, cheapest first:

1. Pure-unit tests (always run): the ``xtbx`` launcher resolver, the pre-opt
   dispatcher, the atom-count helper, and the ``auto`` engine size-gate.
2. GeoInit drop-in (using the bundled ``geoinit`` package):
   ``run_geoinit_preopt`` must be a safe in-place drop-in for ``run_uff_preopt``
   — same atom count, still-parseable .xyz, finite coordinates.
3. End-to-end pipeline regression (opt-in): set ``KNF_RUN_XTB_TESTS=1`` and have
   ``xtb`` on PATH. Runs the full pipeline on a water dimer and asserts a valid
   9-D KNF vector is produced for the explicit geoinit+xtb and legacy UFF+xtb
   paths.

Run just the fast unit layer:
    pytest tests/test_engine_regression.py -q
Run the full end-to-end smoke:
    KNF_RUN_XTB_TESTS=1 pytest tests/test_engine_regression.py -q      # bash
    $env:KNF_RUN_XTB_TESTS=1; pytest tests/test_engine_regression.py -q  # PowerShell
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import csv
import zipfile
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from knf_core.engine import jobs as engine_jobs
from knf_core.engine import processing as engine_processing
from knf_core.engine.events import EventKind
from knf_core.engine.types import RunOptions
from knf_core.cli import app as cli_app
from knf_core.cli import commands as cli_commands
from knf_core.cli import interactive as cli_interactive
from knf_core.cli.argv_preprocess import normalize_argv
from knf_core.cli.options import apply_execution_shortcuts, build_run_options, validate_flag_combinations
from knf_core import knf_vector, wrapper, xtb
from knf_core import main as core_main
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


def _make_fake_xtbx_runtime(path: Path, *, full_gpu: bool = False) -> Path:
    (path / "bin").mkdir(parents=True)
    (path / "lib").mkdir()
    (path / "params").mkdir()
    for rel in (
        "bin/xtb-cpu.exe",
        "bin/xtb.exe",
        "params/param_gfn2-xtb.txt",
    ):
        (path / rel).write_text("placeholder", encoding="utf-8")
    if full_gpu:
        for name in ("cublas64_12.dll", "cublasLt64_12.dll", "cusolver64_11.dll"):
            (path / "lib" / name).write_text("placeholder", encoding="utf-8")
    return path

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


def _fake_knf_payload(index: int) -> dict:
    vector = [round(index + offset / 10, 6) for offset in range(1, 10)]
    return {
        "KNF_vector": vector,
        "SNCI": round(0.1 * index, 6),
        "SCDI": round(0.2 * index, 6),
        "SCDI_variance": round(0.03 * index, 6),
        "metadata": {"f2_defined": True},
    }


def _normalize_batch_json(path: Path) -> dict:
    def normalize_generated_values(value):
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                if key == "calibration_id" and isinstance(item, str) and item.startswith("KUID-MVP-1.0-"):
                    out[key] = "<kuid-calibration-id>"
                elif isinstance(item, str) and ("/" in item or "\\" in item):
                    out[key] = Path(item).name
                else:
                    out[key] = normalize_generated_values(item)
            return out
        if isinstance(value, list):
            return [normalize_generated_values(item) for item in value]
        return value

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload = normalize_generated_values(payload)
    payload.pop("generated_at_utc", None)
    payload["results_root"] = "<results>"
    payload["input_directory"] = "<input>"
    if isinstance(payload.get("summary"), dict):
        payload["summary"]["total_time_seconds"] = "<time>"
    for collection_name in ("records", "knf_results"):
        for item in payload.get(collection_name, []) or []:
            if "input_file" in item:
                item["input_file"] = Path(item["input_file"]).name
            if "result_dir" in item:
                item["result_dir"] = Path(item["result_dir"]).name
            if "elapsed_seconds" in item:
                item["elapsed_seconds"] = "<time>"
    normalizers = payload.get("normalization_and_quadrants") or {}
    for key in ("quadrant_json", "quadrant_plot_png"):
        if normalizers.get(key):
            normalizers[key] = Path(normalizers[key]).name
    return payload


# ---------------------------------------------------------------------------
# 1. Launcher resolution
# ---------------------------------------------------------------------------

def test_xtb_invocation_missing_launcher_raises():
    with pytest.raises(FileNotFoundError):
        wrapper._xtb_invocation("definitely-not-a-real-xtb-launcher")


def test_xtb_invocation_wraps_cmd_shim(monkeypatch):
    """.cmd/.bat launchers must be invoked through ``cmd /c`` on Windows."""
    monkeypatch.setattr(wrapper.os, "name", "nt")
    monkeypatch.setattr(wrapper.shutil, "which", lambda name: r"C:\tools\xtb.cmd")
    assert wrapper._xtb_invocation("xtb") == ["cmd", "/c", r"C:\tools\xtb.cmd"]


def test_xtb_invocation_uses_bundled_xtbx():
    argv = wrapper._xtb_invocation("xtbx")
    assert argv[0] == sys.executable
    assert Path(argv[1]).name == "_runner.py"
    assert Path(argv[1]).parent.name == "nciforge_xtbx"


def test_xtb_invocation_native_exe_passthrough(monkeypatch):
    monkeypatch.setattr(wrapper.os, "name", "nt")
    monkeypatch.setattr(wrapper.shutil, "which", lambda name: r"C:\xtb\bin\xtb.exe")
    assert wrapper._xtb_invocation("xtb") == [r"C:\xtb\bin\xtb.exe"]


@pytest.mark.skipif(not _XTB_AVAILABLE, reason="xtb not on PATH")
def test_xtb_invocation_resolves_real_xtb():
    argv = wrapper._xtb_invocation("xtb")
    assert isinstance(argv, list) and argv
    assert "xtb" in argv[-1].lower()


def test_xtb_wrapper_can_force_xtbx_gpu(monkeypatch, tmp_path):
    xyz = _write_xyz(tmp_path / "m.xyz")
    calls = []

    monkeypatch.setattr(wrapper, "_xtb_invocation", lambda xtb_cmd="xtb": ["xtbx-runner"])

    def fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, errors=None, check=None):
        calls.append(list(cmd))
        Path(cwd, "xtbopt.xyz").write_text("6\nfake opt\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)
    wrapper.run_xtb_opt(xyz, xtb_cmd="xtbx", force_gpu=True)

    assert "--gpu" in calls[0]
    assert calls[0][0] == "xtbx-runner"


def test_xtb_streaming_progress_writes_log_and_emits_events(tmp_path):
    log_path = tmp_path / "xtb.log"
    events = []

    return_code = wrapper._run_xtb_streaming(
        [
            sys.executable,
            "-c",
            "print('cycle 7 energy -1.234 grad 0.056', flush=True)",
        ],
        cwd=str(tmp_path),
        log_path=str(log_path),
        progress_callback=events.append,
        stage="opt",
        launcher="xtbx",
        use_gpu=True,
    )

    assert return_code == 0
    assert "cycle 7 energy -1.234 grad 0.056" in log_path.read_text(encoding="utf-8")
    output_events = [event for event in events if event.get("event") == "output"]
    assert output_events
    assert output_events[-1]["cycle"] == 7
    assert output_events[-1]["energy"] == "-1.234"
    assert output_events[-1]["gradient"] == "0.056"
    assert "cycle 7" in output_events[-1]["message"]
    assert "Energy -1.234 Eh" in output_events[-1]["message"]


def test_xtb_sp_streaming_success_does_not_raise(monkeypatch, tmp_path):
    """A successful streamed SP run must be accepted like the non-streamed path."""
    xyz = _write_xyz(tmp_path / "input.xyz")
    calls = []

    def fake_streaming(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        return 0

    monkeypatch.setattr(wrapper, "_run_xtb_streaming", fake_streaming)

    wrapper.run_xtb_sp(str(xyz), progress_callback=lambda event: None)

    assert len(calls) == 1
    assert calls[0][1]["stage"] == "sp"


def test_xtb_progress_parser_reads_cycle_table_energy():
    payload = wrapper._parse_xtb_progress_line("   12     -1493.782345     0.000123     0.00456")
    assert payload["cycle"] == 12
    assert payload["energy"] == "-1493.782345"


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


def test_xtbx_runtime_is_bundled_with_nciforge_checkout(monkeypatch):
    from nciforge_xtbx import cli as xtbx_cli

    monkeypatch.delenv("NCIFORGE_XTBX_RUNTIME", raising=False)
    monkeypatch.delenv("XTB_GPU_PKG", raising=False)
    monkeypatch.delenv("NCIFORGE_XTBX_CONFIG", raising=False)
    monkeypatch.setattr(xtbx_cli, "resolve_bash", lambda: "bash")

    runtime = xtbx_cli.resolve_runtime([])
    repo_runtime = (
        Path(__file__).resolve().parents[1]
        / "nciforge_xtbx"
        / "runtime"
        / "xtb-win-release"
    )
    assert runtime == repo_runtime.resolve()
    assert (runtime / "bin" / "xtb-cpu.exe").exists()
    assert xtbx_cli.is_available()


def test_xtbx_gpu_requires_full_runtime(monkeypatch):
    from nciforge_xtbx import cli as xtbx_cli

    bundled = (
        Path(__file__).resolve().parents[1]
        / "nciforge_xtbx"
        / "runtime"
        / "xtb-win-release"
    )
    monkeypatch.setattr(xtbx_cli, "_runtime_candidates", lambda args: [bundled])

    with pytest.raises(xtbx_cli.XtbxUnavailable, match="explicit GPU"):
        xtbx_cli.resolve_runtime(["--gpu"])


def test_xtbx_setup_detects_full_runtime_on_path(monkeypatch, tmp_path):
    from nciforge_xtbx import cli as xtbx_cli

    config = tmp_path / "config" / "xtbx_runtime.json"
    runtime = _make_fake_xtbx_runtime(tmp_path / "external" / "xtb-win-release", full_gpu=True)
    monkeypatch.setenv("NCIFORGE_XTBX_CONFIG", str(config))
    monkeypatch.delenv("NCIFORGE_XTBX_RUNTIME", raising=False)
    monkeypatch.delenv("XTB_GPU_PKG", raising=False)
    monkeypatch.setenv("PATH", str(runtime / "bin"))

    assert xtbx_cli.setup_gpu_runtime(interactive=False) == 0
    payload = json.loads(config.read_text(encoding="utf-8"))
    assert Path(payload["gpu_runtime"]) == runtime.resolve()


def test_xtbx_setup_materializes_managed_runtime_from_cuda_dlls(monkeypatch, tmp_path):
    from nciforge_xtbx import cli as xtbx_cli

    fake_package = tmp_path / "pkg" / "nciforge_xtbx"
    _make_fake_xtbx_runtime(fake_package / "runtime" / "xtb-win-release")

    cuda_bin = tmp_path / "cuda" / "bin"
    cuda_bin.mkdir(parents=True)
    for name in xtbx_cli.CUDA_RUNTIME_DLLS:
        (cuda_bin / name).write_text("cuda", encoding="utf-8")

    config = tmp_path / "config" / "xtbx_runtime.json"
    monkeypatch.setenv("NCIFORGE_XTBX_CONFIG", str(config))
    monkeypatch.delenv("NCIFORGE_XTBX_RUNTIME", raising=False)
    monkeypatch.delenv("XTB_GPU_PKG", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(xtbx_cli, "package_dir", lambda: fake_package)
    monkeypatch.setattr(xtbx_cli, "_cuda_dll_source_dirs", lambda: [cuda_bin])
    monkeypatch.setattr(xtbx_cli, "_known_xtb_runtime_roots", lambda: [])

    assert xtbx_cli.setup_gpu_runtime(interactive=False) == 0
    payload = json.loads(config.read_text(encoding="utf-8"))
    runtime = Path(payload["gpu_runtime"])

    assert runtime == (config.parent / "xtbx-runtime" / "xtb-win-release").resolve()
    assert xtbx_cli.resolve_runtime(["--gpu"]) == runtime
    for rel in ("bin/xtb.exe", "bin/xtb-cpu.exe", "params/param_gfn2-xtb.txt"):
        assert (runtime / rel).exists()
    for name in xtbx_cli.CUDA_RUNTIME_DLLS:
        assert (runtime / "lib" / name).exists()


def test_xtbx_setup_downloads_verified_runtime(monkeypatch, tmp_path):
    from nciforge_xtbx import cli as xtbx_cli

    source = _make_fake_xtbx_runtime(tmp_path / "source" / "xtb-win-release", full_gpu=True)
    archive = tmp_path / "nciforge-xtbx-runtime-v1.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for file in source.rglob("*"):
            if file.is_file():
                zf.write(file, Path("xtb-win-release") / file.relative_to(source))

    config = tmp_path / "config" / "xtbx_runtime.json"
    monkeypatch.setenv("NCIFORGE_XTBX_CONFIG", str(config))
    monkeypatch.setenv("NCIFORGE_XTBX_RUNTIME_URL", archive.as_uri())
    monkeypatch.setenv("NCIFORGE_XTBX_RUNTIME_SHA256", xtbx_cli._sha256_file(archive))
    monkeypatch.delenv("NCIFORGE_XTBX_RUNTIME", raising=False)
    monkeypatch.delenv("XTB_GPU_PKG", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(xtbx_cli, "_cuda_dll_source_dirs", lambda: [])
    monkeypatch.setattr(xtbx_cli, "_known_xtb_runtime_roots", lambda: [])

    assert xtbx_cli.setup_gpu_runtime(interactive=False) == 0
    payload = json.loads(config.read_text(encoding="utf-8"))
    runtime = Path(payload["gpu_runtime"])

    assert runtime == (config.parent / "xtbx-runtime" / "xtb-win-release").resolve()
    assert xtbx_cli.resolve_runtime(["--gpu"]) == runtime
    assert (runtime / "bin" / "xtb.exe").exists()
    for name in xtbx_cli.CUDA_RUNTIME_DLLS:
        assert (runtime / "lib" / name).exists()


def test_xtbx_setup_gpu_runtime_persists_full_runtime(monkeypatch, tmp_path):
    from nciforge_xtbx import cli as xtbx_cli

    config = tmp_path / "xtbx_runtime.json"
    runtime = _make_fake_xtbx_runtime(tmp_path / "full-runtime", full_gpu=True)
    monkeypatch.setenv("NCIFORGE_XTBX_CONFIG", str(config))
    monkeypatch.delenv("NCIFORGE_XTBX_RUNTIME", raising=False)
    monkeypatch.delenv("XTB_GPU_PKG", raising=False)

    assert xtbx_cli.setup_gpu_runtime(str(runtime), interactive=False) == 0
    payload = json.loads(config.read_text(encoding="utf-8"))
    assert Path(payload["gpu_runtime"]) == runtime.resolve()
    assert xtbx_cli.resolve_runtime(["--gpu"]) == runtime.resolve()


def test_xtbx_gpu_runtime_override_is_one_shot(monkeypatch, tmp_path):
    from nciforge_xtbx import cli as xtbx_cli

    runtime = _make_fake_xtbx_runtime(tmp_path / "full-runtime", full_gpu=True)
    monkeypatch.delenv("NCIFORGE_XTBX_RUNTIME", raising=False)

    args = xtbx_cli._consume_gpu_runtime_arg(
        ["--gpu-runtime", str(runtime), "--gpu", "--version"]
    )
    assert args == ["--gpu", "--version"]
    assert xtbx_cli.resolve_runtime(args) == runtime.resolve()


def test_xtbx_setup_gpu_does_not_treat_next_flag_as_path():
    from nciforge_xtbx import cli as xtbx_cli

    assert xtbx_cli._setup_path_from_args(["--setup-gpu", "--version"]) == (
        True,
        None,
    )


def test_default_engine_policy_is_geoinit_xtbx(tmp_path):
    xyz = _write_xyz(tmp_path / "m.xyz")
    pipe = KNFPipeline(input_file=xyz, output_root=str(tmp_path / "Results"))
    assert pipe.preopt_engine == "geoinit"
    assert pipe.xtb_engine == "xtbx"


def test_gpu_torch_missing_runs_cuda_setup_before_failing(monkeypatch):
    class Args:
        nci_backend = "torch"
        nci_device = "cuda"
        gpu = True

    state = {"torch_ok": False}
    calls = []

    def fake_torch_available():
        return state["torch_ok"], "No module named 'torch'"

    def fake_cuda_setup(allow_prompt=True):
        calls.append(allow_prompt)
        state["torch_ok"] = True

    monkeypatch.setattr(core_main, "_is_torch_available", fake_torch_available)
    monkeypatch.setattr(core_main, "_ensure_cuda_runtime_for_gpu_mode", fake_cuda_setup)

    core_main._resolve_cpu_backend_when_torch_missing(Args())
    assert calls == [True]


def test_live_repaint_disabled_for_legacy_windows_console(monkeypatch):
    class ConsoleStub:
        is_terminal = True
        is_dumb_terminal = False
        legacy_windows = True

    monkeypatch.setattr(core_main.os, "name", "nt")
    monkeypatch.setattr(core_main.sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("NCIFORGE_FORCE_LIVE", raising=False)
    monkeypatch.delenv("NCIFORGE_NO_LIVE", raising=False)

    assert core_main._live_repaint_supported(ConsoleStub()) is False


def test_live_repaint_env_overrides(monkeypatch):
    class ConsoleStub:
        is_terminal = False
        is_dumb_terminal = True
        legacy_windows = True

    monkeypatch.setattr(core_main.sys.stdout, "isatty", lambda: False)
    monkeypatch.setenv("NCIFORGE_FORCE_LIVE", "1")
    monkeypatch.delenv("NCIFORGE_NO_LIVE", raising=False)
    assert core_main._live_repaint_supported(ConsoleStub()) is True

    monkeypatch.delenv("NCIFORGE_FORCE_LIVE", raising=False)
    monkeypatch.setenv("NCIFORGE_NO_LIVE", "1")
    assert core_main._live_repaint_supported(ConsoleStub()) is False


def test_cli_progress_uses_plain_output_for_unsupported_windows_terminal(monkeypatch):
    class ConsoleStub:
        is_terminal = True
        is_dumb_terminal = False

    monkeypatch.setattr(cli_commands.os, "name", "nt")
    monkeypatch.setattr(cli_commands.sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("NCIFORGE_FORCE_LIVE", raising=False)
    monkeypatch.delenv("NCIFORGE_NO_LIVE", raising=False)
    monkeypatch.delenv("WT_SESSION", raising=False)
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    monkeypatch.delenv("ANSICON", raising=False)
    monkeypatch.delenv("ConEmuANSI", raising=False)

    assert cli_commands._live_progress_supported(ConsoleStub()) is False


def test_cli_progress_force_live_override(monkeypatch):
    class ConsoleStub:
        is_terminal = False
        is_dumb_terminal = True

    monkeypatch.setenv("NCIFORGE_FORCE_LIVE", "1")

    assert cli_commands._live_progress_supported(ConsoleStub()) is True


def test_engine_run_options_match_current_cli_defaults():
    defaults = {field.name: field.default for field in fields(RunOptions)}
    assert defaults == {
        "charge": 0,
            "spin": 1,
            "water": False,
            "hydration_fragment_mode": False,
            "force": False,
        "clean": False,
        "debug": False,
        "processing": "auto",
        "multi": False,
        "single": False,
        "workers": None,
        "output_dir": None,
        "batches": None,
        "compile_existing": False,
        "universal_kuid": False,
        "merge_master_csv": None,
        "merge_new_csv": None,
        "merge_output_dir": None,
        "overwrite_master_csv": False,
        "ram_per_job": 50.0,
        "refresh_autoconfig": False,
        "quiet_config": False,
        "full_files": False,
        "enable_stop_key": False,
        "interactive_quadrant_plot": False,
        "atlas_bundle": False,
        "gpu": False,
        "cpu": False,
        "multiwfn": False,
        "nci_backend": "torch",
        "nci_grid_spacing": 0.2,
        "nci_grid_padding": 3.0,
        "nci_device": "cpu",
        "nci_dtype": "float32",
        "nci_batch_size": 250000,
        "nci_eig_batch_size": 200000,
        "nci_rho_floor": 1e-12,
        "nci_apply_primitive_norm": False,
        "scdi_var_min": None,
        "scdi_var_max": None,
        "wbo_mode": "xtb",
        "preopt": "geoinit",
        "xtb_engine": "xtbx",
        "xtb_gpu_atoms": 350,
        "sp": False,
        "seed_contact": False,
        "refresh_first_run": False,
        "multiwfn_path": None,
        "knf": False,
        "project_root": None,
    }
    assert "input_path" not in defaults


def test_compile_existing_results_from_water_result_folders(tmp_path):
    results_root = tmp_path / "Water-Results"
    for idx in range(1, 3):
        result_dir = results_root / f"mol_{idx}_water_cluster"
        result_dir.mkdir(parents=True)
        (result_dir / "knf_water.json").write_text(
            json.dumps(_fake_knf_payload(idx), indent=2),
            encoding="utf-8",
        )
        (result_dir / "output_water.txt").write_text("ok", encoding="utf-8")

    args = RunOptions(water=True)
    result = engine_jobs.run_compile_existing_results_job(str(results_root), args)

    assert result is not None
    assert result.mode == "compile_existing"
    assert [record.status for record in result.records] == ["success", "success"]
    assert Path(result.aggregate_json_path).name == "batch_knf_water.json"
    assert Path(result.aggregate_csv_path).name == "batch_knf_unified_water.csv"
    assert Path(result.batch_delta_json_path).name == "batch_delta_water.json"

    with open(result.aggregate_csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["File"] for row in rows] == ["mol_1_water_cluster", "mol_2_water_cluster"]
    assert rows[0]["f1"] == "1.1"


def test_engine_batch_job_matches_legacy_batch_outputs(monkeypatch, tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    for idx in range(1, 3):
        _write_xyz(input_dir / f"mol_{idx}.xyz")

    def fake_process_file(file_path, args, output_root=None, batch_size=1):
        file_name = Path(file_path).name
        index = int(Path(file_path).stem.split("_")[-1])
        result_dir = Path(output_root) / Path(file_path).stem
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "knf.json").write_text(
            json.dumps(_fake_knf_payload(index), indent=2),
            encoding="utf-8",
        )
        return True, None, float(index)

    monkeypatch.setattr(core_main, "process_file", fake_process_file)
    monkeypatch.setattr(engine_jobs, "process_file", fake_process_file)
    monkeypatch.setattr(core_main, "_live_repaint_supported", lambda console: False)

    legacy_root = tmp_path / "legacy_results"
    engine_root = tmp_path / "engine_results"
    base_args = dict(
        output_dir=None,
        water=False,
        force=True,
        processing="single",
        workers=1,
        ram_per_job=50.0,
        refresh_autoconfig=False,
        nci_backend="torch",
        nci_device="cpu",
        enable_stop_key=False,
        interactive_quadrant_plot=False,
    )

    legacy_args = SimpleNamespace(**base_args)
    legacy_args.output_dir = str(legacy_root)
    core_main.run_batch_directory(str(input_dir), legacy_args)

    engine_args = RunOptions(**{key: value for key, value in base_args.items() if hasattr(RunOptions, key)})
    engine_args.output_dir = str(engine_root)
    result = engine_jobs.run_batch_directory_job(str(input_dir), engine_args)

    assert result is not None
    assert result.mode == "single"
    assert result.workers == 1
    assert [record.status for record in result.records] == ["success", "success"]

    with open(legacy_root / "batch_knf_unified.csv", newline="", encoding="utf-8") as legacy_file:
        legacy_rows = list(csv.DictReader(legacy_file))
    with open(engine_root / "batch_knf_unified.csv", newline="", encoding="utf-8") as engine_file:
        engine_rows = list(csv.DictReader(engine_file))
    assert engine_rows == legacy_rows

    assert _normalize_batch_json(engine_root / "batch_knf.json") == _normalize_batch_json(
        legacy_root / "batch_knf.json"
    )


def test_single_file_job_emits_stage_progress_event(monkeypatch, tmp_path):
    xyz = _write_xyz(tmp_path / "mol.xyz")
    events = []

    def fake_process_file(file_path, args, output_root=None, batch_size=1, progress_callback=None):
        if progress_callback is not None:
            progress_callback(
                {
                    "message": "xTB OPT [xtbx gpu] | cycle 3",
                    "stage": "opt",
                    "cycle": 3,
                    "elapsed_seconds": 1.25,
                }
            )
        return True, None, 2.0

    monkeypatch.setattr(engine_jobs, "process_file", fake_process_file)

    args = SimpleNamespace(output_dir=str(tmp_path / "Results"), water=False, force=True)
    result = engine_jobs.run_single_file_job(xyz, args, on_event=events.append)

    assert result.success is True
    progress_events = [event for event in events if event.kind == EventKind.FILE_STAGE_PROGRESS]
    assert progress_events
    assert progress_events[0].input_file == xyz
    assert progress_events[0].payload["cycle"] == 3
    assert "cycle 3" in (progress_events[0].message or "")


def test_main_cli_dispatches_single_file_to_cli_commands(monkeypatch, tmp_path):
    xyz = _write_xyz(tmp_path / "mol.xyz")
    called = {}

    monkeypatch.setattr(core_main, "_ensure_utf8_stdout", lambda: None)
    monkeypatch.setattr(core_main, "_clear_terminal", lambda: None)
    monkeypatch.setattr(
        core_main,
        "_show_startup_splash",
        lambda: (_ for _ in ()).throw(AssertionError("CLI mode should not print the plain startup splash")),
    )
    monkeypatch.setattr(cli_app, "resolve_cpu_backend_when_torch_missing", lambda args: None)
    monkeypatch.setattr(cli_app.first_run, "ensure_first_run_setup", lambda **kwargs: True)
    monkeypatch.setattr(cli_app, "probe_missing_dependencies", lambda **kwargs: [])
    monkeypatch.setattr(cli_app, "try_write_atlas_bundle_from_existing_outputs", lambda args: None)
    monkeypatch.setattr(cli_app, "maybe_write_atlas_bundle", lambda args: None)

    def fake_run_single_file(file_path, args):
        called["file_path"] = file_path
        called["args"] = args

    monkeypatch.setattr(core_main.cli_commands, "run_single_file", fake_run_single_file)
    monkeypatch.setattr(core_main.sys, "argv", ["knf", xyz, "--force", "--single"])

    core_main.main()

    assert Path(called["file_path"]).name == "mol.xyz"
    assert called["args"].force is True
    assert called["args"].processing == "single"
    assert Path(called["args"].input_path).name == "mol.xyz"


def test_typer_argv_preprocess_preserves_legacy_forms():
    assert normalize_argv(["full", "input.xyz"]) == ["input.xyz"]
    assert normalize_argv(["input_dir", "--batches"]) == ["input_dir", "--batches", "0"]
    assert normalize_argv(["input_dir", "--batches", "--force"]) == [
        "input_dir",
        "--batches",
        "0",
        "--force",
    ]
    assert normalize_argv(["input_dir", "--batches", "4"]) == ["input_dir", "--batches", "4"]
    assert normalize_argv(["input_dir", "--batches=4"]) == ["input_dir", "--batches=4"]


def test_typer_options_validate_and_apply_shortcuts():
    opts = build_run_options(multi=True, single=True)
    assert validate_flag_combinations(opts) == "Use only one of --multi or --single."

    opts = build_run_options(gpu=True, cpu=True)
    assert validate_flag_combinations(opts) == "Use only one of --gpu or --cpu."

    opts = build_run_options(batches=0, universal_kuid=True)
    assert validate_flag_combinations(opts) == "Use either --batches or --universal-kuid, not both in the same command."

    opts = build_run_options(compile_existing=True, batches=0)
    assert validate_flag_combinations(opts) == "Use either --compile-existing or --batches, not both in the same command."

    opts = build_run_options(single=True, gpu=True)
    assert validate_flag_combinations(opts) is None
    apply_execution_shortcuts(opts)
    assert opts.processing == "single"
    assert opts.nci_backend == "torch"
    assert opts.nci_device == "cuda"


def test_typer_help_shows_nci_backend_section():
    result = CliRunner().invoke(cli_app.app, ["--help"])
    assert result.exit_code == 0
    assert "--charge" in result.output
    assert "--xtb-engine" in result.output
    assert "xTB options" in result.output
    assert "--sp" in result.output
    assert "NCI backend options" in result.output
    assert "--nci-backend" in result.output
    assert "--nci-grid-spacing" in result.output
    assert "--nci-apply-primitive-norm" in result.output
    assert "--knf" not in result.output


def test_main_typer_dispatches_full_bare_batches(monkeypatch, tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    called = {}

    monkeypatch.setattr(core_main, "_ensure_utf8_stdout", lambda: None)
    monkeypatch.setattr(core_main, "_clear_terminal", lambda: None)
    monkeypatch.setattr(core_main, "_show_startup_splash", lambda: None)
    monkeypatch.setattr(cli_app, "resolve_cpu_backend_when_torch_missing", lambda args: None)
    monkeypatch.setattr(cli_app.first_run, "ensure_first_run_setup", lambda **kwargs: True)
    monkeypatch.setattr(cli_app, "probe_missing_dependencies", lambda **kwargs: [])
    monkeypatch.setattr(cli_app, "try_write_atlas_bundle_from_existing_outputs", lambda args: None)
    monkeypatch.setattr(cli_app, "maybe_write_atlas_bundle", lambda args: None)

    def fake_run_batched(directory, args):
        called["directory"] = directory
        called["args"] = args

    monkeypatch.setattr(core_main.cli_commands, "run_batch_directory_batched", fake_run_batched)
    monkeypatch.setattr(core_main.sys, "argv", ["knf", "full", str(input_dir), "--batches"])

    core_main.main()

    assert Path(called["directory"]) == input_dir
    assert called["args"].batches == 0


def test_main_typer_dispatches_compile_existing_without_first_run(monkeypatch, tmp_path):
    results_root = tmp_path / "Results"
    results_root.mkdir()
    called = {}

    monkeypatch.setattr(core_main, "_ensure_utf8_stdout", lambda: None)
    monkeypatch.setattr(core_main, "_clear_terminal", lambda: None)
    monkeypatch.setattr(core_main, "_show_startup_splash", lambda: None)
    monkeypatch.setattr(cli_app, "resolve_cpu_backend_when_torch_missing", lambda args: None)
    monkeypatch.setattr(
        cli_app.first_run,
        "ensure_first_run_setup",
        lambda **kwargs: pytest.fail("compile-only mode should not run first-run setup"),
    )
    monkeypatch.setattr(cli_app, "probe_missing_dependencies", lambda **kwargs: [])
    monkeypatch.setattr(cli_app, "try_write_atlas_bundle_from_existing_outputs", lambda args: None)
    monkeypatch.setattr(cli_app, "maybe_write_atlas_bundle", lambda args: None)

    def fake_compile(directory, args):
        called["directory"] = directory
        called["args"] = args

    monkeypatch.setattr(core_main.cli_commands, "run_compile_existing_results", fake_compile)
    monkeypatch.setattr(core_main.sys, "argv", ["knf", str(results_root), "--compile-existing"])

    core_main.main()

    assert Path(called["directory"]) == results_root
    assert called["args"].compile_existing is True


def test_main_typer_mutual_exclusion_errors(monkeypatch, tmp_path):
    xyz = _write_xyz(tmp_path / "mol.xyz")
    monkeypatch.setattr(core_main, "_ensure_utf8_stdout", lambda: None)
    monkeypatch.setattr(core_main, "_clear_terminal", lambda: None)
    monkeypatch.setattr(core_main, "_show_startup_splash", lambda: None)
    monkeypatch.setattr(core_main.sys, "argv", ["knf", xyz, "--gpu", "--cpu"])

    with pytest.raises(SystemExit) as exc_info:
        core_main.main()

    assert exc_info.value.code == 2


def test_no_args_main_enters_interactive_single_file_mode(monkeypatch, tmp_path):
    xyz = _write_xyz(tmp_path / "mol.xyz")
    prompts = iter([str(xyz), "cpu"])
    called = {}

    monkeypatch.setattr(core_main, "_ensure_utf8_stdout", lambda: None)
    monkeypatch.setattr(core_main, "_clear_terminal", lambda: None)
    monkeypatch.setattr(core_main, "_show_startup_splash", lambda: None)
    monkeypatch.setattr(core_main.sys, "argv", ["knf"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(prompts))
    monkeypatch.setattr(cli_interactive, "brand_panel", lambda: "")
    monkeypatch.setattr(cli_interactive, "resolve_cpu_backend_when_torch_missing", lambda args: None)
    monkeypatch.setattr(cli_interactive.first_run, "ensure_first_run_setup", lambda **kwargs: True)
    monkeypatch.setattr(cli_interactive, "probe_missing_dependencies", lambda **kwargs: [])

    def fake_run_single_file(file_path, args):
        called["file_path"] = file_path
        called["args"] = args

    monkeypatch.setattr(cli_interactive.commands, "run_single_file", fake_run_single_file)

    core_main.main()

    assert Path(called["file_path"]) == Path(xyz)
    assert called["args"].force is True
    assert called["args"].clean is True
    assert called["args"].debug is True
    assert called["args"].enable_stop_key is True
    assert called["args"].cpu is True
    assert called["args"].nci_device == "cpu"


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


def test_pipeline_route_small_single_stays_cpu(tmp_path):
    small = _write_xyz(tmp_path / "small.xyz")  # 6 atoms

    # GPU in play (CUDA NCI), but a single small molecule must stay on CPU.
    pipe = _pipeline_for(
        tmp_path, small, xtb_engine="xtbx", xtb_gpu_available=True, xtb_batch_size=1
    )
    route = pipe._resolve_xtb_route(small)
    assert route.launcher == "xtbx"
    assert route.use_gpu is False

    # Explicit --gpu on a single small molecule is honored.
    forced = _pipeline_for(
        tmp_path,
        small,
        xtb_engine="xtbx",
        xtb_gpu_available=True,
        xtb_explicit_gpu=True,
        xtb_batch_size=1,
    )
    assert forced._resolve_xtb_route(small).use_gpu is True

    # Explicit stock xtb never uses the GPU, even with an explicit preference.
    stock = _pipeline_for(
        tmp_path,
        small,
        xtb_engine="xtb",
        xtb_gpu_available=True,
        xtb_explicit_gpu=True,
    )
    stock_route = stock._resolve_xtb_route(small)
    assert stock_route.launcher == "xtb"
    assert stock_route.use_gpu is False


def test_build_pipeline_decouples_xtb_gpu_from_nci_cuda(tmp_path):
    small = _write_xyz(tmp_path / "small.xyz")  # 6 atoms
    big_header = "400\nbig\n" + "H 0.0 0.0 0.0\n" * 400
    big = _write_xyz(tmp_path / "big.xyz", big_header)

    # NCI on CUDA makes the GPU AVAILABLE to the xtb router, but does not force it.
    cuda_args = RunOptions(nci_backend="torch", nci_device="cuda", xtb_engine="xtbx")
    cuda_pipe = engine_processing._build_pipeline(
        small, cuda_args, output_root=str(tmp_path / "ResultsCuda")
    )
    assert cuda_pipe.xtb_gpu_available is True
    assert cuda_pipe.xtb_explicit_gpu is False
    # Single small molecule on a CUDA run: xtb stays on CPU (no forced --gpu)...
    assert cuda_pipe._resolve_xtb_route(small).use_gpu is False
    # ...but a large molecule on the same run routes to the GPU.
    assert cuda_pipe._resolve_xtb_route(big).use_gpu is True

    # A pure-CPU NCI run never makes the GPU available to the xtb stage.
    cpu_args = RunOptions(nci_backend="torch", nci_device="cpu", xtb_engine="xtbx")
    cpu_pipe = engine_processing._build_pipeline(
        small, cpu_args, output_root=str(tmp_path / "ResultsCpu")
    )
    assert cpu_pipe.xtb_gpu_available is False
    assert cpu_pipe._resolve_xtb_route(big).use_gpu is False

    # The --gpu shortcut sets an explicit GPU preference for the xtb stage too.
    gpu_args = RunOptions(nci_backend="torch", nci_device="cuda", xtb_engine="xtbx", gpu=True)
    gpu_pipe = engine_processing._build_pipeline(
        small, gpu_args, output_root=str(tmp_path / "ResultsGpu")
    )
    assert gpu_pipe.xtb_explicit_gpu is True
    assert gpu_pipe._resolve_xtb_route(small).use_gpu is True  # single small + --gpu -> GPU

    # Many small molecules under --gpu: batch-aware -> stay on CPU per molecule.
    batch_pipe = engine_processing._build_pipeline(
        small, gpu_args, output_root=str(tmp_path / "ResultsBatch"), batch_size=50
    )
    assert batch_pipe._resolve_xtb_route(small).use_gpu is False


def test_sp_only_pipeline_skips_preopt_and_xtb_optimization(monkeypatch, tmp_path):
    xyz = _write_xyz(tmp_path / "dimer.xyz")
    calls = []

    def fail_preopt(*args, **kwargs):
        raise AssertionError("preopt should not run in SP-only mode")

    def fail_opt(*args, **kwargs):
        raise AssertionError("xTB optimization should not run in SP-only mode")

    def fail_contact_seed(*args, **kwargs):
        raise AssertionError("contact seeding should not run in strict SP-only mode")

    def fake_sp(
        filepath,
        charge=0,
        uhf=0,
        use_water=False,
        xtb_cmd="xtb",
        force_gpu=False,
        include_hess=True,
        include_esp=False,
        progress_callback=None,
    ):
        calls.append((Path(filepath).name, charge, uhf, use_water, xtb_cmd, force_gpu, include_hess, include_esp))
        cwd = Path(filepath).parent
        (cwd / "xtb.log").write_text("fake xtb log", encoding="utf-8")
        (cwd / "wbo").write_text("1 2 0.123\n", encoding="utf-8")
        (cwd / "molden.input").write_text("[Molden Format]\n", encoding="utf-8")

    monkeypatch.setattr(wrapper, "run_preopt", fail_preopt)
    monkeypatch.setattr(wrapper, "run_xtb_opt", fail_opt)
    monkeypatch.setattr(wrapper, "run_xtb_sp", fake_sp)
    monkeypatch.setattr("knf_core.pipeline.geometry.promote_hbond_interaction", fail_contact_seed)
    monkeypatch.setattr(KNFPipeline, "_resolve_xtb_cmd", lambda self, path: "xtb")
    monkeypatch.setattr(
        "knf_core.pipeline.xtb.parse_xtb_log",
        lambda path, **kwargs: {"f4": 1.0, "f5": 2.0},
    )
    monkeypatch.setattr(
        "knf_core.pipeline.xtb.compute_wbo_from_molden_details",
        lambda *args, **kwargs: {
            "max_inter_wbo": 0.3,
            "max_wbo_global": 0.4,
            "inter_pair_count": 1,
            "inter_max_pair": {"atom_i_1based": 1, "atom_j_1based": 2, "wbo": 0.3},
            "overlap_model": "test",
            "n_ao": 2,
        },
    )
    monkeypatch.setattr(
        "knf_core.pipeline.xtb.parse_wbo_pair_map",
        lambda *args, **kwargs: {(1, 2): 0.123},
    )

    pipe = KNFPipeline(
        input_file=xyz,
        output_root=str(tmp_path / "Results"),
        force=True,
        xtb_engine="xtb",
        sp_only=True,
        seed_contact=True,
        wbo_mode="native",
    )
    context = pipe.run_pre_nci_stage()

    assert calls == [("input.xyz", 0, 0, False, "xtb", False, False, False)]
    assert context["xtb_sp_only"] is True
    assert context["xtb_sp_include_hess"] is False
    assert context["xtb_sp_include_esp"] is False
    assert Path(context["xtb_geometry_file"]).name == "input.xyz"
    assert context["contact_seed"] == {
        "requested": True,
        "applied": False,
        "reason": "disabled_by_strict_sp",
    }
    assert not (Path(pipe.results_dir) / "xtbopt.xyz").exists()

    def xyz_coordinates(path):
        lines = Path(path).read_text(encoding="utf-8").splitlines()[2:]
        return [[float(value) for value in line.split()[1:4]] for line in lines if line.strip()]

    assert xyz_coordinates(context["xtb_geometry_file"]) == xyz_coordinates(xyz)


def test_xtb_sp_can_skip_esp_and_hessian(monkeypatch, tmp_path):
    xyz = tmp_path / "input.xyz"
    xyz.write_text("1\n\nH 0 0 0\n", encoding="utf-8")
    calls = []

    def fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, errors=None, check=None):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)

    wrapper.run_xtb_sp(str(xyz), include_hess=False, include_esp=False)

    assert calls
    assert "--esp" not in calls[0]
    assert "--hess" not in calls[0]
    assert "--molden" in calls[0]
    assert "--wbo" in calls[0]


def test_xtb_sp_skips_esp_by_default(monkeypatch, tmp_path):
    xyz = tmp_path / "input.xyz"
    xyz.write_text("1\n\nH 0 0 0\n", encoding="utf-8")
    calls = []

    def fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, errors=None, check=None):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)

    wrapper.run_xtb_sp(str(xyz))

    assert calls
    assert "--esp" not in calls[0]
    assert "--hess" in calls[0]
    assert "--molden" in calls[0]
    assert "--wbo" in calls[0]


def test_xtb_log_parser_can_mark_missing_f5_unavailable(tmp_path):
    log_path = tmp_path / "xtb.log"
    log_path.write_text(
        """
molecular dipole:
                 x           y           z       tot (Debye)
 q only:        0.000       0.000       0.000
   full:        0.100       0.200       0.300       1.234

normal termination of xtb
""",
        encoding="utf-8",
    )

    with pytest.raises(xtb.MissingPolarizabilityError):
        xtb.parse_xtb_log(str(log_path))

    parsed = xtb.parse_xtb_log(str(log_path), require_polarizability=False)
    assert parsed["f4"] == pytest.approx(1.234)
    assert parsed["f5"] is None
    assert parsed["f5_available"] is False
    assert "Polarizability" in parsed["f5_unavailable_reason"]


def test_xtb_log_parser_reads_mojibaked_alpha_polarizability(tmp_path):
    log_path = tmp_path / "xtb.log"
    log_path.write_bytes(
        b"""
molecular dipole:
                 x           y           z       tot (Debye)
 q only:        0.000       0.000       0.000
   full:        0.100       0.200       0.300       1.234

 Mol. \xc3\x8e\xc2\xb1(0) /au        :         94.392756

normal termination of xtb
"""
    )

    parsed = xtb.parse_xtb_log(str(log_path))
    assert parsed["f4"] == pytest.approx(1.234)
    assert parsed["f5"] == pytest.approx(94.392756)
    assert parsed["f5_available"] is True


def test_output_txt_writes_na_for_missing_f5(tmp_path):
    result = knf_vector.KNFResult(
        SNCI=0.1,
        SCDI=None,
        SCDI_variance=0.0,
        KNF_vector=[1.0, 2.0, 0.3, 1.234, None, 4.0, 0.01, 0.02, 0.03],
        metadata={"f5_available": False},
    )

    output_path = tmp_path / "output.txt"
    knf_vector.write_output_txt(str(output_path), result)

    assert "f5 (Pol):       n/a au" in output_path.read_text(encoding="utf-8")


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
    assert all(
        isinstance(v, (int, float)) and math.isfinite(v)
        for idx, v in enumerate(vector)
        if idx != 4
    )
    assert vector[4] is None or (
        isinstance(vector[4], (int, float)) and math.isfinite(vector[4])
    )


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


@pytest.mark.skipif(
    not (_RUN_XTB and _XTB_AVAILABLE),
    reason="set KNF_RUN_XTB_TESTS=1 and have xtb on PATH",
)
def test_real_strict_sp_preserves_supplied_coordinates_and_uses_xtb_wbo(tmp_path):
    xyz = _write_xyz(
        tmp_path / "strict_sp_water_dimer.xyz",
        WATER_DIMER_XYZ.replace("-1.551007", "-1.551007123456"),
    )
    pipeline = KNFPipeline(
        input_file=xyz,
        output_root=str(tmp_path / "ResultsStrictSP"),
        force=True,
        xtb_engine="xtb",
        sp_only=True,
    )
    context = pipeline.run_pre_nci_stage()

    def coordinates(path):
        rows = Path(path).read_text(encoding="utf-8").splitlines()[2:]
        return [[float(value) for value in row.split()[1:4]] for row in rows if row.strip()]

    assert Path(context["xtb_geometry_file"]).read_bytes() == Path(xyz).read_bytes()
    assert coordinates(context["xtb_geometry_file"]) == coordinates(xyz)
    assert context["xtb_sp_only"] is True
    assert context["contact_seed"]["applied"] is False
    assert context["wbo_mode"] == "xtb"
    assert context["f3_definition"] == "parsed_xtb_interfragment_wiberg_bond_order"
    assert context["f3_status"] == "production"
