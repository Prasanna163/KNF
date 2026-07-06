from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler

from ..engine import jobs
from ..engine.discovery import _BATCH_LEGACY_CSV_NAMES, _BATCH_PRIMARY_CSV_NAME
from ..engine.events import EventKind, JobEvent
from ..engine.naming import _final_output_name
from .presentation import sysinfo
from .presentation.formatting import (
    ACCENT,
    FAIL_STYLE,
    MUTED,
    OK_STYLE,
    build_console,
    display_name,
    display_path,
    fmt_elapsed,
)
from .presentation.panels import (
    brand_banner,
    failed_files_table,
    progress_panel,
    recent_jobs_panel,
    specs_columns,
    stat_tiles,
    status_label,
    summary_panel,
)


def _make_console() -> Console:
    """Build a Console that renders clean, continuous Unicode boxes and colors.

    Delegates terminal setup to :func:`build_console`, which forces Rich's modern
    VT renderer so the dashboard shows rounded frames and repaints its live region
    in place — even when this checkout's ``main()`` was bypassed (e.g. a shadowing
    ``knf_core``).
    """
    return build_console()


def _live_progress_supported(console: Console) -> bool:
    if os.environ.get("NCIFORGE_FORCE_LIVE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if os.environ.get("NCIFORGE_NO_LIVE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if not bool(getattr(sys.stdout, "isatty", lambda: False)()):
        return False
    if not bool(getattr(console, "is_terminal", False)):
        return False
    if bool(getattr(console, "is_dumb_terminal", False)):
        return False
    if os.name == "nt":
        ansi_env = (
            os.environ.get("WT_SESSION")
            or os.environ.get("TERM_PROGRAM")
            or os.environ.get("ANSICON")
            or os.environ.get("ConEmuANSI", "").strip().lower() == "on"
            # build_console forces the modern renderer; trust it when it wins.
            # Default to legacy when the attribute is absent (conservative).
            or not bool(getattr(console, "legacy_windows", True))
        )
        if not ansi_env:
            return False
    return True


class _live_logging:
    """Route logging through the live console so records render *above* the
    dashboard instead of tearing through its repainting region.

    A ``StreamHandler`` created by ``logging.basicConfig`` holds the pre-Live
    ``sys.stderr`` and writes straight past Rich's cursor control, which is a
    classic source of duplicated/garbled live output. For the duration of the
    ``with`` block we swap the root handlers for a :class:`RichHandler` bound to
    the live console, then restore the originals.
    """

    def __init__(self, console: Console):
        self._console = console
        self._saved_handlers = None
        self._saved_level = None

    def __enter__(self) -> "_live_logging":
        root = logging.getLogger()
        self._saved_handlers = root.handlers[:]
        self._saved_level = root.level
        handler = RichHandler(
            console=self._console,
            show_path=False,
            markup=False,
            rich_tracebacks=True,
            log_time_format="[%H:%M:%S]",
        )
        root.handlers = [handler]
        return self

    def __exit__(self, *exc) -> None:
        root = logging.getLogger()
        if self._saved_handlers is not None:
            root.handlers = self._saved_handlers
        if self._saved_level is not None:
            root.setLevel(self._saved_level)
        return None


# ---------------------------------------------------------------------------
# Live batch dashboard
# ---------------------------------------------------------------------------
def _derive(state: dict) -> dict:
    elapsed = time.perf_counter() - float(state.get("started_at") or time.perf_counter())
    completed = int(state.get("completed") or 0)
    total = int(state.get("total") or 0)
    done = int(state.get("succeeded") or 0)
    failed = int(state.get("failed") or 0)
    running = max(0, int(state.get("active_workers") or 0))
    jobs_per_min = (completed / elapsed) * 60 if completed and elapsed > 0 else 0.0
    avg = (elapsed / completed) if completed else None
    remaining = max(0, total - completed)
    eta = (remaining * avg) if avg else None
    return {
        "elapsed": elapsed,
        "completed": completed,
        "total": total,
        "done": done,
        "failed": failed,
        "running": running,
        "jobs_per_min": jobs_per_min,
        "eta": eta,
    }


def _stat_metrics(state: dict) -> list[tuple[str, str, str]]:
    d = _derive(state)
    return [
        ("Done", str(d["done"]), OK_STYLE),
        ("Failed", str(d["failed"]), FAIL_STYLE if d["failed"] else MUTED),
        ("Running", str(d["running"]), ACCENT),
        ("Elapsed", fmt_elapsed(d["elapsed"]), "bold"),
        ("ETA", fmt_elapsed(d["eta"]) if d["eta"] is not None else "--:--", f"bold {ACCENT}"),
        ("Throughput", f"{d['jobs_per_min']:.1f}/min", "bold"),
    ]


def _batch_specs(state, system_rows):
    config_rows = sysinfo.run_config_rows(
        state.get("args"),
        mode=state.get("mode"),
        workers=state.get("workers"),
        results_root=state.get("results_root"),
    )
    return specs_columns(config_rows, system_rows)


def _batch_dashboard(state, completed_rows) -> Group:
    """Compact, bounded live region (progress + stats + recent jobs).

    Static context (brand + specs) is printed once *above* the live region so
    this stays comfortably shorter than the terminal, which is what keeps Rich's
    Live repainting in place instead of stacking frames.
    """
    d = _derive(state)
    detail = f"ETA {fmt_elapsed(d['eta'])}" if d["eta"] is not None else ""
    return Group(
        progress_panel(d["completed"], d["total"], detail=detail),
        stat_tiles(_stat_metrics(state)),
        recent_jobs_panel(completed_rows, limit=8),
    )


# ---------------------------------------------------------------------------
# Single file
# ---------------------------------------------------------------------------
def run_single_file(file_path: str, args):
    console = _make_console()
    t0 = time.perf_counter()
    use_live_progress = _live_progress_supported(console)

    results_root = jobs.resolve_results_root(file_path, getattr(args, "output_dir", None))
    console.print(brand_banner())
    config_rows = sysinfo.run_config_rows(args, mode="single", results_root=results_root)
    config_rows.insert(0, ("File", display_name(file_path)))
    console.print(specs_columns(config_rows, sysinfo.system_rows(args)))

    if use_live_progress:
        with _live_logging(console), Live(
            progress_panel(0, 1, detail="processing…", pulse=True),
            console=console,
            refresh_per_second=8,
            transient=False,
            vertical_overflow="crop",
        ) as live:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(jobs.run_single_file_job, file_path, args)
                while not future.done():
                    time.sleep(0.1)
                    live.update(progress_panel(0, 1, detail="processing…", pulse=True))
                result = future.result()
            live.update(progress_panel(1, 1, detail="done"))
    else:
        console.print(f"Running single job: {display_name(file_path)}")
        result = jobs.run_single_file_job(file_path, args)

    total_time = result.elapsed_seconds if result.elapsed_seconds > 0 else (time.perf_counter() - t0)
    throughput = ((1 / total_time) * 3600) if (result.success and total_time > 0) else 0.0
    summary_rows = [
        ("Total files", "1"),
        ("Success", "1" if result.success else "0"),
        ("Failed", "0" if result.success else "1"),
        ("Total time", fmt_elapsed(total_time)),
        ("Molecule time", f"{result.elapsed_seconds:.1f}s" if result.elapsed_seconds > 0 else "n/a"),
        ("Throughput", f"{throughput:.1f} jobs/hour" if result.success else "n/a"),
    ]
    if result.output_root:
        summary_rows.append(("Output", display_path(result.output_root)))
    if result.success and isinstance(result.kuid_summary, dict):
        if result.kuid_summary.get("updated"):
            summary_rows.append(("KUID", str(result.kuid_summary.get("kuid", ""))))
            calibration_file = str(result.kuid_summary.get("calibration_file", ""))
            calibration_source = str(result.kuid_summary.get("calibration_source", "")).strip()
            summary_rows.append(
                (
                    "KUID Calibration",
                    f"{calibration_file} ({calibration_source})" if calibration_source else calibration_file,
                )
            )
        else:
            issue = result.kuid_summary.get("error") or result.kuid_summary.get("reason")
            if issue:
                summary_rows.append(("KUID", f"not updated ({issue})"))

    console.print(
        summary_panel(
            summary_rows,
            title="Process Complete" if result.success else "Process Failed",
            tone="ok" if result.success else "fail",
        )
    )
    if not result.success:
        console.print(failed_files_table([(os.path.basename(file_path), str(result.error))]))
    return result


# ---------------------------------------------------------------------------
# Batch directory
# ---------------------------------------------------------------------------
def run_batch_directory(
    directory: str,
    args,
    file_paths: list[str] | None = None,
    results_root_override: str | None = None,
):
    console = _make_console()
    completed_rows: list[tuple[str, str, str]] = []
    failures: list[tuple[str, str]] = []
    use_live_progress = _live_progress_supported(console)
    system_rows = sysinfo.system_rows(args)
    state = {
        "args": args,
        "started_at": time.perf_counter(),
        "total": 0,
        "completed": 0,
        "succeeded": 0,
        "failed": 0,
        "stopped": 0,
        "active_workers": 0,
        "mode": getattr(args, "processing", "auto"),
        "workers": getattr(args, "workers", None),
        "results_root": "",
    }
    live: Live | None = None
    specs_shown = False

    console.print(brand_banner())

    def refresh_dashboard() -> None:
        if live is not None:
            live.update(_batch_dashboard(state, completed_rows))

    def on_event(event: JobEvent) -> None:
        nonlocal specs_shown
        if event.kind == EventKind.BATCH_STARTED:
            payload = event.payload or {}
            state["started_at"] = time.perf_counter()
            state["total"] = int(event.total or 0)
            state["mode"] = payload.get("mode", state["mode"])
            state["workers"] = payload.get("workers", state["workers"])
            state["results_root"] = payload.get("results_root", "")
            state["active_workers"] = 0
            if not specs_shown:
                # Printed once; Rich Live renders this above the live region.
                console.print(_batch_specs(state, system_rows))
                specs_shown = True
            if live is None:
                console.print(
                    f"Batch started: {event.total or 0} file(s), "
                    f"mode={state['mode']}, workers={state['workers']}"
                )
            refresh_dashboard()
        elif event.kind == EventKind.FILE_STARTED:
            state["active_workers"] = int(state.get("active_workers") or 0) + 1
            if live is None and event.message:
                console.print(event.message)
            refresh_dashboard()
        elif event.kind in {EventKind.FILE_SUCCEEDED, EventKind.FILE_FAILED, EventKind.FILE_STOPPED}:
            state["completed"] = int(event.completed or (int(state.get("completed") or 0) + 1))
            state["active_workers"] = max(0, int(state.get("active_workers") or 0) - 1)
            elapsed = f"{(event.elapsed_seconds or 0.0):.1f}s" if event.elapsed_seconds else "-"
            if event.kind == EventKind.FILE_SUCCEEDED:
                state["succeeded"] = int(state.get("succeeded") or 0) + 1
                completed_rows.append((display_name(event.input_file or ""), elapsed, status_label("success")))
                if live is None:
                    console.print(
                        f"[{event.completed or len(completed_rows)}/{event.total or '?'}] "
                        f"OK {display_name(event.input_file or '')} ({elapsed})"
                    )
            elif event.kind == EventKind.FILE_FAILED:
                state["failed"] = int(state.get("failed") or 0) + 1
                completed_rows.append((display_name(event.input_file or ""), elapsed, status_label("failed")))
                failures.append((os.path.basename(event.input_file or ""), event.message or "failed"))
                if live is None:
                    console.print(
                        f"[{event.completed or len(completed_rows)}/{event.total or '?'}] "
                        f"FAIL {display_name(event.input_file or '')} ({elapsed})"
                    )
            else:
                state["stopped"] = int(state.get("stopped") or 0) + 1
                completed_rows.append((display_name(event.input_file or ""), elapsed, status_label("stopped")))
                if live is None:
                    console.print(event.message or "Stop requested.")
            refresh_dashboard()
        elif live is None and event.kind == EventKind.FILE_SKIPPED and event.message:
            console.print(event.message)

    if use_live_progress:
        with _live_logging(console), Live(
            _batch_dashboard(state, completed_rows),
            console=console,
            refresh_per_second=8,
            transient=False,
            vertical_overflow="crop",
        ) as live_ctx:
            live = live_ctx
            result = jobs.run_batch_directory_job(
                directory=directory,
                options=args,
                file_paths=file_paths,
                results_root_override=results_root_override,
                on_event=on_event,
            )
            refresh_dashboard()
        live = None
    else:
        result = jobs.run_batch_directory_job(
            directory=directory,
            options=args,
            file_paths=file_paths,
            results_root_override=results_root_override,
            on_event=on_event,
        )

    if result is None:
        console.print(f"No molecular files found in {directory}.")
        return None

    success_count = sum(1 for record in result.records if record.status == "success")
    failed_count = sum(1 for record in result.records if record.status == "failed")
    completed_non_stopped = max(0, len(result.records) - result.stopped_count)
    throughput = (completed_non_stopped / result.total_time_seconds) * 3600 if result.total_time_seconds > 0 else 0.0
    avg_per_molecule = result.total_time_seconds / completed_non_stopped if completed_non_stopped else 0.0

    if not use_live_progress:
        console.print(recent_jobs_panel(completed_rows, limit=15))
    summary_rows = [
        ("Total files", str(len(result.records))),
        ("Success", str(success_count)),
        ("Failed", str(failed_count)),
        ("Stopped", str(result.stopped_count)),
        ("Total time", fmt_elapsed(result.total_time_seconds)),
        ("Avg per molecule", f"{avg_per_molecule:.1f}s" if completed_non_stopped else "n/a"),
        ("Throughput", f"{throughput:.1f} jobs/hour" if completed_non_stopped else "n/a"),
        ("Batch JSON", display_path(result.aggregate_json_path or "n/a")),
        ("Batch CSV", display_path(result.aggregate_csv_path or "n/a")),
    ]
    if result.skipped_existing:
        summary_rows.insert(4, ("Skipped existing", str(result.skipped_existing)))
    if result.batch_delta_json_path:
        summary_rows.append(("Batch Delta JSON", display_path(result.batch_delta_json_path)))
    if result.batch_delta_txt_path:
        summary_rows.append(("Batch Delta TXT", display_path(result.batch_delta_txt_path)))
    quadrant_payload = result.quadrant_payload or {}
    if quadrant_payload.get("quadrant_plot_png"):
        summary_rows.append(("Quadrant Plot", display_path(quadrant_payload["quadrant_plot_png"])))
    elif quadrant_payload.get("plot_error"):
        summary_rows.append(("Quadrant Plot", f"not generated ({quadrant_payload['plot_error']})"))
    if quadrant_payload.get("quadrant_json"):
        summary_rows.append(("Quadrant JSON", display_path(quadrant_payload["quadrant_json"])))

    tone = "ok" if (failed_count == 0 and result.stopped_count == 0) else ("warn" if failed_count == 0 else "fail")
    title = "Batch Completed" if failed_count == 0 else "Batch Completed With Failures"
    console.print(summary_panel(summary_rows, title=title, tone=tone))
    if failures:
        console.print(failed_files_table(failures))
    return result


def run_universal_kuid(directory: str, args):
    result = jobs.run_universal_kuid_job(directory, args)
    if result is None:
        water_mode = bool(getattr(args, "water", False))
        print(
            f"No {_final_output_name('batch_knf.json', water_mode)} or "
            f"{_final_output_name(_BATCH_PRIMARY_CSV_NAME, water_mode)} "
            f"(legacy {_final_output_name(_BATCH_LEGACY_CSV_NAMES[0], water_mode)} also supported) "
            f"files found under {directory}."
        )
        return None

    print(f"Combined results root: {result['output_root']}")
    print(f"Combined Batch JSON:  {result['batch_json']}")
    print(f"Combined Batch CSV:   {result['batch_csv']}")
    return result


def run_batch_directory_batched(directory: str, args):
    result = jobs.run_batch_directory_batched_job(directory, args)
    if result is None:
        print(f"No molecular files found in {directory}.")
        return None

    combined = result["combined"]
    print(f"\nCombined results root: {combined['output_root']}")
    print(f"Combined Batch JSON:  {combined['batch_json']}")
    print(f"Combined Batch CSV:   {combined['batch_csv']}")
    return result
