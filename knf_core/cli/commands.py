from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from ..engine import jobs
from ..engine.discovery import _BATCH_LEGACY_CSV_NAMES, _BATCH_PRIMARY_CSV_NAME
from ..engine.events import EventKind, JobEvent
from ..engine.naming import _final_output_name
from .presentation.formatting import display_name, display_path, fmt_elapsed, supports_unicode_terminal
from .presentation.panels import brand_panel, calculation_results_panel, failed_files_table, key_value_panel


def _print_process_complete(console: Console, rows: list[tuple[str, str]], *, success: bool = True) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    for key, value in rows:
        table.add_row(str(key), str(value))
    console.print(
        Panel(
            table,
            title="PROCESS COMPLETE",
            title_align="left",
            border_style=("green" if success else "red"),
            padding=(0, 1),
            box=(box.ROUNDED if supports_unicode_terminal() else box.ASCII),
        )
    )


def run_single_file(file_path: str, args):
    console = Console()
    t0 = time.perf_counter()
    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
    task_id = progress.add_task("Single Job", total=1)

    console.print(brand_panel())
    console.print(
        key_value_panel(
            [
                ("Mode", "single"),
                ("File", display_name(file_path)),
                ("Output", display_path(jobs.resolve_results_root(file_path, getattr(args, "output_dir", None)))),
            ],
            title="SYSTEM INITIALIZATION",
        )
    )

    with progress:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(jobs.run_single_file_job, file_path, args)
            while not future.done():
                time.sleep(0.2)
            result = future.result()
            progress.advance(task_id, 1)

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
        summary_rows.append(("Output", result.output_root))
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

    _print_process_complete(console, summary_rows, success=result.success)
    if not result.success:
        console.print(failed_files_table([(os.path.basename(file_path), str(result.error))]))
    return result


def run_batch_directory(
    directory: str,
    args,
    file_paths: list[str] | None = None,
    results_root_override: str | None = None,
):
    console = Console()
    completed_rows: list[tuple[str, str, str]] = []
    failures: list[tuple[str, str]] = []
    task_id = None

    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )

    def on_event(event: JobEvent) -> None:
        nonlocal task_id
        if event.kind == EventKind.BATCH_STARTED:
            task_id = progress.add_task("Overall", total=event.total or 0)
        elif event.kind in {EventKind.FILE_SUCCEEDED, EventKind.FILE_FAILED, EventKind.FILE_STOPPED}:
            if task_id is not None:
                progress.advance(task_id, 1)
            elapsed = f"{(event.elapsed_seconds or 0.0):.1f}s" if event.elapsed_seconds else "-"
            if event.kind == EventKind.FILE_SUCCEEDED:
                completed_rows.append((display_name(event.input_file or ""), elapsed, "[green]OK[/green]"))
            elif event.kind == EventKind.FILE_FAILED:
                completed_rows.append((display_name(event.input_file or ""), elapsed, "[red]FAIL[/red]"))
                failures.append((os.path.basename(event.input_file or ""), event.message or "failed"))
            else:
                completed_rows.append((display_name(event.input_file or ""), elapsed, "[yellow]STOP[/yellow]"))

    console.print(brand_panel())
    with progress:
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

    console.print(calculation_results_panel(completed_rows))
    summary_rows = [
        ("Total files", str(len(result.records))),
        ("Success", str(success_count)),
        ("Failed", str(failed_count)),
        ("Stopped", str(result.stopped_count)),
        ("Total time", fmt_elapsed(result.total_time_seconds)),
        ("Avg per molecule", f"{avg_per_molecule:.1f}s" if completed_non_stopped else "n/a"),
        ("Throughput", f"{throughput:.1f} jobs/hour" if completed_non_stopped else "n/a"),
        ("Batch JSON", result.aggregate_json_path or "n/a"),
        ("Batch CSV", result.aggregate_csv_path or "n/a"),
    ]
    if result.skipped_existing:
        summary_rows.insert(4, ("Skipped existing", str(result.skipped_existing)))
    if result.batch_delta_json_path:
        summary_rows.append(("Batch Delta JSON", result.batch_delta_json_path))
    if result.batch_delta_txt_path:
        summary_rows.append(("Batch Delta TXT", result.batch_delta_txt_path))
    quadrant_payload = result.quadrant_payload or {}
    if quadrant_payload.get("quadrant_plot_png"):
        summary_rows.append(("Quadrant Plot", quadrant_payload["quadrant_plot_png"]))
    elif quadrant_payload.get("plot_error"):
        summary_rows.append(("Quadrant Plot", f"not generated ({quadrant_payload['plot_error']})"))
    if quadrant_payload.get("quadrant_json"):
        summary_rows.append(("Quadrant JSON", quadrant_payload["quadrant_json"]))

    _print_process_complete(console, summary_rows, success=(failed_count == 0))
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
