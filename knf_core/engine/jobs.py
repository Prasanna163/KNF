from __future__ import annotations

import csv
import hashlib
import logging
import os
import shutil
import sys
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, TimeoutError, as_completed
from queue import Queue

from .. import autoconfig, utils
from .aggregate import write_batch_aggregate_json
from .batch_sources import _combine_batch_sources, _discover_universal_batch_sources
from .constants import STOP_KEY, VALID_INPUT_EXTS
from .discovery import (
    _BATCH_LEGACY_CSV_NAMES,
    _BATCH_PRIMARY_CSV_NAME,
    _batch_primary_csv_path,
    _cleanup_compound_knf_json_outputs,
    _dedupe_batch_records,
    _discover_input_files,
    _existing_batch_csv_path,
    _has_reusable_compound_outputs,
    _load_existing_batch_records,
    _normalize_batch_file_name,
    _resolve_requested_batch_count,
    _split_evenly,
    _sum_elapsed_seconds,
    resolve_results_root,
)
from .events import EventKind, JobEvent, OnEvent
from .kuid_ops import _run_kuid_for_single_result
from .naming import _final_output_name
from .processing import process_file, process_file_post_nci, process_file_pre_nci
from .types import BatchFileRecord, BatchResult, SingleFileResult


def _emit(on_event: OnEvent | None, event: JobEvent) -> None:
    if on_event is not None:
        on_event(event)


def _poll_stop_key(enable_stop_key: bool) -> bool:
    if not enable_stop_key:
        return False
    try:
        if os.name == "nt":
            import msvcrt

            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch and ch.lower() == STOP_KEY:
                    return True
            return False
        import select

        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if ready:
            ch = sys.stdin.read(1)
            return bool(ch and ch.lower() == STOP_KEY)
        return False
    except Exception:
        return False


def run_single_file_job(file_path: str, options) -> SingleFileResult:
    results_root = resolve_results_root(file_path, getattr(options, "output_dir", None))
    success, error, elapsed = process_file(file_path, options, output_root=results_root)

    kuid_summary = None
    if success:
        try:
            kuid_summary = _run_kuid_for_single_result(
                file_path=file_path,
                results_root=results_root,
                water=bool(getattr(options, "water", False)),
            )
        except Exception as exc:
            kuid_summary = {"ran": True, "updated": False, "error": str(exc)}
        _cleanup_compound_knf_json_outputs(
            results_root,
            water=bool(getattr(options, "water", False)),
        )

    return SingleFileResult(
        input_file=os.path.abspath(file_path),
        success=bool(success),
        error=None if success else str(error),
        elapsed_seconds=float(elapsed or 0.0),
        output_root=results_root,
        kuid_summary=kuid_summary,
    )


def _normalize_job_files(directory: str, file_paths: list[str] | None) -> list[str]:
    if file_paths is None:
        return _discover_input_files(directory)

    files = []
    for file_path in file_paths:
        full_path = os.path.abspath(file_path)
        if not os.path.isfile(full_path):
            continue
        ext = utils.normalized_extension(os.path.basename(full_path))
        if ext in VALID_INPUT_EXTS:
            files.append(full_path)
    files.sort()
    return files


def _copy_failed_input(
    *,
    file_path: str,
    error: str,
    failed_root: str,
    failed_manifest: str,
    failed_manifest_written: bool,
) -> bool:
    try:
        os.makedirs(failed_root, exist_ok=True)
        src = os.path.abspath(file_path)
        base = os.path.basename(src)
        stem, ext = os.path.splitext(base)
        dst = os.path.join(failed_root, base)
        if os.path.exists(dst):
            short_hash = hashlib.sha1(src.encode("utf-8")).hexdigest()[:8]
            dst = os.path.join(failed_root, f"{stem}__{short_hash}{ext}")
        shutil.copy2(src, dst)
        with open(failed_manifest, "a", newline="", encoding="utf-8") as mf:
            writer = csv.writer(mf)
            if not failed_manifest_written and os.path.getsize(failed_manifest) == 0:
                writer.writerow(["source_file", "failed_copy", "error"])
                failed_manifest_written = True
            writer.writerow([src, dst, str(error)])
    except Exception as copy_exc:
        logging.warning("Failed to copy failed input to %s for %s: %s", failed_root, file_path, copy_exc)
    return failed_manifest_written


def _batch_file_record_from_dict(record: dict) -> BatchFileRecord:
    return BatchFileRecord(
        input_file=str(record.get("input_file", "")),
        status=record.get("status") or "failed",
        elapsed_seconds=float(record.get("elapsed_seconds") or 0.0),
        error=record.get("error"),
        knf=record.get("knf") if isinstance(record.get("knf"), dict) else None,
    )


def run_batch_directory_job(
    directory: str,
    options,
    file_paths: list[str] | None = None,
    results_root_override: str | None = None,
    on_event: OnEvent | None = None,
) -> BatchResult | None:
    """Run all valid molecular files in a directory without CLI rendering."""
    results_root = results_root_override or resolve_results_root(
        directory,
        getattr(options, "output_dir", None),
    )
    water_mode = bool(getattr(options, "water", False))
    aggregate_csv_path = _batch_primary_csv_path(results_root, water=water_mode)
    aggregate_json_path = os.path.join(results_root, _final_output_name("batch_knf.json", water_mode))
    existing_resume_csv_path = _existing_batch_csv_path(results_root, water=water_mode)
    files = _normalize_job_files(directory, file_paths)

    if not files:
        _emit(
            on_event,
            JobEvent(
                EventKind.BATCH_FINISHED,
                message=f"No molecular files found in {directory}.",
                total=0,
            ),
        )
        return None

    batch_input_root = os.path.abspath(directory)
    failed_root = os.path.join(batch_input_root, "failed")
    failed_manifest = os.path.join(failed_root, "failed_manifest.csv")
    failed_manifest_written = False

    existing_batch_records = []
    skipped_existing = 0
    has_resume_outputs = os.path.exists(aggregate_json_path) or os.path.exists(existing_resume_csv_path)
    if has_resume_outputs and not bool(getattr(options, "force", False)):
        resume_state = _load_existing_batch_records(
            directory=directory,
            results_root=results_root,
            water=water_mode,
        )
        existing_batch_records = resume_state.get("records") or []
        processed_names = resume_state.get("processed_names") or set()
        for warning in (resume_state.get("warnings") or []):
            logging.warning(warning)

        if processed_names:
            pending_files = []
            recovered_missing_outputs = 0
            for file_path in files:
                key = _normalize_batch_file_name(os.path.basename(file_path))
                if key in processed_names:
                    if _has_reusable_compound_outputs(file_path, results_root, water=water_mode):
                        continue
                    recovered_missing_outputs += 1
                pending_files.append(file_path)

            skipped_existing = len(files) - len(pending_files)
            files = pending_files
            if skipped_existing:
                _emit(
                    on_event,
                    JobEvent(
                        EventKind.FILE_SKIPPED,
                        message=f"Resume mode skipped {skipped_existing} existing file(s).",
                        completed=skipped_existing,
                    ),
                )
            if recovered_missing_outputs:
                _emit(
                    on_event,
                    JobEvent(
                        EventKind.FILE_STARTED,
                        message=(
                            f"Resume re-queued {recovered_missing_outputs} file(s) "
                            "because compound outputs are missing."
                        ),
                    ),
                )

        if not files:
            if existing_batch_records:
                refresh_mode = str(getattr(options, "processing", "auto")).lower()
                if refresh_mode == "auto":
                    refresh_mode = "multi" if len(existing_batch_records) > 1 else "single"
                refresh_workers = max(1, int(getattr(options, "workers", 1) or 1))
                aggregate_total_time = _sum_elapsed_seconds(existing_batch_records)
                (
                    refreshed_json,
                    refreshed_csv,
                    quadrant_payload,
                    refreshed_delta_json,
                    refreshed_delta_txt,
                ) = write_batch_aggregate_json(
                    directory=directory,
                    results_root=results_root,
                    records=existing_batch_records,
                    mode=refresh_mode,
                    workers=refresh_workers,
                    total_time=aggregate_total_time,
                    water=water_mode,
                    interactive_quadrant_plot=False,
                )
                result = BatchResult(
                    input_directory=os.path.abspath(directory),
                    results_root=results_root,
                    records=[_batch_file_record_from_dict(record) for record in existing_batch_records],
                    mode=refresh_mode,
                    workers=refresh_workers,
                    total_time_seconds=aggregate_total_time,
                    aggregate_json_path=refreshed_json,
                    aggregate_csv_path=refreshed_csv,
                    batch_delta_json_path=refreshed_delta_json,
                    batch_delta_txt_path=refreshed_delta_txt,
                    quadrant_payload=quadrant_payload,
                    skipped_existing=skipped_existing,
                )
                _emit(
                    on_event,
                    JobEvent(EventKind.BATCH_FINISHED, completed=0, total=0, payload={"result": result}),
                )
                return result
            _emit(
                on_event,
                JobEvent(
                    EventKind.BATCH_FINISHED,
                    message=(
                        f"No new molecular files found in {directory}; all detected files "
                        f"are already listed in {aggregate_csv_path}."
                    ),
                    completed=0,
                    total=0,
                ),
            )
            return None

    mode = str(getattr(options, "processing", "auto")).lower()
    if mode == "auto":
        mode = "multi" if len(files) > 1 else "single"
    use_gpu_overlap = (
        mode == "multi"
        and len(files) > 1
        and str(getattr(options, "nci_backend", "") or "").strip().lower() == "torch"
        and str(getattr(options, "nci_device", "") or "").strip().lower() == "cuda"
    )

    if mode == "multi" and len(files) > 1:
        if getattr(options, "workers", None) is None:
            cfg = autoconfig.resolve_multi_config(
                n_jobs=len(files),
                ram_per_job_mb=getattr(options, "ram_per_job", 50.0),
                project_root=(getattr(options, "project_root", None) or os.getcwd()),
                force_refresh=bool(getattr(options, "refresh_autoconfig", False)),
            )
            workers = cfg.workers
            autoconfig.apply_env_inplace(cfg)
        else:
            workers = max(1, int(getattr(options, "workers")))
            logical_threads = os.cpu_count() or 1
            omp = max(1, logical_threads // workers)
            os.environ.update(
                {
                    "OMP_NUM_THREADS": str(omp),
                    "MKL_NUM_THREADS": str(omp),
                    "OPENBLAS_NUM_THREADS": str(omp),
                    "VECLIB_MAXIMUM_THREADS": str(omp),
                    "NUMEXPR_NUM_THREADS": str(omp),
                }
            )
    else:
        workers = 1

    enable_stop_key = bool(getattr(options, "enable_stop_key", False))
    interactive_quadrant_plot = bool(getattr(options, "interactive_quadrant_plot", False))
    stop_requested = False
    stop_notice_emitted = False
    total = len(files)
    completed = 0
    succeeded = 0
    stopped_count = 0
    failures: list[tuple[str, str]] = []
    batch_records: list[dict] = []
    t0 = time.perf_counter()

    _emit(
        on_event,
        JobEvent(
            EventKind.BATCH_STARTED,
            completed=0,
            total=total,
            payload={"mode": mode, "workers": workers, "results_root": results_root},
        ),
    )

    def maybe_request_stop() -> bool:
        nonlocal stop_requested, stop_notice_emitted
        if not stop_requested and _poll_stop_key(enable_stop_key):
            stop_requested = True
        if stop_requested and not stop_notice_emitted:
            _emit(
                on_event,
                JobEvent(
                    EventKind.FILE_STOPPED,
                    message="Stop requested. Completing running tasks and finalizing partial outputs.",
                    completed=completed,
                    total=total,
                ),
            )
            stop_notice_emitted = True
        return stop_requested

    def add_success(file_path: str, elapsed: float):
        nonlocal completed, succeeded
        completed += 1
        succeeded += 1
        batch_records.append(
            {
                "input_file": file_path,
                "status": "success",
                "elapsed_seconds": elapsed,
                "error": None,
            }
        )
        _emit(
            on_event,
            JobEvent(
                EventKind.FILE_SUCCEEDED,
                input_file=file_path,
                completed=completed,
                total=total,
                elapsed_seconds=float(elapsed or 0.0),
            ),
        )

    def add_failure(file_path: str, error: str, elapsed: float):
        nonlocal completed, failed_manifest_written
        completed += 1
        failures.append((file_path, str(error)))
        failed_manifest_written = _copy_failed_input(
            file_path=file_path,
            error=str(error),
            failed_root=failed_root,
            failed_manifest=failed_manifest,
            failed_manifest_written=failed_manifest_written,
        )
        batch_records.append(
            {
                "input_file": file_path,
                "status": "failed",
                "elapsed_seconds": elapsed,
                "error": str(error),
            }
        )
        _emit(
            on_event,
            JobEvent(
                EventKind.FILE_FAILED,
                input_file=file_path,
                message=str(error),
                completed=completed,
                total=total,
                elapsed_seconds=float(elapsed or 0.0),
            ),
        )

    def add_stopped(file_path: str, elapsed: float = 0.0, reason: str = "Stopped by user before processing."):
        nonlocal completed, stopped_count
        completed += 1
        stopped_count += 1
        batch_records.append(
            {
                "input_file": file_path,
                "status": "stopped",
                "elapsed_seconds": elapsed,
                "error": reason,
            }
        )
        _emit(
            on_event,
            JobEvent(
                EventKind.FILE_STOPPED,
                input_file=file_path,
                message=reason,
                completed=completed,
                total=total,
                elapsed_seconds=float(elapsed or 0.0),
            ),
        )

    if mode == "single" or len(files) == 1:
        queue = Queue()
        for path in files:
            queue.put(path)
        while not queue.empty():
            maybe_request_stop()
            if stop_requested:
                while not queue.empty():
                    pending_file = queue.get()
                    add_stopped(pending_file)
                    queue.task_done()
                break

            file_path = queue.get()
            _emit(on_event, JobEvent(EventKind.FILE_STARTED, input_file=file_path, completed=completed, total=total))
            success, error, elapsed = process_file(file_path, options, output_root=results_root)
            if success:
                add_success(file_path, elapsed)
            else:
                add_failure(file_path, error, elapsed)
            queue.task_done()
    elif use_gpu_overlap:
        with ThreadPoolExecutor(max_workers=workers) as cpu_executor, ThreadPoolExecutor(max_workers=1) as gpu_executor:
            pre_futures = {
                cpu_executor.submit(process_file_pre_nci, file_path, options, results_root): file_path
                for file_path in files
            }
            for file_path in files:
                _emit(on_event, JobEvent(EventKind.FILE_STARTED, input_file=file_path, completed=completed, total=total))
            post_futures = {}
            pre_cancel_applied = False

            while pre_futures or post_futures:
                maybe_request_stop()
                if stop_requested and not pre_cancel_applied:
                    for future, file_path in list(pre_futures.items()):
                        if future.cancel():
                            pre_futures.pop(future, None)
                            add_stopped(file_path)
                    pre_cancel_applied = True

                done_pre = []
                if pre_futures:
                    try:
                        for future in as_completed(pre_futures, timeout=0.2):
                            done_pre.append(future)
                            if len(done_pre) >= workers:
                                break
                    except TimeoutError:
                        pass

                for future in done_pre:
                    file_path = pre_futures.pop(future)
                    if future.cancelled():
                        add_stopped(file_path)
                        continue
                    try:
                        success, error, pre_elapsed, pipeline, context = future.result()
                    except CancelledError:
                        add_stopped(file_path)
                        continue
                    except Exception as exc:
                        success, error, pre_elapsed, pipeline, context = False, str(exc), 0.0, None, None

                    if success and not stop_requested:
                        post_future = gpu_executor.submit(process_file_post_nci, pipeline, context, file_path)
                        post_futures[post_future] = (file_path, pre_elapsed)
                    elif success:
                        add_stopped(
                            file_path,
                            elapsed=pre_elapsed,
                            reason="Stopped by user after pre-NCI stage.",
                        )
                    else:
                        add_failure(file_path, error, pre_elapsed)

                done_post = []
                if post_futures:
                    try:
                        for future in as_completed(post_futures, timeout=0.2):
                            done_post.append(future)
                            break
                    except TimeoutError:
                        pass

                for future in done_post:
                    file_path, pre_elapsed = post_futures.pop(future)
                    if future.cancelled():
                        add_stopped(file_path, elapsed=pre_elapsed)
                        continue
                    try:
                        success, error, post_elapsed = future.result()
                    except CancelledError:
                        add_stopped(file_path, elapsed=pre_elapsed)
                        continue
                    except Exception as exc:
                        success, error, post_elapsed = False, str(exc), 0.0

                    total_elapsed_file = pre_elapsed + post_elapsed
                    if success:
                        add_success(file_path, total_elapsed_file)
                    else:
                        add_failure(file_path, error, total_elapsed_file)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for file_path in files:
                _emit(on_event, JobEvent(EventKind.FILE_STARTED, input_file=file_path, completed=completed, total=total))
                futures[executor.submit(process_file, file_path, options, results_root)] = file_path
            cancellation_applied = False

            while futures:
                maybe_request_stop()
                if stop_requested and not cancellation_applied:
                    for future, file_path in list(futures.items()):
                        if future.cancel():
                            futures.pop(future, None)
                            add_stopped(file_path)
                    cancellation_applied = True

                done_futures = []
                try:
                    for future in as_completed(futures, timeout=1):
                        done_futures.append(future)
                        if len(done_futures) >= workers:
                            break
                except TimeoutError:
                    pass

                for future in done_futures:
                    file_path = futures.pop(future)
                    if future.cancelled():
                        add_stopped(file_path)
                        continue
                    try:
                        success, error, elapsed = future.result()
                    except CancelledError:
                        add_stopped(file_path)
                        continue
                    except Exception as exc:
                        success, error, elapsed = False, str(exc), 0.0

                    if success:
                        add_success(file_path, elapsed)
                    else:
                        add_failure(file_path, error, elapsed)

    total_time = time.perf_counter() - t0
    merged_records = _dedupe_batch_records(existing_batch_records + batch_records)
    aggregate_total_time = _sum_elapsed_seconds(merged_records)
    aggregate_json_path, aggregate_csv_path, quadrant_payload, batch_delta_json_path, batch_delta_txt_path = write_batch_aggregate_json(
        directory=directory,
        results_root=results_root,
        records=merged_records,
        mode=mode,
        workers=workers,
        total_time=aggregate_total_time,
        water=water_mode,
        interactive_quadrant_plot=(interactive_quadrant_plot or stop_requested),
    )
    _cleanup_compound_knf_json_outputs(results_root, water=water_mode)

    result = BatchResult(
        input_directory=os.path.abspath(directory),
        results_root=results_root,
        records=[_batch_file_record_from_dict(record) for record in merged_records],
        mode=mode,
        workers=workers,
        total_time_seconds=total_time,
        aggregate_json_path=aggregate_json_path,
        aggregate_csv_path=aggregate_csv_path,
        batch_delta_json_path=batch_delta_json_path,
        batch_delta_txt_path=batch_delta_txt_path,
        quadrant_payload=quadrant_payload,
        skipped_existing=skipped_existing,
        stopped_count=stopped_count,
        failures=failures,
    )
    _emit(
        on_event,
        JobEvent(
            EventKind.BATCH_FINISHED,
            completed=completed,
            total=total,
            elapsed_seconds=total_time,
            payload={"result": result},
        ),
    )
    return result


def run_universal_kuid_job(directory: str, options) -> dict | None:
    water_mode = bool(getattr(options, "water", False))
    output_base = resolve_results_root(directory, getattr(options, "output_dir", None))
    combined_output_root = os.path.join(output_base, "Combined Results")
    source_specs = _discover_universal_batch_sources(directory, water=water_mode)

    if not source_specs:
        return None

    return _combine_batch_sources(
        source_directory=directory,
        source_specs=source_specs,
        output_root=combined_output_root,
        water=water_mode,
        mode="universal_kuid_recompute",
    )


def run_batch_directory_batched_job(
    directory: str,
    options,
    on_event: OnEvent | None = None,
) -> dict | None:
    files = _discover_input_files(directory)
    if not files:
        return None

    batch_count = _resolve_requested_batch_count(
        requested_batches=getattr(options, "batches", None),
        total_files=len(files),
        workers_hint=getattr(options, "workers", None),
    )
    partitions = [part for part in _split_evenly(files, batch_count) if part]
    if not partitions:
        return None

    water_mode = bool(getattr(options, "water", False))
    output_base = resolve_results_root(directory, getattr(options, "output_dir", None))
    batches_output_root = os.path.join(output_base, "Batches")
    combined_output_root = os.path.join(output_base, "Combined Results")
    os.makedirs(batches_output_root, exist_ok=True)

    source_specs = []
    batch_results = []
    for idx, batch_files in enumerate(partitions, start=1):
        source_batch = f"batch_{idx:02d}"
        batch_results_root = os.path.join(batches_output_root, source_batch)
        os.makedirs(batch_results_root, exist_ok=True)

        batch_result = run_batch_directory_job(
            directory=directory,
            options=options,
            file_paths=batch_files,
            results_root_override=batch_results_root,
            on_event=on_event,
        )
        batch_results.append(batch_result)

        batch_json = os.path.join(batch_results_root, _final_output_name("batch_knf.json", water_mode))
        batch_csv = _existing_batch_csv_path(batch_results_root, water=water_mode)
        if os.path.exists(batch_json):
            source_specs.append({"source_batch": source_batch, "path": batch_json, "type": "json"})
        elif os.path.exists(batch_csv):
            source_specs.append({"source_batch": source_batch, "path": batch_csv, "type": "csv"})
        else:
            logging.warning(
                "Batch source outputs not found for %s (expected %s or %s/%s).",
                source_batch,
                batch_json,
                os.path.join(batch_results_root, _final_output_name(_BATCH_PRIMARY_CSV_NAME, water_mode)),
                os.path.join(batch_results_root, _final_output_name(_BATCH_LEGACY_CSV_NAMES[0], water_mode)),
            )

    if not source_specs:
        return None

    combined = _combine_batch_sources(
        source_directory=directory,
        source_specs=source_specs,
        output_root=combined_output_root,
        water=water_mode,
        mode="combined_from_internal_batches",
    )
    return {"combined": combined, "batches": batch_results}
