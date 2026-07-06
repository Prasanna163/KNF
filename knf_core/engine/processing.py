import gc
import logging
import time

from ..pipeline import KNFPipeline


def _best_effort_release_memory() -> None:
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch  # type: ignore
        if bool(getattr(torch, "cuda", None)) and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _is_transient_file_error(exc: Exception) -> bool:
    if isinstance(exc, (PermissionError, FileNotFoundError, BlockingIOError)):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in {5, 11, 13, 16, 32}:
        return True
    msg = str(exc).lower()
    markers = (
        "permission denied",
        "access is denied",
        "being used by another process",
        "resource temporarily unavailable",
        "cannot open",
        "file is locked",
    )
    return any(m in msg for m in markers)


def _is_oom_error(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    markers = ("out of memory", "cuda oom", "cuda out of memory", "cublas_status_alloc_failed")
    return any(m in msg for m in markers)


def _build_pipeline(
    file_path: str,
    args,
    output_root: str = None,
    batch_size: int = 1,
    progress_callback=None,
) -> KNFPipeline:
    xtb_engine = str(getattr(args, "xtb_engine", "xtbx") or "xtbx").strip().lower()
    # The GPU is "available" to the xTB router when this run is configured to use
    # CUDA for the NCI grid (only set once a CUDA-capable GPU is detected). This
    # decouples the per-molecule xTB GPU decision from the NCI device: the
    # throughput-aware router (route_xtb) decides whether xtb actually uses it,
    # rather than blindly forcing --gpu on every molecule of a CUDA run.
    xtb_gpu_available = (
        str(getattr(args, "nci_backend", "") or "").strip().lower() == "torch"
        and str(getattr(args, "nci_device", "") or "").strip().lower() == "cuda"
    )
    xtb_explicit_gpu = bool(getattr(args, "gpu", False))
    return KNFPipeline(
        input_file=file_path,
        charge=args.charge,
        spin=args.spin,
        water=bool(getattr(args, "water", False)),
        force=args.force,
        clean=args.clean,
        debug=args.debug,
        output_root=output_root,
        keep_full_files=args.full_files,
        nci_backend=args.nci_backend,
        nci_grid_spacing=args.nci_grid_spacing,
        nci_grid_padding=args.nci_grid_padding,
        nci_device=args.nci_device,
        nci_dtype=args.nci_dtype,
        nci_batch_size=args.nci_batch_size,
        nci_eig_batch_size=args.nci_eig_batch_size,
        nci_rho_floor=args.nci_rho_floor,
        nci_apply_primitive_norm=args.nci_apply_primitive_norm,
        scdi_var_min=args.scdi_var_min,
        scdi_var_max=args.scdi_var_max,
        wbo_mode=getattr(args, "wbo_mode", "native"),
        preopt_engine=getattr(args, "preopt", "geoinit"),
        xtb_engine=xtb_engine,
        xtb_gpu_atom_cutoff=getattr(args, "xtb_gpu_atoms", 350),
        xtb_gpu_available=xtb_gpu_available,
        xtb_explicit_gpu=xtb_explicit_gpu,
        xtb_batch_size=batch_size,
        sp_only=bool(getattr(args, "sp", False)),
        progress_callback=progress_callback,
    )


def process_file(file_path: str, args, output_root: str = None, batch_size: int = 1, progress_callback=None):
    """Runs the pipeline for a single file and returns status."""
    start = time.perf_counter()
    attempts = 3
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            pipeline = _build_pipeline(
                file_path,
                args,
                output_root=output_root,
                batch_size=batch_size,
                progress_callback=progress_callback,
            )
            pipeline.run()
            return True, None, time.perf_counter() - start
        except Exception as e:
            last_error = e
            retryable = _is_transient_file_error(e) or _is_oom_error(e)
            if _is_oom_error(e):
                _best_effort_release_memory()
            if retryable and attempt < attempts:
                wait_s = 1.5 * attempt
                logging.warning("Retrying %s after error (%s/%s): %s", file_path, attempt, attempts, e)
                time.sleep(wait_s)
                continue
            if args.debug:
                logging.exception(f"Error processing {file_path}:")
            else:
                logging.error(f"Error processing {file_path}: {e}")
            return False, str(e), time.perf_counter() - start
    return False, str(last_error), time.perf_counter() - start


def process_file_pre_nci(file_path: str, args, output_root: str = None, batch_size: int = 1, progress_callback=None):
    """Runs pre-NCI stages only (geometry + xTB) and returns pipeline context."""
    start = time.perf_counter()
    attempts = 3
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            pipeline = _build_pipeline(
                file_path,
                args,
                output_root=output_root,
                batch_size=batch_size,
                progress_callback=progress_callback,
            )
            context = pipeline.run_pre_nci_stage()
            return True, None, time.perf_counter() - start, pipeline, context
        except Exception as e:
            last_error = e
            retryable = _is_transient_file_error(e) or _is_oom_error(e)
            if _is_oom_error(e):
                _best_effort_release_memory()
            if retryable and attempt < attempts:
                wait_s = 1.5 * attempt
                logging.warning("Retrying pre-NCI %s after error (%s/%s): %s", file_path, attempt, attempts, e)
                time.sleep(wait_s)
                continue
            if args.debug:
                logging.exception(f"Pre-NCI error processing {file_path}:")
            else:
                logging.error(f"Pre-NCI error processing {file_path}: {e}")
            return False, str(e), time.perf_counter() - start, None, None
    return False, str(last_error), time.perf_counter() - start, None, None


def process_file_post_nci(pipeline: KNFPipeline, context: dict, file_path: str):
    """Runs post-NCI stage (NCI + SNCI/SCDI + final output write)."""
    start = time.perf_counter()
    attempts = 3
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            pipeline.run_post_nci_stage(context)
            return True, None, time.perf_counter() - start
        except Exception as e:
            last_error = e
            retryable = _is_transient_file_error(e) or _is_oom_error(e)
            if _is_oom_error(e):
                _best_effort_release_memory()
            if retryable and attempt < attempts:
                wait_s = 1.5 * attempt
                logging.warning("Retrying post-NCI %s after error (%s/%s): %s", file_path, attempt, attempts, e)
                time.sleep(wait_s)
                continue
            if pipeline.debug:
                logging.exception(f"Post-NCI error processing {file_path}:")
            else:
                logging.error(f"Post-NCI error processing {file_path}: {e}")
            return False, str(e), time.perf_counter() - start
    return False, str(last_error), time.perf_counter() - start
