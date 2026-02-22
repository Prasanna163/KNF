import argparse
import sys
import os
import shutil
import logging
import time
import json
import csv
import statistics
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError, CancelledError
from datetime import datetime, timezone
from .pipeline import KNFPipeline
from . import utils
from . import autoconfig
from . import first_run
import psutil
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.table import Table

CLI_TITLE = "KNF-Core v1.0"
DISPLAY_NAME_LIMIT = 40
STOP_KEY = "q"

def check_dependencies(multiwfn_path: str = None, nci_backend: str = "torch"):
    """Checks if required external tools are available in PATH."""
    # Attempt to add Multiwfn to PATH if missing
    utils.ensure_multiwfn_in_path(explicit_path=multiwfn_path)

    missing = []
    
    if not shutil.which('obabel'):
        missing.append('obabel (Open Babel)')
        
    if not shutil.which('xtb'):
        missing.append('xtb (Extended Tight Binding)')
        
    backend = (nci_backend or "multiwfn").strip().lower()
    if backend == "multiwfn" and not shutil.which('Multiwfn') and not shutil.which('Multiwfn.exe'):
        missing.append('Multiwfn')
        
    if missing:
        print("WARNING: The following required tools were not found in your PATH:")
        for tool in missing:
            print(f"  - {tool}")
        print("Please resolve these dependencies for full functionality.")
        print("-" * 50)

def _build_pipeline(file_path: str, args, output_root: str = None) -> KNFPipeline:
    return KNFPipeline(
        input_file=file_path,
        charge=args.charge,
        spin=args.spin,
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
    )


def process_file(file_path: str, args, output_root: str = None):
    """Runs the pipeline for a single file and returns status."""
    start = time.perf_counter()
    try:
        pipeline = _build_pipeline(file_path, args, output_root=output_root)
        pipeline.run()
        return True, None, time.perf_counter() - start
    except Exception as e:
        if args.debug:
            logging.exception(f"Error processing {file_path}:")
        else:
            logging.error(f"Error processing {file_path}: {e}")
        return False, str(e), time.perf_counter() - start


def process_file_pre_nci(file_path: str, args, output_root: str = None):
    """Runs pre-NCI stages only (geometry + xTB) and returns pipeline context."""
    start = time.perf_counter()
    try:
        pipeline = _build_pipeline(file_path, args, output_root=output_root)
        context = pipeline.run_pre_nci_stage()
        return True, None, time.perf_counter() - start, pipeline, context
    except Exception as e:
        if args.debug:
            logging.exception(f"Pre-NCI error processing {file_path}:")
        else:
            logging.error(f"Pre-NCI error processing {file_path}: {e}")
        return False, str(e), time.perf_counter() - start, None, None


def process_file_post_nci(pipeline: KNFPipeline, context: dict, file_path: str):
    """Runs post-NCI stage (NCI + SNCI/SCDI + final output write)."""
    start = time.perf_counter()
    try:
        pipeline.run_post_nci_stage(context)
        return True, None, time.perf_counter() - start
    except Exception as e:
        if pipeline.debug:
            logging.exception(f"Post-NCI error processing {file_path}:")
        else:
            logging.error(f"Post-NCI error processing {file_path}: {e}")
        return False, str(e), time.perf_counter() - start


def _fmt_elapsed(seconds: float) -> str:
    seconds = int(max(0, round(seconds)))
    mm, ss = divmod(seconds, 60)
    hh, mm = divmod(mm, 60)
    if hh > 0:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"


def _display_name(file_path: str) -> str:
    stem = os.path.splitext(os.path.basename(file_path))[0]
    cleaned = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in stem)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_") or stem
    label = cleaned + os.path.splitext(os.path.basename(file_path))[1]
    if len(label) > DISPLAY_NAME_LIMIT:
        return label[: DISPLAY_NAME_LIMIT - 3] + "..."
    return label


def _active_tool_ram_mb() -> float:
    total = 0
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            name = (p.info.get("name") or "").lower()
            if "xtb" in name or "multiwfn" in name:
                mem = p.info.get("memory_info")
                total += int(getattr(mem, "rss", 0))
        except Exception:
            continue
    return total / (1024 * 1024)


def _self_ram_mb() -> float:
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _safe_float(value):
    try:
        if value is None:
            return None
        val = float(value)
        if val != val:  # NaN
            return None
        return val
    except (TypeError, ValueError):
        return None


def _normalize_minmax(values, invert: bool = False):
    finite = [v for v in values if v is not None]
    if not finite:
        return [None] * len(values)

    vmin = min(finite)
    vmax = max(finite)
    if abs(vmax - vmin) <= 1e-12:
        return [0.5 if v is not None else None for v in values]

    out = []
    for v in values:
        if v is None:
            out.append(None)
            continue
        normalized = (v - vmin) / (vmax - vmin)
        if invert:
            normalized = 1.0 - normalized
        out.append(max(0.0, min(1.0, float(normalized))))
    return out


def _classify_quadrant(x: float, y: float, mx: float, my: float) -> str:
    if x >= mx and y >= my:
        return "Q1"
    if x < mx and y >= my:
        return "Q2"
    if x < mx and y < my:
        return "Q3"
    return "Q4"


def _compute_norm_and_quadrants(
    enriched_records: list[dict],
    results_root: str,
    interactive_plot: bool = False,
):
    normalized_rows = []
    for entry in enriched_records:
        if entry.get("status") != "success":
            continue
        knf_data = entry.get("knf") or {}
        snci_val = _safe_float(knf_data.get("SNCI"))
        scdi_val = _safe_float(knf_data.get("SCDI"))
        scdi_var = _safe_float(knf_data.get("SCDI_variance"))
        normalized_rows.append(
            {
                "entry": entry,
                "file_name": entry.get("input_file_name", ""),
                "snci": snci_val,
                "scdi": scdi_val,
                "scdi_variance": scdi_var,
                "snci_norm": None,
                "scdi_norm": None,
            }
        )

    if normalized_rows:
        snci_values = [row["snci"] for row in normalized_rows]
        snci_norm = _normalize_minmax(snci_values, invert=False)
        for row, norm_val in zip(normalized_rows, snci_norm):
            row["snci_norm"] = norm_val

        has_complete_scdi = all(row["scdi"] is not None for row in normalized_rows)
        if has_complete_scdi:
            for row in normalized_rows:
                row["scdi_norm"] = max(0.0, min(1.0, float(row["scdi"])))
            scdi_norm_source = "SCDI"
        else:
            variance_values = [row["scdi_variance"] for row in normalized_rows]
            scdi_norm = _normalize_minmax(variance_values, invert=True)
            for row, norm_val in zip(normalized_rows, scdi_norm):
                row["scdi_norm"] = norm_val
            scdi_norm_source = "SCDI_variance_inverse_minmax"

        for row in normalized_rows:
            row["entry"]["SNCI_Norm"] = row["snci_norm"]
            row["entry"]["SCDI_Norm"] = row["scdi_norm"]
    else:
        scdi_norm_source = None

    valid_plot_rows = [
        row
        for row in normalized_rows
        if row["snci_norm"] is not None and row["scdi_norm"] is not None
    ]
    if not valid_plot_rows:
        return {
            "SNCI_Norm_source": "minmax",
            "SCDI_Norm_source": scdi_norm_source,
            "median_SNCI_Norm": None,
            "median_SCDI_Norm": None,
            "quadrants": {},
            "quadrant_json": None,
            "quadrant_plot_png": None,
            "plot_error": "No successful normalized rows available.",
        }

    snci_norm_vals = [row["snci_norm"] for row in valid_plot_rows]
    scdi_norm_vals = [row["scdi_norm"] for row in valid_plot_rows]
    median_x = float(statistics.median(snci_norm_vals))
    median_y = float(statistics.median(scdi_norm_vals))

    quadrants = {
        "Q1": {"count": 0, "files": []},
        "Q2": {"count": 0, "files": []},
        "Q3": {"count": 0, "files": []},
        "Q4": {"count": 0, "files": []},
    }
    for row in valid_plot_rows:
        q = _classify_quadrant(row["snci_norm"], row["scdi_norm"], median_x, median_y)
        quadrants[q]["count"] += 1
        quadrants[q]["files"].append(row["file_name"])
        row["entry"]["quadrant"] = q

    quadrant_json_path = os.path.join(results_root, "snci_scdi_quadrants.json")
    with open(quadrant_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "median_SNCI_Norm": median_x,
                "median_SCDI_Norm": median_y,
                "quadrants": quadrants,
            },
            f,
            indent=2,
        )

    plot_png_path = os.path.join(results_root, "snci_scdi_quadrants.png")
    plot_error = None
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        ax.scatter(snci_norm_vals, scdi_norm_vals, s=45, alpha=0.75)
        ax.axvline(median_x, color="crimson", linestyle="--", linewidth=1.8, label=f"Median X = {median_x:.4f}")
        ax.axhline(median_y, color="darkgreen", linestyle="--", linewidth=1.8, label=f"Median Y = {median_y:.4f}")
        ax.set_xlabel("SNCI_Norm")
        ax.set_ylabel("SCDI_Norm")
        ax.set_title("SCDI_Norm vs SNCI_Norm with Median Quadrants")
        ax.grid(True, linestyle="--", alpha=0.4)

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_left = x_min + 0.08 * (x_max - x_min)
        x_right = x_min + 0.58 * (x_max - x_min)
        y_top = y_min + 0.94 * (y_max - y_min)
        y_bottom = y_min + 0.40 * (y_max - y_min)

        ax.text(x_right, y_top, f"Q1\n(n={quadrants['Q1']['count']})", fontsize=13, fontweight="bold")
        ax.text(x_left, y_top, f"Q2\n(n={quadrants['Q2']['count']})", fontsize=13, fontweight="bold")
        ax.text(x_left, y_bottom, f"Q3\n(n={quadrants['Q3']['count']})", fontsize=13, fontweight="bold")
        ax.text(x_right, y_bottom, f"Q4\n(n={quadrants['Q4']['count']})", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(plot_png_path)
        if interactive_plot:
            plt.show()
        plt.close(fig)
    except Exception as e:
        plot_error = str(e)
        plot_png_path = None

    return {
        "SNCI_Norm_source": "minmax",
        "SCDI_Norm_source": scdi_norm_source,
        "median_SNCI_Norm": median_x,
        "median_SCDI_Norm": median_y,
        "quadrants": quadrants,
        "quadrant_json": quadrant_json_path,
        "quadrant_plot_png": plot_png_path,
        "plot_error": plot_error,
    }


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


def run_single_file(file_path: str, args):
    results_root = resolve_results_root(file_path, args.output_dir)
    console = Console()
    logical = psutil.cpu_count(logical=True) or (os.cpu_count() or 1)
    physical = psutil.cpu_count(logical=False) or max(1, logical // 2)

    peak_cpu = 0.0
    peak_ram = 0.0
    t0 = time.perf_counter()
    psutil.cpu_percent(interval=None)

    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
    task_id = progress.add_task("Single Job", total=1)

    def render(status_text: str, status_style: str):
        avg_cpu = psutil.cpu_percent(interval=None)
        ram_mb = max(_active_tool_ram_mb(), _self_ram_mb())
        nonlocal peak_cpu, peak_ram
        if avg_cpu >= 0:
            peak_cpu = max(peak_cpu, avg_cpu)
        peak_ram = max(peak_ram, ram_mb)

        header = Table.grid(padding=(0, 2))
        header.add_column(style="bold")
        header.add_column()
        header.add_row("KNF-Core", "v1.0")
        header.add_row("Detected", f"{physical}C / {logical}T")
        header.add_row("Mode", "single")
        header.add_row("File", _display_name(file_path))
        header.add_row("Output", results_root)
        header.add_row("Avg CPU", f"{avg_cpu:.1f}%")
        header.add_row("RAM", f"{ram_mb:.1f} MB")
        header.add_row("Status", f"[{status_style}]{status_text}[/{status_style}]")

        return Group(
            Panel(header, title="KNF-Core Single Run", border_style="cyan"),
            progress,
        )

    success = False
    error = None
    elapsed = 0.0
    with Live(render("running", "yellow"), console=console, refresh_per_second=5, transient=False) as live:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(process_file, file_path, args, results_root)
            while not future.done():
                live.update(render("running", "yellow"))
                time.sleep(0.4)

            success, error, elapsed = future.result()
            progress.advance(task_id, 1)
            live.update(render("completed" if success else "failed", "green" if success else "red"))

    total_time = elapsed if elapsed > 0 else (time.perf_counter() - t0)
    throughput = ((1 / total_time) * 3600) if (success and total_time > 0) else 0.0

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Total files", "1")
    summary.add_row("Success", "1" if success else "0")
    summary.add_row("Failed", "0" if success else "1")
    summary.add_row("Total time", _fmt_elapsed(total_time))
    summary.add_row("Molecule time", f"{elapsed:.1f}s" if elapsed > 0 else "n/a")
    summary.add_row("Throughput", f"{throughput:.1f} jobs/hour" if success else "n/a")
    summary.add_row("Peak CPU", f"{peak_cpu:.1f}%")
    summary.add_row("Peak RAM", f"{peak_ram:.1f} MB")
    console.print(Panel(summary, title="Run Completed", border_style="green" if success else "red"))

    if not success:
        fail_table = Table(title="Failure", expand=True)
        fail_table.add_column("File")
        fail_table.add_column("Error")
        fail_table.add_row(os.path.basename(file_path), str(error))
        console.print(fail_table)

def resolve_results_root(input_path: str, output_dir: str = None) -> str:
    """Resolves the top-level Results directory."""
    if output_dir:
        return os.path.abspath(output_dir)

    if os.path.isdir(input_path):
        return os.path.join(os.path.abspath(input_path), "Results")

    return os.path.join(os.path.dirname(os.path.abspath(input_path)), "Results")


def write_batch_aggregate_json(
    directory: str,
    results_root: str,
    records: list[dict],
    mode: str,
    workers: int,
    total_time: float,
    interactive_quadrant_plot: bool = False,
):
    """Writes combined JSON and CSV payloads for batch outputs."""
    aggregate_path = os.path.join(results_root, "batch_knf.json")
    aggregate_csv_path = os.path.join(results_root, "batch_knf.csv")
    os.makedirs(results_root, exist_ok=True)

    enriched_records = []
    knf_results = []
    success_count = 0
    failure_count = 0
    stopped_count = 0

    for record in records:
        input_file = os.path.abspath(record["input_file"])
        stem = os.path.splitext(os.path.basename(input_file))[0]
        result_dir = os.path.join(results_root, stem)
        knf_path = os.path.join(result_dir, "knf.json")

        entry = {
            "input_file": input_file,
            "input_file_name": os.path.basename(input_file),
            "result_dir": result_dir,
            "status": record["status"],
            "elapsed_seconds": round(float(record.get("elapsed_seconds", 0.0)), 4),
            "error": record.get("error"),
            "knf": None,
        }

        if record["status"] == "success" and os.path.exists(knf_path):
            try:
                with open(knf_path, "r", encoding="utf-8") as f:
                    knf_data = json.load(f)
                entry["knf"] = knf_data
                knf_results.append(
                    {
                        "input_file": input_file,
                        "input_file_name": os.path.basename(input_file),
                        "result_dir": result_dir,
                        "knf": knf_data,
                    }
                )
                success_count += 1
            except Exception as e:
                entry["status"] = "failed"
                entry["error"] = f"Failed to read knf.json: {e}"
                failure_count += 1
        elif record["status"] == "success":
            entry["status"] = "failed"
            entry["error"] = "Missing knf.json output."
            failure_count += 1
        elif record["status"] == "stopped":
            stopped_count += 1
        else:
            failure_count += 1

        enriched_records.append(entry)

    quadrant_payload = _compute_norm_and_quadrants(
        enriched_records=enriched_records,
        results_root=results_root,
        interactive_plot=interactive_quadrant_plot,
    )

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": os.path.abspath(directory),
        "results_root": os.path.abspath(results_root),
        "mode": mode,
        "workers": workers,
        "summary": {
            "total_files": len(records),
            "successful_files": success_count,
            "failed_files": failure_count,
            "stopped_files": stopped_count,
            "total_time_seconds": round(float(total_time), 4),
        },
        "normalization_and_quadrants": quadrant_payload,
        "records": enriched_records,
        "knf_results": knf_results,
    }

    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_fields = ["File"] + [f"f{i}" for i in range(1, 10)] + ["SNCI", "SCDI", "SCDI_variance", "SNCI_Norm", "SCDI_Norm"]
    with open(aggregate_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for entry in enriched_records:
            knf_data = entry.get("knf") or {}
            knf_vector = knf_data.get("KNF_vector") or []
            row = {
                "File": entry.get("input_file_name", ""),
                "SNCI": knf_data.get("SNCI", ""),
                "SCDI": knf_data.get("SCDI", ""),
                "SCDI_variance": knf_data.get("SCDI_variance", ""),
                "SNCI_Norm": entry.get("SNCI_Norm", ""),
                "SCDI_Norm": entry.get("SCDI_Norm", ""),
            }
            for idx in range(9):
                row[f"f{idx + 1}"] = knf_vector[idx] if idx < len(knf_vector) else ""
            writer.writerow(row)

    return aggregate_path, aggregate_csv_path, quadrant_payload

def run_batch_directory(directory: str, args):
    """Runs the pipeline for all valid files in a directory using a queue."""
    valid_exts = {'.xyz', '.sdf', '.mol', '.pdb', '.mol2'}

    files = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if not os.path.isfile(full_path):
            continue
        ext = utils.normalized_extension(entry)
        if ext in valid_exts:
            files.append(full_path)
    files.sort()
    
    if not files:
        print(f"No molecular files found in {directory}.")
        return

    results_root = resolve_results_root(directory, args.output_dir)
    mode = args.processing.lower()
    if mode == "auto":
        mode = "multi" if len(files) > 1 else "single"
    use_gpu_overlap = (
        mode == "multi"
        and len(files) > 1
        and (args.nci_backend or "").strip().lower() == "torch"
        and (args.nci_device or "").strip().lower() == "cuda"
    )
    workers = 1
    failures = []
    batch_records = []
    succeeded = 0
    stopped_count = 0

    if mode == 'multi' and len(files) > 1:
        if args.workers is None:
            cfg = autoconfig.resolve_multi_config(
                n_jobs=len(files),
                ram_per_job_mb=args.ram_per_job,
                project_root=os.getcwd(),
                force_refresh=args.refresh_autoconfig
            )
            workers = cfg.workers
            autoconfig.apply_env_inplace(cfg)
        else:
            workers = max(1, args.workers)
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
        cfg = None

    logical = psutil.cpu_count(logical=True) or (os.cpu_count() or 1)
    physical = psutil.cpu_count(logical=False) or max(1, logical // 2)
    console = Console()
    enable_stop_key = bool(getattr(args, "enable_stop_key", False))
    interactive_quadrant_plot = bool(getattr(args, "interactive_quadrant_plot", False))
    stop_requested = False
    stop_notice_emitted = False

    if enable_stop_key:
        console.print(
            f"[yellow]Stop control enabled: press '{STOP_KEY}' to stop new jobs and finish safely.[/yellow]"
        )

    completed_rows = []
    total = len(files)
    completed = 0
    peak_cpu = 0.0
    peak_ram = 0.0
    t0 = time.perf_counter()
    psutil.cpu_percent(interval=None)

    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )
    task_id = progress.add_task("Overall", total=total)

    def maybe_request_stop() -> bool:
        nonlocal stop_requested, stop_notice_emitted
        if not stop_requested and _poll_stop_key(enable_stop_key):
            stop_requested = True
        if stop_requested and not stop_notice_emitted:
            console.print("[yellow]Stop requested. Completing running tasks and finalizing partial outputs...[/yellow]")
            stop_notice_emitted = True
        return stop_requested

    def add_success(file_path: str, elapsed: float):
        nonlocal completed, succeeded
        completed += 1
        succeeded += 1
        progress.advance(task_id, 1)
        batch_records.append(
            {
                "input_file": file_path,
                "status": "success",
                "elapsed_seconds": elapsed,
                "error": None,
            }
        )
        completed_rows.append((_display_name(file_path), f"{elapsed:.1f}s", "[green]OK[/green]"))

    def add_failure(file_path: str, error: str, elapsed: float):
        nonlocal completed
        completed += 1
        progress.advance(task_id, 1)
        failures.append((file_path, error))
        batch_records.append(
            {
                "input_file": file_path,
                "status": "failed",
                "elapsed_seconds": elapsed,
                "error": str(error),
            }
        )
        completed_rows.append((_display_name(file_path), f"{elapsed:.1f}s", "[red]FAIL[/red]"))

    def add_stopped(file_path: str, elapsed: float = 0.0, reason: str = "Stopped by user before processing."):
        nonlocal completed, stopped_count
        completed += 1
        stopped_count += 1
        progress.advance(task_id, 1)
        batch_records.append(
            {
                "input_file": file_path,
                "status": "stopped",
                "elapsed_seconds": elapsed,
                "error": reason,
            }
        )
        elapsed_text = f"{elapsed:.1f}s" if elapsed > 0 else "-"
        completed_rows.append((_display_name(file_path), elapsed_text, "[yellow]STOP[/yellow]"))

    def render(active_workers: int):
        elapsed = time.perf_counter() - t0
        avg_cpu = psutil.cpu_percent(interval=None)
        ram_mb = _active_tool_ram_mb()
        nonlocal peak_cpu, peak_ram
        peak_cpu = max(peak_cpu, avg_cpu)
        peak_ram = max(peak_ram, ram_mb)
        rate = completed / max(elapsed, 1e-6)
        eta = (total - completed) / max(rate, 1e-6) if completed else 0.0

        header = Table.grid(padding=(0, 2))
        header.add_column(style="bold")
        header.add_column()
        header.add_row("KNF-Core", "v1.0")
        header.add_row("Detected", f"{physical}C / {logical}T")
        if mode == "multi":
            if use_gpu_overlap:
                header.add_row("Mode", f"multi (cpu->gpu overlap: {workers} CPU + 1 GPU)")
            else:
                mode_text = "auto" if args.workers is None else "manual"
                header.add_row("Mode", f"multi ({mode_text} -> {workers} workers)")
        else:
            header.add_row("Mode", "single")
        header.add_row("Files", str(total))
        header.add_row("Output", results_root)
        header.add_row("Active Workers", str(active_workers))
        header.add_row("Avg CPU", f"{avg_cpu:.1f}%")
        header.add_row("RAM", f"{ram_mb:.1f} MB")
        header.add_row("ETA", _fmt_elapsed(eta))

        jobs = Table(title="Completed Jobs", expand=True)
        jobs.add_column("File", overflow="fold")
        jobs.add_column("Time", justify="right", width=8)
        jobs.add_column("Status", width=8)
        for row in completed_rows[-15:]:
            jobs.add_row(*row)
        if not completed_rows:
            jobs.add_row("-", "-", "running")

        return Group(
            Panel(header, title="KNF-Core Batch Summary", border_style="cyan"),
            progress,
            jobs,
        )

    if mode == 'single' or len(files) == 1:
        with Live(render(active_workers=1), console=console, refresh_per_second=5, transient=False) as live:
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
                    live.update(render(active_workers=0))
                    break

                file_path = queue.get()
                success, error, elapsed = process_file(file_path, args, output_root=results_root)
                if success:
                    add_success(file_path, elapsed)
                else:
                    add_failure(file_path, error, elapsed)
                live.update(render(active_workers=0))
                queue.task_done()
    elif use_gpu_overlap:
        with Live(render(active_workers=workers + 1), console=console, refresh_per_second=5, transient=False) as live:
            with ThreadPoolExecutor(max_workers=workers) as cpu_executor, ThreadPoolExecutor(max_workers=1) as gpu_executor:
                pre_futures = {
                    cpu_executor.submit(process_file_pre_nci, file_path, args, results_root): file_path
                    for file_path in files
                }
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
                        except Exception as e:
                            success, error, pre_elapsed, pipeline, context = False, str(e), 0.0, None, None

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
                        except Exception as e:
                            success, error, post_elapsed = False, str(e), 0.0

                        total_elapsed_file = pre_elapsed + post_elapsed
                        if success:
                            add_success(file_path, total_elapsed_file)
                        else:
                            add_failure(file_path, error, total_elapsed_file)

                    active_workers = min(workers, len(pre_futures)) + (1 if post_futures else 0)
                    live.update(render(active_workers=active_workers))
    else:
        with Live(render(active_workers=workers), console=console, refresh_per_second=5, transient=False) as live:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_file, file_path, args, results_root): file_path
                    for file_path in files
                }
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
                        except Exception as e:  # Defensive fallback for executor failures
                            success, error, elapsed = False, str(e), 0.0

                        if success:
                            add_success(file_path, elapsed)
                        else:
                            add_failure(file_path, error, elapsed)

                    active_workers = min(workers, len(futures))
                    live.update(render(active_workers=active_workers))

    total_time = time.perf_counter() - t0
    completed_non_stopped = max(0, total - stopped_count)
    throughput = (completed_non_stopped / total_time) * 3600 if total_time > 0 else 0.0
    avg_per_molecule = total_time / completed_non_stopped if completed_non_stopped else 0.0
    aggregate_json_path, aggregate_csv_path, quadrant_payload = write_batch_aggregate_json(
        directory=directory,
        results_root=results_root,
        records=batch_records,
        mode=mode,
        workers=workers,
        total_time=total_time,
        interactive_quadrant_plot=(interactive_quadrant_plot or stop_requested),
    )

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Total files", str(total))
    summary.add_row("Success", str(succeeded))
    summary.add_row("Failed", str(len(failures)))
    summary.add_row("Stopped", str(stopped_count))
    summary.add_row("Total time", _fmt_elapsed(total_time))
    summary.add_row("Avg per molecule", f"{avg_per_molecule:.1f}s" if completed_non_stopped else "n/a")
    summary.add_row("Throughput", f"{throughput:.1f} jobs/hour" if completed_non_stopped else "n/a")
    summary.add_row("Peak CPU", f"{peak_cpu:.1f}%")
    summary.add_row("Peak RAM", f"{peak_ram:.1f} MB")
    summary.add_row("Batch JSON", aggregate_json_path)
    summary.add_row("Batch CSV", aggregate_csv_path)
    if quadrant_payload.get("quadrant_plot_png"):
        summary.add_row("Quadrant Plot", quadrant_payload["quadrant_plot_png"])
    elif quadrant_payload.get("plot_error"):
        summary.add_row("Quadrant Plot", f"not generated ({quadrant_payload['plot_error']})")
    if quadrant_payload.get("quadrant_json"):
        summary.add_row("Quadrant JSON", quadrant_payload["quadrant_json"])
    panel_title = "Batch Stopped" if stop_requested else "Batch Completed"
    panel_color = "yellow" if stop_requested else "green"
    console.print(Panel(summary, title=panel_title, border_style=panel_color))

    if failures:
        fail_table = Table(title="Failed Files", expand=True)
        fail_table.add_column("File")
        fail_table.add_column("Error")
        for file_path, error in failures:
            fail_table.add_row(os.path.basename(file_path), str(error))
        console.print(fail_table)

def main():
    # If no arguments provided -> Interactive mode (simplified)
    if len(sys.argv) == 1:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        print("\n------------------------------------------------------------")
        print("      KNF-Core Interactive Mode")
        print("------------------------------------------------------------\n")
        
        while True:
            input_path = input("Enter path to input file or folder (or 'q' to quit): ").strip()
            if input_path.lower() == 'q':
                sys.exit(0)
                
            input_path = input_path.strip('"').strip("'")
            input_path = utils.resolve_artifacted_path(input_path)
            
            if not os.path.exists(input_path):
                print(f"Error: Path '{input_path}' not found.")
                continue
            break

        nci_mode = input(
            "Run mode [default/gpu/multiwfn] (default: default): "
        ).strip().lower()
        if nci_mode not in {"", "default", "gpu", "multiwfn"}:
            print(f"Unknown mode '{nci_mode}'. Using default.")
            nci_mode = "default"
        
        # Define defaults for interactive mode
        class Args:
            charge = 0
            spin = 1
            force = True # "Just do this" implies run it.
            clean = True # "Just do this" often implies fresh run.
            debug = True # Helpful for user to see what's happening.
            output_dir = None
            processing = 'auto'
            workers = None
            ram_per_job = 50.0
            refresh_autoconfig = False
            quiet_config = False
            full_files = False
            nci_backend = "torch"
            nci_grid_spacing = 0.2
            nci_grid_padding = 3.0
            nci_device = "cpu"
            nci_dtype = "float32"
            nci_batch_size = 250000
            nci_eig_batch_size = 200000
            nci_rho_floor = 1e-12
            nci_apply_primitive_norm = False
            scdi_var_min = None
            scdi_var_max = None
            enable_stop_key = True
            interactive_quadrant_plot = False
            
        args = Args()

        if nci_mode == "gpu":
            args.nci_backend = "torch"
            args.nci_device = "cuda"
        elif nci_mode == "multiwfn":
            args.nci_backend = "multiwfn"
            args.nci_device = "auto"

        first_ok = first_run.ensure_first_run_setup(
            require_multiwfn=(args.nci_backend == "multiwfn"),
        )
        check_dependencies(nci_backend=args.nci_backend)
        if not first_ok:
            print("First-time setup is incomplete. Please install missing tools and run again.")
            sys.exit(1)
        
        if os.path.isdir(input_path):
            mode = input("Processing mode [auto/single/multi] (default: auto): ").strip().lower()
            if mode in {'auto', 'single', 'multi'}:
                args.processing = mode
            run_batch_directory(input_path, args)
        else:
            run_single_file(input_path, args)
            
        print("\nDone.")
        return

    # Batch/CLI Mode
    # Supports: knf full <file/folder> [options]
    # Or just: knf <file/folder> [options] (as alias)
    
    # Remove 'full' if present to normalize
    if len(sys.argv) > 1 and sys.argv[1] == 'full':
        sys.argv.pop(1)
        
    parser = argparse.ArgumentParser(description="KNF-Core Execution")
    parser.add_argument('input_path', help="Path to input molecular file or directory")
    parser.add_argument('--charge', type=int, default=0, help="Total system charge")
    parser.add_argument('--spin', type=int, default=1, help="Total system spin multiplicity")
    parser.add_argument('--force', action='store_true', help="Force recomputation")
    parser.add_argument('--clean', action='store_true', help="Clean results")
    parser.add_argument('--debug', action='store_true', help="Debug logging")
    parser.add_argument(
        '--processing',
        '--processes',
        dest='processing',
        choices=['auto', 'single', 'multi'],
        default='auto',
        help="Processing mode: auto (default), single, multi"
    )
    parser.add_argument(
        '--multi',
        action='store_true',
        help="Shortcut for --processing multi"
    )
    parser.add_argument(
        '--single',
        action='store_true',
        help="Shortcut for --processing single"
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help="Optional override for worker threads. Default: auto-decide in multi mode"
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help="Custom output directory. Default: <input>/Results"
    )
    parser.add_argument(
        '--ram-per-job',
        type=float,
        default=50.0,
        help="Estimated RAM in MB per concurrent job for auto-config"
    )
    parser.add_argument(
        '--refresh-autoconfig',
        action='store_true',
        help="Recompute and overwrite one-time auto-config cache"
    )
    parser.add_argument(
        '--quiet-config',
        action='store_true',
        help="Hide auto-configuration summary banner"
    )
    parser.add_argument(
        '--full-files',
        action='store_true',
        help=(
            "Keep all intermediate and large files. "
            "Default behavior is storage-efficient cleanup."
        ),
    )
    parser.add_argument(
        '--enable-stop-key',
        action='store_true',
        help=f"Enable graceful stop during batch runs by pressing '{STOP_KEY}'.",
    )
    parser.add_argument(
        '--interactive-quadrant-plot',
        action='store_true',
        help="Open an interactive SNCI_Norm vs SCDI_Norm quadrant plot window after batch aggregation.",
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help="Shortcut: use torch NCI backend on CUDA"
    )
    parser.add_argument(
        '--multiwfn',
        action='store_true',
        help="Use Multiwfn backend for NCI instead of default Torch backend"
    )
    parser.add_argument(
        '--nci-backend',
        choices=['multiwfn', 'torch'],
        default='torch',
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nci-grid-spacing',
        type=float,
        default=0.2,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nci-grid-padding',
        type=float,
        default=3.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nci-device',
        default='cpu',
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nci-dtype',
        choices=['float32', 'float64'],
        default='float32',
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nci-batch-size',
        type=int,
        default=250000,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nci-eig-batch-size',
        type=int,
        default=200000,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nci-rho-floor',
        type=float,
        default=1e-12,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nci-apply-primitive-norm',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--scdi-var-min',
        type=float,
        default=None,
        help="Fixed global Var_min for SCDI normalization.",
    )
    parser.add_argument(
        '--scdi-var-max',
        type=float,
        default=None,
        help="Fixed global Var_max for SCDI normalization.",
    )
    parser.add_argument(
        '--refresh-first-run',
        action='store_true',
        help="Re-run one-time first-run setup and overwrite its cached state"
    )
    parser.add_argument(
        '--multiwfn-path',
        default=None,
        help="Path to Multiwfn executable or folder (saved for future runs)"
    )
    
    args = parser.parse_args()
    args.input_path = utils.resolve_artifacted_path(args.input_path)

    if args.multi and args.single:
        parser.error("Use only one of --multi or --single.")
    if args.gpu and args.multiwfn:
        parser.error("Use only one of --gpu or --multiwfn.")

    if args.multi:
        args.processing = "multi"
    elif args.single:
        args.processing = "single"

    if args.multiwfn:
        args.nci_backend = "multiwfn"
        args.nci_device = "auto"
    elif args.gpu:
        args.nci_backend = "torch"
        args.nci_device = "cuda"
        args.nci_dtype = "float64"

    # Configure logging after CLI args are known.
    if args.debug:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

    first_ok = first_run.ensure_first_run_setup(
        force=args.refresh_first_run,
        multiwfn_path=args.multiwfn_path,
        require_multiwfn=(args.nci_backend == "multiwfn"),
    )
    check_dependencies(multiwfn_path=args.multiwfn_path, nci_backend=args.nci_backend)
    if not first_ok:
        logging.error("First-time setup is incomplete. Install missing tools and retry.")
        sys.exit(1)
    
    # If user provided flags, use them.
    # Default behavior for 'knf <file>' without flags is now determined by argparse defaults.
    
    if os.path.isdir(args.input_path):
        run_batch_directory(args.input_path, args)
    else:
        run_single_file(args.input_path, args)

if __name__ == "__main__":
    main()
