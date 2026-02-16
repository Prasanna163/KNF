import argparse
import sys
import os
import shutil
import logging
import time
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from .pipeline import KNFPipeline
from . import utils
from . import autoconfig
import psutil
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.table import Table

CLI_TITLE = "KNF-Core v1.0"
DISPLAY_NAME_LIMIT = 40

def check_dependencies():
    """Checks if required external tools are available in PATH."""
    # Attempt to add Multiwfn to PATH if missing
    utils.ensure_multiwfn_in_path()

    missing = []
    
    if not shutil.which('obabel'):
        missing.append('obabel (Open Babel)')
        
    if not shutil.which('xtb'):
        missing.append('xtb (Extended Tight Binding)')
        
    if not shutil.which('Multiwfn') and not shutil.which('Multiwfn.exe'):
        missing.append('Multiwfn')
        
    if missing:
        print("WARNING: The following required tools were not found in your PATH:")
        for tool in missing:
            print(f"  - {tool}")
        print("Please resolve these dependencies for full functionality.")
        print("-" * 50)

def process_file(file_path: str, args, output_root: str = None):
    """Runs the pipeline for a single file and returns status."""
    start = time.perf_counter()
    try:
        pipeline = KNFPipeline(
            input_file=file_path,
            charge=args.charge,
            spin=args.spin,
            force=args.force,
            clean=args.clean,
            debug=args.debug,
            output_root=output_root
        )
        pipeline.run()
        return True, None, time.perf_counter() - start
    except Exception as e:
        if args.debug:
            logging.exception(f"Error processing {file_path}:")
        else:
            logging.error(f"Error processing {file_path}: {e}")
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

def run_batch_directory(directory: str, args):
    """Runs the pipeline for all valid files in a directory using a queue."""
    valid_exts = {'.xyz', '.sdf', '.mol', '.pdb', '.mol2'}
    
    files = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in valid_exts
    )
    
    if not files:
        print(f"No molecular files found in {directory}.")
        return

    results_root = resolve_results_root(directory, args.output_dir)
    mode = args.processing.lower()
    workers = 1
    failures = []
    succeeded = 0

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
                file_path = queue.get()
                success, error, elapsed = process_file(file_path, args, output_root=results_root)
                completed += 1
                progress.advance(task_id, 1)
                if success:
                    succeeded += 1
                    completed_rows.append((_display_name(file_path), f"{elapsed:.1f}s", "[green]OK[/green]"))
                else:
                    failures.append((file_path, error))
                    completed_rows.append((_display_name(file_path), f"{elapsed:.1f}s", "[red]FAIL[/red]"))
                live.update(render(active_workers=0))
                queue.task_done()
    else:
        with Live(render(active_workers=workers), console=console, refresh_per_second=5, transient=False) as live:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_file, file_path, args, results_root): file_path
                    for file_path in files
                }

                while futures:
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
                        try:
                            success, error, elapsed = future.result()
                        except Exception as e:  # Defensive fallback for executor failures
                            success, error, elapsed = False, str(e), 0.0

                        completed += 1
                        progress.advance(task_id, 1)
                        if success:
                            succeeded += 1
                            completed_rows.append((_display_name(file_path), f"{elapsed:.1f}s", "[green]OK[/green]"))
                        else:
                            completed_rows.append((_display_name(file_path), f"{elapsed:.1f}s", "[red]FAIL[/red]"))
                            failures.append((file_path, error))

                    active_workers = min(workers, len(futures))
                    live.update(render(active_workers=active_workers))

    total_time = time.perf_counter() - t0
    throughput = (total / total_time) * 3600 if total_time > 0 else 0.0
    avg_per_molecule = total_time / total if total else 0.0

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Total files", str(total))
    summary.add_row("Success", str(succeeded))
    summary.add_row("Failed", str(len(failures)))
    summary.add_row("Total time", _fmt_elapsed(total_time))
    summary.add_row("Avg per molecule", f"{avg_per_molecule:.1f}s")
    summary.add_row("Throughput", f"{throughput:.1f} jobs/hour")
    summary.add_row("Peak CPU", f"{peak_cpu:.1f}%")
    summary.add_row("Peak RAM", f"{peak_ram:.1f} MB")
    console.print(Panel(summary, title="Batch Completed", border_style="green"))

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
        check_dependencies()
        print("\n------------------------------------------------------------")
        print("      KNF-Core Interactive Mode")
        print("------------------------------------------------------------\n")
        
        while True:
            # Only ask for input path. No other prompts.
            input_path = input("Enter path to input file or folder (or 'q' to quit): ").strip()
            if input_path.lower() == 'q':
                sys.exit(0)
                
            input_path = input_path.strip('"').strip("'")
            
            if not os.path.exists(input_path):
                print(f"Error: Path '{input_path}' not found.")
                continue
            break
        
        # Define defaults for interactive mode
        class Args:
            charge = 0
            spin = 1
            force = True # "Just do this" implies run it.
            clean = True # "Just do this" often implies fresh run.
            debug = True # Helpful for user to see what's happening.
            output_dir = None
            processing = 'single'
            workers = None
            ram_per_job = 50.0
            refresh_autoconfig = False
            quiet_config = False
            
        args = Args()
        
        if os.path.isdir(input_path):
            mode = input("Processing mode [single/multi] (default: single): ").strip().lower()
            if mode in {'single', 'multi'}:
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
        choices=['single', 'multi'],
        default='single',
        help="Batch processing mode for directories (alias: --processes)"
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
    
    args = parser.parse_args()

    # Configure logging after CLI args are known.
    if args.debug:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

    check_dependencies()
    
    # If user provided flags, use them.
    # Default behavior for 'knf <file>' without flags is now determined by argparse defaults.
    
    if os.path.isdir(args.input_path):
        run_batch_directory(args.input_path, args)
    else:
        run_single_file(args.input_path, args)

if __name__ == "__main__":
    main()
