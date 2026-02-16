import argparse
import sys
import os
import shutil
import logging
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from .pipeline import KNFPipeline
from . import utils
from . import autoconfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

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
        return True, None
    except Exception as e:
        if args.debug:
            logging.exception(f"Error processing {file_path}:")
        else:
            logging.error(f"Error processing {file_path}: {e}")
        return False, str(e)

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
            if not args.quiet_config:
                autoconfig.print_config(cfg)
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
            if not args.quiet_config:
                print("")
                print("KNF-Core Auto Multi Configuration")
                print("---------------------------------")
                print("Source:          manual workers override")
                print(f"CPU:             unknown physical / {logical_threads} logical")
                print(f"Workers:         {workers}")
                print(f"OMP threads:     {omp} per process")
                print(f"Total threads:   {workers * omp} ({(min(workers * omp, logical_threads) / logical_threads) * 100:.0f}% of logical)")
                print("")
    else:
        workers = 1

    print(f"Found {len(files)} files in {directory}.")
    print(f"Processing mode: {mode} (workers={workers})")
    print(f"Results root: {results_root}")

    if mode == 'single' or len(files) == 1:
        queue = Queue()
        for path in files:
            queue.put(path)

        while not queue.empty():
            file_path = queue.get()
            print(f"\nProcessing: {os.path.basename(file_path)}")
            success, error = process_file(file_path, args, output_root=results_root)
            if not success:
                failures.append((file_path, error))
            queue.task_done()
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_file, file_path, args, results_root): file_path
                for file_path in files
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    success, error = future.result()
                except Exception as e:  # Defensive fallback for executor failures
                    success, error = False, str(e)

                status = "OK" if success else "FAILED"
                print(f"{status}: {os.path.basename(file_path)}")
                if not success:
                    failures.append((file_path, error))

    if failures:
        print(f"\nCompleted with {len(failures)} failed file(s):")
        for file_path, error in failures:
            print(f"- {os.path.basename(file_path)}: {error}")
    else:
        print("\nAll files processed successfully.")

def main():
    check_dependencies()

    # If no arguments provided -> Interactive mode (simplified)
    if len(sys.argv) == 1:
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
            results_root = resolve_results_root(input_path, args.output_dir)
            process_file(input_path, args, output_root=results_root)
            
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
    
    # If user provided flags, use them.
    # Default behavior for 'knf <file>' without flags is now determined by argparse defaults.
    
    if os.path.isdir(args.input_path):
        run_batch_directory(args.input_path, args)
    else:
        results_root = resolve_results_root(args.input_path, args.output_dir)
        process_file(args.input_path, args, output_root=results_root)

if __name__ == "__main__":
    main()
