import argparse
import sys
import os
import shutil
import logging
from .pipeline import KNFPipeline
from . import utils

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

def process_file(file_path: str, args):
    """Runs the pipeline for a single file."""
    try:
        # The user wants "Store eything in a folder with the name of the compound."
        # The pipeline class handles creation of results directory.
        # Currently it creates {filename}_knf.
        # We can leave it as is or modify pipeline if needed.
        # pipeline.py uses: self.results_dir = os.path.splitext(os.path.abspath(input_file))[0] + "_knf"
        # If input is 'test.sdf', result is 'test_knf'. This is reasonable.

        pipeline = KNFPipeline(
            input_file=file_path,
            charge=args.charge,
            spin=args.spin,
            force=args.force,
            clean=args.clean,
            debug=args.debug
        )
        pipeline.run()
    except Exception as e:
        if args.debug:
            logging.exception(f"Error processing {file_path}:")
        else:
            logging.error(f"Error processing {file_path}: {e}")

def run_batch_directory(directory: str, args):
    """Runs the pipeline for all valid files in a directory."""
    valid_exts = {'.xyz', '.sdf', '.mol', '.pdb', '.mol2'}
    
    files = [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if not files:
        print(f"No molecular files found in {directory}.")
        return

    print(f"Found {len(files)} files in {directory}. Processing sequentially...")
    
    for f in files:
        file_path = os.path.join(directory, f)
        print(f"\nProcessing: {f}")
        process_file(file_path, args)

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
            
        args = Args()
        
        if os.path.isdir(input_path):
            run_batch_directory(input_path, args)
        else:
            process_file(input_path, args)
            
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
    
    args = parser.parse_args()
    
    # If user provided flags, use them.
    # Default behavior for 'knf <file>' without flags is now determined by argparse defaults.
    
    if os.path.isdir(args.input_path):
        run_batch_directory(args.input_path, args)
    else:
        process_file(args.input_path, args)

if __name__ == "__main__":
    main()
