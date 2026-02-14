import argparse
import sys
import os
import shutil
import logging
import subprocess

from .pipeline import KNFPipeline
from . import utils, converter, geometry, wrapper, multiwfn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def check_dependencies():
    """Checks if required external tools are available in PATH."""
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


def _build_work_paths(input_path: str):
    abs_input = os.path.abspath(input_path)
    base_name = os.path.splitext(os.path.basename(abs_input))[0]
    work_dir = os.path.join(os.path.dirname(abs_input), base_name)
    input_dir = os.path.join(work_dir, "input")
    results_dir = os.path.join(work_dir, "results")
    utils.ensure_directory(input_dir)
    utils.ensure_directory(results_dir)
    return abs_input, base_name, work_dir, input_dir, results_dir


def _prepare_interactive_input(input_path: str):
    abs_input, base_name, work_dir, input_dir, results_dir = _build_work_paths(input_path)
    target_xyz = converter.ensure_xyz(abs_input, input_dir)
    mol = geometry.load_molecule(target_xyz)
    fragments = geometry.detect_fragments(mol)
    atom_count = mol.GetNumAtoms()

    return {
        "input_path": abs_input,
        "base_name": base_name,
        "work_dir": work_dir,
        "input_dir": input_dir,
        "results_dir": results_dir,
        "target_xyz": target_xyz,
        "mol": mol,
        "fragments": fragments,
        "atom_count": atom_count,
    }


def _copy_xyz_to_results(target_xyz: str, results_dir: str):
    work_xyz = os.path.join(results_dir, "input.xyz")
    if os.path.abspath(target_xyz) != os.path.abspath(work_xyz):
        utils.safe_copy(target_xyz, work_xyz)
    return work_xyz


def _run_xtb_with_log(filepath: str, charge: int, uhf: int, flags: list, log_name: str):
    cwd = os.path.dirname(os.path.abspath(filepath))
    filename = os.path.basename(filepath)
    cmd = ['xtb', filename] + flags + ['--charge', str(charge), '--uhf', str(uhf)]
    log_path = os.path.join(cwd, log_name)

    with open(log_path, 'w') as log:
        subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)

    return cmd, log_path


def _run_nci_from_optimized(results_dir: str, optimized_xyz: str, charge: int, uhf: int):
    print("Launching xTB single-point for Multiwfn inputs...")
    wrapper.run_xtb_sp(optimized_xyz, charge, uhf)

    molden_file = os.path.join(results_dir, 'molden.input')
    if not os.path.exists(molden_file):
        raise FileNotFoundError(f"Expected Molden file not found: {molden_file}")

    print("Launching Multiwfn NCI Analysis...")
    multiwfn.run_multiwfn(molden_file, results_dir)

    raw_output = os.path.join(results_dir, 'output.txt')
    nci_grid = os.path.join(results_dir, 'nci_grid.txt')
    if os.path.exists(raw_output):
        os.replace(raw_output, nci_grid)
        print(f"Saved NCI grid data: {nci_grid}")
    else:
        print("Multiwfn finished. Did not detect output.txt for automatic renaming.")


def _post_optimization_menu(context: dict, charge: int, spin: int, optimized_xyz: str):
    results_dir = context["results_dir"]
    uhf = spin - 1

    while True:
        print("\nNEXT STEPS:")
        print("  [1] Continue to NCI Analysis (Multiwfn)")
        print("  [2] Perform another xTB calculation")
        print("  [3] View optimization geometry path")
        print("  [4] Export results and exit")
        choice = input("> ").strip()

        if choice == "1":
            try:
                _run_nci_from_optimized(results_dir, optimized_xyz, charge, uhf)
                print("NCI analysis finished.")
            except Exception as e:
                print(f"NCI step failed: {e}")
            continue
        if choice == "2":
            return
        if choice == "3":
            print(f"Optimized geometry: {optimized_xyz}")
            continue
        if choice == "4":
            print(f"Results directory: {results_dir}")
            return "exit"

        print("Invalid choice. Please select 1, 2, 3, or 4.")


def _interactive_xtb_explorer(context: dict):
    charge = 0
    spin = 1
    method = "GFN2-xTB"

    work_xyz = _copy_xyz_to_results(context["target_xyz"], context["results_dir"])
    results_dir = context["results_dir"]

    options = {
        "1": ("Single Point Energy", []),
        "2": ("Gradient Calculation", ["--grad"]),
        "3": ("Geometry Optimization", ["--opt"]),
        "4": ("Ionization Potential (IP)", ["--vipea"]),
        "5": ("Electron Affinity (EA)", ["--vipea"]),
        "6": ("IP + EA Combined", ["--vipea"]),
        "7": ("Electrophilicity Index", ["--vipea"]),
        "8": ("Fukui Indices", ["--vfukui"]),
        "9": ("Electrostatic Potential", ["--esp"]),
        "10": ("Population Analysis", ["--pop"]),
        "11": ("Bond Order Analysis", ["--wbo"]),
        "12": ("Orbital Localization", ["--lmo"]),
        "13": ("Molecular Dynamics", ["--md"]),
        "14": ("Conformer Search", ["--md"]),
        "15": ("Frequency Calculation", ["--hess"]),
    }

    while True:
        uhf = spin - 1
        molecule_name = os.path.basename(work_xyz)
        print("\nStep 3: xTB EXPLORER")
        print(f"Molecule: {molecule_name} | {method} | Charge: {charge} | UHF: {uhf}\n")
        print("BASIC CALCULATIONS")
        print("  [1] Single Point Energy")
        print("  [2] Gradient Calculation")
        print("  [3] Geometry Optimization")
        print("\nELECTRONIC PROPERTIES")
        print("  [4] Ionization Potential (IP)")
        print("  [5] Electron Affinity (EA)")
        print("  [6] IP + EA Combined")
        print("  [7] Electrophilicity Index")
        print("  [8] Fukui Indices")
        print("\nADVANCED ANALYSIS")
        print("  [9] Electrostatic Potential")
        print("  [10] Population Analysis")
        print("  [11] Bond Order Analysis")
        print("  [12] Orbital Localization")
        print("\nDYNAMICS")
        print("  [13] Molecular Dynamics")
        print("  [14] Conformer Search")
        print("  [15] Frequency Calculation")
        print("\nSOLVATION")
        print("  [16] Add Implicit Solvent (ALPB)")
        print("\nSETTINGS")
        print(f"  [S] Change method (current: {method})")
        print(f"  [C] Change charge/multiplicity (current: charge={charge}, multiplicity={spin})")
        print("  [H] Help")
        print("  [Q] Return to main menu")

        choice = input("> ").strip().lower()

        if choice == "q":
            return

        if choice == "h":
            print("Help: choose an option, inspect the generated xTB command, then confirm execution.")
            continue

        if choice == "s":
            print("Methods available in this build: GFN2-xTB (default).")
            print("Method switching menu is reserved for future extension.")
            continue

        if choice == "c":
            new_charge = input("Enter charge (integer): ").strip()
            new_spin = input("Enter multiplicity (integer): ").strip()
            try:
                charge = int(new_charge)
                spin = int(new_spin)
                if spin < 1:
                    raise ValueError("Multiplicity must be >= 1")
            except Exception:
                print("Invalid values. Keeping previous charge/multiplicity.")
            continue

        if choice == "16":
            solvent = input("Enter ALPB solvent name (example: water, acetone): ").strip()
            if not solvent:
                print("No solvent entered.")
                continue
            flags = ["--alpb", solvent]
            cmd = ['xtb', os.path.basename(work_xyz)] + flags + ['--charge', str(charge), '--uhf', str(uhf)]
            print("Prepared command:")
            print("  " + " ".join(cmd))
            run_now = input("Run this command now? [Y/n]: ").strip().lower()
            if run_now in ("", "y", "yes"):
                try:
                    _, log_path = _run_xtb_with_log(work_xyz, charge, uhf, flags, "xtb_alpb.log")
                    print(f"Completed. Log: {log_path}")
                except Exception as e:
                    print(f"xTB command failed: {e}")
            continue

        if choice not in options:
            print("Invalid option. Please choose from the menu.")
            continue

        label, flags = options[choice]
        cmd = ['xtb', os.path.basename(work_xyz)] + flags + ['--charge', str(charge), '--uhf', str(uhf)]

        print(f"\nSelected: {label}")
        print("Prepared command:")
        print("  " + " ".join(cmd))
        run_now = input("Run this command now? [Y/n]: ").strip().lower()
        if run_now not in ("", "y", "yes"):
            continue

        try:
            if choice == "3":
                print("\nStarting Geometry Optimization...")
                _, log_path = _run_xtb_with_log(work_xyz, charge, uhf, flags, "xtb_opt.log")
                optimized_xyz = os.path.join(results_dir, "xtbopt.xyz")
                if os.path.exists(optimized_xyz):
                    print("Optimization finished.")
                    print(f"Optimized geometry saved: {optimized_xyz}")
                    result = _post_optimization_menu(context, charge, spin, optimized_xyz)
                    if result == "exit":
                        return
                else:
                    print("Optimization completed but xtbopt.xyz was not found.")
                print(f"Log: {log_path}")
            else:
                log_name = f"xtb_option_{choice}.log"
                _, log_path = _run_xtb_with_log(work_xyz, charge, uhf, flags, log_name)
                print(f"Completed. Log: {log_path}")
        except Exception as e:
            print(f"xTB command failed: {e}")


def _interactive_main_menu():
    print("\n=== KNF-CORE v2.0 ===\n")

    while True:
        print("Step 1: INPUT MOLECULE")
        input_path = input("> Enter file path: ").strip().strip('"').strip("'")
        if input_path.lower() == 'q':
            return
        if not os.path.exists(input_path):
            print(f"Path not found: {input_path}\n")
            continue
        break

    try:
        context = _prepare_interactive_input(input_path)
    except Exception as e:
        print(f"Failed to prepare input: {e}")
        return

    fragments = context["fragments"]
    atom_count = context["atom_count"]
    print(f"\nLoaded: {os.path.basename(context['input_path'])} ({atom_count} atoms, charge=0, multiplicity=1)")
    print(f"Detected: {len(fragments)} fragment(s)")

    while True:
        print("\nStep 2: SELECT MODE")
        print("  [1] Automated Pipeline (Full KNF descriptor generation)")
        print("  [2] Interactive xTB Explorer")
        print("  [3] Custom Workflow Builder")
        mode = input("> ").strip()

        if mode == "1":
            class Args:
                charge = 0
                spin = 1
                force = True
                clean = True
                debug = True

            process_file(context["input_path"], Args())
            return

        if mode == "2":
            _interactive_xtb_explorer(context)
            return

        if mode == "3":
            print("Custom Workflow Builder is currently routed to xTB Explorer in this version.")
            _interactive_xtb_explorer(context)
            return

        print("Invalid choice. Please choose 1, 2, or 3.")


def main():
    check_dependencies()

    # No arguments: interactive guided mode
    if len(sys.argv) == 1:
        _interactive_main_menu()
        print("\nDone.")
        return

    # Batch/CLI Mode
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

    if os.path.isdir(args.input_path):
        run_batch_directory(args.input_path, args)
    else:
        process_file(args.input_path, args)


if __name__ == "__main__":
    main()
