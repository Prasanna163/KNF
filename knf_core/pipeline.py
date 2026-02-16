import os
import shutil
import logging
from pathlib import Path

from . import utils, geometry, xtb, multiwfn, snci, scdi, knf_vector, converter, wrapper

class KNFPipeline:
    def __init__(
        self,
        input_file: str,
        charge: int = 0,
        spin: int = 1,
        force: bool = False,
        clean: bool = False,
        debug: bool = False,
        output_root: str = None,
    ):
        self.input_file = os.path.abspath(input_file)
        self.charge = charge
        self.spin = spin
        self.force = force
        self.clean = clean
        self.debug = debug
        
        self.base_name = Path(self.input_file).stem
        default_output_root = os.path.join(os.path.dirname(self.input_file), "Results")
        self.output_root = os.path.abspath(output_root) if output_root else default_output_root
        self.work_dir = os.path.join(self.output_root, self.base_name)
        self.input_dir = os.path.join(self.work_dir, 'input')
        self.results_dir = self.work_dir
        
    def setup_directories(self):
        """Creates directory structure."""
        utils.ensure_directory(self.output_root)

        if self.clean and os.path.exists(self.work_dir):
            logging.warning(f"Cleaning directory: {self.work_dir}")
            shutil.rmtree(self.work_dir)
            
        utils.ensure_directory(self.input_dir)
        utils.ensure_directory(self.results_dir)
        
    def run(self):
        """Executes the full pipeline."""
        utils.setup_logging(self.debug)
        logging.info(f"Starting KNF-Core pipeline for {self.input_file}")
        
        self.setup_directories()
        
        # 1. Prepare Input
        # Convert to XYZ using converter if needed, or just copy/ensure it's there.
        # The user wants: "automatically detects the file type and if its anything other than .xyz, it safely converts it to .xyz using obabel"
        # So we use converter.ensure_xyz.
        
        logging.info("Preparing input file...")
        target_xyz = converter.ensure_xyz(self.input_file, self.input_dir)
            
        # 2. Geometry & Fragment Detection
        # We load the confirmed XYZ file.
        # geometry.load_molecule now handles XYZ with bond perception.
        
        logging.info("Analyzing geometry...")
        mol = geometry.load_molecule(target_xyz)
        fragments = geometry.detect_fragments(mol)
        pair_indices = None
        
        if len(fragments) == 1:
            logging.info("Detected single molecule.")
        elif len(fragments) == 2:
            logging.info("Detected two-fragment complex.")
        else:
            logging.info(f"Detected multi-fragment complex ({len(fragments)} fragments).")
            
        # Compute Geometry Descriptors (f1, f2)
        # Check if we assume complex calculation even for single molecule?
        # Plan: "If 1 fragment -> treat as molecule... Single fragment but KNF requested -> geometry-only mode"
        # Wait, if 1 fragment, f1 (COM dist) is 0? undefined?
        # If single fragment, f1=0, f2=180?
        # The prompt says: "Single fragment but KNF requested -> geometry-only mode"
        # This implies we might skip electronic parts or handle them differently.
        # But for "1 fragment -> treat as molecule", xTB and descriptors can still be run (dipole, pol).
        # But f1 (inter-frag dist) and f2 (HB angle) and f3 (Max intermolecular WBO) don't make sense.
        # We will set them to default values: f1=0, f2=180, f3=0.
        
        if len(fragments) == 2:
            f1 = geometry.compute_fragment_distance(mol, fragments[0], fragments[1])
            f2 = geometry.detect_hb_angle(mol, fragments[0], fragments[1])
            pair_indices = [0, 1]
        elif len(fragments) > 2:
            distances = []
            for i in range(len(fragments)):
                for j in range(i + 1, len(fragments)):
                    distances.append(geometry.compute_fragment_distance(mol, fragments[i], fragments[j]))
            f1 = float(sum(distances) / len(distances)) if distances else 0.0
            f2 = 180.0
            logging.info(
                f"Using average COM distance across {len(distances)} fragment pairs; HB angle fixed at 180.0."
            )
        else:
            f1 = 0.0
            f2 = 180.0
            
        # 3. xTB Optimization
        # Output: optimized.xyz
        # If input is .xyz, we run on it.
        # We should use the standardized XYZ from RDKit if it was converted?
        # Let's trust xTB to handle the input file copied.
        
        optimized_xyz = os.path.join(self.results_dir, 'xtbopt.xyz')
        xtb_log_opt = os.path.join(self.results_dir, 'xtb_opt.log') # We need to handle log path
        
        # We run optimization in results_dir usually, or input_dir?
        # To make it resumable, we check if output exists.
        
        # Issue: `xtb.run_xtb_optimization` runs in the dir of the file.
        # So we should copy input to results_dir and run there? Or run in input_dir?
        # Better to run in `results_dir`.
        
        # Prepare working file for xTB
        # Copy target_xyz (from input_dir) to results_dir to run xTB there.
        # This ensures all xTB outputs (wbo, molden, etc.) stay in results_dir.
        
        work_xyz = os.path.join(self.results_dir, 'input.xyz')
        if os.path.abspath(target_xyz) != os.path.abspath(work_xyz):
            if not os.path.exists(work_xyz) or self.force:
                utils.safe_copy(target_xyz, work_xyz)
        
        if not os.path.exists(optimized_xyz) or self.force:
            # We run xTB on work_xyz
            # Uhf = spin - 1 
            uhf = self.spin - 1
            # But xTB expects --uhf N (number of unpaired electrons). 
            # If Multiplicity=1, unpaired=0. Correct.
            
            wrapper.run_xtb_opt(work_xyz, self.charge, uhf)
            # xTB produces 'xtbopt.xyz' in the same dir
            
        # 4. xTB Single Point
        # Check for wbo, molden, cosmo
        # Files: xtbopt.xyz (input), wbo, molden.input, ...
        # xTB outputs 'wbo', 'molden.input', '*cosmo'
        
        # Expected outputs
        wbo_file = os.path.join(self.results_dir, 'wbo')
        molden_file = os.path.join(self.results_dir, 'molden.input')
        # cosmo file name varies. Usually 'g98.out_cosmo' or similar if input was g98.
        # But here input is xtbopt.xyz. It typically generates 'xtbopt.xyz.cosmo' or similar.
        # We'll check for any .cosmo file in the dir.
        
        if not os.path.exists(wbo_file) or not os.path.exists(molden_file) or self.force:
            uhf = self.spin - 1
            wrapper.run_xtb_sp(optimized_xyz, self.charge, uhf)
            
        # Identify COSMO file
        cosmo_files = [f for f in os.listdir(self.results_dir) if f.endswith('.cosmo')]
        cosmo_file = os.path.join(self.results_dir, cosmo_files[0]) if cosmo_files else None
        
        # Extract xTB Descriptors (f3, f4, f5)
        xtb_log = os.path.join(self.results_dir, 'xtb.log')
        try:
            xtb_data = xtb.parse_xtb_log(xtb_log)
            f3 = xtb_data.get('f3', 0.0)
            f4 = xtb_data.get('f4', 0.0)
            f5 = xtb_data.get('f5', 0.0)
        except Exception as e:
            logging.error(f"Failed to extract xTB descriptors: {e}")
            raise e
            
        # 5. Multiwfn
        # Inputs: molden.input
        # Output: output.txt (the grid text file)
        # Note: The plan calls the final report 'output.txt' too.
        # We should distinguish them. Let's call grid file 'grid_data.txt'.
        # I need to modify `multiwfn.py` or rename the file after generation.
        # Multiwfn's output file depends on settings.
        # If we can't control it easily, we might collide.
        # BUT, the plan says: "Column 1/2/3... The file will be saved as output.txt"
        # AND "17. OUTPUT FILES ... output.txt Human-readable..."
        # This is a collision in the spec.
        # I will name the grid file `nci_grid.txt` to avoid collision with the final report.
        # I'll check if Multiwfn produced `output.txt` and rename it.
        
        # 5. Multiwfn
        # Inputs: molden.input
        # Output: output.txt (the grid text file)
        
        nci_grid_file = os.path.join(self.results_dir, 'output.txt') # Initial name
        final_grid_path = os.path.join(self.results_dir, 'nci_grid.txt')
        
        multiwfn_success = False
        
        if not os.path.exists(final_grid_path) or self.force:
             multiwfn.run_multiwfn(molden_file, self.results_dir)
             if os.path.exists(nci_grid_file):
                 os.rename(nci_grid_file, final_grid_path)
                 multiwfn_success = True
             else:
                 raise RuntimeError("Multiwfn executed but did not produce expected output.")
        else:
             multiwfn_success = True


        # 6. SNCI
        f6 = f7 = f8 = f9 = 0.0
        snci_val = 0.0
        
        if multiwfn_success and os.path.exists(final_grid_path):
            try:
                snci_val = snci.compute_snci(final_grid_path)
                nci_stats = snci.compute_nci_statistics(final_grid_path)
                f6 = nci_stats['f6']
                f7 = nci_stats['f7']
                f8 = nci_stats['f8']
                f9 = nci_stats['f9']
            except Exception as e:
                logging.error(f"SNCI computation failed: {e}")
        
        # 7. SCDI
        scdi_var = 0.0
        if cosmo_file:
            try:
                scdi_var = scdi.compute_scdi(cosmo_file)
            except Exception as e:
                logging.error(f"SCDI computation failed: {e}")
        
        # 8. Assemble KNF
        vector = knf_vector.assemble_knf_vector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
        
        result = knf_vector.KNFResult(
            SNCI=snci_val,
            SCDI_variance=scdi_var,
            KNF_vector=vector,
            metadata={
                'charge': self.charge,
                'spin': self.spin,
                'fragments': len(fragments),
                'geometry_fragment_pair': pair_indices,
                'multiwfn_status': 'success' if multiwfn_success else 'skipped'
            }
        )
        
        # 9. Write outputs
        final_output_txt = os.path.join(self.results_dir, 'output.txt')
        final_json = os.path.join(self.results_dir, 'knf.json')
        
        knf_vector.write_output_txt(final_output_txt, result)
        knf_vector.write_knf_json(final_json, result)
        
        logging.info("KNF-Core pipeline completed (potentially with warnings).")
        logging.info(f"Results saved to {self.results_dir}")
