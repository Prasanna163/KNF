import os
import shutil
import logging
from pathlib import Path
from rdkit import Chem

from . import utils, geometry, xtb, multiwfn, snci, scdi, knf_vector, converter, wrapper


def _final_output_name(filename: str, water: bool) -> str:
    if not water:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}_water{ext}"

class KNFPipeline:
    def __init__(
        self,
        input_file: str,
        charge: int = 0,
        spin: int = 1,
        water: bool = False,
        force: bool = False,
        clean: bool = False,
        debug: bool = False,
        output_root: str = None,
        keep_full_files: bool = False,
        storage_efficient=None,
        nci_backend: str = "torch",
        nci_grid_spacing: float = 0.2,
        nci_grid_padding: float = 3.0,
        nci_device: str = "auto",
        nci_dtype: str = "float32",
        nci_batch_size: int = 250000,
        nci_eig_batch_size: int = 200000,
        nci_rho_floor: float = 1e-12,
        nci_apply_primitive_norm: bool = False,
        scdi_var_min: float = None,
        scdi_var_max: float = None,
        wbo_mode: str = "native",
    ):
        self.input_file = utils.resolve_artifacted_path(input_file)
        self.charge = charge
        self.spin = spin
        self.water = bool(water)
        self.force = force
        self.clean = clean
        self.debug = debug
        if storage_efficient is not None:
            self.keep_full_files = not bool(storage_efficient)
        else:
            self.keep_full_files = bool(keep_full_files)
        self.nci_backend = (nci_backend or "torch").strip().lower()
        self.nci_grid_spacing = nci_grid_spacing
        self.nci_grid_padding = nci_grid_padding
        self.nci_device = nci_device
        self.nci_dtype = nci_dtype
        self.nci_batch_size = nci_batch_size
        self.nci_eig_batch_size = nci_eig_batch_size
        self.nci_rho_floor = nci_rho_floor
        self.nci_apply_primitive_norm = bool(nci_apply_primitive_norm)
        self.scdi_var_min = scdi_var_min
        self.scdi_var_max = scdi_var_max
        self.wbo_mode = (wbo_mode or "native").strip().lower()
        
        self.base_name = Path(self.input_file).stem
        default_output_root = os.path.join(os.path.dirname(self.input_file), "Results")
        self.output_root = os.path.abspath(output_root) if output_root else default_output_root
        self.work_dir = os.path.join(self.output_root, self.base_name)
        self.input_dir = os.path.join(self.work_dir, 'input')
        self.results_dir = self.work_dir

    def _cleanup_storage_heavy_files(self):
        """Deletes large intermediate files to reduce per-job storage."""
        heavy_names = [
            "nci_grid.txt",
            "nci_grid.npz",
            "nci_grid_data.txt",
            "xtb_esp.dat",
            "xtb_esp_profile.dat",
            "xtb_esp.cosmo",
            "xtb.cosmo",
            "xtbrestart",
            "molden.input",
            "wbo",
            "charges",
            "dislin.png",
            "multiwfn.inp",
            "xtb.log",
            "xtb_opt.log",
            "xtbopt.xyz",
            "input.xyz",
        ]
        removed = 0
        skipped = 0

        for name in heavy_names:
            path = os.path.join(self.results_dir, name)
            if not os.path.exists(path):
                continue
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                removed += 1
            except Exception as e:
                skipped += 1
                logging.warning(f"Storage cleanup skipped for {path}: {e}")

        for cosmo_path in Path(self.results_dir).glob("*.cosmo"):
            try:
                if cosmo_path.is_file():
                    cosmo_path.unlink()
                    removed += 1
            except Exception as e:
                skipped += 1
                logging.warning(f"Storage cleanup skipped for {cosmo_path}: {e}")

        if os.path.isdir(self.input_dir):
            try:
                shutil.rmtree(self.input_dir)
                removed += 1
            except Exception as e:
                skipped += 1
                logging.warning(f"Storage cleanup skipped for {self.input_dir}: {e}")

        logging.info(
            f"Storage-efficient cleanup complete for {self.base_name}: removed={removed}, skipped={skipped}"
        )

    def _stage(self, index: int, name: str):
        if self.debug:
            logging.info(f"[{index}/5] {name}")
        
    def setup_directories(self):
        """Creates directory structure."""
        utils.ensure_directory(self.output_root)

        if self.clean and os.path.exists(self.work_dir):
            logging.warning(f"Cleaning directory: {self.work_dir}")
            shutil.rmtree(self.work_dir)
            
        utils.ensure_directory(self.input_dir)
        utils.ensure_directory(self.results_dir)
        
    def run_pre_nci_stage(self) -> dict:
        """Runs geometry + xTB stages and returns context needed for NCI/finalization."""
        utils.setup_logging(self.debug)
        logging.info(f"Starting KNF-Core pipeline for {self.input_file}")

        self.setup_directories()

        self._stage(1, "Geometry")
        target_xyz = converter.ensure_xyz(self.input_file, self.input_dir)

        # ---- .xyz format warning ------------------------------------
        input_ext = os.path.splitext(self.input_file)[1].lower()
        if input_ext == '.xyz':
            logging.warning(
                "⚠️  Direct .xyz input detected. Be careful — .xyz files "
                "carry no bond/connectivity information, so bond perception "
                "is heuristic and may unintentionally alter bonds and affect "
                "final convergence.  For best results, prefer "
                ".mol / .mol2 / .sdf formats."
            )

        mol = geometry.load_molecule(target_xyz)
        fragments = geometry.detect_fragments(mol)

        # Promote intermolecular contact for 2-fragment donor/acceptor systems.
        if len(fragments) == 2:
            hb_seed = geometry.promote_hbond_interaction(mol, fragments[0], fragments[1])
            if hb_seed.get("applied"):
                logging.info(
                    "Applied H-bond interaction seeding: D=%s H=%s A=%s",
                    hb_seed.get("d_idx"),
                    hb_seed.get("h_idx"),
                    hb_seed.get("a_idx"),
                )
            else:
                logging.info("H-bond interaction seeding skipped: %s", hb_seed.get("reason"))
        pair_indices = None

        if len(fragments) == 1:
            logging.info("Detected single molecule.")
        elif len(fragments) == 2:
            logging.info("Detected two-fragment complex.")
        else:
            logging.info(f"Detected multi-fragment complex ({len(fragments)} fragments).")

        if len(fragments) == 2:
            f1 = geometry.compute_fragment_distance(mol, fragments[0], fragments[1])
            pair_indices = [0, 1]
        elif len(fragments) > 2:
            distances = []
            for i in range(len(fragments)):
                for j in range(i + 1, len(fragments)):
                    distances.append(geometry.compute_fragment_distance(mol, fragments[i], fragments[j]))
            f1 = float(sum(distances) / len(distances)) if distances else 0.0
            logging.info(
                "Using average COM distance across %s fragment pairs.",
                len(distances),
            )
        else:
            f1 = 0.0

        # f2 is finalized after xTB descriptor extraction, using weighted D-H...A triplets.
        f2_info = {
            "f2": float("nan"),
            "f2_defined": 0,
            "triplet_count": 0,
            "weight_sum": 0.0,
            "undefined_reason": "not_computed",
            "weight_model": "inv_ha_distance*(1+wbo_da)*(1+nci_local)",
            "top_triplets": [],
        }

        optimized_xyz = os.path.join(self.results_dir, 'xtbopt.xyz')
        work_xyz = os.path.join(self.results_dir, 'input.xyz')
        if not os.path.exists(work_xyz) or self.force:
            # Persist potentially re-oriented fragment geometry for downstream UFF/xTB.
            geometry.write_xyz(mol, work_xyz)

        if not os.path.exists(optimized_xyz) or self.force:
            uhf = self.spin - 1
            # ---- UFF pre-optimisation --------------------------------
            self._stage(2, "UFF Pre-optimisation")
            wrapper.run_uff_preopt(work_xyz)

            self._stage(3, "xTB Opt")
            wrapper.run_xtb_opt(work_xyz, self.charge, uhf, use_water=self.water)


        wbo_file = os.path.join(self.results_dir, 'wbo')
        molden_file = os.path.join(self.results_dir, 'molden.input')
        if not os.path.exists(wbo_file) or not os.path.exists(molden_file) or self.force:
            uhf = self.spin - 1
            self._stage(4, "xTB SP")
            wrapper.run_xtb_sp(optimized_xyz, self.charge, uhf, use_water=self.water)

        cosmo_files = sorted(f for f in os.listdir(self.results_dir) if f.endswith('.cosmo'))
        cosmo_file = None
        if "xtb.cosmo" in cosmo_files:
            cosmo_file = os.path.join(self.results_dir, "xtb.cosmo")
        elif cosmo_files:
            cosmo_file = os.path.join(self.results_dir, cosmo_files[0])
        elif self.water:
            logging.warning(
                "xTB ALPB water mode does not emit a .cosmo file; SCDI will be unavailable for %s.",
                self.base_name,
            )

        xtb_log = os.path.join(self.results_dir, 'xtb.log')
        try:
            xtb_data = xtb.parse_xtb_log(xtb_log)
            f4 = xtb_data.get('f4', 0.0)
            f5 = xtb_data.get('f5', 0.0)
            if self.wbo_mode == "native":
                wbo_native = xtb.compute_wbo_from_molden_details(
                    molden_file,
                    fragments=fragments,
                    use_identity_overlap=True,
                )
                f3 = wbo_native["max_inter_wbo"]
                wbo_max_global = wbo_native["max_wbo_global"]
            elif self.wbo_mode == "xtb":
                f3 = xtb.parse_interfragment_wbo(wbo_file, fragments, xtb_log_path=xtb_log)
                wbo_max_global = xtb.parse_max_wbo(wbo_file, xtb_log_path=xtb_log)
                wbo_native = {
                    "inter_pair_count": None,
                    "inter_max_pair": None,
                    "overlap_model": None,
                    "n_ao": None,
                }
            else:
                raise ValueError(f"Unsupported wbo_mode '{self.wbo_mode}'. Use 'native' or 'xtb'.")
        except Exception as e:
            logging.error(f"Failed to extract xTB descriptors: {e}")
            raise e

        if len(fragments) >= 2:
            f2_mol = Chem.Mol(mol)
            if os.path.exists(optimized_xyz):
                try:
                    optimized_mol = geometry.load_molecule(optimized_xyz)
                    if optimized_mol.GetNumAtoms() == mol.GetNumAtoms():
                        src_conf = optimized_mol.GetConformer()
                        dst_conf = f2_mol.GetConformer()
                        for atom_idx in range(mol.GetNumAtoms()):
                            p = src_conf.GetAtomPosition(atom_idx)
                            dst_conf.SetAtomPosition(atom_idx, (float(p.x), float(p.y), float(p.z)))
                    else:
                        logging.warning(
                            "Optimized geometry atom count (%s) does not match input (%s); "
                            "falling back to input geometry for f2.",
                            optimized_mol.GetNumAtoms(),
                            mol.GetNumAtoms(),
                        )
                except Exception as e:
                    logging.warning("Failed to load optimized geometry for f2 weighting: %s", e)

            wbo_pair_map = {}
            try:
                wbo_pair_map = xtb.parse_wbo_pair_map(wbo_file, xtb_log_path=xtb_log)
            except Exception as e:
                logging.warning("Failed to parse WBO pair map for f2 weighting; using geometry-only weights: %s", e)

            f2_info = geometry.compute_weighted_hbond_angle(
                mol=f2_mol,
                fragments=fragments,
                wbo_by_pair=wbo_pair_map,
                nci_strength_by_triplet=None,
            )
            if not f2_info.get("f2_defined"):
                logging.info(
                    "f2 undefined after weighted triplet evaluation (%s).",
                    f2_info.get("undefined_reason"),
                )
            else:
                logging.info(
                    "f2 weighted over %s triplets (weight_sum=%.6f).",
                    f2_info.get("triplet_count", 0),
                    float(f2_info.get("weight_sum", 0.0)),
                )
        else:
            f2_info = {
                "f2": float("nan"),
                "f2_defined": 0,
                "triplet_count": 0,
                "weight_sum": 0.0,
                "undefined_reason": "single_fragment",
                "weight_model": "inv_ha_distance*(1+wbo_da)*(1+nci_local)",
                "top_triplets": [],
            }

        f2 = float(f2_info.get("f2", float("nan")))

        return {
            "f1": f1,
            "f2": f2,
            "f2_defined": int(f2_info.get("f2_defined", 0)),
            "f2_triplet_count": int(f2_info.get("triplet_count", 0)),
            "f2_weight_sum": float(f2_info.get("weight_sum", 0.0)),
            "f2_undefined_reason": f2_info.get("undefined_reason"),
            "f2_weight_model": f2_info.get("weight_model"),
            "f2_top_triplets": f2_info.get("top_triplets", []),
            "f3": f3,
            "f4": f4,
            "f5": f5,
            "wbo_max_global": wbo_max_global,
            "wbo_inter_pair_count": wbo_native["inter_pair_count"],
            "wbo_inter_max_pair": wbo_native["inter_max_pair"],
            "wbo_overlap_model": wbo_native["overlap_model"],
            "wbo_native_n_ao": wbo_native["n_ao"],
            "wbo_mode": self.wbo_mode,
            "molden_file": molden_file,
            "cosmo_file": cosmo_file,
            "pair_indices": pair_indices,
            "fragment_count": len(fragments),
        }

    def run_post_nci_stage(self, context: dict):
        """Runs NCI + SNCI/SCDI + KNF assembly using a precomputed xTB context."""
        f1 = context["f1"]
        f2 = context["f2"]
        f2_defined = int(context.get("f2_defined", 0))
        f2_triplet_count = int(context.get("f2_triplet_count", 0))
        f2_weight_sum = float(context.get("f2_weight_sum", 0.0))
        f2_undefined_reason = context.get("f2_undefined_reason")
        f2_weight_model = context.get("f2_weight_model")
        f2_top_triplets = context.get("f2_top_triplets") or []
        f3 = context["f3"]
        f4 = context["f4"]
        f5 = context["f5"]
        wbo_max_global = context.get("wbo_max_global")
        wbo_inter_pair_count = context.get("wbo_inter_pair_count")
        wbo_inter_max_pair = context.get("wbo_inter_max_pair")
        wbo_overlap_model = context.get("wbo_overlap_model")
        wbo_native_n_ao = context.get("wbo_native_n_ao")
        wbo_mode = context.get("wbo_mode")
        molden_file = context["molden_file"]
        cosmo_file = context["cosmo_file"]
        pair_indices = context["pair_indices"]
        fragment_count = int(context["fragment_count"])

        nci_grid_file = os.path.join(self.results_dir, 'output.txt')
        final_grid_text_path = os.path.join(self.results_dir, 'nci_grid.txt')
        final_grid_binary_path = os.path.join(self.results_dir, 'nci_grid.npz')
        nci_data_path = final_grid_binary_path if self.nci_backend == "torch" else final_grid_text_path

        nci_success = False
        nci_engine_metadata = None

        if not os.path.exists(nci_data_path) or self.force:
            if self.nci_backend == "multiwfn":
                self._stage(5, "NCI (Multiwfn)")
                multiwfn.run_multiwfn(molden_file, self.results_dir)
                if os.path.exists(nci_grid_file):
                    os.replace(nci_grid_file, final_grid_text_path)
                    nci_success = True
                else:
                    raise RuntimeError("Multiwfn executed but did not produce expected output.")
            elif self.nci_backend == "torch":
                self._stage(5, "NCI (Torch Experimental)")
                from .nci_torch import run_nci_torch
                text_export_path = final_grid_text_path if self.keep_full_files else None
                nci_engine_metadata = run_nci_torch(
                    molden_path=molden_file,
                    output_path=final_grid_binary_path,
                    output_text_path=text_export_path,
                    spacing_angstrom=self.nci_grid_spacing,
                    padding_angstrom=self.nci_grid_padding,
                    device=self.nci_device,
                    dtype=self.nci_dtype,
                    batch_size=self.nci_batch_size,
                    eig_batch_size=self.nci_eig_batch_size,
                    rho_floor=self.nci_rho_floor,
                    output_units="bohr",
                    apply_primitive_normalization=self.nci_apply_primitive_norm,
                )
                nci_success = os.path.exists(final_grid_binary_path)
                if not nci_success:
                    raise RuntimeError("Torch NCI backend finished without producing nci_grid.npz.")
                logging.info(
                    "Torch NCI backend done: device=%s basis=%s grid=%s elapsed=%.2fs",
                    nci_engine_metadata.get("device"),
                    nci_engine_metadata.get("n_basis"),
                    nci_engine_metadata.get("grid_shape"),
                    nci_engine_metadata.get("elapsed_seconds", 0.0),
                )
            else:
                raise ValueError(
                    f"Unsupported nci_backend '{self.nci_backend}'. "
                    "Use 'multiwfn' or 'torch'."
                )
        else:
            nci_success = True

        f6 = f7 = f8 = f9 = 0.0
        snci_val = 0.0
        if nci_success and os.path.exists(nci_data_path):
            try:
                snci_val = snci.compute_snci(nci_data_path)
                nci_stats = snci.compute_nci_statistics(nci_data_path)
                f6 = nci_stats['f6']
                f7 = nci_stats['f7']
                f8 = nci_stats['f8']
                f9 = nci_stats['f9']
            except Exception as e:
                logging.error(f"SNCI computation failed: {e}")

        scdi_var = 0.0
        scdi_value = None
        if cosmo_file:
            try:
                scdi_metrics = scdi.compute_scdi_metrics(
                    cosmo_file,
                    var_min=self.scdi_var_min,
                    var_max=self.scdi_var_max,
                )
                scdi_var = scdi_metrics.variance
                scdi_value = scdi_metrics.scdi
            except Exception as e:
                logging.error(f"SCDI computation failed: {e}")

        vector = knf_vector.assemble_knf_vector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
        result = knf_vector.KNFResult(
            SNCI=snci_val,
            SCDI=scdi_value,
            SCDI_variance=scdi_var,
            KNF_vector=vector,
            metadata={
                'charge': self.charge,
                'spin': self.spin,
                'fragments': fragment_count,
                'geometry_fragment_pair': pair_indices,
                'f2_defined': f2_defined,
                'f2_triplet_count': f2_triplet_count,
                'f2_weight_sum': f2_weight_sum,
                'f2_undefined_reason': f2_undefined_reason,
                'f2_weight_model': f2_weight_model,
                'f2_top_triplets': f2_top_triplets,
                'wbo_max_global': wbo_max_global,
                'wbo_inter_pair_count': wbo_inter_pair_count,
                'wbo_inter_max_pair': wbo_inter_max_pair,
                'wbo_overlap_model': wbo_overlap_model,
                'wbo_native_n_ao': wbo_native_n_ao,
                'wbo_mode': wbo_mode,
                'xtb_water': self.water,
                'nci_backend': self.nci_backend,
                'nci_status': 'success' if nci_success else 'skipped',
                'nci_data_path': nci_data_path,
                'nci_engine_metadata': nci_engine_metadata,
            }
        )

        final_output_txt = os.path.join(self.results_dir, _final_output_name('output.txt', self.water))
        final_json = os.path.join(self.results_dir, _final_output_name('knf.json', self.water))
        knf_vector.write_output_txt(final_output_txt, result)
        knf_vector.write_knf_json(final_json, result)
        if self.water:
            delta_txt = os.path.join(self.results_dir, _final_output_name('delta.txt', self.water))
            delta_json = os.path.join(self.results_dir, _final_output_name('delta.json', self.water))
            reference_json = os.path.join(self.results_dir, 'knf.json')
            knf_vector.write_water_delta_outputs(
                delta_txt_path=delta_txt,
                delta_json_path=delta_json,
                water_result=result,
                reference_json_path=reference_json,
                water_json_path=final_json,
            )

        if not self.keep_full_files:
            self._cleanup_storage_heavy_files()

        logging.info("KNF-Core pipeline completed (potentially with warnings).")
        logging.info(f"Results saved to {self.results_dir}")

    def run(self):
        """Executes the full pipeline."""
        context = self.run_pre_nci_stage()
        self.run_post_nci_stage(context)
