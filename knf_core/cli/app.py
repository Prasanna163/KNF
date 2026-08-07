from __future__ import annotations

import logging
import os
import sys

import typer

from .. import first_run, utils
from ..engine.atlas import maybe_write_atlas_bundle, try_write_atlas_bundle_from_existing_outputs
from ..engine.dependencies import probe_missing_dependencies
from ..engine.discovery import _cleanup_submission_auxiliary_outputs
from ..engine.gpu import ensure_cuda_runtime_for_gpu_mode, resolve_cpu_backend_when_torch_missing
from ..engine.constants import CLI_NAME
from ..engine.batch_sources import _merge_master_and_batch_csv
from . import commands
from .argv_preprocess import normalize_argv
from .dependency_report import print_missing_tools_warning
from .options import apply_execution_shortcuts, build_run_options, validate_flag_combinations

NCI_HELP_PANEL = "NCI backend options"
XTB_HELP_PANEL = "xTB options"

app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command(context_settings={"help_option_names": ["-h", "--help"]})
def cli(
    input_path: str = typer.Argument(..., help="Path to input molecular file or directory"),
    charge: int = typer.Option(0, "--charge", help="Total system charge"),
    spin: int = typer.Option(1, "--spin", help="Total system spin multiplicity"),
    water: bool = typer.Option(
        False,
        "--water",
        help="Use xTB '--alpb water' for optimization and single-point instead of the default '--cosmo water'.",
        rich_help_panel=XTB_HELP_PANEL,
    ),
    hydration_fragment_mode: bool = typer.Option(
        False,
        "--hydration-fragment-mode",
        help=(
            "Group explicit H2O components into one water-cluster fragment B and all "
            "non-water components into solute fragment A. The NCI calculation remains volumetric 3D."
        ),
        rich_help_panel=NCI_HELP_PANEL,
    ),
    force: bool = typer.Option(False, "--force", help="Force recomputation"),
    clean: bool = typer.Option(False, "--clean", help="Clean results"),
    debug: bool = typer.Option(False, "--debug", help="Debug logging"),
    processing: str = typer.Option(
        "auto",
        "--processing",
        "--processes",
        help="Processing mode: auto (default), single, multi",
    ),
    multi: bool = typer.Option(False, "--multi", help="Shortcut for --processing multi"),
    single: bool = typer.Option(False, "--single", help="Shortcut for --processing single"),
    workers: int | None = typer.Option(
        None,
        "--workers",
        help="Optional override for worker threads. Default: auto-decide in multi mode",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        help="Custom output directory. Default: <input>/Results",
    ),
    batches: int | None = typer.Option(
        None,
        "--batches",
        help=(
            "Split directory inputs into evenly sized batches. Use '--batches N' "
            "to force N batches, or '--batches' for auto batch count."
        ),
    ),
    compile_existing: bool = typer.Option(
        False,
        "--compile-existing",
        help=(
            "Compile final batch CSV/JSON files from existing per-molecule result folders "
            "without running more molecules."
        ),
    ),
    universal_kuid: bool = typer.Option(
        False,
        "--universal-kuid",
        help=(
            "Recompute a universal KUID/KUID-Intensive calibration by combining "
            "existing batch_knf outputs discovered under the input directory."
        ),
    ),
    merge_master_csv: str | None = typer.Option(None, "--merge-master-csv", help="Path to the master batch CSV."),
    merge_new_csv: str | None = typer.Option(
        None,
        "--merge-new-csv",
        help="Path to the new batch CSV to append into the master set.",
    ),
    merge_output_dir: str | None = typer.Option(
        None,
        "--merge-output-dir",
        help="Output directory for merged universal-KUID outputs. Default: <master_csv_dir>/Combined Results",
    ),
    overwrite_master_csv: bool = typer.Option(
        False,
        "--overwrite-master-csv",
        help="Overwrite --merge-master-csv with the merged/recomputed CSV output.",
    ),
    ram_per_job: float = typer.Option(50.0, "--ram-per-job", help="Estimated RAM in MB per concurrent job for auto-config"),
    refresh_autoconfig: bool = typer.Option(
        False,
        "--refresh-autoconfig",
        help="Recompute and overwrite one-time auto-config cache",
    ),
    quiet_config: bool = typer.Option(False, "--quiet-config", help="Hide auto-configuration summary banner"),
    full_files: bool = typer.Option(
        False,
        "--full-files",
        help="Keep all intermediate and large files. Default behavior is storage-efficient cleanup.",
    ),
    enable_stop_key: bool = typer.Option(
        False,
        "--enable-stop-key",
        help="Enable graceful stop during batch runs by pressing 'q'.",
    ),
    interactive_quadrant_plot: bool = typer.Option(
        False,
        "--interactive-quadrant-plot",
        help="Open an interactive SNCI_Norm vs SCDI_Norm quadrant plot window after batch aggregation.",
    ),
    atlas_bundle: bool = typer.Option(
        False,
        "--atlas-bundle",
        help=(
            "Generate a canonical atlas submission bundle after KNF execution "
            "(submission_bundle/atlas_submission.csv + manifest.json)."
        ),
    ),
    gpu: bool = typer.Option(
        False,
        "--gpu",
        help=(
            "Smart GPU mode: run torch NCI on CUDA with adaptive packet routing "
            "(CUDA OOM auto-fallback to CPU for that molecule, then retry GPU next molecule)."
        ),
        rich_help_panel=NCI_HELP_PANEL,
    ),
    cpu: bool = typer.Option(
        False,
        "--cpu",
        help=(
            "Force CPU execution. Prefers torch CPU backend when torch is installed; "
            "otherwise auto-falls back to Multiwfn CPU backend."
        ),
        rich_help_panel=NCI_HELP_PANEL,
    ),
    multiwfn: bool = typer.Option(
        False,
        "--multiwfn",
        help="Use Multiwfn backend for NCI instead of default Torch backend",
        rich_help_panel=NCI_HELP_PANEL,
    ),
    nci_backend: str = typer.Option(
        "torch",
        "--nci-backend",
        help="NCI backend: torch or multiwfn.",
        rich_help_panel=NCI_HELP_PANEL,
    ),
    nci_grid_spacing: float = typer.Option(
        0.2,
        "--nci-grid-spacing",
        help="NCI grid spacing in Angstrom.",
        rich_help_panel=NCI_HELP_PANEL,
    ),
    nci_grid_padding: float = typer.Option(
        3.0,
        "--nci-grid-padding",
        help="NCI grid padding around the molecule in Angstrom.",
        rich_help_panel=NCI_HELP_PANEL,
    ),
    nci_device: str = typer.Option(
        "cpu",
        "--nci-device",
        help="Torch NCI device: cpu or cuda.",
        rich_help_panel=NCI_HELP_PANEL,
    ),
    nci_dtype: str = typer.Option(
        "float32",
        "--nci-dtype",
        help="Torch NCI numeric precision: float32 or float64.",
        rich_help_panel=NCI_HELP_PANEL,
    ),
    nci_batch_size: int = typer.Option(
        250000,
        "--nci-batch-size",
        help="Torch NCI grid-point batch size.",
        rich_help_panel=NCI_HELP_PANEL,
    ),
    nci_eig_batch_size: int = typer.Option(
        200000,
        "--nci-eig-batch-size",
        help="Torch NCI eigenvalue batch size.",
        rich_help_panel=NCI_HELP_PANEL,
    ),
    nci_rho_floor: float = typer.Option(
        1e-12,
        "--nci-rho-floor",
        help="Minimum electron-density floor used by the Torch NCI backend.",
        rich_help_panel=NCI_HELP_PANEL,
    ),
    nci_apply_primitive_norm: bool = typer.Option(
        False,
        "--nci-apply-primitive-norm",
        help="Apply primitive Gaussian normalization in the Torch NCI backend.",
        rich_help_panel=NCI_HELP_PANEL,
    ),
    scdi_var_min: float | None = typer.Option(None, "--scdi-var-min", help="Fixed global Var_min for SCDI normalization."),
    scdi_var_max: float | None = typer.Option(None, "--scdi-var-max", help="Fixed global Var_max for SCDI normalization."),
    wbo_mode: str = typer.Option(
        "xtb",
        "--wbo-mode",
        help=(
            "f3 source: xtb (production default; parsed interfragment Wiberg bond order) "
            "or native (experimental identity-overlap density-coupling estimate)."
        ),
    ),
    preopt: str = typer.Option(
        "geoinit",
        "--preopt",
        help="Pre-optimisation engine before xTB: 'geoinit' (default, basin-safe warm-start) or 'uff' (RDKit UFF fallback).",
        rich_help_panel=XTB_HELP_PANEL,
    ),
    xtb_engine: str = typer.Option(
        "xtbx",
        "--xtb-engine",
        help=(
            "xTB launcher for opt + single-point: 'xtbx' (default, unified native Windows CPU/GPU front-end), "
            "'xtb' (stock CPU build), or 'auto' (size-gate to xtbx at/above --xtb-gpu-atoms, otherwise stock xtb)."
        ),
        rich_help_panel=XTB_HELP_PANEL,
    ),
    xtb_gpu_atoms: int = typer.Option(
        350,
        "--xtb-gpu-atoms",
        help=(
            "Atom-count cutoff for '--xtb-engine auto': systems with at least this many atoms route to xtbx (GPU). "
            "Default 350."
        ),
        rich_help_panel=XTB_HELP_PANEL,
    ),
    sp: bool = typer.Option(
        False,
        "--sp",
        help=(
            "Strict single-point mode: preserve the supplied coordinates exactly, skip "
            "contact seeding, pre-optimisation, and geometry optimisation."
        ),
        rich_help_panel=XTB_HELP_PANEL,
    ),
    seed_contact: bool = typer.Option(
        False,
        "--seed-contact",
        help=(
            "Opt in to donor-acceptor contact seeding before xTB. This can translate "
            "fragment coordinates and is never implied by --sp."
        ),
        rich_help_panel=XTB_HELP_PANEL,
    ),
    refresh_first_run: bool = typer.Option(
        False,
        "--refresh-first-run",
        help="Re-run one-time first-run setup and overwrite its cached state",
    ),
    multiwfn_path: str | None = typer.Option(
        None,
        "--multiwfn-path",
        help="Path to Multiwfn executable or folder (saved for future runs)",
    ),
    knf: bool = typer.Option(False, "--knf", hidden=True),
):
    args = build_run_options(
        charge=charge,
        spin=spin,
        water=water,
        hydration_fragment_mode=hydration_fragment_mode,
        force=force,
        clean=clean,
        debug=debug,
        processing=processing,
        multi=multi,
        single=single,
        workers=workers,
        output_dir=output_dir,
        batches=batches,
        compile_existing=compile_existing,
        universal_kuid=universal_kuid,
        merge_master_csv=merge_master_csv,
        merge_new_csv=merge_new_csv,
        merge_output_dir=merge_output_dir,
        overwrite_master_csv=overwrite_master_csv,
        ram_per_job=ram_per_job,
        refresh_autoconfig=refresh_autoconfig,
        quiet_config=quiet_config,
        full_files=full_files,
        enable_stop_key=enable_stop_key,
        interactive_quadrant_plot=interactive_quadrant_plot,
        atlas_bundle=atlas_bundle,
        gpu=gpu,
        cpu=cpu,
        multiwfn=multiwfn,
        nci_backend=nci_backend,
        nci_grid_spacing=nci_grid_spacing,
        nci_grid_padding=nci_grid_padding,
        nci_device=nci_device,
        nci_dtype=nci_dtype,
        nci_batch_size=nci_batch_size,
        nci_eig_batch_size=nci_eig_batch_size,
        nci_rho_floor=nci_rho_floor,
        nci_apply_primitive_norm=nci_apply_primitive_norm,
        scdi_var_min=scdi_var_min,
        scdi_var_max=scdi_var_max,
        wbo_mode=wbo_mode,
        preopt=preopt,
        xtb_engine=xtb_engine,
        xtb_gpu_atoms=xtb_gpu_atoms,
        sp=sp,
        seed_contact=seed_contact,
        refresh_first_run=refresh_first_run,
        multiwfn_path=multiwfn_path,
        knf=knf,
        project_root=os.getcwd(),
    )
    error = validate_flag_combinations(args)
    if error:
        raise typer.BadParameter(error)
    apply_execution_shortcuts(args)

    input_path = utils.resolve_artifacted_path(input_path)
    # Atlas helpers still consume the historical argparse-style attribute.
    # Keep it off the dataclass contract while preserving that call surface.
    setattr(args, "input_path", input_path)

    if args.debug:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

    try:
        resolve_cpu_backend_when_torch_missing(args)
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc

    atlas_from_existing = None if args.compile_existing else try_write_atlas_bundle_from_existing_outputs(args)
    if atlas_from_existing:
        if args.atlas_bundle:
            cleanup_root = atlas_from_existing.get("source_results_root") or os.path.dirname(
                atlas_from_existing.get("bundle_dir", "")
            )
            if cleanup_root:
                removed = _cleanup_submission_auxiliary_outputs(cleanup_root, water=args.water)
                if removed:
                    print(f"Removed {len(removed)} auxiliary file(s) from {cleanup_root}.")
        print("\nAtlas bundle created from existing batch outputs (no recomputation).")
        print(f"Source CSV: {atlas_from_existing['source_csv']}")
        print(f"Bundle dir: {atlas_from_existing['bundle_dir']}")
        print(f"CSV:        {atlas_from_existing['csv_path']}")
        print(f"Manifest:   {atlas_from_existing['manifest_path']}")
        return

    merge_mode = bool(args.merge_master_csv and args.merge_new_csv)
    if not args.universal_kuid and not args.compile_existing and not merge_mode:
        first_ok = first_run.ensure_first_run_setup(
            force=args.refresh_first_run,
            multiwfn_path=args.multiwfn_path,
            require_multiwfn=(args.nci_backend == "multiwfn"),
        )
        missing = probe_missing_dependencies(
            multiwfn_path=args.multiwfn_path,
            nci_backend=args.nci_backend,
            xtb_engine=args.xtb_engine,
        )
        if missing:
            print_missing_tools_warning(missing)
        if (
            (args.nci_backend or "").strip().lower() == "torch"
            and (args.nci_device or "").strip().lower() == "cuda"
            and not bool(args.gpu)
        ):
            try:
                ensure_cuda_runtime_for_gpu_mode(allow_prompt=True)
            except RuntimeError as exc:
                logging.error("GPU setup check failed: %s", exc)
                raise typer.Exit(1) from exc
        if not first_ok:
            logging.error("First-time setup is incomplete. Install missing tools and retry.")
            raise typer.Exit(1)

    if merge_mode:
        merge_result = _merge_master_and_batch_csv(args.merge_master_csv, args.merge_new_csv, args)
        print(f"Merged results root: {merge_result['output_root']}")
        print(f"Combined Batch JSON: {merge_result['batch_json']}")
        print(f"Combined Batch CSV:  {merge_result['batch_csv']}")
        if merge_result.get("master_csv_updated"):
            print(f"Updated master CSV:  {merge_result['master_csv_updated']}")
    elif os.path.isdir(input_path):
        if args.compile_existing:
            commands.run_compile_existing_results(input_path, args)
        elif args.universal_kuid:
            commands.run_universal_kuid(input_path, args)
        elif args.batches is not None:
            commands.run_batch_directory_batched(input_path, args)
        else:
            commands.run_batch_directory(input_path, args)
    else:
        if args.compile_existing:
            raise typer.BadParameter("--compile-existing requires a directory input path.")
        if args.universal_kuid:
            raise typer.BadParameter("--universal-kuid requires a directory input path.")
        if args.batches is not None:
            raise typer.BadParameter("--batches requires a directory input path.")
        commands.run_single_file(input_path, args)

    bundle_info = maybe_write_atlas_bundle(args)
    if bundle_info:
        if args.atlas_bundle:
            cleanup_root = os.path.dirname(bundle_info.get("bundle_dir", ""))
            if cleanup_root:
                removed = _cleanup_submission_auxiliary_outputs(cleanup_root, water=args.water)
                if removed:
                    print(f"Removed {len(removed)} auxiliary file(s) from {cleanup_root}.")
        print("\nAtlas bundle created")
        print(f"Bundle dir: {bundle_info['bundle_dir']}")
        print(f"CSV:        {bundle_info['csv_path']}")
        print(f"Manifest:   {bundle_info['manifest_path']}")


def main(argv: list[str] | None = None, *, prog_name: str | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    try:
        app(args=normalize_argv(raw_args), prog_name=prog_name or CLI_NAME.lower(), standalone_mode=True)
    except SystemExit as exc:
        if exc.code in (0, None):
            return
        raise
