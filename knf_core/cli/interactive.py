from __future__ import annotations

import logging
import os
import sys

from rich.console import Console

from .. import first_run, utils
from ..engine.dependencies import probe_missing_dependencies
from ..engine.gpu import resolve_cpu_backend_when_torch_missing
from ..engine.types import RunOptions
from . import commands
from .dependency_report import print_missing_tools_warning
from .presentation.panels import brand_panel


def run_interactive() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    Console().print(brand_panel())
    print()

    while True:
        input_path = input("Enter path to input file or folder (or 'q' to quit): ").strip()
        if input_path.lower() == "q":
            sys.exit(0)

        input_path = input_path.strip('"').strip("'")
        input_path = utils.resolve_artifacted_path(input_path)

        if not os.path.exists(input_path):
            print(f"Error: Path '{input_path}' not found.")
            continue
        break

    nci_mode = input("Run mode [default/cpu/gpu/multiwfn] (default: default): ").strip().lower()
    if nci_mode not in {"", "default", "cpu", "gpu", "multiwfn"}:
        print(f"Unknown mode '{nci_mode}'. Using default.")
        nci_mode = "default"

    args = RunOptions(
        force=True,
        clean=True,
        debug=True,
        enable_stop_key=True,
        project_root=os.getcwd(),
    )

    if nci_mode == "cpu":
        args.cpu = True
        args.nci_device = "cpu"
    elif nci_mode == "gpu":
        args.gpu = True
        args.nci_backend = "torch"
        args.nci_device = "cuda"
    elif nci_mode == "multiwfn":
        args.nci_backend = "multiwfn"
        args.nci_device = "cpu"

    try:
        resolve_cpu_backend_when_torch_missing(args)
    except RuntimeError as exc:
        print(f"Backend setup failed: {exc}")
        sys.exit(1)

    first_ok = first_run.ensure_first_run_setup(require_multiwfn=(args.nci_backend == "multiwfn"))
    missing = probe_missing_dependencies(nci_backend=args.nci_backend, xtb_engine=args.xtb_engine)
    if missing:
        print_missing_tools_warning(missing)
    if not first_ok:
        print("First-time setup is incomplete. Please install missing tools and run again.")
        sys.exit(1)

    if os.path.isdir(input_path):
        mode = input("Processing mode [auto/single/multi] (default: auto): ").strip().lower()
        if mode in {"auto", "single", "multi"}:
            args.processing = mode
        commands.run_batch_directory(input_path, args)
    else:
        commands.run_single_file(input_path, args)

    print("\nDone.")
