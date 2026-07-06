from __future__ import annotations

from dataclasses import fields

from ..engine.types import RunOptions

_RUN_OPTION_FIELDS = {field.name for field in fields(RunOptions)}


def build_run_options(**values) -> RunOptions:
    payload = {key: value for key, value in values.items() if key in _RUN_OPTION_FIELDS}
    return RunOptions(**payload)


def validate_flag_combinations(options: RunOptions) -> str | None:
    if options.multi and options.single:
        return "Use only one of --multi or --single."
    if options.gpu and options.multiwfn:
        return "Use only one of --gpu or --multiwfn."
    if options.gpu and options.cpu:
        return "Use only one of --gpu or --cpu."
    if options.cpu and options.multiwfn:
        return "Use only one of --cpu or --multiwfn."
    if options.processing not in {"auto", "single", "multi"}:
        return "--processing must be one of: auto, single, multi."
    if options.batches is not None and options.batches < 0:
        return "--batches must be a positive integer, or provided without a value for auto mode."
    if options.batches is not None and options.universal_kuid:
        return "Use either --batches or --universal-kuid, not both in the same command."
    if bool(options.merge_master_csv) ^ bool(options.merge_new_csv):
        return "Use both --merge-master-csv and --merge-new-csv together."
    if (options.batches is not None or options.universal_kuid) and (
        options.merge_master_csv or options.merge_new_csv
    ):
        return "--merge-master-csv/--merge-new-csv cannot be combined with --batches or --universal-kuid."
    if options.nci_backend not in {"torch", "multiwfn"}:
        return "--nci-backend must be one of: torch, multiwfn."
    if options.nci_dtype not in {"float32", "float64"}:
        return "--nci-dtype must be one of: float32, float64."
    if options.wbo_mode not in {"native", "xtb"}:
        return "--wbo-mode must be one of: native, xtb."
    if options.preopt not in {"uff", "geoinit"}:
        return "--preopt must be one of: uff, geoinit."
    if options.xtb_engine not in {"xtb", "xtbx", "auto"}:
        return "--xtb-engine must be one of: xtb, xtbx, auto."
    return None


def apply_execution_shortcuts(options: RunOptions) -> RunOptions:
    if options.multi:
        options.processing = "multi"
    elif options.single:
        options.processing = "single"

    if options.multiwfn:
        options.nci_backend = "multiwfn"
        options.nci_device = "cpu"
    elif options.gpu:
        options.nci_backend = "torch"
        options.nci_device = "cuda"
        options.nci_dtype = "float32"
    elif options.cpu:
        options.nci_backend = "torch"
        options.nci_device = "cpu"
    return options
