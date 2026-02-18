# KNF-Core Runtime Report: I/O, Sequencing, Parallelization, and Queueing

## Scope
This report describes the **current implementation** in this repository (not an idealized target design), with focus on:
- End-to-end execution order
- File and process I/O
- Parallelization strategy
- Queueing model

Primary code paths reviewed:
- `knf_core/main.py`
- `knf_core/pipeline.py`
- `knf_core/wrapper.py`
- `knf_core/multiwfn.py`
- `knf_core/converter.py`
- `knf_core/xtb.py`
- `knf_core/snci.py`
- `knf_core/scdi.py`
- `knf_core/knf_vector.py`
- `knf_core/autoconfig.py`
- `knf_core/first_run.py`
- `knf_core/nci_torch/*`

## 1. High-Level Runtime Architecture

### 1.1 Control Plane
`knf_core/main.py` is the top-level orchestrator.

It supports:
- Interactive mode (`knf` with no args)
- CLI mode (`knf <input_path> [flags]`)
- Single-file processing
- Batch directory processing

### 1.2 Worker Unit
Each molecule/file is handled by **one `KNFPipeline` instance** (`knf_core/pipeline.py`), which runs stages sequentially for that job.

### 1.3 NCI Engine Selection
Inside `KNFPipeline.run()`:
- `nci_backend == "multiwfn"` -> external Multiwfn process
- `nci_backend == "torch"` -> in-process PyTorch backend (`knf_core/nci_torch/pipeline.py` + `engine.py`)

Current default in CLI parser is `torch` on CPU unless flags override.

## 2. End-to-End Sequencing (Per Molecule)

`KNFPipeline.run()` executes this sequence:

1. Setup directories
2. Input normalization to XYZ (`converter.ensure_xyz`)
3. Geometry loading and fragment detection (`geometry.*`)
4. xTB optimization (`wrapper.run_xtb_opt`)
5. xTB single-point/property run (`wrapper.run_xtb_sp`)
6. NCI grid generation (Multiwfn or Torch)
7. SNCI + NCI stats (`snci.compute_*`)
8. SCDI (`scdi.compute_scdi`)
9. KNF vector assembly + write final outputs (`knf_vector.write_*`)
10. Optional storage cleanup (`_cleanup_storage_heavy_files`)

Important execution property:
- **Per-file pipeline is fully sequential**. No stage-level parallel fanout within one molecule job.

## 3. Detailed I/O Map

## 3.1 Directory Layout
For input `<name>.*` and root `<ResultsRoot>`:
- Work dir: `<ResultsRoot>/<name>/`
- Input staging dir: `<ResultsRoot>/<name>/input/`

If no custom output directory is provided:
- Single file -> sibling `Results/`
- Batch directory -> `<input_dir>/Results/`

## 3.2 Input Ingestion and Conversion
`converter.ensure_xyz(input_path, output_dir)`:
- If source is `.xyz`: copies/stages to `<work>/input/<name>.xyz`
- Else: runs `obabel` conversion and writes staged XYZ there

I/O characteristics:
- Read original molecular file
- Write staged XYZ in `input/`

## 3.3 xTB Stage I/O
`wrapper.run_xtb_opt(work_xyz, charge, uhf)`:
- Executes subprocess in `results_dir`
- Produces `xtbopt.xyz` (and typical xTB intermediates)

`wrapper.run_xtb_sp(xtbopt.xyz, charge, uhf)`:
- Executes subprocess in `results_dir`
- Stdout/stderr redirected to `xtb.log`
- Produces artifacts used later: `molden.input`, `wbo`, `.cosmo` files, etc.

I/O characteristics:
- Heavy external process I/O
- File-based handoff into next stages

## 3.4 NCI Stage I/O

### Multiwfn backend
`multiwfn.run_multiwfn(molden_file, results_dir)`:
- Writes `multiwfn.inp`
- Runs `Multiwfn <molden>` with script piped on stdin
- Writes `multiwfn.log`
- Expects `output.txt`, then pipeline renames/moves to `nci_grid.txt`

### Torch backend
`nci_torch.run_nci_torch(...)`:
- Reads `molden.input`
- Builds grid in-memory
- Computes rho / RDG / sign(lambda2)rho in-memory (CPU or CUDA)
- Writes binary grid to `nci_grid.npz` by default
- Optionally writes text grid `nci_grid.txt` when full-file retention is enabled

Torch export behavior:
- Converts tensors to CPU NumPy before file write
- Stores coordinate axes and field tensors in compressed NPZ
- Text rows (`x y z sign(lambda2)rho rdg`) are optional, not default

## 3.5 Descriptor Post-Processing I/O

`snci` accepts both text and NPZ grid payloads:
- Text path: full scan + parse (`nci_grid.txt`)
- Binary path: direct load from `nci_grid.npz`
- Used for SNCI and f6-f9 stats in both cases

`scdi.compute_scdi(<cosmo>)`:
- Parses COSMO segment information block
- Computes area-weighted variance-like metric

`knf_vector.write_output_txt(output.txt)` and `write_knf_json(knf.json)`:
- Writes final human and machine outputs

Batch mode aggregate outputs (`main.write_batch_aggregate_json`):
- `<ResultsRoot>/batch_knf.json`
- `<ResultsRoot>/batch_knf.csv`

## 3.6 Storage-Efficient Cleanup I/O
By default:
- Deletes many heavy intermediates from results directory
- Retains final summaries (e.g., `output.txt`, `knf.json`, batch aggregates)

To keep all intermediates, use `--full-files`.

## 4. Parallelization Model

## 4.1 Inter-File Parallelization (Batch)
In batch `multi` mode (`run_batch_directory`):
- Uses `ThreadPoolExecutor(max_workers=workers)`
- Submits one future per input file (`process_file`)
- Harvests completions via `as_completed(..., timeout=1)` in refresh loop

Concurrency granularity:
- **File-level parallelism only**

Each worker thread executes a full pipeline for one molecule, including blocking subprocess calls to xTB/Multiwfn.

## 4.2 Worker Count Selection
If user does not pass `--workers`:
- `autoconfig.resolve_multi_config(...)` determines worker count from:
  - Physical cores
  - Available RAM
  - Optional benchmark hints
  - Job count
- Applies BLAS/OMP env vars (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc.)

If user passes `--workers`:
- Manual worker count used
- OMP/BLAS thread caps derived as `logical_threads // workers`

## 4.3 Intra-File Parallelization
Per-file `KNFPipeline.run()` is sequential.

Within torch backend, parallelism is numerical and device-side:
- Basis evaluation and tensor math are vectorized in PyTorch
- Density computation uses chunk loop over grid points (`batch_size`)
- Hessian eigen decomposition uses batched chunks (`eig_batch_size`)
- On CUDA eig failure, code falls back to CPU eig for that chunk

There is **no multiprocessing queue** inside torch compute; chunking is a memory/control mechanism, not independent worker scheduling.

## 5. Queueing System (Current State)

## 5.1 Single-Path Queue (Serial)
In single-mode branch inside `run_batch_directory`:
- Creates in-memory `Queue()`
- Enqueues file paths
- Dequeues and processes one-by-one

This is a simple FIFO structure for serial processing; no parallel consumers.

## 5.2 Multi-Path Pending Set (Parallel)
In multi-mode branch:
- No `Queue` object is used
- Pending work is represented by a `dict` of `future -> file_path`
- Completion polling via `as_completed`

So practically, queueing is handled by `ThreadPoolExecutor` internal task queue + the futures map.

## 5.3 What Is Not Present
There is currently no:
- Persistent queue (disk-backed)
- Priority queue
- Retry queue/backoff policy
- Work-stealing layer beyond Python executor defaults
- Distributed scheduler

Failure handling is immediate per future; failures are collected and reported.

## 6. Synchronization and Data Handoffs

Main synchronization boundaries:
- Stage boundaries inside pipeline are strict and sequential.
- File outputs are used as next-stage inputs (e.g., `molden.input` -> NCI stage).
- In batch mode, each file pipeline is isolated by per-file results directories.

No shared mutable scientific data across worker jobs beyond environment variables and process-wide tool path settings.

## 7. Error Propagation and Recovery Behavior

- `process_file` wraps `pipeline.run()` in try/except and returns `(success, error, elapsed)`.
- Batch runner collects failures and prints final failure table.
- No automatic retries for failed files.
- Torch eig on CUDA has local fallback to CPU for failing chunks.
- Missing expected artifacts (e.g., NCI output not produced) raise runtime errors.

## 8. Performance-Relevant I/O/Compute Split

Given current design, dominant costs typically are:
- External tools (xTB/Multiwfn) subprocess runtime
- Torch export/write path for dense grids (NPZ by default; text when `--full-files` is used)

Torch path timings are already surfaced in metadata (`parse_molden`, `build_grid`, `compute_fields`, `export_grid`), enabling direct decomposition of compute vs write time.

## 9. Current Default UX/Execution Behavior (As Implemented)

- CLI default NCI backend is `torch` on CPU.
- Multiwfn is opt-in via `--multiwfn`.
- CUDA torch is opt-in via `--gpu`.
- Storage-efficient cleanup is default; full retention is opt-in via `--full-files`.
- Interactive mode asks for simplified run mode (`default/gpu/multiwfn`).

## 10. Practical Summary

Current system is a **hybrid sequential-per-job + parallel-across-jobs** architecture:
- Sequential pipeline stages per molecule
- ThreadPool-based concurrent molecules in batch mode
- Heavy file-based IPC across external scientific tools
- PyTorch internal vectorization/chunking for torch NCI backend
- Minimal queueing abstraction (FIFO queue in serial path; futures map in parallel path)

This keeps behavior deterministic and straightforward, but there is no advanced queue orchestration layer yet.
