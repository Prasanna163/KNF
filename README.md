# KNF-CORE

KNF-CORE is an automated computational chemistry pipeline for generating KNF descriptors from molecular structure files.

It combines xTB + Multiwfn + KNF post-processing to produce:
- SNCI
- SCDI variance
- 9D KNF vector (`f1` ... `f9`)

Current package version in this branch: `1.0.3`

## What Is Included

- Automatic input conversion to XYZ when needed (Open Babel)
- Single-file and directory processing
- Automatic multiprocessing recommendation + worker auto-config
- One-time first-run setup (dependency checks + multiprocessing suggestion)
- Multiwfn detection (auto + manual path registration)
- Optional experimental GPU NCI backend (`--nci-backend torch`)
- Dockerized runtime for CLI

Fragment handling:
- `1` fragment: `f1 = 0.0`, `f2 = 180.0`
- `2` fragments: `f1` = COM distance, `f2` = detected H-bond angle
- `>2` fragments: `f1` = average COM distance over unique pairs, `f2 = 180.0`

## Requirements

- Python `>=3.8`
- External tools in `PATH`:
  - `xtb`
  - `obabel` (Open Babel)
  - `Multiwfn`

Optional for experimental NCI backend:
- `torch` with CUDA support (or CPU fallback)

## Install

From source:

```bash
git clone https://github.com/Prasanna163/KNF.git
cd KNF
pip install -e .
```

From PyPI:

```bash
pip install KNF
```

## First-Run Setup

On first execution, KNF runs one-time setup that:
- checks external dependencies
- attempts automatic install for some tools when possible
- computes and prints a multiprocessing recommendation

One-time state is stored under:
- `~/.knf/first_run_state.json`

Force refresh:

```bash
knf <input_path> --refresh-first-run
```

## Multiwfn Detection and Manual Path Setup

KNF checks Multiwfn in this order:
- current `PATH`
- `KNF_MULTIWFN_PATH` env var
- saved path from `~/.knf/tool_paths.json`
- common local locations + shallow scan

Manual setup via CLI (saved for future runs):

```bash
knf <input_path> --multiwfn-path "E:\path\to\Multiwfn.exe"
```

You can also provide the folder containing `Multiwfn.exe`.

## CLI Usage

Basic run:

```bash
knf input_molecule.sdf
```

Useful options:
- `--charge <int>`
- `--spin <int>`
- `--force`
- `--clean`
- `--debug`
- `--processing <single|multi>` (alias: `--processes`)
- `--workers <int>`
- `--output-dir <path>`
- `--ram-per-job <MB>`
- `--refresh-autoconfig`
- `--storage-efficient`
- `--refresh-first-run`
- `--multiwfn-path <path>`
- `--nci-backend <multiwfn|torch>`
- `--nci-grid-spacing <float>`
- `--nci-grid-padding <float>`
- `--nci-device <auto|cuda|cpu>`
- `--nci-dtype <float32|float64>`
- `--nci-batch-size <int>`
- `--nci-rho-floor <float>`
- `--nci-apply-primitive-norm` (optional; default off, usually keep off for xTB Molden)

Examples:

```bash
knf example.mol --charge 0 --force
knf ./molecules --processing multi --force
knf ./molecules --processing multi --workers 4 --ram-per-job 200
knf example.mol --nci-backend torch --nci-device cuda --nci-grid-spacing 0.2
```

## Experimental Torch NCI Backend

Set `--nci-backend torch` to run the internal Molden parser + grid + density/derivatives/Hessian/eigen/RDG path instead of Multiwfn.

Current scope:
- Cartesians shells are supported for basis expansion.
- If a Molden file uses spherical `d/f/g` shells, the backend will stop with a clear error.

## Python API

```python
from knf_core.pipeline import KNFPipeline

pipeline = KNFPipeline(
    input_file="test.sdf",
    charge=0,
    spin=1,
)
pipeline.run()
```

## Docker

Build image:

```bash
docker build -t knf-core:latest .
```

Run CLI on a single file:

```bash
docker run --rm -v "$(pwd):/work" -w /work knf-core:latest example.mol --charge 0 --force
```

Use Compose:

```bash
docker compose up --build
```

Full container guide: `README.DOCKER.md`

## Output Layout

Default output root:
- file input: `<input_parent>/Results/<input_stem>/`
- directory input: `<input_dir>/Results/<file_stem>/`

Typical output files:
- `knf.json`
- `output.txt`
- `xtbopt.xyz`
- intermediates (`wbo`, `molden.input`, `nci_grid.txt`, etc.)

Batch mode writes:
- `batch_knf.json`
- `batch_knf.csv`

## Releasing

Release steps: `RELEASE.md`

## License

MIT (`LICENSE`)
