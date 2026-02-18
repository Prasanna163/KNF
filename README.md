# KNF-CORE (KNF-GPU Branch)

KNF-CORE is an automated computational chemistry pipeline that generates:
- SNCI
- SCDI variance
- 9D KNF vector (`f1` to `f9`)

from molecular structure files using xTB + NCI backend + KNF post-processing.

Current package version in this branch: `1.0.3`

## Branch Highlights

This `KNF-GPU` branch includes:
- Torch-based NCI backend (`--nci-backend torch`) with CPU/CUDA execution.
- Multiwfn backend still supported (`--nci-backend multiwfn`).
- GPU overlap scheduler in batch mode (CPU pre-NCI + single GPU post-NCI lane) when using `torch + cuda`.
- Storage-efficient default output behavior (intermediates removed by default, keep with `--full-files`).
- Robust filename/path artifact handling for mojibake/Unicode path variants.
- xTB optimization capped to 50 cycles (`--cycles 50`) and pipeline continues if `xtbopt.xyz` exists.
- Batch aggregate outputs: `batch_knf.json` and `batch_knf.csv`.

## Fragment Handling

- `1` fragment: `f1 = 0.0`, `f2 = 180.0`
- `2` fragments: `f1` = COM distance, `f2` = detected H-bond angle
- `>2` fragments: `f1` = average COM distance over unique pairs, `f2 = 180.0`

## Requirements

- Python `>=3.8`
- External tools in `PATH`:
  - `xtb`
  - `obabel` (Open Babel)
- `Multiwfn` is required only when using `--nci-backend multiwfn`

Optional:
- `torch` (for Torch NCI backend; CUDA optional)

## Install

From source:

```bash
git clone https://github.com/Prasanna163/KNF.git
cd KNF
pip install -e .
```

Install with Torch extra:

```bash
pip install -e ".[torch-nci]"
```

From PyPI:

```bash
pip install KNF
```

## First-Run Setup

On first execution, KNF runs one-time setup that:
- checks external dependencies
- attempts automatic install for some tools (when available)
- computes multiprocessing recommendation

State file:
- `~/.knf/first_run_state.json`

Force refresh:

```bash
knf <input_path> --refresh-first-run
```

## Multiwfn Detection and Path Registration

Search order:
- current `PATH`
- `KNF_MULTIWFN_PATH` env var
- saved path `~/.knf/tool_paths.json`
- common local locations + shallow scan

Manual registration:

```bash
knf <input_path> --multiwfn-path "E:\path\to\Multiwfn.exe"
```

You can also pass a folder containing `Multiwfn.exe`.

## CLI Usage

Basic:

```bash
knf input_molecule.sdf
```

### Core options

- `--charge <int>`
- `--spin <int>`
- `--force`
- `--clean`
- `--debug`
- `--processing <auto|single|multi>`
- `--multi` / `--single` (shortcuts)
- `--workers <int>`
- `--output-dir <path>`
- `--ram-per-job <MB>`
- `--refresh-autoconfig`
- `--quiet-config`
- `--full-files`
- `--refresh-first-run`
- `--multiwfn-path <path>`

### Backend options

- `--gpu` shortcut: sets `torch + cuda + float64`
- `--multiwfn` shortcut: sets `multiwfn + auto`
- `--nci-backend <torch|multiwfn>`

### Advanced NCI options (hidden in default `--help`, but supported)

- `--nci-grid-spacing <float>`
- `--nci-grid-padding <float>`
- `--nci-device <cpu|cuda|auto>`
- `--nci-dtype <float32|float64>`
- `--nci-batch-size <int>`
- `--nci-eig-batch-size <int>`
- `--nci-rho-floor <float>`
- `--nci-apply-primitive-norm`

### Examples

```bash
knf example.mol --force
knf ./molecules --processing multi --workers 4 --ram-per-job 200
knf example.mol --nci-backend torch --nci-device cuda --nci-dtype float64
knf example.mol --gpu
knf example.mol --multiwfn
```

## Torch NCI Backend Notes

- Uses internal Molden parser + grid + RDG pipeline.
- Supports Cartesian shells for basis expansion.
- Spherical `d/f/g` shell Molden inputs are currently not supported.
- SNCI/statistics can be computed from either text grid (`.txt`) or compressed grid (`.npz`).

## Output Layout

Default output root:
- file input: `<input_parent>/Results/<input_stem>/`
- directory input: `<input_dir>/Results/<file_stem>/`

Final outputs:
- `knf.json`
- `output.txt`

Batch root outputs:
- `batch_knf.json`
- `batch_knf.csv`

When `--full-files` is used, intermediate artifacts are retained (for example NCI grid artifacts and xTB/Multiwfn intermediates). Without it, storage-efficient cleanup runs automatically.

## Compare Script

Use `scripts/compare_nci.py` to compare Multiwfn and Torch NCI outputs/correlation and timing behavior.

## Docker

Build:

```bash
docker build -t knf-core:latest .
```

Run:

```bash
docker run --rm -v "$(pwd):/work" -w /work knf-core:latest example.mol --charge 0 --force
```

Compose:

```bash
docker compose up --build
```

See `README.DOCKER.md` for full details.

## Releasing

Release steps are documented in `RELEASE.md`.

## License

MIT (`LICENSE`)
