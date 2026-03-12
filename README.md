# KNF-CORE (KNF-GPU Branch)

KNF-CORE is an automated computational chemistry pipeline that generates:
- SNCI
- SCDI (normalized; when fixed bounds are provided)
- SCDI variance (raw VarA)
- 9D KNF vector (`f1` to `f9`)

from molecular structure files using xTB + NCI backend + KNF post-processing.

Current package version in this branch: `1.0.5`

## Branch Highlights

This `KNF-GPU` branch includes:
- Torch-based NCI backend (`--nci-backend torch`) with CPU/CUDA execution.
- Multiwfn backend still supported (`--nci-backend multiwfn`).
- GPU overlap scheduler in batch mode (CPU pre-NCI + single GPU post-NCI lane) when using `torch + cuda`.
- Storage-efficient default output behavior (intermediates removed by default, keep with `--full-files`).
- Robust filename/path artifact handling for mojibake/Unicode path variants.
- xTB optimization capped to 50 cycles (`--cycles 50`) and pipeline continues if `xtbopt.xyz` exists.
- Batch aggregate outputs: `batch_knf.json` and `batch_knf.csv` (or `*_water.*` when `--water` is used).
- Optional graceful mid-run stop in batch mode (`--enable-stop-key`, press `q`).
- Batch normalized/quadrant outputs: `SNCI_Norm`, `SCDI_Norm`, quadrant PNG + JSON.
- Native Molden-based WBO is now the default (`--wbo-mode native`) for `f3`.

## Fragment Handling

- `1` fragment: `f1 = 0.0`; `f2` is undefined (`NaN`) with `f2_defined = 0`.
- `>=2` fragments:
  - `f1` = COM distance for 2 fragments, or average COM distance over unique fragment pairs for multi-fragment systems.
  - `f2` is a weighted D-H...A angle over all candidate cross-fragment donor-H-acceptor triplets:
    - `f2 = sum_j(w_j * theta_j) / sum_j(w_j)`
    - default weight model: inverse H...A distance, reweighted by interfragment donor-acceptor WBO when available.
  - if no meaningful triplets exist, `f2` remains undefined (`NaN`) and `f2_defined = 0` (no fake `180.0` fallback).

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
- `--water` (switch xTB opt/SP from default `--cosmo water` to `--alpb water`; ALPB mode does not produce `.cosmo`, so SCDI is unavailable)
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
- `--enable-stop-key` (press `q` during batch processing to stop new jobs safely)
- `--interactive-quadrant-plot`
- `--refresh-first-run`
- `--multiwfn-path <path>`
- `--scdi-var-min <float>`
- `--scdi-var-max <float>`
- `--wbo-mode <native|xtb>` (default: `native`)

SCDI normalization can also be provided globally via:
- `KNF_SCDI_VAR_MIN`
- `KNF_SCDI_VAR_MAX`

If bounds are not provided, KNF still computes and reports raw `SCDI_variance` (VarA), and `SCDI` is emitted as `null`/`n/a`.

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
knf example.mol --water
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

With `--water`, final outputs are suffixed for easier comparison:
- `knf_water.json`
- `output_water.txt`
- `delta_water.json`
- `delta_water.txt`

Batch root outputs:
- `batch_knf.json`
- `batch_knf.csv`
- `snci_scdi_quadrants.png`
- `snci_scdi_quadrants.json`

With `--water`, batch-level final outputs are similarly suffixed:
- `batch_knf_water.json`
- `batch_knf_water.csv`
- `batch_delta_water.json`
- `batch_delta_water.txt`
- `snci_scdi_quadrants_water.png`
- `snci_scdi_quadrants_water.json`

`batch_knf.csv` includes normalized columns:
- `SNCI_Norm`
- `SCDI_Norm`
- `f2_defined` (`1` when weighted D-H...A angle is defined, `0` when `f2` is undefined/NaN)

When `--full-files` is used, intermediate artifacts are retained (for example NCI grid artifacts and xTB/Multiwfn intermediates). Without it, storage-efficient cleanup runs automatically.

## Compare Script

Use `scripts/compare_nci.py` to compare Multiwfn and Torch NCI outputs/correlation and timing behavior.

## Docker

Build:

```bash
docker build -t knf-core:1.0.5 -t knf-core:latest .
```

Run:

```bash
docker run --rm -v "$(pwd):/work" -w /work knf-core:1.0.5 example.mol --charge 0 --force
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
