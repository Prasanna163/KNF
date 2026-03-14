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
- Batch aggregate outputs: `batch_knf.json` and `batch_knf_unified_kuid_intensive.csv` (or `*_water.*` when `--water` is used).
- KUID-MVP output by default (single + batch): deterministic `KNF_vector (f1..f9) -> KUID` encoding with min-max calibration metadata.
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
- `--batches [N]` (directory mode only; split inputs into even batches, run each batch separately, then create `Combined Results` with recomputed universal KUID)
- `--universal-kuid` (directory mode only; discover existing batch outputs under the directory and recompute a universal combined KUID dataset)
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
knf ./molecules --batches 4
knf ./existing_runs --universal-kuid
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

With `--batches` (directory mode):
- per-batch outputs are written under `<results_root>/Batches/batch_XX/`
- a merged universal recalculation is written under `<results_root>/Combined Results/`

With `--universal-kuid` (directory mode):
- KNF scans subfolders for existing `batch_knf.json` / `batch_knf_unified_kuid_intensive.csv` (legacy `batch_knf.csv` is still accepted)
- writes a merged universal recalculation under `<results_root>/Combined Results/`

Final outputs:
- `knf.json`
- `output.txt`

Single-file runs also emit:
- `kuid_calibration.json` (in the `Results` root, alongside per-file result folders)

With `--water`, final outputs are suffixed for easier comparison:
- `knf_water.json`
- `output_water.txt`
- `delta_water.json`
- `delta_water.txt`

Batch root outputs:
- `batch_knf.json`
- `batch_knf_unified_kuid_intensive.csv`
- `kuid_calibration.json`
- `kuid_intensive_calibration.json`
- `kuid_prefix_index.json` (legacy-compatible alias of topology prefix index)
- `kuid_topology_prefix_index.json`
- `kuid_instance_prefix_index.json`
- `kuid_full_topology_bridge.json`
- `kuid_full_topology_bridge.csv`
- `kuid_reverse_index.json`
- `kuid_reverse_index.csv`
- `kuid_topology_reverse_index.json`
- `kuid_topology_reverse_index.csv`
- `kuid_intensive_family_distribution.csv`
- `kuid_intensive_family_distribution.png`
- `snci_scdi_quadrants.png`
- `snci_scdi_quadrants.json`

With `--water`, batch-level final outputs are similarly suffixed:
- `batch_knf_water.json`
- `batch_knf_unified_kuid_intensive_water.csv`
- `kuid_calibration_water.json`
- `batch_delta_water.json`
- `batch_delta_water.txt`
- `snci_scdi_quadrants_water.png`
- `snci_scdi_quadrants_water.json`

`batch_knf_unified_kuid_intensive.csv` includes normalized columns:
- `SNCI_Norm`
- `SCDI_Norm`
- `KUID_raw` (18 hex chars; `00-FF` per feature in canonical order `f1..f9`)
- `KUID` (18-char uppercase hex, no separators; same canonical order `f1..f9`)
- `KUID_Cluster` (display format `f1f2f3-f4f5-f6f7-f8f9`)
- `KUID_Intensive_raw` (5 hex chars; one nibble each for `f3,f4,f7,f8,f9`)
- `KUID_Intensive` (display format `X-X-X-X-X`)
- `KUID_Intensive_Cluster` (display format `f3f4f7-f8f9`)
- `KUID_prefix2` (`f3` only; covalency/WBO character)
- `KUID_prefix4` (`f3+f4`; adds dipole/electronic environment)
- `KUID_prefix6` (`f3+f4+f7`; adds mean electron density at NCI surface)
- `f2_defined` (`1` when weighted D-H...A angle is defined, `0` when `f2` is undefined/NaN)

When `f2` is undefined (`f2_defined = 0`), full 9D KUID still remains available: encoding uses an internal `f2` surrogate at the calibration upper bound (so the `f2` bin maps to `FF`) while preserving `f2_defined = 0`; `KUID-Intensive` remains available from `f3,f4,f7,f8,f9`.

`batch_knf_unified_kuid_intensive.csv` stores `SCDI_variance` (the scalar retained for SCDI tracking) and does not include the optional legacy `SCDI` column.

`knf.json` and `batch_knf.json` include both KUID representations by default:
- `kuid` (full KUID)
- `kuid_intensive` (KUID-Intensive)

## KUID Model

KNF uses a two-layer identifier model to separate chemistry topology from geometry-specific identity.

### 1) KUID-Intensive (Topology Passport)
- Feature set: `f3,f4,f7,f8,f9`
- Purpose: answer **"What kind of interaction is this?"**
- Behavior:
  - scale-robust and cross-dataset comparable
  - no `f2` dependency (so no undefined-angle failure mode)
  - stable for family-level grouping, atlas curation, and universal comparisons

### 2) KUID Full (Instance Address)
- Feature set: `f1..f9`
- Purpose: answer **"Which specific interaction instance is this?"**
- Behavior:
  - includes geometric context (`f1` distance, `f2` angle, etc.)
  - intentionally more specific and structure-dependent
  - suitable for exact-instance addressing and fine-grained lookup

### Why Both Exist
- Some complexes (for example dispersion or pi-stacking dominated cases) do not define a donor-H-acceptor angle naturally.
- In those rows, `f2_defined = 0`.
- Full KUID still remains available: encoder uses an internal surrogate bin for undefined `f2` (upper-bound mapping, often `FF`) while preserving the explicit `f2_defined` flag.
- KUID-Intensive remains NaN-free and directly comparable across all rows.

### Prefix Semantics (Current)
- `KUID_prefix2`: `f3` only
- `KUID_prefix4`: `f3+f4`
- `KUID_prefix6`: `f3+f4+f7`
- Full topology passport remains `KUID_Intensive_raw = f3+f4+f7+f8+f9`

### Recommended Use
- Topology clustering/search: use KUID-Intensive indexes.
- Instance-level retrieval: use full KUID indexes.
- Mixed workflows: use the bridge map (`kuid_full_topology_bridge.*`) to move between full instance IDs and topology passports.

Batch runs also emit KUID indexing/statistics artifacts:
- `kuid_family_stats.json`
- `kuid_family_stats.csv`
- `kuid_prefix_index.json` (legacy-compatible alias of topology prefix index)
- `kuid_topology_prefix_index.json` (topology passport index: `f3`, `f3+f4`, `f3+f4+f7`)
- `kuid_instance_prefix_index.json` (instance address index from full KUID prefixes)
- `kuid_full_topology_bridge.json` / `kuid_full_topology_bridge.csv` (maps full KUID -> topology passport)
- `kuid_reverse_index.json`
- `kuid_reverse_index.csv`
- `kuid_topology_reverse_index.json`
- `kuid_topology_reverse_index.csv`
- `kuid_intensive_family_distribution.csv`
- `kuid_intensive_family_distribution.png`

Programmatic KUID distance/search helpers are available in `knf_core/kuid_index.py` (for example byte-level Hamming distance and nearest-neighbor ranking).

Incremental batch resume:
- When running on a directory, if `batch_knf_unified_kuid_intensive.csv` already exists (legacy `batch_knf.csv` also supported) and `--force` is not used, KNF matches filenames against the existing batch CSV and computes only newly added files (already-listed files are skipped).

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
