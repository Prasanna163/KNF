<img width="3000" height="3000" alt="NCIForge2_trans" src="https://github.com/user-attachments/assets/8b2a3809-7275-4aa2-8a7c-c3c1987aef84" />

# NCIForge

NCIForge is a physics-informed descriptor pipeline for non-covalent interactions.
It computes KNF descriptors, SNCI/SCDI metrics, and KUID-family indexing artifacts
from molecular structure files using xTB + NCI backend + KNF post-processing.

Current package version: `1.0.8`  
Release milestone tag: `v1`

## What It Covers

- Generate KNF feature vectors (`f1..f9`) for single files and batches
- Generate full `KUID` identifiers (instance-level)
- Generate `KUID-Intensive` identifiers (topology-family level)
- Build reusable lookup artifacts (prefix, reverse, and bridge indexes)
- Recompute universal KUID outputs from existing batch folders (`--universal-kuid`)
- Generate canonical atlas bundle files (`--atlas-bundle`)

## Requirements

- Python `>=3.8`
- External tools in `PATH`:
  - `xtb`
  - `obabel`
- `Multiwfn` only when using `--nci-backend multiwfn`

Optional:

- `torch` for Torch NCI backend (CPU or CUDA)

## Install

Fastest local install (from this repo):

Windows PowerShell:

```powershell
./scripts/install_nciforge.ps1
```

macOS/Linux:

```bash
bash ./scripts/install_nciforge.sh
```

This opens an interactive CLI wizard that asks:

- install scope: `global` or `local` virtual environment
- PyTorch mode: `cpu`, `gpu`, or `skip`
- whether to auto-setup external dependencies (`xtb`, `obabel`, etc.)

Manual source install:

```bash
git clone https://github.com/Prasanna163/KNF.git
cd KNF
pip install -e .
```

Install with Torch extra:

```bash
pip install -e ".[torch-nci]"
```

## CLI Commands

Primary CLI:

- `nciforge`

Compatibility alias (still supported):

- `knf`

## Example Runs

Single molecule:

```bash
nciforge example.mol
```

Batch run:

```bash
nciforge ./molecules --processing multi --workers 4 --force
```

GPU run (Torch CUDA):

```bash
nciforge ./molecules --gpu
```

`--gpu` enables smart GPU routing by default:

- Runs Torch NCI on CUDA first.
- If a molecule hits CUDA OOM, that molecule is re-routed to CPU automatically.
- The next molecule retries GPU automatically.
- Uses memory-friendlier `float32` by default.

CPU run (no GPU):

```bash
nciforge ./molecules --cpu
```

`--cpu` forces CPU execution. If PyTorch is unavailable, KNF auto-falls back to
the Multiwfn CPU backend (when Multiwfn is installed).

Split into batches:

```bash
nciforge ./molecules --batches 4
```

Recompute universal KUID calibration from existing batch outputs:

```bash
nciforge ./existing_runs --universal-kuid
```

Generate canonical atlas submission bundle:

```bash
nciforge ./molecules --atlas-bundle
```

## Outputs

Single-run outputs include:

- `knf.json` (contains `kuid` and `kuid_intensive` sections)
- `kuid_calibration.json`

Batch root outputs include:

- `batch_knf.json`
- `batch_knf_unified.csv`
- `kuid_calibration.json`
- `kuid_intensive_calibration.json`
- `kuid_prefix_index.json`
- `kuid_topology_prefix_index.json`
- `kuid_instance_prefix_index.json`
- `kuid_full_topology_bridge.json`
- `kuid_full_topology_bridge.csv`
- `kuid_reverse_index.json`
- `kuid_reverse_index.csv`
- `kuid_topology_reverse_index.json`
- `kuid_topology_reverse_index.csv`
- `kuid_family_stats.json`
- `kuid_family_stats.csv`
- `kuid_intensive_family_distribution.csv`
- `kuid_intensive_family_distribution.png`

With `--water`, water-suffixed variants are emitted (for example `*_water.json`, `*_water.csv`).

## Atlas Submission Bundle

When `--atlas-bundle` is supplied, NCIForge writes:

- `submission_bundle/atlas_submission.csv`
- `submission_bundle/manifest.json`

If prior batch outputs already exist, running `--atlas-bundle` can parse those
existing CSV outputs and generate the bundle without recomputing the KNF pipeline.
Legacy batch CSV names are still supported for upgrade compatibility.

After bundle creation, submission runs clean up auxiliary analysis/index artifacts
in the same results root to keep exports lightweight.

`atlas_submission.csv` includes:

- `molecule_name`, `charge`, `spin`
- `f1..f9`, `SNCI`, `SCDI`, `SCDI_variance`
- `backend`, `device`, `xtb_version`, `knf_core_version`
- `nci_grid_spacing`, `nci_grid_padding`, `water_mode`
- `KUID_raw`, `KUID_Cluster`
- `KUID_Intensive_raw`, `KUID_Intensive_Cluster`
- `instance_hash` (`sha256(f1..f9,charge,spin,xtb_version,nci_grid_spacing,nci_grid_padding)[:8]`)

## Incremental Resume

When `batch_knf_unified.csv` already exists (legacy names are also supported,
including older `atlas_submission.csv`), and `--force` is not set, existing rows
are reused and only new files are processed.

## Docker

Container workflows are documented in [`README.DOCKER.md`](README.DOCKER.md).

## Releasing

Release steps are documented in [`RELEASE.md`](RELEASE.md).

## License

MIT (`LICENSE`)
