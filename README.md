# KUID Branch

This branch is focused on KUID generation, calibration, and indexing workflows.

Current package version in this branch: `1.0.5`

## Scope

- Generate full `KUID` identifiers for single and batch runs
- Generate topology-oriented `KUID-Intensive` identifiers
- Persist calibration metadata for reproducible encoding
- Build lookup artifacts (prefix, reverse, and bridge indexes)
- Recompute universal KUID outputs from existing batch folders (`--universal-kuid`)

## KUID Representations

### 1) Full KUID (Instance Address)

- Feature set: `f1..f9`
- Format: 18 hex chars (`00-FF` per feature in canonical order)
- Intended use: exact instance lookup and deduplication

### 2) KUID-Intensive (Topology Passport)

- Feature set: `f3,f4,f7,f8,f9`
- Format: 5 hex chars (display form: `X-X-X-X-X`)
- Intended use: family-level grouping and topology comparison

### Undefined `f2` behavior

- If `f2` is undefined, the pipeline preserves `f2_defined = 0`.
- Full KUID still remains available by using an internal surrogate bin during encoding.
- KUID-Intensive remains directly comparable because it does not depend on `f2`.

### Prefix semantics

- `KUID_prefix2`: `f3`
- `KUID_prefix4`: `f3+f4`
- `KUID_prefix6`: `f3+f4+f7`
- Full topology passport: `KUID_Intensive_raw = f3+f4+f7+f8+f9`

## Requirements

- Python `>=3.8`
- External tools in `PATH`:
  - `xtb`
  - `obabel`
- `Multiwfn` only when using `--nci-backend multiwfn`

Optional:

- `torch` for Torch NCI backend

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

## Run KUID Workflows

Primary CLI command in this branch:

- `kuid`

Single molecule:

```bash
kuid example.mol --force
```

Batch run:

```bash
kuid ./molecules --processing multi --workers 4 --force
```

Split into batches and emit combined universal outputs:

```bash
kuid ./molecules --batches 4
```

Recompute universal KUID from existing batch outputs:

```bash
kuid ./existing_runs --universal-kuid
```

## KUID Files Emitted

Single-run outputs include:

- `knf.json` (contains `kuid` and `kuid_intensive` sections)
- `kuid_calibration.json`

Batch root outputs include:

- `batch_knf.json`
- `batch_knf_unified_kuid_intensive.csv`
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

## Key CSV Columns

`batch_knf_unified_kuid_intensive.csv` includes:

- `KUID_raw`
- `KUID`
- `KUID_Cluster`
- `KUID_Intensive_raw`
- `KUID_Intensive`
- `KUID_Intensive_Cluster`
- `KUID_prefix2`
- `KUID_prefix4`
- `KUID_prefix6`
- `f2_defined`

## Incremental Resume

When `batch_knf_unified_kuid_intensive.csv` already exists and `--force` is not set, existing rows are reused and only new files are processed.

## Docker

KUID-focused Docker usage is documented in `README.DOCKER.md`.

## Releasing

Release steps are documented in `RELEASE.md`.

## License

MIT (`LICENSE`)
