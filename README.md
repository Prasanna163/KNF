# KNF-CORE

KNF-CORE is an automated computational chemistry pipeline for generating KNF descriptors from molecular structure files. It combines xTB + Multiwfn + custom post-processing to produce:

- SNCI
- SCDI variance
- 9D KNF vector (`f1` ... `f9`)

## Version

Current package version in this branch: `1.0`

## What This Branch Includes

- Automatic input conversion to XYZ when needed (via Open Babel)
- Single-file and directory processing modes
- Auto-configured multi-worker batch mode
- Dockerized runtime (`Dockerfile`, `docker-compose.yml`, container entrypoint)
- Updated fragment handling rules

Fragment behavior:

- `1` fragment: `f1 = 0.0`, `f2 = 180.0`
- `2` fragments: `f1` = COM distance, `f2` = detected H-bond angle
- `>2` fragments: `f1` = average COM distance over all unique pairs, `f2 = 180.0`

## Requirements

- Python `>=3.8`
- `xtb` in `PATH`
- `obabel` (Open Babel) in `PATH`
- `Multiwfn` in `PATH`

## Install

From PyPI:

```bash
pip install KNF
```

From source:

```bash
git clone https://github.com/Prasanna163/KNF.git
cd KNF
pip install .
```

## CLI Usage

Basic run:

```bash
knf input_molecule.sdf
```

Useful options:

- `--charge <int>`: total charge (default `0`)
- `--spin <int>`: multiplicity (default `1`)
- `--force`: recompute stages
- `--clean`: remove prior working folder for that input
- `--debug`: verbose logging
- `--processing <single|multi>` (alias: `--processes`)
- `--workers <int>`: explicit workers for multi mode
- `--output-dir <path>`: custom results root
- `--ram-per-job <MB>`: RAM hint for auto worker selection
- `--refresh-autoconfig`: regenerate auto-config cache

Example:

```bash
knf example.mol --charge 0 --force
```

Directory batch example:

```bash
knf ./molecules --processing multi --force
```

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

## Output Layout

Default output root:

- File input: `<input_parent>/Results/<input_stem>/`
- Directory input: `<input_dir>/Results/<file_stem>/`

Typical output files:

- `knf.json`
- `output.txt`
- `xtbopt.xyz`
- xTB/Multiwfn intermediates (`wbo`, `molden.input`, `nci_grid.txt`, etc.)

## Docker

Quick run:

```bash
docker build -t knf-core:latest .
docker run --rm -v "$(pwd):/work" -w /work knf-core:latest example.mol --charge 0 --force
```

Compose:

```bash
docker compose up --build
```

Full Docker documentation is in `README.DOCKER.md`.

## Releasing

PyPI release steps are documented in `RELEASE.md`.

## License

MIT. See `LICENSE`.
