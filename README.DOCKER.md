# KNF-CORE Docker Guide

This guide covers containerized KNF CLI usage.

## What The Image Contains

The Docker image installs:
- Python 3.11
- KNF package (`knf`)
- xTB (from conda-forge)
- Open Babel (`obabel`)
- Multiwfn (Linux no-GUI binary)

Runtime environment exported in image/entrypoint:
- `PATH=/opt/conda/bin:/opt/conda/condabin:/opt/Multiwfn:$PATH`
- `KNF_MULTIWFN_PATH=/opt/Multiwfn/Multiwfn`
- `XTBHOME=/opt/conda`
- `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` (default `4`)

## Files

- `Dockerfile`: image build definition
- `docker-compose.yml`: ready-to-run CLI service
- `scripts/docker-entrypoint.sh`: entrypoint wrapper for `knf`
- `.dockerignore`: build-context exclusions

## Build

```bash
docker build -t knf-core:latest .
```

Build includes:
- `xtb` install via `micromamba install -c conda-forge xtb`
- optional torch extra via `pip install .[torch-nci]`

## Run: CLI

Single molecule:

```bash
docker run --rm -v "$(pwd):/work" -w /work knf-core:latest example.mol --charge 0 --force
```

Directory batch:

```bash
docker run --rm -v "$(pwd):/work" -w /work knf-core:latest molecules --processing multi --force
```

Interactive shell:

```bash
docker run --rm -it -v "$(pwd):/work" -w /work knf-core:latest bash
```

## Docker Compose

Run default compose service:

```bash
docker compose up --build
```

`docker-compose.yml` runs:

```text
example.mol --charge 0 --force
```

Edit `command` to run your own inputs/options.

Compose also provides default environment wiring for Multiwfn/xTB:
- `KNF_MULTIWFN_PATH=/opt/Multiwfn/Multiwfn`
- `XTBHOME=/opt/conda`
- thread env vars for BLAS/OMP

## Output Behavior

With `-v "$(pwd):/work"`:
- inputs are read from your host folder
- results are written back to host `Results/...`

Common outputs:
- `knf.json`
- `output.txt`
- `xtbopt.xyz`
- `batch_knf.json` / `batch_knf.csv` (batch mode)

## Health/Tool Checks

Inside container:

```bash
knf --help
xtb --version
obabel -V
command -v Multiwfn
echo "$KNF_MULTIWFN_PATH"
echo "$XTBHOME"
```

## Windows PowerShell Notes

Use `${PWD}` in mount expressions:

```powershell
docker run --rm -v "${PWD}:/work" -w /work knf-core:latest example.mol --charge 0 --force
```

## Troubleshooting

- Build issues after dependency changes:
  - `docker build --no-cache -t knf-core:latest .`
- No output files:
  - verify mounted path and input path inside `/work`
