# NCIForge Docker Guide

This guide covers containerized NCIForge workflows.

## Image Contents

The Docker image installs:

- Python 3.11
- Project package and CLI (`nciforge`, with `knf` alias)
- HTTP API command (`nciforge-api`)
- PyTorch (CPU build)
- xTB (conda-forge)
- Open Babel (`obabel`)
- RDKit
- Matplotlib (headless mode)
- Multiwfn (Linux no-GUI binary)

The Linux container uses the conda-forge `xtb` executable by default through
`NCIFORGE_DEFAULT_XTB_ENGINE=xtb`. You can still pass `--xtb-engine ...` for
CLI runs or set `xtb_engine` in API job options to override it. The packaged
`xtbx` launcher is intended for the bundled Windows runtime path.

Runtime environment includes:

- `PATH=/opt/conda/bin:/opt/conda/condabin:/opt/Multiwfn:$PATH`
- `NCIFORGE_MULTIWFN_PATH=/opt/Multiwfn/Multiwfn`
- `KUID_MULTIWFN_PATH=/opt/Multiwfn/Multiwfn` (compat)
- `KNF_MULTIWFN_PATH=/opt/Multiwfn/Multiwfn` (compat)
- `NCIFORGE_DEFAULT_XTB_ENGINE=xtb`
- `XTBHOME=/opt/conda`
- `MPLBACKEND=Agg`
- `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` (default `4`)

## Files

- `Dockerfile`: build definition
- `docker-compose.yml`: ready-to-run NCIForge service
- `scripts/docker-entrypoint.sh`: entrypoint wrapper
- `.dockerignore`: build-context exclusions

## Build

```bash
docker build -t nciforge:1.0.9 -t nciforge:latest .
```

## Run

Single molecule:

```bash
docker run --rm -v "$(pwd):/work" -w /work nciforge:1.0.9 example.mol --charge 0 --force --xtb-engine xtb
```

Directory batch:

```bash
docker run --rm -v "$(pwd):/work" -w /work nciforge:1.0.9 molecules --processing multi --force --xtb-engine xtb
```

Universal KUID recompute:

```bash
docker run --rm -v "$(pwd):/work" -w /work nciforge:1.0.9 existing_runs --universal-kuid
```

Interactive shell:

```bash
docker run --rm -it -v "$(pwd):/work" -w /work nciforge:1.0.9 bash
```

HTTP API:

```bash
docker run --rm -p 8000:8000 -v "$(pwd):/work" -w /work nciforge:1.0.9 api
```

Then open:

```text
http://127.0.0.1:8000/health
```

## Docker Compose

```bash
docker compose up --build
```

Default compose command:

```text
example.mol --charge 0 --force --xtb-engine xtb
```

Update `command` in `docker-compose.yml` for your own workload. To run the API
with Compose, use:

```yaml
command: ["api"]
```

## Outputs

With `-v "$(pwd):/work"`:

- inputs are read from your host folder
- result artifacts are written back under `Results/...`

Common outputs include:

- `knf.json` (contains `kuid` + `kuid_intensive`)
- `batch_knf.json`
- `batch_knf_unified.csv`
- `kuid_calibration.json`
- `kuid_intensive_calibration.json`
- `kuid_*index*.json` / `kuid_*index*.csv`
- `submission_bundle/atlas_submission.csv` (when `--atlas-bundle` is used)

## Health Checks

Inside container:

```bash
nciforge --help
knf --help
geoinit --help
nciforge-api --help
xtb --version
obabel -V
command -v Multiwfn
echo "$NCIFORGE_MULTIWFN_PATH"
echo "$XTBHOME"
```

## Windows PowerShell

Use `${PWD}` in mount expressions:

```powershell
docker run --rm -v "${PWD}:/work" -w /work nciforge:1.0.9 example.mol --charge 0 --force --xtb-engine xtb
```

## Troubleshooting

- Build issues after dependency changes:
  - `docker build --no-cache -t nciforge:1.0.9 -t nciforge:latest .`
- No output files:
  - verify mounted path and input path inside `/work`

