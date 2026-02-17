# KNF-CORE Docker Guide

This guide covers containerized KNF CLI usage.

## What The Image Contains

The Docker image installs:
- Python 3.11
- KNF package (`knf`)
- xTB
- Open Babel (`obabel`)
- Multiwfn (Linux no-GUI binary)

## Files

- `Dockerfile`: image build definition
- `docker-compose.yml`: ready-to-run CLI service
- `scripts/docker-entrypoint.sh`: entrypoint wrapper for `knf`
- `.dockerignore`: build-context exclusions

## Build

```bash
docker build -t knf-core:latest .
```

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
