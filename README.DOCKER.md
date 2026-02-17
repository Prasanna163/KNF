# KNF-CORE Docker Guide

This guide covers containerized KNF usage for both CLI and GUI.

## What The Image Contains

The Docker image installs:
- Python 3.11
- KNF package (`knf`, `knf-gui`)
- xTB
- Open Babel (`obabel`)
- Multiwfn (Linux no-GUI binary)
- RDKit (for GUI molecule rendering endpoint)

## Files

- `Dockerfile`: image build definition
- `docker-compose.yml`: CLI and GUI services
- `scripts/docker-entrypoint.sh`: dispatches to `knf` or `knf-gui`
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

## Run: GUI

```bash
docker run --rm -p 8787:8787 -v "$(pwd):/work" -w /work -e KNF_GUI_HOST=0.0.0.0 -e KNF_GUI_PORT=8787 knf-core:latest knf-gui
```

Open:
- `http://127.0.0.1:8787`

## Docker Compose

`docker-compose.yml` defines:
- `knf-cli` (profile: `cli`)
- `knf-gui` (profile: `gui`)

Run CLI profile:

```bash
docker compose --profile cli run --rm knf-cli
```

Run GUI profile:

```bash
docker compose --profile gui up --build knf-gui
```

The compose file also mounts a named volume `knf-cache` to preserve one-time KNF state under `/home/mambauser/.knf`.

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
knf-gui --help
xtb --version
obabel -V
command -v Multiwfn
```

## Windows PowerShell Notes

Use `${PWD}` in mount expressions:

```powershell
docker run --rm -v "${PWD}:/work" -w /work knf-core:latest example.mol --charge 0 --force
```

GUI on Windows:

```powershell
docker run --rm -p 8787:8787 -v "${PWD}:/work" -w /work -e KNF_GUI_HOST=0.0.0.0 knf-core:latest knf-gui
```

## Troubleshooting

- Build issues after dependency changes:
  - `docker build --no-cache -t knf-core:latest .`
- No output files:
  - verify mounted path and input path inside `/work`
- GUI unreachable:
  - confirm port mapping `-p 8787:8787`
  - confirm `KNF_GUI_HOST=0.0.0.0`
