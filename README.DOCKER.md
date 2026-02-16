# KNF-CORE Docker Guide

This document explains how to build and run KNF-CORE in Docker for this branch.

## What is included in the container

The image builds a complete KNF runtime with:

- Python 3.11
- KNF package (`knf` CLI)
- xTB
- Open Babel (`obabel`)
- Multiwfn (no-GUI Linux binary)

## Files used

- `Dockerfile`: container build definition
- `docker-compose.yml`: ready-to-run compose service
- `scripts/docker-entrypoint.sh`: startup wrapper that runs `knf`
- `.dockerignore`: excludes large/unneeded files from build context

## Quick start

1. Build image:

```bash
docker build -t knf-core:latest .
```

2. Run a single molecule from current folder:

```bash
docker run --rm -v "$(pwd):/work" -w /work knf-core:latest example.mol --charge 0 --force
```

3. Check CLI help:

```bash
docker run --rm knf-core:latest --help
```

## Docker Compose

Build and run the default compose command:

```bash
docker compose up --build
```

Current default command in `docker-compose.yml` runs:

```text
example.mol --charge 0 --force
```

Change the `command` field in `docker-compose.yml` to run your own input and options.

## Output behavior

- Input files are read from `/work` (host directory mounted into container).
- Results are written back to your host under:
  - `Results/<input_stem>/`

Typical output files:

- `knf.json`
- `output.txt`
- `xtbopt.xyz`
- xTB/Multiwfn intermediates (`wbo`, `molden.input`, `nci_grid.txt`, etc.)

## Useful run patterns

Run a whole directory in multi mode:

```bash
docker run --rm -v "$(pwd):/work" -w /work knf-core:latest molecules --processing multi --force
```

Run with explicit workers:

```bash
docker run --rm -v "$(pwd):/work" -w /work knf-core:latest molecules --processing multi --workers 4 --force
```

Open an interactive shell:

```bash
docker run --rm -it -v "$(pwd):/work" -w /work knf-core:latest bash
```

## Health/dependency checks

Inside a running container, verify core tools:

```bash
knf --help
xtb --version
obabel -V
command -v Multiwfn
```

## Notes for Windows users

- In PowerShell, use `${PWD}` if `$(pwd)` causes issues:

```powershell
docker run --rm -v "${PWD}:/work" -w /work knf-core:latest example.mol --charge 0 --force
```

- Ensure Docker Desktop file sharing is enabled for your project drive.

## Troubleshooting

- `ModuleNotFoundError` during CLI startup:
  - Rebuild image without cache:
  - `docker build --no-cache -t knf-core:latest .`

- `Multiwfn` not found:
  - Confirm image built successfully and entrypoint is not overridden incorrectly.

- No output created:
  - Confirm input path exists inside mounted `/work`.
  - Run with `--debug` for detailed logs.

- Slow or very large build context:
  - Keep heavy folders excluded in `.dockerignore` (already configured).

## Branch-specific behavior

Fragment handling in this branch:

- 1 fragment: `f1 = 0.0`, `f2 = 180.0`
- 2 fragments: `f1` is COM distance, `f2` is detected H-bond angle
- More than 2 fragments: `f1` is average pairwise COM distance, `f2 = 180.0`

## Recommended workflow

1. Put inputs in the repository root (or a subfolder).
2. Run container with a bind mount to `/work`.
3. Read outputs from `Results/...` on host.
4. Commit only required artifacts (avoid committing generated intermediates).
