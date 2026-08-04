# NCIForge Release Checklist

## 1) Update version

Update all version references for the release (example `1.0.7`):

- `setup.py` (`version=...`)
- `knf_core/main.py` (`CLI_VERSION`)
- `README.md` version line
- Docker tag references in:
  - `README.DOCKER.md`
  - `docker-compose.yml`
  - `Dockerfile` (`NCIFORGE_VERSION`)

## 2) Build package artifacts

```bash
python -m build
```

Expected artifacts:

- `dist/nciforge-<version>.tar.gz`
- `dist/nciforge-<version>-py3-none-any.whl`

## 3) Validate package metadata

```bash
python -m twine check dist/nciforge-<version>*
```

## 4) Upload to PyPI

```bash
python -m twine upload dist/nciforge-<version>*
```

## 5) Verify publish on PyPI

```bash
python -m pip index versions nciforge
```

## 6) Create GitHub tag and release

Release convention:

- Tag: `v<version>` (example: `v1.0.7`)
- Release title: `NCIForge v<version>`

Commands:

```bash
git checkout main
git pull
git tag -a v<version> -m "NCIForge v<version>"
git push origin main
git push origin v<version>
gh release create v<version> --title "NCIForge v<version>" --notes-file RELEASE_NOTES.md
```

## 7) Docker smoke test (recommended)

Build:

```bash
docker build -t nciforge:<version> -t nciforge:latest .
```

CLI smoke run:

```bash
docker run --rm -v "$(pwd):/work" -w /work nciforge:<version> example.mol --charge 0 --force
```
