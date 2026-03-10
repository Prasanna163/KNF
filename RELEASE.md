# KNF Release Checklist

## 1) Update version

Update all version references for the release (example `1.0.5`):

- `setup.py` (`version=...`)
- `knf_core/main.py` (`CLI_TITLE`)
- `README.md` version line
- Docker tag references in:
  - `README.md`
  - `README.DOCKER.md`
  - `docker-compose.yml`

## 2) Build package artifacts

```bash
python -m build
```

Expected artifacts:

- `dist/knf-<version>.tar.gz`
- `dist/knf-<version>-py3-none-any.whl`

## 3) Validate package metadata

```bash
python -m twine check dist/knf-<version>*
```

## 4) Upload to PyPI

```bash
python -m twine upload dist/knf-<version>*
```

## 5) Verify publish on PyPI

```bash
python -m pip index versions KNF
```

## 6) Create GitHub tag and release

Naming convention used in this repo:

- Tag: `v<version>-knf-gpu` (example: `v1.0.5-knf-gpu`)
- Release title: `KNF-Core v<version> (KNF-GPU)`

Commands:

```bash
git checkout KNF-GPU
git pull
git tag -a v<version>-knf-gpu -m "KNF-Core v<version> (KNF-GPU)"
git push origin KNF-GPU
git push origin v<version>-knf-gpu
gh release create v<version>-knf-gpu --title "KNF-Core v<version> (KNF-GPU)" --notes-file RELEASE_NOTES.md
```

## 7) Docker smoke test (recommended)

Build:

```bash
docker build -t knf-core:<version> -t knf-core:latest .
```

CLI smoke run:

```bash
docker run --rm -v "$(pwd):/work" -w /work knf-core:<version> example.mol --charge 0 --force
```
