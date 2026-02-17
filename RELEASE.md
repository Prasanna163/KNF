# PyPI Release Checklist

## 1) Update version

Edit `setup.py` and bump `version='x.y.z'`.

## 2) Build package artifacts

```bash
python -m build
```

Expected artifacts:

- `dist/knf-<version>.tar.gz`
- `dist/knf-<version>-py3-none-any.whl`

## 3) Validate metadata

```bash
python -m twine check dist/knf-<version>*
```

## 4) Upload to PyPI

```bash
python -m twine upload dist/knf-<version>*
```

## 5) Verify publish

```bash
python -m pip index versions KNF
```

## 6) Docker smoke test (recommended)

Build:

```bash
docker build -t knf-core:latest .
```

CLI smoke run:

```bash
docker run --rm -v "$(pwd):/work" -w /work knf-core:latest example.mol --charge 0 --force
```

GUI smoke run:

```bash
docker run --rm -p 8787:8787 -v "$(pwd):/work" -w /work -e KNF_GUI_HOST=0.0.0.0 knf-core:latest knf-gui
```
