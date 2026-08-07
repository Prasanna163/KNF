# Contributing to NCIForge

Thank you for helping improve NCIForge. Changes to scientific software need to
remain reviewable, reproducible, and compatible with existing result records.

## Before opening a change

1. Open an issue or discussion for changes to descriptor definitions,
   scientific defaults, output schemas, or routing policy.
2. Create a focused branch from `main`.
3. Keep generated run directories, virtual environments, credentials, and
   machine-specific paths out of commits.
4. Add regression tests for changed behaviour.

## Development setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[api,torch-nci,plots]"
python -m pip install pytest build twine
```

Run the standard checks:

```bash
python -m pytest tests -q
python -m compileall -q knf_core geoinit nciforge_xtbx nciforge_cli.py
python -m build
python -m twine check dist/*
```

Tests that execute an external xTB installation are opt-in through
`KNF_RUN_XTB_TESTS=1`.

## Scientific compatibility

- Preserve `knf.json`, `output.txt`, and the established `f1`--`f9` contract
  unless a versioned migration has been agreed.
- Do not silently replace undefined scientific quantities with zero.
- Record protocol changes in metadata and `CHANGELOG.md`.
- Do not combine incompatible `f3`, grid, backend, or calibration definitions
  in one K-UID population.
- Distinguish unit/smoke fixtures from chemically validated calculations.

## Pull requests

Describe the problem, cause, change, validation evidence, and compatibility
impact. Keep unrelated generated artifacts out of the review. By contributing,
you agree that your contribution is distributed under the repository's MIT
License.
