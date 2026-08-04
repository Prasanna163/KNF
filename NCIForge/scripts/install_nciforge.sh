#!/usr/bin/env bash
set -euo pipefail
PYTHON_EXE="${PYTHON_EXE:-python3}"
"$PYTHON_EXE" "./scripts/install_nciforge_cli.py"
