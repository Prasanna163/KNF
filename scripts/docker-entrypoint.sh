#!/usr/bin/env bash
set -euo pipefail

# Ensure tool paths are present even when shell activation is skipped.
export PATH="/opt/conda/bin:/opt/conda/condabin:/opt/Multiwfn:${PATH}"
export KNF_MULTIWFN_PATH="${KNF_MULTIWFN_PATH:-/opt/Multiwfn/Multiwfn}"
export XTBHOME="${XTBHOME:-/opt/conda}"

if [[ "${1:-}" == "bash" || "${1:-}" == "sh" ]]; then
  exec "$@"
fi

if [[ $# -eq 0 ]]; then
  exec knf --help
fi

exec knf "$@"
