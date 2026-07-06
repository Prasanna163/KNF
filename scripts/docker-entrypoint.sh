#!/usr/bin/env bash
set -euo pipefail

# Ensure tool paths are present even when shell activation is skipped.
export PATH="/opt/conda/bin:/opt/conda/condabin:/opt/Multiwfn:${PATH}"
export NCIFORGE_MULTIWFN_PATH="${NCIFORGE_MULTIWFN_PATH:-/opt/Multiwfn/Multiwfn}"
export KUID_MULTIWFN_PATH="${KUID_MULTIWFN_PATH:-$NCIFORGE_MULTIWFN_PATH}"
export KNF_MULTIWFN_PATH="${KNF_MULTIWFN_PATH:-$NCIFORGE_MULTIWFN_PATH}"
export XTBHOME="${XTBHOME:-/opt/conda}"

if [[ "${1:-}" == "bash" || "${1:-}" == "sh" ]]; then
  exec "$@"
fi

if [[ $# -eq 0 ]]; then
  exec nciforge --help
fi

if [[ "${1:-}" == "api" ]]; then
  shift
  exec nciforge-api --host 0.0.0.0 --port 8000 "$@"
fi

if [[ "${1:-}" == "nciforge-api" ]]; then
  shift
  exec nciforge-api "$@"
fi

default_xtb_engine="${NCIFORGE_DEFAULT_XTB_ENGINE:-}"
if [[ -n "${default_xtb_engine}" && "${1:-}" != "--help" && "${1:-}" != "-h" ]]; then
  has_xtb_engine=0
  for arg in "$@"; do
    if [[ "${arg}" == "--xtb-engine" || "${arg}" == --xtb-engine=* ]]; then
      has_xtb_engine=1
      break
    fi
  done
  if [[ "${has_xtb_engine}" -eq 0 ]]; then
    set -- "$@" --xtb-engine "${default_xtb_engine}"
  fi
fi

exec nciforge "$@"
