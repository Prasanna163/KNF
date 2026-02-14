#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "bash" || "${1:-}" == "sh" ]]; then
  exec "$@"
fi

if [[ $# -eq 0 ]]; then
  exec knf --help
fi

exec knf "$@"

