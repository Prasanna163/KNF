#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "bash" || "${1:-}" == "sh" ]]; then
  exec "$@"
fi

if [[ $# -eq 0 ]]; then
  exec knf --help
fi

case "${1:-}" in
  knf|knf-gui)
    exec "$@"
    ;;
  gui)
    shift || true
    exec knf-gui "$@"
    ;;
  *)
    exec knf "$@"
    ;;
esac

