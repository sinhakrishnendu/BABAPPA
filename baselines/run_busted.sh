#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY_BIN="${BABAPPA_PYTHON:-python3}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Pinned default image used by adapter when backend=docker
export BABAPPA_HYPHY_BACKEND="${BABAPPA_HYPHY_BACKEND:-auto}"

exec "${PY_BIN}" -m babappa.baseline_adapters --method busted "$@"
