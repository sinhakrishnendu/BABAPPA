#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_babappa}"
mkdir -p "${MPLCONFIGDIR}"

./.venv/bin/python -m pytest -q
