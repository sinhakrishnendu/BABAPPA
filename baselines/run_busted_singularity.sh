#!/usr/bin/env bash
set -euo pipefail

export BABAPPA_HYPHY_BACKEND="singularity"
exec "$(cd "$(dirname "$0")" && pwd)/run_busted.sh" "$@"
