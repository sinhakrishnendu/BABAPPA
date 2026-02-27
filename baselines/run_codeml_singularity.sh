#!/usr/bin/env bash
set -euo pipefail

export BABAPPA_CODEML_BACKEND="singularity"
exec "$(cd "$(dirname "$0")" && pwd)/run_codeml.sh" "$@"
