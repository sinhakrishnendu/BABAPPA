#!/usr/bin/env bash
set -euo pipefail
echo 'USER-RUN ONLY - DO NOT EXECUTE IN CODEX'
babappa parse-codeml-reference --codeml-dir . --outdir ../codeml_parsed
