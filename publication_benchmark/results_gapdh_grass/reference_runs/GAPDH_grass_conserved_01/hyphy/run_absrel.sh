#!/usr/bin/env bash
set -euo pipefail
echo 'USER-RUN ONLY - DO NOT EXECUTE IN CODEX'
hyphy absrel --alignment alignment.fasta --tree tree_foreground.nwk --branches Foreground --output absrel.json
