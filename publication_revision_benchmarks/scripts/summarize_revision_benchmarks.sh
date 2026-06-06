#!/usr/bin/env bash
set -euo pipefail
echo "Revision benchmark summary collection"
find . -maxdepth 3 -type f \( -name '*summary*.md' -o -name '*summary*.tsv' -o -name '*results*.tsv' \) | sort
