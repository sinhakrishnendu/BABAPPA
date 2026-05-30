#!/usr/bin/env bash
set -euo pipefail
test -f package/deployable_model_manifest.json
test -f package/model_card.md
find package/models -name '*.pt' | grep -q .
echo 'Deployable model package structure looks present.'
