#!/usr/bin/env bash
set -euo pipefail
echo 'USER-RUN ONLY - DO NOT EXECUTE IN CODEX'
babappa add-prefiltered-family-to-pilot --workspace real_empirical_pilot --prefilter-dir real_empirical_pilot/prefilter/conserved_control_01 --panel-id conserved_control_01 --expected-category likely_negative --reference-status planned
