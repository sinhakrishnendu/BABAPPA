#!/usr/bin/env bash
set -euo pipefail
echo 'MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING'
babappa add-prefiltered-family-to-pilot --workspace real_empirical_pilot --prefilter-dir real_empirical_pilot/prefilter/conserved_control_01 --panel-id conserved_control_01 --expected-category likely_negative --reference-status planned
