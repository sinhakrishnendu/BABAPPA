#!/usr/bin/env bash
set -euo pipefail
echo 'MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING'

babappa prefilter-empirical-family \
  --cds-fasta candidate.cds.fasta \
  --tree-file tree.treefile \
  --foreground Arabidopsis_thaliana \
  --outdir prefilter/WRKY_candidate_02_close
