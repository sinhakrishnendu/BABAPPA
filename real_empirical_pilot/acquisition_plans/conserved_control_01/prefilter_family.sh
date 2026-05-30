#!/usr/bin/env bash
set -euo pipefail
echo 'USER-RUN ONLY - DO NOT EXECUTE IN CODEX'
babappa prefilter-empirical-family --cds-fasta real_empirical_pilot/input/cds/conserved_control_01.cds.fasta --tree-file real_empirical_pilot/input/trees/conserved_control_01.treefile --foreground Arabidopsis_thaliana --outdir real_empirical_pilot/prefilter/conserved_control_01 --max-mean-pdistance 0.25 --min-taxa 6 --min-codons 100
