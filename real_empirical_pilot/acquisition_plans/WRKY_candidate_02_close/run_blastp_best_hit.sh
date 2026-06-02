#!/usr/bin/env bash
set -euo pipefail
echo 'MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING'

# Run BLASTP best-hit search for Arabidopsis_thaliana AT2G38470.
# makeblastdb -in proteome.fasta -dbtype prot
# blastp -query query.fasta -db proteome.fasta -outfmt 6 -max_target_seqs 5 > best_hits.tsv
