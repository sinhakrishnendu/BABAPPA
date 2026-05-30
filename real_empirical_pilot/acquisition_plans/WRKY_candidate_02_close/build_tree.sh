#!/usr/bin/env bash
set -euo pipefail
echo 'USER-RUN ONLY - DO NOT EXECUTE IN CODEX'

# Align proteins/CDS and infer an ML tree.
# mafft --auto candidate.protein.fasta > candidate.protein.aln.fasta
# iqtree -s candidate.protein.aln.fasta -m MFP -bb 1000 -nt AUTO -pre tree
