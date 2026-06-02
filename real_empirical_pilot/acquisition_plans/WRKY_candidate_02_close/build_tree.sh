#!/usr/bin/env bash
set -euo pipefail
echo 'MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING'

# Align proteins/CDS and infer an ML tree.
# mafft --auto candidate.protein.fasta > candidate.protein.aln.fasta
# iqtree -s candidate.protein.aln.fasta -m MFP -bb 1000 -nt AUTO -pre tree
