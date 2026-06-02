# Known-Truth Benchmark Report

## Benchmark Purpose

This benchmark evaluates BABAPPA against explicit simulated truth labels.

## Known-Truth Design

Families: 12
Truth classes: {'null': 6, 'ood_null': 2, 'positive': 4}

## BABAPPA Performance Against Truth

Gene AUROC: 1.0
Gene AUPRC: 1.0

## OOD Abstention Performance

OOD abstention rate: 1.0
OOD false-call rate: 0.0

## Calibration/FDR

See `calibration_evaluation/` and `manuscript_table_power.tsv`.

## Reference-Comparison Status

Reference methods should be evaluated against the same simulation truth, not treated as truth.

## Claim Boundary

This known-truth simulation benchmark supports simulation validation and conservative method claims. It does not support empirical discovery claims by itself.

## Recommended Manuscript Tables

- `manuscript_table_simulation_truth.tsv`
- `manuscript_table_ood_abstention.tsv`
- `manuscript_table_power.tsv`
