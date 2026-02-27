# BABAPPA Revision Experiment Plan

This plan operationalizes the reviewer-requested validation suite using the current CLI.

## 1) Null Calibration Grid (Type-I Error)

Vary:
- gene length `L`: 300, 900, 1800 nt
- calibration size `N`: 200, 1000, 5000
- tree depth / branch lengths: short vs long trees
- neutral-model mismatch: perturb `kappa`, codon frequencies, or branch lengths between training and test simulation

Primary metric:
- empirical type-I error at alpha = 0.05 and 0.01

## 2) Power Grid (Biologically Grounded Alternatives)

Use omega-mixture alternatives:
- selected fraction: 0.1, 0.3, 0.5
- `omega_alt`: 1.5, 2.5, 4.0

Primary metric:
- power at fixed alpha under rank-based p-values

## 3) Baseline Comparisons

Current built-in baseline:
- entropy mean statistic calibrated by the same neutral simulation process

Planned external baselines:
- HyPhy BUSTED / RELAX
- codeml branch-site models

## 4) Runtime and Practicality

Report:
- wall-clock time per command
- memory usage
- scaling with `L`, `N`, and number of genes

## 5) Reproducibility Requirements

For every run:
- save command manifest (`--manifest`)
- save JSON outputs and TSV records
- pin random seeds
- archive tree and codon-frequency config files

## Example Command Template

```bash
babappa benchmark \
  --tree-file tree.nwk \
  --training-replicates 300 \
  --training-length 900 \
  --gene-length 900 \
  --calibration-size 1000 \
  --n-null-genes 200 \
  --n-alt-genes 200 \
  --selected-fraction 0.3 \
  --omega-alt 2.5 \
  --tail right \
  --output runs/bench_L900_N1000.json \
  --records-tsv runs/bench_L900_N1000.tsv \
  --manifest runs/bench_L900_N1000.manifest.json
```
