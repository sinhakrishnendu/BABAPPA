# Known-Truth Simulation Benchmark

The recommended known-truth benchmark is the script-based BABAPPA/aBSREL workflow in:

```text
benchmarks/known_truth_absrel/
```

The benchmark compares BABAPPA and HyPhy aBSREL against explicit simulator labels. The simulator labels are the truth. aBSREL is an external comparator, not a truth source.

## Why Known Truth Is Necessary

Empirical data do not provide complete labels for selected branches, selected sites, null families, or OOD conditions. A known-truth simulation benchmark makes AUROC, AUPRC, power, false-call behavior, empirical FDR, and OOD abstention measurable.

## Public Commands

Smoke profile:

```bash
bash benchmarks/known_truth_absrel/run_smoke.sh
bash benchmarks/known_truth_absrel/compare_smoke.sh
```

Pilot profile:

```bash
bash benchmarks/known_truth_absrel/run_pilot.sh
bash benchmarks/known_truth_absrel/run_absrel_pilot.sh
bash benchmarks/known_truth_absrel/compare_pilot.sh
```

Paper profile:

```bash
bash benchmarks/known_truth_absrel/run_paper.sh
```

Generated outputs are written under `benchmark_runs/`.

## Truth Schema

Each run writes:

- `truth/family_truth.tsv`;
- `truth/branch_site_truth.tsv`;
- `truth/selected_sites.tsv`;
- `truth/selected_branches.tsv`;
- `manifest.tsv`;
- per-family simulated codon FASTA and tree files.

Truth files are benchmark labels only. They must never be used as empirical inference inputs or scoring features.

## Metrics

The comparison reports AUROC, AUPRC, precision, recall/power, specificity, F1, MCC, FPR, FNR, empirical FDR, failure rate, and OOD false-call rate when enough evaluable families are available.

## Manuscript Interpretation

Known-truth performance supports simulation-validation claims about BABAPPA's conservative, OOD-gated behavior. It does not support unsupported empirical discovery claims. Empirical claims still require dataset-specific QC, OOD status, native calibration, controls, and biological interpretation.
