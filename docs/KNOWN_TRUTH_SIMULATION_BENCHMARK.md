# Known-Truth Simulation Benchmark

BABAPPA-BENCH-SIM-v1 is the primary validation layer for BABAPPA. It benchmarks the software against families where the truth is known by construction: null families, positive branch-site families, OOD families, alignment-difficulty families, and saturation-tier stress cases.

## Why Known Truth Is Necessary

HyPhy and codeml are important external likelihood-reference methods, but their empirical calls are not ground truth. A method paper needs a benchmark where the selected branches, selected sites, null families, and OOD conditions are known. BABAPPA-BENCH-SIM-v1 provides that benchmark.

## Benchmark Regimes

The design includes:

- null low/moderate/high/extreme divergence;
- null alignment-difficult, short-gene, few-taxa, and long-branch cases;
- weak, moderate, and strong branch-site positive regimes;
- sparse, clustered, multi-branch, short-foreground, and long-foreground positives;
- OOD stress regimes including extreme saturation, high gap burden, compositional bias, paralogy-like mixtures, tree-mismatch-like inputs, too-short genes, and too-few taxa.

## Truth Schema

Each family stores:

- `family_truth.json`;
- `branch_site_truth.tsv`;
- `selected_sites.tsv`;
- `selected_branches.tsv`;
- `regime_metadata.json`;
- simulated CDS FASTA;
- simulated tree.

Dataset-level truth is stored in `benchmark_truth_manifest.tsv` and `benchmark_truth_manifest.json`.

Truth files are for benchmark evaluation only. They must never be used as empirical inference inputs or scoring features.

## Metrics

The evaluator reports:

- gene-level AUROC, AUPRC, precision, recall, specificity, F1, MCC, FPR, FNR, empirical FDR;
- branch-site AUROC/AUPRC and called-row precision/recall when row-level scores are available;
- OOD abstention rate and OOD false-call rate;
- stratified metrics by saturation tier, applicability, gene length, taxon count, foreground branch length, effect size, selected-site fraction, and alignment difficulty;
- BH q-values, FDR, and power under the simulated truth labels.

## Smoke, Pilot, Paper, And Extended Profiles

Create the design:

```bash
babappa design-known-truth-benchmark \
  --outdir known_truth_benchmark_design_v1 \
  --benchmark-name BABAPPA-BENCH-SIM-v1 \
  --seed 42
```

Create a smoke plan:

```bash
babappa plan-known-truth-benchmark \
  --profile smoke \
  --design-dir known_truth_benchmark_design_v1 \
  --outdir known_truth_benchmark_plan_smoke \
  --methods identity,mafft,babappalign,muscle \
  --device auto \
  --threads 2 \
  --max-workers 1
```

Create user-run plans:

```bash
babappa plan-known-truth-benchmark --profile pilot --design-dir known_truth_benchmark_design_v1 --outdir known_truth_benchmark_plan_pilot
babappa plan-known-truth-benchmark --profile paper --design-dir known_truth_benchmark_design_v1 --outdir known_truth_benchmark_plan_paper
babappa plan-known-truth-benchmark --profile extended --design-dir known_truth_benchmark_design_v1 --outdir known_truth_benchmark_plan_extended
```

Pilot, paper, and extended profiles are long-run jobs. They should be executed offline by the user, not inside short interactive coding sessions.

## Reference Comparison

Reference methods should be compared against the same simulation truth:

```bash
babappa plan-known-truth-reference-comparison \
  --benchmark-dir known_truth_benchmark_paper \
  --outdir known_truth_benchmark_paper/reference_comparison_plan \
  --tools codeml,absrel,meme \
  --max-families 100
```

codeml, aBSREL, and MEME are external comparators, not truth labels.

## Manuscript Interpretation

Known-truth performance supports claims about simulation-validated behavior, OOD abstention, calibration, FDR, and power. It does not support unsupported empirical discovery claims. Empirical claims still require dataset-specific QC, OOD status, native calibration, controls, and biological interpretation.

