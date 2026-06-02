# BABAPPA Known-Truth Benchmark

BABAPPA-BENCH-SIM-v1 is the benchmark layer for evaluating BABAPPA against explicit simulator truth. It has two layers:

1. Primary validation: BABAPPA versus known simulator truth.
2. Comparator validation: HyPhy aBSREL versus the same known simulator truth.

aBSREL is a comparator, not ground truth. This benchmark does not position BABAPPA as an aBSREL replacement. BABAPPA is evaluated as a complementary, alignment-aware, OOD-gated, simulation-calibrated branch-site support framework.

## Why Known Simulator Truth Comes First

Empirical datasets do not provide complete branch-site truth labels. Simulator truth makes it possible to measure AUROC, AUPRC, power, FDR, OOD false-call behavior, and calibration against labels that are actually known.

## Benchmark Profiles

- `smoke`: 12 families. Small enough for quick validation.
- `pilot`: 300 families. User-run only.
- `paper`: 5000 families. User-run only after the pilot passes.
- `extended`: 20000 families. Optional user-run profile.

## Regimes

The design includes null, positive branch-site, OOD null, high-saturation, short/few-taxa, long-branch, gap/bias stress, and paralogy-like stress regimes.

## Truth Files

Each simulated family writes:

- `family_truth.json`
- `branch_site_truth.tsv`
- `selected_sites.tsv`
- `selected_branches.tsv`

Truth files are benchmark labels only. They must not be used as empirical inference inputs.

## Run Smoke

```bash
babappa run-known-truth-benchmark \
  --profile smoke \
  --outdir known_truth_benchmark_smoke \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --methods identity,mafft,babappalign,muscle \
  --device auto \
  --seed 42
```

## Select aBSREL Subset

```bash
babappa select-known-truth-absrel-subset \
  --benchmark-dir known_truth_benchmark_smoke \
  --outdir known_truth_absrel_subset_smoke \
  --max-families 12 \
  --stratify-by regime,truth_class,ood_status,saturation_tier
```

## Plan aBSREL Comparator

```bash
babappa plan-known-truth-absrel-comparison \
  --subset known_truth_absrel_subset_smoke/absrel_subset.tsv \
  --outdir known_truth_absrel_comparison_plan_smoke \
  --alignment-source mafft_codon \
  --user-run-only true
```

The generated run script checks HyPhy availability, creates per-family output folders, runs aBSREL, writes logs, skips completed families unless forced, and never includes codeml or MEME.

## Parse Pending Or Completed aBSREL Results

```bash
babappa parse-known-truth-absrel-results \
  --absrel-run-dir known_truth_absrel_comparison_plan_smoke \
  --truth-dir known_truth_benchmark_smoke \
  --outdir known_truth_absrel_results_smoke
```

If aBSREL outputs are absent, families are marked `pending_not_run` rather than causing the parser to fail.

## Compare BABAPPA And aBSREL Against Truth

```bash
babappa compare-known-truth-babappa-absrel \
  --babappa-report known_truth_benchmark_smoke/report \
  --absrel-results known_truth_absrel_results_smoke/absrel_results.tsv \
  --truth-dir known_truth_benchmark_smoke \
  --outdir known_truth_babappa_absrel_comparison_smoke
```

The comparison uses simulator truth as the ground truth. aBSREL is an external comparator only.

## Pilot Before Paper

Generate a pilot plan:

```bash
babappa plan-known-truth-benchmark-suite \
  --profile pilot \
  --outdir known_truth_benchmark_plan_pilot \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --include-absrel true \
  --absrel-max-families 300 \
  --device auto \
  --conda-env molevo
```

Validate the plan before running:

```bash
babappa validate-known-truth-benchmark-plan \
  --plan-dir known_truth_benchmark_plan_pilot
```

Only scale to the paper profile after the pilot benchmark is interpretable.

## Expected Manuscript Tables

- `manuscript_table_simulation_truth.tsv`
- `manuscript_table_ood_abstention.tsv`
- `manuscript_table_power.tsv`
- `manuscript_table_babappa_absrel_comparison.tsv`
- `manuscript_table_runtime_failure.tsv`

## Claim Boundary

BABAPPA is not a likelihood-method replacement. The benchmark supports comparative known-truth validation of BABAPPA as a conservative complementary framework. Empirical discovery claims require separate empirical calibration, controls, and biological interpretation.

