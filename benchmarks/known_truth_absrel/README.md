# Simplified Known-Truth BABAPPA/aBSREL Benchmark

This folder contains the public benchmark workflow for comparing BABAPPA and HyPhy aBSREL against explicit simulator truth. The simulator labels are the ground truth. aBSREL is an external comparator evaluated against those same labels.

BABAPPA is not presented here as a replacement for aBSREL. The benchmark tests BABAPPA as a conservative, OOD-gated, simulation-trained branch-site support framework.

## Workflow

The workflow is intentionally script-first rather than a large BABAPPA CLI subsystem:

1. Simulate a known-truth codon dataset.
2. Run BABAPPA direct prediction on each MSA/tree pair.
3. Prepare aBSREL input folders.
4. Run aBSREL where requested.
5. Compare both methods against simulator truth.
6. Render benchmark tables and Markdown reports.

## Smoke Run

The smoke profile has 12 families and is meant for quick validation:

```bash
bash benchmarks/known_truth_absrel/run_smoke.sh
bash benchmarks/known_truth_absrel/run_absrel_smoke.sh
bash benchmarks/known_truth_absrel/compare_smoke.sh
bash benchmarks/known_truth_absrel/diagnose_smoke.sh
```

Outputs are written to:

```text
benchmark_runs/known_truth_absrel_smoke/
```

By default the smoke scripts use 4 parallel family jobs. Override with:

```bash
BABAPPA_BENCH_JOBS=2 bash benchmarks/known_truth_absrel/run_smoke.sh
BABAPPA_BENCH_JOBS=2 bash benchmarks/known_truth_absrel/run_absrel_smoke.sh
```

## Pilot Run

The pilot profile has 300 families and is intended for user-run offline benchmarking:

```bash
bash benchmarks/known_truth_absrel/run_pilot.sh
bash benchmarks/known_truth_absrel/run_absrel_pilot.sh
bash benchmarks/known_truth_absrel/compare_pilot.sh
```

Outputs are written to:

```text
benchmark_runs/known_truth_absrel_pilot/
```

The pilot default is 12 parallel family jobs. To monitor from another terminal:

```bash
bash benchmarks/known_truth_absrel/monitor_pilot.sh
```

## Paper Run

The paper profile has 5000 families. Run this only when the pilot output looks correct:

```bash
bash benchmarks/known_truth_absrel/run_paper.sh
bash benchmarks/known_truth_absrel/run_absrel_paper.sh
bash benchmarks/known_truth_absrel/compare_paper.sh
bash benchmarks/known_truth_absrel/apply_paper_frozen_policy.sh
```

Outputs are written to:

```text
benchmark_runs/known_truth_absrel_paper/
```

The paper default is 14 parallel family jobs, leaving headroom on an 18-core machine:

```bash
BABAPPA_BENCH_JOBS=14 bash benchmarks/known_truth_absrel/run_paper.sh
BABAPPA_BENCH_JOBS=14 bash benchmarks/known_truth_absrel/run_absrel_paper.sh
```

If the machine is stable, try 16 jobs. Do not start with all 18 cores because aBSREL and Python subprocess overhead benefit from memory and scheduling headroom.

Monitor from another terminal:

```bash
bash benchmarks/known_truth_absrel/monitor_paper.sh
```

## Independent Validation Run

If a threshold is selected after inspecting the paper profile, it must not be claimed as final on that same profile. Freeze it as a validation candidate and test it unchanged on the independent validation profile:

```bash
bash benchmarks/known_truth_absrel/run_validation.sh
bash benchmarks/known_truth_absrel/run_absrel_validation.sh
bash benchmarks/known_truth_absrel/compare_validation.sh
```

Outputs are written to:

```text
benchmark_runs/known_truth_absrel_validation/
```

The validation profile uses a different random seed from the paper profile and applies:

```text
benchmarks/known_truth_absrel/threshold_policy_validation_candidate.yaml
```

These paper-derived candidate thresholds are validation hypotheses only until this independent run completes. Do not retune them on validation results.

Monitor from another terminal:

```bash
bash benchmarks/known_truth_absrel/monitor_validation.sh
```

Runs are foreground scripts: they parallelize internally and return only when the active stage is complete. They do not detach themselves with `nohup` or background jobs.

Resume behavior:

- BABAPPA skips a family when a valid `gene_summary.tsv` and score table already exist, unless `BABAPPA_FORCE=1`.
- aBSREL skips a family when its JSON exists and parses via the official `test results -> positive test results` field, unless `BABAPPA_FORCE=1`.
- Final TSV files are written only by the main process after worker completion, sorted by `family_id`.

## Output Files

Each run creates:

- `families/`: simulated codon alignments and trees.
- `truth/family_truth.tsv`: family-level simulator truth.
- `truth/branch_site_truth.tsv`: selected branch-site pairs.
- `truth/selected_sites.tsv`: selected sites.
- `truth/selected_branches.tsv`: selected foreground branches.
- `manifest.tsv`: runnable family manifest.
- `babappa_results.tsv`: BABAPPA family-level result table.
- `babappa_scores/`: per-family BABAPPA output folders.
- `babappa_failures.tsv`: failed BABAPPA families, if any.
- `absrel_results.tsv`: aBSREL family-level parser output.
- `absrel_json/`: aBSREL JSON outputs.
- `absrel_failures.tsv`: failed aBSREL families, if any.
- `method_comparison.tsv`: per-family comparison against truth.
- `manuscript_table_babappa_vs_absrel.tsv`: compact metrics table.
- `benchmark_summary.md`: report for manuscript support.

## Metrics

The comparison step reports AUROC, AUPRC, precision, recall/power, specificity, F1, MCC, FPR, FNR, empirical FDR, failure rate, and OOD false-call rate whenever enough evaluated families are present.

## Interpretation

This benchmark is for known-truth simulation validation. Empirical datasets have no simulator labels, so empirical aBSREL calls must not be treated as truth. In this benchmark, both BABAPPA and aBSREL are measured against the same simulated labels.
