# BABAPPA Known-Truth Benchmark

The public known-truth benchmark is intentionally small in surface area:

```text
benchmarks/known_truth_absrel/
```

It compares BABAPPA and HyPhy aBSREL against explicit simulator truth. aBSREL is a comparator, not ground truth. BABAPPA is evaluated as a complementary, OOD-gated, simulation-trained branch-site support framework rather than as an aBSREL replacement.

## Profiles

- `smoke`: 12 families, quick validation.
- `pilot`: 300 families, user-run offline.
- `paper`: 5000 families, user-run after pilot review.

## Commands

Smoke:

```bash
bash benchmarks/known_truth_absrel/run_smoke.sh
bash benchmarks/known_truth_absrel/compare_smoke.sh
```

Pilot:

```bash
bash benchmarks/known_truth_absrel/run_pilot.sh
bash benchmarks/known_truth_absrel/run_absrel_pilot.sh
bash benchmarks/known_truth_absrel/compare_pilot.sh
```

Paper:

```bash
bash benchmarks/known_truth_absrel/run_paper.sh
```

## Outputs

Runs are written under `benchmark_runs/` and include simulator truth files, BABAPPA result tables, aBSREL result tables, method-comparison tables, and compact manuscript tables.

## Claim Boundary

The simulator labels are the truth in this benchmark. BABAPPA is not a likelihood-method replacement. The benchmark supports known-truth validation of conservative BABAPPA behavior and does not create empirical discovery claims.
