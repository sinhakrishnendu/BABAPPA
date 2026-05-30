# BABAPPA Run Layout

Future long-running BABAPPA outputs should be grouped under `runs/` instead of written directly into the repository root. This prevents root-directory sprawl, makes cleanup safer, and keeps controlled, external, and final benchmark artifacts separable.

Recommended layout:

```text
runs/
  controlled_10k/
  external_1k_fast/
  external_10k_fast/
  final_100k/
```

## Guidance

- Do not move existing completed outputs automatically; preserve their current paths for reproducibility.
- New controlled 10K reruns should go under `runs/controlled_10k/`.
- New fast external 1K validation should go under `runs/external_1k_fast/`.
- New fast external 10K validation should go under `runs/external_10k_fast/` only after 1K method-policy review.
- Final 100K benchmark outputs should go under `runs/final_100k/`.
- Keep generated user-run scripts inside the run directory that owns their outputs.
- Keep `method_policy_*`, reports, summaries, logs, and expected-output manifests with the run they describe.

## Current Workspace Caveat

This recommendation is forward-looking. Existing root-level directories such as `site_neural_external_aligner_validation_high`, `report_external_aligner_validation_high`, and controlled 10K outputs should not be deleted or moved by automated tooling.
