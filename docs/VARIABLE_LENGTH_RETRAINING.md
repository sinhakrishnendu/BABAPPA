# Variable-Length Normalized Retraining

BABAPPA now includes a storage-safe retraining plan for a second deployable branch-site model family using `conservative_branch_site_normalized_v2`.

## Why retrain

The first deployable package was trained on a narrower simulation envelope. Direct user MSAs with very different gene lengths can push raw length and raw site-index features outside the checkpoint feature distribution. BABAPPA now detects that condition and marks the run out of domain, but a better deployable model should be trained with:

- broader simulated `n_taxa` coverage,
- broader simulated `n_codons` coverage,
- normalized/log length features,
- relative site-position features rather than raw zero-based site indices,
- the same explicit simulator branch-site truth labels.

## New feature policy

`conservative_branch_site_normalized_v2` keeps conservative branch-site features but excludes:

- foreground identity columns,
- raw zero-based site index columns,
- raw `n_taxa`,
- raw `n_codons`.

It keeps derived features such as:

- `site_relative_position`,
- `site_centered_position`,
- `site_terminal_distance`,
- `log_n_taxa`,
- `log_n_codons`.

## Generate the plan

```bash
babappa plan-variable-length-100k-retraining \
  --outdir variable_length_retraining_plan \
  --workspace branch_site_v2_100k_workspace \
  --package-outdir deployable_model_conservative_branch_site_v2_100k_mps \
  --n-families-per-tier 25000 \
  --device mps \
  --threads 18 \
  --batch-size 64 \
  --min-free-gb 250
```

This writes scripts and manifests only; it does not start retraining.

## Run with storage cleanup

The generated script defaults to quarantine mode. Quarantine is safer but does not free disk if the quarantine stays on the same volume.

To actually free space after each completed chunk, run:

```bash
BABAPPA_RETRAIN_CLEANUP_MODE=delete \
BABAPPA_RETRAIN_DELETE_INTERMEDIATES=YES \
bash variable_length_retraining_plan/run_variable_length_100k_retraining.sh
```

The script deletes only reproducible intermediate directories after their downstream feature table or retained model artifact validates. It keeps model, calibration, threshold, aggregation, control, and summary artifacts.

## Monitor and validate

```bash
bash variable_length_retraining_plan/monitor_variable_length_100k_retraining.sh
bash variable_length_retraining_plan/validate_variable_length_100k_retraining.sh
```

After all tiers validate, package the new deployable model:

```bash
bash variable_length_retraining_plan/package_variable_length_deployable.sh
```

## Storage caveat

Even with cleanup enabled, this is a long 100K retraining job. Keep at least 250 GB free before each stage. For a 2 TB machine, do not run other large benchmarks at the same time.
