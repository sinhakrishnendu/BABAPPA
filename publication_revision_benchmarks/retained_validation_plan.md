# Fully Retained Validation Plan

Benchmark: `BABAPPA-RETAINED-VALIDATION-v1`

Families: 10000

Purpose: address the conditional-pass reproducibility concern with a smaller fully retained known-truth validation profile.

Retained artifacts:
- `simulation_manifest`
- `family_truth_files`
- `input_cds`
- `tree_files`
- `feature_tables`
- `scores`
- `applicability_reports`
- `threshold_policy_outputs`
- `evaluation_tables`
- `checksums`

Primary metrics:
- AUROC
- AUPRC
- FDR
- MCC
- OOD false-call rate
- non-OOD positive recall

This profile is designed to be small enough to archive completely while large enough to audit the main known-truth operating claims.
