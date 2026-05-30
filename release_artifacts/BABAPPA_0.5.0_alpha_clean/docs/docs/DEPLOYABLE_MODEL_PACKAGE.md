# Deployable Model Package

BABAPPA `0.4.8-alpha` can package the retained conservative explicit branch-truth 100K MPS model artifacts into a research-alpha deployable bundle and use it in the guarded empirical bridge and small diagnostic empirical pilot-panel workflow.

The Cycle 39 100K result was a conditional pass: retained summaries, explicit truth audit, model artifacts, calibration, aggregation, controls, and stage markers support the completed run, but raw/intermediate artifacts were intentionally pruned after validation. That makes the package suitable for controlled simulation-trained deployment scaffolding, not final empirical branch-site inference.

## Package Command

```bash
babappa package-deployable-model --run-name explicit_branch_truth_100k_mps --model-dirs branch_site_neural_explicit_branch_truth_100k_mps_low_streamed,branch_site_neural_explicit_branch_truth_100k_mps_moderate_streamed,branch_site_neural_explicit_branch_truth_100k_mps_high_streamed,branch_site_neural_explicit_branch_truth_100k_mps_extreme_streamed --calibration-dirs branch_site_calibration_explicit_branch_truth_100k_mps_low_streamed,branch_site_calibration_explicit_branch_truth_100k_mps_moderate_streamed,branch_site_calibration_explicit_branch_truth_100k_mps_high_streamed,branch_site_calibration_explicit_branch_truth_100k_mps_extreme_streamed --truth-audit-dir branch_truth_status_audit_explicit_branch_truth_100k_mps --validation-report explicit_branch_truth_100k_mps_final_validation_report.json --feature-policy conservative_branch_site --truth-mode explicit --methods identity,mafft,babappalign,muscle --outdir deployable_model_conservative_branch_site_100k_mps
```

## Package Contents

- `model_manifest.json`
- `model_card.md`
- `feature_schema.json`
- `calibration_schema.json`
- `training_envelope.json`
- `tier_models/`
- `tier_calibrations/`
- `checksums.sha256`
- `validation_summary.json`
- `limitations.md`
- `README.md`

The package copies only lightweight checkpoint, model metadata, history, metrics, and calibration metadata files. It excludes raw simulations, alignments, branch-site datasets, simulator truth, and oracle labels.

## Validate

```bash
babappa validate-deployable-model-package --package-dir deployable_model_conservative_branch_site_100k_mps
```

The validator checks manifest/card/schema presence, checksums, feature policy, explicit truth mode, no proxy labels, supported methods, PRANK/T-Coffee exclusion from production defaults, known warnings, and the simulation-supervised empirical claim boundary.

## Smoke Load

```bash
babappa smoke-load-deployable-model --package-dir deployable_model_conservative_branch_site_100k_mps --device auto --outdir deployable_model_load_smoke
```

When PyTorch is available this loads each tier model and performs a tiny synthetic forward pass. If PyTorch is unavailable, the smoke records metadata-only mode with a clear warning.

## Claim Boundary

This package is simulation-trained. It used explicit simulator branch-site truth during validation. It must not consume simulator truth during empirical inference and it is not final empirical branch-site inference. Empirical use requires input QC, OOD/applicability gates, simulation-matched calibration, and external benchmark panels.

## Cycle 42 Empirical Pilot Panels

After package validation, use `babappa validate-empirical-pilot-panel` and `babappa run-empirical-pilot-panel` on a small curated manifest. The pilot runner chains input QC, empirical alignment ensemble, feature extraction, feature audit, applicability/OOD gating, model scoring, simulation-matched calibration planning, and guarded reporting per family. Classical codeml/HyPhy workflows are planned with `babappa plan-classical-reference-workflows`; they are not executed by BABAPPA. Pilot summaries remain diagnostic and explicitly avoid empirical discovery claims.

Cycle 43 adds `babappa prepare-real-empirical-pilot-workspace` for a real-data manifest template and readiness report. Missing real CDS/tree inputs block the pilot run by design.

## Cycle 41 Empirical Bridge

The empirical bridge runs:

```bash
babappa validate-empirical-input ...
babappa run-empirical-alignment-ensemble ...
babappa extract-empirical-branch-site-features ...
babappa audit-empirical-features ...
babappa empirical-applicability ...
babappa score-empirical-branch-sites ...
babappa plan-simulation-matched-calibration ...
babappa make-empirical-branch-site-report ...
```

Scoring requires PyTorch. If PyTorch is unavailable, BABAPPA fails clearly rather than producing metadata-only scores.
