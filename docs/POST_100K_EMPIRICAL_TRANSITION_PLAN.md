# BABAPPA Post-100K Empirical Transition Plan

## Status

The conservative explicit branch-truth 100K Apple Silicon/MPS validation supports BABAPPA as a simulation-trained research-alpha method. It does not establish final empirical branch-site inference.

## What 100K Proves

- The explicit simulator branch-site truth pathway scales to 100,000 families.
- The production-fast method set `identity,mafft,babappalign,muscle` can support the conservative branch-site workflow.
- The `conservative_branch_site` feature policy remains learnable across low, moderate, high, and extreme saturation tiers.
- Tier-aware branch-site neural checkpoints, calibration artifacts, aggregation outputs, and destructive controls are available for deployable-model packaging.

## What 100K Does Not Prove

- It does not prove empirical biological positive-selection calls.
- It does not validate hidden paralogy, annotation-error, recombination-rich empirical panels, or out-of-distribution taxon/length/composition regimes.
- It does not replace empirical calibration, OOD gating, or benchmarking against established codon-model workflows.
- It does not remove the `context_only_shortcut_high` caution; that caveat must remain visible in manuscripts and model cards.

## Deployable Model Packaging Requirements

- Package all four 100K tier checkpoints with matching model metadata and calibration artifacts.
- Include the 100K cross-tier summary and explicit truth-status audit.
- Include a model card stating simulation-supervised status, feature policy, truth mode, aligner set, tier-selection logic, and empirical claim boundaries.
- Validate package integrity without retraining.
- The Cycle 40 package target is `deployable_model_conservative_branch_site_100k_mps`; it must exclude raw simulator truth, raw oracle labels, raw branch-site datasets, simulations, and alignments.
- The package validator must keep `empirical_claim_status` at `not_final_empirical_inference`.

## Empirical Input Validation

- Require codon-valid FASTA input, taxon naming consistency, no frameshifts, no premature stop codons unless explicitly allowed, and method-specific alignment QC.
- Record alignment method provenance and BABAPPAlign model-cache status.
- Reject or quarantine inputs that fail site-map, frame, or method-policy checks.
- Cycle 41 implements the first tiny empirical bridge: `validate-empirical-input`, `run-empirical-alignment-ensemble`, `extract-empirical-branch-site-features`, `audit-empirical-features`, `empirical-applicability`, `score-empirical-branch-sites`, and `make-empirical-branch-site-report`.

## Simulation-Matched Calibration

- Start empirical scoring with tier-aware 100K calibration artifacts.
- Add a simulation-matched calibration report that compares empirical feature distributions against the 100K training/evaluation distribution.
- Treat calibration mismatch as an abstention or low-applicability warning, not as a positive-selection result.
- Use `babappa plan-simulation-matched-calibration` to create USER-RUN ONLY proposed null-simulation commands from empirical QC summaries; this planner does not run heavy simulations.
- Cycle 41 upgrades the planner to use real empirical QC fields when available and to write both null and optional alternative simulation command templates.

## OOD and Applicability Gate

- Build an applicability layer over taxon count, codon length, gap burden, saturation proxies, branch/site mappability, feature distribution shift, and alignment-method disagreement.
- Report `in_domain`, `borderline`, or `out_of_domain` before emitting branch-site conclusions.
- Out-of-domain families should produce diagnostic output only.

## Benchmarking Against Codon-Model Workflows

- Compare against CODEML-style branch-site tests and HyPhy-style workflows where appropriate.
- Evaluate concordant positives, BABAPPA-only positives, classical-only positives, and abstained/low-applicability cases separately.
- Do not tune BABAPPA to imitate classical methods; use them as external reference points and failure-mode probes.
- Use `babappa plan-external-benchmark-panel` to generate USER-RUN ONLY BABAPPA, codeml, and HyPhy command templates before any benchmark execution.

## Real Case-Study Panel

- Use a small, curated empirical pilot panel before any broad scan.
- Cycle 42 implements the small curated empirical pilot-panel framework, reference workflow planning for codeml/HyPhy-style tools, BABAPPA/reference comparison summaries, and claim-boundary validation.
- Cycle 43 creates a real empirical pilot workspace/template and readiness/decision reports. Missing real data produce `NEED_INPUT_REPAIR`, not fabricated data or a forced run.
- Include known positives, likely negatives, saturated families, alignment-difficult families, and families with literature-supported mechanistic interpretation.
- Preserve all empirical preprocessing, alignment, site-map, model, calibration, and report artifacts.

## Manuscript Claim Boundary

The current defensible claim is: BABAPPA has passed large-scale simulation-supervised explicit branch-site validation under conservative feature policy and production-fast alignment workflows.

The current non-defensible claim is: BABAPPA has proven empirical branch-site selection inference on real biological datasets.

In short, BABAPPA remains not final empirical branch-site inference until empirical calibration, OOD gating, and external benchmark validation are complete.

## Recommended Next Cycle

Package the tier-aware conservative 100K model family, implement simulation-matched empirical calibration, add OOD/applicability gating, and run a small empirical pilot with strict claim boundaries.

## Cycle 42 Pilot Boundary

The pilot panel is diagnostic. It may reveal input-QC failures, alignment sensitivity, OOD behavior, calibration gaps, or disagreement with codeml/HyPhy-style references. It must not be framed as empirical discovery or final branch-site inference until calibration and external biological interpretation are complete.
