# Publication Revision Benchmark Plan

This document records the remaining validation work needed to strengthen BABAPPA from a conservative simulation-trained method paper into a stronger empirical methods submission. It does not report completed results unless the corresponding data files have been filled and analyses have actually been run.

## Why This Layer Exists

The current evidence is strong under explicit simulation truth and independent known-truth validation. The weakest remaining scientific point is empirical transfer: a reader still needs to see how BABAPPA behaves on literature-supported biological positives, matched negatives, and empirical stress cases.

The revision benchmark layer therefore adds four planned analyses:

1. Known biological positive controls.
2. Simulator-to-real transfer tests.
3. Sensitivity analysis for key thresholds and model choices.
4. A smaller fully retained validation profile to address the conditional 100K pass.

Generate the plan with:

```bash
babappa plan-publication-revision-benchmarks \
  --outdir publication_revision_benchmarks \
  --retained-validation-families 10000 \
  --null-replicates 1000 \
  --threads 8 \
  --device auto
```

## Generated Outputs

The command writes:

- `known_positive_control_panel_template.tsv`
- `empirical_transfer_panel_template.tsv`
- `sensitivity_analysis_grid.tsv`
- `revision_response_matrix.tsv`
- `retained_validation_plan.json`
- `retained_validation_plan.md`
- `publication_revision_plan.json`
- `publication_revision_plan.md`
- `scripts/run_known_positive_controls.sh`
- `scripts/run_empirical_transfer_panel.sh`
- `scripts/run_sensitivity_analysis.sh`
- `scripts/run_retained_validation_profile.sh`
- `scripts/summarize_revision_benchmarks.sh`

These scripts are reviewable long-run scripts. They are not executed by the planner.

## Known Positive Controls

The positive-control template intentionally contains placeholders rather than fabricated accessions. A usable positive control should have:

- a defensible codon MSA;
- a tree with matching labels;
- a justified foreground branch or clade;
- literature or experimental evidence for episodic selection;
- expected selected regions or sites where available;
- matched negative controls if possible.

Possible categories include compact viral envelope panels, influenza HA-like antigenic panels, and plant NLR/R-gene families. These should be used only when homology, foreground labeling, and biological interpretation are defensible.

## Simulator-to-Real Transfer

The transfer panel should include:

- at least one literature-supported positive family;
- at least one conserved negative family;
- at least one OOD/stress family expected to be rejected or marked diagnostic only.

BABAPPA should not be declared empirically discovery-ready until this panel is interpretable.

## Sensitivity Analysis

The sensitivity grid covers:

- score threshold;
- p-distance/tier boundaries;
- calibration temperature;
- architecture width;
- training seed.

The purpose is not to optimize after seeing empirical examples. The purpose is to show whether the main claims are robust to reasonable perturbations.

## Fully Retained Validation Profile

The 100K validation completed with a conditional pass because heavy raw intermediates were pruned. The retained-validation plan proposes a smaller profile, recommended at 10,000 families, where compact inputs, truth files, feature tables, score tables, summaries, and checksums are retained and archived.

This does not replace the 100K result. It repairs auditability for skeptical readers.

## Manuscript Use

Until these analyses are completed, the manuscript should say:

- known positive-control evaluation is planned or in progress;
- simulator-to-real transfer remains a limitation;
- hyperparameter sensitivity is planned or should be interpreted cautiously;
- the 100K result is a conditional pass with retained summaries, and a smaller fully retained profile is the reproducibility repair path.

Do not state that empirical positive-control recovery or transfer validation has been completed unless real output files exist.
