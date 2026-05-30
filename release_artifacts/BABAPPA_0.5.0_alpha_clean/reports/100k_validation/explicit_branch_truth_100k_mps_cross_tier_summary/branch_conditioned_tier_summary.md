# BABAPPA explicit branch-truth 100K cross-tier summary

## Executive conclusion

Conservative explicit branch-truth workflow is technically validated at 100K scale. Extreme-tier performance remains strong but reduced relative to low/moderate tiers. Branch-level and gene-level aggregation are strong. Results support moving from bulk simulation validation to targeted ablation, empirical pilot, and manuscript integration work.

## Scientific boundary

This is simulation-supervised branch-conditioned research-alpha validation using direct explicit simulator branch-site truth where available. It is not final empirical branch-site inference.

## Completed tiers

- low: complete (label status: `explicit_simulator_branch_truth`)
- moderate: complete (label status: `explicit_simulator_branch_truth`)
- high: complete (label status: `explicit_simulator_branch_truth`)
- extreme: complete (label status: `explicit_simulator_branch_truth`)

## Branch-site neural performance

| Tier | n | positives | test AUROC |
| --- | ---: | ---: | ---: |
| low | 50000 | 18778 | 0.998605 |
| moderate | 50000 | 18778 | 0.996962 |
| high | 50000 | 18778 | 0.994527 |
| extreme | 50000 | 18778 | 0.989955 |

## Branch-level aggregation

| Tier | n | positives | all AUROC |
| --- | ---: | ---: | ---: |
| low | 173654 | 44613 | 0.999797 |
| moderate | 173654 | 44613 | 0.999421 |
| high | 173654 | 44613 | 0.998777 |
| extreme | 173736 | 44649 | 0.997873 |

## Branch-to-gene aggregation

| Tier | n | positives | all AUROC |
| --- | ---: | ---: | ---: |
| low | 63099 | 46422 | 0.999697 |
| moderate | 63099 | 46422 | 0.999616 |
| high | 63099 | 46422 | 0.999508 |
| extreme | 63145 | 46475 | 0.99943 |

## Calibration and threshold-policy behavior

Calibration and threshold-policy artifacts are summarized where present. Missing calibration or policy artifacts are warnings rather than summary failures because they are optional for the cross-tier audit layer.

- low: temperature `1`, selected threshold `0.01`, branch-site profiles `7`, aggregation profiles `7`
- moderate: temperature `1`, selected threshold `0.022`, branch-site profiles `7`, aggregation profiles `7`
- high: temperature `1.05`, selected threshold `0.062`, branch-site profiles `7`, aggregation profiles `7`
- extreme: temperature `0.95`, selected threshold `0.214`, branch-site profiles `7`, aggregation profiles `7`

## Branch aggregation controls

- low: observed branch-control AUROC `0.999797`
- moderate: observed branch-control AUROC `0.999421`
- high: observed branch-control AUROC `0.998777`
- extreme: observed branch-control AUROC `0.997873`

## Saturation robustness

Low-to-extreme degradation is visible at branch-site neural level (0.998605 to 0.989955), while branch and gene aggregation remain strong.

## Aligner-policy inheritance

This summary reads completed branch-conditioned 100K outputs and inherits the production-fast aligner-policy context: identity, MAFFT, BABAPPAlign, and MUSCLE. It does not run alignments, neural training, 10K generation, or 100K generation.

## Label-truth status

No proxy label tiers were detected in the summary metadata.

## Branch feature policy

- Recommended branch feature policy: `conservative_branch_site`
- Full-context/full_model performance is treated as a context-aware upper-bound, not the main conservative branch-site claim.
- Warning: `context_only_shortcut_high`; context-only features are highly predictive.
- Ablation summary: `branch_context_ablation_explicit_1k_summary`

## Limitations

- Simulation-supervised research-alpha evidence only.
- Explicit simulator branch-site truth is present in the summary metadata; empirical branch-site claims still require empirical deployment and calibration.
- Branch-site full-context models are upper-bound diagnostics when foreground/context-only features are highly predictive.
- 100K explicit branch-truth validation is complete for this simulation-supervised decision point.

## Recommended next step

Use the completed 100K explicit branch-truth validation as the current research-alpha baseline; next prioritize targeted ablations, empirical pilot design, artifact/abstention heads, and manuscript integration.
