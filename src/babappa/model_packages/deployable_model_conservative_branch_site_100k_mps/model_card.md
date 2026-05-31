# BABAPPA conservative_branch_site 100K MPS model card

## Model name

`babappa_conservative_branch_site_100k_mps`

## Intended use

Research-alpha simulation-supervised scoring of branch-site candidates after BABAPPA input QC, alignment ensemble, conservative feature extraction, OOD/applicability checks, and simulation-matched calibration.

## Not intended use

Not final empirical branch-site inference. Not a replacement for empirical calibration, external benchmark panels, or domain review.

## Training data

Simulation-trained on the completed conservative explicit branch-truth 100K Apple Silicon/MPS validation run.

## Validation scale

100,000 simulated families across low, moderate, high, and extreme saturation tiers.

## Explicit branch-truth status

Validation used explicit simulator branch-site truth. No simulator truth is used during empirical inference.

## Feature policy

`conservative_branch_site`

## Supported aligners

`identity`, `mafft`, `babappalign`, `muscle`

## Saturation-tier behavior

Tier-aware package with low, moderate, high, and extreme checkpoints. Extreme remains strong but is the hardest tier.

## Performance table

| tier | AUROC | F1 | MCC | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| low | 0.998605 | 0.991482 | 0.986347 | 0.988048 | 0.994941 |
| moderate | 0.996962 | 0.986097 | 0.977704 | 0.982604 | 0.989616 |
| high | 0.994527 | 0.975462 | 0.960693 | 0.974995 | 0.975929 |
| extreme | 0.989955 | 0.956958 | 0.930738 | 0.945105 | 0.969113 |

## Calibration table

| tier | temperature | selected threshold | target FDR |
| --- | ---: | ---: | ---: |
| low | 1.000000 | 0.010000 | 0.100000 |
| moderate | 1.000000 | 0.022000 | 0.100000 |
| high | 1.050000 | 0.062000 | 0.100000 |
| extreme | 0.950000 | 0.214000 | 0.100000 |

## Controls interpretation

Destructive controls support that branch-label randomization degrades signal, but controls are simulation-supervised and do not by themselves establish empirical validity.

## Known limitations

- context_only_shortcut_high
- foreground_context_columns_present
- simulation_supervised_only
- conditional_pass_due_pruned_raw_intermediates

## Empirical-use warning

This package is simulation-trained and simulation-supervised. It is not final empirical branch-site inference.

## Required empirical workflow

input QC -> alignment ensemble -> feature extraction -> applicability/OOD check -> score -> simulation-matched calibration -> report

## Citation/manuscript placeholder

BABAPPA manuscript citation to be added after empirical benchmark validation.

## Version and checksums

- BABAPPA version: `0.4.4-alpha`
- Checksums: `checksums.sha256`
