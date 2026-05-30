# Explicit branch-truth 100K MPS final validation report

## Executive decision

**CONDITIONAL PASS**: The 100K simulation-supervised validation passes on retained summaries, model artifacts, truth audits, method policies, and stage markers; raw/intermediate directories were pruned after completion and cannot be directly revalidated.

## Run identity

- run_name: `explicit_branch_truth_100k_mps`
- platform: `macOS-26.5-arm64-arm-64bit-Mach-O`
- machine: `arm64`
- device: `mps`
- tiers: `['low', 'moderate', 'high', 'extreme']`
- families_per_tier: `25000`
- total_families: `100000`
- methods: `['identity', 'mafft', 'babappalign', 'muscle']`
- truth_mode: `explicit`
- feature_policy: `conservative_branch_site`
- babappalign_device: `mps`
- babappalign_workers: `4`
- mps_batch_size: `64`
- total_runtime: `not_computed_from_logs`
- disk: `{'total_gib': 1858.1, 'used_gib': 1370.8, 'free_gib': 487.4}`

## Validation completeness

- complete stage markers: `104`
- partial markers: `0`
- raw/intermediate validator status: pruned raw simulation/alignment/tensor/branch-site-dataset directories are recorded as archival notes, not silent passes.

## Simulation and truth

| tier | audit | label status | explicit truth | proxy labels | rows | positives |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| low | explicit_truth_ok | explicit_simulator_branch_truth | True | False | 240000000 | 752220 |
| moderate | explicit_truth_ok | explicit_simulator_branch_truth | True | False | 240000000 | 752220 |
| high | explicit_truth_ok | explicit_simulator_branch_truth | True | False | 240000000 | 752220 |
| extreme | explicit_truth_ok | explicit_simulator_branch_truth | True | False | 240000024 | 752221 |

## Alignment and method policy

- low: usable `babappalign,identity,mafft,muscle`; quarantined `none`
- moderate: usable `babappalign,identity,mafft,muscle`; quarantined `none`
- high: usable `babappalign,identity,mafft,muscle`; quarantined `none`
- extreme: usable `babappalign,identity,mafft,muscle`; quarantined `none`

## Branch-site neural performance

| tier | test AUROC | precision | recall | F1 | MCC | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 0.998605 | 0.988048 | 0.994941 | 0.991482 | 0.986347 | 0.993580 |
| moderate | 0.996962 | 0.982604 | 0.989616 | 0.986097 | 0.977704 | 0.989520 |
| high | 0.994527 | 0.974995 | 0.975929 | 0.975462 | 0.960693 | 0.981560 |
| extreme | 0.989955 | 0.945105 | 0.969113 | 0.956958 | 0.930738 | 0.967260 |

## Calibration

| tier | temperature | selected threshold | raw ECE | calibrated ECE | warnings |
| --- | ---: | ---: | ---: | ---: | --- |
| low | 1.000000 | 0.010000 | 0.003860 | 0.003860 |  |
| moderate | 1.000000 | 0.022000 | 0.003588 | 0.003588 |  |
| high | 1.050000 | 0.062000 | 0.004269 | 0.003559 |  |
| extreme | 0.950000 | 0.214000 | 0.016840 | 0.016672 |  |

## Aggregation

| tier | branch all AUROC | gene all AUROC | branch rows | gene rows |
| --- | ---: | ---: | ---: | ---: |
| low | 0.999797 | 0.999697 | 173654 | 63099 |
| moderate | 0.999421 | 0.999616 | 173654 | 63099 |
| high | 0.998777 | 0.999508 | 173654 | 63099 |
| extreme | 0.997873 | 0.999430 | 173736 | 63145 |

## Controls

Destructive controls support that branch-label randomization collapses toward random, while partial prevalence-preserving controls can remain high and should not be overinterpreted.

| tier | control | observed AUROC | mean AUROC | destructive enough |
| --- | --- | ---: | ---: | --- |
| low | `branch_score_permutation_within_family` | 0.999797 | 0.895382 | yes |
| low | `degree_prevalence_matched_null` | 0.999797 | 0.895373 | yes |
| low | `within_family_branch_label_shuffle` | 0.999797 | 0.917840 | yes |
| moderate | `branch_score_permutation_within_family` | 0.999421 | 0.892325 | yes |
| moderate | `degree_prevalence_matched_null` | 0.999421 | 0.892338 | yes |
| moderate | `within_family_branch_label_shuffle` | 0.999421 | 0.914266 | yes |
| high | `branch_score_permutation_within_family` | 0.998777 | 0.889123 | yes |
| high | `degree_prevalence_matched_null` | 0.998777 | 0.889110 | yes |
| high | `within_family_branch_label_shuffle` | 0.998777 | 0.910686 | yes |
| extreme | `branch_score_permutation_within_family` | 0.997873 | 0.881425 | yes |
| extreme | `degree_prevalence_matched_null` | 0.997873 | 0.881334 | yes |
| extreme | `within_family_branch_label_shuffle` | 0.997873 | 0.901379 | yes |

## Leakage, OOD, and scientific cautions

- simulation_supervised_only
- no_final_empirical_inference_claim
- foreground_context_columns_present in leakage audits
- branch_context_ablation:context_only_shortcut_high carried forward as policy caution
- raw simulation/alignment/tensor/branch-site-dataset trees were pruned after completed validation; preserved summaries and model artifacts remain

## Recommendation

- package_deployable_model: yes
- empirical_mode_scaffolding: yes
- empirical_inference_claims: no
- next_work: Package conservative_branch_site tier-aware 100K models, add simulation-matched empirical calibration, and build OOD/applicability gates before empirical claims.
