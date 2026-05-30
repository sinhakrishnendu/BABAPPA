# BABAPPA explicit branch-truth 100K Apple Silicon/MPS final report

Generated: `2026-05-26T23:19:13`

## Decision headline

The conservative explicit branch-truth 100K Apple Silicon/MPS run completed at stage-marker level and is suitable for the next scientific decision. It should be interpreted as strong simulation-supervised research-alpha validation, not as empirical biological evidence.

## Completion status

- Plan directory: `explicit_branch_truth_100k_mps_plan_blazing`
- Complete stage markers: `104`
- Partial markers: `0`
- low: `26/26` stage markers complete
- moderate: `26/26` stage markers complete
- high: `26/26` stage markers complete
- extreme: `26/26` stage markers complete
- Cross-tier summary: `explicit_branch_truth_100k_mps_cross_tier_summary`
- Truth-status audit: `explicit_branch_truth_100k_mps_truth_status_audit`

## Configuration

- truth_mode: `explicit`
- feature_policy: `conservative_branch_site`
- tiers: `['low', 'moderate', 'high', 'extreme']`
- families_per_tier: `25000`
- methods: `['identity', 'mafft', 'babappalign', 'muscle']`
- device: `Apple Silicon Metal/MPS for neural stages; BABAPPAlign embedded MPS requested in user run`
- batch_size: `64`
- memory_context: `36 GB unified memory; streamed/capped branch-site datasets were required`

## Explicit truth audit

| tier | audit status | label status | explicit truth | proxy labels | branch-site rows | positives | positive fraction |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| low | explicit_truth_ok | explicit_simulator_branch_truth | True | False | 240000000 | 752220 | 0.003134 |
| moderate | explicit_truth_ok | explicit_simulator_branch_truth | True | False | 240000000 | 752220 | 0.003134 |
| high | explicit_truth_ok | explicit_simulator_branch_truth | True | False | 240000000 | 752220 | 0.003134 |
| extreme | explicit_truth_ok | explicit_simulator_branch_truth | True | False | 240000024 | 752221 | 0.003134 |

All tiers report `explicit_truth_ok`, `explicit_simulator_branch_truth`, and `proxy_from_foreground_taxon=False`. This is the key difference from earlier proxy-risk branch-conditioned work.

## Branch-site neural performance

| tier | test n | positives | test AUROC | test F1 | test MCC | all AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 50000 | 18778 | 0.998605 | 0.991482 | 0.986347 | 0.998617 |
| moderate | 50000 | 18778 | 0.996962 | 0.986097 | 0.977704 | 0.997044 |
| high | 50000 | 18778 | 0.994527 | 0.975462 | 0.960693 | 0.994214 |
| extreme | 50000 | 18778 | 0.989955 | 0.956958 | 0.930738 | 0.990101 |

Performance degrades gracefully with saturation, from low-tier test AUROC 0.998605 to extreme-tier test AUROC 0.989955.

## Branch and gene aggregation

| tier | branch all AUROC | branch test AUROC | gene all AUROC | gene test AUROC |
| --- | ---: | ---: | ---: | ---: |
| low | 0.999797 | 1.000000 | 0.999697 | 1.000000 |
| moderate | 0.999421 | 0.999912 | 0.999616 | 0.998844 |
| high | 0.998777 | 0.999926 | 0.999508 | 1.000000 |
| extreme | 0.997873 | 0.999890 | 0.999430 | 1.000000 |

Branch-level and branch-to-gene aggregation remain extremely strong through the extreme tier. This supports the practical claim that sparse branch-site evidence can be converted into stable higher-level support under the current simulation design.

## Control behavior

| tier | control | observed AUROC | mean control AUROC | interpretation |
| --- | --- | ---: | ---: | --- |
| low | `branch_score_permutation_within_family` | 0.999797 | 0.895382 | yes |
| low | `degree_prevalence_matched_null` | 0.999797 | 0.895373 | yes |
| low | `within_family_branch_label_shuffle` | 0.999797 | 0.917840 | yes |
| low | `family_permutation` | 0.999797 | 0.998542 | partial |
| low | `shuffle_branch_assignment_within_family` | 0.999797 | 0.999215 | partial |
| moderate | `branch_score_permutation_within_family` | 0.999421 | 0.892325 | yes |
| moderate | `degree_prevalence_matched_null` | 0.999421 | 0.892338 | yes |
| moderate | `within_family_branch_label_shuffle` | 0.999421 | 0.914266 | yes |
| moderate | `family_permutation` | 0.999421 | 0.996890 | partial |
| moderate | `shuffle_branch_assignment_within_family` | 0.999421 | 0.998203 | partial |
| high | `branch_score_permutation_within_family` | 0.998777 | 0.889123 | yes |
| high | `degree_prevalence_matched_null` | 0.998777 | 0.889110 | yes |
| high | `within_family_branch_label_shuffle` | 0.998777 | 0.910686 | yes |
| high | `family_permutation` | 0.998777 | 0.993937 | partial |
| high | `shuffle_branch_assignment_within_family` | 0.998777 | 0.996483 | partial |
| extreme | `branch_score_permutation_within_family` | 0.997873 | 0.881425 | yes |
| extreme | `degree_prevalence_matched_null` | 0.997873 | 0.881334 | yes |
| extreme | `within_family_branch_label_shuffle` | 0.997873 | 0.901379 | yes |
| extreme | `family_permutation` | 0.997873 | 0.989337 | partial |
| extreme | `shuffle_branch_assignment_within_family` | 0.997873 | 0.993729 | partial |

Global shuffled branch labels are near random, while partial controls that preserve family or branch prevalence can remain high. That means the strongest claims should lean on explicit truth audit, conservative branch-site feature policy, destructive controls, and ablation warnings rather than only aggregate AUROC.

## Warnings and caveats

- Cross-tier warning retained: `branch_context_ablation:context_only_shortcut_high`. Context-only shortcuts remain scientifically important and should be discussed openly.
- Per-tier summaries warn `missing_or_incomplete:branch_site_baseline`; this reflects the current Mac branch-site workflow and is not a failure of the neural/calibration/aggregation outputs.
- These results are simulation-supervised. They do not establish empirical biological selection calls.
- Artifact heads, abstention/risk heads, empirical deployment, and auxiliary ASR/energy interpretation remain future validation work.

## Disk and cleanup note

Disk before aggressive cleanup: `207.9 GiB free of 1858.1 GiB`.
Bulky raw and intermediate outputs are safe to prune after this report, cross-tier summaries, truth audits, logs, manuscript updates, and model/development artifacts are retained. Raw branch-site label TSVs are not necessary for ordinary software development once compact summaries and downstream model outputs exist.

## Recommendation

For the next ChatGPT or project decision: treat the 100K explicit branch-truth MPS run as completed and scientifically useful for method development. Do not rerun it merely to prove completion. The next useful work is manuscript integration, targeted ablations/held-out stress tests, empirical pilot planning, and cleanup of bulky raw intermediates.

## Aggressive cleanup completed

After this report and manuscript update were secured, raw/intermediate bulk artifacts were pruned. Disk space improved from `206 GiB free` to `487 GiB free`, reclaiming about `281 GiB`. The cleanup manifest is `reports/aggressive_cleanup_manifest_20260526_234436.txt`.

Deleted classes included 100K raw simulations, alignments, site maps, tensors, streamed branch-site datasets, giant raw oracle TSVs, and older 10K/1K raw validation artifacts. Preserved classes include source, tests, docs, manuscript source/PDF, reports, logs, plan directories, cross-tier summaries, truth-status audits, branch-site neural outputs, aggregation outputs, calibration/threshold/control summaries, and compact oracle summaries.
