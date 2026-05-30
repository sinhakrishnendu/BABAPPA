# BABAPPA explicit branch-truth 10K MPS handoff report for 100K decision

## Copy-paste instruction for ChatGPT

Use this report to decide whether BABAPPA should proceed from conservative explicit branch-truth 10K MPS validation to a guarded 100K MPS run. Treat the evidence as simulation-supervised research-alpha validation, not empirical inference.

## Executive recommendation

**Recommendation: GO for 100K, but only with the existing Apple Silicon safeguards.**

Rationale: the conservative explicit branch-truth 10K MPS run completed all tiers, all core stages, all production-fast aligners, explicit branch-site truth checks, neural training, aggregation, controls, and summaries. No scientific artifact blocker was found. The remaining cautions affect interpretation and operational risk, not whether the 100K validation run is justified.

## Run identity

| Field | Value |
| --- | --- |
| Run | `explicit_branch_truth_10k_mps` |
| BABAPPA version | `0.4.2-alpha` |
| Platform target | Apple Silicon Metal/MPS, 36 GB unified memory |
| Tiers | low, moderate, high, extreme |
| Families per tier | 2500 |
| Total families | 10000 |
| Methods | identity, mafft, babappalign, muscle |
| Feature policy | `conservative_branch_site` |
| Truth mode | `explicit` |
| Neural device | `mps` |
| BABAPPAlign backend/device | embedded / `mps` |
| BABAPPAlign effective workers | 4 |
| CPU workers | 18 |
| MPS batch size | 128 |
| Main log | `logs/explicit_branch_truth_10k_mps_20260523_212935.log` |
| Runtime from log stat | 2026-05-23 21:29:35 IST to 2026-05-24 03:13:17 IST (~5h 43m 41s) |
| Stage markers | 104 complete, 0 partial |
| Selected 10K output size | 39.2 GB |
| Current free disk | ~1.4 TiB from `df -h .` |

## Completion and validation status

- All expected top-level stage markers are present for all four tiers.
- No `.partial` markers remain.
- Generated cross-tier summary validation: OK, with one warning (`branch_context_ablation:context_only_shortcut_high`).
- Cross-tier truth-status audit validation: OK, 0 warnings.
- Generated per-tier validate script note: it exits nonzero when validating a single-tier truth audit because that validator expects all tiers. The cross-tier truth-status audit validates OK, and this is treated as a validation-script granularity issue rather than a run artifact failure.

## Tier-level simulation and truth

| Tier | Families OK/fail | Branch truth status | Branch-site truth rows | Branch positive rows | Mean p-distance |
| --- | ---: | --- | ---: | ---: | ---: |
| low | 2500/0 | `explicit_truth_ok` | 6000000 | 18330 | 0.021733 |
| moderate | 2500/0 | `explicit_truth_ok` | 6000000 | 18330 | 0.040877 |
| high | 2500/0 | `explicit_truth_ok` | 6000000 | 18330 | 0.077638 |
| extreme | 2500/0 | `explicit_truth_ok` | 6000000 | 18330 | 0.145486 |

## Explicit branch-site labels

| Tier | Label status | Proxy used | Label rows | Positive rows | Positive fraction |
| --- | --- | --- | ---: | ---: | ---: |
| low | `explicit_simulator_branch_truth` | False | 24000000 | 73320 | 0.003055 |
| moderate | `explicit_simulator_branch_truth` | False | 24000000 | 73320 | 0.003055 |
| high | `explicit_simulator_branch_truth` | False | 24000000 | 73320 | 0.003055 |
| extreme | `explicit_simulator_branch_truth` | False | 24000008 | 73321 | 0.003055 |

Cross-tier truth audit says every audited tier uses `direct_simulator_branch_site_truth`; proxy labels were not detected.

## Alignment and method policy

| Tier | Family-method OK/fail | Methods run | Quarantined methods | BABAPPAlign | Warnings |
| --- | ---: | --- | --- | --- | --- |
| low | 10000/0 | identity, mafft, babappalign, muscle | {} | embedded/mps, workers=4 | [] |
| moderate | 10000/0 | identity, mafft, babappalign, muscle | {} | embedded/mps, workers=4 | [] |
| high | 10000/0 | identity, mafft, babappalign, muscle | {} | embedded/mps, workers=4 | [] |
| extreme | 10000/0 | identity, mafft, babappalign, muscle | {} | embedded/mps, workers=4 | [] |

All four production-fast methods passed policy thresholds in every tier: identity, mafft, babappalign, muscle. No PRANK/T-Coffee were used.

## Branch-site dataset after downsampling/capping

| Tier | Rows | Positive rows | Positive fraction | Dataset warnings |
| --- | ---: | ---: | ---: | --- |
| low | 438913 | 73320 | 0.167049 | [] |
| moderate | 438913 | 73320 | 0.167049 | [] |
| high | 438913 | 73320 | 0.167049 | [] |
| extreme | 438733 | 73319 | 0.167115 | [] |

## Conservative branch-site neural performance

| Tier | Device | Epochs/best | Test n | Test positives | Test AUROC | Test accuracy | Test precision | Test recall | Test F1 | Test MCC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low | `mps` | 10/7 | 23378 | 3960 | 0.998403 | 0.993584 | 0.965315 | 0.99798 | 0.981376 | 0.977699 |
| moderate | `mps` | 10/10 | 23378 | 3960 | 0.995264 | 0.989862 | 0.958385 | 0.982828 | 0.970453 | 0.964447 |
| high | `mps` | 10/10 | 23378 | 3960 | 0.993223 | 0.984002 | 0.931217 | 0.977778 | 0.95393 | 0.944668 |
| extreme | `mps` | 10/9 | 23378 | 3960 | 0.989422 | 0.959321 | 0.818818 | 0.975758 | 0.890425 | 0.870701 |

Saturation degradation exists but remains controlled: test AUROC drops from ~0.9984 in low to ~0.9894 in extreme.

## Branch and gene aggregation

| Tier | Branch all AUROC | Branch test AUROC | Gene all AUROC | Gene test AUROC | Branch rows | Gene rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 1 | 1 | 1 | 1 | 75297 | 9993 |
| moderate | 0.999979 | 1 | 1 | 1 | 75297 | 9993 |
| high | 0.999985 | 1 | 1 | 1 | 75297 | 9993 |
| extreme | 0.999876 | 0.999963 | 1 | 1 | 75341 | 9993 |

## Controls

| Tier | Observed AUROC | Shuffle branch labels mean | Branch-score permutation mean | Degree/prevalence null mean | Within-family branch-label shuffle mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| low | 1 | 0.499547 | 0.761833 | 0.762108 | 0.911042 |
| moderate | 0.999979 | 0.499429 | 0.76309 | 0.762948 | 0.913942 |
| high | 0.999985 | 0.499552 | 0.755494 | 0.755673 | 0.907749 |
| extreme | 0.999876 | 0.499427 | 0.748251 | 0.748123 | 0.902119 |

Interpretation: the strongest destructive controls degrade substantially, especially global branch-label shuffle (~0.499 AUROC). Some partial controls remain high, which is expected when family/prevalence structure is preserved; these are cautions for interpretation, not blockers for 100K validation.

## Calibration and thresholds

| Tier | Temperature | Selected threshold | Warnings |
| --- | ---: | ---: | --- |
| low | 0.95 | 0.42 | [] |
| moderate | 0.95 | 0.134 | [] |
| high | 0.95 | 0.34 | [] |
| extreme | 1.1 | 0.828 | [] |

## Warnings and limitations

- `foreground_context_columns_present` appears in branch-site leakage audits for every tier. Forbidden label columns and near-perfect univariate columns were not detected, but this remains an interpretation caution.
- Cross-tier summary carries `branch_context_ablation:context_only_shortcut_high`; foreground/context-only shortcut risk remains scientifically important.
- Per-tier run summaries warn `missing_or_incomplete:branch_site_baseline`; the 10K Mac MPS workflow used branch-site neural, calibration, aggregation, and controls, not the older baseline stage.
- The generated validate script has a per-tier truth-audit validation granularity issue, as noted above; the cross-tier truth audit validates OK.
- This is simulation-supervised research-alpha validation. It supports 100K validation, not final empirical branch-site inference claims.

## 100K operational forecast

- 10K main run runtime from log stat was about 5h 43m on this machine. A 100K run is expected to take roughly 10x longer, likely multiple days depending on thermal/load behavior.
- Selected 10K output directories occupy about 39.2 GB. A 100K run could plausibly use several hundred GB, with branch-site oracle labels dominating disk use.
- Current free disk is about 1.4 TiB, which is operationally adequate if the 100K plan remains streamed/capped and does not duplicate failed partial outputs.
- Keep `BABAPPA_BABAPPALIGN_WORKERS=4`, `BABAPPA_MPS_BATCH_SIZE=64` for 100K as planned, and run tier-by-tier with markers/resume.

## Decision recommendation for ChatGPT

Decision: **Proceed with 100K MPS validation under safeguards.**

Do not proceed if the goal is final empirical inference; proceed if the goal is larger-scale conservative explicit branch-truth validation. There are no run-completion or artifact-integrity blockers in the 10K results. The main scientific caveat is context/foreground shortcut risk, which should be handled by conservative feature policy, controls, and careful interpretation rather than by blocking the 100K validation run.

## Required 100K safeguards

- Keep conservative_branch_site as production feature policy.
- Keep truth-mode explicit.
- Keep methods identity, mafft, babappalign, muscle only; PRANK/T-Coffee diagnostic only.
- Keep BABAPPAlign embedded MPS worker cap at 4 on 36 GB unified memory unless benchmarked otherwise.
- Run 100K tier-by-tier, never monolithically; keep stage markers and memory guard.
- Do not interpret as empirical branch-site inference; this is simulation-supervised research-alpha evidence.
- Preserve and review context-only shortcut warning and foreground-context leakage warning when writing scientific claims.

## Suggested 100K go/no-go answer

If asked for a concise decision: **GO for 100K as a guarded validation run. Do not use it as final empirical claim generation. Keep explicit truth, conservative_branch_site, streamed/capped outputs, 4 BABAPPAlign MPS workers, and tier-by-tier resume.**
