# BABAPPA branch truth-status audit

## Scientific boundary

All audited tiers report explicit branch-site truth availability with status explicit_truth_ok.

## Truth-status by tier

| Tier | Audit status | Label status | Explicit truth | Proxy labels | Rows | Positives | Derivation |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| low | `explicit_truth_ok` | `explicit_simulator_branch_truth` | True | False | 240000000 | 752220 | `direct_simulator_branch_site_truth` |
| moderate | `explicit_truth_ok` | `explicit_simulator_branch_truth` | True | False | 240000000 | 752220 | `direct_simulator_branch_site_truth` |
| high | `explicit_truth_ok` | `explicit_simulator_branch_truth` | True | False | 240000000 | 752220 | `direct_simulator_branch_site_truth` |
| extreme | `explicit_truth_ok` | `explicit_simulator_branch_truth` | True | False | 240000024 | 752221 | `direct_simulator_branch_site_truth` |

## Branch IDs and taxa

- low: branch IDs `taxon_001, taxon_002, taxon_003, taxon_004, taxon_005, taxon_006, taxon_007, taxon_008`; foreground taxa `unavailable`
- moderate: branch IDs `taxon_001, taxon_002, taxon_003, taxon_004, taxon_005, taxon_006, taxon_007, taxon_008`; foreground taxa `unavailable`
- high: branch IDs `taxon_001, taxon_002, taxon_003, taxon_004, taxon_005, taxon_006, taxon_007, taxon_008`; foreground taxa `unavailable`
- extreme: branch IDs `taxon_001, taxon_002, taxon_003, taxon_004, taxon_005, taxon_006, taxon_007, taxon_008`; foreground taxa `unavailable`

## Recommendation

Explicit simulator branch-site truth is available for all audited tiers. Proceed to explicit branch-truth validation at larger scale only after checking foreground-context ablation and aggregation controls.
Do not run final 100K until explicit branch-site truth validation passes.
