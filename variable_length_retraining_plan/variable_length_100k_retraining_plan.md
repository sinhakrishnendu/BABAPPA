# BABAPPA variable-length normalized-v2 100K retraining plan

- Target families: `100000`
- Feature policy: `conservative_branch_site_normalized_v2`
- Workspace: `branch_site_v2_100k_workspace`
- Package output: `deployable_model_conservative_branch_site_v2_100k_mps`
- Cleanup default: `quarantine`; set `BABAPPA_RETRAIN_CLEANUP_MODE=delete` and `BABAPPA_RETRAIN_DELETE_INTERMEDIATES=YES` to actually free disk after completed chunks.

## Why this plan exists

The previous deployable model used raw length/site features that can become far outside its training envelope on real user MSAs. This plan retrains with normalized/log length features and broader simulated gene lengths and taxon counts.

## Storage policy

Each chunk is simulated, aligned, mapped, tensorized, indexed, labelled, and reduced to a branch-site feature table. After that feature table validates, raw simulation, alignment, tensor, index, and label intermediates are cleaned before the next chunk.

## First command

```bash
bash variable_length_retraining_plan/run_variable_length_100k_retraining.sh
```
