# BABAPPA branch-site truth upgrade plan

## Purpose

The next simulator upgrade should emit explicit branch-site selected-event truth so branch-conditioned labels no longer need to be inferred from foreground taxon or gene-level selected-site proxies.

## Required truth fields per family

- `family_id`
- `tree`
- `foreground_branch_id`
- `foreground_taxon`
- `branch_length`
- `selected_sites`
- `selected_site_by_branch`
- `y_branch_site` matrix
- `selection_event_id`
- `omega/background/foreground parameters if available`
- `saturation tier`
- `alignment method after mapping`

## Required files

- `family_XXXX.branch_truth.json`
- `branch_truth_manifest.json`
- `branch_site_truth.tsv`

## Required scientific behavior

- A site can be selected on one branch and not on another.
- Branch-site label must not be inferred only from gene-level `selected_sites`.
- Foreground branch must be explicit.
- Internal branches should be supported later, but leaf foreground is acceptable for the first version.
- Simulator should emit enough truth for branch-conditioned labels without proxy.

## Prototype validation

The first explicit-truth validation should be a 1K prototype across low, moderate, high, and extreme tiers, using `identity,mafft,babappalign,muscle`. It should validate that `y_branch_site` is loaded directly from simulator truth, that proxy labels are not used, and that branch IDs are stable through site mapping.

## Cycle 34 implementation status

Implemented in `0.4.1-alpha`:

- The simulator emits per-family `family_XXXX.branch_truth.json`.
- The simulator emits dataset-level `branch_truth_manifest.json`.
- The simulator emits dataset-level `branch_site_truth.tsv`.
- `validate-sim --require-branch-truth` requires explicit branch-site truth artifacts.
- `audit-sim` reports branch truth presence, row counts, positive row counts, and branch truth status.
- Branch-label extraction supports `--truth-mode auto|explicit|required|proxy` and prioritizes `explicit_simulator_branch_truth`.
- Proxy labels are fallback only.

Remaining limitations:

- Leaf foreground branches are supported.
- Internal branches are future work.
- Empirical branch-site inference is still not claimed.

## 100K gate

Final 100K should remain deferred until explicit branch-site truth validation passes.
