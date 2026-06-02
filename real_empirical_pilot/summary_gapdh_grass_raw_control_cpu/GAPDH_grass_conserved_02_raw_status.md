# GAPDH Grass Raw Control Status

This report summarizes the updated `GAPDH_grass_conserved_02_raw` run performed after duplicate `Phala.08G200500` isoforms were removed.

## Input

- FASTA: `real_empirical_pilot/input/cds/GAPDH_grass_conserved_02_raw.cds.fasta`
- Tree: `real_empirical_pilot/input/trees/GAPDH_grass_conserved_02_raw.treefile`
- Manifest: `real_empirical_pilot/manifest/gapdh_grass_raw_control_panel.tsv`
- Taxa: 6
- CDS gaps: none
- CDS length divisible by 3: yes
- CDS/tree tips match: yes

## Prefilter

- Decision: `diagnostic_only`
- Mean raw p-distance: `0.701344`
- Recommended action from prefilter: use as an OOD/saturated diagnostic stress test or build a closer-taxa panel.

## BABAPPA CPU Fallback Run

- Run directory: `real_empirical_pilot/babappa_run_gapdh_grass_raw_control_cpu`
- Summary directory: `real_empirical_pilot/summary_gapdh_grass_raw_control_cpu`
- Scoring status: `ok`
- Applicability after alignment-aware check: `in_domain`
- Tier model: `high`
- Device: `cpu`
- Score rows: `4974`
- BABAPPA result class: `positive`
- Max gene support: `1.0`
- Called positive rows: `361`

## Alignment Caveat

The result is not cleanly interpretable as a conserved negative control:

- `identity` failed because raw CDS lengths are unequal.
- `mafft` failed in the restricted shell due local `/dev/stderr` permission behavior.
- `babappalign` completed.
- `muscle` completed.
- The method policy quarantined all methods because site-map quality was poor.

## Interpretation

Do not treat this as a biological positive-selection result or as a clean conserved negative control. The current safest interpretation is:

`saturated_or_alignment_sensitive_diagnostic_positive_with_method_policy_failure`

Recommended next step: rerun the user script from the normal terminal/molevo environment so MAFFT and MPS are available outside the restricted shell environment. If the fresh run still quarantines all methods, curate a closer/cleaner GAPDH ortholog panel before using it as a negative control.
