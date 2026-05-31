# DGAPDH_candidate_02_purified Interpretation

## Run Status

- Run directory: `real_empirical_pilot/babappa_run_dgapdh_purified`
- Summary directory: `real_empirical_pilot/summary_dgapdh_purified`
- Panel status: `ok`
- Summary validation: `ok`
- Scoring status: `ok`
- Score rows: `2429`

## Input/QC

- Taxa: `7`
- Codons: `320`
- Foreground: `SMAR006602-PA`
- Raw p-distance: `0.683663`
- Alignment-aware p-distance used by applicability: `0.324778`
- Applicability: `borderline`
- Recommended tier: `extreme`

## Purification Effect

Compared with `DGAPDH_candidate_01`, this purified dataset removes the long-branch/outlier sequence `BL20647_evm1`.

Improvements:

- Sequence lengths are tightly clustered: 960-1023 nt.
- Length ratio is approximately 1.066.
- The previous single positive row disappeared.
- BABAPPA result class changed from diagnostic positive to negative.

Remaining limitation:

- The family remains highly divergent by raw p-distance and borderline by alignment-aware applicability.

## Alignment Ensemble

- `identity`: failed because raw unaligned CDS lengths are unequal.
- `mafft`: failed in the current environment.
- `babappalign`: completed.
- `muscle`: failed because the aligned length was not divisible by 3.

Method-policy result:

- `identity`: quarantine
- `mafft`: quarantine
- `babappalign`: quarantine, site-map quality
- `muscle`: quarantine
- Usable methods: none

## BABAPPA Scores

- BABAPPA result class: `negative`
- Max gene support: `2.97112e-25`
- Called positive rows: `0`
- Diagnostic-only: `False`
- Model tier used: `extreme`

## Interpretation

This is the best dGAPDH control result so far. It behaves as a BABAPPA negative after purification, and removing `BL20647_evm1` eliminated the previous one-row diagnostic-positive signal.

However, it is still **not** a fully clean conserved negative-control benchmark because:

- Applicability is `borderline`, not fully `in_domain`.
- All methods are quarantined by method policy.
- BABAPPAlign completed, but site-map quality is poor.

Safest label:

`borderline_negative_after_outlier_removal_with_method_policy_failure`

## Recommended Next Step

This purified dGAPDH set is worth keeping as a promising negative-control candidate, but for manuscript-grade interpretation it needs either:

1. A closer dGAPDH taxon set with lower alignment-aware p-distance, or
2. A rerun in a terminal/environment where MAFFT and MUSCLE generate acceptable codon/site maps, followed by external reference checks.
