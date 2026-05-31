# DGAPDH_candidate_01 Interpretation

## Run Status

- Run directory: `real_empirical_pilot/babappa_run_dgapdh_candidate`
- Summary directory: `real_empirical_pilot/summary_dgapdh_candidate`
- Panel status: `ok`
- Summary validation: `ok`
- Scoring status: `ok`
- Score rows: `2808`

## Input/QC

- Taxa: `8`
- Codons: `306`
- Foreground: `SMAR006602-PA`
- Raw p-distance: `0.680309`
- Alignment-aware p-distance used by applicability: `0.350375`
- Applicability: `out_of_domain`
- Recommended tier: `extreme`

## Prefilter

- Decision: `diagnostic_only`
- Main flags:
  - high raw p-distance
  - long-branch outlier: `BL20647_evm1`
  - unusually divergent pair: `0.746548`

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

- BABAPPA result class: `positive`
- Max gene support: `0.835383`
- Called positive rows: `1`
- Called branch: `SMAR006602-PA`
- Diagnostic-only: `True`
- Model tier used: `extreme`

## Interpretation

This result should **not** be interpreted as biological positive selection and should **not** be used as a clean conserved negative-control result.

The dataset is structurally valid and BABAPPA completed scoring, but the applicability gate marks it out-of-domain and the method policy quarantines all methods. The single called row on `SMAR006602-PA` is best interpreted as a diagnostic stress-test signal under high divergence/site-map uncertainty.

Safest label:

`out_of_domain_single_row_diagnostic_positive_with_method_policy_failure`

## Recommended Next Step

For a usable negative-control dGAPDH benchmark, build a closer-taxa set and remove or replace long-branch outliers, especially `BL20647_evm1`. The goal should be an alignment-aware p-distance below the empirical OOD gate and at least one non-quarantined production aligner.
