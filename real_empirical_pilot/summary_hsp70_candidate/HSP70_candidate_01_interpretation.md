# HSP70_candidate_01 Interpretation

## Run Status

- Run directory: `real_empirical_pilot/babappa_run_hsp70_candidate`
- Summary directory: `real_empirical_pilot/summary_hsp70_candidate`
- Panel status: `ok`
- Summary validation: `ok`
- Scoring status: `ok`
- Score rows: `6320`

## Input/QC

- Taxa: `8`
- Codons: `337`
- Foreground: `Cucsa.010680|Cucsa.010680.1`
- Raw p-distance: `0.686362`
- Alignment-aware p-distance used by applicability: `0.437231`
- Applicability: `out_of_domain`
- Recommended tier: `extreme`

## Prefilter

- Decision: `reject_possible_paralogy`
- Main flags:
  - duplicate species/isoforms: `ChfasH1`
  - extreme length ratio: `2.288`
  - long-branch outliers: `Glyma.13G224100|Glyma.13G224100.1`, `Psat06G0270900|Psat06G0270900-T1`
  - unusually divergent pair: `0.74184`

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
- Max gene support: `0`
- Called positive rows: `0`
- Diagnostic-only: `True`
- Model tier used: `extreme`

## Interpretation

This result should **not** be used as biological evidence for absence or presence of positive selection.

The software completed and abstained from a positive call, which is a useful diagnostic behavior. However, the family is out-of-domain, prefilter rejects it as possible paralogy/high divergence, and method policy quarantines every alignment method. The safest interpretation is:

`out_of_domain_diagnostic_negative_with_paralogy_and_alignment_quality_failure`

## Recommended Next Step

For an interpretable HSP70 analysis, curate one ortholog/isoform per species, remove or replace extreme long-branch/short-fragment sequences, and build a closer-taxa HSP70 panel before rerunning BABAPPA and external references.
