# GAPDH_grass_conserved_02_raw Interpretation

## Run Status

- Run directory: `real_empirical_pilot/babappa_run_gapdh_grass_raw_control_fresh`
- Summary directory: `real_empirical_pilot/summary_gapdh_grass_raw_control_fresh`
- Panel status: `ok`
- Summary validation: `ok`
- Device: `mps`
- Scoring status: `ok`
- Score rows: `7404`

## Input/QC

- Taxa: `6`
- Codons: `289`
- Foreground: `LOC_Os02g07490_LOC_Os02g07490.1`
- Raw p-distance: `0.701344`
- Alignment-aware p-distance used by applicability: `0.227090`
- Applicability: `in_domain`
- Recommended tier: `high`

## Alignment Ensemble

- `identity`: failed, because raw unaligned CDS lengths are unequal.
- `mafft`: completed.
- `babappalign`: completed.
- `muscle`: completed.

Method-policy result:

- `identity`: quarantine
- `mafft`: quarantine, site-map quality
- `babappalign`: quarantine, site-map quality
- `muscle`: quarantine, site-map quality
- Usable methods: none

## BABAPPA Scores

Gene-level support by method:

- `mafft`: max probability `1.0`, called rows `170`
- `babappalign`: max probability `1.0`, called rows `183`
- `muscle`: max probability `1.0`, called rows `178`

Panel-level result:

- BABAPPA result class: `positive`
- Max gene support: `1`
- Total called positive rows: `531`
- Diagnostic-only flag from scoring: `False`

## Interpretation

This result should **not** be treated as a clean conserved negative-control result.

The run is technically successful, and BABAPPA produces a diagnostic-positive signal. However, the method-policy layer quarantines every alignment method because site-map quality is poor. The input also has high raw sequence divergence. Therefore the scientifically safest interpretation is:

`diagnostic_positive_with_alignment/site-map_quality_failure`

This should be treated as a failure-mode/control-design finding, not as biological evidence for positive selection in GAPDH.

## Recommended Next Step

Build a cleaner conserved negative control by using a closer taxonomic panel and/or a longer, less alignment-fragmented ortholog set. For this GAPDH panel specifically, external codeml/HyPhy reference templates can be generated, but even if references are negative, the method-policy failure already prevents this from being a clean BABAPPA negative-control benchmark.
