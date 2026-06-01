# BABAPPA Branch-Site Prediction

BABAPPA used the supplied codon MSA as the authoritative alignment. No realignment or aligner-disagreement analysis was performed.

- status: `ok`
- applicability: `in_domain`
- foreground requested: `LOC_Os02g07490_LOC_Os02g07490.1`
- branches scored: `1`
- branch-site rows: `419`
- called positive branch-site rows: `0`
- max gene support: `0.0`
- result class: `diagnostic_negative`
- BABAPPA-native evidence class: `not_significant_under_babappa_native_null`
- BABAPPA-native result class: `babappa_native_negative`
- p_BABAPPA called rows: `1.0`
- p_BABAPPA max gene support: `1.0`

## Main Outputs

- branch-site predictions: `publication_benchmark/results_gapdh_grass/babappa/GAPDH_grass_conserved_01/branch_site_predictions.tsv`
- branch summaries: `publication_benchmark/results_gapdh_grass/babappa/GAPDH_grass_conserved_01/branch_predictions.tsv`
- gene summary: `publication_benchmark/results_gapdh_grass/babappa/GAPDH_grass_conserved_01/gene_summary.tsv`
- BABAPPA-native null calibration: `publication_benchmark/results_gapdh_grass/babappa/GAPDH_grass_conserved_01/babappa_native_null/babappa_native_null_report.md`

## Interpretation Boundary

BABAPPA can now report standalone BABAPPA-native calibrated evidence using its own branch-shuffle feature null. This is designed to be a complementary evidence system, not a codeml/HyPhy dependency. For manuscript use, report the BABAPPA-native null backend, replicate count, p-like values, OOD status, and biological context.
