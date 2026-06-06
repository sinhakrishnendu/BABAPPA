# BABAPPA Branch-Site Prediction

BABAPPA used the supplied codon MSA as the authoritative alignment. No realignment or aligner-disagreement analysis was performed.

- status: `ok`
- applicability: `in_domain`
- foreground requested: `all`
- branches scored: `26`
- branch-site rows: `22802`
- called positive branch-site rows: `0`
- max gene support: `0.0`
- result class: `diagnostic_negative`
- BABAPPA-native evidence class: `not_significant_under_babappa_native_null`
- BABAPPA-native result class: `babappa_native_negative`
- p_BABAPPA called rows: `1.0`
- p_BABAPPA max gene support: `1.0`

## Main Outputs

- branch-site predictions: `publication_revision_benchmarks/known_positive_control_results_with_h3_null100/known_positive_hiv_env/branch_site_predictions.tsv`
- branch summaries: `publication_revision_benchmarks/known_positive_control_results_with_h3_null100/known_positive_hiv_env/branch_predictions.tsv`
- gene summary: `publication_revision_benchmarks/known_positive_control_results_with_h3_null100/known_positive_hiv_env/gene_summary.tsv`
- BABAPPA-native null calibration: `publication_revision_benchmarks/known_positive_control_results_with_h3_null100/known_positive_hiv_env/babappa_native_null/babappa_native_null_report.md`

## Interpretation Boundary

BABAPPA can now report standalone BABAPPA-native calibrated evidence using its own branch-shuffle feature null. This is designed to be a complementary evidence system, not a codeml/HyPhy dependency. For manuscript use, report the BABAPPA-native null backend, replicate count, p-like values, OOD status, and biological context.
