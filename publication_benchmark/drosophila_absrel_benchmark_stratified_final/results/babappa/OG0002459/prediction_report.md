# BABAPPA Branch-Site Prediction

BABAPPA used the supplied codon MSA as the authoritative alignment. No realignment or aligner-disagreement analysis was performed.

- status: `ok`
- applicability: `in_domain`
- foreground requested: `leaves`
- branches scored: `12`
- branch-site rows: `5292`
- called positive branch-site rows: `5286`
- max gene support: `0.4014289975166321`
- result class: `diagnostic_positive`
- BABAPPA-native evidence class: `strong_babappa_native_support`
- BABAPPA-native result class: `babappa_native_calibrated_support`
- p_BABAPPA called rows: `0.000999000999000999`
- p_BABAPPA max gene support: `0.13886113886113885`

## Main Outputs

- branch-site predictions: `publication_benchmark/drosophila_absrel_benchmark_stratified_final/results/babappa/OG0002459/branch_site_predictions.tsv`
- branch summaries: `publication_benchmark/drosophila_absrel_benchmark_stratified_final/results/babappa/OG0002459/branch_predictions.tsv`
- gene summary: `publication_benchmark/drosophila_absrel_benchmark_stratified_final/results/babappa/OG0002459/gene_summary.tsv`
- BABAPPA-native null calibration: `publication_benchmark/drosophila_absrel_benchmark_stratified_final/results/babappa/OG0002459/babappa_native_null/babappa_native_null_report.md`

## Interpretation Boundary

BABAPPA can now report standalone BABAPPA-native calibrated evidence using its own branch-shuffle feature null. This is designed to be a complementary evidence system, not a codeml/HyPhy dependency. For manuscript use, report the BABAPPA-native null backend, replicate count, p-like values, OOD status, and biological context.
