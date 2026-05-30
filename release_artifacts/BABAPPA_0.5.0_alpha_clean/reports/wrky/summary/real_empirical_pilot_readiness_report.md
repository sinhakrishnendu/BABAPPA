# Real Empirical Pilot Readiness

- status: `NEED_INPUT_REPAIR`
- manifest: `real_empirical_pilot/manifest/real_empirical_pilot_panel.tsv`
- manifest rows: `12`
- missing inputs: `24`

## Claim Boundary

BABAPPA model is simulation-trained. No simulator truth was used for empirical inference. Scores are diagnostic until simulation-matched calibration and external benchmark interpretation are complete. Out-of-domain cases are not positive-selection calls. Reference-tool disagreement must be interpreted biologically, not automatically treated as BABAPPA failure.

## Missing Inputs

- missing_cds_fasta:wrky_candidate_01:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/wrky_candidate_01.cds.fasta
- missing_tree_file:wrky_candidate_01:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/wrky_candidate_01.treefile
- missing_cds_fasta:constans_like_candidate_01:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/constans_like_candidate_01.cds.fasta
- missing_tree_file:constans_like_candidate_01:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/constans_like_candidate_01.treefile
- missing_cds_fasta:immune_candidate_01:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/immune_candidate_01.cds.fasta
- missing_tree_file:immune_candidate_01:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/immune_candidate_01.treefile
- missing_cds_fasta:housekeeping_negative_01:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/housekeeping_negative_01.cds.fasta
- missing_tree_file:housekeeping_negative_01:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/housekeeping_negative_01.treefile
- missing_cds_fasta:housekeeping_negative_02:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/housekeeping_negative_02.cds.fasta
- missing_tree_file:housekeeping_negative_02:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/housekeeping_negative_02.treefile
- missing_cds_fasta:gst_detox_candidate_01:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/gst_detox_candidate_01.cds.fasta
- missing_tree_file:gst_detox_candidate_01:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/gst_detox_candidate_01.treefile
- missing_cds_fasta:alignment_sensitive_01:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/alignment_sensitive_01.cds.fasta
- missing_tree_file:alignment_sensitive_01:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/alignment_sensitive_01.treefile
- missing_cds_fasta:alignment_sensitive_02:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/alignment_sensitive_02.cds.fasta
- missing_tree_file:alignment_sensitive_02:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/alignment_sensitive_02.treefile
- missing_cds_fasta:saturated_01:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/saturated_01.cds.fasta
- missing_tree_file:saturated_01:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/saturated_01.treefile
- missing_cds_fasta:saturated_02:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/saturated_02.cds.fasta
- missing_tree_file:saturated_02:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/saturated_02.treefile
- missing_cds_fasta:short_low_information_01:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/short_low_information_01.cds.fasta
- missing_tree_file:short_low_information_01:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/short_low_information_01.treefile
- missing_cds_fasta:paralogy_risk_01:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/paralogy_risk_01.cds.fasta
- missing_tree_file:paralogy_risk_01:/Users/krishnendu/Documents/GitHub/BABAPPA/real_empirical_pilot/input/paralogy_risk_01.treefile

## Next Action

Populate real_empirical_pilot/input with real CDS FASTA/tree files and set real foreground taxa.
