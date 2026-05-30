# WRKY_candidate_02_close Interpretation Status

## Decision

- decision: `diagnostic_positive_pending_reference_and_calibration`
- manuscript-ready: `False`

## Why This Is Not Yet A Discovery Claim

Diagnostic empirical evidence only. Not manuscript-ready and not a final positive-selection discovery claim until simulation-matched calibration and codeml/HyPhy-style reference comparison are interpreted.

## Reference Tests

- codeml: Branch-site model A versus null on the marked foreground branch.
- HyPhy: aBSREL tests branch-level episodic selection; MEME tests site-level episodic selection.

## Calibration

Simulation-matched null calibration estimates family-specific score behavior before interpretation.

## Next User-Run Commands

- `cd real_empirical_pilot/reference_runs/WRKY_candidate_02_close/codeml && bash run_codeml_modelA.sh && bash run_codeml_null.sh && bash parse_codeml_lrt.sh`
- `cd real_empirical_pilot/reference_runs/WRKY_candidate_02_close/hyphy && bash run_absrel.sh && bash run_meme.sh`
- `bash real_empirical_pilot/babappa_run_wrky_close_raw_alignmentaware/per_family/WRKY_candidate_02_close/simulation_matched_calibration_plan/run_wrky_close_matched_nulls.sh`
- `babappa compare-empirical-reference-results --babappa-panel-run real_empirical_pilot/babappa_run_wrky_close_raw_alignmentaware --reference-results real_empirical_pilot/reference_results/WRKY_candidate_02_close_reference_results_template.tsv --outdir real_empirical_pilot/comparison/WRKY_candidate_02_close`
