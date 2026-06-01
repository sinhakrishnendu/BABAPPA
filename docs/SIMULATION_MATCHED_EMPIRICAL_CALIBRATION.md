# Simulation-Matched Empirical Calibration

BABAPPA empirical calibration is a research-alpha scaffold in `v0.8.0`. It plans and validates simulation-matched calibration workflows for empirical diagnostic results, but completed calibration is still required before any final empirical branch-site inference claim.

## Why Calibration Is Required

The 100K MPS package is simulation-trained under explicit simulator branch-site truth. Real empirical datasets can differ in taxon count, codon length, gap burden, composition, saturation, tree shape, alignment disagreement, paralogy risk, and annotation quality. Those differences must be measured before BABAPPA scores are interpreted.

## Planner Command

```bash
babappa plan-simulation-matched-calibration --empirical-validation-dir empirical_input_validation --deployable-model-package deployable_model_conservative_branch_site_100k_mps --outdir simulation_matched_calibration_plan
```

The planner reads available empirical QC summaries and proposes matched null-simulation parameters:

- `n_taxa`
- `n_codons`
- GC content if available
- mean pairwise p-distance
- saturation proxy
- gap fraction
- foreground branch/taxon
- tree shape summary
- alignment disagreement
- ambiguous base fraction if available

Outputs:

- `simulation_matched_calibration_plan.json`
- `simulation_matched_calibration_plan.md`
- `proposed_null_simulation_commands.sh`
- `proposed_alt_simulation_commands.sh`
- `expected_outputs.json`

The generated shell script is marked USER-RUN ONLY and keeps the proposed heavy command commented. It is a decision aid, not an execution step.

## Empirical Scoring Scaffold

```bash
babappa plan-empirical-scoring --cds-fasta tests/data/empirical_smoke/tiny_empirical.cds.fasta --tree tests/data/empirical_smoke/tiny_empirical.treefile --foreground taxon1 --deployable-model-package deployable_model_conservative_branch_site_100k_mps --outdir empirical_scoring_plan_smoke --methods identity,mafft,babappalign,muscle --device auto
```

The current scoring scaffold wires the guarded tiny pipeline:

input validation -> empirical alignment ensemble -> empirical feature extraction -> feature safety audit -> applicability/OOD check -> deployable-model scoring -> simulation-matched calibration planning -> empirical report.

The generated script stops on invalid package/input, forbidden truth-derived columns, and out-of-domain applicability unless diagnostic-only out-of-domain scoring is explicitly allowed.

## Safety Boundary

Empirical scoring must never read truth-derived input columns or files:

- `branch_site_truth.tsv`
- `selected_sites`
- `truth.json`
- `branch_truth.json`
- oracle label columns
- `y_branch_site`
- `y_site`
- `gene_label`

Any empirical result produced before QC, OOD gating, simulation-matched calibration, and benchmark comparison is diagnostic only and not final empirical branch-site inference.

## Pilot-Panel Use

Cycle 42 pilot panels call the simulation-matched calibration planner for each small family after empirical QC and applicability checks. The generated proposed null and optional alternative simulation commands are marked USER-RUN ONLY and are not executed during the pilot smoke. The pilot summary reports calibration as planned, not run.

Cycle 43 real-pilot readiness stops before scoring when real CDS/tree files are missing. Once real inputs pass validation, the pilot runner creates per-family `simulation_matched_calibration_plan/` directories but still does not execute null simulations.

## Cycle 48 WRKY Null Calibration Pilot

`WRKY_candidate_02_close` is the first in-domain diagnostic empirical pilot. Its BABAPPA score remains non-manuscript-ready until reference disagreement, feature-level matched-null results, and biological controls are interpreted together.

For the WRKY close-taxa pilot, the user-run feature-level matched-null calibration command is:

```bash
babappa run-simulation-matched-null-calibration --evidence-pack real_empirical_pilot/evidence_packs/WRKY_candidate_02_close --outdir real_empirical_pilot/calibration_runs/WRKY_candidate_02_close_null100 --n-null 100 --seed 20260530 --device mps
babappa validate-simulation-matched-null-calibration --calibration-dir real_empirical_pilot/calibration_runs/WRKY_candidate_02_close_null100
```

The completed WRKY run has 100/100 scored feature-level null replicates, validation status `ok`, `p_empirical_called_rows=0.009900990099009901`, and `p_empirical_support=1.0`. This is mixed diagnostic support: the called-row burden is unusual, but maximum gene support is not. It remains feature-level matched null scoring, not full raw sequence simulation/alignment replay.

The matched-null runner supports resumable user-run scoring. Codex only runs tiny smoke tests or dry runs, and BABAPPA does not fabricate null p-like percentiles; percentiles are written only from completed scored null replicates. Reports must remain diagnostic while `null_scoring_completed` is false.

## Long-Run Handoff Policy

Codex does not execute heavy empirical calibration, broad empirical scans, retraining, 10K/100K simulations, or long aligner/reference batches. Codex generates reproducible USER-RUN scripts, validators, parsers, and reports; the user executes long runs locally/offline and returns logs or summaries for interpretation.

Cycle 49 permits only tiny smoke runs in Codex, such as:

```bash
babappa run-simulation-matched-null-calibration --plan-dir real_empirical_pilot/babappa_run_wrky_close_raw_alignmentaware/per_family/WRKY_candidate_02_close/simulation_matched_calibration_plan --deployable-model-package deployable_model_conservative_branch_site_100k_mps --outdir real_empirical_pilot/calibration_runs/WRKY_candidate_02_close_null3_smoke --n-replicates 3 --device auto --seed 42 --fast-null-mode
```

Large or repeated calibration runs remain USER-RUN ONLY.
