# BABAPPA

BABAPPA is the Branch-site Alignment-Bias-Aware Probabilistic Positive-selection Analyzer.

BABAPPA is a research-alpha, simulation-trained command-line framework for guarded branch-site analysis. It combines alignment ensembles, branch-site feature extraction, applicability/OOD gates, simulation-matched calibration planning, and external reference-workflow planning.

Current package version: `0.4.9-alpha`.

## User Manual Contents

- Important Scientific Boundary
- What BABAPPA Does
- Recommended End-User Workflow
- Installation
- External Dependencies
- Apple Silicon / MPS Setup
- Deployable Model Package
- Real Empirical Pilot: Input Staging
- Real Empirical Pilot: OOD Prefiltering
- Real Empirical Pilot: Run
- Simulation-Matched Calibration Planning
- Classical Reference Workflow Planning
- If Trees Are Missing
- Tiny Smoke Test
- Output Files To Inspect
- Troubleshooting
- More Documentation

## Important Scientific Boundary

BABAPPA can now run small guarded empirical diagnostic pilots, but it is not a final empirical positive-selection caller. Do not report BABAPPA scores as biological discovery claims unless later validation cycles add simulation-matched calibration, OOD acceptance, and external benchmark concordance.

BABAPPA empirical reports should be read as diagnostic evidence:

- The deployable model is simulation-trained.
- No simulator truth is used during empirical inference.
- OOD cases are diagnostic-only, not positive-selection calls.
- Real empirical interpretation requires input QC, alignment sensitivity review, applicability/OOD checks, simulation-matched calibration, and codeml/HyPhy-style comparison where appropriate.

## What BABAPPA Does

- Validates real CDS FASTA and tree inputs.
- Screens candidate gene families before scoring to avoid overly divergent or likely-paralogous sets.
- Runs a production empirical alignment ensemble: `identity`, `mafft`, `babappalign`, `muscle`.
- Extracts conservative branch-site features that avoid simulator-truth leakage.
- Scores branch-site rows with a packaged simulation-trained `conservative_branch_site` model family.
- Marks OOD cases as diagnostic-only.
- Plans simulation-matched calibration runs without executing heavy simulations automatically.
- Plans codeml/HyPhy-style reference workflows without running those tools automatically.
- Summarizes pilot panels and enforces claim-boundary wording.

## Recommended End-User Workflow

Use this order for real empirical work:

1. Install BABAPPA and external aligners.
2. Stage real CDS FASTA and tree files.
3. Validate and sanitize inputs.
4. Prefilter each family for OOD/divergence/paralogy risk.
5. Add only accepted or caution-accepted families to the pilot manifest.
6. Validate readiness.
7. Run a small empirical pilot.
8. Review applicability/OOD and diagnostic scores.
9. Plan simulation-matched calibration.
10. Plan codeml/HyPhy reference workflows.
11. Interpret results as diagnostic until calibration and reference comparison are complete.

## Installation

From the repository root:

```bash
pip install -e .
```

For development and testing:

```bash
pip install -e ".[dev]"
```

Check the installed version and command list:

```bash
babappa --version
babappa --help
```

## External Dependencies

For empirical pilots, install these outside BABAPPA:

- Python 3.10 or newer.
- PyTorch if you want model scoring. On Apple Silicon, install a PyTorch build with MPS support.
- `mafft`.
- `muscle`.
- `babappalign`.
- Optional for tree building: `iqtree2` or `iqtree`.
- Optional for reference comparisons: `codeml` and HyPhy.

BABAPPAlign also needs its BABAPPAScore model cache:

```bash
mkdir -p "$HOME/.cache/babappalign/models"
curl -L "https://zenodo.org/record/18053201/files/babappascore.pt" -o "$HOME/.cache/babappalign/models/babappascore.pt"
```

Check aligner availability:

```bash
babappa check-aligners
```

## Apple Silicon / MPS Setup

Apple Silicon/MPS support is research-alpha. It is useful for lightweight smoke tests and guarded empirical scoring when PyTorch MPS is available.

Recommended shell settings:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
```

Check neural/MPS support:

```bash
babappa check-neural-env
babappa smoke-mps-training --outdir mps_smoke --device auto --batch-size 32 --max-items 512
babappa validate-mps-smoke --smoke-dir mps_smoke
babappa benchmark-apple-silicon --outdir apple_silicon_benchmark --device auto --batch-sizes 32,64,128 --max-items 1024
```

If MPS fails during scoring, retry with `--device cpu` or a smaller batch size.

## Deployable Model Package

The standard simulation-trained package directory is:

```text
deployable_model_conservative_branch_site_100k_mps
```

Validate it before empirical use:

```bash
babappa validate-deployable-model-package --package-dir deployable_model_conservative_branch_site_100k_mps
babappa smoke-load-deployable-model --package-dir deployable_model_conservative_branch_site_100k_mps --device auto --outdir deployable_model_load_smoke
```

The package is simulation-trained and conservative. It excludes raw simulator truth, raw oracle labels, raw branch-site datasets, simulations, and alignments.

## Real Empirical Pilot: Input Staging

Create a workspace:

```bash
babappa prepare-real-empirical-pilot-workspace --workspace real_empirical_pilot --max-families 12
babappa prepare-real-pilot-inputs --workspace real_empirical_pilot --manifest real_empirical_pilot_panel.tsv --outdir real_empirical_pilot/input_staging
```

Put real input files at the suggested paths:

```text
real_empirical_pilot/input/cds/<panel_id>.cds.fasta
real_empirical_pilot/input/trees/<panel_id>.treefile
```

Or import one family:

```bash
babappa import-real-pilot-family \
  --workspace real_empirical_pilot \
  --panel-id FAMILY_ID \
  --gene-family "GENE_FAMILY" \
  --species-group "SPECIES_GROUP" \
  --cds-fasta /path/to/family.cds.fasta \
  --tree-file /path/to/family.treefile \
  --foreground TAXON_NAME \
  --expected-category likely_positive \
  --reference-status planned \
  --notes "real pilot candidate"
```

For batch imports, create a TSV with columns `panel_id`, `gene_family`, `species_group`, `cds_fasta`, `tree_file`, `foreground`, `expected_category`, `reference_status`, and `notes`, then run:

```bash
babappa import-real-pilot-batch --workspace real_empirical_pilot --batch-manifest real_empirical_pilot/import_batch.tsv
```

Useful input repair helpers:

```bash
babappa sanitize-cds-fasta --input input.cds.fasta --output cleaned.cds.fasta --report sanitize_report.json --mode strict
babappa list-foreground-candidates --cds-fasta real_empirical_pilot/input/cds/FAMILY_ID.cds.fasta --tree-file real_empirical_pilot/input/trees/FAMILY_ID.treefile --outdir real_empirical_pilot/foreground_candidates/FAMILY_ID
babappa discover-local-pilot-files --search-dir /path/to/user/data --outdir real_empirical_pilot/local_discovery
```

Validate readiness:

```bash
babappa validate-real-pilot-readiness --workspace real_empirical_pilot --manifest real_empirical_pilot_panel.tsv --outdir real_empirical_pilot/readiness
```

Do not run the pilot until `ready_to_run` is `true`.

## Real Empirical Pilot: OOD Prefiltering

Before scoring a real family, screen it:

```bash
babappa prefilter-empirical-family \
  --cds-fasta real_empirical_pilot/input/cds/FAMILY_ID.cds.fasta \
  --tree-file real_empirical_pilot/input/trees/FAMILY_ID.treefile \
  --foreground TAXON_NAME \
  --outdir real_empirical_pilot/prefilter/FAMILY_ID \
  --max-mean-pdistance 0.35 \
  --min-taxa 6 \
  --min-codons 100
```

Decision meanings:

- `accept`: eligible for guarded empirical pilot.
- `accept_with_caution`: usable, but interpret carefully.
- `reject_too_divergent`: choose closer taxa.
- `reject_too_short`: choose a longer coding region or different family.
- `reject_too_few_taxa`: add more close orthologs.
- `reject_tree_mismatch`: repair FASTA/tree labels.
- `reject_possible_paralogy`: curate likely orthologs.
- `diagnostic_only`: useful as a stress test, not as empirical evidence.

If a family is `accept` or `accept_with_caution`, add it to the real pilot manifest:

```bash
babappa add-prefiltered-family-to-pilot \
  --workspace real_empirical_pilot \
  --prefilter-dir real_empirical_pilot/prefilter/FAMILY_ID \
  --panel-id FAMILY_ID \
  --expected-category likely_positive \
  --reference-status planned
```

For Arabidopsis-like WRKY pilots, start with close Brassicaceae taxa:

```bash
babappa recommend-target-taxa --pilot-type plant_close --outdir real_empirical_pilot/target_taxa_recommendations
```

Plan an OOD-gated family acquisition workflow:

```bash
babappa plan-ood-aware-family-build \
  --family-id WRKY_candidate_02_close \
  --query-species Arabidopsis_thaliana \
  --query-gene-or-locus AT2G38470 \
  --target-taxa-file real_empirical_pilot/target_taxa_recommendations/recommended_target_taxa.tsv \
  --outdir real_empirical_pilot/acquisition_plans/WRKY_candidate_02_close \
  --max-mean-pdistance 0.35 \
  --min-taxa 6 \
  --min-codons 100
```

Generated acquisition scripts are marked `USER-RUN ONLY`. Review and edit source URLs/queries before running them.

Summarize OOD state across the workspace:

```bash
babappa summarize-empirical-ood --workspace real_empirical_pilot --outdir real_empirical_pilot/ood_summary
```

## Real Empirical Pilot: Run

Run only a small curated panel first, usually 8-12 families:

```bash
babappa run-empirical-pilot-panel \
  --panel-manifest real_empirical_pilot/manifest/real_empirical_pilot_panel.tsv \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir real_empirical_pilot/babappa_run \
  --methods identity,mafft,babappalign,muscle \
  --device auto \
  --max-families 12
```

Summarize and validate the report:

```bash
babappa summarize-empirical-pilot-panel --panel-run real_empirical_pilot/babappa_run --outdir real_empirical_pilot/summary
babappa validate-empirical-pilot-summary --summary-dir real_empirical_pilot/summary
```

## Simulation-Matched Calibration Planning

Plan calibration simulations from empirical QC. This writes scripts and expected outputs but does not run heavy simulations:

```bash
babappa plan-simulation-matched-calibration \
  --empirical-validation-dir real_empirical_pilot/babappa_run/per_family/FAMILY_ID/empirical_input_validation \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir real_empirical_pilot/babappa_run/per_family/FAMILY_ID/simulation_matched_calibration_plan
```

## Classical Reference Workflow Planning

Generate codeml/HyPhy command templates:

```bash
babappa plan-classical-reference-workflows \
  --panel-manifest real_empirical_pilot/manifest/real_empirical_pilot_panel.tsv \
  --outdir real_empirical_pilot/reference_plan \
  --tools codeml,hyphy
```

If manually curated reference results are available:

```bash
babappa compare-empirical-reference-results \
  --babappa-panel-run real_empirical_pilot/babappa_run \
  --reference-results real_empirical_pilot/reference_results/reference_results.tsv \
  --outdir real_empirical_pilot/comparison
```

## WRKY Close-Taxa Reference And Calibration Status

`WRKY_candidate_02_close` is BABAPPA's first in-domain diagnostic empirical pilot. It uses close Brassicaceae WRKY33/AT2G38470 homologs, passes the OOD gate, and scores with the moderate deployable model. This is not manuscript-ready and must not be described as a positive-selection discovery until matched-null calibration and codeml/HyPhy-style reference results are interpreted.

Cycle 48 adds one-family reference/calibration helpers:

```bash
babappa install-reference-tools-plan --outdir real_empirical_pilot/reference_runs/WRKY_candidate_02_close/tool_install_plan
babappa check-reference-tools --outdir real_empirical_pilot/reference_runs/WRKY_candidate_02_close/tool_check
babappa parse-codeml-reference --codeml-dir real_empirical_pilot/reference_runs/WRKY_candidate_02_close/codeml --outdir real_empirical_pilot/reference_runs/WRKY_candidate_02_close/codeml_parsed
babappa parse-hyphy-reference --hyphy-dir real_empirical_pilot/reference_runs/WRKY_candidate_02_close/hyphy --outdir real_empirical_pilot/reference_runs/WRKY_candidate_02_close/hyphy_parsed
babappa build-reference-results-table --panel-id WRKY_candidate_02_close --codeml-parsed real_empirical_pilot/reference_runs/WRKY_candidate_02_close/codeml_parsed --hyphy-parsed real_empirical_pilot/reference_runs/WRKY_candidate_02_close/hyphy_parsed --outdir real_empirical_pilot/reference_results/WRKY_candidate_02_close
babappa run-simulation-matched-null-calibration --plan-dir real_empirical_pilot/babappa_run_wrky_close_raw_alignmentaware/per_family/WRKY_candidate_02_close/simulation_matched_calibration_plan --deployable-model-package deployable_model_conservative_branch_site_100k_mps --outdir real_empirical_pilot/calibration_runs/WRKY_candidate_02_close_null100 --n-replicates 100 --device auto --seed 42
babappa validate-simulation-matched-null-calibration --calibration-dir real_empirical_pilot/calibration_runs/WRKY_candidate_02_close_null100
babappa make-wrky-reference-calibration-report --evidence-pack real_empirical_pilot/evidence_packs/WRKY_candidate_02_close --babappa-panel-run real_empirical_pilot/babappa_run_wrky_close_raw_alignmentaware --reference-results real_empirical_pilot/reference_results/WRKY_candidate_02_close/reference_results.tsv --comparison-dir real_empirical_pilot/comparison/WRKY_candidate_02_close --matched-null-calibration real_empirical_pilot/calibration_runs/WRKY_candidate_02_close_null100 --outdir real_empirical_pilot/summary/WRKY_candidate_02_close_reference_calibration_report
```

If codeml or HyPhy are missing, BABAPPA records `pending_tool_missing` rather than failing. The matched-null runner supports resumable user-run scoring, but Codex only runs tiny `--fast-null-mode` smoke tests. BABAPPA does not fabricate null percentiles; p-like percentiles are written only from completed scored null replicates.

## Long-Run Handoff Policy

Codex does not execute heavy empirical calibration, broad scans, retraining, 10K/100K simulations, or long aligner/reference batches. For long-running work, BABAPPA generates reproducible scripts marked `USER-RUN ONLY`, and the user runs them locally/offline in the intended environment. The user can then return logs, summaries, and validation reports for interpretation.

For WRKY null calibration, Codex may run only a tiny `--fast-null-mode` smoke with at most 3 null replicates. The real 100-null command is handed off:

```bash
bash real_empirical_pilot/calibration_runs/WRKY_candidate_02_close_null100/run_user_wrky_null100.sh
```

## If Trees Are Missing

Plan tree building without executing it:

```bash
babappa plan-real-pilot-tree-building \
  --workspace real_empirical_pilot \
  --manifest real_empirical_pilot_panel.tsv \
  --outdir real_empirical_pilot/tree_building_plan \
  --method iqtree
```

For sequence-download and IQ-TREE preparation, see generated real-pilot scripts and the docs in `real_empirical_pilot/REAL_INPUT_GUIDE.md`.

## Tiny Smoke Test

Use the built-in empirical smoke data to check the pipeline:

```bash
babappa validate-empirical-input --cds-fasta tests/data/empirical_smoke/tiny_empirical.cds.fasta --tree tests/data/empirical_smoke/tiny_empirical.treefile --foreground taxon1 --outdir empirical_input_smoke
babappa run-empirical-alignment-ensemble --cds-fasta tests/data/empirical_smoke/tiny_empirical.cds.fasta --tree tests/data/empirical_smoke/tiny_empirical.treefile --foreground taxon1 --outdir empirical_alignment_smoke --methods identity,mafft,babappalign,muscle --require-babappalign true --threads 4
babappa extract-empirical-branch-site-features --empirical-validation-dir empirical_input_smoke --alignment-dir empirical_alignment_smoke --deployable-model-package deployable_model_conservative_branch_site_100k_mps --outdir empirical_features_smoke --foreground taxon1
babappa audit-empirical-features --features empirical_features_smoke/empirical_branch_site_features.tsv --deployable-model-package deployable_model_conservative_branch_site_100k_mps --outdir empirical_feature_audit_smoke
babappa empirical-applicability --empirical-validation-dir empirical_input_smoke --empirical-feature-dir empirical_features_smoke --deployable-model-package deployable_model_conservative_branch_site_100k_mps --outdir empirical_applicability_smoke
```

## Output Files To Inspect

Common outputs:

- `*_validation.json`, `*_validation.md`: pass/warning/fail checks.
- `empirical_family_prefilter.json`: p-distance, codon count, OOD decision, paralogy-risk flags.
- `empirical_applicability.json`: in-domain, borderline, or out-of-domain status.
- `empirical_branch_site_scores.tsv`: diagnostic branch-site scores if scoring succeeds.
- `empirical_gene_support.tsv`: gene-level support summary.
- `simulation_matched_calibration_plan.md`: calibration plan; not executed automatically.
- `empirical_pilot_panel_summary.md`: panel-level summary and claim-boundary text.

## Troubleshooting

- Missing BABAPPAlign model: run the cache command in the External Dependencies section.
- `babappalign` unavailable: install it in the active environment and run `babappa check-aligners`.
- MPS unavailable: use `--device cpu`, or install a PyTorch build with MPS support.
- Very high p-distance: choose closer taxa; BABAPPA should mark such families OOD or diagnostic-only.
- Tree tips do not match FASTA IDs: sanitize IDs and regenerate or repair the tree.
- Internal stop codons: repair the CDS set or exclude the sequence.
- Most families OOD: redesign the pilot around closer species and likely orthologs.

## More Documentation

- Apple Silicon/MPS: `docs/APPLE_SILICON_MPS.md`
- Deployable package: `docs/DEPLOYABLE_MODEL_PACKAGE.md`
- Empirical pilot panels: `docs/EMPIRICAL_PILOT_PANEL.md`
- Simulation-matched calibration: `docs/SIMULATION_MATCHED_EMPIRICAL_CALIBRATION.md`
- OOD-aware family selection: `docs/OOD_AWARE_EMPIRICAL_FAMILY_SELECTION.md`
- Empirical transition plan: `docs/POST_100K_EMPIRICAL_TRANSITION_PLAN.md`

## Developer Cycle Notes

The sections below preserve historical development notes and smoke-test recipes. End users should normally follow the manual above.

Historical validation note: Branch-conditioned 10K streamed validation completed, but Branch-conditioned labels may be proxy-derived only for older simulations or explicit fallback modes. Current guarded validation prioritizes explicit branch-site simulator truth. Final 100K is deferred until explicit branch-truth validation passes.

## Cycle 2 Simulator Smoke Test

Run a small deterministic simulation:

```bash
babappa simulate --outdir sim_smoke --n-families 3 --n-taxa 6 --n-codons 60 --seed 42 --positive-rate 0.5 --saturation-tier moderate
```

Validate the generated simulation directory:

```bash
babappa validate-sim --sim-dir sim_smoke
```

Generated outputs include:

- `*.fasta`: extant coding sequences for each simulated family
- `*.treefile`: simple balanced Newick tree
- `*.truth.json`: positive-selection truth labels and selected sites
- `*.homology.tsv`: codon-position homology map
- `*.events.tsv`: mutation event log
- `*.meta.json`: per-family simulator metadata
- `manifest.json`: top-level output manifest

This simulator is an initial lightweight implementation. It is not yet the final saturation-aware codon-likelihood simulator described in the manuscript.

## Cycle 3 Simulation Audit

Generate a small simulation:

```bash
babappa simulate --outdir sim_smoke --n-families 3 --n-taxa 6 --n-codons 60 --seed 42 --positive-rate 0.5 --saturation-tier moderate
```

Validate its required file structure:

```bash
babappa validate-sim --sim-dir sim_smoke
```

Run dataset-level QC:

```bash
babappa audit-sim --sim-dir sim_smoke --outdir sim_smoke/audit
```

Audit outputs:

- `family_audit.tsv`: one row per family with sequence-length, event, p-distance, saturation, and warning diagnostics
- `dataset_summary.json`: dataset-level QC summary with status counts, positive-family counts, saturation-tier counts, and p-distance ranges

For large simulations, users should run:

```bash
babappa simulate --outdir sim_large --n-families 10000 --n-taxa 16 --n-codons 300 --seed 42 --positive-rate 0.5 --saturation-tier moderate
babappa audit-sim --sim-dir sim_large --outdir sim_large/audit
```

Do not run large simulations during smoke-test development cycles.

## Cycle 4 Alignment-Ensemble Scaffold

Generate a small simulation:

```bash
babappa simulate --outdir sim_smoke --n-families 3 --n-taxa 6 --n-codons 60 --seed 42 --positive-rate 0.5 --saturation-tier moderate
```

Audit the simulated dataset:

```bash
babappa audit-sim --sim-dir sim_smoke --outdir sim_smoke/audit
```

Create internal alignment scaffold channels:

```bash
babappa align-sim --sim-dir sim_smoke --outdir align_smoke --methods identity,codon_dropout --seed 42 --dropout-rate 0.02
```

Validate the alignment directory:

```bash
babappa validate-align --align-dir align_smoke
```

The `identity` method is a truth-preserving internal alignment scaffold that copies equal-length simulated CDS records as codon alignments. The `codon_dropout` method is a simple codon-level perturbation channel that replaces whole codons with `---` without introducing frameshifts. External aligners will be integrated in a later cycle.

This module exists to stabilize the alignment-output schema before real aligners are added.

## Cycle 5 Tensorization

Generate a small simulation:

```bash
babappa simulate --outdir sim_smoke --n-families 3 --n-taxa 6 --n-codons 60 --seed 42 --positive-rate 0.5 --saturation-tier moderate
```

Create internal alignment scaffold channels:

```bash
babappa align-sim --sim-dir sim_smoke --outdir align_smoke --methods identity,codon_dropout --seed 42 --dropout-rate 0.02
```

Build deterministic tensor shards:

```bash
babappa build-tensors --sim-dir sim_smoke --align-dir align_smoke --outdir tensors_smoke --methods identity,codon_dropout
```

Validate the tensor dataset:

```bash
babappa validate-tensors --tensor-dir tensors_smoke
```

Tensor outputs:

- `*.tensor.npz`: compressed NumPy shard storing codon-token tensors
- Channel 0 stores deterministic codon IDs
- Channel 1 stores the gap-codon indicator when enabled
- `*.labels.json`: simulator-truth labels for each family
- `tensor_audit.tsv`: tensor shape and gap-burden summary
- `tensor_manifest.json`: dataset-level tensor manifest and codon vocabulary

No neural network is trained in this cycle.

## Cycle 6 Dataset Indexing And Splits

Generate a small simulation:

```bash
babappa simulate --outdir sim_smoke --n-families 10 --n-taxa 6 --n-codons 60 --seed 42 --positive-rate 0.5 --saturation-tier moderate
```

Create alignment scaffold channels:

```bash
babappa align-sim --sim-dir sim_smoke --outdir align_smoke --methods identity,codon_dropout --seed 42 --dropout-rate 0.02
```

Build tensor shards:

```bash
babappa build-tensors --sim-dir sim_smoke --align-dir align_smoke --outdir tensors_smoke --methods identity,codon_dropout
```

Create the dataset index:

```bash
babappa index-dataset --tensor-dir tensors_smoke --outdir dataset_smoke --methods identity,codon_dropout --seed 42
```

Validate the dataset index:

```bash
babappa validate-index --index-dir dataset_smoke
```

Dataset index outputs:

- `features.tsv`: simple non-leaking gene-level features derived from tensor arrays
- `splits.tsv`: deterministic train/val/calib/test assignments
- `dataset_index.json`: split counts, method list, and index metadata

Family-level splitting keeps all methods of the same family in the same split. No neural network is trained in this cycle.

## Cycle 7 NumPy Baseline Model

Generate a small simulation:

```bash
babappa simulate --outdir sim_smoke --n-families 20 --n-taxa 6 --n-codons 60 --seed 42 --positive-rate 0.5 --saturation-tier moderate
```

Create alignment scaffold channels:

```bash
babappa align-sim --sim-dir sim_smoke --outdir align_smoke --methods identity,codon_dropout --seed 42 --dropout-rate 0.02
```

Build tensor shards:

```bash
babappa build-tensors --sim-dir sim_smoke --align-dir align_smoke --outdir tensors_smoke --methods identity,codon_dropout
```

Create the dataset index:

```bash
babappa index-dataset --tensor-dir tensors_smoke --outdir dataset_smoke --methods identity,codon_dropout --seed 42
```

Train the NumPy sanity baseline:

```bash
babappa train-baseline --dataset-dir dataset_smoke --outdir baseline_smoke --seed 42 --epochs 300 --learning-rate 0.05 --l2 0.001
```

Validate model artifacts:

```bash
babappa validate-baseline --model-dir baseline_smoke
```

This is a NumPy logistic-regression sanity baseline, not the final BABAPPA neural model. It verifies that dataset splits, labels, feature extraction, model saving, prediction, and metrics are working. Final branch-site deep learning will be added later.

## Cycle 8 Baseline Calibration

Generate a small simulation:

```bash
babappa simulate --outdir sim_smoke --n-families 30 --n-taxa 6 --n-codons 60 --seed 42 --positive-rate 0.5 --saturation-tier moderate
```

Create alignment scaffold channels:

```bash
babappa align-sim --sim-dir sim_smoke --outdir align_smoke --methods identity,codon_dropout --seed 42 --dropout-rate 0.02
```

Build tensor shards:

```bash
babappa build-tensors --sim-dir sim_smoke --align-dir align_smoke --outdir tensors_smoke --methods identity,codon_dropout
```

Create the dataset index:

```bash
babappa index-dataset --tensor-dir tensors_smoke --outdir dataset_smoke --methods identity,codon_dropout --seed 42
```

Train the NumPy sanity baseline:

```bash
babappa train-baseline --dataset-dir dataset_smoke --outdir baseline_smoke --seed 42 --epochs 300 --learning-rate 0.05 --l2 0.001
```

Calibrate probabilities and select an empirical threshold:

```bash
babappa calibrate-baseline --model-dir baseline_smoke --outdir calibration_smoke --target-fdr 0.10 --calibration-method temperature
```

Validate calibration artifacts:

```bash
babappa validate-calibration --calibration-dir calibration_smoke
```

Calibration uses the held-out calibration split. Temperature scaling is used as a simple probability calibration method, and an empirical threshold is selected to satisfy the target FDR on the calibration split when possible.

This is still the NumPy baseline, not the final neural BABAPPA model.

## Cycle 9 Consolidated Reporting

Generate a consolidated report from existing BABAPPA outputs:

```bash
babappa make-report \
  --outdir report_smoke \
  --title "BABAPPA smoke report" \
  --sim-dir sim_smoke \
  --sim-audit-dir sim_smoke/audit \
  --align-dir align_smoke \
  --tensor-dir tensors_smoke \
  --dataset-dir dataset_smoke \
  --baseline-dir baseline_smoke \
  --calibration-dir calibration_smoke
```

Validate report artifacts:

```bash
babappa validate-report --report-dir report_smoke
```

Report outputs:

- `report_summary.json`: machine-readable summary of discovered artifacts, QC summaries, splits, baseline metrics, calibration settings, thresholding, warnings, and generated files
- `report.md`: human-readable run report intended for transparent review and reproducibility

The report documents QC, splits, baseline metrics, calibration, thresholding, and limitations. It does not claim final biological performance.

## Cycle 10 Neural Infrastructure

Inspect the optional PyTorch environment:

```bash
babappa check-neural-env
```

Inspect neural dataset rows without requiring PyTorch:

```bash
babappa inspect-neural-data \
  --dataset-dir dataset_smoke \
  --split train \
  --methods identity,codon_dropout \
  --max-items 8
```

Create a small neural-data batch if PyTorch is available:

```bash
babappa smoke-neural-batch \
  --dataset-dir dataset_smoke \
  --split train \
  --methods identity,codon_dropout \
  --batch-size 4
```

Cycle 10 does not train a model. PyTorch is optional at the package level and is not installed automatically. `check-neural-env` reports CUDA, MPS, and CPU availability. `inspect-neural-data` works without PyTorch. `smoke-neural-batch` requires PyTorch and verifies batch loading before real training is implemented.

## Cycle 11 Neural Smoke Training

Check the optional PyTorch environment:

```bash
babappa check-neural-env
```

Train the minimal gene-level neural smoke classifier:

```bash
babappa train-neural-smoke \
  --dataset-dir dataset_smoke \
  --outdir neural_smoke \
  --device auto \
  --methods identity,codon_dropout \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 0.001
```

Validate neural smoke artifacts:

```bash
babappa validate-neural-smoke --model-dir neural_smoke
```

Cycle 11 adds the first PyTorch training loop. It trains a minimal gene-level smoke classifier and verifies device movement, checkpointing, prediction, and metrics before larger-scale training.

This is not the final BABAPPA branch-site neural architecture. Alignment-ensemble branch-site inference, artifact heads, saturation-risk heads, and reliability/abstention logic will come later.

## Cycle 12 Scale-Ready Neural Training

Train the scale-ready gene-level neural model:

```bash
babappa train-neural \
  --dataset-dir dataset_smoke \
  --outdir neural_train \
  --device auto \
  --methods identity,codon_dropout \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --embedding-dim 32 \
  --hidden-dim 64 \
  --dropout 0.1 \
  --early-stopping-patience 8 \
  --monitor-metric val_loss
```

Validate scale-ready neural artifacts:

```bash
babappa validate-neural --model-dir neural_train
```

Cycle 12 adds the first scale-ready neural trainer. It is still gene-level, but it saves best and last checkpoints, supports early stopping, logs training history, and evaluates train, validation, calibration, and test splits.

This is the correct command family for 1000-family and later 10k-family scale tests. The final BABAPPA branch-site architecture will be implemented later.

## Cycle 13 Neural Calibration

Train the current gene-level neural model:

```bash
babappa train-neural \
  --dataset-dir dataset_smoke \
  --outdir neural_train \
  --device auto \
  --methods identity,codon_dropout \
  --epochs 30 \
  --batch-size 32
```

Calibrate neural probabilities and select an empirical threshold:

```bash
babappa calibrate-neural \
  --model-dir neural_train \
  --outdir neural_calibration \
  --target-fdr 0.10 \
  --calibration-method temperature
```

Validate neural calibration artifacts:

```bash
babappa validate-neural-calibration \
  --calibration-dir neural_calibration
```

Generate a report that includes neural training and calibration:

```bash
babappa make-report \
  --outdir report_smoke \
  --title "BABAPPA neural report" \
  --dataset-dir dataset_smoke \
  --neural-dir neural_train \
  --neural-calibration-dir neural_calibration
```

Neural calibration uses the held-out calibration split. Temperature scaling is used as a simple calibration method, and the operating threshold is selected by empirical FDR when possible.

This is still gene-level neural calibration, not final branch-site inference.

## Cycle 14 Run Summary and Model Comparison

Create a compact diagnostic overview for an existing BABAPPA run:

```bash
babappa summarize-run \
  --outdir run_summary_smoke \
  --title "BABAPPA smoke run summary" \
  --dataset-dir dataset_smoke \
  --neural-dir neural_train \
  --neural-calibration-dir neural_calibration \
  --report-dir report_smoke
```

Validate the run summary:

```bash
babappa validate-run-summary --summary-dir run_summary_smoke
```

Compare baseline and neural raw/calibrated metrics:

```bash
babappa compare-models \
  --outdir model_compare_smoke \
  --baseline-metrics baseline_smoke/baseline_metrics.json \
  --baseline-calibrated-metrics calibration_smoke/baseline_calibrated_metrics.json \
  --neural-metrics neural_train/neural_metrics.json \
  --neural-calibrated-metrics neural_calibration/neural_calibrated_metrics.json
```

Validate the comparison artifacts:

```bash
babappa validate-model-comparison --compare-dir model_compare_smoke
```

`summarize-run` provides a compact diagnostic overview of a full BABAPPA run. `compare-models` compares baseline and neural raw/calibrated metrics across train, validation, calibration, test, and all splits.

These commands should be used after 1000-family and 10k-family experiments. They help inspect workflow completeness and metric differences, but they do not claim final biological performance.

## Cycle 15 Prediction Diagnostics and Neural v2

Diagnose prediction collapse or weak threshold behavior:

```bash
babappa diagnose-predictions \
  --predictions neural_train_1000/predictions/neural_predictions.tsv \
  --metrics neural_train_1000/neural_metrics.json \
  --calibration neural_calibration_1000/neural_calibration.json \
  --outdir neural_diag_1000 \
  --model-name neural_v1
```

Validate diagnostic artifacts:

```bash
babappa validate-prediction-diagnostics --diag-dir neural_diag_1000
```

Train the improved gene-level neural v2 model:

```bash
babappa train-neural-v2 \
  --dataset-dir dataset_neural_1000 \
  --outdir neural_v2_1000 \
  --device auto \
  --methods identity,codon_dropout \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --embedding-dim 32 \
  --hidden-dim 64 \
  --dropout 0.1 \
  --early-stopping-patience 8 \
  --monitor-metric val_loss
```

Neural v1 collapsed into all-negative thresholded predictions in the first 1000-family test. `diagnose-predictions` summarizes score distributions and threshold curves so collapse, weak separation, inverted signal, and calibration limitations are visible.

Neural v2 uses contrastive pooling and automatic positive-class weighting. This is still a gene-level repair cycle, not final branch-site BABAPPA inference.

## Cycle 16 Threshold-Policy Profiling

Profile neural v2 operating points on existing predictions:

```bash
babappa threshold-policy \
  --predictions neural_v2_1000/predictions/neural_predictions.tsv \
  --outdir neural_v2_policy_1000 \
  --probability-column prob_positive \
  --selection-split calib \
  --target-fdr 0.10 \
  --precision-floor 0.80 \
  --recall-floor 0.80 \
  --model-name neural_v2_1000
```

Validate threshold-policy artifacts:

```bash
babappa validate-threshold-policy --policy-dir neural_v2_policy_1000
```

`threshold-policy` writes a threshold curve plus operating-point profiles for `default_0_5`, `strict_fdr`, `max_f1`, `max_mcc`, `balanced_youden`, `high_precision`, and `high_recall`.

Strict FDR thresholding is useful for conservative discovery but may have low recall. Max-F1 and MCC profiles are useful for classification diagnostics. The high-recall profile is useful for screening. Threshold-policy reports help decide whether to proceed to larger datasets or improve model architecture.

This is still gene-level BABAPPA. It is not the final branch-site architecture.

## Cycle 17 Stratified Evaluation

Evaluate neural v2 predictions by split, saturation tier, alignment method, and their combinations:

```bash
babappa stratified-eval \
  --predictions neural_v2_1000/predictions/neural_predictions.tsv \
  --outdir neural_v2_stratified_1000 \
  --model-name neural_v2_1000 \
  --probability-column prob_positive \
  --threshold-policy-dir neural_v2_policy_1000
```

Validate stratified evaluation artifacts:

```bash
babappa validate-stratified-eval --eval-dir neural_v2_stratified_1000
```

`stratified-eval` reports performance by split, saturation tier, method, split x saturation, split x method, saturation x method, and split x saturation x method. When a threshold-policy directory is supplied, every selected operating-point profile is evaluated.

This evaluation layer is required before adding saturation-risk heads. Current saturation tiers are simulator-level labels, and current alignment methods remain identity/codon_dropout scaffolds.

## Cycle 18 Multi-Saturation Panels

Create a small four-tier saturation panel:

```bash
babappa make-saturation-panel \
  --outdir saturation_panel_smoke \
  --n-families-per-tier 3 \
  --tiers low,moderate,high,extreme \
  --n-taxa 6 \
  --n-codons 60 \
  --seed 42 \
  --positive-rate 0.5 \
  --methods identity,codon_dropout
```

Validate the panel:

```bash
babappa validate-saturation-panel --panel-dir saturation_panel_smoke
```

Merge tier-specific dataset indexes into one trainable dataset:

```bash
babappa merge-datasets \
  --dataset-dirs saturation_panel_smoke/tiers/low/dataset,saturation_panel_smoke/tiers/moderate/dataset,saturation_panel_smoke/tiers/high/dataset,saturation_panel_smoke/tiers/extreme/dataset \
  --names low,moderate,high,extreme \
  --outdir dataset_saturation_smoke \
  --seed 42 \
  --resplit
```

Validate the merged dataset:

```bash
babappa validate-merged-dataset --dataset-dir dataset_saturation_smoke
```

`make-saturation-panel` creates separate tier-specific simulations, audits, alignments, tensors, and dataset indexes. `merge-datasets` combines those indexes into one trainable dataset while preserving `saturation_tier`, source dataset, original family ID, and resolvable tensor paths.

This enables future neural training and stratified evaluation across low, moderate, high, and extreme saturation. It is still a gene-level benchmark substrate, not the final branch-site BABAPPA architecture.

## Cycle 19 Saturation-Aware Neural Training

Train the saturation-aware gene-level neural v3 model on a merged multi-saturation dataset:

```bash
babappa train-neural-saturation \
  --dataset-dir dataset_saturation_1000 \
  --outdir neural_v3_saturation_1000 \
  --device auto \
  --methods identity,codon_dropout \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --embedding-dim 32 \
  --hidden-dim 64 \
  --dropout 0.1 \
  --early-stopping-patience 8 \
  --monitor-metric val_loss
```

Run saturation-stratified calibration:

```bash
babappa calibrate-stratified \
  --predictions neural_v3_saturation_1000/predictions/neural_predictions.tsv \
  --outdir neural_v3_saturation_stratified_calibration_1000 \
  --group-column saturation_tier \
  --target-fdr 0.10
```

Validate the stratified calibration artifacts:

```bash
babappa validate-stratified-calibration \
  --calibration-dir neural_v3_saturation_stratified_calibration_1000
```

Saturation-aware training uses tier IDs, a saturation embedding, optional saturation-balanced sampling, and optional inverse-frequency group-weighted loss. Stratified calibration fits per-tier temperature and threshold settings when the calibration split is large enough, otherwise it falls back to global calibration with explicit warnings.

This is still gene-level BABAPPA. It is a required saturation-robustness repair step before final branch-site architecture, artifact heads, saturation-risk heads, and reliability/abstention logic.

## Cycle 20 Neural Ablation and Repair

Diagnose an existing neural run:

```bash
babappa diagnose-neural \
  --model-dir neural_v3_saturation_1000 \
  --outdir neural_v3_diag_1000 \
  --model-name neural_v3_saturation_1000
```

Train one controlled ablation variant:

```bash
babappa train-neural \
  --dataset-dir dataset_saturation_1000 \
  --outdir neural_ablation_embed_only_1000 \
  --training-preset saturation_embed_only \
  --device auto \
  --epochs 30
```

Compare ablation runs:

```bash
babappa compare-ablations \
  --outdir ablation_compare_1000 \
  --model-dirs neural_v2_saturation_1000,neural_v3_saturation_1000 \
  --names v2_contrastive,v3_saturation_full
```

Cycle 20 exists because the full saturation-aware v3 model underperformed contrastive v2 on the 4-tier benchmark. The immediate scientific question is which component harmed learning: saturation embedding, saturation inverse-frequency loss, or saturation-balanced sampling.

`train-neural --training-preset` supports controlled presets including `contrastive_v2`, `saturation_embed_only`, `saturation_group_weight_only`, `saturation_sampler_only`, `saturation_full_v3`, `contrastive_class_weighted`, and `contrastive_unweighted`.

Do not proceed to 10k or final branch-site inference until the ablation matrix identifies a stable gene-level configuration.

## Cycle 21 Ranking-Aware Repair and Label-Signal Audit

Audit whether the current dataset-level features contain simple label signal:

```bash
babappa audit-label-signal \
  --dataset-dir dataset_saturation_1000 \
  --outdir label_signal_saturation_1000
```

Train the ranking-aware site-attention repair model:

```bash
babappa train-neural-ranking \
  --dataset-dir dataset_saturation_1000 \
  --outdir neural_ranking_site_attention_1000 \
  --device auto \
  --methods identity,codon_dropout \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --rank-weight 0.2
```

Ranking-aware loss directly pressures positive examples to score above negative examples within a batch. The site-attention model is intended to recover sparse codon-site signals that can be washed out by global pooling.

The label-signal audit checks simple tensor-derived features before attributing weak neural performance to architecture alone. Weak univariate feature signal does not prove tensors are unlearnable, but it suggests sparse or deeper representation is needed.

This remains gene-level BABAPPA. It is not final branch-site inference.

## Cycle 22 Leakage Audit and Stability Benchmark

Audit feature tables for truth-derived leakage columns before training feature-based models:

```bash
babappa audit-leakage \
  --dataset-dir dataset_saturation_1000 \
  --outdir leakage_audit_saturation_1000
```

Run a repeated-seed stability benchmark across repaired neural presets:

```bash
babappa stability-benchmark \
  --dataset-dir dataset_saturation_1000 \
  --outdir stability_benchmark_1000 \
  --seeds 42,43,44 \
  --presets contrastive_v2,saturation_embed_only,site_attention_ranked \
  --device auto \
  --epochs 10
```

The leakage audit prevents truth-derived columns such as `n_selected_sites`, `selected_sites`, labels, truth labels, and foreground metadata from contaminating baselines or future feature-based models. Neural tensor models do not consume these feature columns directly, but reports still flag them because they can bias baseline comparisons.

The stability benchmark creates repeated deterministic resplits and compares presets across seeds. It should be used to decide whether weak AUROC reflects architecture, split instability, or insufficient learnable signal. Do not scale to 10k until leakage is excluded and stability is established.

## Cycle 23 Site-Level Oracle Learning

Extract oracle selected-site targets from an existing tensor dataset:

```bash
babappa extract-site-labels \
  --dataset-dir dataset_saturation_1000 \
  --outdir site_oracle_saturation_1000
```

Build a site-level feature table while preserving split, saturation, method, and foreground context:

```bash
babappa build-site-dataset \
  --dataset-dir dataset_saturation_1000 \
  --oracle-labels site_oracle_saturation_1000/site_oracle_labels.tsv \
  --outdir site_dataset_saturation_1000 \
  --negative-downsample-ratio 20 \
  --seed 42
```

Audit the site-level dataset for oracle leakage:

```bash
babappa audit-site-leakage \
  --site-dataset-dir site_dataset_saturation_1000 \
  --outdir site_leakage_saturation_1000
```

Train a minimal NumPy site-level baseline:

```bash
babappa train-site-baseline \
  --site-dataset-dir site_dataset_saturation_1000 \
  --outdir site_baseline_saturation_1000
```

Site-level oracle labels are supervised targets, not inference inputs. This cycle pivots away from unstable whole-gene binary classification toward local site-level signal extraction. Foreground information is treated as biological context and is audited separately from target leakage.

This remains simulation-supervised method development. It is not real empirical inference yet, and it is not the final branch-site BABAPPA architecture.

## Cycle 24 Site-Level Neural and Aggregation

Train a site-level neural classifier on the oracle site dataset:

```bash
babappa train-site-neural \
  --site-dataset-dir site_dataset_saturation_1000 \
  --outdir site_neural_saturation_1000 \
  --device auto \
  --epochs 30 \
  --batch-size 256 \
  --monitor-metric val_auroc
```

Calibrate site-level probabilities:

```bash
babappa calibrate-site-neural \
  --model-dir site_neural_saturation_1000 \
  --outdir site_neural_calibration_saturation_1000 \
  --target-fdr 0.10
```

Profile site-level operating points:

```bash
babappa site-threshold-policy \
  --predictions site_neural_saturation_1000/site_neural_predictions.tsv \
  --outdir site_neural_policy_saturation_1000
```

Evaluate site performance by split, saturation tier, and method:

```bash
babappa site-stratified-eval \
  --predictions site_neural_saturation_1000/site_neural_predictions.tsv \
  --outdir site_neural_stratified_saturation_1000 \
  --threshold-policy-dir site_neural_policy_saturation_1000
```

Aggregate site probabilities into gene/family-level support:

```bash
babappa aggregate-sites \
  --predictions site_neural_saturation_1000/site_neural_predictions.tsv \
  --gene-dataset-dir dataset_saturation_1000 \
  --outdir site_to_gene_saturation_1000
```

Site-level neural learning follows the successful site-level baseline. Site-to-gene aggregation is the intended route to family-level support: gene-level inference should be derived from local site evidence rather than direct whole-gene classification.

This remains oracle-supervised simulation development and is not empirical branch-site inference.

## Cycle 25 Site Robustness and Calibration Repair

Compare the transparent NumPy site baseline against the site-level neural scorer:

```bash
babappa compare-site-models \
  --site-baseline-dir site_baseline_saturation_1000 \
  --site-neural-dir site_neural_saturation_1000 \
  --site-stratified-eval-dir site_neural_stratified_saturation_1000 \
  --site-aggregation-dir site_to_gene_saturation_1000 \
  --outdir site_model_compare_saturation_1000
```

Run null and decoy controls for site-to-gene aggregation:

```bash
babappa aggregation-controls \
  --predictions site_neural_saturation_1000/site_neural_predictions.tsv \
  --gene-dataset-dir dataset_saturation_1000 \
  --outdir site_aggregation_controls_saturation_1000 \
  --n-permutations 50 \
  --seed 42
```

Select operating points at the aggregation level:

```bash
babappa aggregation-threshold-policy \
  --aggregation-dir site_to_gene_saturation_1000 \
  --outdir site_to_gene_policy_saturation_1000 \
  --score-column max_site_probability \
  --label-column gene_label \
  --selection-split calib \
  --target-fdr 0.10
```

Try nonparametric quantile calibration and compare calibration methods:

```bash
babappa calibrate-site-neural \
  --model-dir site_neural_saturation_1000 \
  --outdir site_neural_calibration_quantile_saturation_1000 \
  --target-fdr 0.10 \
  --calibration-method quantile \
  --n-bins 20

babappa compare-site-calibrations \
  --calibration-dirs site_neural_calibration_saturation_1000,site_neural_calibration_quantile_saturation_1000 \
  --names temperature,quantile \
  --outdir site_calibration_compare_saturation_1000
```

Run a capped repeated-seed site-neural stability benchmark:

```bash
babappa site-stability-benchmark \
  --site-dataset-dir site_dataset_saturation_1000 \
  --outdir site_stability_saturation_1000 \
  --seeds 42,43,44 \
  --device auto \
  --epochs 10 \
  --batch-size 256 \
  --max-train-items 50000 \
  --max-val-items 10000 \
  --max-calib-items 10000 \
  --max-test-items 10000
```

Site neural is strong, but robustness across seeds and aggregation null controls are required before release claims. Strict site-level FDR remains difficult; aggregation-level thresholds may be more relevant for gene-level discovery. Perfect site-to-gene AUROC should be treated as an oracle-simulation upper bound until it exceeds decoy controls.

## Cycle 26 Large-Run Planning

Generate a 10K validation plan without executing the benchmark:

```bash
babappa plan-large-run \
  --scale 10000 \
  --families-per-tier 2500 \
  --outdir large_run_plan_10k \
  --negative-downsample-ratio 10

babappa validate-large-run-plan \
  --plan-dir large_run_plan_10k
```

Generate the final 100K benchmark plan after the 10K run has been inspected:

```bash
babappa plan-large-run \
  --scale 100000 \
  --families-per-tier 25000 \
  --outdir large_run_plan_100k \
  --negative-downsample-ratio 5

babappa validate-large-run-plan \
  --plan-dir large_run_plan_100k
```

At Cycle 26, the 10K run was the next validation scale and the 100K run was the final benchmark scale. The fast external-aligner 10K validation is now complete, so final 100K planning should wait until the branch-conditioned 10K validation passes. Codex prepares command templates, expected row counts, monitoring commands, and validation checklists only; the user runs these jobs offline. Generated `large_run_commands.sh` files are executable user-run scripts marked `USER-RUN ONLY — DO NOT EXECUTE IN CODEX`; `large_run_commands_commented_reference.sh` keeps a fully commented reference copy.

## Cycle 27 External Aligner Injection

Inspect optional external alignment backends:

```bash
babappa check-aligners
```

BABAPPAlign also needs its BABAPPAScore model cache. `check-aligners` reports the expected path, whether the model is present, and an install command. A direct smoke check is available:

```bash
babappa smoke-aligner --method babappalign --outdir aligner_smoke
```

If the model is missing:

```bash
mkdir -p "$HOME/.cache/babappalign/models"
curl -L "https://zenodo.org/record/18053201/files/babappascore.pt" -o "$HOME/.cache/babappalign/models/babappascore.pt"
```

Run a tiny mixed internal/external alignment ensemble without requiring external tools:

```bash
babappa align-external \
  --sim-dir sim_aligner_smoke_c27 \
  --outdir align_external_smoke_c27 \
  --methods identity,mafft,babappalign,muscle \
  --require-available false \
  --threads 1
```

Build and validate aligned-site to original-site maps:

```bash
babappa build-site-map \
  --sim-dir sim_aligner_smoke_c27 \
  --align-dir align_external_smoke_c27 \
  --outdir site_map_external_smoke_c27

babappa validate-site-map \
  --site-map-dir site_map_external_smoke_c27
```

Use mapped site labels when external aligners insert or shift codon columns:

```bash
babappa extract-site-labels \
  --dataset-dir dataset_external_smoke_c27 \
  --outdir site_oracle_external_smoke_c27 \
  --site-map-dir site_map_external_smoke_c27 \
  --aligned-site-mode mapped

babappa build-site-dataset \
  --dataset-dir dataset_external_smoke_c27 \
  --oracle-labels site_oracle_external_smoke_c27/site_oracle_labels.tsv \
  --outdir site_dataset_external_smoke_c27 \
  --require-mappable-sites
```

External aligners are optional. MAFFT, PRANK, and BABAPPAlign executables are detected with `shutil.which`; BABAPPAlign additionally fails fast with `babappalign_model_missing` when its model cache is absent, unless the user explicitly passes `--allow-missing-babappalign`. Site-map preservation is required for oracle site labels: site-level training must attach `y_site` to aligned codon coordinates through the original simulated site index. Aligner ensembles should be tested at 1K before any 10K external-aligner validation.

## Cycle 29 Fast External-Aligner Recovery

Cycle 29 moves external-aligner validation onto a feasible fast-ensemble track.

Default fast mapped-site ensemble:

```bash
identity,mafft,babappalign,muscle
```

Policy decisions:

- `identity`, `mafft`, `babappalign`, and `muscle` are production-default methods for fast external validation.
- `muscle` is optional at runtime and is skipped gracefully if unavailable unless `--require-available true` is used.
- `prank` remains implemented but is diagnostic-only and excluded from default planners.
- `tcoffee`/`t_coffee` remains optional diagnostic and is excluded unless explicitly requested.
- External workflows should run `aligner-method-policy` after site-map construction and tensorize only methods marked `usable` or `caution`.

Plan a fast external-aligner validation without executing it:

```bash
babappa plan-external-aligner-validation \
  --panel-dir saturation_panel_external_1k \
  --outdir external_aligner_validation_plan_fast_1k \
  --methods identity,mafft,babappalign,muscle \
  --tiers low,moderate,high,extreme \
  --negative-downsample-ratio 10
```

Complete reports for already generated low/moderate/high external tiers without running heavy alignment jobs:

```bash
babappa plan-complete-external-tier-reports \
  --tiers low,moderate,high \
  --outdir external_completed_tier_report_plan
```

Plan recovery of the missing external extreme tier without executing it:

```bash
babappa plan-external-extreme-recovery \
  --panel-dir saturation_panel_external_1k \
  --outdir external_extreme_recovery_plan \
  --methods identity,mafft,babappalign,muscle \
  --negative-downsample-ratio 10 \
  --timeout-seconds 300
```

Generated scripts are marked `USER-RUN ONLY` and must be launched by the user, not by Codex. Do not scale external-aligner workflows to 10K until method policy, calibration, aggregation threshold policies, and the missing extreme tier decision have been reviewed.

## Cycle 32 Branch-Site Research-Alpha Transition

BABAPPA `0.4.0-alpha` starts the branch-conditioned branch-site research-alpha layer on top of the validated fast external-aligner 10K pipeline.

The completed external 10K remains a site-evidence validation: site-level oracle learning plus site-to-gene aggregation. It must not be described as empirical branch-site inference.

Branch-conditioned validation is the next step:

```bash
babappa plan-branch-conditioned-10k \
  --outdir branch_conditioned_10k_plan
```

The generated plan reuses completed fast external 10K datasets and site maps where possible. It preserves the production-fast method policy of `identity`, `mafft`, `babappalign`, and MUSCLE with method-policy quarantine; PRANK and T-Coffee remain diagnostic only.

The branch-conditioned commands add branch-site oracle labels, branch-aware datasets, leakage audits, NumPy and lightweight neural baselines, calibration, threshold policies, branch/site aggregation, decoy controls, validators, and run summaries. If explicit simulator branch-site truth is unavailable, labels are reported as foreground-taxon proxy labels rather than silently fabricated.

Cycle 32B makes branch-site dataset construction memory-safe: `build-branch-site-dataset` streams label rows, downsamples negatives before tensor feature extraction, and supports hard caps such as `--max-output-rows`. The branch-conditioned 10K plan uses `_streamed` output directories, `--negative-downsample-ratio 5`, and `--max-output-rows 1000000` per tier by default.

Final 100K should wait until branch-conditioned 10K validation passes.

## Storage Cleanup and Repository Slimming

BABAPPA heavy validation runs can create very large reproducible intermediates. Do
not delete them by hand without first producing an inventory.

Dry-run audit:

```bash
babappa audit-storage \
  --root . \
  --outdir storage_cleanup_audit \
  --target-size-gb 10
```

The audit writes:

- `storage_inventory.tsv/json`
- `keep_list.tsv`
- `remove_candidates.tsv`
- `archive_candidates.tsv`
- `cleanup_dry_run.md`
- `du_top_100.txt`
- user-run scripts for quarantine, archive, deletion after review, and validation

The generated quarantine script moves reproducible candidates into a dated folder
under the home directory and does not permanently delete anything:

```bash
bash storage_cleanup_audit/quarantine_large_reproducible_outputs.sh
```

After inspecting the quarantine, run the lightweight validator:

```bash
bash storage_cleanup_audit/validate_after_cleanup.sh
```

Permanent deletion is intentionally separated into
`storage_cleanup_audit/delete_quarantine_after_review.sh` and requires
`CONFIRM_DELETE=YES`. Keep source, tests, docs, deployable package artifacts,
final reports, WRKY evidence packs, and user-created empirical CDS/tree inputs.
