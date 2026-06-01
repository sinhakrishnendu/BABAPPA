# BABAPPA

BABAPPA is the **Branch-site Alignment-Bias-Aware Probabilistic Positive-selection Analyzer**.

Current source version: `v0.8.0`  
Release archive label: `v0.8.0`  
Status: research-alpha, simulation-trained, standalone BABAPPA-native calibrated evidence workflow

BABAPPA supports branch-site positive-selection investigation from a user-supplied codon MSA and treefile. The main user-facing command treats the supplied MSA as the authoritative alignment, scores requested foreground branches, and reports candidate branch-site episodic-selection support using a deployable simulation-trained model plus a BABAPPA-native empirical null calibration. Alignment ensembles and codeml/HyPhy comparison are optional diagnostic comparators, not dependencies for BABAPPA to issue its own calibrated evidence statement.

BABAPPA is intended to become a standalone complementary software system beside codeml and HyPhy. It does **not** claim likelihood-model equivalence to those tools, and it does not use their null models internally. Instead, BABAPPA reports BABAPPA-native calibrated support classes from its own simulation-trained scoring model and empirical feature-null calibration. For publication, users should report the BABAPPA evidence class, native null replicate count, p-like values, OOD/applicability status, and biological context.

Version `v0.8.0` makes the direct end-user workflow the central interface: supply an aligned codon MSA, supply a matching treefile, choose foreground branches, and receive branch-site predictions with aligned and de-gapped codon coordinates. It also makes CDS integrity stricter and clearer: terminal stop codons are accepted with warnings, while internal stops, frame errors, missing ATG starts, duplicate IDs, and tree/MSA label mismatches stop execution before scoring.

## Contents

- Project status and scientific boundary
- What BABAPPA does
- What BABAPPA does not do
- Installation
- Quick start
- Typical workflows
- Input requirements
- Aligners
- Output interpretation
- Reproducibility
- Storage cleanup and maintenance
- Troubleshooting
- Citation and manuscript status
- Developer notes

## Project Status And Scientific Boundary

BABAPPA has completed conservative explicit branch-truth simulation validation at 100,000 families on Apple Silicon/MPS. It has a validated deployable simulation-trained model package:

```text
deployable_model_conservative_branch_site_100k_mps
```

The deployable package validates successfully:

- status: `ok`
- failures: `0`
- warnings: `0`

The empirical bridge can process small real empirical diagnostic pilots, but BABAPPA scores are not final discovery claims.

Historical validation note: Branch-conditioned 10K streamed validation completed before the final 100K MPS run. Branch-conditioned labels may be proxy-derived in older or non-explicit workflows, so BABAPPA now distinguishes those cases from explicit branch-site simulator truth. A previous gate stated, "Final 100K is deferred until explicit branch-truth validation passes"; that gate has now been satisfied with a conditional-pass 100K explicit-truth validation. Unsupported empirical discovery language remains blocked; BABAPPA-native calibrated support can be reported when the native null and QC outputs support it.

The simulation phase is oracle-supervised because simulator truth is known during validation. That oracle-supervised evidence is never supplied as an empirical inference input.

> **Empirical interpretation warning**
>
> A raw BABAPPA diagnostic-positive score is not, by itself, a publishable empirical positive-selection claim. A manuscript-ready BABAPPA result should include BABAPPA-native null calibration, input QC/applicability status, biological controls or rationale, and the exact BABAPPA version/model package. codeml/HyPhy can be used as external comparators, but BABAPPA does not depend on them to report BABAPPA-native evidence.

## What BABAPPA Does

BABAPPA can:

- predict branch-site support directly from a user-provided aligned codon MSA and matching treefile;
- score one foreground tip, a comma-separated set of foreground tips, or all tree tips;
- validate empirical CDS FASTA and tree inputs;
- run optional alignment ensembles for diagnostic sensitivity analysis;
- construct site maps and method-policy reports;
- extract conservative empirical branch-site features;
- audit empirical feature tables for forbidden truth-derived columns;
- score branch-site rows using a packaged simulation-trained model;
- run BABAPPA-native empirical null calibration for direct MSA/tree predictions;
- report BABAPPA-native p-like values and calibrated support classes;
- classify empirical inputs as `in_domain`, `borderline`, or `out_of_domain`;
- mark OOD cases as `diagnostic_only`;
- produce guarded diagnostic reports;
- prepare and parse codeml/HyPhy-style reference workflows as optional comparators;
- plan and run conservative feature-level matched empirical calibration;
- audit storage and generate safe cleanup scripts for large reproducible outputs.

BABAPPA helps decide whether a dataset is suitable for branch-site positive-selection interpretation and provides a standalone BABAPPA evidence system. It remains research-alpha software: results should be reported as BABAPPA-native calibrated support, not as a classical likelihood-ratio test.

## What BABAPPA Does Not Do

BABAPPA does not:

- provide codeml/HyPhy-equivalent likelihood-ratio tests;
- use codeml or HyPhy internally as a required null model;
- make strong empirical claims from uncalibrated raw scores;
- use simulator truth during empirical inference;
- silently accept out-of-domain empirical inputs as positive-selection calls;
- serve as a clinical, agricultural, regulatory, or policy decision tool.

## Long-Run Handoff Policy

Codex and other assisted-maintenance sessions should not execute heavy empirical calibration, broad empirical scans, retraining, 10K/100K simulations, or long aligner/reference batches. The expected workflow is to generate reproducible USER-RUN scripts, validators, parsers, and reports; the user runs long jobs locally or offline and returns summaries/logs for interpretation.

## Installation

After PyPI release:

```bash
python -m pip install babappa
```

Clone and install from source:

```bash
git clone <REPOSITORY_URL> BABAPPA
cd BABAPPA
python -m pip install -e .
```

For neural scoring, install BABAPPA in an environment with PyTorch available, for example the `molevo` conda environment used during development. The PyPI/source package includes the lightweight deployable model package used by the default predictor.

For development and tests:

```bash
python -m pip install -e ".[dev]"
```

Check the installed version:

```bash
babappa --version
```

Expected for this release:

```text
0.8.0
```

Run tests:

```bash
python -m pytest -q
```

The full test count may change as tests are added. A release candidate should pass the full local suite before publishing.

## External Dependencies

Required Python dependencies are installed through the package. Empirical and reference workflows may also need external command-line tools:

- MAFFT
- MUSCLE
- BABAPPAlign
- optional IQ-TREE2/IQ-TREE for tree building
- optional codeml from PAML
- optional HyPhy
- optional PyTorch for deployable model scoring

Check aligners:

```bash
babappa check-aligners
```

BABAPPAlign requires the BABAPPAScore model cache:

```bash
mkdir -p "$HOME/.cache/babappalign/models"
curl -L "https://zenodo.org/record/18053201/files/babappascore.pt" -o "$HOME/.cache/babappalign/models/babappascore.pt"
```

The BABAPPAlign model is small enough to keep. Generated BABAPPAlign embedding caches can be very large and may be safely regenerated.

## Apple Silicon / MPS

Apple Silicon/MPS support is research-alpha. It is useful for smoke tests, lightweight empirical scoring, and the completed 100K MPS validation.

Recommended shell settings:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
```

Check neural environment:

```bash
babappa check-neural-env
```

Run MPS smoke:

```bash
babappa smoke-mps-training --outdir mps_smoke --device auto --batch-size 32 --max-items 512
babappa validate-mps-smoke --smoke-dir mps_smoke
```

Light benchmark:

```bash
babappa benchmark-apple-silicon --outdir apple_silicon_benchmark --device auto --batch-sizes 32,64,128 --max-items 1024
```

If MPS fails, retry the relevant scoring stage with `--device cpu` or a smaller batch size.

## Quick Start

Inspect commands:

```bash
babappa --help
```

### The Simplest Use Case

If you have exactly what BABAPPA expects, an aligned codon MSA and a matching treefile, run:

```bash
babappa predict-branch-sites \
  --msa aligned_gene.cds.fasta \
  --tree aligned_gene.treefile \
  --foreground leaves \
  --outdir aligned_gene_babappa \
  --device auto \
  --null-replicates 1000
```

This does the core job:

1. validates that the MSA is a plausible CDS alignment;
2. validates that tree tips and MSA IDs match;
3. scores every tree leaf as a foreground branch;
4. writes branch-site predictions;
5. writes de-gapped branch coordinates for easier biological interpretation;
6. runs BABAPPA-native null calibration when `--null-replicates` is greater than zero.

For a quick check before a long run:

```bash
babappa predict-branch-sites \
  --msa aligned_gene.cds.fasta \
  --tree aligned_gene.treefile \
  --foreground leaves \
  --outdir aligned_gene_babappa_dryrun \
  --dry-run
```

Launch the interactive predictor:

```bash
babappa
```

BABAPPA will ask for:

1. aligned codon MSA FASTA path
2. treefile path
3. foreground mode: `leaves`/`all`/`specific`

`leaves` is the default and scores every tree tip. `all` is accepted as the same thing for direct tip-branch scoring. `specific` asks for comma-separated tree-tip labels. Interactive mode uses the default 100 BABAPPA-native null replicates. Use the explicit `predict-branch-sites` command with `--null-replicates` when you want quick uncalibrated scoring (`0`) or manuscript-strength calibration (`1000+`).

### Main End-User Command: MSA + Tree To Branch-Site Calls

If you already have a codon MSA and a tree whose tip labels match the MSA IDs, this is the intended front door:

```bash
babappa predict-branch-sites \
  --msa my_gene.codon_aligned.fasta \
  --tree my_gene.treefile \
  --foreground leaves \
  --model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir my_gene_babappa_prediction \
  --device auto \
  --null-replicates 1000
```

To score only selected tree tips as foreground branches:

```bash
babappa predict-branch-sites \
  --msa my_gene.codon_aligned.fasta \
  --tree my_gene.treefile \
  --foreground Arabidopsis_thaliana,Arabidopsis_lyrata \
  --model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir my_gene_babappa_prediction \
  --device mps \
  --null-replicates 1000
```

BABAPPA does not realign input for this command. The user-supplied MSA is the alignment used for prediction. The prediction table reports both `msa_codon_site`/`aligned_codon_site` and `branch_degapped_codon_site`, so users can locate a call in the alignment column and in the de-gapped sequence coordinate of the scored branch.

The `--null-replicates` option is the standalone BABAPPA evidence layer. It runs a BABAPPA-native branch-shuffle feature null for the same empirical MSA/tree feature table and reports p-like values such as `p_babappa_called_rows` and `p_babappa_max_gene_support`. Use `--null-replicates 0` only for quick checking; use `100` for a pilot; use `1000` or more when you want a BABAPPA-native result that can be reported in a paper as BABAPPA evidence.

Main outputs:

- `branch_site_predictions.tsv`: site-by-branch scores and calls
- `branch_predictions.tsv`: branch-level support summary
- `gene_summary.tsv`: gene-level diagnostic summary
- `babappa_native_null/`: BABAPPA-native empirical null scores, summary, and observed-vs-null report when `--null-replicates > 0`
- `prediction_report.md`: human-readable report
- `qc_report.md`: input/applicability summary

### How To Read The Main Output Files

`branch_site_predictions.tsv` is the file most users will inspect first. Important columns include:

- `branch_id`: foreground branch/tip being scored;
- `msa_codon_site`: one-based codon column in the supplied MSA;
- `aligned_codon_site`: aligned codon coordinate, retained for compatibility with older workflows;
- `branch_degapped_codon_site`: one-based codon coordinate in the foreground sequence after removing gapped codons;
- `branch_codon`: foreground codon at that alignment position;
- `score`: BABAPPA branch-site score;
- `called_positive`: whether the row crossed the selected BABAPPA threshold.

`branch_predictions.tsv` summarizes each scored foreground branch. Use it to see whether support is concentrated on one branch or spread across many branches.

`gene_summary.tsv` summarizes the family. It records:

- input size and tier model;
- applicability/OOD status;
- diagnostic result class;
- maximum gene support;
- number of called branch-site rows;
- BABAPPA-native null replicate count;
- p-like native-null values;
- final BABAPPA-native result class.

`prediction_report.md` is the readable report to start from when writing notes or a manuscript methods/results paragraph.

Dry-run mode validates the MSA/tree and builds the feature table without model scoring:

```bash
babappa predict-branch-sites \
  --msa my_gene.codon_aligned.fasta \
  --tree my_gene.treefile \
  --foreground leaves \
  --outdir my_gene_babappa_dryrun \
  --dry-run
```

### Standalone BABAPPA-Native Evidence For Papers

For a paper, the recommended BABAPPA-native command is:

```bash
babappa predict-branch-sites \
  --msa my_gene.codon_aligned.fasta \
  --tree my_gene.treefile \
  --foreground leaves \
  --outdir my_gene_babappa_prediction_paper \
  --device auto \
  --null-replicates 1000
```

Report these fields from `gene_summary.tsv` and `prediction_report.md`:

- `result_class`
- `babappa_native_result_class`
- `babappa_native_evidence_class`
- `babappa_native_null_replicates`
- `p_babappa_called_rows`
- `p_babappa_max_gene_support`
- `p_babappa_max_branch_support`
- `applicability_status`
- `tier_model`

Suggested wording:

> BABAPPA identified BABAPPA-native calibrated branch-site support using the supplied codon MSA and tree as authoritative inputs. The result was calibrated against BABAPPA's branch-shuffle empirical feature null with N replicates. This is a BABAPPA-native evidence statement and is complementary to, but not mathematically identical with, codeml/HyPhy likelihood-ratio tests.

### Internal Pipeline Commands

Validate the deployable package:

```bash
babappa validate-deployable-model-package --package-dir deployable_model_conservative_branch_site_100k_mps
```

Validate a tiny empirical input:

```bash
babappa validate-empirical-input \
  --cds-fasta tests/data/empirical_smoke/tiny_empirical.cds.fasta \
  --tree tests/data/empirical_smoke/tiny_empirical.treefile \
  --foreground taxon1 \
  --outdir empirical_input_smoke
```

Run a tiny empirical alignment ensemble:

```bash
babappa run-empirical-alignment-ensemble \
  --cds-fasta tests/data/empirical_smoke/tiny_empirical.cds.fasta \
  --tree tests/data/empirical_smoke/tiny_empirical.treefile \
  --foreground taxon1 \
  --outdir empirical_alignment_smoke \
  --methods identity,mafft,babappalign,muscle \
  --require-babappalign true \
  --threads 4
```

Extract empirical branch-site features:

```bash
babappa extract-empirical-branch-site-features \
  --empirical-validation-dir empirical_input_smoke \
  --alignment-dir empirical_alignment_smoke \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir empirical_features_smoke \
  --foreground taxon1
```

Audit feature safety:

```bash
babappa audit-empirical-features \
  --features empirical_features_smoke/empirical_branch_site_features.tsv \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir empirical_feature_audit_smoke
```

Run applicability/OOD gate:

```bash
babappa empirical-applicability \
  --empirical-validation-dir empirical_input_smoke \
  --empirical-feature-dir empirical_features_smoke \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir empirical_applicability_smoke
```

Score only after validation, feature audit, and applicability have run:

```bash
babappa score-empirical-branch-sites \
  --features empirical_features_smoke/empirical_branch_site_features.tsv \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --applicability-dir empirical_applicability_smoke \
  --outdir empirical_scores_smoke \
  --device auto
```

Plan simulation-matched calibration before writing the final diagnostic report:

```bash
babappa plan-simulation-matched-calibration \
  --empirical-validation-dir empirical_input_smoke \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir simulation_matched_calibration_plan_smoke
```

Generate report:

```bash
babappa make-empirical-branch-site-report \
  --outdir empirical_report_smoke \
  --empirical-validation-dir empirical_input_smoke \
  --alignment-dir empirical_alignment_smoke \
  --feature-dir empirical_features_smoke \
  --feature-audit-dir empirical_feature_audit_smoke \
  --applicability-dir empirical_applicability_smoke \
  --scoring-dir empirical_scores_smoke \
  --simulation-matched-calibration-plan simulation_matched_calibration_plan_smoke \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps
```

## Typical Workflows

### 1. Simulation Validation Workflow

Use simulation commands for development and validation, not empirical discovery.

Tiny simulation:

```bash
babappa simulate --outdir sim_smoke --n-families 3 --n-taxa 6 --n-codons 60 --seed 42 --positive-rate 0.5 --saturation-tier moderate
babappa validate-sim --sim-dir sim_smoke
babappa audit-sim --sim-dir sim_smoke --outdir sim_smoke/audit
```

Alignment and feature-building commands include:

```bash
babappa align-sim --sim-dir sim_smoke --outdir align_smoke
babappa validate-align --align-dir align_smoke
babappa build-site-map --sim-dir sim_smoke --align-dir align_smoke --outdir site_map_smoke
babappa validate-site-map --site-map-dir site_map_smoke
```

Heavy 10K/100K plans are user-run only and should not be launched casually.

### 2. Deployable Model Package Validation

The validated package is:

```text
deployable_model_conservative_branch_site_100k_mps
```

Validate package integrity:

```bash
babappa validate-deployable-model-package --package-dir deployable_model_conservative_branch_site_100k_mps
```

Smoke-load package:

```bash
babappa smoke-load-deployable-model \
  --package-dir deployable_model_conservative_branch_site_100k_mps \
  --device auto \
  --outdir deployable_model_load_smoke
```

The package includes:

- `model_manifest.json`
- `model_card.md`
- `feature_schema.json`
- `calibration_schema.json`
- `training_envelope.json`
- `tier_models/`
- `tier_calibrations/`
- `checksums.sha256`
- `validation_summary.json`
- `limitations.md`
- `README.md`

### 3. Real Empirical Input Staging

Prepare a real pilot workspace:

```bash
babappa prepare-real-empirical-pilot-workspace --workspace real_empirical_pilot --max-families 12
babappa prepare-real-pilot-inputs --workspace real_empirical_pilot --manifest real_empirical_pilot_panel.tsv --outdir real_empirical_pilot/input_staging
```

Canonical input paths:

```text
real_empirical_pilot/input/cds/<panel_id>.cds.fasta
real_empirical_pilot/input/trees/<panel_id>.treefile
```

Import one family:

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

Batch import:

```bash
babappa import-real-pilot-batch --workspace real_empirical_pilot --batch-manifest real_empirical_pilot/import_batch.tsv
```

Validate readiness:

```bash
babappa validate-real-pilot-readiness \
  --workspace real_empirical_pilot \
  --manifest real_empirical_pilot_panel.tsv \
  --outdir real_empirical_pilot/readiness
```

Do not run the pilot until readiness says `ready_to_run: true`.

### 4. Empirical Diagnostic Workflow

Screen a family before scoring:

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

Run a small guarded panel:

```bash
babappa run-empirical-pilot-panel \
  --panel-manifest real_empirical_pilot/manifest/real_empirical_pilot_panel.tsv \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir real_empirical_pilot/babappa_run \
  --methods identity,mafft,babappalign,muscle \
  --device auto \
  --max-families 12
```

Summarize and validate the panel:

```bash
babappa summarize-empirical-pilot-panel --panel-run real_empirical_pilot/babappa_run --outdir real_empirical_pilot/summary
babappa validate-empirical-pilot-summary --summary-dir real_empirical_pilot/summary
```

### 5. WRKY-Style Close-Taxa Pilot Workflow

For Arabidopsis-like WRKY families, do not mix very distant plant taxa at first. Start with closer Brassicaceae-heavy taxa:

```bash
babappa recommend-target-taxa --pilot-type plant_close --outdir real_empirical_pilot/target_taxa_recommendations
```

Plan an OOD-aware family build:

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

Current WRKY interpretation:

- `WRKY_candidate_01`: OOD stress test, mean p-distance `0.725799`, diagnostic-only, no positive call.
- `WRKY_candidate_02_close`: in-domain close-taxa WRKY33/AT2G38470 diagnostic pilot, BABAPPA diagnostic-positive, max gene support `0.177189`, called branch-site rows `6954`.
- codeml Model A vs null: LRT `0.0`, p-value `1.0`, negative.
- HyPhy aBSREL foreground p-value: `1.0`, negative.
- HyPhy MEME minimum p-value: `0.0641705`, negative at 0.05.
- Concordance: `BABAPPA_only`.
- Matched-null calibration: 100 feature-level matched nulls completed and validated with the deployable model package.
- Null result: called branch-site rows were unusual versus the feature-matched null (`p_empirical_called_rows=0.009900990099009901`), but max gene support was not unusual (`p_empirical_support=1.0`).

Correct interpretation: BABAPPA-only with mixed feature-level null support; still inconclusive as an empirical discovery claim because codeml and HyPhy are negative and the null calibration is feature-level rather than full raw sequence simulation/alignment replay.

### 6. Simulation-Matched Calibration Planning

Plan calibration from empirical QC:

```bash
babappa plan-simulation-matched-calibration \
  --empirical-validation-dir real_empirical_pilot/babappa_run/per_family/FAMILY_ID/empirical_input_validation \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir real_empirical_pilot/babappa_run/per_family/FAMILY_ID/simulation_matched_calibration_plan
```

Summarize plan:

```bash
babappa summarize-simulation-matched-calibration-plan \
  --plan-dir real_empirical_pilot/babappa_run/per_family/FAMILY_ID/simulation_matched_calibration_plan \
  --outdir real_empirical_pilot/babappa_run/per_family/FAMILY_ID/simulation_matched_calibration_summary
```

The WRKY 100-null feature-level matched calibration has completed once under user control. It should be treated as diagnostic support only, not as a final empirical p-value system or discovery proof.

Dry-run the evidence-pack calibration command before launching anything long:

```bash
babappa run-simulation-matched-null-calibration \
  --evidence-pack real_empirical_pilot/evidence_packs/WRKY_candidate_02_close \
  --outdir real_empirical_pilot/calibration_runs/WRKY_candidate_02_close_null100_dryrun \
  --n-null 100 \
  --seed 20260530 \
  --device mps \
  --dry-run
```

Dry-run mode validates the evidence pack and writes:

- `calibration_run_plan.json`
- `calibration_run_plan.md`
- `calibration_input_validation.tsv`
- `calibration_status.json`
- `calibration_status.md`

It does not write null distributions, null percentiles, or discovery-supporting results.

To rerun the feature-level matched-null calibration:

```bash
babappa run-simulation-matched-null-calibration \
  --evidence-pack real_empirical_pilot/evidence_packs/WRKY_candidate_02_close \
  --outdir real_empirical_pilot/calibration_runs/WRKY_candidate_02_close_null100 \
  --n-null 100 \
  --seed 20260530 \
  --device mps
```

Current implementation note: the evidence-pack command is operational for safe dry-run/planning and for conservative feature-level matched-null scoring through the deployable model package. This is a BABAPPA-native calibration backend, not a codeml/HyPhy likelihood-ratio null and not a full raw sequence simulation plus alignment replay. Do not interpret staged or dry-run files as completed calibration. Completed feature-level null support may be reported as BABAPPA-native evidence, with the backend and limitations stated explicitly.

### 7. Classical Reference Workflow Planning

Plan codeml/HyPhy templates:

```bash
babappa plan-classical-reference-workflows \
  --panel-manifest real_empirical_pilot/manifest/real_empirical_pilot_panel.tsv \
  --outdir real_empirical_pilot/reference_plan \
  --tools codeml,hyphy
```

Check reference tools:

```bash
babappa check-reference-tools --outdir real_empirical_pilot/reference_runs/WRKY_candidate_02_close/tool_check
```

Parse prepared outputs:

```bash
babappa parse-codeml-reference \
  --codeml-dir real_empirical_pilot/reference_runs/WRKY_candidate_02_close/codeml \
  --outdir real_empirical_pilot/reference_runs/WRKY_candidate_02_close/codeml_parsed

babappa parse-hyphy-reference \
  --hyphy-dir real_empirical_pilot/reference_runs/WRKY_candidate_02_close/hyphy \
  --outdir real_empirical_pilot/reference_runs/WRKY_candidate_02_close/hyphy_parsed
```

Build reference results:

```bash
babappa build-reference-results-table \
  --panel-id WRKY_candidate_02_close \
  --codeml-parsed real_empirical_pilot/reference_runs/WRKY_candidate_02_close/codeml_parsed \
  --hyphy-parsed real_empirical_pilot/reference_runs/WRKY_candidate_02_close/hyphy_parsed \
  --outdir real_empirical_pilot/reference_results/WRKY_candidate_02_close
```

Compare:

```bash
babappa compare-empirical-reference-results \
  --babappa-panel-run real_empirical_pilot/babappa_run_wrky_close_raw_alignmentaware \
  --reference-results real_empirical_pilot/reference_results/WRKY_candidate_02_close/reference_results.tsv \
  --outdir real_empirical_pilot/comparison/WRKY_candidate_02_close
```

### 8. Publication Benchmark Pipeline

The repository also includes a separate manuscript-only benchmarking harness:

```text
publication_benchmark/
```

This is not required for normal BABAPPA use. It exists to compare BABAPPA-native calibrated evidence with codeml and HyPhy on a curated publication panel.

Typical user-run sequence:

```bash
bash publication_benchmark/scripts/01_run_babappa_native.sh publication_benchmark/panel_template.tsv publication_benchmark/results
bash publication_benchmark/scripts/02_prepare_codeml_hyphy.sh publication_benchmark/panel_template.tsv publication_benchmark/results
bash publication_benchmark/scripts/03_run_codeml_hyphy_user.sh publication_benchmark/results
bash publication_benchmark/scripts/04_parse_and_compare.sh publication_benchmark/panel_template.tsv publication_benchmark/results
bash publication_benchmark/scripts/05_make_publication_tables.sh publication_benchmark/panel_template.tsv publication_benchmark/results
```

Use this for manuscript benchmark tables only. It should not be confused with the normal end-user command, and it does not make BABAPPA dependent on codeml or HyPhy.

## Input Requirements

Empirical inputs should include:

- CDS FASTA with codon-valid sequences;
- tree file with tips matching FASTA IDs;
- foreground taxon or branch label;
- optional metadata describing expected category and reference status;
- close enough taxa for the current training envelope;
- at least 6 taxa preferred;
- at least 100 codons preferred.

### CDS Integrity Gate

BABAPPA checks that the supplied alignment is biologically plausible CDS before it scores anything. This gate is intentionally strict because a deep-learning score on a broken CDS alignment is not meaningful.

By default, BABAPPA stops with an explicit failure if it finds:

- sequence length not divisible by 3;
- unequal MSA sequence lengths;
- duplicate FASTA IDs;
- tree tips that do not match FASTA IDs;
- missing requested foreground label;
- first non-gap codon is not `ATG`;
- true internal stop codon;
- too few taxa or too few codons.

BABAPPA continues with explicit warnings for:

- terminal stop codons at the natural CDS end;
- ambiguous bases;
- gaps;
- high gap fraction;
- high pairwise p-distance or saturation warnings.

Terminal stop codons are common in real CDS exports. They are not treated as internal stops and do not block execution. The warning exists so the final report is transparent.

If your MSA starts after the biological start codon because you intentionally aligned a CDS fragment, use the diagnostic override:

```bash
babappa predict-branch-sites \
  --msa fragment.codon_aligned.fasta \
  --tree fragment.treefile \
  --foreground leaves \
  --allow-missing-start-codon \
  --outdir fragment_babappa
```

Use this only when you are sure the input is a valid in-frame CDS fragment. The report will still record the missing-start condition.

Internal stop codons should normally be fixed at the data-curation stage. `--allow-stop-codons` is a diagnostic override only; terminal stops do not need it.

Input checks include:

- duplicate sequence IDs;
- CDS length divisibility by 3;
- first non-gap codon is `ATG` by default;
- internal stop codons;
- terminal stop codons, which are accepted as normal CDS endings but reported as warnings;
- ambiguous base fraction;
- gap fraction;
- pairwise p-distance;
- saturation proxy;
- foreground validity;
- tree-tip compatibility.

Do not provide simulator truth or oracle labels during empirical inference. Forbidden empirical input columns include:

- `branch_site_truth`
- `selected_sites`
- `truth`
- `branch_truth`
- `oracle`
- `y_branch_site`
- `y_site`
- `gene_label`
- `positive_label`
- `simulated_label`

## Aligners

For the main command, BABAPPA does not run aligners. The supplied codon MSA is the authoritative input:

```bash
babappa predict-branch-sites --msa aligned.codon.fasta --tree treefile --foreground leaves --outdir prediction
```

Optional diagnostic alignment/sensitivity workflows can use:

- `identity`
- `mafft`
- `babappalign`
- `muscle`

Diagnostic-only aligners:

- PRANK
- T-Coffee

Alignment ensemble robustness matters only when the user wants to test sensitivity to homology uncertainty. It is not required for the core user-supplied-MSA prediction workflow.

## Output Interpretation

Common terms:

- `diagnostic-positive`: BABAPPA scored support above its current diagnostic threshold before native-null interpretation.
- `babappa_native_calibrated_support`: BABAPPA is diagnostic-positive and the observed result is unusual under the BABAPPA-native empirical feature null. This is the primary standalone BABAPPA evidence class.
- `strong_babappa_native_support`: stronger native-null support, typically when at least one p-like BABAPPA metric is at or below 0.01 with sufficient replicates.
- `not_significant_under_babappa_native_null`: raw BABAPPA scores were not unusual under the BABAPPA-native null; do not present as BABAPPA-supported selection.
- `underpowered_native_null`: too few null replicates were run for manuscript interpretation.
- `diagnostic_only`: output may be useful for stress testing or triage but should not be interpreted as positive selection.
- `in_domain`: empirical input appears compatible with the training envelope.
- `borderline`: empirical input has warnings and should be interpreted cautiously.
- `out_of_domain`: empirical input falls outside the current training envelope; abstain from biological interpretation.
- `BABAPPA_only`: BABAPPA-native evidence is present but codeml/HyPhy comparators are negative or absent. This is reportable as BABAPPA evidence, not as cross-method consensus.
- `concordant_positive`: BABAPPA-native evidence and at least one external reference workflow support compatible evidence.
- `reference_only`: reference tool positive but BABAPPA not supportive; inspect alignment, OOD, and model limitations.
- `calibration_pending`: BABAPPA-native null calibration has not completed; do not report calibrated BABAPPA support.
- `feature_matched_calibration_complete`: feature-level matched null scoring has completed. Report the backend explicitly; it is BABAPPA-native evidence, not a codeml/HyPhy likelihood-ratio p-value.

Responsible reporting language:

- use "diagnostic support" or "guarded empirical score";
- for standalone BABAPPA claims, prefer "BABAPPA-native calibrated support" and report `babappa_native_result_class`;
- report applicability/OOD status;
- report `--null-replicates`, native-null backend, and all p-like `p_babappa_*` values;
- report codeml/HyPhy only when used as optional external comparators;
- avoid saying BABAPPA is a codeml/HyPhy replacement or that BABAPPA p-like values are likelihood-ratio p-values.

## Reproducibility

Important retained artifacts:

- deployable package: `deployable_model_conservative_branch_site_100k_mps`
- final 100K validation report: `explicit_branch_truth_100k_mps_final_validation_report.md/json/tsv`
- cross-tier summary: `explicit_branch_truth_100k_mps_cross_tier_summary/`
- truth audit: `branch_truth_status_audit_explicit_branch_truth_100k_mps/`
- WRKY evidence pack: `real_empirical_pilot/evidence_packs/WRKY_candidate_02_close/`
- Git readiness report: `GIT_PUSH_READINESS_REPORT.md`

Existing Zenodo-ready archive:

```text
BABAPPA_v0.8.0_release_zenodo_YYYYMMDD.tar.xz
```

Checksum:

```text
pending for the v0.8.0 release archive
```

Validate package:

```bash
babappa validate-deployable-model-package --package-dir deployable_model_conservative_branch_site_100k_mps
```

Validate WRKY evidence pack:

```bash
babappa validate-empirical-evidence-pack --evidence-pack real_empirical_pilot/evidence_packs/WRKY_candidate_02_close
```

Run tests:

```bash
python -m pytest -q
```

## Storage Cleanup And User Maintenance

BABAPPA simulations can generate very large reproducible outputs. Audit before deleting anything:

```bash
babappa audit-storage --root . --outdir storage_cleanup_audit --target-size-gb 10
```

Outputs include:

- `storage_inventory.tsv`
- `storage_inventory.json`
- `storage_summary.md`
- `keep_list.tsv`
- `remove_candidates.tsv`
- `archive_candidates.tsv`
- `cleanup_dry_run.md`
- `du_top_100.txt`
- `quarantine_large_reproducible_outputs.sh`
- `delete_quarantine_after_review.sh`
- `archive_key_reports.sh`
- `validate_after_cleanup.sh`

Move candidates to quarantine only:

```bash
bash storage_cleanup_audit/quarantine_large_reproducible_outputs.sh
```

Validate after cleanup:

```bash
bash storage_cleanup_audit/validate_after_cleanup.sh
```

Do not run the permanent delete script until the quarantine has been manually reviewed. The delete script requires `CONFIRM_DELETE=YES`.

Recent storage note: the large system storage issue was caused by a generated BABAPPAlign embeddings cache at `$HOME/.cache/babappalign/embeddings`, not by the BABAPPA Git checkout. The required model file `$HOME/.cache/babappalign/models/babappascore.pt` should be preserved.

## Troubleshooting

### Missing aligners

Run:

```bash
babappa check-aligners
```

If BABAPPAlign reports a missing model, install `babappascore.pt` into `$HOME/.cache/babappalign/models/`.

### MPS/CUDA/CPU device problems

Run:

```bash
babappa check-neural-env
```

Use `--device cpu` if MPS/CUDA fails or if a tensor operation is unsupported.

### Very high p-distance or OOD input

Use closer taxa. For plant WRKY pilots, start with close Brassicaceae panels rather than broad monocot/dicot/legume mixtures.

### codeml/HyPhy disagreement

Treat disagreement conservatively. BABAPPA-only positive signals require matched-null calibration, controls, and biological review.

### Pruned intermediates

Some raw 100K intermediates were intentionally pruned after validation. Use retained summaries, audits, stage markers, model artifacts, checksums, and cleanup manifests for reproducibility.

### Package validation failure

Check that `model_manifest.json`, schemas, checksums, tier models, tier calibrations, and validation summary are present.

### Git cleanup confusion

Generated heavy outputs should not be committed. Use:

```bash
git status --short
git diff --stat
git diff --cached --stat
```

## Citation And Manuscript Status

BABAPPA is currently described by a research-alpha software/methods manuscript in:

```text
Manuscript/BABAPPA_method_paper_auxiliary_saturation.tex
```

No final publication DOI is available yet. Use the repository and release archive metadata until a formal citation is assigned.

Citation placeholder:

```text
Sinha K. BABAPPA: a research-alpha, simulation-trained framework for guarded branch-site positive-selection support under alignment uncertainty. Manuscript in preparation.
```

## PyPI Release Workflow

The package metadata lives in `pyproject.toml`, and the console entry point is:

```text
babappa = "babappa.cli:main"
```

Build locally:

```bash
python -m pip install -e ".[dev]"
python -m build
python -m twine check dist/*
```

Upload to TestPyPI first:

```bash
python -m twine upload --repository testpypi dist/*
```

Then test installation in a fresh environment. Upload to PyPI only after the TestPyPI package installs and `babappa --version` plus `babappa --help` work.

## Developer Notes

Check version:

```bash
babappa --version
```

Run tests:

```bash
python -m pytest -q
```

Inspect Git state:

```bash
git status --short
git diff --stat
git diff --cached --stat
```

Do not commit:

- raw 10K/100K simulations;
- raw alignments;
- tensor shards;
- branch-site datasets;
- prediction tables from heavy runs;
- logs;
- temporary work directories;
- generated BABAPPAlign embeddings caches;
- raw empirical downloads;
- BLAST databases or downloaded genomes/proteomes.

Commit and archive:

- source code;
- tests;
- docs;
- examples;
- manuscript source/PDF;
- deployable package metadata and selected lightweight model artifacts;
- final validation reports;
- evidence-pack manifests and summaries;
- checksums;
- cleanup manifests.

## Scientific Bottom Line

BABAPPA is now oriented around the original end-user goal: supply an aligned codon MSA and treefile, choose foreground branches, and receive branch-site calls with de-gapped site coordinates and BABAPPA-native calibrated evidence. codeml and HyPhy remain valuable external comparators, but BABAPPA is not dependent on them to report its own standalone evidence class. The correct manuscript language is "BABAPPA-native calibrated branch-site support" with full QC, OOD, null-replicate, model-package, and biological-context reporting.

## Minimal End-User Checklist

Before trusting a BABAPPA run, check:

- your FASTA is an aligned codon MSA;
- every sequence length is equal and divisible by 3;
- sequence IDs match tree tip labels exactly;
- every sequence is a plausible CDS or intentional in-frame CDS fragment;
- terminal stop codons are acceptable and recorded as warnings;
- no internal stop codons are present;
- `gene_summary.tsv` reports `in_domain` or a defensible `borderline` status;
- native-null calibration has enough replicates for the claim you want to make;
- the final wording says BABAPPA-native support, not codeml/HyPhy p-value.
