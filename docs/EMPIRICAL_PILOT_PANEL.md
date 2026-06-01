# Empirical Pilot Panel

BABAPPA `v0.7.0` includes a small curated empirical pilot-panel framework. It is diagnostic benchmarking only, not final empirical inference.

## Manifest Schema

Required columns:

- `panel_id`
- `gene_family`
- `species_group`
- `cds_fasta`
- `tree_file`
- `foreground`
- `expected_category`
- `reference_status`
- `notes`

Valid `expected_category` values are `known_positive`, `likely_negative`, `alignment_sensitive`, `saturated`, `short_low_information`, `paralogy_risk`, and `unknown`.

For real pilot templates, `likely_positive` is also accepted as a guarded candidate category. It means evolutionary-interest candidate only, not a discovered positive.

Valid `reference_status` values are `codeml_available`, `hyphy_available`, `both_available`, `unavailable`, and `planned`.

## Commands

```bash
babappa validate-empirical-pilot-panel --panel-manifest tests/data/empirical_pilot_panel/empirical_pilot_panel.tsv --outdir empirical_pilot_panel_validation_smoke
babappa run-empirical-pilot-panel --panel-manifest tests/data/empirical_pilot_panel/empirical_pilot_panel.tsv --deployable-model-package deployable_model_conservative_branch_site_100k_mps --outdir empirical_pilot_panel_run_smoke --methods identity,mafft,babappalign,muscle --device auto --max-families 5
babappa plan-classical-reference-workflows --panel-manifest tests/data/empirical_pilot_panel/empirical_pilot_panel.tsv --outdir classical_reference_workflow_plan_smoke --tools codeml,hyphy
babappa compare-empirical-reference-results --babappa-panel-run empirical_pilot_panel_run_smoke --reference-results tests/data/empirical_pilot_panel/mock_reference_results.tsv --outdir empirical_reference_comparison_smoke
babappa summarize-empirical-pilot-panel --panel-run empirical_pilot_panel_run_smoke --reference-comparison empirical_reference_comparison_smoke --outdir empirical_pilot_panel_summary_smoke
babappa validate-empirical-pilot-summary --summary-dir empirical_pilot_panel_summary_smoke
```

## Real Pilot Workspace

```bash
babappa prepare-real-empirical-pilot-workspace --workspace real_empirical_pilot --max-families 12
babappa validate-empirical-pilot-panel --panel-manifest real_empirical_pilot/manifest/real_empirical_pilot_panel.tsv --outdir real_empirical_pilot/panel_validation
babappa plan-classical-reference-workflows --panel-manifest real_empirical_pilot/manifest/real_empirical_pilot_panel.tsv --outdir real_empirical_pilot/reference_plan --tools codeml,hyphy
babappa make-real-empirical-pilot-decision-report --workspace real_empirical_pilot --outdir real_empirical_pilot/summary
babappa prepare-real-pilot-inputs --workspace real_empirical_pilot --manifest real_empirical_pilot_panel.tsv --outdir real_empirical_pilot/input_staging
babappa validate-real-pilot-readiness --workspace real_empirical_pilot --manifest real_empirical_pilot_panel.tsv --outdir real_empirical_pilot/readiness
```

If the real manifest contains missing files or placeholder foregrounds, BABAPPA writes a readiness/decision report and does not run the empirical pilot. Populate `real_empirical_pilot/input/` with real CDS FASTA/tree files before running `run-empirical-pilot-panel`.

Cycle 44 input staging also supports `import-real-pilot-family`, `import-real-pilot-batch`, `sanitize-cds-fasta`, `list-foreground-candidates`, `plan-real-pilot-tree-building`, and `discover-local-pilot-files`.

## OOD-Aware Family Selection

Cycle 46 adds a pre-scoring gate for real empirical families:

```bash
babappa prefilter-empirical-family --cds-fasta real_empirical_pilot/input/cds/WRKY_candidate_01.cds.fasta --tree-file real_empirical_pilot/input/trees/WRKY_candidate_01.treefile --foreground Arabidopsis_thaliana --outdir real_empirical_pilot/prefilter/WRKY_candidate_01
babappa recommend-target-taxa --pilot-type plant_close --outdir real_empirical_pilot/target_taxa_recommendations
babappa plan-ood-aware-family-build --family-id WRKY_candidate_02_close --query-species Arabidopsis_thaliana --query-gene-or-locus AT2G38470 --target-taxa-file real_empirical_pilot/target_taxa_recommendations/recommended_target_taxa.tsv --outdir real_empirical_pilot/acquisition_plans/WRKY_candidate_02_close --max-mean-pdistance 0.35 --min-taxa 6 --min-codons 100
```

The first real WRKY family was useful as a stress-test/failure-mode case, not as a discovery. Its mean p-distance was far above the recommended first-pilot gate, so BABAPPA treats it as OOD and diagnostic-only.

`WRKY_candidate_02_close` is the first accepted in-domain diagnostic empirical pilot. It should be used as the reference/calibration development case, not as a manuscript discovery claim. Its current status is BABAPPA diagnostic-positive but codeml and HyPhy negative. A 100-replicate feature-level matched-null calibration completed with unusual called-row burden (`p_empirical_called_rows=0.009900990099009901`) but non-unusual maximum gene support (`p_empirical_support=1.0`). The interpretation is therefore BABAPPA-only with mixed feature-level null support, and inconclusive as an empirical discovery claim.

## Long-Run Handoff Policy

Codex does not execute heavy empirical calibration or broad scans. It generates USER-RUN scripts and lightweight validators; the user runs long jobs locally/offline and returns summaries/logs for interpretation.

## Claim Boundary

Every pilot summary must state:

- BABAPPA model is simulation-trained.
- No simulator truth was used for empirical inference.
- Scores are diagnostic until simulation-matched calibration and external benchmark interpretation are complete.
- Out-of-domain cases are not positive-selection calls.
- Reference-tool disagreement must be interpreted biologically, not automatically treated as BABAPPA failure.

The built-in panel under `tests/data/empirical_pilot_panel/` is synthetic empirical-like smoke data only. It is not a real biological positive/negative benchmark.
