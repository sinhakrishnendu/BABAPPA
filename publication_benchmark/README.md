# BABAPPA Publication Benchmark Pipeline

This directory is a manuscript benchmarking harness, not a required BABAPPA runtime dependency.

Purpose:

- run BABAPPA-native calibrated predictions on a curated panel;
- run codeml/PAML branch-site references on the same panel;
- run HyPhy aBSREL/MEME references on the same panel;
- parse outputs into a shared comparison table;
- generate publication-ready benchmark tables for the manuscript.

Scientific boundary:

- BABAPPA remains standalone and can report BABAPPA-native calibrated evidence without codeml or HyPhy.
- codeml and HyPhy are used here as external manuscript comparators, not as ground truth and not as BABAPPA dependencies.
- Disagreement should be interpreted biologically and methodologically, not automatically as a failure of any one method.
- This pipeline is for selected benchmark panels only. Do not use it for broad empirical scans without a separate design.

## Files

- `panel_template.tsv`: input manifest template for benchmark families.
- `reference_results_schema.tsv`: schema for combined codeml/HyPhy results.
- `benchmark_config.env`: editable defaults.
- `scripts/01_run_babappa_native.sh`: runs BABAPPA direct MSA/tree prediction with BABAPPA-native null calibration.
- `scripts/02_prepare_codeml_hyphy.sh`: prepares codeml and HyPhy reference folders.
- `scripts/03_run_codeml_hyphy_user.sh`: manual execution execution script for codeml/HyPhy.
- `scripts/04_parse_and_compare.sh`: parses reference outputs and creates BABAPPA/reference comparisons.
- `scripts/05_make_publication_tables.sh`: writes compact manuscript tables from available outputs.

## Recommended Panel Design

Use 8-20 curated families:

- known or literature-supported positives;
- likely conserved negatives;
- alignment-sensitive cases;
- saturated/high-divergence cases;
- short/low-information cases;
- paralogy-risk cases.

Each row should have a curated codon MSA and tree. The benchmark should not realign inputs unless the manuscript explicitly studies alignment sensitivity.

## Minimal Workflow

Review and edit:

```bash
publication_benchmark/panel_template.tsv
publication_benchmark/benchmark_config.env
```

Then run, one stage at a time:

```bash
bash publication_benchmark/scripts/01_run_babappa_native.sh publication_benchmark/panel_template.tsv publication_benchmark/results
bash publication_benchmark/scripts/02_prepare_codeml_hyphy.sh publication_benchmark/panel_template.tsv publication_benchmark/results
bash publication_benchmark/scripts/03_run_codeml_hyphy_user.sh publication_benchmark/results
bash publication_benchmark/scripts/04_parse_and_compare.sh publication_benchmark/panel_template.tsv publication_benchmark/results
bash publication_benchmark/scripts/05_make_publication_tables.sh publication_benchmark/panel_template.tsv publication_benchmark/results
```

The scripts are marked MANUAL EXECUTION SCRIPT. They are not executed automatically.

## Manuscript Reporting

Report BABAPPA fields:

- `babappa_native_result_class`
- `babappa_native_evidence_class`
- `p_babappa_called_rows`
- `p_babappa_max_gene_support`
- `applicability_status`
- `tier_model`

Report reference fields:

- codeml Model A vs null LRT/p-value/result class;
- HyPhy aBSREL foreground p-value/result class;
- HyPhy MEME minimum p-value/result class.

Use wording such as:

> BABAPPA was benchmarked against codeml and HyPhy on a curated manuscript panel. BABAPPA evidence was generated independently using its native empirical null calibration; codeml and HyPhy were used as external comparators.
