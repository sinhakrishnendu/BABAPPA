# BABAPPA Storage Cleanup Dry Run

No files were moved or deleted by this audit.

- Current project size: `183.7M`
- Estimated movable archive/remove size: `4.2M`
- Expected size after quarantine: `179.5M`

## Proposed Top-Level Quarantine Moves

| path | size | recommendation | reason |
|---|---:|---|---|
| `src/babappa/__pycache__/cli.cpython-311.pyc` | 443.9K | remove | cache/log/temp artifact |
| `src/babappa/empirical/__pycache__/reference_eval.cpython-311.pyc` | 128.1K | remove | cache/log/temp artifact |
| `src/babappa/empirical/__pycache__/bridge.cpython-311.pyc` | 88.1K | remove | cache/log/temp artifact |
| `src/babappa/reports/__pycache__/summary.cpython-311.pyc` | 84.3K | remove | cache/log/temp artifact |
| `src/babappa/branch/__pycache__/truth_plan.cpython-311.pyc` | 83.2K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_external_aligners_and_site_maps.cpython-311-pytest-9.0.3.pyc` | 75.2K | remove | cache/log/temp artifact |
| `src/babappa/empirical/__pycache__/pilot_panel.cpython-311.pyc` | 72.1K | remove | cache/log/temp artifact |
| `src/babappa/branch/__pycache__/controls.cpython-311.pyc` | 61.4K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_branch_conditioned.cpython-311-pytest-9.0.3.pyc` | 58.5K | remove | cache/log/temp artifact |
| `src/babappa/branch/__pycache__/dataset.cpython-311.pyc` | 57.9K | remove | cache/log/temp artifact |
| `src/babappa/benchmarks/__pycache__/external_aligner_validation_plan.cpython-311.pyc` | 57.6K | remove | cache/log/temp artifact |
| `src/babappa/empirical/__pycache__/input_staging.cpython-311.pyc` | 57.4K | remove | cache/log/temp artifact |
| `src/babappa/deploy/__pycache__/package.cpython-311.pyc` | 54.3K | remove | cache/log/temp artifact |
| `src/babappa/align/__pycache__/external.cpython-311.pyc` | 54.1K | remove | cache/log/temp artifact |
| `src/babappa/reports/__pycache__/run_summary.cpython-311.pyc` | 53.6K | remove | cache/log/temp artifact |
| `src/babappa/branch/__pycache__/cycle39_report.cpython-311.pyc` | 52.5K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_branch_truth_c34.cpython-311-pytest-9.0.3.pyc` | 51.9K | remove | cache/log/temp artifact |
| `src/babappa/empirical/__pycache__/family_prefilter.cpython-311.pyc` | 48.4K | remove | cache/log/temp artifact |
| `src/babappa/reports/__pycache__/external_tier_summary.cpython-311.pyc` | 46.7K | remove | cache/log/temp artifact |
| `src/babappa/branch/__pycache__/summary.cpython-311.pyc` | 46.6K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_saturation_aware_training.cpython-311-pytest-9.0.3.pyc` | 41.8K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_cycle41_empirical_bridge.cpython-311-pytest-9.0.3.pyc` | 41.7K | remove | cache/log/temp artifact |
| `src/babappa/branch/__pycache__/mps_preflight.cpython-311.pyc` | 40.6K | remove | cache/log/temp artifact |
| `src/babappa/training/__pycache__/neural_train_full.cpython-311.pyc` | 38.5K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_saturation_panel_and_merge.cpython-311-pytest-9.0.3.pyc` | 38.4K | remove | cache/log/temp artifact |
| `src/babappa/branch/__pycache__/oracle_labels.cpython-311.pyc` | 37.5K | remove | cache/log/temp artifact |
| `.pytest_cache` | 35.6K | remove | cache/log/temp artifact |
| `src/babappa/calibration/__pycache__/threshold_policy.cpython-311.pyc` | 35.5K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_site_level_oracle_and_baseline.cpython-311-pytest-9.0.3.pyc` | 33.8K | remove | cache/log/temp artifact |
| `src/babappa/simulate/__pycache__/audit.cpython-311.pyc` | 33.7K | remove | cache/log/temp artifact |
| `src/babappa/branch/__pycache__/truth_audit.cpython-311.pyc` | 33.4K | remove | cache/log/temp artifact |
| `src/babappa/branch/__pycache__/context_ablation.cpython-311.pyc` | 33.3K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_site_neural_and_aggregation.cpython-311-pytest-9.0.3.pyc` | 33.0K | remove | cache/log/temp artifact |
| `src/babappa/site/__pycache__/oracle_labels.cpython-311.pyc` | 32.8K | remove | cache/log/temp artifact |
| `src/babappa/simulate/__pycache__/simulator.cpython-311.pyc` | 32.8K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_apple_silicon_mps_c37.cpython-311-pytest-9.0.3.pyc` | 32.1K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_threshold_policy.cpython-311-pytest-9.0.3.pyc` | 31.6K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_prediction_diagnostics_and_neural_v2.cpython-311-pytest-9.0.3.pyc` | 30.9K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_stratified_eval.cpython-311-pytest-9.0.3.pyc` | 30.4K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_branch_summary_truth.cpython-311-pytest-9.0.3.pyc` | 29.9K | remove | cache/log/temp artifact |
| `src/babappa/maintenance/__pycache__/storage_audit.cpython-311.pyc` | 29.6K | remove | cache/log/temp artifact |
| `src/babappa/reports/__pycache__/stratified_eval.cpython-311.pyc` | 29.3K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_cycle47_reference_eval.cpython-311-pytest-9.0.3.pyc` | 29.3K | remove | cache/log/temp artifact |
| `src/babappa/empirical/__pycache__/calibration.cpython-311.pyc` | 29.0K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_cycle42_empirical_pilot_panel.cpython-311-pytest-9.0.3.pyc` | 28.8K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_run_summary_and_compare.cpython-311-pytest-9.0.3.pyc` | 28.4K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_neural_ablation_and_diagnostics.cpython-311-pytest-9.0.3.pyc` | 28.3K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_cycle40_deploy.cpython-311-pytest-9.0.3.pyc` | 27.7K | remove | cache/log/temp artifact |
| `src/babappa/align/__pycache__/site_map.cpython-311.pyc` | 27.1K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_site_robustness_and_controls.cpython-311-pytest-9.0.3.pyc` | 26.6K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_cycle44_real_input_staging.cpython-311-pytest-9.0.3.pyc` | 26.5K | remove | cache/log/temp artifact |
| `src/babappa/site/__pycache__/dataset.cpython-311.pyc` | 26.4K | remove | cache/log/temp artifact |
| `src/babappa/branch/__pycache__/neural_train.cpython-311.pyc` | 26.4K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_ranking_repair_and_signal_audit.cpython-311-pytest-9.0.3.pyc` | 26.2K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_mps_preflight_c37e.cpython-311-pytest-9.0.3.pyc` | 25.7K | remove | cache/log/temp artifact |
| `src/babappa/benchmarks/__pycache__/large_run_plan.cpython-311.pyc` | 25.4K | remove | cache/log/temp artifact |
| `src/babappa/site/__pycache__/neural_train.cpython-311.pyc` | 25.3K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_baseline.cpython-311-pytest-9.0.3.pyc` | 24.8K | remove | cache/log/temp artifact |
| `src/babappa/site/__pycache__/baseline.cpython-311.pyc` | 24.8K | remove | cache/log/temp artifact |
| `src/babappa/reports/__pycache__/prediction_diagnostics.cpython-311.pyc` | 24.2K | remove | cache/log/temp artifact |
| `src/babappa/datasets/__pycache__/index.cpython-311.pyc` | 24.0K | remove | cache/log/temp artifact |
| `src/babappa/training/__pycache__/mps.cpython-311.pyc` | 23.8K | remove | cache/log/temp artifact |
| `src/babappa/calibration/__pycache__/stratified_calibration.cpython-311.pyc` | 23.7K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_leakage_and_stability.cpython-311-pytest-9.0.3.pyc` | 23.7K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_neural_full_training.cpython-311-pytest-9.0.3.pyc` | 23.6K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_cycle46_ood_family_prefilter.cpython-311-pytest-9.0.3.pyc` | 23.6K | remove | cache/log/temp artifact |
| `src/babappa/calibration/__pycache__/baseline.cpython-311.pyc` | 23.5K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_branch_feature_policy_c36.cpython-311-pytest-9.0.3.pyc` | 23.1K | remove | cache/log/temp artifact |
| `src/babappa/models/__pycache__/baseline.cpython-311.pyc` | 22.8K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_neural_calibration.cpython-311-pytest-9.0.3.pyc` | 22.7K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_calibration.cpython-311-pytest-9.0.3.pyc` | 22.7K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_cycle48_reference_execution.cpython-311-pytest-9.0.3.pyc` | 22.6K | remove | cache/log/temp artifact |
| `src/babappa/tensors/__pycache__/build.cpython-311.pyc` | 22.5K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_large_run_plan.cpython-311-pytest-9.0.3.pyc` | 22.5K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_tensors.cpython-311-pytest-9.0.3.pyc` | 22.3K | remove | cache/log/temp artifact |
| `src/babappa/branch/__pycache__/calibration.cpython-311.pyc` | 22.2K | remove | cache/log/temp artifact |
| `src/babappa/reports/__pycache__/ablation_compare.cpython-311.pyc` | 22.0K | remove | cache/log/temp artifact |
| `src/babappa/branch/__pycache__/plan.cpython-311.pyc` | 21.6K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_datasets.cpython-311-pytest-9.0.3.pyc` | 21.4K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_align.cpython-311-pytest-9.0.3.pyc` | 21.3K | remove | cache/log/temp artifact |
| `src/babappa/reports/__pycache__/label_signal_audit.cpython-311.pyc` | 21.2K | remove | cache/log/temp artifact |
| `src/babappa/branch/__pycache__/aggregation.cpython-311.pyc` | 21.0K | remove | cache/log/temp artifact |
| `src/babappa/datasets/__pycache__/merge.cpython-311.pyc` | 20.9K | remove | cache/log/temp artifact |
| `src/babappa/training/__pycache__/neural_train.cpython-311.pyc` | 20.5K | remove | cache/log/temp artifact |
| `src/babappa/reports/__pycache__/neural_diagnostics.cpython-311.pyc` | 20.3K | remove | cache/log/temp artifact |
| `src/babappa/align/__pycache__/method_policy.cpython-311.pyc` | 20.3K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_neural_smoke_training.cpython-311-pytest-9.0.3.pyc` | 20.1K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_reports.cpython-311-pytest-9.0.3.pyc` | 19.9K | remove | cache/log/temp artifact |
| `src/babappa/training/__pycache__/neural_model.cpython-311.pyc` | 19.7K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_cycle49_babappa_only_handoff.cpython-311-pytest-9.0.3.pyc` | 19.7K | remove | cache/log/temp artifact |
| `src/babappa/training/__pycache__/neural_data.cpython-311.pyc` | 19.2K | remove | cache/log/temp artifact |
| `src/babappa/site/__pycache__/aggregation_controls.cpython-311.pyc` | 19.1K | remove | cache/log/temp artifact |
| `src/babappa/branch/__pycache__/baseline.cpython-311.pyc` | 18.9K | remove | cache/log/temp artifact |
| `src/babappa/site/__pycache__/calibration.cpython-311.pyc` | 17.8K | remove | cache/log/temp artifact |
| `src/babappa/benchmarks/__pycache__/stability.cpython-311.pyc` | 17.6K | remove | cache/log/temp artifact |
| `src/babappa/site/__pycache__/stability.cpython-311.pyc` | 17.5K | remove | cache/log/temp artifact |
| `src/babappa/reports/__pycache__/leakage_audit.cpython-311.pyc` | 17.0K | remove | cache/log/temp artifact |
| `src/babappa/branch/__pycache__/leakage.cpython-311.pyc` | 16.6K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_neural_infra.cpython-311-pytest-9.0.3.pyc` | 16.2K | remove | cache/log/temp artifact |
| `tests/__pycache__/test_storage_audit.cpython-311-pytest-9.0.3.pyc` | 16.0K | remove | cache/log/temp artifact |

Review `remove_candidates.tsv` and `archive_candidates.tsv` before running the quarantine script.
The generated quarantine script uses `mv` only; permanent deletion is a separate manual-confirmation script.
