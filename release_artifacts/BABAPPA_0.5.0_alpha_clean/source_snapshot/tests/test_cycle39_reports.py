import json
from pathlib import Path

from babappa.branch.cycle39_report import (
    DeployableModelPackagePlanConfig,
    Final100KValidationReportConfig,
    ValidationScaleComparisonConfig,
    build_final_100k_validation_report,
    compare_validation_scales,
    plan_deployable_model_package,
)
from babappa.datasets.index import write_tsv


def _write_summary(path: Path, run: str, auroc: float = 0.99) -> None:
    path.mkdir(parents=True)
    tier_rows = []
    neural_rows = []
    branch_rows = []
    gene_rows = []
    calibration_rows = []
    controls_rows = []
    thresholds_rows = []
    for tier in ["low", "moderate", "high", "extreme"]:
        tier_rows.append(
            {
                "tier": tier,
                "status": "complete",
                "run_summary_dir": f"branch_site_run_summary_{run}_{tier}_streamed",
                "branch_site_label_status": "explicit_simulator_branch_truth",
                "branch_site_rows": "240000000",
                "branch_site_positives": "752220",
                "branch_site_neural_test_auroc": str(auroc),
                "branch_site_neural_all_auroc": str(auroc),
                "branch_level_all_auroc": "0.999",
                "branch_level_test_auroc": "0.999",
                "gene_level_all_auroc": "0.999",
                "gene_level_test_auroc": "0.999",
                "calibration_temperature": "1.0",
                "calibration_selected_threshold": "0.1",
                "branch_site_threshold_policy_profiles": "7",
                "branch_aggregation_threshold_policy_profiles": "7",
                "controls_observed_branch_auroc": "0.999",
                "run_summary_warnings": "missing_or_incomplete:branch_site_baseline",
                "optional_warnings": "",
            }
        )
        neural_rows.append(
            {
                "tier": tier,
                "level": "branch_site_neural",
                "split": "test",
                "n": "50000",
                "positives": "100",
                "negatives": "49900",
                "auroc": str(auroc),
                "accuracy": "0.9",
                "precision": "0.8",
                "recall": "0.7",
                "f1": "0.75",
                "mcc": "0.7",
                "specificity": "0.9",
            }
        )
        for rows, level in [(branch_rows, "branch_level_aggregation"), (gene_rows, "branch_to_gene_aggregation")]:
            rows.append(
                {
                    "tier": tier,
                    "level": level,
                    "split": "all",
                    "n": "100",
                    "positives": "50",
                    "negatives": "50",
                    "auroc": "0.999",
                    "accuracy": "0.9",
                    "precision": "0.9",
                    "recall": "0.9",
                    "f1": "0.9",
                    "mcc": "0.8",
                    "specificity": "0.9",
                }
            )
        calibration_rows.append(
            {
                "tier": tier,
                "calibration_dir": f"branch_site_calibration_{run}_{tier}_streamed",
                "temperature": "1.0",
                "selected_threshold": "0.1",
                "target_fdr": "0.1",
                "calibration_split_size": "50",
                "calibration_split_positive_count": "10",
                "raw_brier": "0.1",
                "calibrated_brier": "0.1",
                "raw_ece": "0.1",
                "calibrated_ece": "0.1",
                "warnings": "",
            }
        )
        controls_rows.append(
            {
                "tier": tier,
                "control": "global_shuffled_branch_labels",
                "observed_auroc": "0.999",
                "mean_auroc": "0.5",
                "q05_auroc": "0.49",
                "q95_auroc": "0.51",
                "min_auroc": "0.49",
                "max_auroc": "0.51",
                "std_auroc": "0.01",
                "n_permutations": "10",
                "empirical_p_value": "0.01",
                "control_interpretation": "shuffle",
                "expected_behavior": "random",
                "whether_control_is_destructive_enough": "yes",
            }
        )
    write_tsv(path / "branch_conditioned_tier_summary.tsv", tier_rows, list(tier_rows[0]))
    write_tsv(path / "branch_site_neural_performance.tsv", neural_rows, list(neural_rows[0]))
    write_tsv(path / "branch_aggregation_performance.tsv", branch_rows, list(branch_rows[0]))
    write_tsv(path / "branch_gene_aggregation_performance.tsv", gene_rows, list(gene_rows[0]))
    write_tsv(path / "branch_calibration_summary.tsv", calibration_rows, list(calibration_rows[0]))
    write_tsv(path / "branch_controls_summary.tsv", controls_rows, list(controls_rows[0]))
    write_tsv(path / "branch_threshold_policy_summary.tsv", thresholds_rows, ["tier"])
    (path / "branch_conditioned_tier_summary.json").write_text(json.dumps({"status": "ok"}))


def _write_truth(path: Path) -> None:
    path.mkdir(parents=True)
    rows = [
        {
            "tier": tier,
            "label_dir": f"branch_site_oracle_explicit_branch_truth_100k_mps_{tier}",
            "audit_status": "explicit_truth_ok",
            "label_status": "explicit_simulator_branch_truth",
            "explicit_branch_site_truth_available": "True",
            "proxy_from_foreground_taxon": "False",
            "not_available": "False",
            "n_branch_site_rows": "240000000",
            "n_positive_branch_sites": "752220",
            "positive_branch_site_fraction": "0.003",
        }
        for tier in ["low", "moderate", "high", "extreme"]
    ]
    write_tsv(path / "branch_truth_status_audit.tsv", rows, list(rows[0]))


def test_compare_validation_scales_handles_missing_optional_fields(tmp_path: Path) -> None:
    small = tmp_path / "small"
    large = tmp_path / "large"
    _write_summary(small, "explicit_branch_truth_10k_mps", 0.98)
    _write_summary(large, "explicit_branch_truth_100k_mps", 0.99)
    result = compare_validation_scales(
        ValidationScaleComparisonConfig(
            small_run="explicit_branch_truth_10k_mps",
            large_run="explicit_branch_truth_100k_mps",
            small_summary=str(small),
            large_summary=str(large),
            outdir=str(tmp_path / "comparison"),
        )
    )
    assert result["status"] == "ok"
    payload = json.loads((tmp_path / "comparison" / "scale_comparison.json").read_text())
    assert payload["rows"][0]["delta_branch_site_neural_test_auroc"] > 0


def test_final_report_builder_contains_simulation_caution(tmp_path: Path) -> None:
    summary = tmp_path / "summary"
    truth = tmp_path / "truth"
    plan = tmp_path / "explicit_branch_truth_100k_mps_plan_blazing"
    marker_dir = plan / "stage_markers"
    marker_dir.mkdir(parents=True)
    for tier in ["low", "moderate", "high", "extreme"]:
        for i in range(26):
            (marker_dir / f".stage_complete_{tier}_{i}").write_text("")
    (plan / "preflight_report.json").write_text(json.dumps({"status": "pass", "n_checks": 1, "n_fail": 0}))
    (plan / "mps_plan_script_validation.json").write_text(json.dumps({"status": "pass", "n_checks": 1, "n_fail": 0}))
    _write_summary(summary, "explicit_branch_truth_100k_mps", 0.99)
    _write_truth(truth)
    result = build_final_100k_validation_report(
        Final100KValidationReportConfig(
            run_name="explicit_branch_truth_100k_mps",
            summary_dir=str(summary),
            truth_audit_dir=str(truth),
            plan_dir=str(plan),
            comparison_dir=None,
            outdir=str(tmp_path),
        )
    )
    assert result["decision"] == "CONDITIONAL PASS"
    text = (tmp_path / "explicit_branch_truth_100k_mps_final_validation_report.md").read_text()
    assert "simulation_supervised_only" in text
    assert "no_final_empirical_inference_claim" in text


def test_deployable_model_package_planner_refuses_missing_artifacts(tmp_path: Path, monkeypatch) -> None:
    summary = tmp_path / "summary"
    truth = tmp_path / "truth"
    _write_summary(summary, "explicit_branch_truth_100k_mps", 0.99)
    _write_truth(truth)
    monkeypatch.chdir(tmp_path)
    result = plan_deployable_model_package(
        DeployableModelPackagePlanConfig(
            run_name="explicit_branch_truth_100k_mps",
            summary_dir=str(summary),
            truth_audit_dir=str(truth),
            outdir=str(tmp_path / "package_plan"),
        )
    )
    assert result["blocked"] is True
    assert result["missing_artifacts"]


def test_empirical_transition_plan_contains_claim_boundary() -> None:
    text = Path("docs/POST_100K_EMPIRICAL_TRANSITION_PLAN.md").read_text()
    assert "does not establish final empirical branch-site inference" in text
    assert "non-defensible claim" in text
