import json
from pathlib import Path

from babappa.datasets.index import write_tsv
from babappa.empirical.pilot_panel import (
    CLAIM_BOUNDARY_TEXT,
    EmpiricalPilotPanelSummaryConfig,
    EmpiricalPilotSummaryValidationConfig,
    RealEmpiricalPilotDecisionReportConfig,
    RealEmpiricalPilotWorkspaceConfig,
    make_real_empirical_pilot_decision_report,
    prepare_real_empirical_pilot_workspace,
    summarize_empirical_pilot_panel,
    validate_empirical_pilot_summary,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_real_pilot_workspace_generator_works(tmp_path: Path) -> None:
    result = prepare_real_empirical_pilot_workspace(
        RealEmpiricalPilotWorkspaceConfig(str(tmp_path / "real_empirical_pilot"), max_families=12)
    )
    workspace = tmp_path / "real_empirical_pilot"
    assert result["manifest_created"] is True
    assert result["families"] == 12
    for dirname in ["input", "manifest", "babappa_run", "reference_plan", "reference_results", "comparison", "summary", "logs"]:
        assert (workspace / dirname).is_dir()
    assert (workspace / "manifest" / "real_empirical_pilot_panel.tsv").exists()


def test_readiness_report_generated_if_manifest_files_missing(tmp_path: Path) -> None:
    result = prepare_real_empirical_pilot_workspace(
        RealEmpiricalPilotWorkspaceConfig(str(tmp_path / "real_empirical_pilot"), max_families=8)
    )
    workspace = tmp_path / "real_empirical_pilot"
    payload = json.loads((workspace / "summary" / "real_empirical_pilot_readiness_report.json").read_text())
    assert result["status"] == "NEED_INPUT_REPAIR"
    assert payload["n_missing_inputs"] > 0
    assert payload["run_babappa_pilot"] is False
    assert "simulation-trained" in (workspace / "summary" / "real_empirical_pilot_readiness_report.md").read_text()


def test_real_pilot_summary_handles_absent_reference_results(tmp_path: Path) -> None:
    run = tmp_path / "babappa_run"
    run.mkdir()
    _write_json(run / "panel_run_manifest.json", {"status": "ok"})
    write_tsv(
        run / "panel_run_summary.tsv",
        [{"panel_id": "real1", "input_status": "pass", "applicability_status": "borderline", "scoring_status": "ok", "diagnostic_only": "False"}],
        ["panel_id", "input_status", "applicability_status", "scoring_status", "diagnostic_only"],
    )
    result = summarize_empirical_pilot_panel(
        EmpiricalPilotPanelSummaryConfig(panel_run=str(run), outdir=str(tmp_path / "summary"))
    )
    payload = json.loads((tmp_path / "summary" / "empirical_pilot_panel_summary.json").read_text())
    assert result["status"] == "ok"
    assert payload["reference_comparison"] is None
    assert payload["reference_concordance_counts"] == {}


def test_real_pilot_decision_report_includes_claim_boundary(tmp_path: Path) -> None:
    workspace = tmp_path / "real_empirical_pilot"
    prepare_real_empirical_pilot_workspace(RealEmpiricalPilotWorkspaceConfig(str(workspace), max_families=8))
    result = make_real_empirical_pilot_decision_report(
        RealEmpiricalPilotDecisionReportConfig(workspace=str(workspace), outdir=str(workspace / "summary"))
    )
    text = (workspace / "summary" / "real_empirical_pilot_decision_report.md").read_text()
    assert result["decision"] == "NEED_INPUT_REPAIR"
    assert "No simulator truth was used for empirical inference" in text
    assert "not ready for claims" in text


def test_forbidden_discovery_language_rejected_by_summary_validator(tmp_path: Path) -> None:
    summary = tmp_path / "summary"
    summary.mkdir()
    _write_json(summary / "empirical_pilot_panel_summary.json", {"status": "ok", "claim_boundary": CLAIM_BOUNDARY_TEXT})
    (summary / "empirical_pilot_panel_summary.md").write_text(
        CLAIM_BOUNDARY_TEXT + "\npositive selection discovered\n",
        encoding="utf-8",
    )
    result = validate_empirical_pilot_summary(EmpiricalPilotSummaryValidationConfig(str(summary)))
    assert result["status"] == "fail"
    assert any(item.startswith("forbidden_discovery_language") for item in result["failures"])
