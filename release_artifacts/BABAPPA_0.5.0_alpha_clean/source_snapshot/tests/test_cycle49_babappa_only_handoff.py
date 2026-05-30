import json
from pathlib import Path

from babappa.datasets.index import read_tsv, write_tsv
from babappa.empirical.reference_eval import (
    BabappaOnlyResultAuditConfig,
    BabappaOnlySignalInterpretationConfig,
    CloseTaxaControlFamilyPlanConfig,
    SimulationMatchedNullCalibrationConfig,
    audit_babappa_only_result,
    interpret_babappa_only_signal,
    plan_close_taxa_control_family,
    run_simulation_matched_null_calibration,
    write_wrky_matched_null_script,
)
from test_cycle48_reference_execution import _mock_null_plan, _write_json


def test_null_calibration_runner_completes_tiny_fast_mode(tmp_path: Path) -> None:
    plan = _mock_null_plan(tmp_path)
    result = run_simulation_matched_null_calibration(
        SimulationMatchedNullCalibrationConfig(str(plan), "package", str(tmp_path / "null3"), 3, fast_null_mode=True)
    )
    assert result["n_replicates_completed"] == 3
    summary = json.loads((tmp_path / "null3" / "matched_null_summary.json").read_text())
    assert summary["p_empirical_support"] is not None
    assert (tmp_path / "null3" / ".stage_complete_generate_nulls").exists()
    assert (tmp_path / "null3" / ".stage_complete_score_nulls").exists()
    assert (tmp_path / "null3" / ".stage_complete_summarize_nulls").exists()


def test_user_run_null100_scripts_are_marked_user_run_only(tmp_path: Path) -> None:
    script = write_wrky_matched_null_script(str(tmp_path / "plan"), str(tmp_path / "null100"))
    assert "USER-RUN ONLY" in script.read_text()
    for name in [
        "run_user_wrky_null100.sh",
        "monitor_user_wrky_null100.sh",
        "validate_user_wrky_null100.sh",
        "summarize_user_wrky_null100.sh",
    ]:
        path = tmp_path / "null100" / name
        assert path.exists()
        assert "USER-RUN ONLY" in path.read_text()


def test_babappa_only_interpreter_returns_inconclusive_if_null_missing(tmp_path: Path) -> None:
    _write_json(tmp_path / "report.json", {"decision_category": "diagnostic_positive_calibration_pending"})
    write_tsv(
        tmp_path / "refs.tsv",
        [{"panel_id": "family", "tool": "codeml", "test_name": "A", "p_value": "1", "q_value": "", "selected_branch": "", "selected_sites": "", "result_class": "negative", "notes": ""}],
        ["panel_id", "tool", "test_name", "p_value", "q_value", "selected_branch", "selected_sites", "result_class", "notes"],
    )
    _write_json(tmp_path / "null" / "matched_null_summary.json", {"null_scoring_completed": False})
    result = interpret_babappa_only_signal(
        BabappaOnlySignalInterpretationConfig(str(tmp_path / "report.json"), str(tmp_path / "null"), str(tmp_path / "refs.tsv"), str(tmp_path / "interp"))
    )
    assert result["decision"] == "babappa_only_inconclusive"


def test_babappa_only_interpreter_not_supported_if_null_not_extreme(tmp_path: Path) -> None:
    _write_json(tmp_path / "report.json", {"decision_category": "diagnostic_positive_calibration_pending"})
    write_tsv(
        tmp_path / "refs.tsv",
        [{"panel_id": "family", "tool": "codeml", "test_name": "A", "p_value": "1", "q_value": "", "selected_branch": "", "selected_sites": "", "result_class": "negative", "notes": ""}],
        ["panel_id", "tool", "test_name", "p_value", "q_value", "selected_branch", "selected_sites", "result_class", "notes"],
    )
    _write_json(tmp_path / "null" / "matched_null_summary.json", {"null_scoring_completed": True, "p_empirical_support": 0.4, "p_empirical_called_rows": 0.2})
    result = interpret_babappa_only_signal(
        BabappaOnlySignalInterpretationConfig(str(tmp_path / "report.json"), str(tmp_path / "null"), str(tmp_path / "refs.tsv"), str(tmp_path / "interp"))
    )
    assert result["decision"] == "babappa_only_not_supported_by_null"


def test_babappa_only_interpreter_supported_only_when_null_extreme(tmp_path: Path) -> None:
    _write_json(tmp_path / "report.json", {"decision_category": "diagnostic_positive_calibration_pending"})
    write_tsv(
        tmp_path / "refs.tsv",
        [{"panel_id": "family", "tool": "hyphy", "test_name": "MEME", "p_value": "0.9", "q_value": "", "selected_branch": "", "selected_sites": "", "result_class": "negative", "notes": ""}],
        ["panel_id", "tool", "test_name", "p_value", "q_value", "selected_branch", "selected_sites", "result_class", "notes"],
    )
    _write_json(tmp_path / "null" / "matched_null_summary.json", {"null_scoring_completed": True, "p_empirical_support": 0.01, "p_empirical_called_rows": 0.5})
    result = interpret_babappa_only_signal(
        BabappaOnlySignalInterpretationConfig(str(tmp_path / "report.json"), str(tmp_path / "null"), str(tmp_path / "refs.tsv"), str(tmp_path / "interp"))
    )
    assert result["decision"] == "babappa_only_supported_by_null"


def test_negative_control_planner_writes_user_run_scripts(tmp_path: Path) -> None:
    taxa = tmp_path / "taxa.tsv"
    taxa.write_text("taxon_label\nArabidopsis_thaliana\n", encoding="utf-8")
    result = plan_close_taxa_control_family(
        CloseTaxaControlFamilyPlanConfig("conserved_control_01", "Arabidopsis_thaliana", "ACTIN", str(taxa), str(tmp_path / "control"))
    )
    assert result["executed"] is False
    for script in (tmp_path / "control").glob("*.sh"):
        assert "USER-RUN ONLY" in script.read_text()


def test_babappa_only_audit_detects_method_concentrated_scores(tmp_path: Path) -> None:
    family = "family"
    score_dir = tmp_path / "run" / "per_family" / family / "empirical_scores"
    app_dir = tmp_path / "run" / "per_family" / family / "empirical_applicability"
    align_dir = tmp_path / "run" / "per_family" / family / "empirical_alignment"
    score_dir.mkdir(parents=True)
    app_dir.mkdir(parents=True)
    align_dir.mkdir(parents=True)
    _write_json(app_dir / "empirical_applicability.json", {"applicability_status": "in_domain", "recommended_tier": "moderate"})
    _write_json(align_dir / "empirical_alignment_manifest.json", {"status": "ok"})
    write_tsv(
        score_dir / "empirical_branch_site_scores.tsv",
        [
            {"method": "babappalign", "branch_id": "b1", "called_positive": "1", "prob_positive": "0.9"},
            {"method": "babappalign", "branch_id": "b1", "called_positive": "1", "prob_positive": "0.8"},
            {"method": "mafft", "branch_id": "b2", "called_positive": "0", "prob_positive": "0.1"},
        ],
        ["method", "branch_id", "called_positive", "prob_positive"],
    )
    write_tsv(score_dir / "empirical_branch_scores.tsv", [], ["method"])
    write_tsv(score_dir / "empirical_gene_support.tsv", [], ["method"])
    write_tsv(
        tmp_path / "refs.tsv",
        [{"panel_id": family, "tool": "codeml", "test_name": "A", "p_value": "1", "q_value": "", "selected_branch": "", "selected_sites": "", "result_class": "negative", "notes": ""}],
        ["panel_id", "tool", "test_name", "p_value", "q_value", "selected_branch", "selected_sites", "result_class", "notes"],
    )
    result = audit_babappa_only_result(
        BabappaOnlyResultAuditConfig(family, str(tmp_path / "run"), str(tmp_path / "refs.tsv"), str(tmp_path / "audit"))
    )
    assert result["status"] == "warning"
    assert any("babappalign_driven_signal" in warning for warning in result["warnings"])


def test_docs_contain_long_run_handoff_policy() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    calibration = Path("docs/SIMULATION_MATCHED_EMPIRICAL_CALIBRATION.md").read_text(encoding="utf-8")
    assert "Long-Run Handoff Policy" in readme
    assert "Codex does not execute heavy empirical calibration" in calibration
