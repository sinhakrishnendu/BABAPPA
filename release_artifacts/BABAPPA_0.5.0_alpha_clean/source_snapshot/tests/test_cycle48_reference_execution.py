import json
from pathlib import Path

from babappa.datasets.index import read_tsv, write_tsv
from babappa.empirical.reference_eval import (
    CodemlReferenceParseConfig,
    HyphyReferenceParseConfig,
    ReferenceResultsTableConfig,
    ReferenceToolCheckConfig,
    ReferenceToolsInstallPlanConfig,
    SimulationMatchedNullCalibrationConfig,
    SimulationMatchedNullCalibrationValidationConfig,
    WRKYReferenceCalibrationReportConfig,
    build_reference_results_table,
    check_reference_tools,
    install_reference_tools_plan,
    make_wrky_reference_calibration_report,
    parse_codeml_reference,
    parse_hyphy_reference,
    run_simulation_matched_null_calibration,
    validate_simulation_matched_null_calibration,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_install_reference_tools_plan_writes_conda_and_brew_helpers(tmp_path: Path) -> None:
    result = install_reference_tools_plan(ReferenceToolsInstallPlanConfig(str(tmp_path / "install")))
    assert result["status"] == "planned"
    assert "paml hyphy" in (tmp_path / "install" / "install_reference_tools_conda.sh").read_text()
    assert "USER-RUN ONLY" in (tmp_path / "install" / "install_reference_tools_brew.sh").read_text()


def test_check_reference_tools_reports_missing_without_failure(tmp_path: Path, monkeypatch) -> None:
    import babappa.empirical.reference_eval as reference_eval

    monkeypatch.setattr(reference_eval.shutil, "which", lambda _name: None)
    result = check_reference_tools(ReferenceToolCheckConfig(str(tmp_path / "tools")))
    assert result["status"] == "ok"
    assert result["codeml"] is False
    assert result["hyphy"] is False


def test_parse_references_return_pending_tool_missing_if_outputs_absent(tmp_path: Path, monkeypatch) -> None:
    import babappa.empirical.reference_eval as reference_eval

    monkeypatch.setattr(reference_eval.shutil, "which", lambda _name: None)
    codeml_dir = tmp_path / "codeml"
    hyphy_dir = tmp_path / "hyphy"
    codeml_dir.mkdir()
    hyphy_dir.mkdir()
    codeml = parse_codeml_reference(
        CodemlReferenceParseConfig(str(codeml_dir), str(tmp_path / "codeml_parsed"))
    )
    hyphy = parse_hyphy_reference(
        HyphyReferenceParseConfig(str(hyphy_dir), str(tmp_path / "hyphy_parsed"))
    )
    assert codeml["status"] == "pending_tool_missing"
    assert hyphy["status"] == "pending_tool_missing"


def test_build_reference_results_table_writes_pending_rows_when_tools_absent(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "codeml_parsed" / "codeml_reference_parse.json",
        {"status": "pending_tool_missing", "result_class": "pending_tool_missing"},
    )
    _write_json(
        tmp_path / "hyphy_parsed" / "hyphy_reference_parse.json",
        {"status": "pending_tool_missing", "result_class": "pending_tool_missing"},
    )
    result = build_reference_results_table(
        ReferenceResultsTableConfig(
            panel_id="family",
            codeml_parsed=str(tmp_path / "codeml_parsed"),
            hyphy_parsed=str(tmp_path / "hyphy_parsed"),
            outdir=str(tmp_path / "refs"),
        )
    )
    rows = read_tsv(tmp_path / "refs" / "reference_results.tsv")
    assert result["status"] == "pending_tool_missing"
    assert len(rows) == 3
    assert {row["result_class"] for row in rows} == {"pending_tool_missing"}


def _mock_null_plan(tmp_path: Path) -> Path:
    family = tmp_path / "family"
    plan = family / "simulation_matched_calibration_plan"
    _write_json(
        plan / "simulation_matched_calibration_plan.json",
        {
            "empirical_validation_dir": str(family / "empirical_input"),
            "proposed_simulation_parameters": {
                "n_taxa": 7,
                "n_codons": 490,
                "mean_pairwise_p_distance": 0.1,
                "recommended_tier": "moderate",
                "foreground": "taxon1",
            },
        },
    )
    _write_json(
        family / "empirical_applicability" / "empirical_applicability.json",
        {"recommended_tier": "moderate", "validation": {"p_distance_used": 0.101, "p_distance_source": "alignment_ensemble_mean"}},
    )
    (family / "empirical_scores").mkdir(parents=True, exist_ok=True)
    write_tsv(
        family / "empirical_scores" / "empirical_gene_support.tsv",
        [{"method": "mafft", "max_prob_positive": "0.2", "n_called_positive": "7"}],
        ["method", "max_prob_positive", "n_called_positive"],
    )
    write_tsv(
        family / "empirical_scores" / "empirical_branch_scores.tsv",
        [{"method": "mafft", "max_prob_positive": "0.3", "n_called_positive": "4"}],
        ["method", "max_prob_positive", "n_called_positive"],
    )
    return plan


def test_run_simulation_matched_null_calibration_creates_staged_outputs(tmp_path: Path) -> None:
    plan = _mock_null_plan(tmp_path)
    result = run_simulation_matched_null_calibration(
        SimulationMatchedNullCalibrationConfig(
            plan_dir=str(plan),
            deployable_model_package="package",
            outdir=str(tmp_path / "nulls"),
            n_replicates=5,
            seed=10,
            fast_null_mode=True,
        )
    )
    assert result["status"] == "ok"
    rows = read_tsv(tmp_path / "nulls" / "matched_null_scores.tsv")
    assert len(rows) == 5
    assert rows[0]["status"] == "scored"


def test_validate_simulation_matched_null_calibration_detects_incomplete_run(tmp_path: Path) -> None:
    nulls = tmp_path / "nulls"
    nulls.mkdir()
    _write_json(
        nulls / "matched_null_manifest.json",
        {"n_replicates_requested": 5, "n_replicates_staged": 5, "n_replicates_completed": 0, "observed_values": {"max_gene_support": 0.2}},
    )
    _write_json(
        nulls / "matched_null_summary.json",
        {"null_scoring_completed": False, "p_empirical_support": None, "p_empirical_called_rows": None},
    )
    write_tsv(nulls / "matched_null_scores.tsv", [], ["replicate", "status"])
    result = validate_simulation_matched_null_calibration(
        SimulationMatchedNullCalibrationValidationConfig(str(tmp_path / "nulls"))
    )
    assert result["status"] == "fail"
    assert any("no_scored_null_replicates" in item for item in result["failures"])


def test_integrated_report_refuses_discovery_language_when_pending(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    _write_json(pack / "evidence_pack_validation.json", {"status": "ok"})
    run = tmp_path / "run"
    run.mkdir()
    write_tsv(
        run / "panel_run_summary.tsv",
        [{"panel_id": "WRKY_candidate_02_close", "applicability_status": "in_domain", "babappa_result_class": "positive", "max_gene_support": "0.2", "n_called_positive": "7"}],
        ["panel_id", "applicability_status", "babappa_result_class", "max_gene_support", "n_called_positive"],
    )
    refs = tmp_path / "reference_results.tsv"
    write_tsv(
        refs,
        [{"panel_id": "WRKY_candidate_02_close", "tool": "codeml", "test_name": "A", "p_value": "NA", "q_value": "NA", "selected_branch": "", "selected_sites": "", "result_class": "pending_tool_missing", "notes": ""}],
        ["panel_id", "tool", "test_name", "p_value", "q_value", "selected_branch", "selected_sites", "result_class", "notes"],
    )
    _write_json(tmp_path / "comparison" / "empirical_reference_comparison.json", {"status": "pending_tool_missing"})
    _write_json(tmp_path / "nulls" / "matched_null_summary.json", {"status": "staged_downstream_scoring_missing", "null_scoring_completed": False, "n_replicates_completed": 0})
    result = make_wrky_reference_calibration_report(
        WRKYReferenceCalibrationReportConfig(
            evidence_pack=str(pack),
            babappa_panel_run=str(run),
            reference_results=str(refs),
            comparison_dir=str(tmp_path / "comparison"),
            matched_null_calibration=str(tmp_path / "nulls"),
            outdir=str(tmp_path / "report"),
        )
    )
    text = (tmp_path / "report" / "wrky_reference_calibration_report.md").read_text()
    assert result["decision_category"] == "diagnostic_positive_reference_pending"
    assert "manuscript-ready: `False`" in text
    assert "positive selection discovered" not in text.lower()
