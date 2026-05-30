import json
from pathlib import Path

from babappa.datasets.index import read_tsv, write_tsv
from babappa.empirical.pilot_panel import (
    EmpiricalReferenceComparisonConfig,
    compare_empirical_reference_results,
)
from babappa.empirical.reference_eval import (
    CodemlReferenceParseConfig,
    CodemlReferencePrepConfig,
    EmpiricalEvidencePackConfig,
    EmpiricalEvidencePackValidationConfig,
    HyphyReferenceParseConfig,
    HyphyReferencePrepConfig,
    ReferenceResultsTemplateConfig,
    ReferenceToolCheckConfig,
    SimulationMatchedCalibrationSummaryConfig,
    WRKYInterpretationStatusConfig,
    check_reference_tools,
    freeze_empirical_evidence_pack,
    make_wrky_interpretation_status,
    parse_codeml_reference,
    parse_hyphy_reference,
    prepare_codeml_reference,
    prepare_hyphy_reference,
    summarize_simulation_matched_calibration_plan,
    validate_empirical_evidence_pack,
    write_reference_results_template,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tiny_cds_tree(tmp_path: Path) -> tuple[Path, Path]:
    cds = tmp_path / "family.cds.fasta"
    tree = tmp_path / "family.treefile"
    cds.write_text(
        ">taxon1\nATGGCTGCTTAA\n>taxon2\nATGGCTGCCTAA\n",
        encoding="utf-8",
    )
    tree.write_text("(taxon1:0.1,taxon2:0.1);\n", encoding="utf-8")
    return cds, tree


def _minimal_pack_sources(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    cds, tree = _tiny_cds_tree(tmp_path)
    prefilter = tmp_path / "prefilter"
    family = tmp_path / "family"
    prefilter.mkdir()
    _write_json(prefilter / "empirical_family_prefilter.json", {"decision": "accept"})
    _write_json(
        family / "empirical_applicability" / "empirical_applicability.json",
        {"applicability_status": "in_domain", "recommended_tier": "moderate"},
    )
    _write_json(
        family / "empirical_scores" / "empirical_scoring_manifest.json",
        {"status": "ok", "diagnostic_only": False, "tier_model": "moderate"},
    )
    write_tsv(
        family / "empirical_scores" / "empirical_branch_site_scores.tsv",
        [{"branch": "taxon1", "site": "1", "prob_positive": "0.1"}],
        ["branch", "site", "prob_positive"],
    )
    write_tsv(
        family / "empirical_scores" / "empirical_gene_support.tsv",
        [{"method": "mafft", "max_prob_positive": "0.1", "n_called_positive": "0"}],
        ["method", "max_prob_positive", "n_called_positive"],
    )
    (family / "empirical_scores" / "branch_site_truth.tsv").write_text("forbidden\n", encoding="utf-8")
    panel = tmp_path / "panel_run_summary.tsv"
    write_tsv(
        panel,
        [{"panel_id": "family", "applicability_status": "in_domain", "diagnostic_only": "False"}],
        ["panel_id", "applicability_status", "diagnostic_only"],
    )
    return cds, tree, prefilter, family, panel


def test_evidence_pack_excludes_simulator_truth_and_validates(tmp_path: Path) -> None:
    cds, tree, prefilter, family, panel = _minimal_pack_sources(tmp_path)
    outdir = tmp_path / "pack"
    freeze_empirical_evidence_pack(
        EmpiricalEvidencePackConfig(
            family_id="family",
            outdir=str(outdir),
            cds_fasta=str(cds),
            tree_file=str(tree),
            foreground="taxon1",
            babappa_family_dir=str(family),
            panel_run_summary=str(panel),
            prefilter_dir=str(prefilter),
        )
    )
    assert not any("truth" in str(path) for path in outdir.rglob("*"))
    result = validate_empirical_evidence_pack(EmpiricalEvidencePackValidationConfig(str(outdir)))
    assert result["status"] == "ok"


def test_codeml_planner_writes_model_templates_and_foreground_tree(tmp_path: Path) -> None:
    cds, tree = _tiny_cds_tree(tmp_path)
    result = prepare_codeml_reference(
        CodemlReferencePrepConfig(str(cds), str(tree), "taxon1", str(tmp_path / "codeml"))
    )
    assert result["status"] == "prepared"
    assert "model = 2" in (tmp_path / "codeml" / "codeml_modelA.ctl").read_text()
    assert "taxon1#1" in (tmp_path / "codeml" / "tree_foreground.nwk").read_text()
    assert "USER-RUN ONLY" in (tmp_path / "codeml" / "run_codeml_modelA.sh").read_text()


def test_hyphy_planner_writes_absrel_meme_templates(tmp_path: Path) -> None:
    cds, tree = _tiny_cds_tree(tmp_path)
    result = prepare_hyphy_reference(
        HyphyReferencePrepConfig(str(cds), str(tree), "taxon1", str(tmp_path / "hyphy"))
    )
    assert result["status"] == "prepared"
    assert "absrel" in (tmp_path / "hyphy" / "run_absrel.sh").read_text()
    assert "meme" in (tmp_path / "hyphy" / "run_meme.sh").read_text()
    assert "taxon1{Foreground}" in (tmp_path / "hyphy" / "tree_foreground.nwk").read_text()
    expected = json.loads((tmp_path / "hyphy" / "expected_outputs.json").read_text())
    assert expected["hyphy_safe_alignment"]["stop_codons_replaced_with_NNN"] >= 1


def test_reference_tool_checker_handles_missing_tools_gracefully(tmp_path: Path, monkeypatch) -> None:
    import babappa.empirical.reference_eval as reference_eval

    monkeypatch.setattr(reference_eval.shutil, "which", lambda _name: None)
    result = check_reference_tools(ReferenceToolCheckConfig(str(tmp_path / "tools")))
    assert result["status"] == "ok"
    assert result["codeml"] is False
    assert result["hyphy"] is False


def test_reference_parsers_return_pending_when_outputs_absent(tmp_path: Path) -> None:
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
    assert codeml["status"] in {"pending_not_run", "pending_tool_missing"}
    assert hyphy["status"] in {"pending_not_run", "pending_tool_missing"}


def test_reference_results_template_has_required_schema(tmp_path: Path) -> None:
    result = write_reference_results_template(
        ReferenceResultsTemplateConfig("family", "taxon1", str(tmp_path / "refs"))
    )
    rows = read_tsv(Path(result["path"]))
    assert [row["result_class"] for row in rows] == ["pending", "pending", "pending"]
    assert rows[0]["test_name"] == "branch_site_model_A_vs_null"


def test_reference_comparison_reports_pending_reference_results(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_json(run / "panel_run_manifest.json", {"status": "ok"})
    write_tsv(
        run / "panel_run_summary.tsv",
        [{"panel_id": "family", "babappa_result_class": "positive", "diagnostic_only": "False"}],
        ["panel_id", "babappa_result_class", "diagnostic_only"],
    )
    refs = tmp_path / "refs.tsv"
    write_tsv(
        refs,
        [{"panel_id": "family", "tool": "codeml", "test_name": "A", "p_value": "NA", "q_value": "NA", "selected_branch": "", "selected_sites": "", "result_class": "pending", "notes": ""}],
        ["panel_id", "tool", "test_name", "p_value", "q_value", "selected_branch", "selected_sites", "result_class", "notes"],
    )
    result = compare_empirical_reference_results(
        EmpiricalReferenceComparisonConfig(str(run), str(refs), str(tmp_path / "comparison"))
    )
    assert result["status"] == "pending_reference_results"
    rows = read_tsv(tmp_path / "comparison" / "empirical_reference_comparison.tsv")
    assert rows[0]["concordance_class"] == "pending_reference_results"


def test_calibration_summary_says_not_interpretable_before_calibration(tmp_path: Path) -> None:
    family = tmp_path / "family"
    plan = family / "simulation_matched_calibration_plan"
    _write_json(
        plan / "simulation_matched_calibration_plan.json",
        {
            "empirical_validation_dir": str(family / "empirical_input"),
            "proposed_simulation_parameters": {
                "n_taxa": 7,
                "n_codons": 490,
                "mean_pairwise_p_distance": 0.66,
                "recommended_tier": "extreme",
            },
        },
    )
    _write_json(plan / "expected_outputs.json", {"expected_outputs": ["null_scores.tsv"]})
    _write_json(
        family / "empirical_applicability" / "empirical_applicability.json",
        {
            "recommended_tier": "moderate",
            "validation": {
                "p_distance_used": 0.101,
                "p_distance_source": "alignment_ensemble_mean",
            },
        },
    )
    result = summarize_simulation_matched_calibration_plan(
        SimulationMatchedCalibrationSummaryConfig(str(plan), str(tmp_path / "summary"))
    )
    payload = json.loads(Path(result["json"]).read_text())
    assert payload["interpretable_before_calibration"] is False
    assert payload["matched_tier"] == "moderate"
    assert "not interpretable" in (tmp_path / "summary" / "simulation_matched_calibration_summary.md").read_text()


def test_interpretation_report_contains_not_manuscript_ready_boundary(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    write_tsv(
        run / "panel_run_summary.tsv",
        [{"panel_id": "family", "babappa_result_class": "positive", "diagnostic_only": "False"}],
        ["panel_id", "babappa_result_class", "diagnostic_only"],
    )
    result = make_wrky_interpretation_status(
        WRKYInterpretationStatusConfig(
            family_id="family",
            babappa_panel_run=str(run),
            evidence_pack="pack",
            calibration_summary="summary",
            reference_results="refs",
            outdir=str(tmp_path / "status"),
        )
    )
    text = (tmp_path / "status" / "family_interpretation_status.md").read_text()
    assert result["manuscript_ready"] is False
    assert "manuscript-ready: `False`" in text
    assert "not a final positive-selection discovery claim" in text
