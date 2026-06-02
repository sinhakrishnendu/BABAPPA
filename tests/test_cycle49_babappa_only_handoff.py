import json
from pathlib import Path

from typer.testing import CliRunner

from babappa.cli import app
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

runner = CliRunner()


def _mock_evidence_pack(tmp_path: Path, family: str = "WRKY_candidate_02_close") -> Path:
    pack = tmp_path / family
    (pack / "inputs").mkdir(parents=True)
    (pack / "inputs" / f"{family}.cds.fasta").write_text(">Arabidopsis_thaliana\nATGAAA\n", encoding="utf-8")
    (pack / "inputs" / f"{family}.treefile").write_text("(Arabidopsis_thaliana:0.1);\n", encoding="utf-8")
    scores = pack / "babappa" / "empirical_scores"
    scores.mkdir(parents=True)
    write_tsv(
        scores / "empirical_branch_site_scores.tsv",
        [{"prob_positive": "0.1", "called_positive": "1"}],
        ["prob_positive", "called_positive"],
    )
    write_tsv(
        scores / "empirical_gene_support.tsv",
        [{"method": "mafft", "max_prob_positive": "0.177189", "n_called_positive": "7"}],
        ["method", "max_prob_positive", "n_called_positive"],
    )
    write_tsv(
        scores / "empirical_branch_scores.tsv",
        [{"method": "mafft", "max_prob_positive": "0.2", "n_called_positive": "4"}],
        ["method", "max_prob_positive", "n_called_positive"],
    )
    app = pack / "babappa" / "empirical_applicability"
    app.mkdir(parents=True)
    write_tsv(app / "empirical_applicability.tsv", [{"status": "in_domain"}], ["status"])
    _write_json(
        app / "empirical_applicability.json",
        {
            "recommended_tier": "moderate",
            "validation": {
                "p_distance_used": 0.097198,
                "p_distance_source": "alignment_ensemble_mean",
                "n_taxa": 7,
                "n_codons": 490,
            },
        },
    )
    pref = pack / "prefilter"
    pref.mkdir(parents=True)
    write_tsv(pref / "empirical_family_prefilter.tsv", [{"decision": "accept"}], ["decision"])
    _write_json(
        pack / "simulation_matched_calibration_plan" / "simulation_matched_calibration_plan.json",
        {
            "proposed_simulation_parameters": {
                "n_taxa": 7,
                "n_codons": 490,
                "mean_pairwise_p_distance": 0.1,
                "recommended_tier": "moderate",
                "foreground": "Arabidopsis_thaliana",
            }
        },
    )
    return pack


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


def test_cli_help_includes_evidence_pack_null_calibration() -> None:
    result = runner.invoke(app, ["run-simulation-matched-null-calibration", "--help"])
    assert result.exit_code == 0
    assert "run-simulation-matched-null-calibration" in result.output
    assert "--evidence-pack" in result.output
    assert "--dry-run" in result.output


def test_evidence_pack_dry_run_creates_plan_files(tmp_path: Path) -> None:
    pack = _mock_evidence_pack(tmp_path)
    outdir = tmp_path / "dryrun"
    result = runner.invoke(
        app,
        [
            "run-simulation-matched-null-calibration",
            "--evidence-pack",
            str(pack),
            "--outdir",
            str(outdir),
            "--n-null",
            "100",
            "--seed",
            "20260530",
            "--device",
            "mps",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (outdir / "calibration_run_plan.json").exists()
    assert (outdir / "calibration_run_plan.md").exists()
    assert (outdir / "calibration_input_validation.tsv").exists()
    assert (outdir / "calibration_status.json").exists()
    assert (outdir / "calibration_status.md").exists()
    status = json.loads((outdir / "calibration_status.json").read_text())
    assert status["status"] == "dry_run"
    assert status["heavy_jobs_executed"] is False
    assert not (outdir / "matched_null_scores.tsv").exists()
    assert not (outdir / "matched_null_manifest.json").exists()


def test_evidence_pack_missing_path_fails_clearly(tmp_path: Path) -> None:
    outdir = tmp_path / "dryrun"
    result = runner.invoke(
        app,
        [
            "run-simulation-matched-null-calibration",
            "--evidence-pack",
            str(tmp_path / "missing_pack"),
            "--outdir",
            str(outdir),
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    assert "matched-null calibration input" in result.output
    assert "validation failed" in result.output
    status = json.loads((outdir / "calibration_status.json").read_text())
    assert status["status"] == "fail"


def test_evidence_pack_missing_required_inputs_fails_clearly(tmp_path: Path) -> None:
    pack = tmp_path / "WRKY_candidate_02_close"
    pack.mkdir()
    outdir = tmp_path / "dryrun"
    result = runner.invoke(
        app,
        [
            "run-simulation-matched-null-calibration",
            "--evidence-pack",
            str(pack),
            "--outdir",
            str(outdir),
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    rows = read_tsv(outdir / "calibration_input_validation.tsv")
    assert any(row["status"] == "missing" for row in rows)
    status = json.loads((outdir / "calibration_status.json").read_text())
    assert status["status"] == "fail"


def test_evidence_pack_feature_matched_backend_writes_expected_outputs(tmp_path: Path, monkeypatch) -> None:
    import babappa.empirical.reference_eval as reference_eval

    pack = _mock_evidence_pack(tmp_path)

    def fake_score_nulls(replicate_rows, params, package_dir, device_request, outdir):
        return [
            {
                **row,
                "status": "scored",
                "max_gene_support": 0.05,
                "max_branch_support": 0.04,
                "called_branch_site_rows": 1,
                "max_site_score": 0.05,
                "q95_site_score": 0.04,
                "q99_site_score": 0.045,
            }
            for row in replicate_rows
        ]

    monkeypatch.setattr(reference_eval, "_score_null_replicates_with_model", fake_score_nulls)
    result = run_simulation_matched_null_calibration(
        SimulationMatchedNullCalibrationConfig(
            plan_dir="",
            deployable_model_package="package",
            outdir=str(tmp_path / "nulls"),
            n_replicates=2,
            seed=7,
            evidence_pack=str(pack),
        )
    )
    assert result["status"] == "ok"
    for name in [
        "matched_null_manifest.json",
        "matched_null_manifest.tsv",
        "matched_null_scores.tsv",
        "matched_null_gene_support.tsv",
        "matched_null_branch_site_summary.tsv",
        "matched_null_calibration_summary.json",
        "matched_null_calibration_summary.tsv",
        "matched_null_calibration_report.md",
        "wrky_close_matched_null_interpretation.json",
        "wrky_close_matched_null_interpretation.md",
    ]:
        assert (tmp_path / "nulls" / name).exists()
    manifest = json.loads((tmp_path / "nulls" / "matched_null_manifest.json").read_text())
    assert manifest["calibration_backend"] == "feature_matched_deployable_model_null"
    assert manifest["null_results_fabricated"] is False


def test_user_run_null100_scripts_are_marked_user_run_only(tmp_path: Path) -> None:
    script = write_wrky_matched_null_script(str(tmp_path / "plan"), str(tmp_path / "null100"))
    assert "MANUAL EXECUTION SCRIPT" in script.read_text()
    for name in [
        "run_user_wrky_null100.sh",
        "monitor_user_wrky_null100.sh",
        "validate_user_wrky_null100.sh",
        "summarize_user_wrky_null100.sh",
    ]:
        path = tmp_path / "null100" / name
        assert path.exists()
        assert "MANUAL EXECUTION SCRIPT" in path.read_text()


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
        assert "MANUAL EXECUTION SCRIPT" in script.read_text()


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
    assert "The tooling does not automatically execute heavy empirical calibration" in calibration
