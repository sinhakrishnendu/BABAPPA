import json
import os
from pathlib import Path

import pytest

from babappa.datasets.index import write_tsv
from babappa.empirical.pilot_panel import (
    ClassicalReferenceWorkflowPlanConfig,
    EmpiricalPilotPanelRunConfig,
    EmpiricalPilotPanelSummaryConfig,
    EmpiricalPilotPanelValidationConfig,
    EmpiricalPilotSummaryValidationConfig,
    EmpiricalReferenceComparisonConfig,
    compare_empirical_reference_results,
    plan_classical_reference_workflows,
    run_empirical_pilot_panel,
    summarize_empirical_pilot_panel,
    validate_empirical_pilot_panel,
    validate_empirical_pilot_summary,
)
from babappa.empirical.bridge import EmpiricalAlignmentEnsembleConfig, run_empirical_alignment_ensemble
from test_cycle41_empirical_bridge import _mock_aligners, _synthetic_package, _tiny_inputs, _write_json


PANEL = Path("tests/data/empirical_pilot_panel/empirical_pilot_panel.tsv")
REFERENCES = Path("tests/data/empirical_pilot_panel/mock_reference_results.tsv")


def test_empirical_pilot_panel_validator_accepts_valid_tiny_manifest(tmp_path: Path) -> None:
    result = validate_empirical_pilot_panel(
        EmpiricalPilotPanelValidationConfig(str(PANEL), str(tmp_path / "validation"))
    )
    assert result["status"] == "ok"
    assert result["n_rows"] == 4
    payload = json.loads((tmp_path / "validation" / "empirical_pilot_panel_validation.json").read_text())
    assert payload["category_counts"]["known_positive"] == 1


def test_empirical_pilot_panel_validator_rejects_missing_required_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad_panel.tsv"
    write_tsv(
        bad,
        [{"panel_id": "x", "cds_fasta": "missing.fasta", "tree_file": "missing.tree"}],
        ["panel_id", "cds_fasta", "tree_file"],
    )
    result = validate_empirical_pilot_panel(
        EmpiricalPilotPanelValidationConfig(str(bad), str(tmp_path / "validation"))
    )
    assert result["status"] == "fail"
    assert any(item.startswith("missing_required_columns") for item in result["failures"])


def test_empirical_pilot_panel_validator_catches_duplicate_panel_id(tmp_path: Path) -> None:
    fasta, tree = _tiny_inputs(tmp_path)
    panel = tmp_path / "dup_panel.tsv"
    rows = [
        {
            "panel_id": "dup",
            "gene_family": "g",
            "species_group": "s",
            "cds_fasta": str(fasta),
            "tree_file": str(tree),
            "foreground": "taxon1",
            "expected_category": "unknown",
            "reference_status": "planned",
            "notes": "",
        },
        {
            "panel_id": "dup",
            "gene_family": "g2",
            "species_group": "s",
            "cds_fasta": str(fasta),
            "tree_file": str(tree),
            "foreground": "taxon1",
            "expected_category": "unknown",
            "reference_status": "planned",
            "notes": "",
        },
    ]
    write_tsv(panel, rows, list(rows[0]))
    result = validate_empirical_pilot_panel(
        EmpiricalPilotPanelValidationConfig(str(panel), str(tmp_path / "validation"))
    )
    assert result["status"] == "fail"
    assert any("duplicate_panel_id:dup" in item for item in result["failures"])


def test_pilot_runner_continues_after_failed_qc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = _synthetic_package(tmp_path)
    fasta, tree = _tiny_inputs(tmp_path)
    bad_fasta = tmp_path / "bad.cds.fasta"
    bad_tree = tmp_path / "bad.treefile"
    bad_fasta.write_text(">taxon1\nATGGCTCAA\n>taxon2\nATGGCTCAA\n", encoding="utf-8")
    bad_tree.write_text("(taxon1:0.1,taxon2:0.1);\n", encoding="utf-8")
    panel = tmp_path / "panel.tsv"
    rows = [
        {
            "panel_id": "good1",
            "gene_family": "good1",
            "species_group": "synthetic",
            "cds_fasta": str(fasta),
            "tree_file": str(tree),
            "foreground": "taxon1",
            "expected_category": "unknown",
            "reference_status": "planned",
            "notes": "",
        },
        {
            "panel_id": "bad_qc",
            "gene_family": "bad",
            "species_group": "synthetic",
            "cds_fasta": str(bad_fasta),
            "tree_file": str(bad_tree),
            "foreground": "taxon1",
            "expected_category": "short_low_information",
            "reference_status": "planned",
            "notes": "",
        },
        {
            "panel_id": "good2",
            "gene_family": "good2",
            "species_group": "synthetic",
            "cds_fasta": str(fasta),
            "tree_file": str(tree),
            "foreground": "taxon1",
            "expected_category": "unknown",
            "reference_status": "planned",
            "notes": "",
        },
    ]
    write_tsv(panel, rows, list(rows[0]))

    def fake_score(config):
        outdir = Path(config.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        _write_json(outdir / "empirical_scoring_manifest.json", {"status": "ok", "diagnostic_only": False})
        write_tsv(
            outdir / "empirical_gene_support.tsv",
            [{"family_id": "empirical", "method": "identity", "max_prob_positive": "0.2", "n_called_positive": "0"}],
            ["family_id", "method", "max_prob_positive", "n_called_positive"],
        )
        return {"status": "ok", "diagnostic_only": False}

    import babappa.empirical.pilot_panel as pilot_panel

    monkeypatch.setattr(pilot_panel, "score_empirical_branch_sites", fake_score)
    result = run_empirical_pilot_panel(
        EmpiricalPilotPanelRunConfig(
            panel_manifest=str(panel),
            deployable_model_package=str(package),
            outdir=str(tmp_path / "run"),
            methods="identity",
            device="cpu",
            max_families=3,
            fail_fast=False,
        )
    )
    assert result["families_processed"] == 3
    assert result["qc_fail"] == 1
    summary = (tmp_path / "run" / "panel_run_summary.tsv").read_text()
    assert "good2" in summary


def test_classical_reference_planner_writes_templates_without_executing(tmp_path: Path) -> None:
    result = plan_classical_reference_workflows(
        ClassicalReferenceWorkflowPlanConfig(str(PANEL), str(tmp_path / "classical"), "codeml,hyphy")
    )
    assert result["executed"] is False
    assert "MANUAL EXECUTION SCRIPT" in (tmp_path / "classical" / "codeml_commands.sh").read_text()
    assert (tmp_path / "classical" / "hyphy_commands.sh").exists()


def test_empirical_babappalign_runner_uses_absolute_input_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fasta, tree = _tiny_inputs(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "babappalign").write_text(
        "#!/usr/bin/env bash\n"
        "input=\"${@: -1}\"\n"
        "case \"$input\" in /*) cat \"$input\" ;; *) echo \"relative input: $input\" >&2; exit 2 ;; esac\n",
        encoding="utf-8",
    )
    (bin_dir / "babappalign").chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    result = run_empirical_alignment_ensemble(
        EmpiricalAlignmentEnsembleConfig(
            cds_fasta=str(fasta),
            tree=str(tree),
            foreground="taxon1",
            outdir=str(tmp_path / "alignment"),
            methods="babappalign",
            require_babappalign=True,
        )
    )
    assert result["status"] == "ok"
    assert result["methods_run"] == ["babappalign"]


def test_reference_comparison_classifies_concordant_and_discordant_cases(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_json(run / "panel_run_manifest.json", {"status": "ok", "claim_boundary": "boundary"})
    write_tsv(
        run / "panel_run_summary.tsv",
        [
            {
                "panel_id": "a",
                "applicability_status": "in_domain",
                "babappa_result_class": "positive",
                "max_gene_support": "0.9",
                "diagnostic_only": "False",
            },
            {
                "panel_id": "b",
                "applicability_status": "in_domain",
                "babappa_result_class": "negative",
                "max_gene_support": "0.1",
                "diagnostic_only": "False",
            },
            {
                "panel_id": "c",
                "applicability_status": "out_of_domain",
                "babappa_result_class": "positive",
                "max_gene_support": "0.8",
                "diagnostic_only": "True",
            },
        ],
        ["panel_id", "applicability_status", "babappa_result_class", "max_gene_support", "diagnostic_only"],
    )
    refs = tmp_path / "refs.tsv"
    write_tsv(
        refs,
        [
            {"panel_id": "a", "tool": "codeml", "test_name": "A", "p_value": "0.01", "q_value": "", "selected_branch": "", "selected_sites": "", "result_class": "positive", "notes": ""},
            {"panel_id": "b", "tool": "hyphy", "test_name": "B", "p_value": "0.01", "q_value": "", "selected_branch": "", "selected_sites": "", "result_class": "positive", "notes": ""},
            {"panel_id": "c", "tool": "codeml", "test_name": "C", "p_value": "0.9", "q_value": "", "selected_branch": "", "selected_sites": "", "result_class": "negative", "notes": ""},
        ],
        ["panel_id", "tool", "test_name", "p_value", "q_value", "selected_branch", "selected_sites", "result_class", "notes"],
    )
    result = compare_empirical_reference_results(
        EmpiricalReferenceComparisonConfig(str(run), str(refs), str(tmp_path / "comparison"))
    )
    payload = json.loads((tmp_path / "comparison" / "empirical_reference_comparison.json").read_text())
    assert result["status"] == "ok"
    assert payload["concordance_counts"]["concordant_positive"] == 1
    assert payload["concordance_counts"]["reference_only"] == 1
    assert payload["concordance_counts"]["BABAPPA_abstained"] == 1


def test_pilot_summary_includes_no_empirical_discovery_claim(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_json(run / "panel_run_manifest.json", {"status": "ok"})
    write_tsv(
        run / "panel_run_summary.tsv",
        [{"panel_id": "a", "input_status": "pass", "applicability_status": "in_domain", "scoring_status": "ok", "diagnostic_only": "False"}],
        ["panel_id", "input_status", "applicability_status", "scoring_status", "diagnostic_only"],
    )
    result = summarize_empirical_pilot_panel(
        EmpiricalPilotPanelSummaryConfig(panel_run=str(run), outdir=str(tmp_path / "summary"))
    )
    text = (tmp_path / "summary" / "empirical_pilot_panel_summary.md").read_text()
    assert result["status"] == "ok"
    assert "No empirical discovery claim" in text
    assert "simulation-trained" in text


def test_pilot_summary_validator_fails_if_claim_boundary_missing(tmp_path: Path) -> None:
    summary = tmp_path / "summary"
    summary.mkdir()
    _write_json(summary / "empirical_pilot_panel_summary.json", {"status": "ok"})
    (summary / "empirical_pilot_panel_summary.md").write_text("# Summary\n", encoding="utf-8")
    result = validate_empirical_pilot_summary(EmpiricalPilotSummaryValidationConfig(str(summary)))
    assert result["status"] == "fail"
    assert any(item.startswith("missing_claim_boundary_phrase") for item in result["failures"])


def test_real_pilot_template_exists() -> None:
    assert Path("examples/empirical_pilot_panel_template.tsv").exists()
    readme = Path("examples/EMPIRICAL_PILOT_PANEL_README.md")
    assert readme.exists()
    assert "not final empirical inference" in readme.read_text(encoding="utf-8")
