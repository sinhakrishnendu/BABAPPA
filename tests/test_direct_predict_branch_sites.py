import json
import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

import babappa.cli as cli
from babappa.cli import app
from babappa.datasets.index import read_tsv, write_tsv
from babappa.empirical.bridge import (
    DirectBranchSitePredictionConfig,
    _babappa_native_evidence_class,
    _babappa_native_result_class,
    _branch_shuffle_null_features,
    _right_tail_empirical_p_value,
    _update_direct_gene_summary_with_null,
    _write_direct_prediction_outputs,
    predict_branch_sites,
)


FEATURE_COLUMNS = [
    "site_index_zero",
    "aligned_site_index_zero",
    "original_site_index_zero",
    "site_relative_position",
    "n_taxa",
    "n_codons",
    "codon_id_mean",
    "codon_id_std",
    "codon_id_min",
    "codon_id_max",
    "codon_id_range",
    "codon_id_unique_count",
    "gap_fraction",
    "non_gap_fraction",
    "taxon_codon_variability",
    "foreground_codon_id",
    "foreground_gap",
    "branch_codon_id",
    "branch_gap",
    "background_mean_codon_id",
    "foreground_background_codon_delta",
    "branch_background_codon_delta",
]


def _tiny_inputs(tmp_path: Path) -> tuple[Path, Path]:
    fasta = tmp_path / "tiny.cds.fasta"
    tree = tmp_path / "tiny.treefile"
    fasta.write_text(
        ">taxon1\nATGGCTGCTGCTTAA\n>taxon2\nATGGCTGCCGCTTAA\n>taxon3\nATGGCTGCTGCCTAA\n",
        encoding="utf-8",
    )
    tree.write_text("(taxon1:0.1,(taxon2:0.1,taxon3:0.1):0.1);\n", encoding="utf-8")
    return fasta, tree


def _minimal_package(tmp_path: Path) -> Path:
    package = tmp_path / "package"
    package.mkdir()
    (package / "feature_schema.json").write_text(
        json.dumps(
            {
                "feature_policy": "conservative_branch_site",
                "expected_feature_columns": FEATURE_COLUMNS,
                "blocked_empirical_input_columns": ["branch_site_truth.tsv", "y_branch_site"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (package / "training_envelope.json").write_text("{}\n", encoding="utf-8")
    return package


def test_direct_prediction_dry_run_uses_user_msa_without_realignment(tmp_path: Path) -> None:
    fasta, tree = _tiny_inputs(tmp_path)
    package = _minimal_package(tmp_path)
    result = predict_branch_sites(
        DirectBranchSitePredictionConfig(
            msa=str(fasta),
            tree=str(tree),
            foreground="all",
            model_package=str(package),
            outdir=str(tmp_path / "prediction"),
            dry_run=True,
        )
    )
    manifest = json.loads((tmp_path / "prediction" / "prediction_manifest.json").read_text())
    alignment = json.loads((tmp_path / "prediction" / "user_msa" / "empirical_alignment_manifest.json").read_text())
    site_map = read_tsv(tmp_path / "prediction" / "user_msa" / "site_map" / "user_msa.site_map.tsv")
    features = read_tsv(tmp_path / "prediction" / "features" / "empirical_branch_site_features.tsv")

    assert result["status"] == "dry_run"
    assert alignment["methods_run"] == ["user_msa"]
    assert alignment["realignment_performed"] is False
    assert all(row["aligned_site_index_zero"] == row["original_site_index_zero"] for row in site_map)
    assert manifest["user_msa_is_authoritative"] is True
    assert manifest["realignment_performed"] is False
    assert {row["method"] for row in features} == {"user_msa"}
    assert {row["branch_id"] for row in features} == {"taxon1", "taxon2", "taxon3"}


def test_direct_prediction_foreground_list_limits_scored_branches(tmp_path: Path) -> None:
    fasta, tree = _tiny_inputs(tmp_path)
    package = _minimal_package(tmp_path)
    predict_branch_sites(
        DirectBranchSitePredictionConfig(
            msa=str(fasta),
            tree=str(tree),
            foreground="taxon1,taxon3",
            model_package=str(package),
            outdir=str(tmp_path / "prediction"),
            dry_run=True,
        )
    )
    features = read_tsv(tmp_path / "prediction" / "features" / "empirical_branch_site_features.tsv")
    assert {row["branch_id"] for row in features} == {"taxon1", "taxon3"}
    assert {row["foreground_taxon"] for row in features} == {"taxon1", "taxon3"}


def test_direct_prediction_leaves_alias_scores_all_tips(tmp_path: Path) -> None:
    fasta, tree = _tiny_inputs(tmp_path)
    package = _minimal_package(tmp_path)
    predict_branch_sites(
        DirectBranchSitePredictionConfig(
            msa=str(fasta),
            tree=str(tree),
            foreground="leaves",
            model_package=str(package),
            outdir=str(tmp_path / "prediction"),
            dry_run=True,
        )
    )
    features = read_tsv(tmp_path / "prediction" / "features" / "empirical_branch_site_features.tsv")
    assert {row["branch_id"] for row in features} == {"taxon1", "taxon2", "taxon3"}


def test_direct_prediction_warns_on_terminal_stop_codons(tmp_path: Path) -> None:
    fasta, tree = _tiny_inputs(tmp_path)
    package = _minimal_package(tmp_path)
    predict_branch_sites(
        DirectBranchSitePredictionConfig(
            msa=str(fasta),
            tree=str(tree),
            foreground="taxon1",
            model_package=str(package),
            outdir=str(tmp_path / "prediction"),
            dry_run=True,
        )
    )
    validation = json.loads((tmp_path / "prediction" / "input_validation" / "empirical_input_validation.json").read_text())
    assert validation["status"] == "warning"
    assert any(item.startswith("terminal_stop_codon") for item in validation["warnings"])
    assert not any("internal_stop_codon" in item for item in validation["failures"])


def test_direct_prediction_rejects_internal_stop_codons(tmp_path: Path) -> None:
    fasta = tmp_path / "bad_internal_stop.fasta"
    tree = tmp_path / "bad_internal_stop.tree"
    fasta.write_text(
        ">taxon1\nATGTAAGCTGCT\n>taxon2\nATGGCTGCTGCT\n>taxon3\nATGGCTGCTGCC\n",
        encoding="utf-8",
    )
    tree.write_text("(taxon1:0.1,(taxon2:0.1,taxon3:0.1):0.1);\n", encoding="utf-8")
    package = _minimal_package(tmp_path)

    with pytest.raises(ValueError, match="internal_stop_codon"):
        predict_branch_sites(
            DirectBranchSitePredictionConfig(
                msa=str(fasta),
                tree=str(tree),
                foreground="taxon1",
                model_package=str(package),
                outdir=str(tmp_path / "prediction"),
                dry_run=True,
            )
        )


def test_direct_prediction_rejects_missing_start_codon(tmp_path: Path) -> None:
    fasta = tmp_path / "bad_start.fasta"
    tree = tmp_path / "bad_start.tree"
    fasta.write_text(
        ">taxon1\nGCTGCTGCTTAA\n>taxon2\nATGGCTGCTTAA\n>taxon3\nATGGCTGCCTAA\n",
        encoding="utf-8",
    )
    tree.write_text("(taxon1:0.1,(taxon2:0.1,taxon3:0.1):0.1);\n", encoding="utf-8")
    package = _minimal_package(tmp_path)

    with pytest.raises(ValueError, match="missing_start_codon"):
        predict_branch_sites(
            DirectBranchSitePredictionConfig(
                msa=str(fasta),
                tree=str(tree),
                foreground="taxon1",
                model_package=str(package),
                outdir=str(tmp_path / "prediction"),
                dry_run=True,
            )
        )


def test_direct_prediction_rejects_unequal_length_msa(tmp_path: Path) -> None:
    fasta = tmp_path / "bad.fasta"
    tree = tmp_path / "bad.tree"
    fasta.write_text(">a\nATGGCT\n>b\nATGGCTGCT\n>c\nATGGCT\n", encoding="utf-8")
    tree.write_text("(a:0.1,(b:0.1,c:0.1):0.1);\n", encoding="utf-8")
    package = _minimal_package(tmp_path)

    with pytest.raises(ValueError, match="requires an aligned codon MSA"):
        predict_branch_sites(
            DirectBranchSitePredictionConfig(
                msa=str(fasta),
                tree=str(tree),
                model_package=str(package),
                outdir=str(tmp_path / "prediction"),
                dry_run=True,
            )
        )


def test_predict_branch_sites_cli_help_includes_direct_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["predict-branch-sites", "--help"])
    assert result.exit_code == 0
    assert "--msa" in result.output
    assert "--tree" in result.output
    assert "--foreground" in result.output
    assert "--null-replicates" in result.output


def test_interactive_default_launches_direct_prediction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}

    def fake_predict(config: DirectBranchSitePredictionConfig) -> dict:
        captured["config"] = config
        return {
            "status": "ok",
            "outdir": config.outdir,
            "foreground": config.foreground,
            "n_foregrounds": 3,
            "n_taxa": 3,
            "n_codons": 5,
            "applicability": "in_domain",
            "device": "cpu",
            "branch_site_predictions": str(Path(config.outdir) / "branch_site_predictions.tsv"),
            "report": str(Path(config.outdir) / "prediction_report.md"),
        }

    monkeypatch.setattr(cli, "predict_branch_sites", fake_predict)
    runner = CliRunner()
    result = runner.invoke(app, [], input=f"{tmp_path / 'x.fasta'}\n{tmp_path / 'x.tree'}\n\n")

    assert result.exit_code == 0
    assert captured["config"].foreground == "all"
    assert captured["config"].null_replicates == 100
    assert captured["config"].outdir.endswith("babappa_prediction_x")
    assert "BABAPPA direct branch-site predictor" in result.output


def test_direct_prediction_outputs_include_degapped_branch_site_number(tmp_path: Path) -> None:
    outdir = tmp_path / "prediction"
    scores_dir = outdir / "scores"
    applicability_dir = outdir / "applicability"
    msa_dir = outdir / "user_msa" / "methods" / "user_msa"
    scores_dir.mkdir(parents=True)
    applicability_dir.mkdir(parents=True)
    msa_dir.mkdir(parents=True)
    (msa_dir / "empirical.user_msa.codon.fasta").write_text(
        ">taxon1\nATG---GCT\n>taxon2\nATGGCTGCT\n",
        encoding="utf-8",
    )
    write_tsv(
        scores_dir / "empirical_branch_site_scores.tsv",
        [
            {
                "family_id": "empirical",
                "method": "user_msa",
                "branch_id": "taxon1",
                "foreground_taxon": "taxon1",
                "aligned_site_index_zero": "1",
                "original_site_index_zero": "1",
                "prob_positive": "0.9",
                "called_positive": "1",
                "tier_model": "moderate",
                "calibrated_threshold": "0.5",
                "diagnostic_only": "False",
            },
            {
                "family_id": "empirical",
                "method": "user_msa",
                "branch_id": "taxon2",
                "foreground_taxon": "taxon2",
                "aligned_site_index_zero": "1",
                "original_site_index_zero": "1",
                "prob_positive": "0.9",
                "called_positive": "1",
                "tier_model": "moderate",
                "calibrated_threshold": "0.5",
                "diagnostic_only": "False",
            },
        ],
        [
            "family_id",
            "method",
            "branch_id",
            "foreground_taxon",
            "aligned_site_index_zero",
            "original_site_index_zero",
            "prob_positive",
            "called_positive",
            "tier_model",
            "calibrated_threshold",
            "diagnostic_only",
        ],
    )
    write_tsv(scores_dir / "empirical_branch_scores.tsv", [], ["family_id", "method", "branch_id"])
    write_tsv(scores_dir / "empirical_gene_support.tsv", [], ["family_id", "method"])
    (applicability_dir / "empirical_applicability.json").write_text(
        json.dumps({"applicability_status": "in_domain"}) + "\n",
        encoding="utf-8",
    )

    _write_direct_prediction_outputs(outdir, scores_dir, applicability_dir)
    rows = read_tsv(outdir / "branch_site_predictions.tsv")
    by_branch = {row["branch_id"]: row for row in rows}

    assert by_branch["taxon1"]["msa_codon_site"] == "2"
    assert by_branch["taxon1"]["branch_degapped_codon_site"] == ""
    assert by_branch["taxon1"]["branch_codon"] == "---"
    assert by_branch["taxon2"]["branch_degapped_codon_site"] == "2"


def test_babappa_native_null_helpers_compute_standalone_evidence(tmp_path: Path) -> None:
    p_value = _right_tail_empirical_p_value(10.0, [1.0, 2.0, 10.0, 12.0])
    assert p_value == pytest.approx(3 / 5)
    assert _babappa_native_evidence_class({"p_babappa_called_rows": 0.009}, 100) == "strong_babappa_native_support"
    assert _babappa_native_evidence_class({"p_babappa_called_rows": 0.03}, 100) == "babappa_native_support"
    assert _babappa_native_evidence_class({"p_babappa_called_rows": 0.2}, 100) == "not_significant_under_babappa_native_null"
    assert _babappa_native_evidence_class({"p_babappa_called_rows": 0.001}, 3) == "underpowered_native_null"
    assert _babappa_native_result_class("diagnostic_positive", "babappa_native_support") == "babappa_native_calibrated_support"
    assert (
        _babappa_native_result_class("diagnostic_positive", "not_significant_under_babappa_native_null")
        == "diagnostic_positive_not_supported_by_babappa_native_null"
    )
    assert _babappa_native_result_class("diagnostic_negative", "babappa_native_support") == "babappa_native_negative"

    rows = [
        {"branch_id": "a", "branch_codon_id": "1", "foreground_codon_id": "2", "site_index_zero": "0"},
        {"branch_id": "b", "branch_codon_id": "3", "foreground_codon_id": "4", "site_index_zero": "1"},
    ]
    shuffled = _branch_shuffle_null_features(rows, random.Random(7))
    assert shuffled is not rows
    assert {row["branch_id"] for row in shuffled} == {"a", "b"}
    assert sorted(row["branch_codon_id"] for row in shuffled) == ["1", "3"]

    outdir = tmp_path / "prediction"
    outdir.mkdir()
    write_tsv(
        outdir / "gene_summary.tsv",
        [{"family_id": "empirical", "result_class": "diagnostic_positive"}],
        ["family_id", "result_class"],
    )
    _update_direct_gene_summary_with_null(
        outdir,
        {
            "n_replicates_completed": 100,
            "evidence_class": "babappa_native_support",
            "p_values": {
                "p_babappa_max_gene_support": 0.04,
                "p_babappa_called_rows": 0.02,
                "p_babappa_max_branch_support": 0.05,
                "p_babappa_max_site_score": 0.1,
            },
        },
    )
    updated = read_tsv(outdir / "gene_summary.tsv")[0]
    assert updated["babappa_native_evidence_class"] == "babappa_native_support"
    assert updated["babappa_native_result_class"] == "babappa_native_calibrated_support"
    assert updated["p_babappa_called_rows"] == "0.02"
