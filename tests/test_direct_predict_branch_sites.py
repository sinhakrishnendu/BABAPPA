import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import babappa.cli as cli
from babappa.cli import app
from babappa.datasets.index import read_tsv, write_tsv
from babappa.empirical.bridge import (
    DirectBranchSitePredictionConfig,
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
