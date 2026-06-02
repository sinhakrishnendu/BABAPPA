import json
from pathlib import Path

import pytest
import numpy as np
from typer.testing import CliRunner

from babappa.benchmarks import SaturationPanelConfig, build_saturation_panel
from babappa.branch import (
    BranchAggregationConfig,
    BranchAggregationControlConfig,
    BranchConditioned10kPlanConfig,
    BranchSiteBaselineConfig,
    BranchSiteDatasetConfig,
    BranchSiteNeuralTrainConfig,
    BranchSiteOracleLabelConfig,
    aggregate_branch_sites,
    audit_branch_site_leakage,
    build_branch_site_dataset,
    extract_branch_site_labels,
    plan_branch_conditioned_10k,
    run_branch_aggregation_controls,
    train_branch_site_baseline,
    train_branch_site_neural_model,
    validate_branch_aggregation_controls_dir,
    validate_branch_aggregation_dir,
    validate_branch_site_baseline_dir,
    validate_branch_site_dataset_dir,
    validate_branch_site_label_dir,
    validate_branch_site_leakage_dir,
    validate_branch_site_neural_dir,
)
from babappa.cli import app
from babappa.datasets import read_tsv
from babappa.training import safe_import_torch


torch, _torch_error = safe_import_torch()
runner = CliRunner()


def test_branch_label_extraction_tiny_truth(tmp_path) -> None:
    dataset_dir = _build_tiny_gene_dataset(tmp_path)
    outdir = tmp_path / "branch_labels"

    summary = extract_branch_site_labels(
        BranchSiteOracleLabelConfig(dataset_dir=str(dataset_dir), outdir=str(outdir))
    )
    rows = read_tsv(outdir / "branch_site_oracle_labels.tsv")
    payload = json.loads((outdir / "branch_site_oracle_summary.json").read_text("utf-8"))

    assert summary["status"] == "ok"
    assert payload["branch_site_labels_status"] == "explicit_simulator_branch_truth"
    assert payload["explicit_branch_site_truth_available"] is True
    assert payload["proxy_labels_used"] is False
    assert rows
    assert {"branch_id", "y_branch_site", "y_site", "gene_label"} <= set(rows[0])
    assert any(row["y_branch_site"] == "1" for row in rows)
    assert validate_branch_site_label_dir(outdir)["status"] == "ok"


def test_branch_dataset_excludes_leakage_columns(tmp_path) -> None:
    dataset_dir, branch_dataset = _build_tiny_branch_dataset(tmp_path)
    rows = read_tsv(branch_dataset / "branch_site_features.tsv")
    index = json.loads((branch_dataset / "branch_site_dataset_index.json").read_text("utf-8"))

    assert dataset_dir.exists()
    assert validate_branch_site_dataset_dir(branch_dataset)["status"] == "ok"
    assert rows
    assert "y_site" in rows[0]
    assert "gene_label" in rows[0]
    assert "selected_sites" not in rows[0]
    assert "y_branch_site" not in index["feature_columns"]
    assert "y_site" not in index["feature_columns"]
    assert "gene_label" not in index["feature_columns"]


def test_branch_leakage_audit_flags_forbidden_feature_columns(tmp_path) -> None:
    dataset = tmp_path / "branch_dataset_bad"
    dataset.mkdir()
    (dataset / "branch_site_dataset_index.json").write_text(
        json.dumps({"feature_columns": ["codon_id_mean", "y_site", "gene_label"]}) + "\n",
        encoding="utf-8",
    )
    (dataset / "branch_site_features.tsv").write_text(
        "branch_site_id\tfamily_id\tmethod\tsplit\tsaturation_tier\tbranch_id\tsite_index_zero\ty_branch_site\ty_site\tgene_label\tcodon_id_mean\n"
        "b1\tf1\tidentity\ttrain\tlow\ttaxon_001\t0\t1\t1\t1\t0.8\n"
        "b2\tf1\tidentity\ttrain\tlow\ttaxon_002\t0\t0\t1\t1\t0.2\n",
        encoding="utf-8",
    )

    summary = audit_branch_site_leakage(dataset, tmp_path / "leakage")
    payload = json.loads((tmp_path / "leakage" / "branch_site_leakage_audit.json").read_text("utf-8"))

    assert summary["status"] == "ok"
    assert payload["status"] == "warning"
    assert "y_site" in payload["forbidden_columns_present"]
    assert "gene_label" in payload["forbidden_columns_present"]
    assert validate_branch_site_leakage_dir(tmp_path / "leakage")["status"] == "ok"


def test_branch_baseline_trains_on_tiny_dataset(tmp_path) -> None:
    _dataset_dir, branch_dataset = _build_tiny_branch_dataset(tmp_path)
    outdir = tmp_path / "branch_baseline"

    summary = train_branch_site_baseline(
        BranchSiteBaselineConfig(
            branch_site_dataset_dir=str(branch_dataset),
            outdir=str(outdir),
            epochs=20,
            learning_rate=0.05,
        )
    )

    assert summary["status"] == "ok"
    assert (outdir / "branch_site_baseline_predictions.tsv").exists()
    assert validate_branch_site_baseline_dir(outdir)["status"] == "ok"


def test_branch_aggregation_produces_branch_and_gene_outputs(tmp_path) -> None:
    predictions = _branch_baseline_predictions(tmp_path)
    outdir = tmp_path / "branch_aggregation"

    summary = aggregate_branch_sites(
        BranchAggregationConfig(predictions_tsv=str(predictions), outdir=str(outdir))
    )

    assert summary["status"] == "ok"
    assert (outdir / "branch_site_to_branch_predictions.tsv").exists()
    assert (outdir / "branch_to_gene_predictions.tsv").exists()
    assert validate_branch_aggregation_dir(outdir)["status"] == "ok"


def test_branch_controls_produce_null_auroc_distribution(tmp_path) -> None:
    predictions = _branch_baseline_predictions(tmp_path)
    outdir = tmp_path / "branch_controls"

    summary = run_branch_aggregation_controls(
        BranchAggregationControlConfig(
            predictions_tsv=str(predictions),
            outdir=str(outdir),
            n_permutations=3,
            seed=7,
            workers=2,
        )
    )
    payload = json.loads((outdir / "branch_aggregation_controls.json").read_text("utf-8"))
    rows = read_tsv(outdir / "branch_aggregation_controls.tsv")

    assert summary["status"] == "ok"
    assert summary["requested_workers"] == 2
    assert payload["requested_workers"] == 2
    assert payload["effective_workers"] >= 1
    assert rows
    assert {row["control"] for row in rows}
    assert {
        "within_family_branch_label_shuffle",
        "within_family_site_label_shuffle",
        "branch_score_permutation_within_family",
        "family_label_preserving_random_scores",
        "degree_prevalence_matched_null",
    } <= {row["control"] for row in rows}
    assert "control_interpretation" in rows[0]
    assert "expected_behavior" in rows[0]
    assert "whether_control_is_destructive_enough" in rows[0]
    assert validate_branch_aggregation_controls_dir(outdir)["status"] == "ok"


def test_branch_conditioned_planner_generates_user_run_scripts(tmp_path) -> None:
    outdir = tmp_path / "branch_plan"

    summary = plan_branch_conditioned_10k(
        BranchConditioned10kPlanConfig(outdir=str(outdir), tiers=["low", "extreme"])
    )
    expected = json.loads((outdir / "expected_outputs.json").read_text("utf-8"))
    run_text = (outdir / "run_branch_conditioned_10k.sh").read_text("utf-8")

    assert summary["status"] == "ok"
    for filename in [
        "run_branch_conditioned_10k.sh",
        "monitor_branch_conditioned_10k.sh",
        "validate_branch_conditioned_10k.sh",
        "summarize_branch_conditioned_10k.sh",
        "expected_outputs.json",
        "branch_conditioned_10k_plan.md",
    ]:
        assert (outdir / filename).exists()
    assert "MANUAL EXECUTION SCRIPT" in run_text
    assert "align-external" not in run_text
    assert "train-site-neural" not in run_text
    assert "identity" in json.dumps(expected)
    assert "prank" in json.dumps(expected)


def test_build_branch_site_dataset_streaming_large_synthetic(tmp_path) -> None:
    dataset_dir, labels_path, n_rows, n_positive = _build_synthetic_branch_labels(tmp_path, n_rows=2000, n_positive=20)
    outdir = tmp_path / "branch_dataset_streaming"

    summary = build_branch_site_dataset(
        BranchSiteDatasetConfig(
            dataset_dir=str(dataset_dir),
            branch_site_labels_tsv=str(labels_path),
            outdir=str(outdir),
            negative_downsample_ratio=5,
            seed=123,
        )
    )
    rows = read_tsv(outdir / "branch_site_features.tsv")
    index = json.loads((outdir / "branch_site_dataset_index.json").read_text("utf-8"))

    assert summary["status"] == "ok"
    assert index["streaming"] is True
    assert index["total_input_rows"] == n_rows
    assert int(index["n_positive_branch_sites"]) == n_positive
    assert len(rows) < n_rows
    assert sum(1 for row in rows if row["y_branch_site"] == "1") == n_positive


def test_branch_dataset_caps(tmp_path) -> None:
    dataset_dir, labels_path, _n_rows, n_positive = _build_synthetic_branch_labels(tmp_path, n_rows=1200, n_positive=20)
    outdir = tmp_path / "branch_dataset_capped"

    build_branch_site_dataset(
        BranchSiteDatasetConfig(
            dataset_dir=str(dataset_dir),
            branch_site_labels_tsv=str(labels_path),
            outdir=str(outdir),
            negative_downsample_ratio=50,
            seed=7,
            max_output_rows=50,
        )
    )
    rows = read_tsv(outdir / "branch_site_features.tsv")
    index = json.loads((outdir / "branch_site_dataset_index.json").read_text("utf-8"))

    assert len(rows) <= 50
    assert index["rows_written"] <= 50
    assert int(index["n_positive_branch_sites"]) == n_positive


def test_branch_plan_uses_streaming_caps(tmp_path) -> None:
    outdir = tmp_path / "branch_plan_streamed"

    plan_branch_conditioned_10k(BranchConditioned10kPlanConfig(outdir=str(outdir), tiers=["low"]))
    run_text = (outdir / "run_branch_conditioned_10k.sh").read_text("utf-8")
    expected = json.loads((outdir / "expected_outputs.json").read_text("utf-8"))

    assert "--streaming --max-output-rows 1000000" in run_text
    assert "--negative-downsample-ratio 5" in run_text
    assert "branch_site_dataset_fast_external_10k_low_streamed" in run_text
    assert expected["max_output_rows_per_tier"] == 1_000_000
    assert expected["output_suffix"] == "streamed"


def test_streaming_branch_dataset_summary_does_not_return_all_rows(tmp_path) -> None:
    dataset_dir, labels_path, n_rows, _n_positive = _build_synthetic_branch_labels(tmp_path, n_rows=800, n_positive=8)
    outdir = tmp_path / "branch_dataset_no_row_return"

    summary = build_branch_site_dataset(
        BranchSiteDatasetConfig(
            dataset_dir=str(dataset_dir),
            branch_site_labels_tsv=str(labels_path),
            outdir=str(outdir),
            negative_downsample_ratio=3,
            seed=42,
        )
    )
    index = json.loads((outdir / "branch_site_dataset_index.json").read_text("utf-8"))

    assert "rows" not in summary
    assert summary["total_input_rows"] == n_rows
    assert index["streaming"] is True
    assert index["rows_written"] < index["total_input_rows"]


def test_branch_cli_smoke_extract_and_plan(tmp_path) -> None:
    dataset_dir = _build_tiny_gene_dataset(tmp_path)
    result = runner.invoke(
        app,
        [
            "extract-branch-site-labels",
            "--dataset-dir",
            str(dataset_dir),
            "--outdir",
            str(tmp_path / "branch_labels_cli"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Branch-Site Oracle Labels" in result.output

    plan = runner.invoke(app, ["plan-branch-conditioned-10k", "--outdir", str(tmp_path / "branch_plan_cli")])
    assert plan.exit_code == 0, plan.output
    assert "Branch-Conditioned 10K Plan" in plan.output


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_branch_neural_tiny_if_torch_available(tmp_path) -> None:
    _dataset_dir, branch_dataset = _build_tiny_branch_dataset(tmp_path)
    outdir = tmp_path / "branch_neural"

    summary = train_branch_site_neural_model(
        BranchSiteNeuralTrainConfig(
            branch_site_dataset_dir=str(branch_dataset),
            outdir=str(outdir),
            device="cpu",
            epochs=2,
            batch_size=8,
            hidden_dim=8,
            threads=1,
            max_train_items=64,
            max_val_items=32,
            max_calib_items=32,
            max_test_items=32,
        )
    )

    assert summary["status"] == "ok"
    assert summary["threads"] == 1
    assert (outdir / "branch_site_neural_predictions.tsv").exists()
    assert validate_branch_site_neural_dir(outdir)["status"] == "ok"


def _build_tiny_gene_dataset(tmp_path: Path) -> Path:
    panel_dir = tmp_path / "panel"
    build_saturation_panel(
        SaturationPanelConfig(
            outdir=str(panel_dir),
            n_families_per_tier=12,
            tiers=["low"],
            n_taxa=4,
            n_codons=30,
            seed=42,
            positive_rate=1.0,
            selected_site_fraction=0.1,
            methods=["identity"],
            build_tensors=True,
            index_datasets=True,
        )
    )
    return panel_dir / "tiers" / "low" / "dataset"


def _build_tiny_branch_dataset(tmp_path: Path) -> tuple[Path, Path]:
    dataset_dir = _build_tiny_gene_dataset(tmp_path)
    label_dir = tmp_path / "branch_labels"
    extract_branch_site_labels(
        BranchSiteOracleLabelConfig(dataset_dir=str(dataset_dir), outdir=str(label_dir))
    )
    branch_dataset = tmp_path / "branch_dataset"
    build_branch_site_dataset(
        BranchSiteDatasetConfig(
            dataset_dir=str(dataset_dir),
            branch_site_labels_tsv=str(label_dir / "branch_site_oracle_labels.tsv"),
            outdir=str(branch_dataset),
            negative_downsample_ratio=10,
            seed=42,
        )
    )
    return dataset_dir, branch_dataset


def _branch_baseline_predictions(tmp_path: Path) -> Path:
    _dataset_dir, branch_dataset = _build_tiny_branch_dataset(tmp_path)
    outdir = tmp_path / "branch_baseline"
    train_branch_site_baseline(
        BranchSiteBaselineConfig(
            branch_site_dataset_dir=str(branch_dataset),
            outdir=str(outdir),
            epochs=20,
            learning_rate=0.05,
        )
    )
    return outdir / "branch_site_baseline_predictions.tsv"


def _build_synthetic_branch_labels(tmp_path: Path, n_rows: int, n_positive: int) -> tuple[Path, Path, int, int]:
    dataset_dir = tmp_path / "synthetic_dataset"
    dataset_dir.mkdir()
    tensor_path = dataset_dir / "synthetic.tensor.npz"
    taxa = np.array(["taxon_001", "taxon_002", "taxon_003"])
    X = np.zeros((3, n_rows, 2), dtype=np.float32)
    for taxon_index in range(3):
        X[taxon_index, :, 0] = (np.arange(n_rows) + taxon_index) % 61
    np.savez_compressed(tensor_path, X=X, taxa_order=taxa)
    labels_path = tmp_path / "synthetic_branch_labels.tsv"
    header = [
        "family_id",
        "method",
        "split",
        "saturation_tier",
        "branch_id",
        "foreground_taxon",
        "site_index_zero",
        "aligned_site_index_zero",
        "original_site_index_zero",
        "y_branch_site",
        "y_site",
        "gene_label",
        "foreground_branch_present",
        "branch_label_source",
        "mapping_status",
        "mapping_confidence",
        "mappable_site",
        "original_family_id",
        "source_dataset",
        "tensor_file",
        "labels_file",
        "n_taxa",
        "n_codons",
    ]
    positive_indices = set(range(n_positive))
    with labels_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for index in range(n_rows):
            split = "train" if index < int(n_rows * 0.6) else "val"
            branch_id = taxa[index % len(taxa)]
            is_positive = 1 if index in positive_indices else 0
            foreground = 1 if branch_id == "taxon_001" else 0
            row = [
                f"family_{index // 100:06d}",
                "identity",
                split,
                "low",
                str(branch_id),
                "taxon_001",
                str(index),
                str(index),
                str(index),
                str(is_positive),
                str(is_positive),
                "1",
                str(foreground),
                "proxy_from_foreground_taxon:test",
                "unique",
                "1.0",
                "1",
                "",
                "",
                str(tensor_path),
                "",
                "3",
                str(n_rows),
            ]
            handle.write("\t".join(row) + "\n")
    return dataset_dir, labels_path, n_rows, n_positive
