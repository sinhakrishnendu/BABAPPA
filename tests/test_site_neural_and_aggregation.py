import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from babappa.cli import app
from babappa.datasets import write_tsv
from babappa.site import (
    SiteAggregationConfig,
    SiteCalibrationConfig,
    SiteNeuralDatasetConfig,
    SiteNeuralTrainConfig,
    SiteStratifiedEvalConfig,
    SiteThresholdPolicyConfig,
    aggregate_site_predictions,
    build_site_threshold_policy,
    calibrate_site_model,
    load_site_feature_arrays,
    site_stratified_evaluate,
    train_site_neural_model,
    validate_site_aggregation_dir,
    validate_site_calibration_dir,
    validate_site_neural_dir,
    validate_site_stratified_eval_dir,
    validate_site_threshold_policy_dir,
)
from babappa.site.neural_model import SiteMLPClassifier
from babappa.training import safe_import_torch


torch, _torch_error = safe_import_torch()
runner = CliRunner()


def test_repository_hardening_init_files() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in [
        "src/babappa/init.py",
        "src/babappa/calibration/init.py",
        "src/babappa/reports/init.py",
        "src/babappa/training/init.py",
        "src/babappa/benchmarks/init.py",
        "src/babappa/datasets/init.py",
        "src/babappa/site/init.py",
    ]:
        assert not (root / relative).exists()
    for relative in [
        "src/babappa/__init__.py",
        "src/babappa/calibration/__init__.py",
        "src/babappa/reports/__init__.py",
        "src/babappa/training/__init__.py",
        "src/babappa/benchmarks/__init__.py",
        "src/babappa/datasets/__init__.py",
        "src/babappa/site/__init__.py",
    ]:
        assert (root / relative).exists()


def test_site_neural_dataset_loading(tmp_path) -> None:
    dataset_dir = _synthetic_site_dataset(tmp_path / "site_dataset")

    X, y, metadata, columns = load_site_feature_arrays(
        SiteNeuralDatasetConfig(site_dataset_dir=str(dataset_dir), split="train")
    )

    assert X.shape[0] == y.shape[0] == len(metadata)
    assert X.shape[1] == len(columns)
    assert "codon_id_mean" in columns
    assert "y_site" not in columns
    assert all("positive" not in column for column in columns)


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_site_mlp_forward_if_torch_available() -> None:
    model = SiteMLPClassifier(input_dim=4, hidden_dim=8, dropout=0.0)
    x = torch.randn(5, 4)

    logits = model(x)

    assert list(logits.shape) == [5]


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_train_site_neural_tiny_if_torch_available(tmp_path) -> None:
    dataset_dir = _synthetic_site_dataset(tmp_path / "site_dataset")
    outdir = tmp_path / "site_neural"

    summary = train_site_neural_model(
        SiteNeuralTrainConfig(
            site_dataset_dir=str(dataset_dir),
            outdir=str(outdir),
            device="cpu",
            epochs=2,
            batch_size=8,
            hidden_dim=8,
            max_train_items=32,
            max_val_items=16,
            max_calib_items=16,
            max_test_items=16,
        )
    )

    assert summary["status"] == "ok"
    assert (outdir / "site_neural_predictions.tsv").exists()
    assert (outdir / "site_neural_metrics.json").exists()
    assert validate_site_neural_dir(outdir)["status"] == "ok"


def test_site_calibration_on_synthetic_predictions(tmp_path) -> None:
    model_dir = _synthetic_site_prediction_model(tmp_path / "model")
    outdir = tmp_path / "calibration"

    summary = calibrate_site_model(
        SiteCalibrationConfig(model_dir=str(model_dir), outdir=str(outdir), target_fdr=0.25)
    )

    assert summary["status"] == "ok"
    assert validate_site_calibration_dir(outdir)["status"] == "ok"


def test_site_threshold_policy_on_synthetic_predictions(tmp_path) -> None:
    model_dir = _synthetic_site_prediction_model(tmp_path / "model")
    outdir = tmp_path / "policy"

    summary = build_site_threshold_policy(
        SiteThresholdPolicyConfig(
            predictions_tsv=str(model_dir / "site_neural_predictions.tsv"),
            outdir=str(outdir),
            target_fdr=0.25,
        )
    )

    payload = json.loads((outdir / "site_threshold_profiles.json").read_text("utf-8"))
    assert summary["status"] == "ok"
    assert "max_f1" in payload["profiles"]
    assert validate_site_threshold_policy_dir(outdir)["status"] == "ok"


def test_site_stratified_eval_on_synthetic_predictions(tmp_path) -> None:
    model_dir = _synthetic_site_prediction_model(tmp_path / "model")
    policy_dir = tmp_path / "policy"
    build_site_threshold_policy(
        SiteThresholdPolicyConfig(
            predictions_tsv=str(model_dir / "site_neural_predictions.tsv"),
            outdir=str(policy_dir),
        )
    )
    outdir = tmp_path / "eval"

    summary = site_stratified_evaluate(
        SiteStratifiedEvalConfig(
            predictions_tsv=str(model_dir / "site_neural_predictions.tsv"),
            outdir=str(outdir),
            threshold_policy_dir=str(policy_dir),
        )
    )

    assert summary["status"] == "ok"
    assert validate_site_stratified_eval_dir(outdir)["status"] == "ok"


def test_site_to_gene_aggregation_synthetic(tmp_path) -> None:
    model_dir = _synthetic_site_prediction_model(tmp_path / "model")
    gene_dir = _synthetic_gene_dataset(tmp_path / "gene_dataset")
    outdir = tmp_path / "aggregation"

    summary = aggregate_site_predictions(
        SiteAggregationConfig(
            predictions_tsv=str(model_dir / "site_neural_predictions.tsv"),
            gene_dataset_dir=str(gene_dir),
            outdir=str(outdir),
        )
    )
    payload = json.loads((outdir / "site_to_gene_metrics.json").read_text("utf-8"))

    assert summary["status"] == "ok"
    assert "max_site_probability" in payload["gene_level_metrics_by_score"]
    assert validate_site_aggregation_dir(outdir)["status"] == "ok"


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_cli_train_site_neural_if_torch_available(tmp_path) -> None:
    dataset_dir = _synthetic_site_dataset(tmp_path / "site_dataset")
    result = runner.invoke(
        app,
        [
            "train-site-neural",
            "--site-dataset-dir",
            str(dataset_dir),
            "--outdir",
            str(tmp_path / "site_neural_cli"),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--hidden-dim",
            "8",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Site Neural" in result.output


def test_cli_site_threshold_policy_exits_0(tmp_path) -> None:
    model_dir = _synthetic_site_prediction_model(tmp_path / "model")
    result = runner.invoke(
        app,
        [
            "site-threshold-policy",
            "--predictions",
            str(model_dir / "site_neural_predictions.tsv"),
            "--outdir",
            str(tmp_path / "policy_cli"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Site Threshold Policy" in result.output


def test_cli_aggregate_sites_exits_0(tmp_path) -> None:
    model_dir = _synthetic_site_prediction_model(tmp_path / "model")
    gene_dir = _synthetic_gene_dataset(tmp_path / "gene_dataset")
    result = runner.invoke(
        app,
        [
            "aggregate-sites",
            "--predictions",
            str(model_dir / "site_neural_predictions.tsv"),
            "--gene-dataset-dir",
            str(gene_dir),
            "--outdir",
            str(tmp_path / "aggregation_cli"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Site-to-Gene Aggregation" in result.output


def _synthetic_site_dataset(path: Path) -> Path:
    path.mkdir(parents=True)
    rows = []
    splits = ["train", "val", "calib", "test"]
    for split_index, split in enumerate(splits):
        for index in range(24):
            y = 1 if index % 6 == 0 else 0
            family_id = f"family_{split}_{index // 6:02d}"
            method = "identity" if index % 2 == 0 else "codon_dropout"
            tier = "low" if index % 3 else "high"
            site_index = index
            signal = 0.8 if y else 0.2
            rows.append(
                {
                    "site_id": f"{family_id}::{method}::site_{site_index:06d}",
                    "family_id": family_id,
                    "original_family_id": family_id,
                    "source_dataset": "synthetic",
                    "method": method,
                    "saturation_tier": tier,
                    "split": split,
                    "tensor_file": "synthetic.npz",
                    "labels_file": "synthetic.labels.json",
                    "site_index_zero": site_index,
                    "site_index_one": site_index + 1,
                    "y_site": y,
                    "site_relative_position": site_index / 100,
                    "n_taxa": 4,
                    "n_codons": 100,
                    "codon_id_mean": signal + split_index * 0.01,
                    "codon_id_std": signal,
                    "codon_id_min": 1,
                    "codon_id_max": 50,
                    "codon_id_range": 49,
                    "codon_id_unique_count": 3 + y,
                    "gap_fraction": 0.0,
                    "non_gap_fraction": 1.0,
                    "taxon_codon_variability": signal,
                    "foreground_taxon_present": 0,
                    "foreground_taxon_index": -1,
                    "foreground_codon_id": -1,
                    "background_codon_id_mean": signal,
                    "foreground_background_abs_delta": 0,
                    "foreground_gap": 0,
                }
            )
    fieldnames = list(rows[0].keys())
    write_tsv(path / "site_features.tsv", rows, fieldnames)
    write_tsv(
        path / "site_splits.tsv",
        [
            {
                "site_id": row["site_id"],
                "family_id": row["family_id"],
                "method": row["method"],
                "saturation_tier": row["saturation_tier"],
                "split": row["split"],
                "site_index_zero": row["site_index_zero"],
                "y_site": row["y_site"],
            }
            for row in rows
        ],
        ["site_id", "family_id", "method", "saturation_tier", "split", "site_index_zero", "y_site"],
    )
    (path / "site_dataset_index.json").write_text(
        json.dumps({"n_site_rows": len(rows), "n_positive_sites": sum(row["y_site"] for row in rows)}) + "\n",
        encoding="utf-8",
    )
    return path


def _synthetic_site_prediction_model(path: Path) -> Path:
    path.mkdir(parents=True)
    rows = []
    for split in ["train", "val", "calib", "test"]:
        for family_index in range(4):
            gene_positive = family_index % 2 == 0
            family_id = f"family_{split}_{family_index}"
            for site_index in range(5):
                y = 1 if gene_positive and site_index == 0 else 0
                prob = 0.9 if y else 0.1 + 0.05 * site_index
                rows.append(
                    {
                        "site_id": f"{family_id}::identity::site_{site_index:06d}",
                        "family_id": family_id,
                        "method": "identity",
                        "saturation_tier": "low" if family_index < 2 else "high",
                        "split": split,
                        "site_index_zero": site_index,
                        "y_site": y,
                        "prob_positive": prob,
                        "pred_label": 1 if prob >= 0.5 else 0,
                        "correct": int((prob >= 0.5) == bool(y)),
                    }
                )
    write_tsv(
        path / "site_neural_predictions.tsv",
        rows,
        ["site_id", "family_id", "method", "saturation_tier", "split", "site_index_zero", "y_site", "prob_positive", "pred_label", "correct"],
    )
    return path


def _synthetic_gene_dataset(path: Path) -> Path:
    path.mkdir(parents=True)
    rows = []
    for split in ["train", "val", "calib", "test"]:
        for family_index in range(4):
            rows.append(
                {
                    "family_id": f"family_{split}_{family_index}",
                    "method": "identity",
                    "tensor_file": "synthetic.npz",
                    "gene_label": 1 if family_index % 2 == 0 else 0,
                    "saturation_tier": "low" if family_index < 2 else "high",
                    "split": split,
                }
            )
    write_tsv(
        path / "splits.tsv",
        rows,
        ["family_id", "method", "tensor_file", "gene_label", "saturation_tier", "split"],
    )
    return path
