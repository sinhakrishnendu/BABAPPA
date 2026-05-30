import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from babappa.align import AlignmentConfig, align_simulation_directory
from babappa.cli import app
from babappa.datasets import DatasetIndexConfig, build_dataset_index
from babappa.reports import (
    PredictionDiagnosticsConfig,
    diagnose_predictions,
    validate_prediction_diagnostics_dir,
)
from babappa.simulate import SimulationConfig, simulate_families
from babappa.tensors import TensorBuildConfig, build_tensor_dataset
from babappa.training import (
    ContrastiveGeneClassifier,
    NeuralFullTrainConfig,
    safe_import_torch,
    train_neural_model,
)


torch, _torch_error = safe_import_torch()
runner = CliRunner()


def test_repository_hardening_init_files() -> None:
    root = Path(__file__).resolve().parents[1]

    assert not (root / "src" / "babappa" / "init.py").exists()
    assert not (root / "src" / "babappa" / "calibration" / "init.py").exists()
    assert not (root / "src" / "babappa" / "reports" / "init.py").exists()
    assert (root / "src" / "babappa" / "__init__.py").exists()
    assert (root / "src" / "babappa" / "calibration" / "__init__.py").exists()
    assert (root / "src" / "babappa" / "reports" / "__init__.py").exists()


def test_prediction_diagnostics_on_synthetic_predictions(tmp_path) -> None:
    predictions = tmp_path / "predictions.tsv"
    _write_predictions(
        predictions,
        [
            ("a", "train", 1, 0.80),
            ("b", "train", 0, 0.20),
            ("c", "val", 1, 0.70),
            ("d", "val", 0, 0.30),
        ],
    )
    outdir = tmp_path / "diag"

    summary = diagnose_predictions(
        PredictionDiagnosticsConfig(
            predictions_tsv=str(predictions),
            outdir=str(outdir),
            model_name="synthetic",
        )
    )

    assert summary["status"] == "ok"
    assert (outdir / "prediction_diagnostics.json").exists()
    assert (outdir / "prediction_score_summary.tsv").exists()
    assert (outdir / "threshold_curve.tsv").exists()
    assert (outdir / "prediction_diagnostics.md").exists()
    assert validate_prediction_diagnostics_dir(outdir)["status"] == "ok"


def test_prediction_diagnostics_detects_all_negative(tmp_path) -> None:
    predictions = tmp_path / "collapsed.tsv"
    _write_predictions(
        predictions,
        [
            ("a", "train", 1, 0.10),
            ("b", "train", 1, 0.12),
            ("c", "train", 0, 0.08),
            ("d", "train", 0, 0.09),
        ],
    )
    outdir = tmp_path / "collapsed_diag"

    summary = diagnose_predictions(
        PredictionDiagnosticsConfig(
            predictions_tsv=str(predictions),
            outdir=str(outdir),
            model_name="collapsed",
        )
    )

    assert "all_negative_at_0_5" in summary["warnings"]


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_contrastive_model_forward_if_torch_available() -> None:
    model = ContrastiveGeneClassifier(
        vocab_size=128,
        embedding_dim=32,
        hidden_dim=64,
        dropout=0.1,
    )
    X = torch.zeros((4, 6, 60, 2), dtype=torch.long)
    X[..., 0] = torch.randint(0, 64, (4, 6, 60))

    logits = model(X)

    assert list(logits.shape) == [4]


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_train_neural_v2_tiny_if_torch_available(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "neural_v2"

    summary = train_neural_model(_v2_config(dataset_dir, model_dir))
    meta = json.loads((model_dir / "neural_model_meta.json").read_text("utf-8"))

    assert summary["status"] == "ok"
    assert (model_dir / "checkpoints" / "best_model.pt").exists()
    assert (model_dir / "predictions" / "neural_predictions.tsv").exists()
    assert (model_dir / "neural_metrics.json").exists()
    assert meta["architecture"] == "contrastive"
    assert meta["positive_class_weight"] == "auto"


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_cli_train_neural_v2_exits_0_if_torch_available(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "neural_v2_cli"

    result = runner.invoke(
        app,
        [
            "train-neural-v2",
            "--dataset-dir",
            str(dataset_dir),
            "--outdir",
            str(model_dir),
            "--device",
            "cpu",
            "--methods",
            "identity,codon_dropout",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--max-train-items",
            "8",
            "--max-val-items",
            "4",
            "--max-calib-items",
            "2",
            "--max-test-items",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "Architecture" in result.output


def test_cli_diagnose_predictions_exits_0(tmp_path) -> None:
    predictions = tmp_path / "predictions.tsv"
    _write_predictions(
        predictions,
        [
            ("a", "train", 1, 0.80),
            ("b", "train", 0, 0.20),
        ],
    )

    result = runner.invoke(
        app,
        [
            "diagnose-predictions",
            "--predictions",
            str(predictions),
            "--outdir",
            str(tmp_path / "diag_cli"),
            "--model-name",
            "cli_model",
        ],
    )

    assert result.exit_code == 0
    assert "Prediction Diagnostics" in result.output


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_invalid_architecture_fails(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)

    result = runner.invoke(
        app,
        [
            "train-neural",
            "--dataset-dir",
            str(dataset_dir),
            "--outdir",
            str(tmp_path / "bad_architecture"),
            "--architecture",
            "bad",
        ],
    )

    assert result.exit_code != 0
    assert "architecture must be one of" in result.output


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_invalid_positive_class_weight_fails(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)

    result = runner.invoke(
        app,
        [
            "train-neural",
            "--dataset-dir",
            str(dataset_dir),
            "--outdir",
            str(tmp_path / "bad_weight"),
            "--positive-class-weight",
            "bad",
        ],
    )

    assert result.exit_code != 0
    assert "positive_class_weight must be one of" in result.output


def _write_predictions(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=[
                "family_id",
                "method",
                "split",
                "tensor_file",
                "gene_label",
                "saturation_tier",
                "prob_positive",
                "pred_label",
                "correct",
            ],
        )
        writer.writeheader()
        for family_id, split, label, prob in rows:
            pred = int(prob >= 0.5)
            writer.writerow(
                {
                    "family_id": family_id,
                    "method": "identity",
                    "split": split,
                    "tensor_file": f"{family_id}.npz",
                    "gene_label": label,
                    "saturation_tier": "moderate",
                    "prob_positive": prob,
                    "pred_label": pred,
                    "correct": int(pred == label),
                }
            )


def _v2_config(dataset_dir, model_dir) -> NeuralFullTrainConfig:
    return NeuralFullTrainConfig(
        dataset_dir=str(dataset_dir),
        outdir=str(model_dir),
        device="cpu",
        methods=["identity", "codon_dropout"],
        epochs=2,
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.0001,
        embedding_dim=32,
        hidden_dim=64,
        dropout=0.1,
        architecture="contrastive",
        positive_class_weight="auto",
        max_train_items=16,
        max_val_items=6,
        max_calib_items=4,
        max_test_items=4,
        early_stopping_patience=2,
    )


def _prepare_dataset(tmp_path):
    sim_dir = tmp_path / "sim"
    align_dir = tmp_path / "align"
    tensor_dir = tmp_path / "tensors"
    dataset_dir = tmp_path / "dataset"
    simulate_families(
        SimulationConfig(
            outdir=str(sim_dir),
            n_families=30,
            n_taxa=6,
            n_codons=60,
            seed=42,
            positive_rate=0.5,
            saturation_tier="moderate",
        )
    )
    align_simulation_directory(
        AlignmentConfig(
            sim_dir=str(sim_dir),
            outdir=str(align_dir),
            methods=["identity", "codon_dropout"],
            seed=42,
            dropout_rate=0.02,
        )
    )
    build_tensor_dataset(
        TensorBuildConfig(
            sim_dir=str(sim_dir),
            align_dir=str(align_dir),
            outdir=str(tensor_dir),
            methods=["identity", "codon_dropout"],
        )
    )
    build_dataset_index(
        DatasetIndexConfig(
            tensor_dir=str(tensor_dir),
            outdir=str(dataset_dir),
            methods=["identity", "codon_dropout"],
            seed=42,
        )
    )
    return dataset_dir
