import json

import pytest
from typer.testing import CliRunner

from babappa.align import AlignmentConfig, align_simulation_directory
from babappa.cli import app
from babappa.datasets import DatasetIndexConfig, build_dataset_index, read_tsv
from babappa.simulate import SimulationConfig, simulate_families
from babappa.tensors import TensorBuildConfig, build_tensor_dataset
from babappa.training import (
    NeuralTrainConfig,
    safe_import_torch,
    train_neural_smoke_model,
    validate_neural_smoke_dir,
)


torch, _torch_error = safe_import_torch()
pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch is not available")
runner = CliRunner()


def test_train_neural_smoke_creates_artifacts(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "neural_smoke"

    summary = train_neural_smoke_model(_train_config(dataset_dir, model_dir))

    assert summary["status"] == "ok"
    assert (model_dir / "neural_smoke_checkpoint.pt").exists()
    assert (model_dir / "neural_smoke_model_meta.json").exists()
    assert (model_dir / "neural_smoke_history.tsv").exists()
    assert (model_dir / "neural_smoke_predictions.tsv").exists()
    assert (model_dir / "neural_smoke_metrics.json").exists()


def test_validate_neural_smoke_succeeds(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "neural_validate"
    train_neural_smoke_model(_train_config(dataset_dir, model_dir))

    summary = validate_neural_smoke_dir(model_dir)

    assert summary["status"] == "ok"
    assert summary["n_fail"] == 0


def test_history_has_epochs(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "neural_history"
    train_neural_smoke_model(_train_config(dataset_dir, model_dir, epochs=2))

    rows = read_tsv(model_dir / "neural_smoke_history.tsv")

    assert len(rows) >= 2


def test_predictions_are_probabilities(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "neural_predictions"
    train_neural_smoke_model(_train_config(dataset_dir, model_dir))
    rows = read_tsv(model_dir / "neural_smoke_predictions.tsv")

    assert rows
    for row in rows:
        prob = float(row["prob_positive"])
        assert 0 <= prob <= 1
        assert row["pred_label"] in {"0", "1"}


def test_cli_train_neural_smoke_exits_0(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "neural_cli"

    result = runner.invoke(
        app,
        [
            "train-neural-smoke",
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
            "4",
            "--max-val-items",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "Checkpoint" in result.output


def test_cli_validate_neural_smoke_exits_0(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "neural_cli_validate"
    train_neural_smoke_model(_train_config(dataset_dir, model_dir))

    result = runner.invoke(app, ["validate-neural-smoke", "--model-dir", str(model_dir)])

    assert result.exit_code == 0
    assert "ok" in result.output


def test_invalid_device_fails(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)

    result = runner.invoke(
        app,
        [
            "train-neural-smoke",
            "--dataset-dir",
            str(dataset_dir),
            "--outdir",
            str(tmp_path / "bad_device"),
            "--device",
            "bad_device",
        ],
    )

    assert result.exit_code != 0
    assert "device must be one of" in result.output


def test_corrupt_neural_smoke_missing_checkpoint_fails_validation(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "neural_corrupt"
    train_neural_smoke_model(_train_config(dataset_dir, model_dir))
    (model_dir / "neural_smoke_checkpoint.pt").unlink()

    result = runner.invoke(app, ["validate-neural-smoke", "--model-dir", str(model_dir)])

    assert result.exit_code != 0
    assert "missing neural_smoke_checkpoint.pt" in result.output


def _train_config(dataset_dir, model_dir, epochs: int = 2) -> NeuralTrainConfig:
    return NeuralTrainConfig(
        dataset_dir=str(dataset_dir),
        outdir=str(model_dir),
        device="cpu",
        methods=["identity", "codon_dropout"],
        epochs=epochs,
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.0001,
        embedding_dim=16,
        hidden_dim=32,
        dropout=0.1,
        max_train_items=8,
        max_val_items=4,
    )


def _prepare_dataset(tmp_path):
    sim_dir = tmp_path / "sim"
    align_dir = tmp_path / "align"
    tensor_dir = tmp_path / "tensors"
    dataset_dir = tmp_path / "dataset"
    simulate_families(
        SimulationConfig(
            outdir=str(sim_dir),
            n_families=20,
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
