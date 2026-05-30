import pytest
from typer.testing import CliRunner

from babappa.align import AlignmentConfig, align_simulation_directory
from babappa.cli import app
from babappa.datasets import DatasetIndexConfig, build_dataset_index, read_tsv
from babappa.simulate import SimulationConfig, simulate_families
from babappa.tensors import TensorBuildConfig, build_tensor_dataset
from babappa.training import (
    NeuralFullTrainConfig,
    safe_import_torch,
    train_neural_model,
    validate_neural_model_dir,
)


torch, _torch_error = safe_import_torch()
pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch is not available")
runner = CliRunner()


def test_train_neural_full_creates_artifacts(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "neural_full"

    summary = train_neural_model(_train_config(dataset_dir, model_dir))

    assert summary["status"] == "ok"
    assert (model_dir / "checkpoints" / "best_model.pt").exists()
    assert (model_dir / "checkpoints" / "last_model.pt").exists()
    assert (model_dir / "neural_model_meta.json").exists()
    assert (model_dir / "logs" / "neural_training_history.tsv").exists()
    assert (model_dir / "predictions" / "neural_predictions.tsv").exists()
    assert (model_dir / "neural_metrics.json").exists()


def test_validate_neural_full_succeeds(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "neural_full_validate"
    train_neural_model(_train_config(dataset_dir, model_dir))

    summary = validate_neural_model_dir(model_dir)

    assert summary["status"] == "ok"
    assert summary["n_fail"] == 0


def test_history_has_epoch_rows(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "neural_history"
    train_neural_model(_train_config(dataset_dir, model_dir, epochs=3))

    rows = read_tsv(model_dir / "logs" / "neural_training_history.tsv")

    assert 1 <= len(rows) <= 3


def test_predictions_include_all_available_splits(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "neural_predictions"
    train_neural_model(_train_config(dataset_dir, model_dir))
    rows = read_tsv(model_dir / "predictions" / "neural_predictions.tsv")
    splits = {row["split"] for row in rows}

    assert "train" in splits
    assert "val" in splits


def test_probabilities_valid(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "neural_probabilities"
    train_neural_model(_train_config(dataset_dir, model_dir))
    rows = read_tsv(model_dir / "predictions" / "neural_predictions.tsv")

    assert rows
    for row in rows:
        prob = float(row["prob_positive"])
        assert 0 <= prob <= 1
        assert row["pred_label"] in {"0", "1"}


def test_cli_train_neural_exits_0(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "neural_cli"

    result = runner.invoke(
        app,
        [
            "train-neural",
            "--dataset-dir",
            str(dataset_dir),
            "--outdir",
            str(model_dir),
            "--device",
            "cpu",
            "--methods",
            "identity,codon_dropout",
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--max-train-items",
            "8",
            "--max-val-items",
            "4",
        ],
    )

    assert result.exit_code == 0
    assert "Best checkpoint" in result.output


def test_cli_validate_neural_exits_0(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "neural_cli_validate"
    train_neural_model(_train_config(dataset_dir, model_dir))

    result = runner.invoke(app, ["validate-neural", "--model-dir", str(model_dir)])

    assert result.exit_code == 0
    assert "ok" in result.output


def test_invalid_monitor_metric_fails(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)

    result = runner.invoke(
        app,
        [
            "train-neural",
            "--dataset-dir",
            str(dataset_dir),
            "--outdir",
            str(tmp_path / "bad_monitor"),
            "--monitor-metric",
            "bad_metric",
        ],
    )

    assert result.exit_code != 0
    assert "monitor_metric must be one of" in result.output


def test_corrupt_neural_missing_best_checkpoint_fails_validation(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "neural_corrupt"
    train_neural_model(_train_config(dataset_dir, model_dir))
    (model_dir / "checkpoints" / "best_model.pt").unlink()

    result = runner.invoke(app, ["validate-neural", "--model-dir", str(model_dir)])

    assert result.exit_code != 0
    assert "missing best_model.pt" in result.output


def _train_config(dataset_dir, model_dir, epochs: int = 3) -> NeuralFullTrainConfig:
    return NeuralFullTrainConfig(
        dataset_dir=str(dataset_dir),
        outdir=str(model_dir),
        device="cpu",
        methods=["identity", "codon_dropout"],
        epochs=epochs,
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.0001,
        embedding_dim=32,
        hidden_dim=64,
        dropout=0.1,
        max_train_items=12,
        max_val_items=4,
        max_calib_items=2,
        max_test_items=2,
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
            n_families=24,
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
