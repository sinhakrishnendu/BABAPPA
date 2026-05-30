import json

import numpy as np
from typer.testing import CliRunner

from babappa.align import AlignmentConfig, align_simulation_directory
from babappa.cli import app
from babappa.datasets import DatasetIndexConfig, build_dataset_index, read_tsv
from babappa.models import BaselineTrainConfig, train_baseline_model
from babappa.simulate import SimulationConfig, simulate_families
from babappa.tensors import TensorBuildConfig, build_tensor_dataset


runner = CliRunner()


def test_train_baseline_writes_expected_artifacts(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "baseline"

    summary = train_baseline_model(
        BaselineTrainConfig(dataset_dir=str(dataset_dir), outdir=str(model_dir))
    )

    assert summary["status"] == "ok"
    assert (model_dir / "baseline_model.npz").exists()
    assert (model_dir / "baseline_model_meta.json").exists()
    assert (model_dir / "baseline_predictions.tsv").exists()
    assert (model_dir / "baseline_metrics.json").exists()


def test_validate_baseline_model_dir_succeeds(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "baseline_validate"
    train_baseline_model(
        BaselineTrainConfig(dataset_dir=str(dataset_dir), outdir=str(model_dir))
    )

    result = runner.invoke(app, ["validate-baseline", "--model-dir", str(model_dir)])

    assert result.exit_code == 0
    assert "ok" in result.output


def test_model_npz_arrays_have_expected_shapes(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "baseline_npz"
    train_baseline_model(
        BaselineTrainConfig(dataset_dir=str(dataset_dir), outdir=str(model_dir))
    )

    with np.load(model_dir / "baseline_model.npz", allow_pickle=False) as model:
        assert "weights" in model.files
        assert model["weights"].ndim == 1
        assert model["feature_mean"].shape == model["weights"].shape
        assert model["feature_std"].shape == model["weights"].shape


def test_predictions_are_valid_probabilities_and_labels(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "baseline_predictions"
    train_baseline_model(
        BaselineTrainConfig(dataset_dir=str(dataset_dir), outdir=str(model_dir))
    )
    rows = read_tsv(model_dir / "baseline_predictions.tsv")

    assert rows
    assert "split" in rows[0]
    for row in rows:
        prob = float(row["prob_positive"])
        assert 0 <= prob <= 1
        assert row["pred_label"] in {"0", "1"}


def test_metrics_json_contains_expected_keys(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "baseline_metrics"
    train_baseline_model(
        BaselineTrainConfig(dataset_dir=str(dataset_dir), outdir=str(model_dir))
    )
    metrics = json.loads((model_dir / "baseline_metrics.json").read_text("utf-8"))

    assert "metrics_by_split" in metrics
    assert "all" in metrics["metrics_by_split"]
    assert "train" in metrics["metrics_by_split"]
    assert "accuracy" in metrics["metrics_by_split"]["all"]
    assert "auroc" in metrics["metrics_by_split"]["all"]


def test_cli_train_baseline_exits_successfully(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "baseline_cli"

    result = runner.invoke(
        app,
        [
            "train-baseline",
            "--dataset-dir",
            str(dataset_dir),
            "--outdir",
            str(model_dir),
            "--seed",
            "42",
            "--epochs",
            "50",
            "--learning-rate",
            "0.05",
            "--l2",
            "0.001",
        ],
    )

    assert result.exit_code == 0
    assert "Baseline model path:" in result.output


def test_cli_validate_baseline_exits_successfully(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "baseline_cli_validate"
    train_baseline_model(
        BaselineTrainConfig(dataset_dir=str(dataset_dir), outdir=str(model_dir))
    )

    result = runner.invoke(app, ["validate-baseline", "--model-dir", str(model_dir)])

    assert result.exit_code == 0
    assert "ok" in result.output


def test_invalid_epochs_fail_gracefully(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)

    result = runner.invoke(
        app,
        [
            "train-baseline",
            "--dataset-dir",
            str(dataset_dir),
            "--outdir",
            str(tmp_path / "bad_epochs"),
            "--epochs",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "epochs must be >= 1" in result.output


def test_validate_baseline_fails_when_model_npz_missing(tmp_path) -> None:
    dataset_dir = _prepare_dataset(tmp_path)
    model_dir = tmp_path / "baseline_corrupt"
    train_baseline_model(
        BaselineTrainConfig(dataset_dir=str(dataset_dir), outdir=str(model_dir))
    )
    (model_dir / "baseline_model.npz").unlink()

    result = runner.invoke(app, ["validate-baseline", "--model-dir", str(model_dir)])

    assert result.exit_code != 0
    assert "missing baseline_model.npz" in result.output


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
