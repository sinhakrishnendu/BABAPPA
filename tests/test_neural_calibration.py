import json
import shutil

import pytest
from typer.testing import CliRunner

from babappa.align import AlignmentConfig, align_simulation_directory
from babappa.calibration import (
    NeuralCalibrationConfig,
    calibrate_neural_model,
    validate_neural_calibration_dir,
)
from babappa.cli import app
from babappa.datasets import DatasetIndexConfig, build_dataset_index, read_tsv
from babappa.reports import ReportConfig, generate_report
from babappa.simulate import SimulationConfig, simulate_families
from babappa.tensors import TensorBuildConfig, build_tensor_dataset
from babappa.training import (
    NeuralFullTrainConfig,
    safe_import_torch,
    train_neural_model,
)


torch, _torch_error = safe_import_torch()
pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch is not available")
runner = CliRunner()


@pytest.fixture(scope="module")
def neural_artifacts(tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("neural_calibration")
    dataset_dir = _prepare_dataset(base_dir)
    model_dir = base_dir / "neural_model"
    train_neural_model(_train_config(dataset_dir, model_dir))
    calibration_dir = base_dir / "neural_calibration"
    calibrate_neural_model(
        NeuralCalibrationConfig(
            model_dir=str(model_dir),
            outdir=str(calibration_dir),
            target_fdr=0.10,
            calibration_method="temperature",
        )
    )
    return {
        "base_dir": base_dir,
        "dataset_dir": dataset_dir,
        "model_dir": model_dir,
        "calibration_dir": calibration_dir,
    }


def test_calibrate_neural_creates_artifacts(neural_artifacts) -> None:
    calibration_dir = neural_artifacts["calibration_dir"]

    assert (calibration_dir / "neural_calibration.json").exists()
    assert (calibration_dir / "neural_calibrated_predictions.tsv").exists()
    assert (calibration_dir / "neural_calibrated_metrics.json").exists()


def test_validate_neural_calibration_succeeds(neural_artifacts) -> None:
    summary = validate_neural_calibration_dir(neural_artifacts["calibration_dir"])

    assert summary["status"] == "ok"
    assert summary["n_fail"] == 0


def test_neural_calibration_json_fields(neural_artifacts) -> None:
    payload = json.loads(
        (neural_artifacts["calibration_dir"] / "neural_calibration.json").read_text(
            "utf-8"
        )
    )

    assert payload["temperature"] > 0
    assert 0 <= payload["selected_threshold"] <= 1
    assert "target_fdr" in payload


def test_neural_calibrated_predictions_valid(neural_artifacts) -> None:
    rows = read_tsv(
        neural_artifacts["calibration_dir"] / "neural_calibrated_predictions.tsv"
    )

    assert rows
    for row in rows:
        prob = float(row["prob_positive_calibrated"])
        assert 0 <= prob <= 1
        assert row["pred_label_calibrated"] in {"0", "1"}


def test_cli_calibrate_neural_exits_0(tmp_path, neural_artifacts) -> None:
    calibration_dir = tmp_path / "cli_neural_calibration"

    result = runner.invoke(
        app,
        [
            "calibrate-neural",
            "--model-dir",
            str(neural_artifacts["model_dir"]),
            "--outdir",
            str(calibration_dir),
            "--target-fdr",
            "0.10",
            "--calibration-method",
            "temperature",
        ],
    )

    assert result.exit_code == 0
    assert "Selected threshold" in result.output


def test_cli_validate_neural_calibration_exits_0(neural_artifacts) -> None:
    result = runner.invoke(
        app,
        [
            "validate-neural-calibration",
            "--calibration-dir",
            str(neural_artifacts["calibration_dir"]),
        ],
    )

    assert result.exit_code == 0
    assert "ok" in result.output


def test_invalid_target_fdr_fails(tmp_path, neural_artifacts) -> None:
    result = runner.invoke(
        app,
        [
            "calibrate-neural",
            "--model-dir",
            str(neural_artifacts["model_dir"]),
            "--outdir",
            str(tmp_path / "bad_fdr"),
            "--target-fdr",
            "-0.1",
        ],
    )

    assert result.exit_code != 0
    assert "target_fdr must be between 0 and 1" in result.output


def test_corrupt_neural_calibration_missing_json_fails_validation(
    tmp_path, neural_artifacts
) -> None:
    corrupt_dir = tmp_path / "corrupt_neural_calibration"
    shutil.copytree(neural_artifacts["calibration_dir"], corrupt_dir)
    (corrupt_dir / "neural_calibration.json").unlink()

    result = runner.invoke(
        app,
        ["validate-neural-calibration", "--calibration-dir", str(corrupt_dir)],
    )

    assert result.exit_code != 0
    assert "missing neural_calibration.json" in result.output


def test_report_includes_neural_sections(tmp_path, neural_artifacts) -> None:
    report_dir = tmp_path / "neural_report"
    generate_report(
        ReportConfig(
            dataset_dir=str(neural_artifacts["dataset_dir"]),
            neural_dir=str(neural_artifacts["model_dir"]),
            neural_calibration_dir=str(neural_artifacts["calibration_dir"]),
            outdir=str(report_dir),
            title="BABAPPA neural report",
        )
    )

    payload = json.loads((report_dir / "report_summary.json").read_text("utf-8"))
    markdown = (report_dir / "report.md").read_text("utf-8")

    assert "neural_model" in payload["sections"]
    assert "neural_calibration" in payload["sections"]
    assert "Neural model" in markdown
    assert "Neural calibration" in markdown


def _train_config(dataset_dir, model_dir) -> NeuralFullTrainConfig:
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
        max_train_items=16,
        max_val_items=6,
        max_calib_items=4,
        max_test_items=4,
        early_stopping_patience=2,
    )


def _prepare_dataset(base_dir):
    sim_dir = base_dir / "sim"
    align_dir = base_dir / "align"
    tensor_dir = base_dir / "tensors"
    dataset_dir = base_dir / "dataset"
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
