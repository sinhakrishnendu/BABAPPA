import json

from typer.testing import CliRunner

from babappa.align import AlignmentConfig, align_simulation_directory
from babappa.calibration import (
    BaselineCalibrationConfig,
    calibrate_baseline_model,
    validate_baseline_calibration_dir,
)
from babappa.cli import app
from babappa.datasets import DatasetIndexConfig, build_dataset_index, read_tsv
from babappa.models import BaselineTrainConfig, train_baseline_model
from babappa.simulate import SimulationConfig, simulate_families
from babappa.tensors import TensorBuildConfig, build_tensor_dataset


runner = CliRunner()


def test_calibrate_baseline_writes_expected_artifacts(tmp_path) -> None:
    model_dir = _prepare_baseline(tmp_path)
    calibration_dir = tmp_path / "calibration"

    summary = calibrate_baseline_model(
        BaselineCalibrationConfig(model_dir=str(model_dir), outdir=str(calibration_dir))
    )

    assert summary["status"] == "ok"
    assert (calibration_dir / "baseline_calibration.json").exists()
    assert (calibration_dir / "baseline_calibrated_predictions.tsv").exists()
    assert (calibration_dir / "baseline_calibrated_metrics.json").exists()


def test_validate_baseline_calibration_dir_succeeds(tmp_path) -> None:
    model_dir = _prepare_baseline(tmp_path)
    calibration_dir = tmp_path / "calibration_validate"
    calibrate_baseline_model(
        BaselineCalibrationConfig(model_dir=str(model_dir), outdir=str(calibration_dir))
    )

    summary = validate_baseline_calibration_dir(calibration_dir)

    assert summary["status"] == "ok"
    assert summary["n_fail"] == 0


def test_calibration_json_contains_temperature_and_threshold(tmp_path) -> None:
    model_dir = _prepare_baseline(tmp_path)
    calibration_dir = tmp_path / "calibration_json"
    calibrate_baseline_model(
        BaselineCalibrationConfig(model_dir=str(model_dir), outdir=str(calibration_dir))
    )
    payload = json.loads(
        (calibration_dir / "baseline_calibration.json").read_text("utf-8")
    )

    assert "temperature" in payload
    assert "selected_threshold" in payload
    assert payload["temperature"] > 0
    assert 0 <= payload["selected_threshold"] <= 1


def test_calibrated_predictions_are_valid(tmp_path) -> None:
    model_dir = _prepare_baseline(tmp_path)
    calibration_dir = tmp_path / "calibration_predictions"
    calibrate_baseline_model(
        BaselineCalibrationConfig(model_dir=str(model_dir), outdir=str(calibration_dir))
    )
    rows = read_tsv(calibration_dir / "baseline_calibrated_predictions.tsv")

    assert rows
    for row in rows:
        prob = float(row["prob_positive_calibrated"])
        assert 0 <= prob <= 1
        assert row["pred_label_calibrated"] in {"0", "1"}


def test_calibrated_metrics_json_contains_expected_keys(tmp_path) -> None:
    model_dir = _prepare_baseline(tmp_path)
    calibration_dir = tmp_path / "calibration_metrics"
    calibrate_baseline_model(
        BaselineCalibrationConfig(model_dir=str(model_dir), outdir=str(calibration_dir))
    )
    payload = json.loads(
        (calibration_dir / "baseline_calibrated_metrics.json").read_text("utf-8")
    )

    assert "metrics_by_split_calibrated" in payload
    assert "selected_threshold" in payload
    assert "temperature" in payload


def test_cli_calibrate_baseline_exits_successfully(tmp_path) -> None:
    model_dir = _prepare_baseline(tmp_path, epochs=60)
    calibration_dir = tmp_path / "calibration_cli"

    result = runner.invoke(
        app,
        [
            "calibrate-baseline",
            "--model-dir",
            str(model_dir),
            "--outdir",
            str(calibration_dir),
            "--target-fdr",
            "0.10",
            "--calibration-method",
            "temperature",
        ],
    )

    assert result.exit_code == 0
    assert "Calibrated predictions" in result.output


def test_cli_validate_calibration_exits_successfully(tmp_path) -> None:
    model_dir = _prepare_baseline(tmp_path)
    calibration_dir = tmp_path / "calibration_cli_validate"
    calibrate_baseline_model(
        BaselineCalibrationConfig(model_dir=str(model_dir), outdir=str(calibration_dir))
    )

    result = runner.invoke(
        app,
        ["validate-calibration", "--calibration-dir", str(calibration_dir)],
    )

    assert result.exit_code == 0
    assert "ok" in result.output


def test_invalid_target_fdr_fails_gracefully(tmp_path) -> None:
    model_dir = _prepare_baseline(tmp_path)

    result = runner.invoke(
        app,
        [
            "calibrate-baseline",
            "--model-dir",
            str(model_dir),
            "--outdir",
            str(tmp_path / "bad_fdr"),
            "--target-fdr",
            "1.5",
        ],
    )

    assert result.exit_code != 0
    assert "target_fdr must be between 0 and 1" in result.output


def test_validate_calibration_fails_when_json_missing(tmp_path) -> None:
    model_dir = _prepare_baseline(tmp_path)
    calibration_dir = tmp_path / "calibration_corrupt"
    calibrate_baseline_model(
        BaselineCalibrationConfig(model_dir=str(model_dir), outdir=str(calibration_dir))
    )
    (calibration_dir / "baseline_calibration.json").unlink()

    result = runner.invoke(
        app,
        ["validate-calibration", "--calibration-dir", str(calibration_dir)],
    )

    assert result.exit_code != 0
    assert "missing baseline_calibration.json" in result.output


def _prepare_baseline(tmp_path, epochs: int = 100):
    sim_dir = tmp_path / "sim"
    align_dir = tmp_path / "align"
    tensor_dir = tmp_path / "tensors"
    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "baseline"
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
    train_baseline_model(
        BaselineTrainConfig(
            dataset_dir=str(dataset_dir),
            outdir=str(model_dir),
            epochs=epochs,
        )
    )
    return model_dir
