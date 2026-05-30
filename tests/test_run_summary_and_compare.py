import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from babappa.align import AlignmentConfig, align_simulation_directory
from babappa.calibration import (
    BaselineCalibrationConfig,
    NeuralCalibrationConfig,
    calibrate_baseline_model,
    calibrate_neural_model,
)
from babappa.cli import app
from babappa.datasets import DatasetIndexConfig, build_dataset_index
from babappa.models import BaselineTrainConfig, train_baseline_model
from babappa.reports import (
    ModelCompareConfig,
    ReportConfig,
    RunSummaryConfig,
    compare_models,
    generate_report,
    summarize_run,
    validate_model_comparison_dir,
    validate_run_summary_dir,
)
from babappa.simulate import (
    SimulationConfig,
    audit_simulation_directory,
    simulate_families,
)
from babappa.tensors import TensorBuildConfig, build_tensor_dataset
from babappa.training import NeuralFullTrainConfig, safe_import_torch, train_neural_model


torch, _torch_error = safe_import_torch()
runner = CliRunner()


def test_repository_hardening_init_files() -> None:
    root = Path(__file__).resolve().parents[1]

    assert not (root / "src" / "babappa" / "init.py").exists()
    assert not (root / "src" / "babappa" / "calibration" / "init.py").exists()
    assert (root / "src" / "babappa" / "__init__.py").exists()
    assert (root / "src" / "babappa" / "calibration" / "__init__.py").exists()


def test_summarize_run_dataset_only_without_neural(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset_only"
    dataset_dir.mkdir()
    outdir = tmp_path / "summary_dataset_only"

    summary = summarize_run(
        RunSummaryConfig(
            outdir=str(outdir),
            dataset_dir=str(dataset_dir),
            title="Dataset-only summary",
        )
    )

    assert summary["status"] == "ok"
    assert (outdir / "run_summary.json").exists()
    assert (outdir / "run_summary.md").exists()


@pytest.fixture(scope="module")
def full_outputs(tmp_path_factory):
    if torch is None:
        pytest.skip("PyTorch is not available")

    base_dir = tmp_path_factory.mktemp("run_summary_compare")
    paths = _prepare_full_outputs(base_dir)
    summarize_run(_summary_config(paths, base_dir / "run_summary"))
    compare_models(_compare_config(paths, base_dir / "model_compare"))
    paths["summary_dir"] = base_dir / "run_summary"
    paths["compare_dir"] = base_dir / "model_compare"
    return paths


def test_summarize_run_creates_artifacts(full_outputs) -> None:
    summary_dir = full_outputs["summary_dir"]

    assert (summary_dir / "run_summary.json").exists()
    assert (summary_dir / "run_summary.md").exists()


def test_validate_run_summary_succeeds(full_outputs) -> None:
    summary = validate_run_summary_dir(full_outputs["summary_dir"])

    assert summary["status"] == "ok"
    assert summary["n_fail"] == 0


def test_run_summary_has_recommended_next_action(full_outputs) -> None:
    payload = json.loads(
        (full_outputs["summary_dir"] / "run_summary.json").read_text("utf-8")
    )

    assert payload["recommended_next_action"]


def test_compare_models_creates_artifacts(full_outputs) -> None:
    compare_dir = full_outputs["compare_dir"]

    assert (compare_dir / "model_comparison.json").exists()
    assert (compare_dir / "model_comparison.tsv").exists()
    assert (compare_dir / "model_comparison.md").exists()


def test_validate_model_comparison_succeeds(full_outputs) -> None:
    summary = validate_model_comparison_dir(full_outputs["compare_dir"])

    assert summary["status"] == "ok"
    assert summary["n_fail"] == 0


def test_cli_summarize_run_exits_0(tmp_path, full_outputs) -> None:
    result = runner.invoke(
        app,
        [
            "summarize-run",
            "--outdir",
            str(tmp_path / "cli_summary"),
            "--title",
            "BABAPPA CLI run summary",
            "--sim-dir",
            str(full_outputs["sim_dir"]),
            "--sim-audit-dir",
            str(full_outputs["sim_audit_dir"]),
            "--align-dir",
            str(full_outputs["align_dir"]),
            "--tensor-dir",
            str(full_outputs["tensor_dir"]),
            "--dataset-dir",
            str(full_outputs["dataset_dir"]),
            "--baseline-dir",
            str(full_outputs["baseline_dir"]),
            "--baseline-calibration-dir",
            str(full_outputs["baseline_calibration_dir"]),
            "--neural-dir",
            str(full_outputs["neural_dir"]),
            "--neural-calibration-dir",
            str(full_outputs["neural_calibration_dir"]),
            "--report-dir",
            str(full_outputs["report_dir"]),
        ],
    )

    assert result.exit_code == 0
    assert "Recommended next action" in result.output


def test_cli_compare_models_exits_0(tmp_path, full_outputs) -> None:
    result = runner.invoke(
        app,
        [
            "compare-models",
            "--outdir",
            str(tmp_path / "cli_compare"),
            "--baseline-metrics",
            str(full_outputs["baseline_dir"] / "baseline_metrics.json"),
            "--baseline-calibrated-metrics",
            str(
                full_outputs["baseline_calibration_dir"]
                / "baseline_calibrated_metrics.json"
            ),
            "--neural-metrics",
            str(full_outputs["neural_dir"] / "neural_metrics.json"),
            "--neural-calibrated-metrics",
            str(
                full_outputs["neural_calibration_dir"]
                / "neural_calibrated_metrics.json"
            ),
        ],
    )

    assert result.exit_code == 0
    assert "JSON" in result.output


def test_compare_models_requires_two_inputs(tmp_path) -> None:
    metric_path = tmp_path / "one_metrics.json"
    metric_path.write_text('{"metrics_by_split": {}}\n', encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "compare-models",
            "--outdir",
            str(tmp_path / "bad_compare"),
            "--baseline-metrics",
            str(metric_path),
        ],
    )

    assert result.exit_code != 0
    assert "at least two metric files must be supplied" in result.output


def _prepare_full_outputs(base_dir):
    sim_dir = base_dir / "sim"
    sim_audit_dir = sim_dir / "audit"
    align_dir = base_dir / "align"
    tensor_dir = base_dir / "tensors"
    dataset_dir = base_dir / "dataset"
    baseline_dir = base_dir / "baseline"
    baseline_calibration_dir = base_dir / "baseline_calibration"
    neural_dir = base_dir / "neural"
    neural_calibration_dir = base_dir / "neural_calibration"
    report_dir = base_dir / "report"

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
    audit_simulation_directory(sim_dir, sim_audit_dir)
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
            outdir=str(baseline_dir),
            seed=42,
            epochs=80,
            learning_rate=0.05,
            l2=0.001,
        )
    )
    calibrate_baseline_model(
        BaselineCalibrationConfig(
            model_dir=str(baseline_dir),
            outdir=str(baseline_calibration_dir),
            target_fdr=0.10,
            calibration_method="temperature",
        )
    )
    train_neural_model(
        NeuralFullTrainConfig(
            dataset_dir=str(dataset_dir),
            outdir=str(neural_dir),
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
    )
    calibrate_neural_model(
        NeuralCalibrationConfig(
            model_dir=str(neural_dir),
            outdir=str(neural_calibration_dir),
            target_fdr=0.10,
            calibration_method="temperature",
        )
    )
    generate_report(
        ReportConfig(
            sim_dir=str(sim_dir),
            sim_audit_dir=str(sim_audit_dir),
            align_dir=str(align_dir),
            tensor_dir=str(tensor_dir),
            dataset_dir=str(dataset_dir),
            baseline_dir=str(baseline_dir),
            calibration_dir=str(baseline_calibration_dir),
            neural_dir=str(neural_dir),
            neural_calibration_dir=str(neural_calibration_dir),
            outdir=str(report_dir),
            title="BABAPPA test report",
        )
    )
    return {
        "sim_dir": sim_dir,
        "sim_audit_dir": sim_audit_dir,
        "align_dir": align_dir,
        "tensor_dir": tensor_dir,
        "dataset_dir": dataset_dir,
        "baseline_dir": baseline_dir,
        "baseline_calibration_dir": baseline_calibration_dir,
        "neural_dir": neural_dir,
        "neural_calibration_dir": neural_calibration_dir,
        "report_dir": report_dir,
    }


def _summary_config(paths, outdir):
    return RunSummaryConfig(
        outdir=str(outdir),
        sim_dir=str(paths["sim_dir"]),
        sim_audit_dir=str(paths["sim_audit_dir"]),
        align_dir=str(paths["align_dir"]),
        tensor_dir=str(paths["tensor_dir"]),
        dataset_dir=str(paths["dataset_dir"]),
        baseline_dir=str(paths["baseline_dir"]),
        baseline_calibration_dir=str(paths["baseline_calibration_dir"]),
        neural_dir=str(paths["neural_dir"]),
        neural_calibration_dir=str(paths["neural_calibration_dir"]),
        report_dir=str(paths["report_dir"]),
        title="BABAPPA test run summary",
    )


def _compare_config(paths, outdir):
    return ModelCompareConfig(
        outdir=str(outdir),
        baseline_metrics=str(paths["baseline_dir"] / "baseline_metrics.json"),
        baseline_calibrated_metrics=str(
            paths["baseline_calibration_dir"] / "baseline_calibrated_metrics.json"
        ),
        neural_metrics=str(paths["neural_dir"] / "neural_metrics.json"),
        neural_calibrated_metrics=str(
            paths["neural_calibration_dir"] / "neural_calibrated_metrics.json"
        ),
    )
