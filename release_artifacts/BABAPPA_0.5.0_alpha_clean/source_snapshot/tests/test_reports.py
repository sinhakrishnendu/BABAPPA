import json

from typer.testing import CliRunner

from babappa.align import AlignmentConfig, align_simulation_directory
from babappa.calibration import BaselineCalibrationConfig, calibrate_baseline_model
from babappa.cli import app
from babappa.datasets import DatasetIndexConfig, build_dataset_index
from babappa.models import BaselineTrainConfig, train_baseline_model
from babappa.reports import ReportConfig, generate_report, validate_report_dir
from babappa.simulate import (
    SimulationConfig,
    audit_simulation_directory,
    simulate_families,
)
from babappa.tensors import TensorBuildConfig, build_tensor_dataset


runner = CliRunner()


def test_generate_report_writes_expected_artifacts(tmp_path) -> None:
    paths = _prepare_outputs(tmp_path)
    report_dir = tmp_path / "report"

    summary = generate_report(_report_config(paths, report_dir))

    assert summary["status"] == "ok"
    assert (report_dir / "report_summary.json").exists()
    assert (report_dir / "report.md").exists()


def test_validate_report_dir_succeeds(tmp_path) -> None:
    paths = _prepare_outputs(tmp_path)
    report_dir = tmp_path / "report_validate"
    generate_report(_report_config(paths, report_dir))

    summary = validate_report_dir(report_dir)

    assert summary["status"] == "ok"
    assert summary["n_fail"] == 0


def test_report_summary_contains_expected_sections(tmp_path) -> None:
    paths = _prepare_outputs(tmp_path)
    report_dir = tmp_path / "report_sections"
    generate_report(_report_config(paths, report_dir))
    payload = json.loads((report_dir / "report_summary.json").read_text("utf-8"))

    for section in [
        "simulation",
        "alignment",
        "tensorization",
        "dataset_index",
        "baseline_model",
        "calibration",
    ]:
        assert section in payload["sections"]


def test_report_markdown_contains_expected_sections(tmp_path) -> None:
    paths = _prepare_outputs(tmp_path)
    report_dir = tmp_path / "report_markdown"
    generate_report(_report_config(paths, report_dir))
    markdown = (report_dir / "report.md").read_text("utf-8")

    assert "Simulation summary" in markdown
    assert "Baseline model" in markdown
    assert "Calibration" in markdown
    assert "Limitations" in markdown


def test_cli_make_report_exits_successfully(tmp_path) -> None:
    paths = _prepare_outputs(tmp_path)
    report_dir = tmp_path / "report_cli"

    result = runner.invoke(app, _make_report_args(paths, report_dir))

    assert result.exit_code == 0
    assert "JSON report" in result.output


def test_cli_validate_report_exits_successfully(tmp_path) -> None:
    paths = _prepare_outputs(tmp_path)
    report_dir = tmp_path / "report_cli_validate"
    generate_report(_report_config(paths, report_dir))

    result = runner.invoke(app, ["validate-report", "--report-dir", str(report_dir)])

    assert result.exit_code == 0
    assert "ok" in result.output


def test_make_report_without_input_dirs_fails_gracefully(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "make-report",
            "--outdir",
            str(tmp_path / "empty_report"),
            "--title",
            "No inputs",
        ],
    )

    assert result.exit_code != 0
    assert "at least one input directory must be supplied" in result.output


def test_validate_report_fails_when_markdown_missing(tmp_path) -> None:
    paths = _prepare_outputs(tmp_path)
    report_dir = tmp_path / "report_corrupt"
    generate_report(_report_config(paths, report_dir))
    (report_dir / "report.md").unlink()

    result = runner.invoke(app, ["validate-report", "--report-dir", str(report_dir)])

    assert result.exit_code != 0
    assert "missing report.md" in result.output


def _prepare_outputs(tmp_path):
    sim_dir = tmp_path / "sim"
    align_dir = tmp_path / "align"
    tensor_dir = tmp_path / "tensors"
    dataset_dir = tmp_path / "dataset"
    baseline_dir = tmp_path / "baseline"
    calibration_dir = tmp_path / "calibration"
    sim_audit_dir = sim_dir / "audit"

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
            epochs=80,
        )
    )
    calibrate_baseline_model(
        BaselineCalibrationConfig(
            model_dir=str(baseline_dir),
            outdir=str(calibration_dir),
            target_fdr=0.10,
            calibration_method="temperature",
        )
    )
    return {
        "sim_dir": sim_dir,
        "sim_audit_dir": sim_audit_dir,
        "align_dir": align_dir,
        "tensor_dir": tensor_dir,
        "dataset_dir": dataset_dir,
        "baseline_dir": baseline_dir,
        "calibration_dir": calibration_dir,
    }


def _report_config(paths, report_dir):
    return ReportConfig(
        sim_dir=str(paths["sim_dir"]),
        sim_audit_dir=str(paths["sim_audit_dir"]),
        align_dir=str(paths["align_dir"]),
        tensor_dir=str(paths["tensor_dir"]),
        dataset_dir=str(paths["dataset_dir"]),
        baseline_dir=str(paths["baseline_dir"]),
        calibration_dir=str(paths["calibration_dir"]),
        outdir=str(report_dir),
        title="BABAPPA smoke report",
    )


def _make_report_args(paths, report_dir):
    return [
        "make-report",
        "--outdir",
        str(report_dir),
        "--title",
        "BABAPPA smoke report",
        "--sim-dir",
        str(paths["sim_dir"]),
        "--sim-audit-dir",
        str(paths["sim_audit_dir"]),
        "--align-dir",
        str(paths["align_dir"]),
        "--tensor-dir",
        str(paths["tensor_dir"]),
        "--dataset-dir",
        str(paths["dataset_dir"]),
        "--baseline-dir",
        str(paths["baseline_dir"]),
        "--calibration-dir",
        str(paths["calibration_dir"]),
    ]
