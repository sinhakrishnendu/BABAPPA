import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from babappa.benchmarks import SaturationPanelConfig, build_saturation_panel
from babappa.calibration import (
    StratifiedCalibrationConfig,
    calibrate_by_group,
    validate_stratified_calibration_dir,
)
from babappa.cli import app
from babappa.datasets import DatasetMergeConfig, merge_dataset_indexes
from babappa.reports import (
    ReportConfig,
    RunSummaryConfig,
    generate_report,
    summarize_run,
    validate_report_dir,
    validate_run_summary_dir,
)
from babappa.training import (
    BabappaTensorDataset,
    NeuralDatasetConfig,
    NeuralFullTrainConfig,
    SATURATION_TIER_TO_ID,
    SaturationAwareGeneClassifier,
    collate_babappa_batch,
    safe_import_torch,
    saturation_tier_to_id,
    train_neural_model,
)


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
    ]:
        assert not (root / relative).exists()

    for relative in [
        "src/babappa/__init__.py",
        "src/babappa/calibration/__init__.py",
        "src/babappa/reports/__init__.py",
        "src/babappa/training/__init__.py",
        "src/babappa/benchmarks/__init__.py",
        "src/babappa/datasets/__init__.py",
    ]:
        assert (root / relative).exists()


def test_saturation_tier_to_id() -> None:
    assert SATURATION_TIER_TO_ID == {
        "unknown": 0,
        "low": 1,
        "moderate": 2,
        "high": 3,
        "extreme": 4,
    }
    assert saturation_tier_to_id("low") == 1
    assert saturation_tier_to_id("moderate") == 2
    assert saturation_tier_to_id("high") == 3
    assert saturation_tier_to_id("extreme") == 4
    assert saturation_tier_to_id("missing") == 0
    assert saturation_tier_to_id("") == 0


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_collate_includes_saturation_id(tmp_path) -> None:
    dataset_dir = _build_merged_panel_dataset(tmp_path)
    dataset = BabappaTensorDataset(
        NeuralDatasetConfig(
            dataset_dir=str(dataset_dir),
            split="all",
            max_items=4,
            require_torch=True,
        )
    )
    batch = collate_babappa_batch([dataset[index] for index in range(min(4, len(dataset)))])

    assert "saturation_id" in batch
    assert "saturation_tier" in batch
    assert list(batch["saturation_id"].shape) == [min(4, len(dataset))]
    assert set(int(value) for value in batch["saturation_id"].tolist()) <= {1, 2}


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_saturation_aware_model_forward_if_torch_available() -> None:
    model = SaturationAwareGeneClassifier(
        vocab_size=128,
        embedding_dim=16,
        hidden_dim=32,
        dropout=0.1,
        saturation_embedding_dim=8,
    )
    X = torch.zeros((4, 6, 60, 2), dtype=torch.long)
    X[..., 0] = torch.randint(0, 64, (4, 6, 60))
    saturation_id = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    logits = model(X, saturation_id=saturation_id)

    assert list(logits.shape) == [4]


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_train_neural_saturation_tiny_if_torch_available(tmp_path) -> None:
    dataset_dir = _build_merged_panel_dataset(tmp_path)
    model_dir = tmp_path / "neural_saturation"

    summary = train_neural_model(_saturation_config(dataset_dir, model_dir))
    meta = json.loads((model_dir / "neural_model_meta.json").read_text("utf-8"))

    assert summary["status"] == "ok"
    assert (model_dir / "checkpoints" / "best_model.pt").exists()
    assert (model_dir / "predictions" / "neural_predictions.tsv").exists()
    assert (model_dir / "neural_metrics.json").exists()
    assert meta["architecture"] == "saturation_aware"
    assert meta["group_weighting"] == "saturation_inverse_frequency"
    assert meta["sampler"] == "saturation_balanced"


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_cli_train_neural_saturation_exits_0_if_torch_available(tmp_path) -> None:
    dataset_dir = _build_merged_panel_dataset(tmp_path)

    result = runner.invoke(
        app,
        [
            "train-neural-saturation",
            "--dataset-dir",
            str(dataset_dir),
            "--outdir",
            str(tmp_path / "neural_saturation_cli"),
            "--device",
            "cpu",
            "--methods",
            "identity,codon_dropout",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--embedding-dim",
            "16",
            "--hidden-dim",
            "32",
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
    assert "saturation_aware" in result.output


def test_stratified_calibration_on_synthetic_predictions(tmp_path) -> None:
    predictions = tmp_path / "predictions.tsv"
    _write_stratified_predictions(predictions)
    outdir = tmp_path / "stratified_calibration"

    summary = calibrate_by_group(
        StratifiedCalibrationConfig(
            predictions_tsv=str(predictions),
            outdir=str(outdir),
            group_column="saturation_tier",
            probability_column="prob_positive",
            target_fdr=0.25,
            min_group_calib_n=2,
        )
    )

    assert summary["status"] == "ok"
    assert (outdir / "stratified_calibration.json").exists()
    assert (outdir / "stratified_calibrated_predictions.tsv").exists()
    assert (outdir / "stratified_calibrated_metrics.json").exists()
    assert validate_stratified_calibration_dir(outdir)["status"] == "ok"


def test_cli_calibrate_stratified_exits_0(tmp_path) -> None:
    predictions = tmp_path / "predictions.tsv"
    _write_stratified_predictions(predictions)

    result = runner.invoke(
        app,
        [
            "calibrate-stratified",
            "--predictions",
            str(predictions),
            "--outdir",
            str(tmp_path / "stratified_calibration_cli"),
            "--group-column",
            "saturation_tier",
            "--probability-column",
            "prob_positive",
            "--target-fdr",
            "0.25",
            "--min-group-calib-n",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "Stratified Calibration" in result.output


def test_run_summary_accepts_stratified_calibration_dir(tmp_path) -> None:
    calibration_dir = _make_stratified_calibration(tmp_path)
    summary_dir = tmp_path / "run_summary"

    summarize_run(
        RunSummaryConfig(
            outdir=str(summary_dir),
            stratified_calibration_dir=str(calibration_dir),
            title="Stratified calibration summary",
        )
    )
    payload = json.loads((summary_dir / "run_summary.json").read_text("utf-8"))
    markdown = (summary_dir / "run_summary.md").read_text("utf-8")

    assert validate_run_summary_dir(summary_dir)["status"] == "ok"
    assert payload["status_overview"]["stratified_calibration_present"] is True
    assert payload["stratified_calibration_overview"]["groups"]
    assert "Stratified calibration overview" in markdown


def test_make_report_accepts_stratified_calibration_dir(tmp_path) -> None:
    calibration_dir = _make_stratified_calibration(tmp_path)
    report_dir = tmp_path / "report"

    generate_report(
        ReportConfig(
            outdir=str(report_dir),
            stratified_calibration_dir=str(calibration_dir),
            title="Stratified calibration report",
        )
    )
    payload = json.loads((report_dir / "report_summary.json").read_text("utf-8"))
    markdown = (report_dir / "report.md").read_text("utf-8")

    assert validate_report_dir(report_dir)["status"] == "ok"
    assert "stratified_calibration" in payload["sections"]
    assert "Stratified calibration" in markdown


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_invalid_group_weighting_fails(tmp_path) -> None:
    dataset_dir = _build_merged_panel_dataset(tmp_path)

    result = runner.invoke(
        app,
        [
            "train-neural",
            "--dataset-dir",
            str(dataset_dir),
            "--outdir",
            str(tmp_path / "bad_group_weighting"),
            "--group-weighting",
            "bad",
        ],
    )

    assert result.exit_code != 0
    assert "group_weighting must be one of" in result.output


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_invalid_sampler_fails(tmp_path) -> None:
    dataset_dir = _build_merged_panel_dataset(tmp_path)

    result = runner.invoke(
        app,
        [
            "train-neural",
            "--dataset-dir",
            str(dataset_dir),
            "--outdir",
            str(tmp_path / "bad_sampler"),
            "--sampler",
            "bad",
        ],
    )

    assert result.exit_code != 0
    assert "sampler must be one of" in result.output


def _build_merged_panel_dataset(tmp_path: Path) -> Path:
    panel_dir = tmp_path / "panel"
    build_saturation_panel(
        SaturationPanelConfig(
            outdir=str(panel_dir),
            n_families_per_tier=2,
            tiers=["low", "moderate"],
            n_taxa=4,
            n_codons=30,
            seed=42,
            positive_rate=0.5,
            methods=["identity", "codon_dropout"],
            dropout_rate=0.02,
        )
    )
    merged_dir = tmp_path / "merged"
    merge_dataset_indexes(
        DatasetMergeConfig(
            dataset_dirs=[
                str(panel_dir / "tiers" / "low" / "dataset"),
                str(panel_dir / "tiers" / "moderate" / "dataset"),
            ],
            names=["low", "moderate"],
            outdir=str(merged_dir),
            seed=42,
            resplit=True,
        )
    )
    return merged_dir


def _saturation_config(dataset_dir: Path, model_dir: Path) -> NeuralFullTrainConfig:
    return NeuralFullTrainConfig(
        dataset_dir=str(dataset_dir),
        outdir=str(model_dir),
        device="cpu",
        methods=["identity", "codon_dropout"],
        epochs=2,
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.0001,
        embedding_dim=16,
        hidden_dim=32,
        dropout=0.1,
        architecture="saturation_aware",
        saturation_embedding_dim=8,
        positive_class_weight="auto",
        group_weighting="saturation_inverse_frequency",
        sampler="saturation_balanced",
        max_train_items=8,
        max_val_items=4,
        max_calib_items=2,
        max_test_items=2,
        early_stopping_patience=2,
    )


def _write_stratified_predictions(path: Path) -> None:
    rows = [
        ("low_a", "calib", "low", 1, 0.92),
        ("low_b", "calib", "low", 0, 0.10),
        ("low_c", "test", "low", 1, 0.82),
        ("low_d", "test", "low", 0, 0.15),
        ("moderate_a", "calib", "moderate", 1, 0.85),
        ("moderate_b", "calib", "moderate", 0, 0.20),
        ("moderate_c", "test", "moderate", 1, 0.75),
        ("moderate_d", "test", "moderate", 0, 0.25),
    ]
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
        for family_id, split, tier, label, prob in rows:
            pred = int(prob >= 0.5)
            writer.writerow(
                {
                    "family_id": family_id,
                    "method": "identity",
                    "split": split,
                    "tensor_file": f"{family_id}.npz",
                    "gene_label": label,
                    "saturation_tier": tier,
                    "prob_positive": prob,
                    "pred_label": pred,
                    "correct": int(pred == label),
                }
            )


def _make_stratified_calibration(tmp_path: Path) -> Path:
    predictions = tmp_path / "predictions.tsv"
    calibration_dir = tmp_path / "stratified_calibration_for_report"
    _write_stratified_predictions(predictions)
    calibrate_by_group(
        StratifiedCalibrationConfig(
            predictions_tsv=str(predictions),
            outdir=str(calibration_dir),
            group_column="saturation_tier",
            probability_column="prob_positive",
            target_fdr=0.25,
            min_group_calib_n=2,
        )
    )
    return calibration_dir
