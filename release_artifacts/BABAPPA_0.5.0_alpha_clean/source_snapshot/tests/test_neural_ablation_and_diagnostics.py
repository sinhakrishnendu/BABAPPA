import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from babappa.benchmarks import SaturationPanelConfig, build_saturation_panel
from babappa.calibration import ThresholdPolicyConfig, build_threshold_policy
from babappa.cli import app
from babappa.datasets import DatasetMergeConfig, merge_dataset_indexes
from babappa.reports import (
    AblationCompareConfig,
    NeuralDiagnosticsConfig,
    compare_neural_ablations,
    diagnose_neural_run,
    validate_ablation_comparison_dir,
    validate_neural_diagnostics_dir,
)
from babappa.training import (
    NeuralFullTrainConfig,
    apply_training_preset,
    safe_import_torch,
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


def test_apply_training_preset(tmp_path) -> None:
    dataset_dir = _minimal_dataset_dir(tmp_path)
    base = NeuralFullTrainConfig(
        dataset_dir=str(dataset_dir),
        outdir=str(tmp_path / "model"),
    )

    v2 = apply_training_preset(base, "contrastive_v2")
    full = apply_training_preset(base, "saturation_full_v3")

    assert v2.architecture == "contrastive"
    assert v2.positive_class_weight == "auto"
    assert v2.group_weighting == "none"
    assert v2.sampler == "none"
    assert full.architecture == "saturation_aware"
    assert full.group_weighting == "saturation_inverse_frequency"
    assert full.sampler == "saturation_balanced"


def test_neural_diagnostics_on_synthetic_predictions(tmp_path) -> None:
    model_dir = _fake_model_dir(tmp_path / "fake_model", constant_probs=True)
    outdir = tmp_path / "diag"

    summary = diagnose_neural_run(
        NeuralDiagnosticsConfig(
            model_dir=str(model_dir),
            outdir=str(outdir),
            model_name="fake_model",
        )
    )

    assert summary["status"] == "ok"
    assert validate_neural_diagnostics_dir(outdir)["status"] == "ok"
    assert any("probability_collapse" in warning for warning in summary["warnings"])


def test_threshold_policy_degeneracy_warning(tmp_path) -> None:
    predictions = tmp_path / "degenerate_predictions.tsv"
    _write_prediction_rows(
        predictions,
        [
            ("a", "calib", 1, 0.1),
            ("b", "calib", 1, 0.1),
            ("c", "calib", 1, 0.1),
            ("d", "calib", 1, 0.1),
        ],
    )
    outdir = tmp_path / "policy"

    summary = build_threshold_policy(
        ThresholdPolicyConfig(
            predictions_tsv=str(predictions),
            outdir=str(outdir),
            selection_split="calib",
            threshold_grid_size=11,
        )
    )
    payload = json.loads((outdir / "threshold_profiles.json").read_text("utf-8"))
    max_f1_warnings = payload["profiles"]["max_f1"].get("warnings", [])

    assert summary["status"] == "ok"
    assert (
        "selected_all_or_nearly_all_positive" in max_f1_warnings
        or "selected_boundary_threshold" in max_f1_warnings
    )


def test_ablation_compare_minimal(tmp_path) -> None:
    model_a = _fake_model_dir(tmp_path / "model_a", auroc=0.60)
    model_b = _fake_model_dir(tmp_path / "model_b", auroc=0.55)
    outdir = tmp_path / "compare"

    summary = compare_neural_ablations(
        AblationCompareConfig(
            outdir=str(outdir),
            model_dirs=[str(model_a), str(model_b)],
            names=["model_a", "model_b"],
        )
    )

    assert summary["status"] == "ok"
    assert validate_ablation_comparison_dir(outdir)["status"] == "ok"
    assert summary["recommendation"]["best_model"] == "model_a"


def test_cli_diagnose_neural_exits_0(tmp_path) -> None:
    model_dir = _fake_model_dir(tmp_path / "fake_model")

    result = runner.invoke(
        app,
        [
            "diagnose-neural",
            "--model-dir",
            str(model_dir),
            "--outdir",
            str(tmp_path / "diag_cli"),
            "--model-name",
            "fake_model",
        ],
    )

    assert result.exit_code == 0
    assert "Neural Diagnostics" in result.output


def test_cli_compare_ablations_exits_0(tmp_path) -> None:
    model_a = _fake_model_dir(tmp_path / "model_a", auroc=0.60)
    model_b = _fake_model_dir(tmp_path / "model_b", auroc=0.55)

    result = runner.invoke(
        app,
        [
            "compare-ablations",
            "--outdir",
            str(tmp_path / "compare_cli"),
            "--model-dirs",
            f"{model_a},{model_b}",
            "--names",
            "model_a,model_b",
        ],
    )

    assert result.exit_code == 0
    assert "Ablation Comparison" in result.output


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_tiny_training_ablation_compare_if_torch_available(tmp_path) -> None:
    dataset_dir = _build_merged_panel_dataset(tmp_path)
    contrastive_dir = tmp_path / "contrastive"
    embed_dir = tmp_path / "embed"

    train_neural_model(
        _tiny_train_config(dataset_dir, contrastive_dir, "contrastive_v2")
    )
    train_neural_model(
        _tiny_train_config(dataset_dir, embed_dir, "saturation_embed_only")
    )
    summary = compare_neural_ablations(
        AblationCompareConfig(
            outdir=str(tmp_path / "tiny_compare"),
            model_dirs=[str(contrastive_dir), str(embed_dir)],
            names=["contrastive", "embed_only"],
        )
    )

    assert summary["status"] == "ok"


def _minimal_dataset_dir(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "dataset_index.json").write_text(
        json.dumps({"n_rows": 0, "n_families": 0}) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "splits.tsv").write_text(
        "family_id\tmethod\tsplit\ttensor_file\tgene_label\tsaturation_tier\n",
        encoding="utf-8",
    )
    return dataset_dir


def _fake_model_dir(
    model_dir: Path, constant_probs: bool = False, auroc: float = 0.55
) -> Path:
    (model_dir / "predictions").mkdir(parents=True)
    (model_dir / "logs").mkdir()
    _write_prediction_rows(
        model_dir / "predictions" / "neural_predictions.tsv",
        [
            ("a", "train", 1, 0.49 if constant_probs else 0.80),
            ("b", "train", 0, 0.49 if constant_probs else 0.20),
            ("c", "val", 1, 0.49 if constant_probs else 0.70),
            ("d", "val", 0, 0.49 if constant_probs else 0.30),
        ],
    )
    (model_dir / "logs" / "neural_training_history.tsv").write_text(
        "epoch\ttrain_loss\ttrain_accuracy\ttrain_auroc\tval_loss\tval_accuracy\tval_auroc\tmonitor_metric\tmonitor_value\tis_best\tseconds_elapsed\n"
        f"1\t0.8\t0.5\t0.5\t0.7\t0.5\t{auroc}\tval_loss\t0.7\t1\t0.1\n"
        f"2\t0.6\t0.5\t0.5\t0.8\t0.5\t{auroc}\tval_loss\t0.8\t0\t0.2\n",
        encoding="utf-8",
    )
    meta = {
        "architecture": "contrastive",
        "training_preset": "contrastive_v2",
        "group_weighting": "none",
        "sampler": "none",
        "positive_class_weight": "auto",
        "monitor_metric": "val_loss",
        "best_epoch": 1,
        "epochs_completed": 2,
        "stopped_early": False,
    }
    (model_dir / "neural_model_meta.json").write_text(
        json.dumps(meta) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "metrics_by_split": {
            split: {
                "n": 2,
                "accuracy": 0.5,
                "auroc": auroc,
                "f1": 0.5,
                "mcc": 0.0,
                "precision": 0.5,
                "recall": 0.5,
                "specificity": 0.5,
            }
            for split in ["train", "val", "calib", "test", "all"]
        }
    }
    (model_dir / "neural_metrics.json").write_text(
        json.dumps(metrics) + "\n",
        encoding="utf-8",
    )
    return model_dir


def _write_prediction_rows(path: Path, rows) -> None:
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


def _tiny_train_config(
    dataset_dir: Path, model_dir: Path, preset: str
) -> NeuralFullTrainConfig:
    return NeuralFullTrainConfig(
        dataset_dir=str(dataset_dir),
        outdir=str(model_dir),
        device="cpu",
        methods=["identity", "codon_dropout"],
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.0001,
        embedding_dim=16,
        hidden_dim=32,
        dropout=0.1,
        training_preset=preset,
        max_train_items=8,
        max_val_items=4,
        max_calib_items=2,
        max_test_items=2,
        early_stopping_patience=1,
    )
