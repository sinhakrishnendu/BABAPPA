import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from babappa.benchmarks import SaturationPanelConfig, build_saturation_panel
from babappa.cli import app
from babappa.datasets import DatasetMergeConfig, merge_dataset_indexes
from babappa.reports import (
    AblationCompareConfig,
    LabelSignalAuditConfig,
    audit_label_signal,
    compare_neural_ablations,
    validate_ablation_comparison_dir,
    validate_label_signal_audit_dir,
)
from babappa.training import (
    NeuralFullTrainConfig,
    SiteAttentionGeneClassifier,
    safe_import_torch,
    train_neural_model,
)
from babappa.training.losses import pairwise_rank_loss


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


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_pairwise_rank_loss_if_torch_available() -> None:
    logits = torch.tensor([0.2, 0.8, -0.1], dtype=torch.float32)
    y = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    loss = pairwise_rank_loss(logits, y)

    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_site_attention_forward_if_torch_available() -> None:
    model = SiteAttentionGeneClassifier(
        vocab_size=128,
        embedding_dim=16,
        hidden_dim=32,
        dropout=0.1,
    )
    X = torch.randint(0, 64, (4, 6, 60, 2)).float()

    logits = model(X)

    assert list(logits.shape) == [4]


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_train_neural_ranking_tiny_if_torch_available(tmp_path) -> None:
    dataset_dir = _build_merged_panel_dataset(tmp_path)
    model_dir = tmp_path / "ranking_model"

    train_neural_model(
        NeuralFullTrainConfig(
            dataset_dir=str(dataset_dir),
            outdir=str(model_dir),
            device="cpu",
            methods=["identity", "codon_dropout"],
            epochs=2,
            batch_size=4,
            embedding_dim=16,
            hidden_dim=32,
            training_preset="site_attention_ranked",
            max_train_items=8,
            max_val_items=4,
            max_calib_items=2,
            max_test_items=2,
            early_stopping_patience=2,
        )
    )
    meta = json.loads((model_dir / "neural_model_meta.json").read_text("utf-8"))

    assert (model_dir / "checkpoints" / "best_model.pt").exists()
    assert meta["architecture"] == "site_attention"
    assert meta["loss_mode"] == "bce_rank"


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_cli_train_neural_ranking_if_torch_available(tmp_path) -> None:
    dataset_dir = _build_merged_panel_dataset(tmp_path)
    result = runner.invoke(
        app,
        [
            "train-neural-ranking",
            "--dataset-dir",
            str(dataset_dir),
            "--outdir",
            str(tmp_path / "ranking_cli"),
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
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Neural Training Summary" in result.output


def test_label_signal_audit_on_tiny_dataset(tmp_path) -> None:
    dataset_dir = _synthetic_dataset(tmp_path / "dataset")
    outdir = tmp_path / "label_signal"

    summary = audit_label_signal(
        LabelSignalAuditConfig(dataset_dir=str(dataset_dir), outdir=str(outdir))
    )
    payload = json.loads((outdir / "label_signal_audit.json").read_text("utf-8"))

    assert summary["status"] == "ok"
    assert validate_label_signal_audit_dir(outdir)["status"] == "ok"
    assert payload["top_features_by_auroc_distance"]


def test_cli_audit_label_signal_exits_0(tmp_path) -> None:
    dataset_dir = _synthetic_dataset(tmp_path / "dataset")
    result = runner.invoke(
        app,
        [
            "audit-label-signal",
            "--dataset-dir",
            str(dataset_dir),
            "--outdir",
            str(tmp_path / "label_signal_cli"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Label-Signal Audit" in result.output


def test_invalid_loss_mode_fails(tmp_path) -> None:
    dataset_dir = _synthetic_dataset(tmp_path / "dataset")

    with pytest.raises(ValueError, match="loss_mode"):
        NeuralFullTrainConfig(
            dataset_dir=str(dataset_dir),
            outdir=str(tmp_path / "model"),
            loss_mode="bad_loss",
        )


def test_ablation_compare_accepts_diagnostics_dirs(tmp_path) -> None:
    model_a = _fake_model_dir(tmp_path / "model_a", auroc=0.6)
    model_b = _fake_model_dir(tmp_path / "model_b", auroc=0.55)
    diag_a = _fake_diagnostics_dir(tmp_path / "diag_a", "model_a", std=0.01)
    diag_b = _fake_diagnostics_dir(tmp_path / "diag_b", "model_b", std=0.04)
    outdir = tmp_path / "compare"

    summary = compare_neural_ablations(
        AblationCompareConfig(
            outdir=str(outdir),
            model_dirs=[str(model_a), str(model_b)],
            names=["model_a", "model_b"],
            neural_diagnostics_dirs=[str(diag_a), str(diag_b)],
        )
    )
    payload = json.loads((outdir / "ablation_comparison.json").read_text("utf-8"))

    assert summary["status"] == "ok"
    assert validate_ablation_comparison_dir(outdir)["status"] == "ok"
    assert payload["models"][0]["diagnostics"]["probability_std_all"] == 0.01


def _synthetic_dataset(dataset_dir: Path) -> Path:
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "dataset_index.json").write_text(
        json.dumps({"n_rows": 4, "n_families": 4}) + "\n",
        encoding="utf-8",
    )
    rows = [
        ("a", "identity", "a.npz", 1, "low", 0.9, 0.8),
        ("b", "identity", "b.npz", 0, "low", 0.1, 0.2),
        ("c", "codon_dropout", "c.npz", 1, "high", 0.8, 0.7),
        ("d", "codon_dropout", "d.npz", 0, "high", 0.2, 0.3),
    ]
    with (dataset_dir / "features.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=[
                "family_id",
                "method",
                "tensor_file",
                "gene_label",
                "saturation_tier",
                "signal_feature",
                "weak_feature",
            ],
        )
        writer.writeheader()
        for family_id, method, tensor, label, tier, signal, weak in rows:
            writer.writerow(
                {
                    "family_id": family_id,
                    "method": method,
                    "tensor_file": tensor,
                    "gene_label": label,
                    "saturation_tier": tier,
                    "signal_feature": signal,
                    "weak_feature": weak,
                }
            )
    with (dataset_dir / "splits.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=[
                "family_id",
                "method",
                "tensor_file",
                "gene_label",
                "saturation_tier",
                "split",
            ],
        )
        writer.writeheader()
        for index, (family_id, method, tensor, label, tier, *_rest) in enumerate(rows):
            writer.writerow(
                {
                    "family_id": family_id,
                    "method": method,
                    "tensor_file": tensor,
                    "gene_label": label,
                    "saturation_tier": tier,
                    "split": "train" if index < 2 else "val",
                }
            )
    return dataset_dir


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


def _fake_model_dir(model_dir: Path, auroc: float = 0.55) -> Path:
    model_dir.mkdir(parents=True)
    (model_dir / "neural_model_meta.json").write_text(
        json.dumps(
            {
                "architecture": "contrastive",
                "training_preset": "contrastive_v2",
                "group_weighting": "none",
                "sampler": "none",
                "positive_class_weight": "auto",
                "best_epoch": 1,
                "epochs_completed": 2,
                "stopped_early": False,
            }
        )
        + "\n",
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


def _fake_diagnostics_dir(diag_dir: Path, model_name: str, std: float) -> Path:
    diag_dir.mkdir(parents=True)
    (diag_dir / "neural_diagnostics.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "probability_summary_by_split": {
                    "all": {
                        "prob_std": std,
                        "separation": 0.05,
                        "fraction_ge_0_5": 0.5,
                    }
                },
                "warnings": ["all:probability_collapse"] if std < 0.02 else [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return diag_dir
