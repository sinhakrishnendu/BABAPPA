import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from babappa.benchmarks import (
    SaturationPanelConfig,
    StabilityBenchmarkConfig,
    build_saturation_panel,
    run_stability_benchmark,
    validate_stability_benchmark_dir,
)
from babappa.cli import app
from babappa.datasets import (
    DatasetMergeConfig,
    ResplitDatasetConfig,
    merge_dataset_indexes,
    read_tsv,
    resplit_dataset,
    validate_resplit_dataset_dir,
)
from babappa.models.baseline import get_default_feature_columns
from babappa.reports import (
    LeakageAuditConfig,
    audit_leakage,
    validate_leakage_audit_dir,
)
from babappa.training import safe_import_torch


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


def test_leakage_audit_detects_n_selected_sites(tmp_path) -> None:
    dataset_dir = _synthetic_dataset(tmp_path / "dataset", include_leakage=True)
    outdir = tmp_path / "leakage"

    summary = audit_leakage(
        LeakageAuditConfig(dataset_dir=str(dataset_dir), outdir=str(outdir))
    )
    payload = json.loads((outdir / "leakage_audit.json").read_text("utf-8"))

    assert summary["status"] == "ok"
    assert payload["status"] == "warning"
    assert "n_selected_sites" in payload["recommended_excluded_columns"]
    assert validate_leakage_audit_dir(outdir)["status"] == "ok"


def test_baseline_excludes_leakage_columns() -> None:
    rows = [
        {
            "n_selected_sites": "5",
            "selected_sites": "1,2,3",
            "truth_label": "1",
            "mean_site_codon_id_std": "0.25",
        }
    ]

    columns = get_default_feature_columns(rows)

    assert "mean_site_codon_id_std" in columns
    assert "n_selected_sites" not in columns
    assert "selected_sites" not in columns
    assert "truth_label" not in columns


def test_resplit_dataset_family_disjoint(tmp_path) -> None:
    dataset_dir = _synthetic_dataset(tmp_path / "dataset")
    outdir = tmp_path / "resplit"

    resplit_dataset(ResplitDatasetConfig(dataset_dir=str(dataset_dir), outdir=str(outdir), seed=43))
    summary = validate_resplit_dataset_dir(outdir)
    family_to_splits = {}
    for row in read_tsv(outdir / "splits.tsv"):
        family_to_splits.setdefault(row["family_id"], set()).add(row["split"])

    assert summary["status"] == "ok"
    assert all(len(splits) == 1 for splits in family_to_splits.values())


def test_cli_audit_leakage_exits_0(tmp_path) -> None:
    dataset_dir = _synthetic_dataset(tmp_path / "dataset", include_leakage=True)
    result = runner.invoke(
        app,
        [
            "audit-leakage",
            "--dataset-dir",
            str(dataset_dir),
            "--outdir",
            str(tmp_path / "leakage_cli"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Leakage Audit" in result.output


def test_cli_resplit_dataset_exits_0(tmp_path) -> None:
    dataset_dir = _synthetic_dataset(tmp_path / "dataset")
    result = runner.invoke(
        app,
        [
            "resplit-dataset",
            "--dataset-dir",
            str(dataset_dir),
            "--outdir",
            str(tmp_path / "resplit_cli"),
            "--seed",
            "43",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Dataset Resplit" in result.output


def test_stability_benchmark_structure_no_training(tmp_path) -> None:
    dataset_dir = _synthetic_dataset(tmp_path / "dataset")
    outdir = tmp_path / "stability"

    summary = run_stability_benchmark(
        StabilityBenchmarkConfig(
            dataset_dir=str(dataset_dir),
            outdir=str(outdir),
            seeds=[42, 43],
            presets=["contrastive_v2"],
            run_training=False,
        )
    )

    assert summary["status"] == "ok"
    assert (outdir / "stability_benchmark.json").exists()
    assert (outdir / "stability_results.tsv").exists()
    assert validate_stability_benchmark_dir(outdir)["status"] == "ok"


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_stability_benchmark_tiny_if_torch_available(tmp_path) -> None:
    dataset_dir = _build_merged_panel_dataset(tmp_path)
    outdir = tmp_path / "stability_torch"

    summary = run_stability_benchmark(
        StabilityBenchmarkConfig(
            dataset_dir=str(dataset_dir),
            outdir=str(outdir),
            seeds=[42],
            presets=["contrastive_v2"],
            methods=["identity", "codon_dropout"],
            device="cpu",
            epochs=1,
            batch_size=4,
            max_train_items=8,
            max_val_items=4,
            max_calib_items=2,
            max_test_items=2,
        )
    )

    assert summary["status"] == "ok"
    assert validate_stability_benchmark_dir(outdir)["status"] == "ok"


def _synthetic_dataset(dataset_dir: Path, include_leakage: bool = False) -> Path:
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "dataset_index.json").write_text(
        json.dumps({"n_rows": 8, "n_families": 4}) + "\n",
        encoding="utf-8",
    )
    rows = [
        ("a", "identity", "a.npz", 1, "low", "train", 0.8, 3),
        ("a", "codon_dropout", "a2.npz", 1, "low", "train", 0.7, 3),
        ("b", "identity", "b.npz", 0, "low", "val", 0.2, 0),
        ("b", "codon_dropout", "b2.npz", 0, "low", "val", 0.3, 0),
        ("c", "identity", "c.npz", 1, "high", "calib", 0.9, 2),
        ("c", "codon_dropout", "c2.npz", 1, "high", "calib", 0.8, 2),
        ("d", "identity", "d.npz", 0, "high", "test", 0.1, 0),
        ("d", "codon_dropout", "d2.npz", 0, "high", "test", 0.2, 0),
    ]
    feature_fields = [
        "family_id",
        "method",
        "tensor_file",
        "gene_label",
        "saturation_tier",
        "mean_site_codon_id_std",
    ]
    if include_leakage:
        feature_fields.extend(["n_selected_sites", "truth_label"])
    with (dataset_dir / "features.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=feature_fields, delimiter="\t")
        writer.writeheader()
        for family, method, tensor, label, tier, _split, signal, selected in rows:
            row = {
                "family_id": family,
                "method": method,
                "tensor_file": tensor,
                "gene_label": label,
                "saturation_tier": tier,
                "mean_site_codon_id_std": signal,
            }
            if include_leakage:
                row["n_selected_sites"] = selected
                row["truth_label"] = label
            writer.writerow(row)
    with (dataset_dir / "splits.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family_id",
                "method",
                "tensor_file",
                "gene_label",
                "saturation_tier",
                "split",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for family, method, tensor, label, tier, split, *_rest in rows:
            writer.writerow(
                {
                    "family_id": family,
                    "method": method,
                    "tensor_file": tensor,
                    "gene_label": label,
                    "saturation_tier": tier,
                    "split": split,
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
