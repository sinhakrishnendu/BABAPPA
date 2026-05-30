import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from babappa.cli import app
from babappa.site import (
    AggregationThresholdPolicyConfig,
    SiteAggregationControlConfig,
    SiteCalibrationCompareConfig,
    SiteCalibrationConfig,
    SiteModelCompareConfig,
    SiteStabilityConfig,
    build_aggregation_threshold_policy,
    calibrate_site_model,
    compare_site_calibrations,
    compare_site_models,
    run_site_aggregation_controls,
    run_site_stability_benchmark,
    validate_aggregation_threshold_policy_dir,
    validate_site_aggregation_controls_dir,
    validate_site_calibration_comparison_dir,
    validate_site_calibration_dir,
    validate_site_model_comparison_dir,
    validate_site_stability_dir,
)
from babappa.training.neural_env import safe_import_torch


runner = CliRunner()


def test_repository_hardening_init_files() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in [
        root / "src" / "babappa" / "init.py",
        root / "src" / "babappa" / "calibration" / "init.py",
        root / "src" / "babappa" / "reports" / "init.py",
        root / "src" / "babappa" / "training" / "init.py",
        root / "src" / "babappa" / "benchmarks" / "init.py",
        root / "src" / "babappa" / "datasets" / "init.py",
        root / "src" / "babappa" / "site" / "init.py",
    ]:
        assert not path.exists()
    for path in [
        root / "src" / "babappa" / "__init__.py",
        root / "src" / "babappa" / "calibration" / "__init__.py",
        root / "src" / "babappa" / "reports" / "__init__.py",
        root / "src" / "babappa" / "training" / "__init__.py",
        root / "src" / "babappa" / "benchmarks" / "__init__.py",
        root / "src" / "babappa" / "datasets" / "__init__.py",
        root / "src" / "babappa" / "site" / "__init__.py",
    ]:
        assert path.exists()


def test_compare_site_models_synthetic(tmp_path) -> None:
    baseline_dir, neural_dir = _write_model_metric_dirs(tmp_path)
    outdir = tmp_path / "compare"

    summary = compare_site_models(
        SiteModelCompareConfig(
            outdir=str(outdir),
            site_baseline_dir=str(baseline_dir),
            site_neural_dir=str(neural_dir),
        )
    )

    assert summary["status"] == "ok"
    assert validate_site_model_comparison_dir(outdir)["status"] == "ok"


def test_aggregation_controls_synthetic(tmp_path) -> None:
    predictions, gene_dataset = _write_site_predictions_and_gene_dataset(tmp_path)
    outdir = tmp_path / "controls"

    summary = run_site_aggregation_controls(
        SiteAggregationControlConfig(
            predictions_tsv=str(predictions),
            gene_dataset_dir=str(gene_dataset),
            outdir=str(outdir),
            n_permutations=3,
            seed=7,
        )
    )

    assert summary["status"] == "ok"
    assert validate_site_aggregation_controls_dir(outdir)["status"] == "ok"
    payload = json.loads((outdir / "site_aggregation_controls.json").read_text())
    assert payload["controls"]["random_uniform_probabilities"]["empirical_p_value"] is not None


def test_aggregation_threshold_policy_synthetic(tmp_path) -> None:
    predictions = tmp_path / "site_to_gene_predictions.tsv"
    predictions.write_text(
        "\n".join(
            [
                "family_id\tmethod\tsplit\tsaturation_tier\tgene_label\tmax_site_probability",
                "f1\tidentity\tcalib\tlow\t1\t0.95",
                "f2\tidentity\tcalib\tlow\t0\t0.10",
                "f3\tidentity\ttest\thigh\t1\t0.85",
                "f4\tidentity\ttest\thigh\t0\t0.20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outdir = tmp_path / "agg_policy"

    summary = build_aggregation_threshold_policy(
        AggregationThresholdPolicyConfig(
            predictions_tsv=str(predictions),
            outdir=str(outdir),
            threshold_grid_size=51,
        )
    )

    assert summary["status"] == "ok"
    assert validate_aggregation_threshold_policy_dir(outdir)["status"] == "ok"


def test_quantile_calibration_synthetic(tmp_path) -> None:
    model_dir = _write_site_neural_predictions(tmp_path / "model")
    outdir = tmp_path / "quantile"

    summary = calibrate_site_model(
        SiteCalibrationConfig(
            model_dir=str(model_dir),
            outdir=str(outdir),
            calibration_method="quantile",
            n_bins=3,
            threshold_grid_size=51,
        )
    )

    assert summary["status"] == "ok"
    assert validate_site_calibration_dir(outdir)["status"] == "ok"
    payload = json.loads((outdir / "site_calibration.json").read_text())
    assert payload["calibration_method"] == "quantile"
    assert payload["quantile_mapping"]["n_bins"] >= 1


def test_site_calibration_comparison_synthetic(tmp_path) -> None:
    model_dir = _write_site_neural_predictions(tmp_path / "model")
    temp_dir = tmp_path / "temp"
    quant_dir = tmp_path / "quant"
    calibrate_site_model(SiteCalibrationConfig(model_dir=str(model_dir), outdir=str(temp_dir)))
    calibrate_site_model(
        SiteCalibrationConfig(
            model_dir=str(model_dir),
            outdir=str(quant_dir),
            calibration_method="quantile",
            n_bins=3,
        )
    )
    outdir = tmp_path / "compare_cal"

    summary = compare_site_calibrations(
        SiteCalibrationCompareConfig(
            calibration_dirs=[str(temp_dir), str(quant_dir)],
            names=["temperature", "quantile"],
            outdir=str(outdir),
        )
    )

    assert summary["status"] == "ok"
    assert validate_site_calibration_comparison_dir(outdir)["status"] == "ok"


def test_site_stability_benchmark_structure_no_training(tmp_path) -> None:
    dataset_dir = _write_tiny_site_dataset(tmp_path / "site_dataset")
    outdir = tmp_path / "stability"

    summary = run_site_stability_benchmark(
        SiteStabilityConfig(
            site_dataset_dir=str(dataset_dir),
            outdir=str(outdir),
            seeds=[42, 43],
            run_training=False,
        )
    )

    assert summary["status"] == "ok"
    assert validate_site_stability_dir(outdir)["status"] == "ok"


def test_tiny_site_stability_if_torch_available(tmp_path) -> None:
    torch, _error = safe_import_torch()
    if torch is None:
        pytest.skip("torch unavailable")
    dataset_dir = _write_tiny_site_dataset(tmp_path / "site_dataset")
    outdir = tmp_path / "stability_train"

    summary = run_site_stability_benchmark(
        SiteStabilityConfig(
            site_dataset_dir=str(dataset_dir),
            outdir=str(outdir),
            seeds=[42],
            device="cpu",
            epochs=1,
            batch_size=4,
            max_train_items=8,
            max_val_items=4,
            max_calib_items=4,
            max_test_items=4,
        )
    )

    assert summary["status"] == "ok"
    assert validate_site_stability_dir(outdir)["status"] == "ok"


def test_cli_compare_site_models_exits_0(tmp_path) -> None:
    baseline_dir, neural_dir = _write_model_metric_dirs(tmp_path)
    result = runner.invoke(
        app,
        [
            "compare-site-models",
            "--site-baseline-dir",
            str(baseline_dir),
            "--site-neural-dir",
            str(neural_dir),
            "--outdir",
            str(tmp_path / "cli_compare"),
        ],
    )
    assert result.exit_code == 0


def test_cli_aggregation_threshold_policy_exits_0(tmp_path) -> None:
    predictions = tmp_path / "site_to_gene_predictions.tsv"
    predictions.write_text(
        "family_id\tmethod\tsplit\tsaturation_tier\tgene_label\tmax_site_probability\n"
        "f1\tidentity\tcalib\tlow\t1\t0.9\n"
        "f2\tidentity\tcalib\tlow\t0\t0.1\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "aggregation-threshold-policy",
            "--predictions",
            str(predictions),
            "--outdir",
            str(tmp_path / "cli_policy"),
        ],
    )
    assert result.exit_code == 0


def _metric(n=10, auroc=0.8, f1=0.5):
    return {
        "n": n,
        "positives": n // 2,
        "negatives": n - n // 2,
        "accuracy": 0.7,
        "precision": 0.6,
        "recall": 0.7,
        "specificity": 0.7,
        "f1": f1,
        "mcc": 0.4,
        "auroc": auroc,
    }


def _write_model_metric_dirs(tmp_path):
    baseline = tmp_path / "baseline"
    neural = tmp_path / "neural"
    baseline.mkdir()
    neural.mkdir()
    (baseline / "site_baseline_metrics.json").write_text(
        json.dumps(
            {
                "metrics_by_split": {"all": _metric(auroc=0.8), "val": _metric(auroc=0.78)},
                "metrics_by_saturation_tier": {"low": _metric(auroc=0.82)},
                "metrics_by_method": {"identity": _metric(auroc=0.81)},
            }
        ),
        encoding="utf-8",
    )
    (neural / "site_neural_metrics.json").write_text(
        json.dumps(
            {
                "metrics_by_split": {"all": _metric(auroc=0.9, f1=0.6), "val": _metric(auroc=0.88)},
                "metrics_by_saturation_tier": {"low": _metric(auroc=0.92)},
                "metrics_by_method": {"identity": _metric(auroc=0.91)},
            }
        ),
        encoding="utf-8",
    )
    return baseline, neural


def _write_site_predictions_and_gene_dataset(tmp_path):
    predictions = tmp_path / "site_predictions.tsv"
    rows = [
        "site_id\tfamily_id\tmethod\tsaturation_tier\tsplit\tsite_index_zero\ty_site\tprob_positive",
    ]
    for family, label, split, high in [
        ("f1", 1, "train", True),
        ("f2", 0, "train", False),
        ("f3", 1, "calib", True),
        ("f4", 0, "calib", False),
        ("f5", 1, "test", True),
        ("f6", 0, "test", False),
    ]:
        for site in range(4):
            prob = 0.9 - site * 0.05 if high else 0.1 + site * 0.03
            rows.append(f"{family}::identity::site_{site}\t{family}\tidentity\tlow\t{split}\t{site}\t{int(label and site == 0)}\t{prob}")
    predictions.write_text("\n".join(rows) + "\n", encoding="utf-8")
    dataset = tmp_path / "gene_dataset"
    dataset.mkdir()
    split_rows = ["family_id\tmethod\tsplit\tsaturation_tier\tgene_label"]
    for family, label, split, _high in [
        ("f1", 1, "train", True),
        ("f2", 0, "train", False),
        ("f3", 1, "calib", True),
        ("f4", 0, "calib", False),
        ("f5", 1, "test", True),
        ("f6", 0, "test", False),
    ]:
        split_rows.append(f"{family}\tidentity\t{split}\tlow\t{label}")
    (dataset / "splits.tsv").write_text("\n".join(split_rows) + "\n", encoding="utf-8")
    return predictions, dataset


def _write_site_neural_predictions(model_dir: Path) -> Path:
    model_dir.mkdir()
    rows = [
        "site_id\tfamily_id\tmethod\tsaturation_tier\tsplit\tsite_index_zero\ty_site\tprob_positive\tpred_label\tcorrect",
        "s1\tf1\tidentity\tlow\ttrain\t0\t1\t0.90\t1\t1",
        "s2\tf1\tidentity\tlow\ttrain\t1\t0\t0.10\t0\t1",
        "s3\tf2\tidentity\tlow\tval\t0\t1\t0.80\t1\t1",
        "s4\tf2\tidentity\tlow\tval\t1\t0\t0.20\t0\t1",
        "s5\tf3\tidentity\tlow\tcalib\t0\t1\t0.95\t1\t1",
        "s6\tf3\tidentity\tlow\tcalib\t1\t0\t0.30\t0\t1",
        "s7\tf4\tidentity\tlow\tcalib\t0\t1\t0.70\t1\t1",
        "s8\tf4\tidentity\tlow\tcalib\t1\t0\t0.05\t0\t1",
        "s9\tf5\tidentity\tlow\ttest\t0\t1\t0.85\t1\t1",
        "s10\tf5\tidentity\tlow\ttest\t1\t0\t0.15\t0\t1",
    ]
    (model_dir / "site_neural_predictions.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return model_dir


def _write_tiny_site_dataset(dataset_dir: Path) -> Path:
    dataset_dir.mkdir()
    header = (
        "site_id\tfamily_id\tmethod\tsaturation_tier\tsplit\tsite_index_zero\ty_site\t"
        "site_relative_position\tcodon_id_mean\tcodon_id_std\tgap_fraction"
    )
    rows = [header]
    for split in ["train", "val", "calib", "test"]:
        for i in range(6):
            y = int(i % 3 == 0)
            rows.append(
                f"{split}_{i}\tf_{split}_{i}\tidentity\tlow\t{split}\t{i}\t{y}\t{i/10:.3f}\t{0.8 if y else 0.2}\t{0.1+i/100:.3f}\t0.0"
            )
    (dataset_dir / "site_features.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    splits = ["site_id\tfamily_id\tmethod\tsaturation_tier\tsplit\tsite_index_zero\ty_site"]
    for row in rows[1:]:
        values = row.split("\t")
        splits.append("\t".join(values[:7]))
    (dataset_dir / "site_splits.tsv").write_text("\n".join(splits) + "\n", encoding="utf-8")
    (dataset_dir / "site_dataset_index.json").write_text(
        json.dumps(
            {
                "site_dataset_version": "test",
                "dataset_dir": str(dataset_dir),
                "n_site_rows": 24,
                "split_counts": {"train": 6, "val": 6, "calib": 6, "test": 6},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    return dataset_dir
