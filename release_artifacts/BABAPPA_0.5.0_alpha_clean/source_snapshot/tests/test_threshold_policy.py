import csv
import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from babappa.calibration import (
    ThresholdPolicyConfig,
    build_threshold_policy,
    validate_threshold_policy_dir,
)
from babappa.calibration.threshold_policy import threshold_metrics
from babappa.cli import app
from babappa.reports import ReportConfig, RunSummaryConfig, generate_report, summarize_run


runner = CliRunner()


def test_threshold_policy_on_synthetic_predictions(tmp_path) -> None:
    predictions = tmp_path / "predictions.tsv"
    _write_predictions(
        predictions,
        [
            ("family_001", "train", 1, 0.90),
            ("family_002", "train", 0, 0.20),
            ("family_003", "val", 1, 0.80),
            ("family_004", "val", 0, 0.30),
            ("family_005", "calib", 1, 0.95),
            ("family_006", "calib", 1, 0.88),
            ("family_007", "calib", 0, 0.15),
            ("family_008", "calib", 0, 0.10),
            ("family_009", "test", 1, 0.82),
            ("family_010", "test", 0, 0.25),
        ],
    )
    outdir = tmp_path / "policy"

    summary = build_threshold_policy(
        ThresholdPolicyConfig(
            predictions_tsv=str(predictions),
            outdir=str(outdir),
            model_name="synthetic",
            threshold_grid_size=101,
        )
    )

    assert summary["status"] == "ok"
    assert (outdir / "threshold_profiles.json").exists()
    assert (outdir / "threshold_profiles.tsv").exists()
    assert (outdir / "threshold_profile_metrics.tsv").exists()
    assert (outdir / "threshold_policy_curve.tsv").exists()
    assert (outdir / "threshold_policy.md").exists()
    assert validate_threshold_policy_dir(outdir)["status"] == "ok"


def test_threshold_policy_profiles_exist(tmp_path) -> None:
    outdir = _build_simple_policy(tmp_path)
    payload = json.loads((outdir / "threshold_profiles.json").read_text("utf-8"))

    assert set(payload["profiles"]) == {
        "default_0_5",
        "strict_fdr",
        "max_f1",
        "max_mcc",
        "balanced_youden",
        "high_precision",
        "high_recall",
    }


def test_threshold_policy_handles_no_fdr_threshold(tmp_path) -> None:
    predictions = tmp_path / "weak_predictions.tsv"
    _write_predictions(
        predictions,
        [
            ("family_001", "calib", 0, 0.90),
            ("family_002", "calib", 0, 0.80),
            ("family_003", "calib", 1, 0.20),
            ("family_004", "calib", 1, 0.10),
        ],
    )
    outdir = tmp_path / "weak_policy"

    summary = build_threshold_policy(
        ThresholdPolicyConfig(
            predictions_tsv=str(predictions),
            outdir=str(outdir),
            target_fdr=0.10,
            threshold_grid_size=101,
        )
    )

    assert "no_threshold_met_target_fdr" in summary["warnings"]


def test_threshold_metrics_large_mcc_denominator_does_not_crash() -> None:
    y_true = np.tile(np.array([1, 0, 1, 0], dtype=np.int32), 50_000)
    probs = np.tile(np.array([0.90, 0.80, 0.20, 0.10], dtype=np.float64), 50_000)

    metrics = threshold_metrics(y_true, probs, 0.5)

    assert "mcc" in metrics
    assert metrics["mcc"] is not None


def test_cli_threshold_policy_exits_0(tmp_path) -> None:
    predictions = tmp_path / "predictions.tsv"
    _write_predictions(
        predictions,
        [
            ("family_001", "calib", 1, 0.90),
            ("family_002", "calib", 0, 0.10),
        ],
    )

    result = runner.invoke(
        app,
        [
            "threshold-policy",
            "--predictions",
            str(predictions),
            "--outdir",
            str(tmp_path / "policy_cli"),
            "--model-name",
            "cli_model",
            "--threshold-grid-size",
            "51",
        ],
    )

    assert result.exit_code == 0
    assert "Threshold Policy" in result.output


def test_cli_validate_threshold_policy_exits_0(tmp_path) -> None:
    outdir = _build_simple_policy(tmp_path)

    result = runner.invoke(
        app,
        [
            "validate-threshold-policy",
            "--policy-dir",
            str(outdir),
        ],
    )

    assert result.exit_code == 0
    assert "Status" in result.output
    assert "ok" in result.output


def test_invalid_selection_split_fails(tmp_path) -> None:
    predictions = tmp_path / "predictions.tsv"
    _write_predictions(predictions, [("family_001", "calib", 1, 0.90)])

    result = runner.invoke(
        app,
        [
            "threshold-policy",
            "--predictions",
            str(predictions),
            "--outdir",
            str(tmp_path / "bad_policy"),
            "--selection-split",
            "bad_split",
        ],
    )

    assert result.exit_code != 0
    assert "selection_split must be one of" in result.output


def test_run_summary_accepts_threshold_policy_dir(tmp_path) -> None:
    policy_dir = _build_simple_policy(tmp_path)
    summary_dir = tmp_path / "summary"

    summarize_run(
        RunSummaryConfig(
            outdir=str(summary_dir),
            threshold_policy_dir=str(policy_dir),
            title="Policy-only summary",
        )
    )
    payload = json.loads((summary_dir / "run_summary.json").read_text("utf-8"))

    assert payload["status_overview"]["threshold_policy_present"] is True
    assert payload["threshold_policy_overview"]["selected_profiles"]
    assert "Threshold policy overview" in (
        summary_dir / "run_summary.md"
    ).read_text("utf-8")


def test_report_accepts_threshold_policy_dir(tmp_path) -> None:
    policy_dir = _build_simple_policy(tmp_path)
    report_dir = tmp_path / "report"

    generate_report(
        ReportConfig(
            outdir=str(report_dir),
            threshold_policy_dir=str(policy_dir),
            title="Policy report",
        )
    )
    payload = json.loads((report_dir / "report_summary.json").read_text("utf-8"))
    markdown = (report_dir / "report.md").read_text("utf-8")

    assert "threshold_policy" in payload["sections"]
    assert "Threshold policy" in markdown


def test_repository_hardening_init_files() -> None:
    root = Path(__file__).resolve().parents[1]

    assert not (root / "src" / "babappa" / "init.py").exists()
    assert not (root / "src" / "babappa" / "calibration" / "init.py").exists()
    assert not (root / "src" / "babappa" / "reports" / "init.py").exists()
    assert not (root / "src" / "babappa" / "training" / "init.py").exists()
    assert (root / "src" / "babappa" / "__init__.py").exists()
    assert (root / "src" / "babappa" / "calibration" / "__init__.py").exists()
    assert (root / "src" / "babappa" / "reports" / "__init__.py").exists()
    assert (root / "src" / "babappa" / "training" / "__init__.py").exists()


def _build_simple_policy(tmp_path) -> Path:
    predictions = tmp_path / "predictions.tsv"
    _write_predictions(
        predictions,
        [
            ("family_001", "train", 1, 0.90),
            ("family_002", "train", 0, 0.20),
            ("family_003", "calib", 1, 0.85),
            ("family_004", "calib", 0, 0.15),
            ("family_005", "test", 1, 0.80),
            ("family_006", "test", 0, 0.25),
        ],
    )
    outdir = tmp_path / "policy"
    build_threshold_policy(
        ThresholdPolicyConfig(
            predictions_tsv=str(predictions),
            outdir=str(outdir),
            model_name="simple",
            threshold_grid_size=101,
        )
    )
    return outdir


def _write_predictions(path: Path, rows) -> None:
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
