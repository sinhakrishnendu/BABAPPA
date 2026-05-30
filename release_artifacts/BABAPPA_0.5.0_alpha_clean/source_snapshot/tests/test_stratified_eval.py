import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from babappa.calibration import ThresholdPolicyConfig, build_threshold_policy
from babappa.cli import app
from babappa.reports import (
    ReportConfig,
    RunSummaryConfig,
    StratifiedEvalConfig,
    generate_report,
    stratified_evaluate_predictions,
    summarize_run,
    validate_stratified_eval_dir,
)


runner = CliRunner()


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


def test_stratified_eval_on_synthetic_predictions(tmp_path) -> None:
    predictions = tmp_path / "predictions.tsv"
    _write_predictions(predictions)
    outdir = tmp_path / "stratified"

    summary = stratified_evaluate_predictions(
        StratifiedEvalConfig(
            predictions_tsv=str(predictions),
            outdir=str(outdir),
            model_name="synthetic",
        )
    )

    assert summary["status"] == "ok"
    assert (outdir / "stratified_eval.json").exists()
    assert (outdir / "stratified_metrics.tsv").exists()
    assert (outdir / "stratified_eval.md").exists()
    assert validate_stratified_eval_dir(outdir)["status"] == "ok"


def test_stratified_eval_contains_group_types(tmp_path) -> None:
    outdir = _build_stratified(tmp_path)
    rows = _read_tsv(outdir / "stratified_metrics.tsv")
    group_types = {row["group_type"] for row in rows}

    assert {
        "split",
        "saturation_tier",
        "method",
        "split_x_saturation",
        "split_x_method",
        "saturation_x_method",
        "split_x_saturation_x_method",
    }.issubset(group_types)


def test_stratified_eval_with_threshold_policy(tmp_path) -> None:
    predictions = tmp_path / "predictions.tsv"
    _write_predictions(predictions)
    policy_dir = tmp_path / "policy"
    build_threshold_policy(
        ThresholdPolicyConfig(
            predictions_tsv=str(predictions),
            outdir=str(policy_dir),
            threshold_grid_size=101,
        )
    )
    outdir = tmp_path / "stratified_policy"

    stratified_evaluate_predictions(
        StratifiedEvalConfig(
            predictions_tsv=str(predictions),
            outdir=str(outdir),
            threshold_policy_dir=str(policy_dir),
        )
    )
    profiles = {row["profile"] for row in _read_tsv(outdir / "stratified_metrics.tsv")}

    assert "default_0_5" in profiles
    assert "max_f1" in profiles
    assert len(profiles) > 1


def test_cli_stratified_eval_exits_0(tmp_path) -> None:
    predictions = tmp_path / "predictions.tsv"
    _write_predictions(predictions)

    result = runner.invoke(
        app,
        [
            "stratified-eval",
            "--predictions",
            str(predictions),
            "--outdir",
            str(tmp_path / "stratified_cli"),
            "--model-name",
            "cli_model",
        ],
    )

    assert result.exit_code == 0
    assert "Stratified Evaluation" in result.output


def test_cli_validate_stratified_eval_exits_0(tmp_path) -> None:
    outdir = _build_stratified(tmp_path)

    result = runner.invoke(
        app,
        [
            "validate-stratified-eval",
            "--eval-dir",
            str(outdir),
        ],
    )

    assert result.exit_code == 0
    assert "Status" in result.output
    assert "ok" in result.output


def test_invalid_threshold_fails(tmp_path) -> None:
    predictions = tmp_path / "predictions.tsv"
    _write_predictions(predictions)

    result = runner.invoke(
        app,
        [
            "stratified-eval",
            "--predictions",
            str(predictions),
            "--outdir",
            str(tmp_path / "bad_threshold"),
            "--threshold",
            "1.5",
        ],
    )

    assert result.exit_code != 0
    assert "threshold must be between 0 and 1" in result.output


def test_run_summary_accepts_stratified_eval_dir(tmp_path) -> None:
    stratified_dir = _build_stratified(tmp_path)
    summary_dir = tmp_path / "summary"

    summarize_run(
        RunSummaryConfig(
            outdir=str(summary_dir),
            stratified_eval_dir=str(stratified_dir),
            title="Stratified-only summary",
        )
    )
    payload = json.loads((summary_dir / "run_summary.json").read_text("utf-8"))
    markdown = (summary_dir / "run_summary.md").read_text("utf-8")

    assert payload["status_overview"]["stratified_eval_present"] is True
    assert payload["stratified_eval_overview"]["key_findings"]
    assert "Stratified evaluation overview" in markdown


def test_make_report_accepts_stratified_eval_dir(tmp_path) -> None:
    stratified_dir = _build_stratified(tmp_path)
    report_dir = tmp_path / "report"

    generate_report(
        ReportConfig(
            outdir=str(report_dir),
            stratified_eval_dir=str(stratified_dir),
            title="Stratified report",
        )
    )
    payload = json.loads((report_dir / "report_summary.json").read_text("utf-8"))
    markdown = (report_dir / "report.md").read_text("utf-8")

    assert "stratified_eval" in payload["sections"]
    assert "Stratified evaluation" in markdown


def _build_stratified(tmp_path) -> Path:
    predictions = tmp_path / "predictions.tsv"
    _write_predictions(predictions)
    outdir = tmp_path / "stratified"
    stratified_evaluate_predictions(
        StratifiedEvalConfig(
            predictions_tsv=str(predictions),
            outdir=str(outdir),
            model_name="synthetic",
        )
    )
    return outdir


def _write_predictions(path: Path) -> None:
    rows = [
        ("family_001", "train", "identity", "low", 1, 0.90),
        ("family_002", "train", "identity", "low", 0, 0.10),
        ("family_003", "train", "codon_dropout", "high", 1, 0.70),
        ("family_004", "train", "codon_dropout", "high", 0, 0.40),
        ("family_005", "test", "identity", "low", 1, 0.80),
        ("family_006", "test", "identity", "low", 0, 0.20),
        ("family_007", "test", "codon_dropout", "high", 1, 0.60),
        ("family_008", "test", "codon_dropout", "high", 0, 0.45),
        ("family_009", "calib", "identity", "low", 1, 0.85),
        ("family_010", "calib", "identity", "low", 0, 0.15),
        ("family_011", "val", "codon_dropout", "high", 1, 0.65),
        ("family_012", "val", "codon_dropout", "high", 0, 0.35),
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
        for family_id, split, method, saturation, label, prob in rows:
            pred = int(prob >= 0.5)
            writer.writerow(
                {
                    "family_id": family_id,
                    "method": method,
                    "split": split,
                    "tensor_file": f"{family_id}.npz",
                    "gene_label": label,
                    "saturation_tier": saturation,
                    "prob_positive": prob,
                    "pred_label": pred,
                    "correct": int(pred == label),
                }
            )


def _read_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))
