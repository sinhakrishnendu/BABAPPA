import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from babappa.cli import app
from babappa.reports import (
    ExternalTierSummaryConfig,
    summarize_external_tiers,
    validate_external_tier_summary_dir,
)


runner = CliRunner()


def test_external_tier_summary_allows_missing_optional_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_required_tier(tmp_path, "low", methods=["identity", "mafft"])

    summary = summarize_external_tiers(
        ExternalTierSummaryConfig(
            tiers="low",
            outdir="external_aligner_1k_cross_tier_summary",
        )
    )

    outdir = tmp_path / "external_aligner_1k_cross_tier_summary"
    assert summary["status"] == "ok"
    assert summary["n_warning"] > 0
    assert (outdir / "external_tier_summary.json").exists()
    assert (outdir / "external_tier_summary.md").exists()

    validation = validate_external_tier_summary_dir(outdir)
    assert validation["status"] == "ok"
    assert validation["n_fail"] == 0


def test_external_tier_summary_carries_method_quarantine(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_required_tier(tmp_path, "extreme", methods=["identity", "mafft", "muscle"])
    _write_method_policy(tmp_path, "extreme")

    summarize_external_tiers(
        ExternalTierSummaryConfig(
            tiers=["extreme"],
            outdir="cross_tier",
        )
    )

    rows = _read_tsv(tmp_path / "cross_tier" / "external_method_policy_summary.tsv")
    muscle = [row for row in rows if row["method"] == "muscle"][0]
    assert muscle["recommendation"] == "quarantine"
    payload = json.loads((tmp_path / "cross_tier" / "external_tier_summary.json").read_text("utf-8"))
    assert payload["recommended_10k_method_set"] == [
        "identity",
        "mafft",
        "babappalign",
        "muscle-with-quarantine",
    ]


def test_cli_external_tier_summary_and_validator(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_required_tier(tmp_path, "low", methods=["identity"])
    outdir = tmp_path / "cli_summary"

    result = runner.invoke(
        app,
        [
            "summarize-external-tiers",
            "--tiers",
            "low",
            "--outdir",
            str(outdir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Recommended 10K methods" in result.output

    validate_result = runner.invoke(
        app,
        [
            "validate-external-tier-summary",
            "--summary-dir",
            str(outdir),
        ],
    )
    assert validate_result.exit_code == 0, validate_result.output
    assert "ok" in validate_result.output


def _write_required_tier(tmp_path: Path, tier: str, methods: list[str]) -> None:
    run_dir = tmp_path / f"run_summary_external_aligner_validation_{tier}"
    neural_dir = tmp_path / f"site_neural_external_aligner_validation_{tier}"
    run_dir.mkdir()
    neural_dir.mkdir()
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "generated_files": {"json_summary": str(run_dir / "run_summary.json")},
                "merged_dataset_overview": {
                    "methods": methods,
                    "n_families": 2,
                    "n_rows": 2 * len(methods),
                },
                "run_summary_version": "test",
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    (neural_dir / "site_neural_metrics.json").write_text(
        json.dumps(
            {
                "metrics_by_split": {
                    "all": _metrics(n=20, auroc=0.95),
                    "test": _metrics(n=5, auroc=0.93),
                },
                "site_neural_version": "test",
            }
        ),
        encoding="utf-8",
    )


def _write_method_policy(tmp_path: Path, tier: str) -> None:
    policy_dir = tmp_path / f"method_policy_external_aligner_validation_{tier}"
    policy_dir.mkdir()
    (policy_dir / "method_policy.json").write_text(
        json.dumps(
            {
                "usable_methods": ["identity", "mafft"],
                "quarantined_methods": ["muscle"],
                "methods": [
                    {
                        "method": "identity",
                        "attempted_families": 2,
                        "successful_families": 2,
                        "failed_families": 0,
                        "failure_fraction": "0",
                        "site_map_unique_fraction": "1",
                        "site_map_conflict_fraction": "0",
                        "site_map_frame_error_fraction": "0",
                        "recommendation": "usable",
                        "reason": "passes_policy_thresholds",
                    },
                    {
                        "method": "muscle",
                        "attempted_families": 2,
                        "successful_families": 1,
                        "failed_families": 1,
                        "failure_fraction": "0.5",
                        "site_map_unique_fraction": "0.5",
                        "site_map_conflict_fraction": "0",
                        "site_map_frame_error_fraction": "0.5",
                        "recommendation": "quarantine",
                        "reason": "frame_error_fraction>0",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _metrics(n: int, auroc: float) -> dict:
    return {
        "n": n,
        "positives": n // 2,
        "negatives": n // 2,
        "auroc": auroc,
        "accuracy": 0.9,
        "precision": 0.9,
        "recall": 0.9,
        "f1": 0.9,
        "mcc": 0.8,
        "specificity": 0.9,
    }


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))
