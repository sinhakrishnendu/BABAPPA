import json
from pathlib import Path

from typer.testing import CliRunner

from babappa.benchmarks import FastExternal10kPlanConfig, plan_fast_external_10k
from babappa.cli import app


runner = CliRunner()


def test_fast_external_10k_planner_generates_all_files(tmp_path) -> None:
    outdir = tmp_path / "fast_external_10k_plan"
    summary = plan_fast_external_10k(
        FastExternal10kPlanConfig(
            outdir=str(outdir),
            panel_outdir=str(tmp_path / "saturation_panel_external_fast_10k"),
            families_per_tier=3,
            tiers=["low", "extreme"],
            methods=["identity", "mafft", "babappalign", "muscle"],
        )
    )

    assert summary["status"] == "ok"
    for filename in [
        "run_fast_external_10k.sh",
        "monitor_fast_external_10k.sh",
        "validate_fast_external_10k.sh",
        "summarize_fast_external_10k.sh",
        "expected_outputs.json",
        "fast_external_10k_plan.md",
    ]:
        assert (outdir / filename).exists()


def test_fast_external_10k_scripts_are_user_run_only(tmp_path) -> None:
    outdir = tmp_path / "plan"
    plan_fast_external_10k(
        FastExternal10kPlanConfig(
            outdir=str(outdir),
            panel_outdir=str(tmp_path / "panel"),
            families_per_tier=2,
            tiers=["low"],
        )
    )

    for filename in [
        "run_fast_external_10k.sh",
        "monitor_fast_external_10k.sh",
        "validate_fast_external_10k.sh",
        "summarize_fast_external_10k.sh",
    ]:
        text = (outdir / filename).read_text("utf-8")
        assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail")
        assert "MANUAL EXECUTION SCRIPT" in text
        assert "Review before running" in text


def test_fast_external_10k_run_script_method_policy(tmp_path) -> None:
    outdir = tmp_path / "plan"
    plan_fast_external_10k(
        FastExternal10kPlanConfig(
            outdir=str(outdir),
            panel_outdir=str(tmp_path / "panel"),
            families_per_tier=2,
            tiers=["low"],
            methods=["identity", "mafft", "babappalign", "muscle"],
        )
    )
    run_script = (outdir / "run_fast_external_10k.sh").read_text("utf-8")

    assert "--methods identity,mafft,babappalign,muscle" in run_script
    assert "prank" not in run_script
    assert "tcoffee" not in run_script
    assert "method_policy.tsv" in run_script
    assert "build-tensors" in run_script


def test_fast_external_10k_expected_outputs_parse(tmp_path) -> None:
    outdir = tmp_path / "plan"
    plan_fast_external_10k(
        FastExternal10kPlanConfig(
            outdir=str(outdir),
            panel_outdir=str(tmp_path / "panel"),
            families_per_tier=2500,
            tiers=["low", "moderate", "high", "extreme"],
            methods=["identity", "mafft", "babappalign", "muscle"],
        )
    )

    payload = json.loads((outdir / "expected_outputs.json").read_text("utf-8"))
    assert payload["scale"] == 10000
    assert payload["families_per_tier"] == 2500
    assert payload["expected_family_method_attempts"] == 40000
    assert payload["expected_raw_site_rows_assuming_300_codons"] == 12000000
    assert payload["planner_executed_commands"] == []
    assert payload["diagnostic_exclusions"]["prank"] == "diagnostic only, excluded from default"


def test_fast_external_10k_cli_plan_exits_0(tmp_path) -> None:
    outdir = tmp_path / "cli_plan"
    result = runner.invoke(
        app,
        [
            "plan-fast-external-10k",
            "--outdir",
            str(outdir),
            "--panel-outdir",
            str(tmp_path / "panel"),
            "--families-per-tier",
            "2",
            "--tiers",
            "low",
            "--methods",
            "identity,mafft,babappalign,muscle",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Run script" in result.output
    assert (outdir / "validate_fast_external_10k.sh").exists()
    assert (outdir / "monitor_fast_external_10k.sh").exists()
