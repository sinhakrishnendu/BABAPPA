import json
from pathlib import Path

from typer.testing import CliRunner

from babappa.benchmarks import (
    LargeRunPlanConfig,
    plan_large_run,
    validate_large_run_plan_dir,
)
from babappa.cli import app


runner = CliRunner()


def test_plan_large_run_10k(tmp_path) -> None:
    outdir = tmp_path / "large_run_plan_10k"

    summary = plan_large_run(
        LargeRunPlanConfig(
            scale=10000,
            families_per_tier=2500,
            outdir=str(outdir),
            negative_downsample_ratio=10,
        )
    )

    payload = json.loads((outdir / "expected_outputs.json").read_text("utf-8"))
    assert summary["status"] == "ok"
    assert payload["expected_raw_site_rows"] == 6_000_000
    assert payload["approximate_downsampled_site_rows"] == 1_626_900
    commands = (outdir / "large_run_commands.sh").read_text("utf-8")
    reference = (outdir / "large_run_commands_commented_reference.sh").read_text("utf-8")
    assert commands.startswith("#!/usr/bin/env bash\nset -euo pipefail")
    assert "USER-RUN ONLY" in commands
    assert "\nbabappa make-saturation-panel " in commands
    assert (outdir / "external_aligner_run_commands.sh").exists()
    assert "babappa build-site-map" in (
        outdir / "external_aligner_run_commands.sh"
    ).read_text("utf-8")
    assert "USER-RUN ONLY" in reference
    assert "\n# babappa make-saturation-panel " in reference
    assert validate_large_run_plan_dir(outdir)["status"] == "ok"


def test_plan_large_run_100k(tmp_path) -> None:
    outdir = tmp_path / "large_run_plan_100k"

    summary = plan_large_run(
        LargeRunPlanConfig(
            scale=100000,
            families_per_tier=25000,
            outdir=str(outdir),
            negative_downsample_ratio=5,
        )
    )

    payload = json.loads((outdir / "expected_outputs.json").read_text("utf-8"))
    assert summary["status"] == "ok"
    assert payload["expected_raw_site_rows"] == 60_000_000
    assert payload["approximate_downsampled_site_rows"] == 8_874_000
    assert payload["planner_executed_commands"] == []
    assert (outdir / "large_run_commands_commented_reference.sh").exists()
    assert (outdir / "external_aligner_run_commands.sh").exists()
    assert validate_large_run_plan_dir(outdir)["status"] == "ok"


def test_validate_large_run_plan_cli(tmp_path) -> None:
    outdir = tmp_path / "large_run_plan"
    plan_large_run(
        LargeRunPlanConfig(
            scale=10000,
            families_per_tier=2500,
            outdir=str(outdir),
            negative_downsample_ratio=10,
        )
    )

    result = runner.invoke(
        app,
        [
            "validate-large-run-plan",
            "--plan-dir",
            str(outdir),
        ],
    )

    assert result.exit_code == 0
    assert "ok" in result.output


def test_plan_large_run_cli_writes_executable_and_reference_scripts(tmp_path) -> None:
    outdir = tmp_path / "large_run_plan_cli"

    result = runner.invoke(
        app,
        [
            "plan-large-run",
            "--scale",
            "10000",
            "--families-per-tier",
            "2500",
            "--outdir",
            str(outdir),
            "--negative-downsample-ratio",
            "10",
        ],
    )

    assert result.exit_code == 0
    command_lines = (outdir / "large_run_commands.sh").read_text("utf-8").splitlines()
    reference_lines = (
        outdir / "large_run_commands_commented_reference.sh"
    ).read_text("utf-8").splitlines()
    assert command_lines[0] == "#!/usr/bin/env bash"
    assert command_lines[1] == "set -euo pipefail"
    assert any(line.startswith("babappa ") for line in command_lines)
    assert not any(line.startswith("babappa ") for line in reference_lines)
    assert (outdir / "external_aligner_run_commands.sh").read_text("utf-8").startswith(
        "#!/usr/bin/env bash\nset -euo pipefail"
    )
    assert not (outdir / "planner_execution.log").exists()


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
        root / "src" / "babappa" / "align" / "init.py",
    ]:
        assert not path.exists()
