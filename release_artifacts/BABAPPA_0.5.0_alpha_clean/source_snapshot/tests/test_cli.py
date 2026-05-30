from typer.testing import CliRunner

from babappa.cli import app


runner = CliRunner()


def test_status_exits_successfully() -> None:
    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "BABAPPA is installed" in result.output


def test_simulate_small_parameters_exits_successfully(tmp_path) -> None:
    outdir = tmp_path / "sim_cli"
    result = runner.invoke(
        app,
        [
            "simulate",
            "--outdir",
            str(outdir),
            "--n-families",
            "1",
            "--n-taxa",
            "3",
            "--n-codons",
            "30",
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0
    assert "Manifest path:" in result.output
    assert (outdir / "manifest.json").exists()


def test_validate_sim_on_generated_output_exits_successfully(tmp_path) -> None:
    outdir = tmp_path / "sim_cli_validate"
    simulate_result = runner.invoke(
        app,
        [
            "simulate",
            "--outdir",
            str(outdir),
            "--n-families",
            "1",
            "--n-taxa",
            "3",
            "--n-codons",
            "30",
            "--seed",
            "42",
        ],
    )
    assert simulate_result.exit_code == 0

    validate_result = runner.invoke(app, ["validate-sim", "--sim-dir", str(outdir)])

    assert validate_result.exit_code == 0
    assert "Simulation directory is valid" in validate_result.output


def test_invalid_saturation_tier_exits_nonzero(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "simulate",
            "--outdir",
            str(tmp_path / "bad_sim"),
            "--n-families",
            "1",
            "--n-taxa",
            "3",
            "--n-codons",
            "30",
            "--saturation-tier",
            "invalid",
        ],
    )

    assert result.exit_code != 0
    assert "saturation_tier" in result.output


def test_validate_missing_file_fails_gracefully() -> None:
    result = runner.invoke(app, ["validate", "--input", "missing-file.fasta"])

    assert result.exit_code == 1
    assert "does not exist" in result.output
