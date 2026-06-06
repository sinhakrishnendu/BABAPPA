from typer.testing import CliRunner

from babappa.benchmarks import (
    PublicationRevisionPlanConfig,
    plan_publication_revision_benchmarks,
)
from babappa.cli import app


runner = CliRunner()


def test_publication_revision_planner_writes_reviewer_requested_artifacts(tmp_path):
    outdir = tmp_path / "revision_plan"
    summary = plan_publication_revision_benchmarks(
        PublicationRevisionPlanConfig(outdir=str(outdir), retained_validation_families=120)
    )

    assert summary["status"] == "planned"
    assert (outdir / "known_positive_control_panel_template.tsv").exists()
    assert (outdir / "empirical_transfer_panel_template.tsv").exists()
    assert (outdir / "sensitivity_analysis_grid.tsv").exists()
    assert (outdir / "retained_validation_plan.json").exists()
    assert (outdir / "revision_response_matrix.tsv").exists()

    matrix = (outdir / "revision_response_matrix.tsv").read_text()
    assert "empirical_demonstration_weak" in matrix
    assert "simulator_to_real_transfer_unvalidated" in matrix
    assert "conditional_100k_pass_due_to_pruned_intermediates" in matrix
    assert "hyperparameter_justification" in matrix


def test_publication_revision_scripts_do_not_use_assistant_stamps(tmp_path):
    outdir = tmp_path / "revision_plan"
    plan_publication_revision_benchmarks(PublicationRevisionPlanConfig(outdir=str(outdir)))
    script_text = "\n".join(path.read_text() for path in (outdir / "scripts").glob("*.sh"))

    assert ("Co" + "dex") not in script_text
    assert ("Chat" + "GPT") not in script_text
    assert ("USER-RUN" + " ONLY") not in script_text
    assert "Long-run" in script_text


def test_publication_revision_cli_generates_plan(tmp_path):
    outdir = tmp_path / "cli_revision_plan"
    result = runner.invoke(
        app,
        [
            "plan-publication-revision-benchmarks",
            "--outdir",
            str(outdir),
            "--retained-validation-families",
            "120",
            "--null-replicates",
            "50",
            "--threads",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "Publication Revision Benchmark Plan" in result.output
    assert (outdir / "publication_revision_plan.md").exists()
    assert (outdir / "scripts" / "run_known_positive_controls.sh").exists()
