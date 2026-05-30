import json

from babappa.branch import (
    BranchContextAblationPlanConfig,
    branch_context_profile_columns,
    plan_branch_context_ablation,
)
from babappa.branch.context_ablation import (
    CONTEXT_ONLY_COLUMNS,
    FOREGROUND_ALL_COLUMNS,
)


def test_branch_context_ablation_planner_generates_scripts_without_execution(tmp_path) -> None:
    outdir = tmp_path / "branch_context_ablation_plan"

    summary = plan_branch_context_ablation(
        BranchContextAblationPlanConfig(
            run_name="explicit_branch_truth_1k",
            tiers="low,extreme",
            output_suffix="_streamed",
            outdir=str(outdir),
        )
    )
    run_text = (outdir / "run_branch_context_ablation.sh").read_text("utf-8")
    expected = json.loads((outdir / "expected_outputs.json").read_text("utf-8"))

    assert summary["does_not_run_jobs"] is True
    assert "babappa run-branch-context-ablation" in run_text
    assert "branch_site_dataset_explicit_branch_truth_1k_low_streamed" in run_text
    assert expected["plan_only"] is True
    assert not (tmp_path / "branch_context_ablation_explicit_1k").exists()


def test_no_foreground_all_profile_excludes_all_foreground_columns() -> None:
    feature_columns = [
        "site_relative_position",
        "codon_id_mean",
        *FOREGROUND_ALL_COLUMNS,
    ]

    selected = branch_context_profile_columns(feature_columns, "no_foreground_all")

    assert not (set(selected) & set(FOREGROUND_ALL_COLUMNS))
    assert "site_relative_position" in selected
    assert "codon_id_mean" in selected


def test_context_only_profile_includes_only_context_columns() -> None:
    feature_columns = [
        "site_relative_position",
        "codon_id_mean",
        *CONTEXT_ONLY_COLUMNS,
    ]

    selected = branch_context_profile_columns(feature_columns, "context_only")

    assert selected
    assert set(selected) <= set(CONTEXT_ONLY_COLUMNS)
    assert "site_relative_position" not in selected
    assert "codon_id_mean" not in selected
