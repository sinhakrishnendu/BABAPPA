import json

from babappa.branch import (
    BranchAggregationControlsRerunPlanConfig,
    BranchContextAblationInterpretationConfig,
    ExplicitBranchTruth10kPlanConfig,
    columns_for_policy,
    get_branch_feature_policy,
    interpret_branch_context_ablation,
    list_branch_feature_policies,
    plan_explicit_branch_truth_10k,
    plan_rerun_branch_aggregation_controls,
)
from babappa.branch.context_ablation import FOREGROUND_IDENTITY_COLUMNS


def test_branch_feature_policy_list_contains_all_cycle36_policies() -> None:
    names = {row["policy"] for row in list_branch_feature_policies()}

    assert {
        "full_context",
        "no_foreground_identity",
        "no_foreground_codon_context",
        "no_foreground_all",
        "context_only",
        "conservative_branch_site",
    } <= names


def test_conservative_branch_site_excludes_foreground_identity_columns() -> None:
    feature_columns = [
        "site_relative_position",
        "codon_id_mean",
        "foreground_codon_id",
        *FOREGROUND_IDENTITY_COLUMNS,
    ]

    selected = columns_for_policy(feature_columns, "conservative_branch_site")

    assert not (set(selected) & set(FOREGROUND_IDENTITY_COLUMNS))
    assert "site_relative_position" in selected
    assert "codon_id_mean" in selected
    assert "foreground_codon_id" in selected


def test_context_only_policy_is_diagnostic_and_not_production() -> None:
    policy = get_branch_feature_policy("context_only")

    assert policy.production_default is False
    assert "diagnostic" in policy.recommended_role
    assert "never production default" in policy.recommended_role
    assert policy.warning


def test_ablation_interpreter_flags_high_context_only_shortcut(tmp_path) -> None:
    summary_dir = _write_ablation_summary(tmp_path)
    outdir = tmp_path / "interpretation"

    summary = interpret_branch_context_ablation(
        BranchContextAblationInterpretationConfig(
            summary_dir=str(summary_dir),
            outdir=str(outdir),
        )
    )
    payload = json.loads((outdir / "branch_context_ablation_interpretation.json").read_text("utf-8"))

    assert "context_only_shortcut_high" in summary["warnings"]
    assert "context_only_shortcut_high" in payload["warnings"]
    assert summary["recommended_next_default"] == "conservative_branch_site"


def test_ablation_interpreter_flags_foreground_context_dependence(tmp_path) -> None:
    summary_dir = _write_ablation_summary(tmp_path)
    outdir = tmp_path / "interpretation"

    summary = interpret_branch_context_ablation(
        BranchContextAblationInterpretationConfig(
            summary_dir=str(summary_dir),
            outdir=str(outdir),
        )
    )
    payload = json.loads((outdir / "branch_context_ablation_interpretation.json").read_text("utf-8"))

    assert "foreground_context_dependence" in summary["warnings"]
    assert "non_context_sequence_signal_present" in summary["conclusions"]
    assert payload["non_context_sequence_signal_present"] is True
    assert payload["ten_k_readiness"]["full_context_only_10k"] == "not_ready"


def test_rerun_aggregation_controls_planner_generates_script_with_new_controls(tmp_path) -> None:
    outdir = tmp_path / "controls_rerun_plan"

    summary = plan_rerun_branch_aggregation_controls(
        BranchAggregationControlsRerunPlanConfig(
            run_name="explicit_branch_truth_1k",
            tiers="low",
            output_suffix="_streamed",
            outdir=str(outdir),
            n_permutations=100,
            seed=42,
        )
    )
    expected = json.loads((outdir / "expected_outputs.json").read_text("utf-8"))
    run_text = (outdir / "run_branch_aggregation_controls_rerun.sh").read_text("utf-8")

    assert summary["does_not_run_jobs"] is True
    assert "babappa branch-aggregation-controls" in run_text
    assert "train-branch-site" not in run_text
    assert "--n-permutations 100" in run_text
    assert {
        "within_family_branch_label_shuffle",
        "within_family_site_label_shuffle",
        "branch_score_permutation_within_family",
        "family_label_preserving_random_scores",
        "degree_prevalence_matched_null",
    } <= set(expected["controls_included"])


def test_explicit_branch_truth_10k_planner_uses_conservative_policy_and_explicit_truth(tmp_path) -> None:
    outdir = tmp_path / "explicit_10k_plan"

    summary = plan_explicit_branch_truth_10k(
        ExplicitBranchTruth10kPlanConfig(
            outdir=str(outdir),
            tiers="low",
            methods="identity,mafft",
            feature_policy="conservative_branch_site",
        )
    )
    expected = json.loads((outdir / "expected_outputs.json").read_text("utf-8"))
    run_text = (outdir / "run_explicit_branch_truth_10k.sh").read_text("utf-8")
    markdown = (outdir / "explicit_branch_truth_10k_plan.md").read_text("utf-8")

    assert summary["does_not_run_jobs"] is True
    assert summary["feature_policy"] == "conservative_branch_site"
    assert summary["truth_mode"] == "explicit"
    assert expected["feature_policy"] == "conservative_branch_site"
    assert expected["truth_mode"] == "explicit"
    assert "--truth-mode explicit" in run_text
    assert "--profiles conservative_branch_site" in run_text
    assert "full_context" in markdown


def _write_ablation_summary(tmp_path):
    summary_dir = tmp_path / "ablation_summary"
    summary_dir.mkdir()
    (summary_dir / "branch_context_ablation_summary.tsv").write_text(
        "\n".join(
            [
                "tier\tprofile\tmodel\tn_features\texcluded_columns\ttest_n\ttest_auroc\ttest_f1\ttest_mcc\tall_auroc\tall_f1\tall_mcc\tmetrics_json",
                "low\tfull_model\tbaseline\t10\t\t100\t0.99\t0.90\t0.80\t0.99\t0.90\t0.80\tlow_full.json",
                "low\tcontext_only\tbaseline\t4\t\t100\t0.96\t0.88\t0.76\t0.96\t0.88\t0.76\tlow_context.json",
                "low\tno_foreground_all\tbaseline\t6\tforeground\t100\t0.80\t0.60\t0.40\t0.80\t0.60\t0.40\tlow_nofg.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_dir
