"""Command-line interface for BABAPPA."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from babappa import __version__
from babappa.align import (
    ExternalAlignmentConfig,
    AlignmentConfig,
    MethodPolicyConfig,
    SiteMapConfig,
    align_simulation_directory,
    build_alignment_site_maps,
    build_method_policy,
    detect_aligner_backends,
    run_alignment_ensemble,
    smoke_aligner,
    validate_alignment_directory,
    validate_method_policy_dir,
    validate_site_map_dir,
)
from babappa.benchmarks import (
    ExternalAlignerValidationPlanConfig,
    ExternalCompletedTierReportPlanConfig,
    ExternalExtremeRecoveryPlanConfig,
    FastExternal10kPlanConfig,
    LargeRunPlanConfig,
    SaturationPanelConfig,
    StabilityBenchmarkConfig,
    build_saturation_panel,
    plan_complete_external_tier_reports,
    plan_external_extreme_recovery,
    plan_external_aligner_validation,
    plan_fast_external_10k,
    plan_large_run,
    run_stability_benchmark,
    validate_large_run_plan_dir,
    validate_saturation_panel_dir,
    validate_stability_benchmark_dir,
)
from babappa.branch import (
    BranchAggregationConfig,
    BranchAggregationControlConfig,
    BranchAggregationControlsRerunPlanConfig,
    BranchAggregationThresholdPolicyConfig,
    BranchConditioned10kPlanConfig,
    BranchConditionedTierSummaryConfig,
    BranchContextAblationInterpretationConfig,
    BranchContextAblationPlanConfig,
    BranchContextAblationRunConfig,
    BranchContextAblationSummaryConfig,
    BranchSiteBaselineConfig,
    BranchSiteCalibrationConfig,
    BranchSiteDatasetConfig,
    BranchSiteNeuralTrainConfig,
    BranchSiteOracleLabelConfig,
    BranchSiteRunSummaryConfig,
    BranchSiteThresholdPolicyConfig,
    BranchTruthStatusAuditConfig,
    DeployableModelPackagePlanConfig,
    ExplicitBranchTruth10kPlanConfig,
    ExplicitBranchTruth10kMacPlanConfig,
    ExplicitBranchTruth100kMacPlanConfig,
    ExplicitBranchTruth1kPlanConfig,
    ExplicitBranchTruthPrototypePlanConfig,
    Final100KValidationReportConfig,
    MPSPlanPreflightConfig,
    MPSPlanScriptValidationConfig,
    ValidationScaleComparisonConfig,
    aggregate_branch_sites,
    audit_branch_site_leakage,
    audit_branch_truth_status,
    build_final_100k_validation_report,
    build_branch_aggregation_threshold_policy,
    build_branch_site_dataset,
    build_branch_site_threshold_policy,
    calibrate_branch_site_model,
    compare_validation_scales,
    extract_branch_site_labels,
    interpret_branch_context_ablation,
    list_branch_feature_policies,
    plan_branch_conditioned_10k,
    plan_branch_context_ablation,
    plan_explicit_branch_truth_10k,
    plan_explicit_branch_truth_10k_mac,
    plan_explicit_branch_truth_100k_mac,
    plan_explicit_branch_truth_1k,
    plan_explicit_branch_truth_prototype,
    plan_deployable_model_package,
    plan_rerun_branch_aggregation_controls,
    preflight_explicit_branch_truth_mps_plan,
    run_branch_aggregation_controls,
    run_branch_context_ablation,
    summarize_branch_conditioned_tiers,
    summarize_branch_context_ablation,
    summarize_branch_site_run,
    train_branch_site_baseline,
    train_branch_site_neural_model,
    validate_branch_aggregation_controls_dir,
    validate_branch_aggregation_dir,
    validate_branch_aggregation_threshold_policy_dir,
    validate_branch_conditioned_tier_summary_dir,
    validate_branch_site_baseline_dir,
    validate_branch_site_calibration_dir,
    validate_branch_site_dataset_dir,
    validate_branch_site_label_dir,
    validate_branch_site_leakage_dir,
    validate_branch_site_neural_dir,
    validate_branch_site_run_summary_dir,
    validate_branch_site_threshold_policy_dir,
    validate_branch_truth_status_audit_dir,
    validate_mps_plan_script,
)
from babappa.calibration import (
    BaselineCalibrationConfig,
    NeuralCalibrationConfig,
    StratifiedCalibrationConfig,
    ThresholdPolicyConfig,
    build_threshold_policy,
    calibrate_baseline_model,
    calibrate_by_group,
    calibrate_neural_model,
    validate_baseline_calibration_dir,
    validate_neural_calibration_dir,
    validate_stratified_calibration_dir,
    validate_threshold_policy_dir,
)
from babappa.datasets import (
    DatasetIndexConfig,
    DatasetMergeConfig,
    ResplitDatasetConfig,
    build_dataset_index,
    merge_dataset_indexes,
    resplit_dataset,
    validate_dataset_index,
    validate_merged_dataset_dir,
    validate_resplit_dataset_dir,
)
from babappa.deploy import (
    DeployableModelPackageConfig,
    DeployableModelPackageValidationConfig,
    DeployableModelSmokeConfig,
    package_deployable_model,
    smoke_load_deployable_model,
    validate_deployable_model_package,
)
from babappa.empirical import (
    AddPrefilteredFamilyConfig,
    BabappaOnlyResultAuditConfig,
    BabappaOnlySignalInterpretationConfig,
    CloseTaxaControlFamilyPlanConfig,
    CodemlReferenceParseConfig,
    CodemlReferencePrepConfig,
    ClassicalReferenceWorkflowPlanConfig,
    EmpiricalEvidencePackConfig,
    EmpiricalEvidencePackValidationConfig,
    EmpiricalAlignmentEnsembleConfig,
    EmpiricalApplicabilityConfig,
    EmpiricalBranchSiteReportConfig,
    EmpiricalFamilyAcquisitionPlanConfig,
    EmpiricalFamilyPrefilterConfig,
    EmpiricalBranchSiteScoringConfig,
    EmpiricalFeatureAuditConfig,
    EmpiricalFeatureExtractionConfig,
    EmpiricalInputValidationConfig,
    EmpiricalOODSummaryConfig,
    EmpiricalPilotPanelRunConfig,
    EmpiricalPilotPanelSummaryConfig,
    EmpiricalPilotPanelValidationConfig,
    EmpiricalPilotSummaryValidationConfig,
    EmpiricalReferenceComparisonConfig,
    EmpiricalScoringPlanConfig,
    ExternalBenchmarkPanelPlanConfig,
    HyphyReferenceParseConfig,
    HyphyReferencePrepConfig,
    CdsFastaSanitizeConfig,
    ForegroundCandidateConfig,
    LocalPilotFileDiscoveryConfig,
    OODAwareFamilyBuildPlanConfig,
    RealEmpiricalPilotDecisionReportConfig,
    RealEmpiricalPilotWorkspaceConfig,
    RealPilotBatchImportConfig,
    RealPilotFamilyImportConfig,
    RealPilotInputStagingConfig,
    RealPilotReadinessConfig,
    RealPilotTreeBuildingPlanConfig,
    ReferenceResultsTableConfig,
    ReferenceResultsTemplateConfig,
    ReferenceToolCheckConfig,
    ReferenceToolsInstallPlanConfig,
    SimulationMatchedNullCalibrationConfig,
    SimulationMatchedNullCalibrationValidationConfig,
    SimulationMatchedCalibrationPlanConfig,
    SimulationMatchedCalibrationSummaryConfig,
    TargetTaxaRecommendationConfig,
    WRKYReferenceCalibrationReportConfig,
    WRKYInterpretationStatusConfig,
    add_prefiltered_family_to_pilot,
    audit_babappa_only_result,
    audit_empirical_features,
    build_reference_results_table,
    check_reference_tools,
    compare_empirical_reference_results,
    discover_local_pilot_files,
    extract_empirical_branch_site_features,
    freeze_empirical_evidence_pack,
    install_reference_tools_plan,
    interpret_babappa_only_signal,
    import_real_pilot_batch,
    import_real_pilot_family,
    list_foreground_candidates,
    make_empirical_branch_site_report,
    make_real_empirical_pilot_decision_report,
    make_wrky_reference_calibration_report,
    make_wrky_interpretation_status,
    parse_codeml_reference,
    parse_hyphy_reference,
    plan_classical_reference_workflows,
    plan_close_taxa_control_family,
    plan_empirical_family_acquisition,
    plan_ood_aware_family_build,
    plan_external_benchmark_panel,
    plan_empirical_scoring,
    plan_real_pilot_tree_building,
    plan_simulation_matched_calibration,
    prepare_real_empirical_pilot_workspace,
    prepare_real_pilot_inputs,
    prepare_codeml_reference,
    prepare_hyphy_reference,
    prefilter_empirical_family,
    recommend_target_taxa,
    run_empirical_pilot_panel,
    run_empirical_alignment_ensemble,
    run_empirical_applicability,
    run_simulation_matched_null_calibration,
    sanitize_cds_fasta,
    score_empirical_branch_sites,
    summarize_empirical_ood,
    summarize_empirical_pilot_panel,
    summarize_simulation_matched_calibration_plan,
    validate_empirical_input,
    validate_empirical_evidence_pack,
    validate_empirical_pilot_panel,
    validate_empirical_pilot_summary,
    validate_real_pilot_readiness,
    validate_simulation_matched_null_calibration,
    write_reference_results_template,
    write_wrky_matched_null_script,
)
from babappa.models import (
    BaselineTrainConfig,
    train_baseline_model,
    validate_baseline_model_dir,
)
from babappa.site import (
    AggregationThresholdPolicyConfig,
    OracleSiteLabelConfig,
    SiteAggregationControlConfig,
    SiteAggregationConfig,
    SiteBaselineConfig,
    SiteCalibrationConfig,
    SiteCalibrationCompareConfig,
    SiteDatasetConfig,
    SiteModelCompareConfig,
    SiteNeuralTrainConfig,
    SiteStabilityConfig,
    SiteStratifiedEvalConfig,
    SiteThresholdPolicyConfig,
    aggregate_site_predictions,
    audit_site_dataset_leakage,
    build_aggregation_threshold_policy,
    build_site_dataset,
    build_site_threshold_policy,
    calibrate_site_model,
    compare_site_calibrations,
    compare_site_models,
    extract_oracle_site_labels,
    run_site_aggregation_controls,
    run_site_stability_benchmark,
    site_stratified_evaluate,
    train_site_baseline,
    train_site_neural_model,
    validate_aggregation_threshold_policy_dir,
    validate_site_aggregation_controls_dir,
    validate_site_aggregation_dir,
    validate_site_baseline_dir,
    validate_site_calibration_dir,
    validate_site_calibration_comparison_dir,
    validate_site_dataset_dir,
    validate_site_label_dir,
    validate_site_model_comparison_dir,
    validate_site_neural_dir,
    validate_site_stability_dir,
    validate_site_stratified_eval_dir,
    validate_site_threshold_policy_dir,
)
from babappa.reports import (
    AblationCompareConfig,
    LabelSignalAuditConfig,
    LeakageAuditConfig,
    ModelCompareConfig,
    NeuralDiagnosticsConfig,
    PredictionDiagnosticsConfig,
    ReportConfig,
    RunSummaryConfig,
    ExternalTierSummaryConfig,
    StratifiedEvalConfig,
    audit_label_signal,
    audit_leakage,
    compare_neural_ablations,
    compare_models,
    diagnose_neural_run,
    diagnose_predictions,
    generate_report,
    stratified_evaluate_predictions,
    validate_ablation_comparison_dir,
    validate_label_signal_audit_dir,
    validate_leakage_audit_dir,
    summarize_run,
    summarize_external_tiers,
    validate_external_tier_summary_dir,
    validate_model_comparison_dir,
    validate_neural_diagnostics_dir,
    validate_prediction_diagnostics_dir,
    validate_report_dir,
    validate_run_summary_dir,
    validate_stratified_eval_dir,
)
from babappa.simulate import (
    SimulationConfig,
    audit_simulation_directory,
    simulate_families,
)
from babappa.tensors import (
    TensorBuildConfig,
    build_tensor_dataset,
    validate_tensor_directory,
)
from babappa.training import (
    AppleSiliconBenchmarkConfig,
    MPSTrainingSmokeConfig,
    NeuralDatasetConfig,
    NeuralFullTrainConfig,
    NeuralTrainConfig,
    get_torch_environment,
    inspect_neural_dataset,
    make_smoke_batch,
    run_apple_silicon_benchmark,
    run_mps_training_smoke,
    train_neural_model,
    train_neural_smoke_model,
    validate_mps_smoke_dir,
    validate_neural_model_dir,
    validate_neural_smoke_dir,
)

app = typer.Typer(
    help="BABAPPA: Branch-site Alignment-Bias-Aware Probabilistic Positive-selection Analyzer."
)
console = Console()

PLANNED_MODULES = [
    "simulate",
    "align",
    "tensors",
    "train",
    "calibrate",
    "predict",
    "benchmark",
    "report",
    "asr",
    "energy",
]
REQUIRED_SIMULATION_SUFFIXES = {
    "fasta": ".fasta",
    "treefile": ".treefile",
    "truth": ".truth.json",
    "homology": ".homology.tsv",
    "events": ".events.tsv",
    "meta": ".meta.json",
}
BRANCH_SITE_TRUTH_HEADER = [
    "family_id",
    "saturation_tier",
    "branch_id",
    "foreground_taxon",
    "branch_type",
    "site_index_zero",
    "site_index_one",
    "y_branch_site",
    "selection_event_id",
    "truth_source",
]
EVENTS_HEADER = [
    "family_id",
    "taxon",
    "codon_index_0based",
    "old_codon",
    "new_codon",
    "event_type",
    "is_selected_site",
    "is_foreground",
]
HOMOLOGY_HEADER = ["taxon", "codon_index_0based", "homology_id", "codon"]
AVAILABLE_COMMANDS = [
    "simulate",
    "validate-sim",
    "audit-sim",
    "align-sim",
    "validate-align",
    "check-aligners",
    "smoke-aligner",
    "align-external",
    "build-site-map",
    "validate-site-map",
    "aligner-method-policy",
    "validate-aligner-method-policy",
    "build-tensors",
    "validate-tensors",
    "index-dataset",
    "validate-index",
    "train-baseline",
    "validate-baseline",
    "calibrate-baseline",
    "validate-calibration",
    "make-report",
    "validate-report",
    "check-neural-env",
    "smoke-mps-training",
    "validate-mps-smoke",
    "benchmark-apple-silicon",
    "inspect-neural-data",
    "smoke-neural-batch",
    "train-neural-smoke",
    "validate-neural-smoke",
    "train-neural",
    "train-neural-v2",
    "train-neural-saturation",
    "train-neural-ranking",
    "validate-neural",
    "audit-label-signal",
    "validate-label-signal-audit",
    "audit-leakage",
    "validate-leakage-audit",
    "calibrate-neural",
    "validate-neural-calibration",
    "calibrate-stratified",
    "validate-stratified-calibration",
    "diagnose-predictions",
    "validate-prediction-diagnostics",
    "diagnose-neural",
    "validate-neural-diagnostics",
    "threshold-policy",
    "validate-threshold-policy",
    "stratified-eval",
    "validate-stratified-eval",
    "make-saturation-panel",
    "validate-saturation-panel",
    "merge-datasets",
    "validate-merged-dataset",
    "resplit-dataset",
    "validate-resplit-dataset",
    "stability-benchmark",
    "validate-stability-benchmark",
    "plan-large-run",
    "validate-large-run-plan",
    "plan-external-aligner-validation",
    "plan-complete-external-tier-reports",
    "plan-external-extreme-recovery",
    "plan-fast-external-10k",
    "extract-site-labels",
    "validate-site-labels",
    "build-site-dataset",
    "validate-site-dataset",
    "audit-site-leakage",
    "train-site-baseline",
    "validate-site-baseline",
    "train-site-neural",
    "validate-site-neural",
    "calibrate-site-neural",
    "validate-site-calibration",
    "site-threshold-policy",
    "validate-site-threshold-policy",
    "site-stratified-eval",
    "validate-site-stratified-eval",
    "aggregate-sites",
    "validate-site-aggregation",
    "site-stability-benchmark",
    "validate-site-stability",
    "compare-site-models",
    "validate-site-model-comparison",
    "aggregation-controls",
    "validate-aggregation-controls",
    "aggregation-threshold-policy",
    "validate-aggregation-threshold-policy",
    "compare-site-calibrations",
    "validate-site-calibration-comparison",
    "summarize-run",
    "validate-run-summary",
    "summarize-external-tiers",
    "validate-external-tier-summary",
    "extract-branch-site-labels",
    "validate-branch-site-labels",
    "build-branch-site-dataset",
    "validate-branch-site-dataset",
    "audit-branch-site-leakage",
    "validate-branch-site-leakage",
    "train-branch-site-baseline",
    "validate-branch-site-baseline",
    "train-branch-site-neural",
    "validate-branch-site-neural",
    "calibrate-branch-site-neural",
    "validate-branch-site-calibration",
    "branch-site-threshold-policy",
    "validate-branch-site-threshold-policy",
    "aggregate-branch-sites",
    "validate-branch-aggregation",
    "branch-aggregation-controls",
    "validate-branch-aggregation-controls",
    "plan-rerun-branch-aggregation-controls",
    "branch-aggregation-threshold-policy",
    "validate-branch-aggregation-threshold-policy",
    "summarize-branch-site-run",
    "validate-branch-site-run-summary",
    "plan-branch-conditioned-10k",
    "summarize-branch-conditioned-tiers",
    "validate-branch-conditioned-tier-summary",
    "audit-branch-truth-status",
    "validate-branch-truth-status-audit",
    "plan-branch-context-ablation",
    "run-branch-context-ablation",
    "summarize-branch-context-ablation",
    "interpret-branch-context-ablation",
    "list-branch-feature-policies",
    "plan-explicit-branch-truth-prototype",
    "plan-explicit-branch-truth-1k",
    "plan-explicit-branch-truth-10k",
    "plan-explicit-branch-truth-10k-mac",
    "plan-explicit-branch-truth-100k-mac",
    "validate-mps-plan-script",
    "preflight-explicit-branch-truth-mps-plan",
    "compare-validation-scales",
    "build-explicit-branch-truth-100k-validation-report",
    "plan-deployable-model-package",
    "package-deployable-model",
    "validate-deployable-model-package",
    "smoke-load-deployable-model",
    "plan-simulation-matched-calibration",
    "summarize-simulation-matched-calibration-plan",
    "plan-empirical-scoring",
    "validate-empirical-input",
    "run-empirical-alignment-ensemble",
    "extract-empirical-branch-site-features",
    "audit-empirical-features",
    "empirical-applicability",
    "score-empirical-branch-sites",
    "make-empirical-branch-site-report",
    "plan-external-benchmark-panel",
    "validate-empirical-pilot-panel",
    "run-empirical-pilot-panel",
    "plan-classical-reference-workflows",
    "compare-empirical-reference-results",
    "summarize-empirical-pilot-panel",
    "validate-empirical-pilot-summary",
    "prepare-real-empirical-pilot-workspace",
    "make-real-empirical-pilot-decision-report",
    "prepare-real-pilot-inputs",
    "import-real-pilot-family",
    "import-real-pilot-batch",
    "plan-real-pilot-tree-building",
    "sanitize-cds-fasta",
    "list-foreground-candidates",
    "validate-real-pilot-readiness",
    "discover-local-pilot-files",
    "prefilter-empirical-family",
    "plan-empirical-family-acquisition",
    "recommend-target-taxa",
    "plan-ood-aware-family-build",
    "add-prefiltered-family-to-pilot",
    "summarize-empirical-ood",
    "freeze-empirical-evidence-pack",
    "validate-empirical-evidence-pack",
    "check-reference-tools",
    "install-reference-tools-plan",
    "prepare-codeml-reference",
    "prepare-hyphy-reference",
    "parse-codeml-reference",
    "parse-hyphy-reference",
    "build-reference-results-table",
    "run-simulation-matched-null-calibration",
    "validate-simulation-matched-null-calibration",
    "write-reference-results-template",
    "write-wrky-matched-null-script",
    "make-wrky-interpretation-status",
    "make-wrky-reference-calibration-report",
    "interpret-babappa-only-signal",
    "audit-babappa-only-result",
    "plan-close-taxa-control-family",
    "compare-models",
    "validate-model-comparison",
    "compare-ablations",
    "validate-ablation-comparison",
]


def _version_callback(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit()


@app.callback()
def callback(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the BABAPPA version and exit.",
    )
) -> None:
    """BABAPPA command group."""


@app.command()
def status() -> None:
    """Show installation status and planned modules."""
    console.print(
        "BABAPPA is installed and the Cycle 27 external-aligner site-map layer is available."
    )
    console.print("Available commands:")
    for command in AVAILABLE_COMMANDS:
        console.print(f"- {command}")
    console.print("Planned modules:")
    for module in PLANNED_MODULES:
        console.print(f"- {module}")


@app.command()
def simulate(
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for simulated families.",
    ),
    n_families: int = typer.Option(
        5,
        "--n-families",
        min=1,
        help="Number of families to simulate.",
    ),
    n_taxa: int = typer.Option(
        8,
        "--n-taxa",
        min=3,
        help="Number of extant taxa per family.",
    ),
    n_codons: int = typer.Option(
        120,
        "--n-codons",
        min=30,
        help="Number of codons per coding sequence.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for deterministic simulation.",
    ),
    positive_rate: float = typer.Option(
        0.3,
        "--positive-rate",
        help="Probability that a family contains positive selection.",
    ),
    selected_site_fraction: float = typer.Option(
        0.05,
        "--selected-site-fraction",
        help="Fraction of sites selected in positive families.",
    ),
    mutation_rate: float = typer.Option(
        0.03,
        "--mutation-rate",
        help="Per-codon mutation attempt rate before saturation scaling.",
    ),
    indel_rate: float = typer.Option(
        0.0,
        "--indel-rate",
        help="Reserved for future indel simulation; must be non-negative.",
    ),
    saturation_tier: str = typer.Option(
        "low",
        "--saturation-tier",
        help="Saturation tier: low, moderate, high, or extreme.",
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        min=1,
        help="Parallel worker processes for independent family simulation.",
    ),
) -> None:
    """Run the lightweight Cycle 2 simulator."""
    try:
        config = SimulationConfig(
            outdir=str(outdir),
            n_families=n_families,
            n_taxa=n_taxa,
            n_codons=n_codons,
            seed=seed,
            positive_rate=positive_rate,
            selected_site_fraction=selected_site_fraction,
            mutation_rate=mutation_rate,
            indel_rate=indel_rate,
            saturation_tier=saturation_tier,
            workers=workers,
        )
        summary = simulate_families(config)
    except ValueError as exc:
        console.print(f"Error: invalid simulation configuration: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        console.print(f"Error: could not write simulation output: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Simulation Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Families", str(summary["n_families"]))
    table.add_row("Workers", str(summary.get("workers", 1)))
    table.add_row("Saturation tier", saturation_tier)
    table.add_row("Manifest", summary["manifest"])
    console.print(table)
    console.print(f"Manifest path: {summary['manifest']}")


@app.command("validate-sim")
def validate_sim(
    sim_dir: Path = typer.Option(
        ...,
        "--sim-dir",
        help="Simulation output directory to validate.",
    ),
    require_branch_truth: bool = typer.Option(
        False,
        "--require-branch-truth",
        help="Fail validation if explicit branch-site simulator truth is absent.",
    ),
) -> None:
    """Validate a BABAPPA simulation output directory."""
    warnings: list[str] = []
    errors = _validate_simulation_directory(
        sim_dir,
        require_branch_truth=require_branch_truth,
        warnings=warnings,
    )
    if errors:
        console.print("Simulation directory is invalid:", style="red")
        for error in errors:
            console.print(f"- {error}", style="red")
        if warnings:
            console.print("Warnings:")
            for warning in warnings:
                console.print(f"- {warning}")
        raise typer.Exit(code=1)

    manifest = _read_json(sim_dir / "manifest.json")
    console.print(f"Simulation directory is valid: {sim_dir}")
    console.print(f"Families: {len(manifest.get('family_ids', []))}")
    console.print(f"Branch truth present: {manifest.get('branch_truth_present', False)}")
    console.print(f"Branch truth files: {manifest.get('n_branch_truth_files', 0)}")
    console.print(f"Branch-site truth rows: {manifest.get('n_branch_site_truth_rows', 0)}")
    console.print(f"Branch positive rows: {manifest.get('n_branch_positive_rows', 0)}")
    console.print(f"Branch truth status: {manifest.get('branch_truth_status', 'missing')}")
    if warnings:
        console.print("Warnings:")
        for warning in warnings:
            console.print(f"- {warning}")


@app.command("audit-sim")
def audit_sim(
    sim_dir: Path = typer.Option(
        ...,
        "--sim-dir",
        help="Simulation output directory to audit.",
    ),
    outdir: Optional[Path] = typer.Option(
        None,
        "--outdir",
        help="Output directory for audit files. Defaults to SIM_DIR/audit.",
    ),
) -> None:
    """Run dataset-level QC for a BABAPPA simulation directory."""
    try:
        summary = audit_simulation_directory(sim_dir=sim_dir, outdir=outdir)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not audit simulation directory: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    audit_files = summary["audit_files"]
    table = Table(title="BABAPPA Simulation Audit Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Expected families", str(summary["n_families_expected"]))
    table.add_row("Audited families", str(summary["n_families_audited"]))
    table.add_row("OK", str(summary["n_ok"]))
    table.add_row("Warning", str(summary["n_warning"]))
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Positive families", str(summary["positive_family_count"]))
    table.add_row("Branch truth present", str(summary.get("branch_truth_present", False)))
    table.add_row("Branch truth files", str(summary.get("n_branch_truth_files", 0)))
    table.add_row("Branch-site truth rows", str(summary.get("n_branch_site_truth_rows", 0)))
    table.add_row("Branch positive rows", str(summary.get("n_branch_positive_rows", 0)))
    table.add_row("Branch truth status", str(summary.get("branch_truth_status", "missing")))
    table.add_row(
        "Mean pairwise nt p-distance",
        f"{summary['mean_pairwise_nt_pdist_mean']:.6f}",
    )
    table.add_row("family_audit.tsv", audit_files["family_audit_tsv"])
    table.add_row("dataset_summary.json", audit_files["dataset_summary_json"])
    console.print(table)

    if summary["n_fail"] > 0:
        raise typer.Exit(code=1)


@app.command("align-sim")
def align_sim(
    sim_dir: Path = typer.Option(
        ...,
        "--sim-dir",
        help="Simulation output directory to align.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for alignment scaffold files.",
    ),
    methods: str = typer.Option(
        "identity,codon_dropout",
        "--methods",
        help="Comma-separated alignment scaffold methods.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for deterministic alignment perturbations.",
    ),
    dropout_rate: float = typer.Option(
        0.02,
        "--dropout-rate",
        help="Codon dropout probability for the codon_dropout method.",
    ),
) -> None:
    """Create internal alignment scaffold channels from a simulation directory."""
    parsed_methods = _parse_methods(methods)
    try:
        config = AlignmentConfig(
            sim_dir=str(sim_dir),
            outdir=str(outdir),
            methods=parsed_methods,
            seed=seed,
            dropout_rate=dropout_rate,
        )
        summary = align_simulation_directory(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not align simulation directory: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Alignment Scaffold Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Input simulation directory", str(sim_dir))
    table.add_row("Output alignment directory", summary["outdir"])
    table.add_row("Families", str(summary["n_families"]))
    table.add_row("Methods", _format_methods(summary["methods"]))
    table.add_row("Manifest", summary["manifest"])
    console.print(table)
    console.print(f"Alignment manifest path: {summary['manifest']}")


@app.command("validate-align")
def validate_align(
    align_dir: Path = typer.Option(
        ...,
        "--align-dir",
        help="Alignment scaffold directory to validate.",
    )
) -> None:
    """Validate a BABAPPA alignment scaffold directory."""
    summary = validate_alignment_directory(align_dir)
    table = Table(title="BABAPPA Alignment Validation Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Expected families", str(summary["n_families_expected"]))
    table.add_row("Checked families", str(summary["n_families_checked"]))
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    table.add_row("Methods", _format_methods(summary["methods"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("check-aligners")
def check_aligners(
    json_out: Optional[Path] = typer.Option(
        None,
        "--json-out",
        help="Optional JSON output path for detected aligner status.",
    )
) -> None:
    """Inspect internal and optional external aligner backends."""
    backends = detect_aligner_backends()
    table = Table(title="BABAPPA Aligner Backends")
    for column in (
        "method",
        "kind",
        "available",
        "wrapper_status",
        "runtime_class",
        "production_default",
        "mapped_oracle_default",
        "default_role",
        "command_template",
        "executable",
        "model_status",
        "model_expected_path",
        "model_present",
        "model_size_bytes",
        "install_command",
        "version",
        "notes",
    ):
        table.add_column(column)
    payload = {}
    for name in sorted(backends):
        backend = backends[name]
        payload[name] = backend.as_dict()
        table.add_row(
            backend.name,
            backend.kind,
            str(backend.available),
            backend.wrapper_status or "",
            backend.runtime_class,
            str(backend.production_default),
            str(backend.mapped_oracle_default),
            backend.default_role,
            backend.command_template or "",
            backend.executable or "",
            backend.model_status or "",
            backend.model_expected_path or "",
            "" if backend.model_present is None else str(backend.model_present),
            "" if backend.model_size_bytes is None else str(backend.model_size_bytes),
            backend.install_command or "",
            backend.version or "",
            "; ".join(backend.notes),
        )
    console.print(table)
    babappalign = backends.get("babappalign")
    if babappalign is not None:
        model_table = Table(title="BABAPPAlign Model Cache")
        model_table.add_column("Field")
        model_table.add_column("Value")
        model_table.add_row("executable", babappalign.executable or "")
        model_table.add_row("model_expected_path", babappalign.model_expected_path or "")
        model_table.add_row("model_present", str(babappalign.model_present))
        model_table.add_row(
            "model_size_bytes",
            "" if babappalign.model_size_bytes is None else str(babappalign.model_size_bytes),
        )
        model_table.add_row("model_status", babappalign.model_status or "")
        model_table.add_row("install_command", babappalign.install_command or "")
        console.print(model_table)
        if babappalign.model_status == "model_missing":
            console.print(
                "babappalign_model_missing: "
                f"expected {babappalign.model_expected_path}; "
                f"install with: {babappalign.install_command}",
                style="yellow",
            )
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        console.print(f"Aligner status JSON: {json_out}")


@app.command("smoke-aligner")
def smoke_aligner_command(
    method: str = typer.Option("babappalign", "--method", help="Aligner method to smoke."),
    outdir: Path = typer.Option("aligner_smoke", "--outdir", help="Smoke report directory."),
    device: str = typer.Option("cpu", "--device", help="Device for BABAPPAlign smoke."),
    timeout_seconds: int = typer.Option(60, "--timeout-seconds"),
) -> None:
    """Run a tiny aligner smoke with explicit BABAPPAlign model-cache diagnostics."""
    try:
        summary = smoke_aligner(
            method=method,
            outdir=outdir,
            device=device,
            timeout_seconds=timeout_seconds,
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not run aligner smoke: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    table = Table(title="BABAPPA Aligner Smoke")
    table.add_column("Field")
    table.add_column("Value")
    for key in (
        "method",
        "status",
        "reason",
        "executable",
        "model_expected_path",
        "model_present",
        "model_size_bytes",
        "install_command",
        "report_json",
        "report_md",
    ):
        if key in summary and summary.get(key) is not None:
            table.add_row(key, str(summary.get(key)))
    console.print(table)
    if summary.get("status") == "fail":
        raise typer.Exit(code=1)


@app.command("align-external")
def align_external(
    sim_dir: Path = typer.Option(..., "--sim-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    methods: str = typer.Option(
        "identity,mafft,babappalign,muscle",
        "--methods",
        help="Comma-separated internal/external methods.",
    ),
    seed: int = typer.Option(42, "--seed"),
    require_available: str = typer.Option(
        "false",
        "--require-available",
        help="Require requested external aligners to be installed: true or false.",
    ),
    keep_intermediate: bool = typer.Option(False, "--keep-intermediate"),
    timeout_seconds: int = typer.Option(300, "--timeout-seconds"),
    threads: int = typer.Option(1, "--threads"),
    aligner_subprocess_threads: int = typer.Option(
        1,
        "--aligner-subprocess-threads",
        help=(
            "CPU threads exposed to each per-family external aligner subprocess. "
            "Keep this low when --threads runs many families in parallel."
        ),
    ),
    babappalign_device: str = typer.Option(
        "cpu",
        "--babappalign-device",
        help="BABAPPAlign device for external alignment: auto, cpu, cuda, or mps.",
    ),
    babappalign_backend: str = typer.Option(
        "auto",
        "--babappalign-backend",
        help="BABAPPAlign runner backend: auto, embedded, or cli.",
    ),
    babappalign_workers: int = typer.Option(
        0,
        "--babappalign-workers",
        help="Override parallel workers for BABAPPAlign only; 0 uses --threads.",
    ),
    max_method_failure_fraction: float = typer.Option(
        0.01,
        "--max-method-failure-fraction",
        help="Quarantine a method above this family-method failure fraction.",
    ),
    allow_missing_babappalign: bool = typer.Option(
        False,
        "--allow-missing-babappalign",
        help="Allow BABAPPAlign to be skipped/failed when the BABAPPAScore model cache is missing.",
    ),
) -> None:
    """Run a mixed internal/external alignment ensemble."""
    try:
        summary = run_alignment_ensemble(
            ExternalAlignmentConfig(
                sim_dir=str(sim_dir),
                outdir=str(outdir),
                methods=_parse_methods(methods),
                seed=seed,
                require_available=_parse_bool(require_available),
                keep_intermediate=keep_intermediate,
                timeout_seconds=timeout_seconds,
                threads=threads,
                aligner_subprocess_threads=aligner_subprocess_threads,
                babappalign_device=babappalign_device,
                babappalign_backend=babappalign_backend,
                babappalign_workers=babappalign_workers,
                max_method_failure_fraction=max_method_failure_fraction,
                allow_missing_babappalign=allow_missing_babappalign,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not run alignment ensemble: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    table = Table(title="BABAPPA Alignment Ensemble")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Requested methods", _format_methods(summary["methods_requested"]))
    table.add_row("Run methods", _format_methods(summary["methods_run"]))
    table.add_row("Skipped methods", json.dumps(summary["methods_skipped"], sort_keys=True))
    table.add_row("Quarantined methods", json.dumps(summary["methods_quarantined"], sort_keys=True))
    table.add_row("Family-method OK", str(summary["n_family_method_ok"]))
    table.add_row("Family-method failed", str(summary["n_family_method_failed"]))
    table.add_row("Manifest", summary["manifest"])
    console.print(table)
    _print_warnings(summary["warnings"])


@app.command("build-site-map")
def build_site_map(
    sim_dir: Path = typer.Option(..., "--sim-dir"),
    align_dir: Path = typer.Option(..., "--align-dir"),
    outdir: Optional[Path] = typer.Option(None, "--outdir"),
    methods: Optional[str] = typer.Option(
        None,
        "--methods",
        help="Optional comma-separated methods. Defaults to alignment manifest methods.",
    ),
    require_complete: bool = typer.Option(False, "--require-complete"),
    workers: int = typer.Option(1, "--workers", help="Parallel family-method site-map workers."),
) -> None:
    """Build aligned-site to original-site coordinate maps."""
    try:
        summary = build_alignment_site_maps(
            SiteMapConfig(
                sim_dir=str(sim_dir),
                align_dir=str(align_dir),
                outdir=None if outdir is None else str(outdir),
                methods=_parse_optional_methods(methods),
                require_complete=require_complete,
                workers=workers,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not build site maps: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    table = Table(title="BABAPPA Alignment Site Maps")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Family-method maps", str(summary["n_family_method_maps"]))
    table.add_row("Unique fraction", str(summary["unique_fraction"]))
    table.add_row("Conflict fraction", str(summary["conflict_fraction"]))
    table.add_row("Frame-error fraction", str(summary["frame_error_fraction"]))
    table.add_row("Manifest", summary["manifest"])
    console.print(table)
    _print_warnings(summary["warnings"])


@app.command("validate-site-map")
def validate_site_map(
    site_map_dir: Path = typer.Option(..., "--site-map-dir"),
    methods: Optional[str] = typer.Option(
        None,
        "--methods",
        help="Optional comma-separated methods to enforce QC on.",
    ),
    max_conflict_fraction: Optional[float] = typer.Option(
        None,
        "--max-conflict-fraction",
        help="Fail selected non-quarantined methods above this conflict fraction.",
    ),
    quarantine_methods: Optional[str] = typer.Option(
        None,
        "--quarantine-methods",
        help="Comma-separated methods to report but exempt from selected-method QC failure.",
    ),
) -> None:
    """Validate alignment site-map artifacts."""
    summary = validate_site_map_dir(
        site_map_dir,
        methods=_parse_optional_methods(methods),
        max_conflict_fraction=max_conflict_fraction,
        quarantine_methods=_parse_optional_methods(quarantine_methods),
    )
    _print_validation_table("BABAPPA Alignment Site-Map Validation", summary, "n_maps")
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("aligner-method-policy")
def aligner_method_policy(
    align_dir: Path = typer.Option(..., "--align-dir"),
    site_map_dir: Path = typer.Option(..., "--site-map-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    max_conflict_fraction: float = typer.Option(0.03, "--max-conflict-fraction"),
    max_frame_error_fraction: float = typer.Option(0.0, "--max-frame-error-fraction"),
    max_method_failure_fraction: float = typer.Option(0.01, "--max-method-failure-fraction"),
) -> None:
    """Build method-level usability/quarantine policy for external aligners."""
    try:
        summary = build_method_policy(
            MethodPolicyConfig(
                align_dir=str(align_dir),
                site_map_dir=str(site_map_dir),
                outdir=str(outdir),
                max_conflict_fraction=max_conflict_fraction,
                max_frame_error_fraction=max_frame_error_fraction,
                max_method_failure_fraction=max_method_failure_fraction,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not build aligner method policy: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    table = Table(title="BABAPPA Aligner Method Policy")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Usable methods", _format_methods(summary["usable_methods"]))
    table.add_row("Quarantined methods", _format_methods(summary["quarantined_methods"]))
    table.add_row("Methods", str(summary["n_methods"]))
    table.add_row("Policy JSON", summary["json"])
    console.print(table)


@app.command("validate-aligner-method-policy")
def validate_aligner_method_policy(
    policy_dir: Path = typer.Option(..., "--policy-dir")
) -> None:
    """Validate aligner method-policy artifacts."""
    summary = validate_method_policy_dir(policy_dir)
    _print_validation_table("BABAPPA Aligner Method Policy Validation", summary)
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("build-tensors")
def build_tensors(
    sim_dir: Path = typer.Option(
        ...,
        "--sim-dir",
        help="Simulation output directory containing truth labels.",
    ),
    align_dir: Path = typer.Option(
        ...,
        "--align-dir",
        help="Alignment scaffold directory to tensorize.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for tensor shards.",
    ),
    methods: Optional[str] = typer.Option(
        None,
        "--methods",
        help="Comma-separated methods to tensorize. Defaults to all alignment methods.",
    ),
    workers: int = typer.Option(1, "--workers", help="Parallel family tensor workers."),
) -> None:
    """Build deterministic ML-ready tensor shards from alignment outputs."""
    parsed_methods = _parse_optional_methods(methods)
    try:
        config = TensorBuildConfig(
            sim_dir=str(sim_dir),
            align_dir=str(align_dir),
            outdir=str(outdir),
            methods=parsed_methods,
            workers=workers,
        )
        summary = build_tensor_dataset(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not build tensors: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Tensor Build Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Simulation directory", str(sim_dir))
    table.add_row("Alignment directory", str(align_dir))
    table.add_row("Tensor output directory", summary["outdir"])
    table.add_row("Families", str(summary["n_families"]))
    table.add_row("Methods", _format_methods(summary["methods"]))
    table.add_row("Manifest", summary["manifest"])
    table.add_row("Audit", summary["audit"])
    console.print(table)
    console.print(f"Tensor manifest path: {summary['manifest']}")
    console.print(f"Tensor audit path: {summary['audit']}")


@app.command("validate-tensors")
def validate_tensors(
    tensor_dir: Path = typer.Option(
        ...,
        "--tensor-dir",
        help="Tensor dataset directory to validate.",
    )
) -> None:
    """Validate a BABAPPA tensor dataset directory."""
    summary = validate_tensor_directory(tensor_dir)
    table = Table(title="BABAPPA Tensor Validation Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Tensor files checked", str(summary["n_tensor_files_checked"]))
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("index-dataset")
def index_dataset(
    tensor_dir: Path = typer.Option(
        ...,
        "--tensor-dir",
        help="Tensor dataset directory to index.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for dataset index files.",
    ),
    methods: Optional[str] = typer.Option(
        None,
        "--methods",
        help="Comma-separated methods to index. Defaults to all tensor methods.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for deterministic split assignment.",
    ),
    train_fraction: float = typer.Option(
        0.8,
        "--train-fraction",
        help="Fraction of split units assigned to train.",
    ),
    val_fraction: float = typer.Option(
        0.1,
        "--val-fraction",
        help="Fraction of split units assigned to validation.",
    ),
    calib_fraction: float = typer.Option(
        0.05,
        "--calib-fraction",
        help="Fraction of split units assigned to calibration.",
    ),
    test_fraction: float = typer.Option(
        0.05,
        "--test-fraction",
        help="Fraction of split units assigned to test.",
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        min=1,
        help="Parallel worker processes for tensor feature extraction.",
    ),
) -> None:
    """Build feature tables and deterministic dataset splits."""
    parsed_methods = _parse_optional_methods(methods)
    try:
        config = DatasetIndexConfig(
            tensor_dir=str(tensor_dir),
            outdir=str(outdir),
            methods=parsed_methods,
            seed=seed,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            calib_fraction=calib_fraction,
            test_fraction=test_fraction,
            workers=workers,
        )
        summary = build_dataset_index(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not index tensor dataset: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Dataset Index Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Tensor directory", str(tensor_dir))
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Rows", str(summary["n_rows"]))
    table.add_row("Families", str(summary["n_families"]))
    table.add_row("Methods", _format_methods(summary["methods"]))
    table.add_row("Workers", str(summary.get("workers", 1)))
    table.add_row("Features", summary["features"])
    table.add_row("Splits", summary["splits"])
    table.add_row("Index", summary["index"])
    console.print(table)
    console.print(f"Dataset index path: {summary['index']}")
    console.print(f"Features path: {summary['features']}")
    console.print(f"Splits path: {summary['splits']}")


@app.command("validate-index")
def validate_index(
    index_dir: Path = typer.Option(
        ...,
        "--index-dir",
        help="Dataset index directory to validate.",
    )
) -> None:
    """Validate a BABAPPA dataset index directory."""
    summary = validate_dataset_index(index_dir)
    table = Table(title="BABAPPA Dataset Index Validation Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Rows", str(summary["n_rows"]))
    table.add_row("Families", str(summary["n_families"]))
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("train-baseline")
def train_baseline(
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        help="Dataset index directory to train from.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for baseline model artifacts.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for deterministic initialization.",
    ),
    learning_rate: float = typer.Option(
        0.05,
        "--learning-rate",
        help="Full-batch gradient-descent learning rate.",
    ),
    epochs: int = typer.Option(
        300,
        "--epochs",
        help="Number of logistic-regression training epochs.",
    ),
    l2: float = typer.Option(
        0.001,
        "--l2",
        help="L2 regularization strength.",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        help="Probability threshold for predicted labels.",
    ),
) -> None:
    """Train the NumPy logistic-regression sanity baseline."""
    try:
        config = BaselineTrainConfig(
            dataset_dir=str(dataset_dir),
            outdir=str(outdir),
            seed=seed,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
            threshold=threshold,
        )
        summary = train_baseline_model(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not train baseline model: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Baseline Training Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Dataset directory", str(dataset_dir))
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Model", summary["model"])
    table.add_row("Predictions", summary["predictions"])
    table.add_row("Metrics", summary["metrics"])
    console.print(table)
    _print_metric_summary(summary["metrics_by_split"])
    console.print(f"Baseline model path: {summary['model']}")
    console.print(f"Baseline predictions path: {summary['predictions']}")
    console.print(f"Baseline metrics path: {summary['metrics']}")

    meta = _read_json(Path(summary["meta"]))
    if meta.get("warnings"):
        console.print("Warnings:")
        for warning in meta["warnings"]:
            console.print(f"- {warning}")


@app.command("validate-baseline")
def validate_baseline(
    model_dir: Path = typer.Option(
        ...,
        "--model-dir",
        help="Baseline model artifact directory to validate.",
    )
) -> None:
    """Validate a BABAPPA baseline model artifact directory."""
    summary = validate_baseline_model_dir(model_dir)
    table = Table(title="BABAPPA Baseline Validation Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Predictions", str(summary["n_predictions"]))
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("calibrate-baseline")
def calibrate_baseline(
    model_dir: Path = typer.Option(
        ...,
        "--model-dir",
        help="Baseline model artifact directory to calibrate.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for calibrated baseline artifacts.",
    ),
    target_fdr: float = typer.Option(
        0.10,
        "--target-fdr",
        help="Target empirical FDR for calibration-split threshold selection.",
    ),
    calibration_method: str = typer.Option(
        "temperature",
        "--calibration-method",
        help="Calibration method: none or temperature.",
    ),
    threshold_grid_size: int = typer.Option(
        181,
        "--threshold-grid-size",
        help="Number of empirical threshold candidates to evaluate.",
    ),
    min_threshold: float = typer.Option(
        0.05,
        "--min-threshold",
        help="Minimum threshold candidate.",
    ),
    max_threshold: float = typer.Option(
        0.95,
        "--max-threshold",
        help="Maximum threshold candidate.",
    ),
) -> None:
    """Calibrate baseline probabilities and select an empirical threshold."""
    try:
        config = BaselineCalibrationConfig(
            model_dir=str(model_dir),
            outdir=str(outdir),
            target_fdr=target_fdr,
            threshold_grid_size=threshold_grid_size,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            calibration_method=calibration_method,
        )
        summary = calibrate_baseline_model(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not calibrate baseline model: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Baseline Calibration Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Source model directory", str(model_dir))
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Method", calibration_method)
    table.add_row("Temperature", f"{float(summary['temperature']):.6f}")
    table.add_row("Selected threshold", f"{float(summary['selected_threshold']):.6f}")
    table.add_row("Target FDR", f"{target_fdr:.4f}")
    table.add_row("Calibration JSON", summary["calibration"])
    table.add_row("Calibrated predictions", summary["predictions"])
    table.add_row("Calibrated metrics", summary["metrics"])
    console.print(table)

    metrics = _read_json(Path(summary["metrics"]))
    _print_metric_summary(metrics.get("metrics_by_split_calibrated", {}))

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")


@app.command("validate-calibration")
def validate_calibration(
    calibration_dir: Path = typer.Option(
        ...,
        "--calibration-dir",
        help="Baseline calibration artifact directory to validate.",
    )
) -> None:
    """Validate a BABAPPA baseline calibration artifact directory."""
    summary = validate_baseline_calibration_dir(calibration_dir)
    table = Table(title="BABAPPA Calibration Validation Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Predictions", str(summary["n_predictions"]))
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("make-report")
def make_report(
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for report artifacts.",
    ),
    title: str = typer.Option(
        "BABAPPA run report",
        "--title",
        help="Human-readable report title.",
    ),
    sim_dir: Optional[Path] = typer.Option(
        None,
        "--sim-dir",
        help="Simulation output directory.",
    ),
    sim_audit_dir: Optional[Path] = typer.Option(
        None,
        "--sim-audit-dir",
        help="Simulation audit output directory.",
    ),
    align_dir: Optional[Path] = typer.Option(
        None,
        "--align-dir",
        help="Alignment scaffold output directory.",
    ),
    tensor_dir: Optional[Path] = typer.Option(
        None,
        "--tensor-dir",
        help="Tensor output directory.",
    ),
    dataset_dir: Optional[Path] = typer.Option(
        None,
        "--dataset-dir",
        help="Dataset index output directory.",
    ),
    baseline_dir: Optional[Path] = typer.Option(
        None,
        "--baseline-dir",
        help="Baseline model output directory.",
    ),
    calibration_dir: Optional[Path] = typer.Option(
        None,
        "--calibration-dir",
        help="Baseline calibration output directory.",
    ),
    neural_dir: Optional[Path] = typer.Option(
        None,
        "--neural-dir",
        help="Scale-ready neural training output directory.",
    ),
    neural_calibration_dir: Optional[Path] = typer.Option(
        None,
        "--neural-calibration-dir",
        help="Neural calibration output directory.",
    ),
    stratified_calibration_dir: Optional[Path] = typer.Option(
        None,
        "--stratified-calibration-dir",
        help="Stratified calibration output directory.",
    ),
    threshold_policy_dir: Optional[Path] = typer.Option(
        None,
        "--threshold-policy-dir",
        help="Threshold-policy output directory.",
    ),
    stratified_eval_dir: Optional[Path] = typer.Option(
        None,
        "--stratified-eval-dir",
        help="Stratified evaluation output directory.",
    ),
    neural_diagnostics_dir: Optional[Path] = typer.Option(
        None,
        "--neural-diagnostics-dir",
        help="Neural diagnostics output directory.",
    ),
    ablation_comparison_dir: Optional[Path] = typer.Option(
        None,
        "--ablation-comparison-dir",
        help="Neural ablation comparison output directory.",
    ),
    label_signal_audit_dir: Optional[Path] = typer.Option(
        None,
        "--label-signal-audit-dir",
        help="Label-signal audit output directory.",
    ),
    leakage_audit_dir: Optional[Path] = typer.Option(
        None,
        "--leakage-audit-dir",
        help="Leakage audit output directory.",
    ),
    stability_benchmark_dir: Optional[Path] = typer.Option(
        None,
        "--stability-benchmark-dir",
        help="Stability benchmark output directory.",
    ),
    site_label_dir: Optional[Path] = typer.Option(
        None,
        "--site-label-dir",
        help="Oracle site-label output directory.",
    ),
    site_dataset_dir: Optional[Path] = typer.Option(
        None,
        "--site-dataset-dir",
        help="Site-level dataset output directory.",
    ),
    site_leakage_audit_dir: Optional[Path] = typer.Option(
        None,
        "--site-leakage-audit-dir",
        help="Site leakage audit output directory.",
    ),
    site_baseline_dir: Optional[Path] = typer.Option(
        None,
        "--site-baseline-dir",
        help="Site baseline output directory.",
    ),
    site_neural_dir: Optional[Path] = typer.Option(
        None,
        "--site-neural-dir",
        help="Site neural output directory.",
    ),
    site_calibration_dir: Optional[Path] = typer.Option(
        None,
        "--site-calibration-dir",
        help="Site neural calibration output directory.",
    ),
    site_threshold_policy_dir: Optional[Path] = typer.Option(
        None,
        "--site-threshold-policy-dir",
        help="Site threshold-policy output directory.",
    ),
    site_stratified_eval_dir: Optional[Path] = typer.Option(
        None,
        "--site-stratified-eval-dir",
        help="Site stratified evaluation output directory.",
    ),
    site_aggregation_dir: Optional[Path] = typer.Option(
        None,
        "--site-aggregation-dir",
        help="Site-to-gene aggregation output directory.",
    ),
    site_stability_dir: Optional[Path] = typer.Option(None, "--site-stability-dir"),
    site_model_comparison_dir: Optional[Path] = typer.Option(
        None, "--site-model-comparison-dir"
    ),
    site_aggregation_controls_dir: Optional[Path] = typer.Option(
        None, "--site-aggregation-controls-dir"
    ),
    site_aggregation_threshold_policy_dir: Optional[Path] = typer.Option(
        None, "--site-aggregation-threshold-policy-dir"
    ),
    site_calibration_comparison_dir: Optional[Path] = typer.Option(
        None, "--site-calibration-comparison-dir"
    ),
    saturation_panel_dir: Optional[Path] = typer.Option(
        None,
        "--saturation-panel-dir",
        help="Saturation panel output directory.",
    ),
    merged_dataset_dir: Optional[Path] = typer.Option(
        None,
        "--merged-dataset-dir",
        help="Merged multi-saturation dataset directory.",
    ),
) -> None:
    """Generate a consolidated BABAPPA run report."""
    try:
        config = ReportConfig(
            sim_dir=_optional_path_to_str(sim_dir),
            sim_audit_dir=_optional_path_to_str(sim_audit_dir),
            align_dir=_optional_path_to_str(align_dir),
            tensor_dir=_optional_path_to_str(tensor_dir),
            dataset_dir=_optional_path_to_str(dataset_dir),
            baseline_dir=_optional_path_to_str(baseline_dir),
            calibration_dir=_optional_path_to_str(calibration_dir),
            neural_dir=_optional_path_to_str(neural_dir),
            neural_calibration_dir=_optional_path_to_str(neural_calibration_dir),
            stratified_calibration_dir=_optional_path_to_str(stratified_calibration_dir),
            threshold_policy_dir=_optional_path_to_str(threshold_policy_dir),
            stratified_eval_dir=_optional_path_to_str(stratified_eval_dir),
            neural_diagnostics_dir=_optional_path_to_str(neural_diagnostics_dir),
            ablation_comparison_dir=_optional_path_to_str(ablation_comparison_dir),
            label_signal_audit_dir=_optional_path_to_str(label_signal_audit_dir),
            leakage_audit_dir=_optional_path_to_str(leakage_audit_dir),
            stability_benchmark_dir=_optional_path_to_str(stability_benchmark_dir),
            site_label_dir=_optional_path_to_str(site_label_dir),
            site_dataset_dir=_optional_path_to_str(site_dataset_dir),
            site_leakage_audit_dir=_optional_path_to_str(site_leakage_audit_dir),
            site_baseline_dir=_optional_path_to_str(site_baseline_dir),
            site_neural_dir=_optional_path_to_str(site_neural_dir),
            site_calibration_dir=_optional_path_to_str(site_calibration_dir),
            site_threshold_policy_dir=_optional_path_to_str(site_threshold_policy_dir),
            site_stratified_eval_dir=_optional_path_to_str(site_stratified_eval_dir),
            site_aggregation_dir=_optional_path_to_str(site_aggregation_dir),
            site_stability_dir=_optional_path_to_str(site_stability_dir),
            site_model_comparison_dir=_optional_path_to_str(site_model_comparison_dir),
            site_aggregation_controls_dir=_optional_path_to_str(site_aggregation_controls_dir),
            site_aggregation_threshold_policy_dir=_optional_path_to_str(site_aggregation_threshold_policy_dir),
            site_calibration_comparison_dir=_optional_path_to_str(site_calibration_comparison_dir),
            saturation_panel_dir=_optional_path_to_str(saturation_panel_dir),
            merged_dataset_dir=_optional_path_to_str(merged_dataset_dir),
            outdir=str(outdir),
            title=title,
        )
        summary = generate_report(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not generate report: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Report Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Report output directory", summary["outdir"])
    table.add_row("JSON report", summary["json_report"])
    table.add_row("Markdown report", summary["markdown_report"])
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")


@app.command("validate-report")
def validate_report(
    report_dir: Path = typer.Option(
        ...,
        "--report-dir",
        help="BABAPPA report directory to validate.",
    )
) -> None:
    """Validate a BABAPPA consolidated report directory."""
    summary = validate_report_dir(report_dir)
    table = Table(title="BABAPPA Report Validation Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("summarize-run")
def summarize_run_command(
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for compact run summary artifacts.",
    ),
    title: str = typer.Option(
        "BABAPPA run summary",
        "--title",
        help="Human-readable run summary title.",
    ),
    sim_dir: Optional[Path] = typer.Option(None, "--sim-dir"),
    sim_audit_dir: Optional[Path] = typer.Option(None, "--sim-audit-dir"),
    align_dir: Optional[Path] = typer.Option(None, "--align-dir"),
    tensor_dir: Optional[Path] = typer.Option(None, "--tensor-dir"),
    dataset_dir: Optional[Path] = typer.Option(None, "--dataset-dir"),
    baseline_dir: Optional[Path] = typer.Option(None, "--baseline-dir"),
    baseline_calibration_dir: Optional[Path] = typer.Option(
        None,
        "--baseline-calibration-dir",
        help="Baseline calibration output directory.",
    ),
    neural_dir: Optional[Path] = typer.Option(None, "--neural-dir"),
    neural_calibration_dir: Optional[Path] = typer.Option(
        None,
        "--neural-calibration-dir",
        help="Neural calibration output directory.",
    ),
    stratified_calibration_dir: Optional[Path] = typer.Option(
        None,
        "--stratified-calibration-dir",
        help="Stratified calibration output directory.",
    ),
    threshold_policy_dir: Optional[Path] = typer.Option(
        None,
        "--threshold-policy-dir",
        help="Threshold-policy output directory.",
    ),
    stratified_eval_dir: Optional[Path] = typer.Option(
        None,
        "--stratified-eval-dir",
        help="Stratified evaluation output directory.",
    ),
    neural_diagnostics_dir: Optional[Path] = typer.Option(
        None,
        "--neural-diagnostics-dir",
        help="Neural diagnostics output directory.",
    ),
    ablation_comparison_dir: Optional[Path] = typer.Option(
        None,
        "--ablation-comparison-dir",
        help="Neural ablation comparison output directory.",
    ),
    label_signal_audit_dir: Optional[Path] = typer.Option(
        None,
        "--label-signal-audit-dir",
        help="Label-signal audit output directory.",
    ),
    leakage_audit_dir: Optional[Path] = typer.Option(
        None,
        "--leakage-audit-dir",
        help="Leakage audit output directory.",
    ),
    stability_benchmark_dir: Optional[Path] = typer.Option(
        None,
        "--stability-benchmark-dir",
        help="Stability benchmark output directory.",
    ),
    site_label_dir: Optional[Path] = typer.Option(
        None,
        "--site-label-dir",
        help="Oracle site-label output directory.",
    ),
    site_dataset_dir: Optional[Path] = typer.Option(
        None,
        "--site-dataset-dir",
        help="Site-level dataset output directory.",
    ),
    site_leakage_audit_dir: Optional[Path] = typer.Option(
        None,
        "--site-leakage-audit-dir",
        help="Site leakage audit output directory.",
    ),
    site_baseline_dir: Optional[Path] = typer.Option(
        None,
        "--site-baseline-dir",
        help="Site baseline output directory.",
    ),
    site_neural_dir: Optional[Path] = typer.Option(
        None,
        "--site-neural-dir",
        help="Site neural output directory.",
    ),
    site_calibration_dir: Optional[Path] = typer.Option(
        None,
        "--site-calibration-dir",
        help="Site neural calibration output directory.",
    ),
    site_threshold_policy_dir: Optional[Path] = typer.Option(
        None,
        "--site-threshold-policy-dir",
        help="Site threshold-policy output directory.",
    ),
    site_stratified_eval_dir: Optional[Path] = typer.Option(
        None,
        "--site-stratified-eval-dir",
        help="Site stratified evaluation output directory.",
    ),
    site_aggregation_dir: Optional[Path] = typer.Option(
        None,
        "--site-aggregation-dir",
        help="Site-to-gene aggregation output directory.",
    ),
    site_stability_dir: Optional[Path] = typer.Option(None, "--site-stability-dir"),
    site_model_comparison_dir: Optional[Path] = typer.Option(
        None, "--site-model-comparison-dir"
    ),
    site_aggregation_controls_dir: Optional[Path] = typer.Option(
        None, "--site-aggregation-controls-dir"
    ),
    site_aggregation_threshold_policy_dir: Optional[Path] = typer.Option(
        None, "--site-aggregation-threshold-policy-dir"
    ),
    site_calibration_comparison_dir: Optional[Path] = typer.Option(
        None, "--site-calibration-comparison-dir"
    ),
    saturation_panel_dir: Optional[Path] = typer.Option(
        None,
        "--saturation-panel-dir",
        help="Saturation panel output directory.",
    ),
    merged_dataset_dir: Optional[Path] = typer.Option(
        None,
        "--merged-dataset-dir",
        help="Merged multi-saturation dataset directory.",
    ),
    report_dir: Optional[Path] = typer.Option(None, "--report-dir"),
) -> None:
    """Generate a compact diagnostic summary for a BABAPPA run."""
    try:
        config = RunSummaryConfig(
            outdir=str(outdir),
            sim_dir=_optional_path_to_str(sim_dir),
            sim_audit_dir=_optional_path_to_str(sim_audit_dir),
            align_dir=_optional_path_to_str(align_dir),
            tensor_dir=_optional_path_to_str(tensor_dir),
            dataset_dir=_optional_path_to_str(dataset_dir),
            baseline_dir=_optional_path_to_str(baseline_dir),
            baseline_calibration_dir=_optional_path_to_str(
                baseline_calibration_dir
            ),
            neural_dir=_optional_path_to_str(neural_dir),
            neural_calibration_dir=_optional_path_to_str(neural_calibration_dir),
            stratified_calibration_dir=_optional_path_to_str(
                stratified_calibration_dir
            ),
            threshold_policy_dir=_optional_path_to_str(threshold_policy_dir),
            stratified_eval_dir=_optional_path_to_str(stratified_eval_dir),
            neural_diagnostics_dir=_optional_path_to_str(neural_diagnostics_dir),
            ablation_comparison_dir=_optional_path_to_str(ablation_comparison_dir),
            label_signal_audit_dir=_optional_path_to_str(label_signal_audit_dir),
            leakage_audit_dir=_optional_path_to_str(leakage_audit_dir),
            stability_benchmark_dir=_optional_path_to_str(stability_benchmark_dir),
            site_label_dir=_optional_path_to_str(site_label_dir),
            site_dataset_dir=_optional_path_to_str(site_dataset_dir),
            site_leakage_audit_dir=_optional_path_to_str(site_leakage_audit_dir),
            site_baseline_dir=_optional_path_to_str(site_baseline_dir),
            site_neural_dir=_optional_path_to_str(site_neural_dir),
            site_calibration_dir=_optional_path_to_str(site_calibration_dir),
            site_threshold_policy_dir=_optional_path_to_str(site_threshold_policy_dir),
            site_stratified_eval_dir=_optional_path_to_str(site_stratified_eval_dir),
            site_aggregation_dir=_optional_path_to_str(site_aggregation_dir),
            site_stability_dir=_optional_path_to_str(site_stability_dir),
            site_model_comparison_dir=_optional_path_to_str(site_model_comparison_dir),
            site_aggregation_controls_dir=_optional_path_to_str(site_aggregation_controls_dir),
            site_aggregation_threshold_policy_dir=_optional_path_to_str(site_aggregation_threshold_policy_dir),
            site_calibration_comparison_dir=_optional_path_to_str(site_calibration_comparison_dir),
            saturation_panel_dir=_optional_path_to_str(saturation_panel_dir),
            merged_dataset_dir=_optional_path_to_str(merged_dataset_dir),
            report_dir=_optional_path_to_str(report_dir),
            title=title,
        )
        summary = summarize_run(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not summarize run: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Run Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("JSON summary", summary["json_summary"])
    table.add_row("Markdown summary", summary["markdown_summary"])
    table.add_row("Recommended next action", summary["recommended_next_action"])
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")


@app.command("validate-run-summary")
def validate_run_summary(
    summary_dir: Path = typer.Option(
        ...,
        "--summary-dir",
        help="BABAPPA run summary directory to validate.",
    )
) -> None:
    """Validate a BABAPPA run summary directory."""
    summary = validate_run_summary_dir(summary_dir)
    table = Table(title="BABAPPA Run Summary Validation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("summarize-external-tiers")
def summarize_external_tiers_command(
    tiers: str = typer.Option(
        "low,moderate,high,extreme",
        "--tiers",
        help="Comma-separated external-aligner saturation tiers to summarize.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for the cross-tier external-aligner summary.",
    ),
    run_name: str = typer.Option(
        "external_aligner_validation",
        "--run-name",
        help="Directory infix used before each tier, e.g. external_aligner_validation or fast_external_10k.",
    ),
) -> None:
    """Summarize completed external-aligner tiers into one cross-tier report."""
    try:
        summary = summarize_external_tiers(
            ExternalTierSummaryConfig(tiers=tiers, outdir=str(outdir), run_name=run_name)
        )
    except (OSError, ValueError) as exc:
        console.print(
            f"Error: could not summarize external tiers: {exc}",
            style="red",
        )
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA External Tier Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Tiers included", ",".join(summary["tiers_included"]))
    table.add_row(
        "Recommended 10K methods",
        ",".join(summary["recommended_10k_method_set"]),
    )
    table.add_row("JSON summary", summary["json_summary"])
    table.add_row("Markdown summary", summary["markdown_summary"])
    table.add_row("Warnings", str(summary["n_warning"]))
    console.print(table)
    _print_warnings(summary["warnings"])


@app.command("validate-external-tier-summary")
def validate_external_tier_summary(
    summary_dir: Path = typer.Option(
        ...,
        "--summary-dir",
        help="BABAPPA external-tier summary directory to validate.",
    )
) -> None:
    """Validate a cross-tier external-aligner summary directory."""
    summary = validate_external_tier_summary_dir(summary_dir)
    _print_validation_table("BABAPPA External Tier Summary Validation", summary)
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("compare-models")
def compare_models_command(
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for model comparison artifacts.",
    ),
    baseline_metrics: Optional[Path] = typer.Option(
        None,
        "--baseline-metrics",
        help="Path to baseline_metrics.json.",
    ),
    baseline_calibrated_metrics: Optional[Path] = typer.Option(
        None,
        "--baseline-calibrated-metrics",
        help="Path to baseline_calibrated_metrics.json.",
    ),
    neural_metrics: Optional[Path] = typer.Option(
        None,
        "--neural-metrics",
        help="Path to neural_metrics.json.",
    ),
    neural_calibrated_metrics: Optional[Path] = typer.Option(
        None,
        "--neural-calibrated-metrics",
        help="Path to neural_calibrated_metrics.json.",
    ),
    title: str = typer.Option(
        "BABAPPA model comparison",
        "--title",
        help="Human-readable comparison title.",
    ),
) -> None:
    """Compare baseline and neural raw/calibrated metrics."""
    try:
        config = ModelCompareConfig(
            outdir=str(outdir),
            baseline_metrics=_optional_path_to_str(baseline_metrics),
            baseline_calibrated_metrics=_optional_path_to_str(
                baseline_calibrated_metrics
            ),
            neural_metrics=_optional_path_to_str(neural_metrics),
            neural_calibrated_metrics=_optional_path_to_str(
                neural_calibrated_metrics
            ),
            title=title,
        )
        summary = compare_models(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not compare models: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Model Comparison")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("JSON", summary["json"])
    table.add_row("TSV", summary["tsv"])
    table.add_row("Markdown", summary["markdown"])
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")


@app.command("validate-model-comparison")
def validate_model_comparison(
    compare_dir: Path = typer.Option(
        ...,
        "--compare-dir",
        help="BABAPPA model comparison directory to validate.",
    )
) -> None:
    """Validate a BABAPPA model comparison directory."""
    summary = validate_model_comparison_dir(compare_dir)
    table = Table(title="BABAPPA Model Comparison Validation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("compare-ablations")
def compare_ablations_command(
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for neural ablation comparison artifacts.",
    ),
    model_dirs: str = typer.Option(
        ...,
        "--model-dirs",
        help="Comma-separated neural model artifact directories.",
    ),
    names: Optional[str] = typer.Option(
        None,
        "--names",
        help="Optional comma-separated model names.",
    ),
    stratified_eval_dirs: Optional[str] = typer.Option(
        None,
        "--stratified-eval-dirs",
        help="Optional comma-separated stratified evaluation directories.",
    ),
    threshold_policy_dirs: Optional[str] = typer.Option(
        None,
        "--threshold-policy-dirs",
        help="Optional comma-separated threshold-policy directories.",
    ),
    neural_diagnostics_dirs: Optional[str] = typer.Option(
        None,
        "--neural-diagnostics-dirs",
        help="Optional comma-separated neural diagnostics directories.",
    ),
    title: str = typer.Option(
        "BABAPPA neural ablation comparison",
        "--title",
        help="Human-readable ablation comparison title.",
    ),
) -> None:
    """Compare neural ablation variants using existing metrics."""
    try:
        config = AblationCompareConfig(
            outdir=str(outdir),
            model_dirs=_parse_methods(model_dirs),
            names=_parse_optional_methods(names),
            stratified_eval_dirs=_parse_optional_methods(stratified_eval_dirs),
            threshold_policy_dirs=_parse_optional_methods(threshold_policy_dirs),
            neural_diagnostics_dirs=_parse_optional_methods(neural_diagnostics_dirs),
            title=title,
        )
        summary = compare_neural_ablations(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not compare ablations: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Neural Ablation Comparison")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("JSON", summary["json"])
    table.add_row("TSV", summary["tsv"])
    table.add_row("Markdown", summary["markdown"])
    table.add_row(
        "Recommended model", str(summary["recommendation"].get("best_model"))
    )
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")


@app.command("validate-ablation-comparison")
def validate_ablation_comparison(
    compare_dir: Path = typer.Option(
        ...,
        "--compare-dir",
        help="BABAPPA neural ablation comparison directory to validate.",
    )
) -> None:
    """Validate a BABAPPA neural ablation comparison directory."""
    summary = validate_ablation_comparison_dir(compare_dir)
    table = Table(title="BABAPPA Neural Ablation Comparison Validation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("audit-label-signal")
def audit_label_signal_command(
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        help="Dataset directory containing features.tsv and splits.tsv.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for label-signal audit artifacts.",
    ),
    label_column: str = typer.Option("gene_label", "--label-column"),
    saturation_column: str = typer.Option("saturation_tier", "--saturation-column"),
    method_column: str = typer.Option("method", "--method-column"),
    split_column: str = typer.Option("split", "--split-column"),
) -> None:
    """Audit univariate label signal in dataset feature tables."""
    try:
        config = LabelSignalAuditConfig(
            dataset_dir=str(dataset_dir),
            outdir=str(outdir),
            label_column=label_column,
            saturation_column=saturation_column,
            method_column=method_column,
            split_column=split_column,
        )
        summary = audit_label_signal(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not audit label signal: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Label-Signal Audit")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("JSON", summary["json"])
    table.add_row("Feature summary", summary["features"])
    table.add_row("Markdown", summary["markdown"])
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)
    console.print(summary["interpretation"])


@app.command("validate-label-signal-audit")
def validate_label_signal_audit(
    audit_dir: Path = typer.Option(
        ...,
        "--audit-dir",
        help="Label-signal audit directory to validate.",
    )
) -> None:
    """Validate label-signal audit artifacts."""
    summary = validate_label_signal_audit_dir(audit_dir)
    table = Table(title="BABAPPA Label-Signal Audit Validation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("audit-leakage")
def audit_leakage_command(
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        help="Dataset directory containing features.tsv and splits.tsv.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for leakage audit artifacts.",
    ),
    label_column: str = typer.Option("gene_label", "--label-column"),
    split_column: str = typer.Option("split", "--split-column"),
    method_column: str = typer.Option("method", "--method-column"),
    saturation_column: str = typer.Option("saturation_tier", "--saturation-column"),
) -> None:
    """Audit dataset feature tables for truth-derived leakage columns."""
    try:
        config = LeakageAuditConfig(
            dataset_dir=str(dataset_dir),
            outdir=str(outdir),
            label_column=label_column,
            split_column=split_column,
            method_column=method_column,
            saturation_column=saturation_column,
        )
        summary = audit_leakage(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not audit leakage: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Leakage Audit")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Leakage status", summary["leakage_status"])
    table.add_row("JSON", summary["json"])
    table.add_row("Columns TSV", summary["columns"])
    table.add_row("Markdown", summary["markdown"])
    table.add_row(
        "Recommended exclusions",
        _format_methods(summary["recommended_excluded_columns"]),
    )
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)


@app.command("validate-leakage-audit")
def validate_leakage_audit(
    audit_dir: Path = typer.Option(
        ...,
        "--audit-dir",
        help="Leakage audit directory to validate.",
    )
) -> None:
    """Validate leakage audit artifacts."""
    summary = validate_leakage_audit_dir(audit_dir)
    table = Table(title="BABAPPA Leakage Audit Validation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("check-neural-env")
def check_neural_env(
    prefer_mps: bool = typer.Option(
        False,
        "--prefer-mps/--no-prefer-mps",
        help="Prefer MPS over CUDA when recommending an auto device.",
    ),
) -> None:
    """Inspect optional PyTorch and accelerator availability."""
    env = get_torch_environment(prefer_mps=prefer_mps)
    table = Table(title="BABAPPA Neural Environment")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Platform system", str(env["platform_system"]))
    table.add_row("Platform machine", str(env["platform_machine"]))
    table.add_row("macOS version", str(env["macos_version"] or "n/a"))
    table.add_row("Python executable", str(env["python_executable"]))
    table.add_row("Torch available", str(env["torch_available"]))
    table.add_row("Torch version", str(env["torch_version"]))
    table.add_row("CUDA available", str(env["cuda_available"]))
    table.add_row("CUDA device count", str(env["cuda_device_count"]))
    table.add_row(
        "CUDA device names",
        _format_methods(env["cuda_device_names"]) if env["cuda_device_names"] else "none",
    )
    table.add_row("MPS built", str(env["mps_built"]))
    table.add_row("MPS available", str(env["mps_available"]))
    table.add_row("Recommended device", str(env["recommended_device"]))
    table.add_row("PYTORCH_ENABLE_MPS_FALLBACK", str(env["mps_fallback_env"] or "unset"))
    table.add_row("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(env["mps_high_watermark_env"] or "unset"))
    console.print(table)

    if env["warnings"]:
        console.print("Warnings:")
        for warning in env["warnings"]:
            console.print(f"- {warning}")


@app.command("smoke-mps-training")
def smoke_mps_training(
    outdir: Path = typer.Option("mps_smoke", "--outdir"),
    dataset_dir: Optional[Path] = typer.Option(None, "--dataset-dir"),
    device: str = typer.Option("mps", "--device", help="Device: auto, cpu, cuda, or mps."),
    batch_size: int = typer.Option(32, "--batch-size"),
    max_items: int = typer.Option(512, "--max-items"),
    threads: int = typer.Option(8, "--threads"),
) -> None:
    """Run a tiny MPS forward/backward/checkpoint smoke, or skip gracefully."""
    try:
        summary = run_mps_training_smoke(
            MPSTrainingSmokeConfig(
                outdir=str(outdir),
                dataset_dir=_optional_path_to_str(dataset_dir),
                device=device,
                batch_size=batch_size,
                max_items=max_items,
                threads=threads,
            )
        )
    except (OSError, ValueError, RuntimeError) as exc:
        console.print(f"Error: could not run MPS smoke: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA MPS Smoke",
        summary,
        ["outdir", "status", "device_used", "json", "markdown", "warnings"],
    )


@app.command("validate-mps-smoke")
def validate_mps_smoke(
    smoke_dir: Path = typer.Option(..., "--smoke-dir"),
) -> None:
    """Validate MPS smoke artifacts."""
    summary = validate_mps_smoke_dir(smoke_dir)
    _print_validation_table("BABAPPA MPS Smoke Validation", summary)
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("benchmark-apple-silicon")
def benchmark_apple_silicon(
    outdir: Path = typer.Option("apple_silicon_benchmark", "--outdir"),
    device: str = typer.Option("auto", "--device", help="Device: auto, cpu, cuda, or mps."),
    batch_sizes: str = typer.Option("32,64,128,256", "--batch-sizes"),
    max_items: int = typer.Option(4096, "--max-items"),
    threads: int = typer.Option(8, "--threads"),
    prefer_mps: bool = typer.Option(False, "--prefer-mps/--no-prefer-mps"),
) -> None:
    """Run a lightweight synthetic branch-neural benchmark for Apple Silicon sizing."""
    try:
        summary = run_apple_silicon_benchmark(
            AppleSiliconBenchmarkConfig(
                outdir=str(outdir),
                device=device,
                batch_sizes=batch_sizes,
                max_items=max_items,
                threads=threads,
                prefer_mps=prefer_mps,
            )
        )
    except (OSError, ValueError, RuntimeError) as exc:
        console.print(f"Error: could not run Apple Silicon benchmark: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Apple Silicon Benchmark",
        summary,
        ["outdir", "status", "device_used", "recommended_batch_size", "json", "markdown", "warnings"],
    )


@app.command("inspect-neural-data")
def inspect_neural_data(
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        help="Dataset index directory to inspect.",
    ),
    split: str = typer.Option(
        "train",
        "--split",
        help="Split to inspect: train, val, calib, test, or all.",
    ),
    methods: Optional[str] = typer.Option(
        None,
        "--methods",
        help="Optional comma-separated method filter.",
    ),
    max_items: Optional[int] = typer.Option(
        None,
        "--max-items",
        help="Optional maximum number of sorted rows to inspect.",
    ),
) -> None:
    """Inspect neural dataset rows and one example tensor shard."""
    try:
        config = NeuralDatasetConfig(
            dataset_dir=str(dataset_dir),
            split=split,
            methods=_parse_optional_methods(methods),
            max_items=max_items,
        )
        summary = inspect_neural_dataset(config)
    except (OSError, ValueError, FileNotFoundError) as exc:
        console.print(f"Error: could not inspect neural data: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Neural Data Inspection")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Dataset directory", summary["dataset_dir"])
    table.add_row("Split", summary["split"])
    table.add_row("Methods filter", _format_methods(summary["methods"]) or "all")
    table.add_row("Rows", str(summary["n_rows"]))
    table.add_row("Families", str(summary["n_families"]))
    table.add_row("Class counts", _format_mapping(summary["class_counts"]))
    table.add_row("Methods present", _format_methods(summary["methods_present"]))
    table.add_row("Example tensor shape", str(summary["example_shape"]))
    table.add_row("Example dtype", str(summary["example_dtype"]))
    table.add_row("Example tensor file", str(summary["example_tensor_file"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")


@app.command("smoke-neural-batch")
def smoke_neural_batch(
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        help="Dataset index directory to batch from.",
    ),
    split: str = typer.Option(
        "train",
        "--split",
        help="Split to batch: train, val, calib, test, or all.",
    ),
    methods: Optional[str] = typer.Option(
        None,
        "--methods",
        help="Optional comma-separated method filter.",
    ),
    batch_size: int = typer.Option(
        4,
        "--batch-size",
        help="Number of items to collate into a smoke batch.",
    ),
) -> None:
    """Create one small neural data batch without training a model."""
    try:
        config = NeuralDatasetConfig(
            dataset_dir=str(dataset_dir),
            split=split,
            methods=_parse_optional_methods(methods),
            require_torch=True,
        )
        summary = make_smoke_batch(config, batch_size=batch_size)
    except RuntimeError as exc:
        console.print(str(exc), style="red")
        raise typer.Exit(code=1) from exc
    except (OSError, ValueError, FileNotFoundError) as exc:
        console.print(f"Error: could not create neural smoke batch: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Neural Smoke Batch")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Batch size", str(summary["batch_size"]))
    table.add_row("X shape", str(summary["X_shape"]))
    table.add_row("y shape", str(summary["y_shape"]))
    table.add_row("X dtype", summary["X_dtype"])
    table.add_row("y dtype", summary["y_dtype"])
    table.add_row("Methods", _format_methods(summary["methods"]))
    table.add_row("Split", summary["split"])
    console.print(table)


@app.command("train-neural-smoke")
def train_neural_smoke(
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        help="Dataset index directory to train from.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for neural smoke artifacts.",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device: auto, cpu, cuda, or mps.",
    ),
    methods: Optional[str] = typer.Option(
        None,
        "--methods",
        help="Optional comma-separated method filter.",
    ),
    epochs: int = typer.Option(
        5,
        "--epochs",
        help="Small smoke-training epoch count.",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        help="Training batch size.",
    ),
    learning_rate: float = typer.Option(
        0.001,
        "--learning-rate",
        help="AdamW learning rate.",
    ),
    weight_decay: float = typer.Option(
        0.0001,
        "--weight-decay",
        help="AdamW weight decay.",
    ),
    embedding_dim: int = typer.Option(
        16,
        "--embedding-dim",
        help="Codon embedding dimension.",
    ),
    hidden_dim: int = typer.Option(
        32,
        "--hidden-dim",
        help="Hidden MLP dimension.",
    ),
    dropout: float = typer.Option(
        0.1,
        "--dropout",
        help="MLP dropout probability.",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        help="Probability threshold for predictions.",
    ),
    max_train_items: Optional[int] = typer.Option(
        None,
        "--max-train-items",
        help="Optional cap on sorted train rows for smoke tests.",
    ),
    max_val_items: Optional[int] = typer.Option(
        None,
        "--max-val-items",
        help="Optional cap on sorted validation rows for smoke tests.",
    ),
) -> None:
    """Train the minimal PyTorch gene-level smoke classifier."""
    try:
        config = NeuralTrainConfig(
            dataset_dir=str(dataset_dir),
            outdir=str(outdir),
            device=device,
            methods=_parse_optional_methods(methods),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            threshold=threshold,
            max_train_items=max_train_items,
            max_val_items=max_val_items,
        )
        summary = train_neural_smoke_model(config)
    except RuntimeError as exc:
        console.print(str(exc), style="red")
        raise typer.Exit(code=1) from exc
    except (OSError, ValueError, FileNotFoundError) as exc:
        console.print(f"Error: could not train neural smoke model: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Neural Smoke Training Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Dataset directory", str(dataset_dir))
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Device used", summary["device_used"])
    table.add_row("Checkpoint", summary["checkpoint"])
    table.add_row("History", summary["history"])
    table.add_row("Predictions", summary["predictions"])
    table.add_row("Metrics", summary["metrics"])
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")


@app.command("validate-neural-smoke")
def validate_neural_smoke(
    model_dir: Path = typer.Option(
        ...,
        "--model-dir",
        help="Neural smoke model artifact directory to validate.",
    )
) -> None:
    """Validate neural smoke-training artifacts."""
    summary = validate_neural_smoke_dir(model_dir)
    table = Table(title="BABAPPA Neural Smoke Validation Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Predictions", str(summary["n_predictions"]))
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("train-neural")
def train_neural(
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        help="Dataset index directory to train from.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for scale-ready neural artifacts.",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device: auto, cpu, cuda, or mps.",
    ),
    methods: Optional[str] = typer.Option(
        None,
        "--methods",
        help="Optional comma-separated method filter.",
    ),
    epochs: int = typer.Option(
        30,
        "--epochs",
        help="Maximum training epochs.",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Training batch size.",
    ),
    learning_rate: float = typer.Option(
        0.001,
        "--learning-rate",
        help="AdamW learning rate.",
    ),
    weight_decay: float = typer.Option(
        0.0001,
        "--weight-decay",
        help="AdamW weight decay.",
    ),
    embedding_dim: int = typer.Option(
        32,
        "--embedding-dim",
        help="Codon embedding dimension.",
    ),
    hidden_dim: int = typer.Option(
        64,
        "--hidden-dim",
        help="Hidden MLP dimension.",
    ),
    dropout: float = typer.Option(
        0.1,
        "--dropout",
        help="MLP dropout probability.",
    ),
    saturation_embedding_dim: int = typer.Option(
        8,
        "--saturation-embedding-dim",
        help="Saturation-tier embedding dimension for saturation-aware models.",
    ),
    architecture: str = typer.Option(
        "saturation_aware",
        "--architecture",
        help=(
            "Architecture: small, contrastive, saturation_aware, "
            "site_attention, or site_attention_saturation."
        ),
    ),
    positive_class_weight: str = typer.Option(
        "auto",
        "--positive-class-weight",
        help="Positive class weighting: none or auto.",
    ),
    group_weighting: str = typer.Option(
        "none",
        "--group-weighting",
        help="Group weighting: none or saturation_inverse_frequency.",
    ),
    sampler: str = typer.Option(
        "none",
        "--sampler",
        help="Sampler: none or saturation_balanced.",
    ),
    training_preset: Optional[str] = typer.Option(
        None,
        "--training-preset",
        help=(
            "Optional preset: contrastive_v2, saturation_embed_only, "
            "saturation_group_weight_only, saturation_sampler_only, "
            "saturation_full_v3, contrastive_class_weighted, contrastive_unweighted, "
            "contrastive_ranked, site_attention_ranked, "
            "site_attention_focal_ranked, or site_attention_saturation_ranked."
        ),
    ),
    loss_mode: str = typer.Option(
        "bce_rank",
        "--loss-mode",
        help="Loss mode: bce, bce_rank, focal, or focal_rank.",
    ),
    rank_weight: float = typer.Option(
        0.2,
        "--rank-weight",
        help="Weight for pairwise rank loss in ranked loss modes.",
    ),
    focal_gamma: float = typer.Option(
        2.0,
        "--focal-gamma",
        help="Focal loss gamma for focal loss modes.",
    ),
    min_delta: float = typer.Option(
        0.0,
        "--min-delta",
        help="Minimum monitored-metric improvement for early stopping.",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        help="Probability threshold for predictions.",
    ),
    early_stopping_patience: int = typer.Option(
        8,
        "--early-stopping-patience",
        help="Epochs to wait for monitored-metric improvement.",
    ),
    monitor_metric: str = typer.Option(
        "val_loss",
        "--monitor-metric",
        help="Metric to monitor: val_loss, val_auroc, or val_accuracy.",
    ),
    max_train_items: Optional[int] = typer.Option(
        None,
        "--max-train-items",
        help="Optional cap on sorted train rows for smoke tests.",
    ),
    max_val_items: Optional[int] = typer.Option(
        None,
        "--max-val-items",
        help="Optional cap on sorted validation rows for smoke tests.",
    ),
    max_calib_items: Optional[int] = typer.Option(
        None,
        "--max-calib-items",
        help="Optional cap on sorted calibration rows for smoke tests.",
    ),
    max_test_items: Optional[int] = typer.Option(
        None,
        "--max-test-items",
        help="Optional cap on sorted test rows for smoke tests.",
    ),
    save_every_epoch: bool = typer.Option(
        False,
        "--save-every-epoch",
        help="Save epoch-specific checkpoint files.",
    ),
) -> None:
    """Train the scale-ready gene-level neural model."""
    try:
        config = NeuralFullTrainConfig(
            dataset_dir=str(dataset_dir),
            outdir=str(outdir),
            device=device,
            methods=_parse_optional_methods(methods),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            architecture=architecture,
            saturation_embedding_dim=saturation_embedding_dim,
            positive_class_weight=positive_class_weight,
            group_weighting=group_weighting,
            sampler=sampler,
            training_preset=training_preset,
            loss_mode=loss_mode,
            rank_weight=rank_weight,
            focal_gamma=focal_gamma,
            min_delta=min_delta,
            threshold=threshold,
            max_train_items=max_train_items,
            max_val_items=max_val_items,
            max_calib_items=max_calib_items,
            max_test_items=max_test_items,
            early_stopping_patience=early_stopping_patience,
            monitor_metric=monitor_metric,
            save_every_epoch=save_every_epoch,
        )
        summary = train_neural_model(config)
    except RuntimeError as exc:
        console.print(str(exc), style="red")
        raise typer.Exit(code=1) from exc
    except (OSError, ValueError, FileNotFoundError) as exc:
        console.print(f"Error: could not train neural model: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Neural Training Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Dataset directory", str(dataset_dir))
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Device used", summary["device_used"])
    table.add_row("Architecture", architecture)
    table.add_row("Positive class weight", positive_class_weight)
    table.add_row("Group weighting", group_weighting)
    table.add_row("Sampler", sampler)
    table.add_row("Training preset", str(training_preset))
    table.add_row("Loss mode", loss_mode)
    table.add_row("Rank weight", str(rank_weight))
    table.add_row("Focal gamma", str(focal_gamma))
    table.add_row("Best checkpoint", summary["best_checkpoint"])
    table.add_row("Last checkpoint", summary["last_checkpoint"])
    table.add_row("Metadata", summary["meta"])
    table.add_row("History", summary["history"])
    table.add_row("Predictions", summary["predictions"])
    table.add_row("Metrics", summary["metrics"])
    table.add_row("Best epoch", str(summary["best_epoch"]))
    table.add_row("Stopped early", str(summary["stopped_early"]))
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")


@app.command("train-neural-v2")
def train_neural_v2(
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        help="Dataset index directory to train from.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for neural v2 artifacts.",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device: auto, cpu, cuda, or mps.",
    ),
    methods: Optional[str] = typer.Option(
        None,
        "--methods",
        help="Optional comma-separated method filter.",
    ),
    epochs: int = typer.Option(30, "--epochs", help="Maximum training epochs."),
    batch_size: int = typer.Option(32, "--batch-size", help="Training batch size."),
    learning_rate: float = typer.Option(
        0.001,
        "--learning-rate",
        help="AdamW learning rate.",
    ),
    weight_decay: float = typer.Option(
        0.0001,
        "--weight-decay",
        help="AdamW weight decay.",
    ),
    embedding_dim: int = typer.Option(
        32,
        "--embedding-dim",
        help="Codon embedding dimension.",
    ),
    hidden_dim: int = typer.Option(
        64,
        "--hidden-dim",
        help="Hidden MLP dimension.",
    ),
    dropout: float = typer.Option(
        0.1,
        "--dropout",
        help="MLP dropout probability.",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        help="Probability threshold for predictions.",
    ),
    early_stopping_patience: int = typer.Option(
        8,
        "--early-stopping-patience",
        help="Epochs to wait for monitored-metric improvement.",
    ),
    monitor_metric: str = typer.Option(
        "val_loss",
        "--monitor-metric",
        help="Metric to monitor: val_loss, val_auroc, or val_accuracy.",
    ),
    max_train_items: Optional[int] = typer.Option(
        None,
        "--max-train-items",
        help="Optional cap on sorted train rows for smoke tests.",
    ),
    max_val_items: Optional[int] = typer.Option(
        None,
        "--max-val-items",
        help="Optional cap on sorted validation rows for smoke tests.",
    ),
    max_calib_items: Optional[int] = typer.Option(
        None,
        "--max-calib-items",
        help="Optional cap on sorted calibration rows for smoke tests.",
    ),
    max_test_items: Optional[int] = typer.Option(
        None,
        "--max-test-items",
        help="Optional cap on sorted test rows for smoke tests.",
    ),
) -> None:
    """Train neural v2: contrastive pooling with automatic class weighting."""
    train_neural(
        dataset_dir=dataset_dir,
        outdir=outdir,
        device=device,
        methods=methods,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        saturation_embedding_dim=8,
        architecture="contrastive",
        positive_class_weight="auto",
        group_weighting="none",
        sampler="none",
        training_preset="contrastive_v2",
        loss_mode="bce",
        rank_weight=0.0,
        focal_gamma=2.0,
        min_delta=0.0,
        threshold=threshold,
        early_stopping_patience=early_stopping_patience,
        monitor_metric=monitor_metric,
        max_train_items=max_train_items,
        max_val_items=max_val_items,
        max_calib_items=max_calib_items,
        max_test_items=max_test_items,
        save_every_epoch=False,
    )


@app.command("train-neural-saturation")
def train_neural_saturation(
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        help="Merged saturation-aware dataset directory to train from.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for saturation-aware neural artifacts.",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device: auto, cpu, cuda, or mps.",
    ),
    methods: Optional[str] = typer.Option(
        None,
        "--methods",
        help="Optional comma-separated method filter.",
    ),
    epochs: int = typer.Option(30, "--epochs", help="Maximum training epochs."),
    batch_size: int = typer.Option(32, "--batch-size", help="Training batch size."),
    learning_rate: float = typer.Option(
        0.001,
        "--learning-rate",
        help="AdamW learning rate.",
    ),
    weight_decay: float = typer.Option(
        0.0001,
        "--weight-decay",
        help="AdamW weight decay.",
    ),
    embedding_dim: int = typer.Option(
        32,
        "--embedding-dim",
        help="Codon embedding dimension.",
    ),
    hidden_dim: int = typer.Option(
        64,
        "--hidden-dim",
        help="Hidden MLP dimension.",
    ),
    dropout: float = typer.Option(
        0.1,
        "--dropout",
        help="MLP dropout probability.",
    ),
    saturation_embedding_dim: int = typer.Option(
        8,
        "--saturation-embedding-dim",
        help="Saturation-tier embedding dimension.",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        help="Probability threshold for predictions.",
    ),
    early_stopping_patience: int = typer.Option(
        8,
        "--early-stopping-patience",
        help="Epochs to wait for monitored-metric improvement.",
    ),
    monitor_metric: str = typer.Option(
        "val_loss",
        "--monitor-metric",
        help="Metric to monitor: val_loss, val_auroc, or val_accuracy.",
    ),
    min_delta: float = typer.Option(
        0.0,
        "--min-delta",
        help="Minimum monitored-metric improvement for early stopping.",
    ),
    max_train_items: Optional[int] = typer.Option(
        None,
        "--max-train-items",
        help="Optional cap on sorted train rows for smoke tests.",
    ),
    max_val_items: Optional[int] = typer.Option(
        None,
        "--max-val-items",
        help="Optional cap on sorted validation rows for smoke tests.",
    ),
    max_calib_items: Optional[int] = typer.Option(
        None,
        "--max-calib-items",
        help="Optional cap on sorted calibration rows for smoke tests.",
    ),
    max_test_items: Optional[int] = typer.Option(
        None,
        "--max-test-items",
        help="Optional cap on sorted test rows for smoke tests.",
    ),
) -> None:
    """Train saturation-aware neural v3 with tier embeddings and balanced loss."""
    train_neural(
        dataset_dir=dataset_dir,
        outdir=outdir,
        device=device,
        methods=methods,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        saturation_embedding_dim=saturation_embedding_dim,
        architecture="saturation_aware",
        positive_class_weight="auto",
        group_weighting="saturation_inverse_frequency",
        sampler="saturation_balanced",
        training_preset="saturation_full_v3",
        loss_mode="bce",
        rank_weight=0.0,
        focal_gamma=2.0,
        min_delta=min_delta,
        threshold=threshold,
        early_stopping_patience=early_stopping_patience,
        monitor_metric=monitor_metric,
        max_train_items=max_train_items,
        max_val_items=max_val_items,
        max_calib_items=max_calib_items,
        max_test_items=max_test_items,
        save_every_epoch=False,
    )


@app.command("train-neural-ranking")
def train_neural_ranking(
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        help="Dataset index directory to train from.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for ranking-aware neural artifacts.",
    ),
    device: str = typer.Option("auto", "--device", help="Device: auto, cpu, cuda, or mps."),
    methods: Optional[str] = typer.Option(
        None,
        "--methods",
        help="Optional comma-separated method filter.",
    ),
    epochs: int = typer.Option(30, "--epochs", help="Maximum training epochs."),
    batch_size: int = typer.Option(32, "--batch-size", help="Training batch size."),
    learning_rate: float = typer.Option(0.001, "--learning-rate", help="AdamW learning rate."),
    weight_decay: float = typer.Option(0.0001, "--weight-decay", help="AdamW weight decay."),
    embedding_dim: int = typer.Option(32, "--embedding-dim", help="Codon embedding dimension."),
    hidden_dim: int = typer.Option(64, "--hidden-dim", help="Hidden MLP dimension."),
    dropout: float = typer.Option(0.1, "--dropout", help="MLP dropout probability."),
    threshold: float = typer.Option(0.5, "--threshold", help="Probability threshold."),
    early_stopping_patience: int = typer.Option(
        8,
        "--early-stopping-patience",
        help="Epochs to wait for monitored-metric improvement.",
    ),
    monitor_metric: str = typer.Option(
        "val_loss",
        "--monitor-metric",
        help="Metric to monitor: val_loss, val_auroc, or val_accuracy.",
    ),
    max_train_items: Optional[int] = typer.Option(None, "--max-train-items"),
    max_val_items: Optional[int] = typer.Option(None, "--max-val-items"),
    max_calib_items: Optional[int] = typer.Option(None, "--max-calib-items"),
    max_test_items: Optional[int] = typer.Option(None, "--max-test-items"),
    rank_weight: float = typer.Option(
        0.2,
        "--rank-weight",
        help="Weight for pairwise rank loss.",
    ),
    loss_mode: str = typer.Option(
        "bce_rank",
        "--loss-mode",
        help="Loss mode: bce, bce_rank, focal, or focal_rank.",
    ),
    focal_gamma: float = typer.Option(
        2.0,
        "--focal-gamma",
        help="Focal loss gamma for focal loss modes.",
    ),
) -> None:
    """Train ranking-aware site-attention repair model."""
    train_neural(
        dataset_dir=dataset_dir,
        outdir=outdir,
        device=device,
        methods=methods,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        saturation_embedding_dim=8,
        architecture="site_attention",
        positive_class_weight="auto",
        group_weighting="none",
        sampler="none",
        training_preset="site_attention_ranked",
        loss_mode=loss_mode,
        rank_weight=rank_weight,
        focal_gamma=focal_gamma,
        min_delta=0.0,
        threshold=threshold,
        early_stopping_patience=early_stopping_patience,
        monitor_metric=monitor_metric,
        max_train_items=max_train_items,
        max_val_items=max_val_items,
        max_calib_items=max_calib_items,
        max_test_items=max_test_items,
        save_every_epoch=False,
    )


@app.command("validate-neural")
def validate_neural(
    model_dir: Path = typer.Option(
        ...,
        "--model-dir",
        help="Scale-ready neural model artifact directory to validate.",
    )
) -> None:
    """Validate scale-ready neural training artifacts."""
    summary = validate_neural_model_dir(model_dir)
    table = Table(title="BABAPPA Neural Validation Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Predictions", str(summary["n_predictions"]))
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("calibrate-neural")
def calibrate_neural(
    model_dir: Path = typer.Option(
        ...,
        "--model-dir",
        help="Scale-ready neural model artifact directory to calibrate.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for calibrated neural artifacts.",
    ),
    target_fdr: float = typer.Option(
        0.10,
        "--target-fdr",
        help="Target empirical FDR for calibration-split threshold selection.",
    ),
    calibration_method: str = typer.Option(
        "temperature",
        "--calibration-method",
        help="Calibration method: none or temperature.",
    ),
    threshold_grid_size: int = typer.Option(
        181,
        "--threshold-grid-size",
        help="Number of empirical threshold candidates to evaluate.",
    ),
    min_threshold: float = typer.Option(
        0.05,
        "--min-threshold",
        help="Minimum threshold candidate.",
    ),
    max_threshold: float = typer.Option(
        0.95,
        "--max-threshold",
        help="Maximum threshold candidate.",
    ),
) -> None:
    """Calibrate neural probabilities and select an empirical threshold."""
    try:
        config = NeuralCalibrationConfig(
            model_dir=str(model_dir),
            outdir=str(outdir),
            target_fdr=target_fdr,
            threshold_grid_size=threshold_grid_size,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            calibration_method=calibration_method,
        )
        summary = calibrate_neural_model(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not calibrate neural model: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Neural Calibration Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Source model directory", str(model_dir))
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Method", calibration_method)
    table.add_row("Temperature", f"{float(summary['temperature']):.6f}")
    table.add_row("Selected threshold", f"{float(summary['selected_threshold']):.6f}")
    table.add_row("Target FDR", f"{target_fdr:.4f}")
    table.add_row("Calibration JSON", summary["calibration"])
    table.add_row("Calibrated predictions", summary["predictions"])
    table.add_row("Calibrated metrics", summary["metrics"])
    console.print(table)

    metrics = _read_json(Path(summary["metrics"]))
    _print_metric_summary(metrics.get("metrics_by_split_calibrated", {}))

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")


@app.command("validate-neural-calibration")
def validate_neural_calibration(
    calibration_dir: Path = typer.Option(
        ...,
        "--calibration-dir",
        help="Neural calibration artifact directory to validate.",
    )
) -> None:
    """Validate a BABAPPA neural calibration artifact directory."""
    summary = validate_neural_calibration_dir(calibration_dir)
    table = Table(title="BABAPPA Neural Calibration Validation Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Predictions", str(summary["n_predictions"]))
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("calibrate-stratified")
def calibrate_stratified(
    predictions: Path = typer.Option(
        ...,
        "--predictions",
        help="Prediction TSV to calibrate by group.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for stratified calibration artifacts.",
    ),
    group_column: str = typer.Option(
        "saturation_tier",
        "--group-column",
        help="Column defining calibration groups.",
    ),
    probability_column: str = typer.Option(
        "prob_positive",
        "--probability-column",
        help="Raw probability column.",
    ),
    label_column: str = typer.Option(
        "gene_label",
        "--label-column",
        help="Ground-truth label column.",
    ),
    split_column: str = typer.Option(
        "split",
        "--split-column",
        help="Split column.",
    ),
    target_fdr: float = typer.Option(
        0.10,
        "--target-fdr",
        help="Target empirical FDR for per-group threshold selection.",
    ),
    calibration_method: str = typer.Option(
        "temperature",
        "--calibration-method",
        help="Calibration method: none or temperature.",
    ),
    min_group_calib_n: int = typer.Option(
        20,
        "--min-group-calib-n",
        help="Minimum calibration rows needed before fitting a group-specific calibration.",
    ),
    threshold_grid_size: int = typer.Option(
        181,
        "--threshold-grid-size",
        help="Number of empirical threshold candidates to evaluate.",
    ),
    min_threshold: float = typer.Option(
        0.05,
        "--min-threshold",
        help="Minimum threshold candidate.",
    ),
    max_threshold: float = typer.Option(
        0.95,
        "--max-threshold",
        help="Maximum threshold candidate.",
    ),
) -> None:
    """Calibrate probabilities and thresholds by a grouping column."""
    try:
        config = StratifiedCalibrationConfig(
            predictions_tsv=str(predictions),
            outdir=str(outdir),
            group_column=group_column,
            probability_column=probability_column,
            label_column=label_column,
            split_column=split_column,
            target_fdr=target_fdr,
            calibration_method=calibration_method,
            min_group_calib_n=min_group_calib_n,
            threshold_grid_size=threshold_grid_size,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
        )
        summary = calibrate_by_group(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not run stratified calibration: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Stratified Calibration")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Predictions", str(predictions))
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Calibration JSON", summary["calibration"])
    table.add_row("Calibrated predictions", summary["predictions"])
    table.add_row("Calibrated metrics", summary["metrics"])
    table.add_row("Markdown", summary["markdown"])
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")


@app.command("validate-stratified-calibration")
def validate_stratified_calibration(
    calibration_dir: Path = typer.Option(
        ...,
        "--calibration-dir",
        help="Stratified calibration artifact directory to validate.",
    )
) -> None:
    """Validate a BABAPPA stratified calibration artifact directory."""
    summary = validate_stratified_calibration_dir(calibration_dir)
    table = Table(title="BABAPPA Stratified Calibration Validation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Predictions", str(summary["n_predictions"]))
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("threshold-policy")
def threshold_policy_command(
    predictions: Path = typer.Option(
        ...,
        "--predictions",
        help="Prediction TSV to profile.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for threshold-policy artifacts.",
    ),
    probability_column: str = typer.Option(
        "prob_positive",
        "--probability-column",
        help="Raw probability column.",
    ),
    calibrated_probability_column: Optional[str] = typer.Option(
        None,
        "--calibrated-probability-column",
        help="Optional calibrated probability column; used if present.",
    ),
    label_column: str = typer.Option(
        "gene_label",
        "--label-column",
        help="Ground-truth label column.",
    ),
    split_column: str = typer.Option(
        "split",
        "--split-column",
        help="Split column.",
    ),
    selection_split: str = typer.Option(
        "calib",
        "--selection-split",
        help="Split used to select operating-point thresholds.",
    ),
    target_fdr: float = typer.Option(
        0.10,
        "--target-fdr",
        help="Target empirical FDR for the strict_fdr profile.",
    ),
    precision_floor: float = typer.Option(
        0.80,
        "--precision-floor",
        help="Minimum precision for high_precision profile.",
    ),
    recall_floor: float = typer.Option(
        0.80,
        "--recall-floor",
        help="Minimum recall for high_recall profile.",
    ),
    threshold_grid_size: int = typer.Option(
        501,
        "--threshold-grid-size",
        help="Number of thresholds in the policy grid.",
    ),
    min_threshold: float = typer.Option(
        0.0,
        "--min-threshold",
        help="Minimum threshold in the policy grid.",
    ),
    max_threshold: float = typer.Option(
        1.0,
        "--max-threshold",
        help="Maximum threshold in the policy grid.",
    ),
    degenerate_call_fraction: float = typer.Option(
        0.98,
        "--degenerate-call-fraction",
        help="Called-positive/negative fraction used to warn about degenerate profiles.",
    ),
    min_non_degenerate_threshold: Optional[float] = typer.Option(
        None,
        "--min-non-degenerate-threshold",
        help="Optional minimum threshold when preferring non-degenerate profiles.",
    ),
    model_name: str = typer.Option(
        "model",
        "--model-name",
        help="Model name for threshold-policy outputs.",
    ),
) -> None:
    """Profile thresholds and select operating-point policies."""
    try:
        config = ThresholdPolicyConfig(
            predictions_tsv=str(predictions),
            outdir=str(outdir),
            probability_column=probability_column,
            calibrated_probability_column=calibrated_probability_column,
            label_column=label_column,
            split_column=split_column,
            selection_split=selection_split,
            target_fdr=target_fdr,
            precision_floor=precision_floor,
            recall_floor=recall_floor,
            threshold_grid_size=threshold_grid_size,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            degenerate_call_fraction=degenerate_call_fraction,
            min_non_degenerate_threshold=min_non_degenerate_threshold,
            model_name=model_name,
        )
        summary = build_threshold_policy(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not build threshold policy: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Threshold Policy")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Profiles JSON", summary["profiles_json"])
    table.add_row("Profiles TSV", summary["profiles_tsv"])
    table.add_row("Profile metrics TSV", summary["profile_metrics_tsv"])
    table.add_row("Threshold curve TSV", summary["curve_tsv"])
    table.add_row("Markdown", summary["markdown"])
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")


@app.command("validate-threshold-policy")
def validate_threshold_policy(
    policy_dir: Path = typer.Option(
        ...,
        "--policy-dir",
        help="Threshold-policy directory to validate.",
    )
) -> None:
    """Validate a BABAPPA threshold-policy directory."""
    summary = validate_threshold_policy_dir(policy_dir)
    table = Table(title="BABAPPA Threshold Policy Validation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("stratified-eval")
def stratified_eval_command(
    predictions: Path = typer.Option(
        ...,
        "--predictions",
        help="Prediction TSV to evaluate.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for stratified evaluation artifacts.",
    ),
    model_name: str = typer.Option(
        "model",
        "--model-name",
        help="Model name for output tables.",
    ),
    probability_column: str = typer.Option(
        "prob_positive",
        "--probability-column",
        help="Probability column to evaluate.",
    ),
    label_column: str = typer.Option(
        "gene_label",
        "--label-column",
        help="Ground-truth label column.",
    ),
    split_column: str = typer.Option(
        "split",
        "--split-column",
        help="Split column.",
    ),
    method_column: str = typer.Option(
        "method",
        "--method-column",
        help="Alignment method column.",
    ),
    saturation_column: str = typer.Option(
        "saturation_tier",
        "--saturation-column",
        help="Saturation tier column.",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        help="Fixed threshold used when no threshold-policy directory is supplied.",
    ),
    threshold_policy_dir: Optional[Path] = typer.Option(
        None,
        "--threshold-policy-dir",
        help="Optional threshold-policy directory containing threshold_profiles.json.",
    ),
) -> None:
    """Evaluate predictions by split, saturation tier, and alignment method."""
    try:
        config = StratifiedEvalConfig(
            predictions_tsv=str(predictions),
            outdir=str(outdir),
            model_name=model_name,
            probability_column=probability_column,
            label_column=label_column,
            split_column=split_column,
            method_column=method_column,
            saturation_column=saturation_column,
            threshold=threshold,
            threshold_policy_dir=_optional_path_to_str(threshold_policy_dir),
        )
        summary = stratified_evaluate_predictions(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not run stratified evaluation: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Stratified Evaluation")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("JSON", summary["json"])
    table.add_row("TSV", summary["tsv"])
    table.add_row("Markdown", summary["markdown"])
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")


@app.command("validate-stratified-eval")
def validate_stratified_eval(
    eval_dir: Path = typer.Option(
        ...,
        "--eval-dir",
        help="Stratified evaluation directory to validate.",
    )
) -> None:
    """Validate a BABAPPA stratified evaluation directory."""
    summary = validate_stratified_eval_dir(eval_dir)
    table = Table(title="BABAPPA Stratified Evaluation Validation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("make-saturation-panel")
def make_saturation_panel(
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for the multi-saturation panel.",
    ),
    n_families_per_tier: int = typer.Option(
        10,
        "--n-families-per-tier",
        help="Number of simulated families per saturation tier.",
    ),
    tiers: str = typer.Option(
        "low,moderate,high,extreme",
        "--tiers",
        help="Comma-separated saturation tiers to build.",
    ),
    n_taxa: int = typer.Option(
        8,
        "--n-taxa",
        help="Number of taxa per family.",
    ),
    n_codons: int = typer.Option(
        120,
        "--n-codons",
        help="Number of codons per family.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Base random seed; tier-specific seeds are derived deterministically.",
    ),
    positive_rate: float = typer.Option(
        0.5,
        "--positive-rate",
        help="Probability that a family contains positive selection.",
    ),
    selected_site_fraction: float = typer.Option(
        0.05,
        "--selected-site-fraction",
        help="Fraction of selected sites in positive families.",
    ),
    mutation_rate: float = typer.Option(
        0.03,
        "--mutation-rate",
        help="Base per-codon mutation rate before saturation scaling.",
    ),
    indel_rate: float = typer.Option(
        0.0,
        "--indel-rate",
        help="Reserved for future indel simulation; must be non-negative.",
    ),
    methods: str = typer.Option(
        "identity,codon_dropout",
        "--methods",
        help="Comma-separated alignment scaffold methods.",
    ),
    dropout_rate: float = typer.Option(
        0.02,
        "--dropout-rate",
        help="Codon dropout probability for the codon_dropout method.",
    ),
    build_tensors: bool = typer.Option(
        True,
        "--build-tensors/--no-build-tensors",
        help="Build tensor shards for each tier.",
    ),
    index_datasets: bool = typer.Option(
        True,
        "--index-datasets/--no-index-datasets",
        help="Build dataset indexes for each tier.",
    ),
) -> None:
    """Build a low/moderate/high/extreme saturation benchmark panel."""
    try:
        config = SaturationPanelConfig(
            outdir=str(outdir),
            n_families_per_tier=n_families_per_tier,
            tiers=_parse_methods(tiers),
            n_taxa=n_taxa,
            n_codons=n_codons,
            seed=seed,
            positive_rate=positive_rate,
            selected_site_fraction=selected_site_fraction,
            mutation_rate=mutation_rate,
            indel_rate=indel_rate,
            methods=_parse_methods(methods),
            dropout_rate=dropout_rate,
            build_tensors=build_tensors,
            index_datasets=index_datasets,
        )
        summary = build_saturation_panel(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not build saturation panel: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Saturation Panel")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Panel JSON", summary["panel_json"])
    table.add_row("Panel Markdown", summary["panel_markdown"])
    table.add_row("Tiers", _format_methods(list(summary["tier_outputs"].keys())))
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")


@app.command("validate-saturation-panel")
def validate_saturation_panel(
    panel_dir: Path = typer.Option(
        ...,
        "--panel-dir",
        help="Saturation panel directory to validate.",
    )
) -> None:
    """Validate a BABAPPA saturation panel directory."""
    summary = validate_saturation_panel_dir(panel_dir)
    table = Table(title="BABAPPA Saturation Panel Validation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("merge-datasets")
def merge_datasets(
    dataset_dirs: str = typer.Option(
        ...,
        "--dataset-dirs",
        help="Comma-separated dataset index directories to merge.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for the merged dataset.",
    ),
    names: Optional[str] = typer.Option(
        None,
        "--names",
        help="Optional comma-separated source names matching dataset-dirs.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for deterministic merged split assignment.",
    ),
    resplit: bool = typer.Option(
        True,
        "--resplit/--no-resplit",
        help="Create new deterministic merged splits instead of preserving sources.",
    ),
    train_fraction: float = typer.Option(
        0.8,
        "--train-fraction",
        help="Fraction of merged families assigned to train.",
    ),
    val_fraction: float = typer.Option(
        0.1,
        "--val-fraction",
        help="Fraction of merged families assigned to validation.",
    ),
    calib_fraction: float = typer.Option(
        0.05,
        "--calib-fraction",
        help="Fraction of merged families assigned to calibration.",
    ),
    test_fraction: float = typer.Option(
        0.05,
        "--test-fraction",
        help="Fraction of merged families assigned to test.",
    ),
) -> None:
    """Merge per-tier BABAPPA dataset indexes into one trainable dataset."""
    try:
        config = DatasetMergeConfig(
            dataset_dirs=_parse_methods(dataset_dirs),
            outdir=str(outdir),
            names=_parse_optional_methods(names),
            seed=seed,
            resplit=resplit,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            calib_fraction=calib_fraction,
            test_fraction=test_fraction,
        )
        summary = merge_dataset_indexes(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not merge datasets: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Dataset Merge")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Rows", str(summary["n_rows"]))
    table.add_row("Families", str(summary["n_families"]))
    table.add_row("Saturation tiers", _format_mapping(summary["saturation_tier_counts"]))
    table.add_row("Features", summary["features"])
    table.add_row("Splits", summary["splits"])
    table.add_row("Index", summary["index"])
    console.print(table)


@app.command("validate-merged-dataset")
def validate_merged_dataset(
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        help="Merged dataset directory to validate.",
    )
) -> None:
    """Validate a BABAPPA merged dataset directory."""
    summary = validate_merged_dataset_dir(dataset_dir)
    table = Table(title="BABAPPA Merged Dataset Validation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Rows", str(summary["n_rows"]))
    table.add_row("Families", str(summary["n_families"]))
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("resplit-dataset")
def resplit_dataset_command(
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        help="Existing dataset directory to resplit.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for the resplit dataset.",
    ),
    seed: int = typer.Option(..., "--seed", help="Random seed for split assignment."),
    train_fraction: float = typer.Option(0.8, "--train-fraction"),
    val_fraction: float = typer.Option(0.1, "--val-fraction"),
    calib_fraction: float = typer.Option(0.05, "--calib-fraction"),
    test_fraction: float = typer.Option(0.05, "--test-fraction"),
) -> None:
    """Create a deterministic family-disjoint resplit of an existing dataset."""
    try:
        config = ResplitDatasetConfig(
            dataset_dir=str(dataset_dir),
            outdir=str(outdir),
            seed=seed,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            calib_fraction=calib_fraction,
            test_fraction=test_fraction,
        )
        summary = resplit_dataset(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not resplit dataset: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Dataset Resplit")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Rows", str(summary["n_rows"]))
    table.add_row("Families", str(summary["n_families"]))
    table.add_row("Split counts", _format_mapping(summary["split_counts_rows"]))
    table.add_row("Features", summary["features"])
    table.add_row("Splits", summary["splits"])
    table.add_row("Index", summary["index"])
    console.print(table)


@app.command("validate-resplit-dataset")
def validate_resplit_dataset(
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        help="Resplit dataset directory to validate.",
    )
) -> None:
    """Validate a BABAPPA resplit dataset directory."""
    summary = validate_resplit_dataset_dir(dataset_dir)
    table = Table(title="BABAPPA Resplit Dataset Validation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Rows", str(summary["n_rows"]))
    table.add_row("Families", str(summary["n_families"]))
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("stability-benchmark")
def stability_benchmark_command(
    dataset_dir: Path = typer.Option(..., "--dataset-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    seeds: str = typer.Option("42,43,44", "--seeds"),
    presets: str = typer.Option(
        "contrastive_v2,saturation_embed_only,site_attention_ranked",
        "--presets",
    ),
    methods: str = typer.Option("identity,codon_dropout", "--methods"),
    device: str = typer.Option("cpu", "--device"),
    epochs: int = typer.Option(2, "--epochs"),
    batch_size: int = typer.Option(8, "--batch-size"),
    learning_rate: float = typer.Option(0.001, "--learning-rate"),
    max_train_items: Optional[int] = typer.Option(64, "--max-train-items"),
    max_val_items: Optional[int] = typer.Option(32, "--max-val-items"),
    max_calib_items: Optional[int] = typer.Option(16, "--max-calib-items"),
    max_test_items: Optional[int] = typer.Option(16, "--max-test-items"),
    run_training: bool = typer.Option(
        True,
        "--run-training/--no-run-training",
        help="Train small benchmark models, or only create resplit structure.",
    ),
) -> None:
    """Run repeated-split/repeated-seed stability benchmark."""
    try:
        config = StabilityBenchmarkConfig(
            dataset_dir=str(dataset_dir),
            outdir=str(outdir),
            seeds=[int(seed) for seed in _parse_methods(seeds)],
            presets=_parse_methods(presets),
            methods=_parse_methods(methods),
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_train_items=max_train_items,
            max_val_items=max_val_items,
            max_calib_items=max_calib_items,
            max_test_items=max_test_items,
            run_training=run_training,
        )
        summary = run_stability_benchmark(config)
    except (OSError, ValueError, RuntimeError) as exc:
        console.print(f"Error: could not run stability benchmark: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Stability Benchmark")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("JSON", summary["json"])
    table.add_row("TSV", summary["tsv"])
    table.add_row("Markdown", summary["markdown"])
    table.add_row(
        "Best val preset",
        str(summary["aggregate_summary"].get("best_preset_by_mean_val_auroc")),
    )
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)


@app.command("validate-stability-benchmark")
def validate_stability_benchmark(
    benchmark_dir: Path = typer.Option(
        ...,
        "--benchmark-dir",
        help="Stability benchmark directory to validate.",
    )
) -> None:
    """Validate stability benchmark artifacts."""
    summary = validate_stability_benchmark_dir(benchmark_dir)
    table = Table(title="BABAPPA Stability Benchmark Validation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("extract-site-labels")
def extract_site_labels_command(
    dataset_dir: Path = typer.Option(..., "--dataset-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    site_index_base: str = typer.Option(
        "auto",
        "--site-index-base",
        help="Oracle site index base: auto, zero, or one.",
    ),
    site_map_dir: Optional[Path] = typer.Option(
        None,
        "--site-map-dir",
        help="Optional alignment site-map directory for mapped aligned-site labels.",
    ),
    aligned_site_mode: str = typer.Option(
        "mapped",
        "--aligned-site-mode",
        help="Site-label mode: original or mapped.",
    ),
) -> None:
    """Extract oracle site-level labels from a BABAPPA tensor dataset."""
    try:
        config = OracleSiteLabelConfig(
            dataset_dir=str(dataset_dir),
            outdir=str(outdir),
            site_index_base=site_index_base,
            site_map_dir=_optional_path_to_str(site_map_dir),
            aligned_site_mode=aligned_site_mode,
        )
        summary = extract_oracle_site_labels(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not extract site labels: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Oracle Site Labels")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Labels TSV", summary["site_labels_tsv"])
    table.add_row("Summary JSON", summary["summary_json"])
    table.add_row("Markdown", summary["markdown"])
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)
    _print_warnings(summary["warnings"])


@app.command("validate-site-labels")
def validate_site_labels(
    site_label_dir: Path = typer.Option(
        ...,
        "--site-label-dir",
        help="Oracle site-label output directory.",
    )
) -> None:
    """Validate oracle site-label extraction artifacts."""
    summary = validate_site_label_dir(site_label_dir)
    table = Table(title="BABAPPA Site Label Validation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Rows", str(summary["n_rows"]))
    table.add_row("Positive sites", str(summary["n_positive_sites"]))
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)
    _print_warnings(summary["warnings"])
    _print_failures(summary["failures"])
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("build-site-dataset")
def build_site_dataset_command(
    dataset_dir: Path = typer.Option(..., "--dataset-dir"),
    oracle_labels: Path = typer.Option(
        ...,
        "--oracle-labels",
        help="site_oracle_labels.tsv produced by extract-site-labels.",
    ),
    outdir: Path = typer.Option(..., "--outdir"),
    include_foreground_context: bool = typer.Option(
        True,
        "--foreground-context/--no-foreground-context",
        help="Include foreground-derived numeric context features.",
    ),
    max_sites_per_family_method: Optional[int] = typer.Option(
        None,
        "--max-sites-per-family-method",
        help="Optional cap on site rows per family/method.",
    ),
    negative_downsample_ratio: Optional[float] = typer.Option(
        None,
        "--negative-downsample-ratio",
        help="Keep at most ratio * positives negative sites per split/tier/method group.",
    ),
    seed: int = typer.Option(42, "--seed"),
    require_mappable_sites: bool = typer.Option(
        True,
        "--require-mappable-sites/--allow-unmappable-sites",
        help="Drop site-map rows that do not map uniquely to an original simulated site.",
    ),
) -> None:
    """Build site-level features and splits from oracle site labels."""
    try:
        config = SiteDatasetConfig(
            dataset_dir=str(dataset_dir),
            oracle_labels_tsv=str(oracle_labels),
            outdir=str(outdir),
            include_foreground_context=include_foreground_context,
            max_sites_per_family_method=max_sites_per_family_method,
            negative_downsample_ratio=negative_downsample_ratio,
            seed=seed,
            require_mappable_sites=require_mappable_sites,
        )
        summary = build_site_dataset(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not build site dataset: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Site Dataset")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Rows", str(summary["n_site_rows"]))
    table.add_row("Positive sites", str(summary["n_positive_sites"]))
    table.add_row("Negative sites", str(summary["n_negative_sites"]))
    table.add_row("Features", summary["features"])
    table.add_row("Splits", summary["splits"])
    table.add_row("Index", summary["index"])
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)
    _print_warnings(summary["warnings"])


@app.command("validate-site-dataset")
def validate_site_dataset(
    site_dataset_dir: Path = typer.Option(
        ...,
        "--site-dataset-dir",
        help="Site dataset directory to validate.",
    )
) -> None:
    """Validate a BABAPPA site-level dataset."""
    summary = validate_site_dataset_dir(site_dataset_dir)
    table = Table(title="BABAPPA Site Dataset Validation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Rows", str(summary["n_rows"]))
    table.add_row("Positive sites", str(summary["n_positive_sites"]))
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)
    _print_warnings(summary["warnings"])
    _print_failures(summary["failures"])
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("audit-site-leakage")
def audit_site_leakage_command(
    site_dataset_dir: Path = typer.Option(..., "--site-dataset-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Audit a site-level dataset for oracle leakage risks."""
    try:
        summary = audit_site_dataset_leakage(site_dataset_dir, outdir)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not audit site leakage: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Site Leakage Audit")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Leakage status", summary["leakage_status"])
    table.add_row("JSON", summary["json"])
    table.add_row("Columns TSV", summary["columns_tsv"])
    table.add_row("Markdown", summary["markdown"])
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)
    _print_warnings(summary["warnings"])


@app.command("train-site-baseline")
def train_site_baseline_command(
    site_dataset_dir: Path = typer.Option(..., "--site-dataset-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    seed: int = typer.Option(42, "--seed"),
    epochs: int = typer.Option(300, "--epochs"),
    learning_rate: float = typer.Option(0.05, "--learning-rate"),
    l2: float = typer.Option(0.001, "--l2"),
    positive_class_weight: str = typer.Option("auto", "--positive-class-weight"),
    threshold: float = typer.Option(0.5, "--threshold"),
) -> None:
    """Train a minimal NumPy site-level logistic baseline."""
    try:
        config = SiteBaselineConfig(
            site_dataset_dir=str(site_dataset_dir),
            outdir=str(outdir),
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            positive_class_weight=positive_class_weight,
            threshold=threshold,
        )
        summary = train_site_baseline(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not train site baseline: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Site Baseline")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Model", summary["model"])
    table.add_row("Metadata", summary["meta"])
    table.add_row("Predictions", summary["predictions"])
    table.add_row("Metrics", summary["metrics"])
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)
    _print_warnings(summary["warnings"])


@app.command("validate-site-baseline")
def validate_site_baseline(
    model_dir: Path = typer.Option(
        ...,
        "--model-dir",
        help="Site baseline output directory to validate.",
    )
) -> None:
    """Validate a site-level baseline output directory."""
    summary = validate_site_baseline_dir(model_dir)
    table = Table(title="BABAPPA Site Baseline Validation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Predictions", str(summary["n_predictions"]))
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)
    _print_warnings(summary["warnings"])
    _print_failures(summary["failures"])
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("train-site-neural")
def train_site_neural_command(
    site_dataset_dir: Path = typer.Option(..., "--site-dataset-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    seed: int = typer.Option(42, "--seed"),
    device: str = typer.Option("auto", "--device"),
    epochs: int = typer.Option(30, "--epochs"),
    batch_size: int = typer.Option(256, "--batch-size"),
    learning_rate: float = typer.Option(0.001, "--learning-rate"),
    weight_decay: float = typer.Option(0.0001, "--weight-decay"),
    hidden_dim: int = typer.Option(64, "--hidden-dim"),
    dropout: float = typer.Option(0.1, "--dropout"),
    positive_class_weight: str = typer.Option("auto", "--positive-class-weight"),
    threshold: float = typer.Option(0.5, "--threshold"),
    early_stopping_patience: int = typer.Option(8, "--early-stopping-patience"),
    monitor_metric: str = typer.Option("val_auroc", "--monitor-metric"),
    max_train_items: Optional[int] = typer.Option(None, "--max-train-items"),
    max_val_items: Optional[int] = typer.Option(None, "--max-val-items"),
    max_calib_items: Optional[int] = typer.Option(None, "--max-calib-items"),
    max_test_items: Optional[int] = typer.Option(None, "--max-test-items"),
) -> None:
    """Train a site-level neural MLP classifier."""
    try:
        summary = train_site_neural_model(
            SiteNeuralTrainConfig(
                site_dataset_dir=str(site_dataset_dir),
                outdir=str(outdir),
                seed=seed,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                hidden_dim=hidden_dim,
                dropout=dropout,
                positive_class_weight=positive_class_weight,
                threshold=threshold,
                early_stopping_patience=early_stopping_patience,
                monitor_metric=monitor_metric,
                max_train_items=max_train_items,
                max_val_items=max_val_items,
                max_calib_items=max_calib_items,
                max_test_items=max_test_items,
            )
        )
    except (OSError, ValueError, RuntimeError) as exc:
        console.print(f"Error: could not train site neural model: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    table = Table(title="BABAPPA Site Neural")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Checkpoint", summary["checkpoint"])
    table.add_row("Predictions", summary["predictions"])
    table.add_row("Metrics", summary["metrics"])
    table.add_row("Best epoch", str(summary["best_epoch"]))
    console.print(table)


@app.command("validate-site-neural")
def validate_site_neural(
    model_dir: Path = typer.Option(..., "--model-dir")
) -> None:
    """Validate a site-level neural model directory."""
    summary = validate_site_neural_dir(model_dir)
    _print_validation_table("BABAPPA Site Neural Validation", summary, "n_predictions")
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("calibrate-site-neural")
def calibrate_site_neural_command(
    model_dir: Path = typer.Option(..., "--model-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    target_fdr: float = typer.Option(0.10, "--target-fdr"),
    calibration_method: str = typer.Option("temperature", "--calibration-method"),
    threshold_grid_size: int = typer.Option(501, "--threshold-grid-size"),
    min_threshold: float = typer.Option(0.0, "--min-threshold"),
    max_threshold: float = typer.Option(1.0, "--max-threshold"),
    n_bins: int = typer.Option(20, "--n-bins"),
) -> None:
    """Calibrate site-level neural probabilities."""
    try:
        summary = calibrate_site_model(
            SiteCalibrationConfig(
                model_dir=str(model_dir),
                outdir=str(outdir),
                target_fdr=target_fdr,
                calibration_method=calibration_method,
                threshold_grid_size=threshold_grid_size,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
                n_bins=n_bins,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not calibrate site neural model: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    table = Table(title="BABAPPA Site Calibration")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Temperature", str(summary["temperature"]))
    table.add_row("Selected threshold", str(summary["selected_threshold"]))
    table.add_row("Calibration", summary["calibration"])
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)
    _print_warnings(summary["warnings"])


@app.command("validate-site-calibration")
def validate_site_calibration(
    calibration_dir: Path = typer.Option(..., "--calibration-dir")
) -> None:
    """Validate site calibration artifacts."""
    summary = validate_site_calibration_dir(calibration_dir)
    _print_validation_table("BABAPPA Site Calibration Validation", summary, "n_predictions")
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("site-threshold-policy")
def site_threshold_policy_command(
    predictions: Path = typer.Option(..., "--predictions"),
    outdir: Path = typer.Option(..., "--outdir"),
    probability_column: str = typer.Option("prob_positive", "--probability-column"),
    calibrated_probability_column: Optional[str] = typer.Option(
        None, "--calibrated-probability-column"
    ),
    label_column: str = typer.Option("y_site", "--label-column"),
    split_column: str = typer.Option("split", "--split-column"),
    selection_split: str = typer.Option("calib", "--selection-split"),
    target_fdr: float = typer.Option(0.10, "--target-fdr"),
    precision_floor: float = typer.Option(0.80, "--precision-floor"),
    recall_floor: float = typer.Option(0.80, "--recall-floor"),
    threshold_grid_size: int = typer.Option(501, "--threshold-grid-size"),
    min_threshold: float = typer.Option(0.0, "--min-threshold"),
    max_threshold: float = typer.Option(1.0, "--max-threshold"),
    model_name: str = typer.Option("site_model", "--model-name"),
) -> None:
    """Build site-level operating-point threshold profiles."""
    try:
        summary = build_site_threshold_policy(
            SiteThresholdPolicyConfig(
                predictions_tsv=str(predictions),
                outdir=str(outdir),
                probability_column=probability_column,
                calibrated_probability_column=calibrated_probability_column,
                label_column=label_column,
                split_column=split_column,
                selection_split=selection_split,
                target_fdr=target_fdr,
                precision_floor=precision_floor,
                recall_floor=recall_floor,
                threshold_grid_size=threshold_grid_size,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
                model_name=model_name,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not build site threshold policy: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    table = Table(title="BABAPPA Site Threshold Policy")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Profiles JSON", summary["profiles_json"])
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)
    _print_warnings(summary["warnings"])


@app.command("validate-site-threshold-policy")
def validate_site_threshold_policy(
    policy_dir: Path = typer.Option(..., "--policy-dir")
) -> None:
    """Validate site threshold-policy artifacts."""
    summary = validate_site_threshold_policy_dir(policy_dir)
    _print_validation_table("BABAPPA Site Threshold Policy Validation", summary)
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("site-stratified-eval")
def site_stratified_eval_command(
    predictions: Path = typer.Option(..., "--predictions"),
    outdir: Path = typer.Option(..., "--outdir"),
    probability_column: str = typer.Option("prob_positive", "--probability-column"),
    label_column: str = typer.Option("y_site", "--label-column"),
    threshold: float = typer.Option(0.5, "--threshold"),
    threshold_policy_dir: Optional[Path] = typer.Option(None, "--threshold-policy-dir"),
) -> None:
    """Run site-level stratified evaluation."""
    try:
        summary = site_stratified_evaluate(
            SiteStratifiedEvalConfig(
                predictions_tsv=str(predictions),
                outdir=str(outdir),
                probability_column=probability_column,
                label_column=label_column,
                threshold=threshold,
                threshold_policy_dir=_optional_path_to_str(threshold_policy_dir),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not run site stratified evaluation: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    table = Table(title="BABAPPA Site Stratified Evaluation")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("JSON", summary["json"])
    table.add_row("TSV", summary["tsv"])
    console.print(table)


@app.command("validate-site-stratified-eval")
def validate_site_stratified_eval(
    eval_dir: Path = typer.Option(..., "--eval-dir")
) -> None:
    """Validate site stratified evaluation artifacts."""
    summary = validate_site_stratified_eval_dir(eval_dir)
    _print_validation_table("BABAPPA Site Stratified Evaluation Validation", summary)
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("aggregate-sites")
def aggregate_sites_command(
    predictions: Path = typer.Option(..., "--predictions"),
    gene_dataset_dir: Path = typer.Option(..., "--gene-dataset-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    probability_column: str = typer.Option("prob_positive", "--probability-column"),
) -> None:
    """Aggregate site-level probabilities to gene/family-level support."""
    try:
        summary = aggregate_site_predictions(
            SiteAggregationConfig(
                predictions_tsv=str(predictions),
                gene_dataset_dir=str(gene_dataset_dir),
                outdir=str(outdir),
                probability_column=probability_column,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not aggregate site predictions: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    table = Table(title="BABAPPA Site-to-Gene Aggregation")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Rows", str(summary["n_family_method_rows"]))
    table.add_row("Metrics", summary["metrics"])
    console.print(table)


@app.command("validate-site-aggregation")
def validate_site_aggregation(
    aggregation_dir: Path = typer.Option(..., "--aggregation-dir")
) -> None:
    """Validate site-to-gene aggregation artifacts."""
    summary = validate_site_aggregation_dir(aggregation_dir)
    _print_validation_table("BABAPPA Site Aggregation Validation", summary, "n_rows")
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("site-stability-benchmark")
def site_stability_benchmark_command(
    site_dataset_dir: Path = typer.Option(..., "--site-dataset-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    seeds: str = typer.Option("42,43,44", "--seeds"),
    device: str = typer.Option("cpu", "--device"),
    epochs: int = typer.Option(5, "--epochs"),
    batch_size: int = typer.Option(256, "--batch-size"),
    learning_rate: float = typer.Option(0.001, "--learning-rate"),
    weight_decay: float = typer.Option(0.0001, "--weight-decay"),
    hidden_dim: int = typer.Option(64, "--hidden-dim"),
    dropout: float = typer.Option(0.1, "--dropout"),
    max_train_items: Optional[int] = typer.Option(4096, "--max-train-items"),
    max_val_items: Optional[int] = typer.Option(1024, "--max-val-items"),
    max_calib_items: Optional[int] = typer.Option(1024, "--max-calib-items"),
    max_test_items: Optional[int] = typer.Option(1024, "--max-test-items"),
) -> None:
    """Run a repeated-seed site neural stability benchmark."""
    try:
        summary = run_site_stability_benchmark(
            SiteStabilityConfig(
                site_dataset_dir=str(site_dataset_dir),
                outdir=str(outdir),
                seeds=_parse_int_list(seeds),
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                hidden_dim=hidden_dim,
                dropout=dropout,
                max_train_items=max_train_items,
                max_val_items=max_val_items,
                max_calib_items=max_calib_items,
                max_test_items=max_test_items,
            )
        )
    except (OSError, ValueError, RuntimeError) as exc:
        console.print(f"Error: could not run site stability benchmark: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table("BABAPPA Site Stability Benchmark", summary, ["outdir", "json", "warnings"])


@app.command("validate-site-stability")
def validate_site_stability(
    benchmark_dir: Path = typer.Option(..., "--benchmark-dir")
) -> None:
    """Validate site stability benchmark artifacts."""
    summary = validate_site_stability_dir(benchmark_dir)
    _print_validation_table("BABAPPA Site Stability Validation", summary)
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("compare-site-models")
def compare_site_models_command(
    site_baseline_dir: Path = typer.Option(..., "--site-baseline-dir"),
    site_neural_dir: Path = typer.Option(..., "--site-neural-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    site_stratified_eval_dir: Optional[Path] = typer.Option(None, "--site-stratified-eval-dir"),
    site_aggregation_dir: Optional[Path] = typer.Option(None, "--site-aggregation-dir"),
) -> None:
    """Compare site baseline and site neural model metrics."""
    try:
        summary = compare_site_models(
            SiteModelCompareConfig(
                outdir=str(outdir),
                site_baseline_dir=str(site_baseline_dir),
                site_neural_dir=str(site_neural_dir),
                site_stratified_eval_dir=_optional_path_to_str(site_stratified_eval_dir),
                site_aggregation_dir=_optional_path_to_str(site_aggregation_dir),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not compare site models: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table("BABAPPA Site Model Comparison", summary, ["outdir", "json", "recommendation"])


@app.command("validate-site-model-comparison")
def validate_site_model_comparison(
    compare_dir: Path = typer.Option(..., "--compare-dir")
) -> None:
    """Validate site model comparison artifacts."""
    summary = validate_site_model_comparison_dir(compare_dir)
    _print_validation_table("BABAPPA Site Model Comparison Validation", summary)
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("aggregation-controls")
def aggregation_controls_command(
    predictions: Path = typer.Option(..., "--predictions"),
    gene_dataset_dir: Path = typer.Option(..., "--gene-dataset-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    probability_column: str = typer.Option("prob_positive", "--probability-column"),
    n_permutations: int = typer.Option(50, "--n-permutations"),
    seed: int = typer.Option(42, "--seed"),
    workers: int = typer.Option(1, "--workers", help="Parallel worker processes for permutation controls."),
) -> None:
    """Run null/decoy controls for site-to-gene aggregation."""
    try:
        summary = run_site_aggregation_controls(
            SiteAggregationControlConfig(
                predictions_tsv=str(predictions),
                gene_dataset_dir=str(gene_dataset_dir),
                outdir=str(outdir),
                probability_column=probability_column,
                n_permutations=n_permutations,
                seed=seed,
                workers=workers,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not run aggregation controls: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table("BABAPPA Aggregation Controls", summary, ["outdir", "json", "observed_auroc"])


@app.command("validate-aggregation-controls")
def validate_aggregation_controls(
    controls_dir: Path = typer.Option(..., "--controls-dir")
) -> None:
    """Validate aggregation control artifacts."""
    summary = validate_site_aggregation_controls_dir(controls_dir)
    _print_validation_table("BABAPPA Aggregation Controls Validation", summary)
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("aggregation-threshold-policy")
def aggregation_threshold_policy_command(
    outdir: Path = typer.Option(..., "--outdir"),
    aggregation_dir: Optional[Path] = typer.Option(None, "--aggregation-dir"),
    predictions: Optional[Path] = typer.Option(None, "--predictions"),
    score_column: str = typer.Option("max_site_probability", "--score-column"),
    label_column: str = typer.Option("gene_label", "--label-column"),
    split_column: str = typer.Option("split", "--split-column"),
    selection_split: str = typer.Option("calib", "--selection-split"),
    target_fdr: float = typer.Option(0.10, "--target-fdr"),
    precision_floor: float = typer.Option(0.80, "--precision-floor"),
    recall_floor: float = typer.Option(0.80, "--recall-floor"),
) -> None:
    """Build threshold profiles for site-to-gene aggregation scores."""
    try:
        summary = build_aggregation_threshold_policy(
            AggregationThresholdPolicyConfig(
                aggregation_dir=_optional_path_to_str(aggregation_dir),
                predictions_tsv=_optional_path_to_str(predictions),
                outdir=str(outdir),
                score_column=score_column,
                label_column=label_column,
                split_column=split_column,
                selection_split=selection_split,
                target_fdr=target_fdr,
                precision_floor=precision_floor,
                recall_floor=recall_floor,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not build aggregation threshold policy: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table("BABAPPA Aggregation Threshold Policy", summary, ["outdir", "profiles_json", "warnings"])


@app.command("validate-aggregation-threshold-policy")
def validate_aggregation_threshold_policy(
    policy_dir: Path = typer.Option(..., "--policy-dir")
) -> None:
    """Validate aggregation threshold-policy artifacts."""
    summary = validate_aggregation_threshold_policy_dir(policy_dir)
    _print_validation_table("BABAPPA Aggregation Threshold Policy Validation", summary)
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("extract-branch-site-labels")
def extract_branch_site_labels_command(
    dataset_dir: Path = typer.Option(..., "--dataset-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    site_map_dir: Optional[Path] = typer.Option(None, "--site-map-dir"),
    aligned_site_mode: str = typer.Option("mapped", "--aligned-site-mode"),
    foreground_source: str = typer.Option("auto", "--foreground-source"),
    truth_mode: str = typer.Option("auto", "--truth-mode"),
    streaming_output: bool = typer.Option(
        True,
        "--streaming-output/--no-streaming-output",
    ),
) -> None:
    """Extract branch-conditioned oracle labels from a BABAPPA tensor dataset."""
    try:
        summary = extract_branch_site_labels(
            BranchSiteOracleLabelConfig(
                dataset_dir=str(dataset_dir),
                site_map_dir=_optional_path_to_str(site_map_dir),
                outdir=str(outdir),
                aligned_site_mode=aligned_site_mode,
                foreground_source=foreground_source,
                truth_mode=truth_mode,
                streaming_output=streaming_output,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not extract branch-site labels: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Branch-Site Oracle Labels",
        summary,
        ["outdir", "branch_site_labels_tsv", "summary_json", "warnings"],
    )
    _print_warnings(summary["warnings"])


@app.command("validate-branch-site-labels")
def validate_branch_site_labels(
    label_dir: Path = typer.Option(..., "--label-dir")
) -> None:
    """Validate branch-site oracle label artifacts."""
    summary = validate_branch_site_label_dir(label_dir)
    _print_validation_table("BABAPPA Branch-Site Label Validation", summary, "n_rows")
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("build-branch-site-dataset")
def build_branch_site_dataset_command(
    dataset_dir: Path = typer.Option(..., "--dataset-dir"),
    branch_site_labels: Path = typer.Option(..., "--branch-site-labels"),
    outdir: Path = typer.Option(..., "--outdir"),
    negative_downsample_ratio: Optional[float] = typer.Option(
        None, "--negative-downsample-ratio"
    ),
    seed: int = typer.Option(42, "--seed"),
    require_mappable_sites: bool = typer.Option(
        True,
        "--require-mappable-sites/--allow-unmappable-sites",
    ),
    max_input_rows: Optional[int] = typer.Option(None, "--max-input-rows"),
    max_output_rows: Optional[int] = typer.Option(None, "--max-output-rows"),
    max_rows_per_split: Optional[int] = typer.Option(None, "--max-rows-per-split"),
    max_negatives_per_positive: Optional[float] = typer.Option(None, "--max-negatives-per-positive"),
    streaming: bool = typer.Option(True, "--streaming/--no-streaming"),
) -> None:
    """Build branch-conditioned site features from branch-site labels."""
    try:
        summary = build_branch_site_dataset(
            BranchSiteDatasetConfig(
                dataset_dir=str(dataset_dir),
                branch_site_labels_tsv=str(branch_site_labels),
                outdir=str(outdir),
                negative_downsample_ratio=negative_downsample_ratio,
                seed=seed,
                require_mappable_sites=require_mappable_sites,
                max_input_rows=max_input_rows,
                max_output_rows=max_output_rows,
                max_rows_per_split=max_rows_per_split,
                max_negatives_per_positive=max_negatives_per_positive,
                streaming=streaming,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not build branch-site dataset: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Branch-Site Dataset",
        summary,
        ["outdir", "n_branch_site_rows", "n_positive_branch_sites", "features", "warnings"],
    )
    _print_warnings(summary["warnings"])


@app.command("validate-branch-site-dataset")
def validate_branch_site_dataset(
    branch_site_dataset_dir: Path = typer.Option(..., "--branch-site-dataset-dir")
) -> None:
    """Validate a branch-site dataset."""
    summary = validate_branch_site_dataset_dir(branch_site_dataset_dir)
    _print_validation_table("BABAPPA Branch-Site Dataset Validation", summary, "n_rows")
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("audit-branch-site-leakage")
def audit_branch_site_leakage_command(
    branch_site_dataset_dir: Path = typer.Option(..., "--branch-site-dataset-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Audit branch-site features for leakage and sensitive context."""
    try:
        summary = audit_branch_site_leakage(branch_site_dataset_dir, outdir)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not audit branch-site leakage: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Branch-Site Leakage Audit",
        summary,
        ["outdir", "leakage_status", "json", "warnings"],
    )
    _print_warnings(summary["warnings"])


@app.command("validate-branch-site-leakage")
def validate_branch_site_leakage(
    leakage_dir: Path = typer.Option(..., "--leakage-dir")
) -> None:
    """Validate branch-site leakage audit artifacts."""
    summary = validate_branch_site_leakage_dir(leakage_dir)
    _print_validation_table("BABAPPA Branch-Site Leakage Validation", summary)
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("train-branch-site-baseline")
def train_branch_site_baseline_command(
    branch_site_dataset_dir: Path = typer.Option(..., "--branch-site-dataset-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    seed: int = typer.Option(42, "--seed"),
    epochs: int = typer.Option(300, "--epochs"),
    learning_rate: float = typer.Option(0.05, "--learning-rate"),
    l2: float = typer.Option(0.001, "--l2"),
    positive_class_weight: str = typer.Option("auto", "--positive-class-weight"),
    threshold: float = typer.Option(0.5, "--threshold"),
) -> None:
    """Train a NumPy branch-site logistic baseline."""
    try:
        summary = train_branch_site_baseline(
            BranchSiteBaselineConfig(
                branch_site_dataset_dir=str(branch_site_dataset_dir),
                outdir=str(outdir),
                seed=seed,
                epochs=epochs,
                learning_rate=learning_rate,
                l2=l2,
                positive_class_weight=positive_class_weight,
                threshold=threshold,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not train branch-site baseline: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Branch-Site Baseline",
        summary,
        ["outdir", "model", "predictions", "metrics", "warnings"],
    )
    _print_warnings(summary["warnings"])


@app.command("validate-branch-site-baseline")
def validate_branch_site_baseline(
    model_dir: Path = typer.Option(..., "--model-dir")
) -> None:
    """Validate branch-site baseline artifacts."""
    summary = validate_branch_site_baseline_dir(model_dir)
    _print_validation_table("BABAPPA Branch-Site Baseline Validation", summary, "n_predictions")
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("train-branch-site-neural")
def train_branch_site_neural_command(
    branch_site_dataset_dir: Path = typer.Option(..., "--branch-site-dataset-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    seed: int = typer.Option(42, "--seed"),
    device: str = typer.Option("auto", "--device"),
    epochs: int = typer.Option(30, "--epochs"),
    batch_size: int = typer.Option(256, "--batch-size"),
    learning_rate: float = typer.Option(0.001, "--learning-rate"),
    weight_decay: float = typer.Option(0.0001, "--weight-decay"),
    hidden_dim: int = typer.Option(64, "--hidden-dim"),
    dropout: float = typer.Option(0.1, "--dropout"),
    positive_class_weight: str = typer.Option("auto", "--positive-class-weight"),
    threshold: float = typer.Option(0.5, "--threshold"),
    early_stopping_patience: int = typer.Option(8, "--early-stopping-patience"),
    monitor_metric: str = typer.Option("val_auroc", "--monitor-metric"),
    max_train_items: Optional[int] = typer.Option(None, "--max-train-items"),
    max_val_items: Optional[int] = typer.Option(None, "--max-val-items"),
    max_calib_items: Optional[int] = typer.Option(None, "--max-calib-items"),
    max_test_items: Optional[int] = typer.Option(None, "--max-test-items"),
    feature_policy: str = typer.Option("full_context", "--feature-policy"),
    threads: int = typer.Option(0, "--threads", help="Torch CPU threads for loader/CPU-side tensor work; 0 leaves torch default."),
) -> None:
    """Train a lightweight branch-context neural MLP."""
    try:
        summary = train_branch_site_neural_model(
            BranchSiteNeuralTrainConfig(
                branch_site_dataset_dir=str(branch_site_dataset_dir),
                outdir=str(outdir),
                seed=seed,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                hidden_dim=hidden_dim,
                dropout=dropout,
                positive_class_weight=positive_class_weight,
                threshold=threshold,
                early_stopping_patience=early_stopping_patience,
                monitor_metric=monitor_metric,
                max_train_items=max_train_items,
                max_val_items=max_val_items,
                max_calib_items=max_calib_items,
                max_test_items=max_test_items,
                feature_policy=feature_policy,
                threads=threads,
            )
        )
    except (OSError, ValueError, RuntimeError) as exc:
        console.print(f"Error: could not train branch-site neural model: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Branch-Site Neural",
        summary,
        ["outdir", "checkpoint", "predictions", "metrics", "best_epoch"],
    )


@app.command("validate-branch-site-neural")
def validate_branch_site_neural(
    model_dir: Path = typer.Option(..., "--model-dir")
) -> None:
    """Validate branch-site neural artifacts."""
    summary = validate_branch_site_neural_dir(model_dir)
    _print_validation_table("BABAPPA Branch-Site Neural Validation", summary, "n_predictions")
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("calibrate-branch-site-neural")
def calibrate_branch_site_neural_command(
    model_dir: Path = typer.Option(..., "--model-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    target_fdr: float = typer.Option(0.10, "--target-fdr"),
    calibration_method: str = typer.Option("temperature", "--calibration-method"),
    threshold_grid_size: int = typer.Option(501, "--threshold-grid-size"),
    min_threshold: float = typer.Option(0.0, "--min-threshold"),
    max_threshold: float = typer.Option(1.0, "--max-threshold"),
    n_bins: int = typer.Option(20, "--n-bins"),
) -> None:
    """Calibrate branch-site neural probabilities."""
    try:
        summary = calibrate_branch_site_model(
            BranchSiteCalibrationConfig(
                model_dir=str(model_dir),
                outdir=str(outdir),
                target_fdr=target_fdr,
                calibration_method=calibration_method,
                threshold_grid_size=threshold_grid_size,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
                n_bins=n_bins,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not calibrate branch-site neural model: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Branch-Site Calibration",
        summary,
        ["outdir", "temperature", "selected_threshold", "calibration", "warnings"],
    )
    _print_warnings(summary["warnings"])


@app.command("validate-branch-site-calibration")
def validate_branch_site_calibration(
    calibration_dir: Path = typer.Option(..., "--calibration-dir")
) -> None:
    """Validate branch-site calibration artifacts."""
    summary = validate_branch_site_calibration_dir(calibration_dir)
    _print_validation_table("BABAPPA Branch-Site Calibration Validation", summary, "n_predictions")
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("branch-site-threshold-policy")
def branch_site_threshold_policy_command(
    predictions: Path = typer.Option(..., "--predictions"),
    outdir: Path = typer.Option(..., "--outdir"),
    probability_column: str = typer.Option("prob_positive", "--probability-column"),
    calibrated_probability_column: Optional[str] = typer.Option(
        None, "--calibrated-probability-column"
    ),
    label_column: str = typer.Option("y_branch_site", "--label-column"),
    split_column: str = typer.Option("split", "--split-column"),
    selection_split: str = typer.Option("calib", "--selection-split"),
    target_fdr: float = typer.Option(0.10, "--target-fdr"),
    precision_floor: float = typer.Option(0.80, "--precision-floor"),
    recall_floor: float = typer.Option(0.80, "--recall-floor"),
    threshold_grid_size: int = typer.Option(501, "--threshold-grid-size"),
    min_threshold: float = typer.Option(0.0, "--min-threshold"),
    max_threshold: float = typer.Option(1.0, "--max-threshold"),
    model_name: str = typer.Option("branch_site_model", "--model-name"),
) -> None:
    """Build threshold profiles for branch-site predictions."""
    try:
        summary = build_branch_site_threshold_policy(
            BranchSiteThresholdPolicyConfig(
                predictions_tsv=str(predictions),
                outdir=str(outdir),
                probability_column=probability_column,
                calibrated_probability_column=calibrated_probability_column,
                label_column=label_column,
                split_column=split_column,
                selection_split=selection_split,
                target_fdr=target_fdr,
                precision_floor=precision_floor,
                recall_floor=recall_floor,
                threshold_grid_size=threshold_grid_size,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
                model_name=model_name,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not build branch-site threshold policy: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table("BABAPPA Branch-Site Threshold Policy", summary, ["outdir", "profiles_json", "warnings"])
    _print_warnings(summary["warnings"])


@app.command("validate-branch-site-threshold-policy")
def validate_branch_site_threshold_policy(
    policy_dir: Path = typer.Option(..., "--policy-dir")
) -> None:
    """Validate branch-site threshold-policy artifacts."""
    summary = validate_branch_site_threshold_policy_dir(policy_dir)
    _print_validation_table("BABAPPA Branch-Site Threshold Policy Validation", summary)
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("aggregate-branch-sites")
def aggregate_branch_sites_command(
    predictions: Path = typer.Option(..., "--predictions"),
    outdir: Path = typer.Option(..., "--outdir"),
    probability_column: str = typer.Option("prob_positive", "--probability-column"),
) -> None:
    """Aggregate branch-site probabilities to branch and gene support."""
    try:
        summary = aggregate_branch_sites(
            BranchAggregationConfig(
                predictions_tsv=str(predictions),
                outdir=str(outdir),
                probability_column=probability_column,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not aggregate branch-site predictions: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Branch Aggregation",
        summary,
        ["outdir", "n_site_to_branch_rows", "n_branch_to_gene_rows", "metrics"],
    )


@app.command("validate-branch-aggregation")
def validate_branch_aggregation(
    aggregation_dir: Path = typer.Option(..., "--aggregation-dir")
) -> None:
    """Validate branch aggregation artifacts."""
    summary = validate_branch_aggregation_dir(aggregation_dir)
    _print_validation_table("BABAPPA Branch Aggregation Validation", summary, "n_rows")
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("branch-aggregation-controls")
def branch_aggregation_controls_command(
    predictions: Path = typer.Option(..., "--predictions"),
    outdir: Path = typer.Option(..., "--outdir"),
    probability_column: str = typer.Option("prob_positive", "--probability-column"),
    n_permutations: int = typer.Option(50, "--n-permutations"),
    seed: int = typer.Option(42, "--seed"),
    workers: int = typer.Option(1, "--workers", help="Parallel worker processes for permutation controls."),
) -> None:
    """Run branch aggregation null/decoy controls."""
    try:
        summary = run_branch_aggregation_controls(
            BranchAggregationControlConfig(
                predictions_tsv=str(predictions),
                outdir=str(outdir),
                probability_column=probability_column,
                n_permutations=n_permutations,
                seed=seed,
                workers=workers,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not run branch aggregation controls: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Branch Aggregation Controls",
        summary,
        ["outdir", "json", "observed_auroc", "workers"],
    )


@app.command("validate-branch-aggregation-controls")
def validate_branch_aggregation_controls(
    controls_dir: Path = typer.Option(..., "--controls-dir")
) -> None:
    """Validate branch aggregation controls."""
    summary = validate_branch_aggregation_controls_dir(controls_dir)
    _print_validation_table("BABAPPA Branch Aggregation Controls Validation", summary)
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("plan-rerun-branch-aggregation-controls")
def plan_rerun_branch_aggregation_controls_command(
    run_name: str = typer.Option(..., "--run-name"),
    tiers: str = typer.Option("low,moderate,high,extreme", "--tiers"),
    output_suffix: str = typer.Option("_streamed", "--output-suffix"),
    outdir: Path = typer.Option(..., "--outdir"),
    n_permutations: int = typer.Option(100, "--n-permutations"),
    seed: int = typer.Option(42, "--seed"),
    workers: int = typer.Option(1, "--workers"),
) -> None:
    """Write a script to rerun only branch aggregation controls."""
    try:
        summary = plan_rerun_branch_aggregation_controls(
            BranchAggregationControlsRerunPlanConfig(
                run_name=run_name,
                tiers=tiers,
                output_suffix=output_suffix,
                outdir=str(outdir),
                n_permutations=n_permutations,
                seed=seed,
                workers=workers,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan branch aggregation controls rerun: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Branch Aggregation Controls Rerun Plan",
        summary,
        ["outdir", "run", "expected_outputs", "controls_included", "does_not_run_jobs"],
    )


@app.command("branch-aggregation-threshold-policy")
def branch_aggregation_threshold_policy_command(
    outdir: Path = typer.Option(..., "--outdir"),
    aggregation_dir: Optional[Path] = typer.Option(None, "--aggregation-dir"),
    predictions: Optional[Path] = typer.Option(None, "--predictions"),
    score_column: str = typer.Option("max_branch_probability", "--score-column"),
    label_column: str = typer.Option("gene_label", "--label-column"),
    split_column: str = typer.Option("split", "--split-column"),
    selection_split: str = typer.Option("calib", "--selection-split"),
    target_fdr: float = typer.Option(0.10, "--target-fdr"),
    precision_floor: float = typer.Option(0.80, "--precision-floor"),
    recall_floor: float = typer.Option(0.80, "--recall-floor"),
) -> None:
    """Build threshold profiles for branch-to-gene aggregation."""
    try:
        summary = build_branch_aggregation_threshold_policy(
            BranchAggregationThresholdPolicyConfig(
                outdir=str(outdir),
                aggregation_dir=_optional_path_to_str(aggregation_dir),
                predictions_tsv=_optional_path_to_str(predictions),
                score_column=score_column,
                label_column=label_column,
                split_column=split_column,
                selection_split=selection_split,
                target_fdr=target_fdr,
                precision_floor=precision_floor,
                recall_floor=recall_floor,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not build branch aggregation threshold policy: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table("BABAPPA Branch Aggregation Threshold Policy", summary, ["outdir", "profiles_json", "warnings"])
    _print_warnings(summary["warnings"])


@app.command("validate-branch-aggregation-threshold-policy")
def validate_branch_aggregation_threshold_policy(
    policy_dir: Path = typer.Option(..., "--policy-dir")
) -> None:
    """Validate branch aggregation threshold-policy artifacts."""
    summary = validate_branch_aggregation_threshold_policy_dir(policy_dir)
    _print_validation_table("BABAPPA Branch Aggregation Threshold Policy Validation", summary)
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("summarize-branch-site-run")
def summarize_branch_site_run_command(
    outdir: Path = typer.Option(..., "--outdir"),
    title: str = typer.Option("BABAPPA branch-conditioned validation summary", "--title"),
    branch_site_label_dir: Optional[Path] = typer.Option(None, "--branch-site-label-dir"),
    branch_site_dataset_dir: Optional[Path] = typer.Option(None, "--branch-site-dataset-dir"),
    branch_site_leakage_dir: Optional[Path] = typer.Option(None, "--branch-site-leakage-dir"),
    branch_site_baseline_dir: Optional[Path] = typer.Option(None, "--branch-site-baseline-dir"),
    branch_site_neural_dir: Optional[Path] = typer.Option(None, "--branch-site-neural-dir"),
    branch_site_calibration_dir: Optional[Path] = typer.Option(None, "--branch-site-calibration-dir"),
    branch_aggregation_dir: Optional[Path] = typer.Option(None, "--branch-aggregation-dir"),
    branch_aggregation_controls_dir: Optional[Path] = typer.Option(None, "--branch-aggregation-controls-dir"),
    branch_site_threshold_policy_dir: Optional[Path] = typer.Option(None, "--branch-site-threshold-policy-dir"),
    branch_aggregation_threshold_policy_dir: Optional[Path] = typer.Option(None, "--branch-aggregation-threshold-policy-dir"),
) -> None:
    """Summarize a branch-conditioned validation run."""
    try:
        summary = summarize_branch_site_run(
            BranchSiteRunSummaryConfig(
                outdir=str(outdir),
                title=title,
                branch_site_label_dir=_optional_path_to_str(branch_site_label_dir),
                branch_site_dataset_dir=_optional_path_to_str(branch_site_dataset_dir),
                branch_site_leakage_dir=_optional_path_to_str(branch_site_leakage_dir),
                branch_site_baseline_dir=_optional_path_to_str(branch_site_baseline_dir),
                branch_site_neural_dir=_optional_path_to_str(branch_site_neural_dir),
                branch_site_calibration_dir=_optional_path_to_str(branch_site_calibration_dir),
                branch_aggregation_dir=_optional_path_to_str(branch_aggregation_dir),
                branch_aggregation_controls_dir=_optional_path_to_str(branch_aggregation_controls_dir),
                branch_site_threshold_policy_dir=_optional_path_to_str(branch_site_threshold_policy_dir),
                branch_aggregation_threshold_policy_dir=_optional_path_to_str(branch_aggregation_threshold_policy_dir),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not summarize branch-site run: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table("BABAPPA Branch-Site Run Summary", summary, ["outdir", "json", "markdown", "warnings"])
    _print_warnings(summary["warnings"])


@app.command("validate-branch-site-run-summary")
def validate_branch_site_run_summary(
    summary_dir: Path = typer.Option(..., "--summary-dir")
) -> None:
    """Validate branch-site run summary artifacts."""
    summary = validate_branch_site_run_summary_dir(summary_dir)
    _print_validation_table("BABAPPA Branch-Site Run Summary Validation", summary)
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("plan-branch-conditioned-10k")
def plan_branch_conditioned_10k_command(
    outdir: Path = typer.Option("branch_conditioned_10k_plan", "--outdir"),
    tiers: str = typer.Option("low,moderate,high,extreme", "--tiers"),
    negative_downsample_ratio: float = typer.Option(5.0, "--negative-downsample-ratio"),
    max_output_rows_per_tier: int = typer.Option(1_000_000, "--max-output-rows-per-tier"),
    output_suffix: str = typer.Option("streamed", "--output-suffix"),
    conda_sh: str = typer.Option(
        "/home/rajamosai/miniconda3/etc/profile.d/conda.sh",
        "--conda-sh",
    ),
    conda_env: str = typer.Option("molevo", "--conda-env"),
    neural_epochs: int = typer.Option(10, "--neural-epochs"),
    batch_size: int = typer.Option(256, "--batch-size"),
    max_train_items: int = typer.Option(50000, "--max-train-items"),
    max_val_items: int = typer.Option(10000, "--max-val-items"),
    max_calib_items: int = typer.Option(10000, "--max-calib-items"),
    max_test_items: int = typer.Option(10000, "--max-test-items"),
    n_control_permutations: int = typer.Option(20, "--n-control-permutations"),
) -> None:
    """Write a user-run branch-conditioned 10K validation plan."""
    try:
        summary = plan_branch_conditioned_10k(
            BranchConditioned10kPlanConfig(
                outdir=str(outdir),
                tiers=_parse_methods(tiers),
                negative_downsample_ratio=negative_downsample_ratio,
                max_output_rows_per_tier=max_output_rows_per_tier,
                output_suffix=output_suffix,
                conda_sh=conda_sh,
                conda_env=conda_env,
                neural_epochs=neural_epochs,
                batch_size=batch_size,
                max_train_items=max_train_items,
                max_val_items=max_val_items,
                max_calib_items=max_calib_items,
                max_test_items=max_test_items,
                n_control_permutations=n_control_permutations,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan branch-conditioned 10K: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Branch-Conditioned 10K Plan",
        summary,
        ["outdir", "run", "monitor", "validate", "summarize", "expected_outputs"],
    )


@app.command("summarize-branch-conditioned-tiers")
def summarize_branch_conditioned_tiers_command(
    tiers: str = typer.Option("low,moderate,high,extreme", "--tiers"),
    run_name: str = typer.Option("fast_external_10k_streamed", "--run-name"),
    output_suffix: Optional[str] = typer.Option(None, "--output-suffix"),
    allow_streamed: str = typer.Option("true", "--allow-streamed"),
    ablation_summary_dir: Optional[Path] = typer.Option(None, "--ablation-summary-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Summarize completed branch-conditioned tiers into one cross-tier report."""
    try:
        summary = summarize_branch_conditioned_tiers(
            BranchConditionedTierSummaryConfig(
                tiers=tiers,
                run_name=run_name,
                output_suffix=output_suffix,
                allow_streamed=_parse_bool(allow_streamed),
                ablation_summary_dir=str(ablation_summary_dir) if ablation_summary_dir else None,
                outdir=str(outdir),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not summarize branch-conditioned tiers: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Branch-Conditioned Tier Summary",
        summary,
        ["outdir", "json_summary", "markdown_summary", "tiers_included", "n_warning"],
    )


@app.command("validate-branch-conditioned-tier-summary")
def validate_branch_conditioned_tier_summary(
    summary_dir: Path = typer.Option(..., "--summary-dir")
) -> None:
    """Validate a branch-conditioned cross-tier summary directory."""
    summary = validate_branch_conditioned_tier_summary_dir(summary_dir)
    _print_validation_table("BABAPPA Branch-Conditioned Tier Summary Validation", summary)
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("audit-branch-truth-status")
def audit_branch_truth_status_command(
    tiers: str = typer.Option("low,moderate,high,extreme", "--tiers"),
    run_name: str = typer.Option("fast_external_10k_streamed", "--run-name"),
    output_suffix: Optional[str] = typer.Option(None, "--output-suffix"),
    allow_streamed: str = typer.Option("true", "--allow-streamed"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Audit whether branch-site labels are explicit truth or proxies."""
    try:
        summary = audit_branch_truth_status(
            BranchTruthStatusAuditConfig(
                tiers=tiers,
                run_name=run_name,
                output_suffix=output_suffix,
                allow_streamed=_parse_bool(allow_streamed),
                outdir=str(outdir),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not audit branch truth status: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Branch Truth-Status Audit",
        summary,
        ["outdir", "json", "tsv", "markdown", "explicit_truth_available", "proxy_label_tiers"],
    )


@app.command("validate-branch-truth-status-audit")
def validate_branch_truth_status_audit(
    audit_dir: Path = typer.Option(..., "--audit-dir")
) -> None:
    """Validate a branch truth-status audit directory."""
    summary = validate_branch_truth_status_audit_dir(audit_dir)
    _print_validation_table("BABAPPA Branch Truth-Status Audit Validation", summary)
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("plan-branch-context-ablation")
def plan_branch_context_ablation_command(
    run_name: str = typer.Option(..., "--run-name"),
    tiers: str = typer.Option("low,moderate,high,extreme", "--tiers"),
    output_suffix: str = typer.Option("_streamed", "--output-suffix"),
    outdir: Path = typer.Option(..., "--outdir"),
    profiles: str = typer.Option(
        "full_model,no_foreground_identity,no_foreground_codon_context,no_foreground_all,context_only",
        "--profiles",
    ),
    ablation_root: str = typer.Option("branch_context_ablation_explicit_1k", "--ablation-root"),
    seed: int = typer.Option(42, "--seed"),
    epochs: int = typer.Option(300, "--epochs"),
    learning_rate: float = typer.Option(0.05, "--learning-rate"),
) -> None:
    """Write branch foreground-context ablation scripts without running jobs."""
    try:
        summary = plan_branch_context_ablation(
            BranchContextAblationPlanConfig(
                run_name=run_name,
                tiers=tiers,
                output_suffix=output_suffix,
                outdir=str(outdir),
                profiles=profiles,
                ablation_root=ablation_root,
                seed=seed,
                epochs=epochs,
                learning_rate=learning_rate,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan branch-context ablation: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Branch Context Ablation Plan",
        summary,
        ["outdir", "run", "expected_outputs", "does_not_run_jobs", "profiles"],
    )


@app.command("run-branch-context-ablation")
def run_branch_context_ablation_command(
    branch_site_dataset_dir: Path = typer.Option(..., "--branch-site-dataset-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    profiles: str = typer.Option(
        "full_model,no_foreground_identity,no_foreground_codon_context,no_foreground_all,context_only",
        "--profiles",
    ),
    model: str = typer.Option("baseline", "--model"),
    seed: int = typer.Option(42, "--seed"),
    epochs: int = typer.Option(300, "--epochs"),
    learning_rate: float = typer.Option(0.05, "--learning-rate"),
) -> None:
    """Run baseline foreground-context ablation profiles for one branch-site dataset."""
    try:
        summary = run_branch_context_ablation(
            BranchContextAblationRunConfig(
                branch_site_dataset_dir=str(branch_site_dataset_dir),
                outdir=str(outdir),
                profiles=profiles,
                model=model,
                seed=seed,
                epochs=epochs,
                learning_rate=learning_rate,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not run branch-context ablation: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Branch Context Ablation",
        summary,
        ["outdir", "profiles", "n_profiles", "summary"],
    )


@app.command("summarize-branch-context-ablation")
def summarize_branch_context_ablation_command(
    ablation_dir: Path = typer.Option(..., "--ablation-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Summarize branch foreground-context ablation outputs."""
    try:
        summary = summarize_branch_context_ablation(
            BranchContextAblationSummaryConfig(
                ablation_dir=str(ablation_dir),
                outdir=str(outdir),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not summarize branch-context ablation: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Branch Context Ablation Summary",
        summary,
        ["outdir", "json", "tsv", "markdown", "n_rows"],
    )


@app.command("interpret-branch-context-ablation")
def interpret_branch_context_ablation_command(
    summary_dir: Path = typer.Option(..., "--summary-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Interpret branch-context ablation as a branch feature policy decision."""
    try:
        summary = interpret_branch_context_ablation(
            BranchContextAblationInterpretationConfig(
                summary_dir=str(summary_dir),
                outdir=str(outdir),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not interpret branch-context ablation: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Branch Context Ablation Interpretation",
        summary,
        ["outdir", "json", "warnings", "conclusions", "recommended_next_default"],
    )


@app.command("list-branch-feature-policies")
def list_branch_feature_policies_command() -> None:
    """List named branch-site feature policies."""
    rows = list_branch_feature_policies()
    table = Table(title="BABAPPA Branch Feature Policies")
    for column in [
        "policy",
        "label",
        "recommended_role",
        "production_default",
        "included_columns",
        "excluded_columns",
        "warning",
    ]:
        table.add_column(column)
    for row in rows:
        table.add_row(
            str(row["policy"]),
            str(row["label"]),
            str(row["recommended_role"]),
            str(row["production_default"]),
            str(row["included_columns"]),
            str(row["excluded_columns"]),
            str(row["warning"]),
        )
    console.print(table)


@app.command("plan-explicit-branch-truth-prototype")
def plan_explicit_branch_truth_prototype_command(
    outdir: Path = typer.Option("explicit_branch_truth_prototype_plan", "--outdir"),
    n_families: int = typer.Option(1000, "--n-families", min=1),
    tiers: str = typer.Option("low,moderate,high,extreme", "--tiers"),
    methods: str = typer.Option("identity,mafft,babappalign,muscle", "--methods"),
) -> None:
    """Write a future-facing explicit branch-site truth prototype plan."""
    try:
        summary = plan_explicit_branch_truth_prototype(
            ExplicitBranchTruthPrototypePlanConfig(
                outdir=str(outdir),
                n_families=n_families,
                tiers=tiers,
                methods=methods,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan explicit branch truth prototype: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Explicit Branch-Truth Prototype Plan",
        summary,
        ["outdir", "run", "monitor", "validate", "expected_outputs", "markdown", "does_not_run_jobs"],
    )


@app.command("plan-explicit-branch-truth-1k")
def plan_explicit_branch_truth_1k_command(
    outdir: Path = typer.Option("explicit_branch_truth_1k_plan", "--outdir"),
    n_families_per_tier: int = typer.Option(250, "--n-families-per-tier", min=1),
    tiers: str = typer.Option("low,moderate,high,extreme", "--tiers"),
    methods: str = typer.Option("identity,mafft,babappalign,muscle", "--methods"),
    negative_downsample_ratio: float = typer.Option(5.0, "--negative-downsample-ratio"),
    conda_sh: str = typer.Option(
        "/home/rajamosai/miniconda3/etc/profile.d/conda.sh",
        "--conda-sh",
    ),
    conda_env: str = typer.Option("molevo", "--conda-env"),
) -> None:
    """Write a user-run explicit branch-truth 1K validation plan."""
    try:
        summary = plan_explicit_branch_truth_1k(
            ExplicitBranchTruth1kPlanConfig(
                outdir=str(outdir),
                n_families_per_tier=n_families_per_tier,
                tiers=tiers,
                methods=methods,
                negative_downsample_ratio=negative_downsample_ratio,
                conda_sh=conda_sh,
                conda_env=conda_env,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan explicit branch truth 1K: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Explicit Branch-Truth 1K Plan",
        summary,
        ["outdir", "run", "monitor", "validate", "summarize", "expected_outputs", "does_not_run_jobs"],
    )


@app.command("plan-explicit-branch-truth-10k")
def plan_explicit_branch_truth_10k_command(
    outdir: Path = typer.Option("explicit_branch_truth_10k_plan", "--outdir"),
    n_families_per_tier: int = typer.Option(2500, "--n-families-per-tier", min=1),
    tiers: str = typer.Option("low,moderate,high,extreme", "--tiers"),
    methods: str = typer.Option("identity,mafft,babappalign,muscle", "--methods"),
    feature_policy: str = typer.Option("conservative_branch_site", "--feature-policy"),
    negative_downsample_ratio: float = typer.Option(5.0, "--negative-downsample-ratio"),
    max_output_rows_per_tier: int = typer.Option(1_000_000, "--max-output-rows-per-tier", min=1),
    conda_sh: str = typer.Option(
        "/home/rajamosai/miniconda3/etc/profile.d/conda.sh",
        "--conda-sh",
    ),
    conda_env: str = typer.Option("molevo", "--conda-env"),
) -> None:
    """Write a conservative user-run explicit branch-truth 10K validation plan."""
    try:
        summary = plan_explicit_branch_truth_10k(
            ExplicitBranchTruth10kPlanConfig(
                outdir=str(outdir),
                n_families_per_tier=n_families_per_tier,
                tiers=tiers,
                methods=methods,
                feature_policy=feature_policy,
                negative_downsample_ratio=negative_downsample_ratio,
                max_output_rows_per_tier=max_output_rows_per_tier,
                conda_sh=conda_sh,
                conda_env=conda_env,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan explicit branch truth 10K: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Explicit Branch-Truth 10K Plan",
        summary,
        [
            "outdir",
            "run",
            "monitor",
            "validate",
            "summarize",
            "expected_outputs",
            "feature_policy",
            "truth_mode",
            "does_not_run_jobs",
        ],
    )


@app.command("plan-explicit-branch-truth-10k-mac")
def plan_explicit_branch_truth_10k_mac_command(
    outdir: Path = typer.Option("explicit_branch_truth_10k_mps_plan", "--outdir"),
    n_families_per_tier: int = typer.Option(2500, "--n-families-per-tier", min=1),
    tiers: str = typer.Option("low,moderate,high,extreme", "--tiers"),
    methods: str = typer.Option("identity,mafft,babappalign,muscle", "--methods"),
    feature_policy: str = typer.Option("conservative_branch_site", "--feature-policy"),
    truth_mode: str = typer.Option("explicit", "--truth-mode"),
    negative_downsample_ratio: float = typer.Option(5.0, "--negative-downsample-ratio"),
    max_output_rows_per_tier: int = typer.Option(1_000_000, "--max-output-rows-per-tier", min=1),
    device: str = typer.Option("mps", "--device"),
    batch_size: int = typer.Option(128, "--batch-size"),
    threads: int = typer.Option(8, "--threads"),
    conda_env: str = typer.Option("molevo", "--conda-env"),
    mps_fallback: str = typer.Option("true", "--mps-fallback"),
    mps_high_watermark_ratio: Optional[float] = typer.Option(None, "--mps-high-watermark-ratio"),
    allow_missing_babappalign: bool = typer.Option(
        False,
        "--allow-missing-babappalign",
        help="Generate a plan that allows BABAPPAlign stages to proceed without the local BABAPPAScore model preflight.",
    ),
) -> None:
    """Write a user-run Apple Silicon/MPS explicit branch-truth 10K plan."""
    try:
        summary = plan_explicit_branch_truth_10k_mac(
            ExplicitBranchTruth10kMacPlanConfig(
                outdir=str(outdir),
                n_families_per_tier=n_families_per_tier,
                tiers=tiers,
                methods=methods,
                feature_policy=feature_policy,
                truth_mode=truth_mode,
                negative_downsample_ratio=negative_downsample_ratio,
                max_output_rows_per_tier=max_output_rows_per_tier,
                device=device,
                batch_size=batch_size,
                threads=threads,
                conda_env=conda_env,
                mps_fallback=_parse_bool(mps_fallback),
                mps_high_watermark_ratio=mps_high_watermark_ratio,
                allow_missing_babappalign=allow_missing_babappalign,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan explicit branch truth 10K MPS: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Explicit Branch-Truth 10K MPS Plan",
        summary,
        ["outdir", "run", "monitor", "validate", "summarize", "expected_outputs", "device", "batch_size", "feature_policy", "truth_mode", "does_not_run_jobs"],
    )


@app.command("plan-explicit-branch-truth-100k-mac")
def plan_explicit_branch_truth_100k_mac_command(
    outdir: Path = typer.Option("explicit_branch_truth_100k_mps_plan", "--outdir"),
    n_families_per_tier: int = typer.Option(25000, "--n-families-per-tier", min=1),
    tiers: str = typer.Option("low,moderate,high,extreme", "--tiers"),
    methods: str = typer.Option("identity,mafft,babappalign,muscle", "--methods"),
    feature_policy: str = typer.Option("conservative_branch_site", "--feature-policy"),
    truth_mode: str = typer.Option("explicit", "--truth-mode"),
    negative_downsample_ratio: float = typer.Option(5.0, "--negative-downsample-ratio"),
    max_output_rows_per_tier: int = typer.Option(2_000_000, "--max-output-rows-per-tier", min=1),
    device: str = typer.Option("mps", "--device"),
    batch_size: int = typer.Option(64, "--batch-size"),
    threads: int = typer.Option(8, "--threads"),
    conda_env: str = typer.Option("molevo", "--conda-env"),
    mps_fallback: str = typer.Option("true", "--mps-fallback"),
    mps_high_watermark_ratio: Optional[float] = typer.Option(None, "--mps-high-watermark-ratio"),
    allow_missing_babappalign: bool = typer.Option(
        False,
        "--allow-missing-babappalign",
        help="Generate a plan that allows BABAPPAlign stages to proceed without the local BABAPPAScore model preflight.",
    ),
) -> None:
    """Write a gated user-run Apple Silicon/MPS explicit branch-truth 100K plan."""
    try:
        summary = plan_explicit_branch_truth_100k_mac(
            ExplicitBranchTruth100kMacPlanConfig(
                outdir=str(outdir),
                n_families_per_tier=n_families_per_tier,
                tiers=tiers,
                methods=methods,
                feature_policy=feature_policy,
                truth_mode=truth_mode,
                negative_downsample_ratio=negative_downsample_ratio,
                max_output_rows_per_tier=max_output_rows_per_tier,
                device=device,
                batch_size=batch_size,
                threads=threads,
                conda_env=conda_env,
                mps_fallback=_parse_bool(mps_fallback),
                mps_high_watermark_ratio=mps_high_watermark_ratio,
                allow_missing_babappalign=allow_missing_babappalign,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan explicit branch truth 100K MPS: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Explicit Branch-Truth 100K MPS Plan",
        summary,
        ["outdir", "run", "monitor", "validate", "summarize", "expected_outputs", "device", "batch_size", "blocked_until_10k_passes", "does_not_run_jobs"],
    )


@app.command("validate-mps-plan-script")
def validate_mps_plan_script_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
    scale: Optional[str] = typer.Option(None, "--scale"),
) -> None:
    """Validate generated Mac MPS plan scripts without environment smokes."""
    try:
        summary = validate_mps_plan_script(
            MPSPlanScriptValidationConfig(plan_dir=str(plan_dir), scale=scale)
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not validate MPS plan script: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA MPS Plan Script Validation",
        summary,
        ["status", "n_checks", "n_fail", "n_warn"],
    )
    reports = summary.get("reports", {})
    if reports:
        console.print(f"JSON: {reports.get('json')}")
        console.print(f"TSV: {reports.get('tsv')}")
        console.print(f"Markdown: {reports.get('markdown')}")
    if summary.get("status") == "fail":
        raise typer.Exit(code=1)


@app.command("preflight-explicit-branch-truth-mps-plan")
def preflight_explicit_branch_truth_mps_plan_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
    scale: str = typer.Option(..., "--scale"),
    require_babappalign: str = typer.Option("true", "--require-babappalign"),
    require_mps: str = typer.Option("true", "--require-mps"),
    conda_env: str = typer.Option("molevo", "--conda-env"),
    allow_partial_resume: bool = typer.Option(False, "--allow-partial-resume"),
    run_align_external_smoke: bool = typer.Option(
        False,
        "--run-align-external-smoke",
        help="Run the optional tiny BABAPPA align-external smoke.",
    ),
) -> None:
    """Run fast preflight checks before generated Mac MPS 10K/100K scripts."""
    try:
        summary = preflight_explicit_branch_truth_mps_plan(
            MPSPlanPreflightConfig(
                plan_dir=str(plan_dir),
                scale=scale,
                require_babappalign=_parse_bool(require_babappalign),
                require_mps=_parse_bool(require_mps),
                conda_env=conda_env,
                allow_partial_resume=allow_partial_resume,
                run_align_external_smoke=run_align_external_smoke,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not preflight MPS plan: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA MPS Plan Preflight",
        summary,
        ["status", "n_checks", "n_fail", "n_warn"],
    )
    reports = summary.get("reports", {})
    if reports:
        console.print(f"JSON: {reports.get('json')}")
        console.print(f"TSV: {reports.get('tsv')}")
        console.print(f"Markdown: {reports.get('markdown')}")
    if summary.get("status") == "fail":
        raise typer.Exit(code=1)


@app.command("compare-validation-scales")
def compare_validation_scales_command(
    small_run: str = typer.Option(..., "--small-run"),
    large_run: str = typer.Option(..., "--large-run"),
    small_summary: Path = typer.Option(..., "--small-summary"),
    large_summary: Path = typer.Option(..., "--large-summary"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Compare completed explicit branch-truth validation scales."""
    try:
        summary = compare_validation_scales(
            ValidationScaleComparisonConfig(
                small_run=small_run,
                large_run=large_run,
                small_summary=str(small_summary),
                large_summary=str(large_summary),
                outdir=str(outdir),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not compare validation scales: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Validation Scale Comparison",
        summary,
        ["status", "outdir", "json", "tsv", "markdown", "n_tiers"],
    )


@app.command("build-explicit-branch-truth-100k-validation-report")
def build_explicit_branch_truth_100k_validation_report_command(
    run_name: str = typer.Option("explicit_branch_truth_100k_mps", "--run-name"),
    summary_dir: Path = typer.Option(..., "--summary-dir"),
    truth_audit_dir: Path = typer.Option(..., "--truth-audit-dir"),
    plan_dir: Path = typer.Option(..., "--plan-dir"),
    comparison_dir: Optional[Path] = typer.Option(None, "--comparison-dir"),
    outdir: Path = typer.Option(".", "--outdir"),
) -> None:
    """Build final validation reports for completed explicit branch-truth 100K MPS."""
    try:
        summary = build_final_100k_validation_report(
            Final100KValidationReportConfig(
                run_name=run_name,
                summary_dir=str(summary_dir),
                truth_audit_dir=str(truth_audit_dir),
                plan_dir=str(plan_dir),
                comparison_dir=str(comparison_dir) if comparison_dir else None,
                outdir=str(outdir),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not build final 100K validation report: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Explicit Branch-Truth 100K Validation Report",
        summary,
        ["status", "decision", "json", "tsv", "markdown"],
    )


@app.command("plan-deployable-model-package")
def plan_deployable_model_package_command(
    run_name: str = typer.Option(..., "--run-name"),
    summary_dir: Path = typer.Option(..., "--summary-dir"),
    truth_audit_dir: Path = typer.Option(..., "--truth-audit-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    feature_policy: str = typer.Option("conservative_branch_site", "--feature-policy"),
    truth_mode: str = typer.Option("explicit", "--truth-mode"),
    methods: str = typer.Option("identity,mafft,babappalign,muscle", "--methods"),
) -> None:
    """Plan a deployable conservative model package without packaging automatically."""
    try:
        summary = plan_deployable_model_package(
            DeployableModelPackagePlanConfig(
                run_name=run_name,
                summary_dir=str(summary_dir),
                truth_audit_dir=str(truth_audit_dir),
                outdir=str(outdir),
                feature_policy=feature_policy,
                truth_mode=truth_mode,
                methods=methods.split(","),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan deployable model package: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Deployable Model Package Plan",
        summary,
        ["status", "outdir", "blocked", "manifest", "model_card", "package_script", "validate_script"],
    )
    if summary.get("missing_artifacts"):
        _print_warnings([f"missing_artifact:{item}" for item in summary["missing_artifacts"]])


@app.command("package-deployable-model")
def package_deployable_model_command(
    run_name: str = typer.Option(..., "--run-name"),
    model_dirs: str = typer.Option(..., "--model-dirs"),
    calibration_dirs: str = typer.Option(..., "--calibration-dirs"),
    truth_audit_dir: Path = typer.Option(..., "--truth-audit-dir"),
    validation_report: Path = typer.Option(..., "--validation-report"),
    feature_policy: str = typer.Option("conservative_branch_site", "--feature-policy"),
    truth_mode: str = typer.Option("explicit", "--truth-mode"),
    methods: str = typer.Option("identity,mafft,babappalign,muscle", "--methods"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Package retained 100K conservative branch-site model artifacts."""
    try:
        summary = package_deployable_model(
            DeployableModelPackageConfig(
                run_name=run_name,
                model_dirs=model_dirs,
                calibration_dirs=calibration_dirs,
                truth_audit_dir=str(truth_audit_dir),
                validation_report=str(validation_report),
                feature_policy=feature_policy,
                truth_mode=truth_mode,
                methods=methods,
                outdir=str(outdir),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not package deployable model: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Deployable Model Package",
        summary,
        ["status", "outdir", "manifest", "model_card", "feature_schema", "calibration_schema", "checksums"],
    )
    if summary.get("status") == "blocked":
        _print_warnings(summary.get("blockers", []))
        raise typer.Exit(code=1)


@app.command("validate-deployable-model-package")
def validate_deployable_model_package_command(
    package_dir: Path = typer.Option(..., "--package-dir"),
) -> None:
    """Validate a deployable BABAPPA model package."""
    try:
        summary = validate_deployable_model_package(
            DeployableModelPackageValidationConfig(package_dir=str(package_dir))
        )
    except OSError as exc:
        console.print(f"Error: could not validate deployable model package: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_validation_table("BABAPPA Deployable Model Package Validation", summary)
    if summary.get("status") == "fail":
        raise typer.Exit(code=1)


@app.command("smoke-load-deployable-model")
def smoke_load_deployable_model_command(
    package_dir: Path = typer.Option(..., "--package-dir"),
    device: str = typer.Option("auto", "--device"),
    outdir: Path = typer.Option("deployable_model_load_smoke", "--outdir"),
) -> None:
    """Smoke-load a deployable model package without empirical inference."""
    try:
        summary = smoke_load_deployable_model(
            DeployableModelSmokeConfig(
                package_dir=str(package_dir),
                device=device,
                outdir=str(outdir),
            )
        )
    except OSError as exc:
        console.print(f"Error: could not smoke-load deployable model: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Deployable Model Load Smoke",
        summary,
        ["status", "device", "forward_pass", "metadata_only", "n_fail", "n_warning"],
    )
    if summary.get("status") == "fail":
        raise typer.Exit(code=1)


@app.command("plan-simulation-matched-calibration")
def plan_simulation_matched_calibration_command(
    empirical_validation_dir: Path = typer.Option(..., "--empirical-validation-dir"),
    deployable_model_package: Path = typer.Option(..., "--deployable-model-package"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Plan simulation-matched empirical calibration without running simulations."""
    try:
        summary = plan_simulation_matched_calibration(
            SimulationMatchedCalibrationPlanConfig(
                empirical_validation_dir=str(empirical_validation_dir),
                deployable_model_package=str(deployable_model_package),
                outdir=str(outdir),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan simulation-matched calibration: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Simulation-Matched Calibration Plan",
        summary,
        ["status", "outdir", "json", "markdown", "commands", "heavy_jobs_executed"],
    )
    if summary.get("missing_inputs"):
        _print_warnings(summary["missing_inputs"])


@app.command("summarize-simulation-matched-calibration-plan")
def summarize_simulation_matched_calibration_plan_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Summarize a USER-RUN simulation-matched calibration plan."""
    try:
        summary = summarize_simulation_matched_calibration_plan(
            SimulationMatchedCalibrationSummaryConfig(
                plan_dir=str(plan_dir),
                outdir=str(outdir),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not summarize simulation-matched calibration plan: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Simulation-Matched Calibration Summary",
        summary,
        ["status", "outdir", "json"],
    )


@app.command("validate-empirical-input")
def validate_empirical_input_command(
    cds_fasta: Path = typer.Option(..., "--cds-fasta"),
    tree: Path = typer.Option(..., "--tree"),
    foreground: str = typer.Option(..., "--foreground"),
    outdir: Path = typer.Option(..., "--outdir"),
    allow_stop_codons: bool = typer.Option(False, "--allow-stop-codons"),
    min_taxa: int = typer.Option(3, "--min-taxa"),
    min_codons: int = typer.Option(3, "--min-codons"),
) -> None:
    """Validate empirical CDS FASTA/tree inputs and write QC summaries."""
    try:
        summary = validate_empirical_input(
            EmpiricalInputValidationConfig(
                cds_fasta=str(cds_fasta),
                tree=str(tree),
                foreground=foreground,
                outdir=str(outdir),
                allow_stop_codons=allow_stop_codons,
                min_taxa=min_taxa,
                min_codons=min_codons,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not validate empirical input: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Empirical Input Validation",
        summary,
        ["status", "outdir", "json", "tsv", "markdown", "n_taxa", "n_codons"],
    )
    if summary.get("status") == "fail":
        _print_warnings(summary.get("failures", []))
        raise typer.Exit(code=1)
    if summary.get("warnings"):
        _print_warnings(summary["warnings"])


@app.command("run-empirical-alignment-ensemble")
def run_empirical_alignment_ensemble_command(
    cds_fasta: Path = typer.Option(..., "--cds-fasta"),
    tree: Path = typer.Option(..., "--tree"),
    foreground: str = typer.Option(..., "--foreground"),
    outdir: Path = typer.Option(..., "--outdir"),
    methods: str = typer.Option("identity,mafft,babappalign,muscle", "--methods"),
    require_babappalign: str = typer.Option("true", "--require-babappalign"),
    threads: int = typer.Option(4, "--threads"),
) -> None:
    """Run a tiny empirical alignment ensemble and build site maps/policy."""
    try:
        summary = run_empirical_alignment_ensemble(
            EmpiricalAlignmentEnsembleConfig(
                cds_fasta=str(cds_fasta),
                tree=str(tree),
                foreground=foreground,
                outdir=str(outdir),
                methods=methods,
                require_babappalign=_parse_bool(require_babappalign),
                threads=threads,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not run empirical alignment ensemble: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Empirical Alignment Ensemble",
        summary,
        ["status", "outdir", "methods_run", "manifest", "report"],
    )
    if summary.get("status") == "fail":
        _print_warnings(summary.get("failures", []))
        raise typer.Exit(code=1)
    if summary.get("warnings"):
        _print_warnings(summary["warnings"])


@app.command("extract-empirical-branch-site-features")
def extract_empirical_branch_site_features_command(
    empirical_validation_dir: Path = typer.Option(..., "--empirical-validation-dir"),
    alignment_dir: Path = typer.Option(..., "--alignment-dir"),
    deployable_model_package: Path = typer.Option(..., "--deployable-model-package"),
    outdir: Path = typer.Option(..., "--outdir"),
    foreground: str = typer.Option(..., "--foreground"),
) -> None:
    """Extract empirical branch-site features matching the deployable schema."""
    try:
        summary = extract_empirical_branch_site_features(
            EmpiricalFeatureExtractionConfig(
                empirical_validation_dir=str(empirical_validation_dir),
                alignment_dir=str(alignment_dir),
                deployable_model_package=str(deployable_model_package),
                outdir=str(outdir),
                foreground=foreground,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not extract empirical features: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Empirical Feature Extraction",
        summary,
        ["status", "outdir", "features", "rows", "schema_match"],
    )


@app.command("audit-empirical-features")
def audit_empirical_features_command(
    features: Path = typer.Option(..., "--features"),
    deployable_model_package: Path = typer.Option(..., "--deployable-model-package"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Audit empirical feature tables for forbidden truth-derived columns."""
    try:
        summary = audit_empirical_features(
            EmpiricalFeatureAuditConfig(
                features=str(features),
                deployable_model_package=str(deployable_model_package),
                outdir=str(outdir),
            )
        )
    except OSError as exc:
        console.print(f"Error: could not audit empirical features: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Empirical Feature Audit",
        summary,
        ["status", "outdir", "json", "forbidden_columns"],
    )
    if summary.get("status") == "fail":
        raise typer.Exit(code=1)


@app.command("empirical-applicability")
def empirical_applicability_command(
    empirical_validation_dir: Path = typer.Option(..., "--empirical-validation-dir"),
    empirical_feature_dir: Path = typer.Option(..., "--empirical-feature-dir"),
    deployable_model_package: Path = typer.Option(..., "--deployable-model-package"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Run rule-based empirical applicability/OOD gating."""
    try:
        summary = run_empirical_applicability(
            EmpiricalApplicabilityConfig(
                empirical_validation_dir=str(empirical_validation_dir),
                empirical_feature_dir=str(empirical_feature_dir),
                deployable_model_package=str(deployable_model_package),
                outdir=str(outdir),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not run empirical applicability: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Empirical Applicability",
        summary,
        ["status", "outdir", "json", "recommended_tier", "reasons"],
    )


@app.command("score-empirical-branch-sites")
def score_empirical_branch_sites_command(
    features: Path = typer.Option(..., "--features"),
    deployable_model_package: Path = typer.Option(..., "--deployable-model-package"),
    applicability_dir: Path = typer.Option(..., "--applicability-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Score empirical branch-site rows with the deployable model package."""
    try:
        summary = score_empirical_branch_sites(
            EmpiricalBranchSiteScoringConfig(
                features=str(features),
                deployable_model_package=str(deployable_model_package),
                applicability_dir=str(applicability_dir),
                outdir=str(outdir),
                device=device,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        console.print(f"Error: could not score empirical branch-sites: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Empirical Branch-Site Scoring",
        summary,
        ["status", "outdir", "device", "tier_model", "diagnostic_only", "n_rows"],
    )


@app.command("make-empirical-branch-site-report")
def make_empirical_branch_site_report_command(
    outdir: Path = typer.Option(..., "--outdir"),
    empirical_validation_dir: Path = typer.Option(..., "--empirical-validation-dir"),
    alignment_dir: Path = typer.Option(..., "--alignment-dir"),
    feature_dir: Path = typer.Option(..., "--feature-dir"),
    feature_audit_dir: Path = typer.Option(..., "--feature-audit-dir"),
    applicability_dir: Path = typer.Option(..., "--applicability-dir"),
    scoring_dir: Path = typer.Option(..., "--scoring-dir"),
    simulation_matched_calibration_plan: Path = typer.Option(..., "--simulation-matched-calibration-plan"),
    deployable_model_package: Path = typer.Option(..., "--deployable-model-package"),
) -> None:
    """Assemble a guarded empirical branch-site report."""
    try:
        summary = make_empirical_branch_site_report(
            EmpiricalBranchSiteReportConfig(
                outdir=str(outdir),
                empirical_validation_dir=str(empirical_validation_dir),
                alignment_dir=str(alignment_dir),
                feature_dir=str(feature_dir),
                feature_audit_dir=str(feature_audit_dir),
                applicability_dir=str(applicability_dir),
                scoring_dir=str(scoring_dir),
                simulation_matched_calibration_plan=str(simulation_matched_calibration_plan),
                deployable_model_package=str(deployable_model_package),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not make empirical report: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Empirical Branch-Site Report",
        summary,
        ["status", "outdir", "json", "markdown", "no_simulator_truth_used"],
    )


@app.command("plan-external-benchmark-panel")
def plan_external_benchmark_panel_command(
    panel_manifest: Path = typer.Option(..., "--panel-manifest"),
    deployable_model_package: Path = typer.Option(..., "--deployable-model-package"),
    outdir: Path = typer.Option(..., "--outdir"),
    methods: str = typer.Option("identity,mafft,babappalign,muscle", "--methods"),
    classical_tools: str = typer.Option("codeml,hyphy", "--classical-tools"),
) -> None:
    """Plan external codeml/HyPhy benchmark panel commands without executing them."""
    try:
        summary = plan_external_benchmark_panel(
            ExternalBenchmarkPanelPlanConfig(
                panel_manifest=str(panel_manifest),
                deployable_model_package=str(deployable_model_package),
                outdir=str(outdir),
                methods=methods,
                classical_tools=classical_tools,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan external benchmark panel: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA External Benchmark Panel Plan",
        summary,
        ["status", "outdir", "codeml_template", "hyphy_template", "heavy_jobs_executed"],
    )


@app.command("validate-empirical-pilot-panel")
def validate_empirical_pilot_panel_command(
    panel_manifest: Path = typer.Option(..., "--panel-manifest"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Validate a curated empirical pilot-panel manifest."""
    try:
        summary = validate_empirical_pilot_panel(
            EmpiricalPilotPanelValidationConfig(
                panel_manifest=str(panel_manifest),
                outdir=str(outdir),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not validate empirical pilot panel: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Empirical Pilot Panel Validation",
        summary,
        ["status", "outdir", "n_rows", "n_fail", "n_warning", "json", "markdown"],
    )
    if summary.get("status") == "fail":
        _print_warnings(summary.get("failures", []))
        raise typer.Exit(code=1)
    if summary.get("warnings"):
        _print_warnings(summary["warnings"])


@app.command("run-empirical-pilot-panel")
def run_empirical_pilot_panel_command(
    panel_manifest: Path = typer.Option(..., "--panel-manifest"),
    deployable_model_package: Path = typer.Option(..., "--deployable-model-package"),
    outdir: Path = typer.Option(..., "--outdir"),
    methods: str = typer.Option("identity,mafft,babappalign,muscle", "--methods"),
    device: str = typer.Option("auto", "--device"),
    max_families: int = typer.Option(5, "--max-families"),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
) -> None:
    """Run a small guarded empirical pilot panel."""
    try:
        summary = run_empirical_pilot_panel(
            EmpiricalPilotPanelRunConfig(
                panel_manifest=str(panel_manifest),
                deployable_model_package=str(deployable_model_package),
                outdir=str(outdir),
                methods=methods,
                device=device,
                max_families=max_families,
                fail_fast=fail_fast,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not run empirical pilot panel: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Empirical Pilot Panel Run",
        summary,
        ["status", "outdir", "families_processed", "qc_pass", "qc_fail", "scoring_ok", "report"],
    )
    if summary.get("status") == "fail":
        console.print("One or more pilot families failed; see per-family outputs for diagnostics.", style="yellow")


@app.command("plan-classical-reference-workflows")
def plan_classical_reference_workflows_command(
    panel_manifest: Path = typer.Option(..., "--panel-manifest"),
    outdir: Path = typer.Option(..., "--outdir"),
    tools: str = typer.Option("codeml,hyphy", "--tools"),
) -> None:
    """Generate codeml/HyPhy reference workflow templates without executing tools."""
    try:
        summary = plan_classical_reference_workflows(
            ClassicalReferenceWorkflowPlanConfig(
                panel_manifest=str(panel_manifest),
                outdir=str(outdir),
                tools=tools,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan classical reference workflows: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Classical Reference Workflow Plan",
        summary,
        ["status", "outdir", "tools", "codeml_script", "hyphy_script", "executed"],
    )


@app.command("compare-empirical-reference-results")
def compare_empirical_reference_results_command(
    babappa_panel_run: Path = typer.Option(..., "--babappa-panel-run"),
    reference_results: Path = typer.Option(..., "--reference-results"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Compare BABAPPA pilot diagnostics to reference result summaries."""
    try:
        summary = compare_empirical_reference_results(
            EmpiricalReferenceComparisonConfig(
                babappa_panel_run=str(babappa_panel_run),
                reference_results=str(reference_results),
                outdir=str(outdir),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not compare empirical reference results: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Empirical Reference Comparison",
        summary,
        ["status", "outdir", "n_rows", "concordance_classes", "json"],
    )
    if summary.get("status") == "fail":
        raise typer.Exit(code=1)


@app.command("summarize-empirical-pilot-panel")
def summarize_empirical_pilot_panel_command(
    panel_run: Path = typer.Option(..., "--panel-run"),
    outdir: Path = typer.Option(..., "--outdir"),
    reference_comparison: Optional[Path] = typer.Option(None, "--reference-comparison"),
) -> None:
    """Summarize a guarded empirical pilot panel."""
    try:
        summary = summarize_empirical_pilot_panel(
            EmpiricalPilotPanelSummaryConfig(
                panel_run=str(panel_run),
                outdir=str(outdir),
                reference_comparison=str(reference_comparison) if reference_comparison else None,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not summarize empirical pilot panel: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Empirical Pilot Panel Summary",
        summary,
        ["status", "outdir", "n_families", "claim_boundary_present", "json", "markdown"],
    )


@app.command("validate-empirical-pilot-summary")
def validate_empirical_pilot_summary_command(
    summary_dir: Path = typer.Option(..., "--summary-dir"),
) -> None:
    """Validate empirical pilot summary claim-boundary language."""
    try:
        summary = validate_empirical_pilot_summary(
            EmpiricalPilotSummaryValidationConfig(summary_dir=str(summary_dir))
        )
    except OSError as exc:
        console.print(f"Error: could not validate empirical pilot summary: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_validation_table("BABAPPA Empirical Pilot Summary Validation", summary)
    if summary.get("status") == "fail":
        raise typer.Exit(code=1)


@app.command("prepare-real-empirical-pilot-workspace")
def prepare_real_empirical_pilot_workspace_command(
    workspace: Path = typer.Option("real_empirical_pilot", "--workspace"),
    max_families: int = typer.Option(12, "--max-families"),
) -> None:
    """Create the guarded real empirical pilot workspace and manifest template."""
    try:
        summary = prepare_real_empirical_pilot_workspace(
            RealEmpiricalPilotWorkspaceConfig(
                workspace=str(workspace),
                max_families=max_families,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not prepare real empirical pilot workspace: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Real Empirical Pilot Workspace",
        summary,
        ["status", "workspace", "manifest", "manifest_created", "families", "validation_status", "readiness_report"],
    )
    if summary.get("missing_inputs"):
        _print_warnings([f"missing_inputs:{len(summary['missing_inputs'])}"])


@app.command("make-real-empirical-pilot-decision-report")
def make_real_empirical_pilot_decision_report_command(
    workspace: Path = typer.Option("real_empirical_pilot", "--workspace"),
    outdir: Optional[Path] = typer.Option(None, "--outdir"),
) -> None:
    """Create a guarded real empirical pilot decision report."""
    try:
        summary = make_real_empirical_pilot_decision_report(
            RealEmpiricalPilotDecisionReportConfig(
                workspace=str(workspace),
                outdir=str(outdir) if outdir else None,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not make real empirical pilot decision report: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Real Empirical Pilot Decision Report",
        summary,
        ["status", "outdir", "decision", "not_ready_for_claims", "reference_comparison_status", "markdown"],
    )


@app.command("prepare-real-pilot-inputs")
def prepare_real_pilot_inputs_command(
    workspace: Path = typer.Option("real_empirical_pilot", "--workspace"),
    manifest: str = typer.Option("real_empirical_pilot_panel.tsv", "--manifest"),
    outdir: Path = typer.Option("real_empirical_pilot/input_staging", "--outdir"),
) -> None:
    """Inventory real pilot inputs and suggest canonical FASTA/tree paths."""
    try:
        summary = prepare_real_pilot_inputs(
            RealPilotInputStagingConfig(
                workspace=str(workspace),
                manifest=manifest,
                outdir=str(outdir),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not prepare real pilot inputs: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Real Pilot Input Staging",
        summary,
        ["status", "workspace", "families", "missing_inputs", "ready_to_run", "suggested_paths"],
    )


@app.command("import-real-pilot-family")
def import_real_pilot_family_command(
    workspace: Path = typer.Option("real_empirical_pilot", "--workspace"),
    panel_id: str = typer.Option(..., "--panel-id"),
    gene_family: str = typer.Option(..., "--gene-family"),
    species_group: str = typer.Option(..., "--species-group"),
    cds_fasta: Path = typer.Option(..., "--cds-fasta"),
    tree_file: Path = typer.Option(..., "--tree-file"),
    foreground: str = typer.Option(..., "--foreground"),
    expected_category: str = typer.Option(..., "--expected-category"),
    reference_status: str = typer.Option("planned", "--reference-status"),
    notes: str = typer.Option("", "--notes"),
) -> None:
    """Import one user-supplied CDS FASTA/tree pair into the real pilot workspace."""
    try:
        summary = import_real_pilot_family(
            RealPilotFamilyImportConfig(
                workspace=str(workspace),
                panel_id=panel_id,
                gene_family=gene_family,
                species_group=species_group,
                cds_fasta=str(cds_fasta),
                tree_file=str(tree_file),
                foreground=foreground,
                expected_category=expected_category,
                reference_status=reference_status,
                notes=notes,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not import real pilot family: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Real Pilot Family Import",
        summary,
        ["status", "panel_id", "cds_fasta", "tree_file", "manifest", "report"],
    )


@app.command("import-real-pilot-batch")
def import_real_pilot_batch_command(
    workspace: Path = typer.Option("real_empirical_pilot", "--workspace"),
    batch_manifest: Path = typer.Option(..., "--batch-manifest"),
) -> None:
    """Import a batch of user-supplied real pilot families."""
    try:
        summary = import_real_pilot_batch(
            RealPilotBatchImportConfig(
                workspace=str(workspace),
                batch_manifest=str(batch_manifest),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not import real pilot batch: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Real Pilot Batch Import",
        summary,
        ["status", "workspace", "n_imported", "manifest", "report"],
    )


@app.command("plan-real-pilot-tree-building")
def plan_real_pilot_tree_building_command(
    workspace: Path = typer.Option("real_empirical_pilot", "--workspace"),
    manifest: str = typer.Option("real_empirical_pilot_panel.tsv", "--manifest"),
    outdir: Path = typer.Option("real_empirical_pilot/tree_building_plan", "--outdir"),
    method: str = typer.Option("iqtree", "--method"),
) -> None:
    """Plan USER-RUN tree building for real pilot families missing trees."""
    try:
        summary = plan_real_pilot_tree_building(
            RealPilotTreeBuildingPlanConfig(
                workspace=str(workspace),
                manifest=manifest,
                outdir=str(outdir),
                method=method,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan real pilot tree building: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Real Pilot Tree Building Plan",
        summary,
        ["status", "outdir", "n_trees_to_build", "script", "executed"],
    )


@app.command("sanitize-cds-fasta")
def sanitize_cds_fasta_command(
    input: Path = typer.Option(..., "--input"),
    output: Path = typer.Option(..., "--output"),
    report: Path = typer.Option(..., "--report"),
    mode: str = typer.Option("strict", "--mode"),
) -> None:
    """Sanitize and QC a CDS FASTA before real pilot import."""
    try:
        summary = sanitize_cds_fasta(
            CdsFastaSanitizeConfig(
                input=str(input),
                output=str(output),
                report=str(report),
                mode=mode,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not sanitize CDS FASTA: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA CDS FASTA Sanitation",
        summary,
        ["status", "output", "report", "n_fail", "n_warning", "output_written"],
    )
    if summary.get("status") == "fail":
        raise typer.Exit(code=1)


@app.command("list-foreground-candidates")
def list_foreground_candidates_command(
    cds_fasta: Path = typer.Option(..., "--cds-fasta"),
    tree_file: Path = typer.Option(..., "--tree-file"),
    outdir: Path = typer.Option(..., "--outdir"),
    foreground: Optional[str] = typer.Option(None, "--foreground"),
) -> None:
    """List matching FASTA/tree taxa for choosing a foreground label."""
    try:
        summary = list_foreground_candidates(
            ForegroundCandidateConfig(
                cds_fasta=str(cds_fasta),
                tree_file=str(tree_file),
                outdir=str(outdir),
                foreground=foreground,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not list foreground candidates: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Foreground Candidates",
        summary,
        ["status", "outdir", "matching_tips", "foreground_valid", "json"],
    )


@app.command("validate-real-pilot-readiness")
def validate_real_pilot_readiness_command(
    workspace: Path = typer.Option("real_empirical_pilot", "--workspace"),
    manifest: str = typer.Option("real_empirical_pilot_panel.tsv", "--manifest"),
    outdir: Path = typer.Option("real_empirical_pilot/readiness", "--outdir"),
) -> None:
    """Gate the real empirical pilot before running BABAPPA scoring."""
    try:
        summary = validate_real_pilot_readiness(
            RealPilotReadinessConfig(
                workspace=str(workspace),
                manifest=manifest,
                outdir=str(outdir),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not validate real pilot readiness: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Real Pilot Readiness",
        summary,
        ["status", "ready_to_run", "total_families", "files_missing", "foreground_invalid", "tree_incompatible", "json"],
    )


@app.command("discover-local-pilot-files")
def discover_local_pilot_files_command(
    search_dir: Path = typer.Option(..., "--search-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Discover local FASTA/tree files and suggest candidate pairs."""
    try:
        summary = discover_local_pilot_files(
            LocalPilotFileDiscoveryConfig(
                search_dir=str(search_dir),
                outdir=str(outdir),
            )
        )
    except OSError as exc:
        console.print(f"Error: could not discover local pilot files: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Local Pilot File Discovery",
        summary,
        ["status", "outdir", "n_fasta", "n_tree", "n_pair_suggestions", "manifest_modified"],
    )


@app.command("prefilter-empirical-family")
def prefilter_empirical_family_command(
    cds_fasta: Path = typer.Option(..., "--cds-fasta"),
    tree_file: Path = typer.Option(..., "--tree-file"),
    foreground: str = typer.Option(..., "--foreground"),
    outdir: Path = typer.Option(..., "--outdir"),
    max_mean_pdistance: float = typer.Option(0.35, "--max-mean-pdistance"),
    min_taxa: int = typer.Option(6, "--min-taxa"),
    min_codons: int = typer.Option(100, "--min-codons"),
) -> None:
    """Screen an empirical family for divergence, tree compatibility, and OOD risk."""
    try:
        summary = prefilter_empirical_family(
            EmpiricalFamilyPrefilterConfig(
                cds_fasta=str(cds_fasta),
                tree_file=str(tree_file),
                foreground=foreground,
                outdir=str(outdir),
                max_mean_pdistance=max_mean_pdistance,
                min_taxa=min_taxa,
                min_codons=min_codons,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not prefilter empirical family: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Empirical Family Prefilter",
        summary,
        ["status", "decision", "n_taxa", "n_codons", "mean_pdistance", "recommended_action", "json"],
    )


@app.command("plan-empirical-family-acquisition")
def plan_empirical_family_acquisition_command(
    family_id: str = typer.Option(..., "--family-id"),
    query_species: str = typer.Option(..., "--query-species"),
    query_gene_or_locus: str = typer.Option(..., "--query-gene-or-locus"),
    target_taxa_file: Path = typer.Option(..., "--target-taxa-file"),
    outdir: Path = typer.Option(..., "--outdir"),
    source: str = typer.Option("ensembl_plants", "--source"),
    strategy: str = typer.Option("blastp_best_hit", "--strategy"),
) -> None:
    """Plan target-taxon family acquisition scripts without executing downloads."""
    try:
        summary = plan_empirical_family_acquisition(
            EmpiricalFamilyAcquisitionPlanConfig(
                family_id=family_id,
                query_species=query_species,
                query_gene_or_locus=query_gene_or_locus,
                target_taxa_file=str(target_taxa_file),
                outdir=str(outdir),
                source=source,
                strategy=strategy,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan empirical family acquisition: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Empirical Family Acquisition Plan",
        summary,
        ["status", "outdir", "scripts", "executed"],
    )


@app.command("recommend-target-taxa")
def recommend_target_taxa_command(
    pilot_type: str = typer.Option("plant_close", "--pilot-type"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Recommend a close or moderate target-taxon template for empirical pilots."""
    try:
        summary = recommend_target_taxa(
            TargetTaxaRecommendationConfig(
                pilot_type=pilot_type,
                outdir=str(outdir),
            )
        )
    except OSError as exc:
        console.print(f"Error: could not recommend target taxa: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Target Taxa Recommendation",
        summary,
        ["status", "pilot_type", "outdir", "n_taxa", "recommendation"],
    )


@app.command("plan-ood-aware-family-build")
def plan_ood_aware_family_build_command(
    family_id: str = typer.Option(..., "--family-id"),
    query_species: str = typer.Option(..., "--query-species"),
    query_gene_or_locus: str = typer.Option(..., "--query-gene-or-locus"),
    target_taxa_file: Path = typer.Option(..., "--target-taxa-file"),
    outdir: Path = typer.Option(..., "--outdir"),
    max_mean_pdistance: float = typer.Option(0.35, "--max-mean-pdistance"),
    min_taxa: int = typer.Option(6, "--min-taxa"),
    min_codons: int = typer.Option(100, "--min-codons"),
) -> None:
    """Plan an OOD-gated family build workflow without executing it."""
    try:
        summary = plan_ood_aware_family_build(
            OODAwareFamilyBuildPlanConfig(
                family_id=family_id,
                query_species=query_species,
                query_gene_or_locus=query_gene_or_locus,
                target_taxa_file=str(target_taxa_file),
                outdir=str(outdir),
                max_mean_pdistance=max_mean_pdistance,
                min_taxa=min_taxa,
                min_codons=min_codons,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan OOD-aware family build: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA OOD-Aware Family Build Plan",
        summary,
        ["status", "family_id", "max_mean_pdistance", "script", "executed"],
    )


@app.command("add-prefiltered-family-to-pilot")
def add_prefiltered_family_to_pilot_command(
    workspace: Path = typer.Option("real_empirical_pilot", "--workspace"),
    prefilter_dir: Path = typer.Option(..., "--prefilter-dir"),
    panel_id: str = typer.Option(..., "--panel-id"),
    expected_category: str = typer.Option(..., "--expected-category"),
    reference_status: str = typer.Option("planned", "--reference-status"),
    allow_diagnostic_only: bool = typer.Option(False, "--allow-diagnostic-only"),
) -> None:
    """Import a prefiltered family into the real pilot manifest only if it passes the OOD gate."""
    try:
        summary = add_prefiltered_family_to_pilot(
            AddPrefilteredFamilyConfig(
                workspace=str(workspace),
                prefilter_dir=str(prefilter_dir),
                panel_id=panel_id,
                expected_category=expected_category,
                reference_status=reference_status,
                allow_diagnostic_only=allow_diagnostic_only,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not add prefiltered family: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Add Prefiltered Family",
        summary,
        ["status", "decision", "panel_id", "report"],
    )


@app.command("summarize-empirical-ood")
def summarize_empirical_ood_command(
    workspace: Path = typer.Option("real_empirical_pilot", "--workspace"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Summarize empirical prefilter, applicability, and diagnostic-only OOD status."""
    try:
        summary = summarize_empirical_ood(
            EmpiricalOODSummaryConfig(
                workspace=str(workspace),
                outdir=str(outdir),
            )
        )
    except OSError as exc:
        console.print(f"Error: could not summarize empirical OOD status: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Empirical OOD Summary",
        summary,
        ["status", "outdir", "n_families", "json"],
    )


@app.command("plan-empirical-scoring")
def plan_empirical_scoring_command(
    cds_fasta: Path = typer.Option(..., "--cds-fasta"),
    tree: Path = typer.Option(..., "--tree"),
    foreground: str = typer.Option(..., "--foreground"),
    deployable_model_package: Path = typer.Option(..., "--deployable-model-package"),
    outdir: Path = typer.Option(..., "--outdir"),
    methods: str = typer.Option("identity,mafft,babappalign,muscle", "--methods"),
    device: str = typer.Option("auto", "--device"),
    allow_diagnostic_out_of_domain: bool = typer.Option(False, "--allow-diagnostic-out-of-domain"),
) -> None:
    """Plan empirical scoring scripts without running empirical prediction."""
    try:
        summary = plan_empirical_scoring(
            EmpiricalScoringPlanConfig(
                cds_fasta=str(cds_fasta),
                tree=str(tree),
                foreground=foreground,
                deployable_model_package=str(deployable_model_package),
                outdir=str(outdir),
                methods=methods,
                device=device,
                allow_diagnostic_out_of_domain=allow_diagnostic_out_of_domain,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan empirical scoring: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Empirical Scoring Plan",
        summary,
        ["status", "outdir", "run_script", "validate_script", "summarize_script", "current_stopping_point"],
    )


@app.command("freeze-empirical-evidence-pack")
def freeze_empirical_evidence_pack_command(
    family_id: str = typer.Option(..., "--family-id"),
    outdir: Path = typer.Option(..., "--outdir"),
    cds_fasta: Path = typer.Option(..., "--cds-fasta"),
    tree_file: Path = typer.Option(..., "--tree-file"),
    foreground: str = typer.Option(..., "--foreground"),
    babappa_family_dir: Path = typer.Option(..., "--babappa-family-dir"),
    panel_run_summary: Path = typer.Option(..., "--panel-run-summary"),
    prefilter_dir: Path = typer.Option(..., "--prefilter-dir"),
    summary_report: Optional[Path] = typer.Option(None, "--summary-report"),
    reference_plan_dir: Optional[Path] = typer.Option(None, "--reference-plan-dir"),
    calibration_plan_dir: Optional[Path] = typer.Option(None, "--calibration-plan-dir"),
) -> None:
    """Freeze a small empirical evidence pack without raw simulator truth."""
    try:
        summary = freeze_empirical_evidence_pack(
            EmpiricalEvidencePackConfig(
                family_id=family_id,
                outdir=str(outdir),
                cds_fasta=str(cds_fasta),
                tree_file=str(tree_file),
                foreground=foreground,
                babappa_family_dir=str(babappa_family_dir),
                panel_run_summary=str(panel_run_summary),
                prefilter_dir=str(prefilter_dir),
                summary_report=str(summary_report) if summary_report else "",
                reference_plan_dir=str(reference_plan_dir) if reference_plan_dir else "",
                calibration_plan_dir=str(calibration_plan_dir) if calibration_plan_dir else "",
            )
        )
    except OSError as exc:
        console.print(f"Error: could not freeze empirical evidence pack: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Empirical Evidence Pack",
        summary,
        ["status", "path", "n_files", "checksums"],
    )


@app.command("validate-empirical-evidence-pack")
def validate_empirical_evidence_pack_command(
    evidence_pack: Path = typer.Option(..., "--evidence-pack"),
) -> None:
    """Validate a frozen empirical evidence pack and its claim boundary."""
    try:
        summary = validate_empirical_evidence_pack(
            EmpiricalEvidencePackValidationConfig(evidence_pack=str(evidence_pack))
        )
    except OSError as exc:
        console.print(f"Error: could not validate empirical evidence pack: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Empirical Evidence Pack Validation",
        summary,
        ["status", "json", "markdown"],
    )
    if summary.get("status") == "fail":
        _print_warnings(summary.get("failures", []))
        raise typer.Exit(code=1)


@app.command("check-reference-tools")
def check_reference_tools_command(
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Check codeml/HyPhy availability without requiring execution."""
    try:
        summary = check_reference_tools(ReferenceToolCheckConfig(outdir=str(outdir)))
    except OSError as exc:
        console.print(f"Error: could not check reference tools: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Reference Tool Check",
        summary,
        ["status", "outdir", "codeml", "hyphy"],
    )


@app.command("install-reference-tools-plan")
def install_reference_tools_plan_command(
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Generate USER-RUN conda/brew helper scripts for codeml/PAML and HyPhy."""
    try:
        summary = install_reference_tools_plan(
            ReferenceToolsInstallPlanConfig(outdir=str(outdir))
        )
    except OSError as exc:
        console.print(f"Error: could not plan reference tool installation: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Reference Tool Install Plan",
        summary,
        ["status", "outdir", "conda_script", "brew_script", "executed"],
    )


@app.command("prepare-codeml-reference")
def prepare_codeml_reference_command(
    cds_fasta: Path = typer.Option(..., "--cds-fasta"),
    tree: Path = typer.Option(..., "--tree"),
    foreground: str = typer.Option(..., "--foreground"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Prepare USER-RUN codeml branch-site reference templates."""
    try:
        summary = prepare_codeml_reference(
            CodemlReferencePrepConfig(
                cds_fasta=str(cds_fasta),
                tree=str(tree),
                foreground=foreground,
                outdir=str(outdir),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not prepare codeml reference: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA codeml Reference Preparation",
        summary,
        ["status", "outdir", "modelA", "null", "executed"],
    )


@app.command("prepare-hyphy-reference")
def prepare_hyphy_reference_command(
    cds_fasta: Path = typer.Option(..., "--cds-fasta"),
    tree: Path = typer.Option(..., "--tree"),
    foreground: str = typer.Option(..., "--foreground"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Prepare USER-RUN HyPhy aBSREL/MEME reference templates."""
    try:
        summary = prepare_hyphy_reference(
            HyphyReferencePrepConfig(
                cds_fasta=str(cds_fasta),
                tree=str(tree),
                foreground=foreground,
                outdir=str(outdir),
            )
        )
    except OSError as exc:
        console.print(f"Error: could not prepare HyPhy reference: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA HyPhy Reference Preparation",
        summary,
        ["status", "outdir", "absrel", "meme", "executed"],
    )


@app.command("parse-codeml-reference")
def parse_codeml_reference_command(
    codeml_dir: Path = typer.Option(..., "--codeml-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Parse codeml outputs if present; otherwise report pending."""
    try:
        summary = parse_codeml_reference(
            CodemlReferenceParseConfig(codeml_dir=str(codeml_dir), outdir=str(outdir))
        )
    except OSError as exc:
        console.print(f"Error: could not parse codeml reference: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA codeml Reference Parse",
        summary,
        ["status", "outdir"],
    )


@app.command("parse-hyphy-reference")
def parse_hyphy_reference_command(
    hyphy_dir: Path = typer.Option(..., "--hyphy-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Parse HyPhy outputs if present; otherwise report pending."""
    try:
        summary = parse_hyphy_reference(
            HyphyReferenceParseConfig(hyphy_dir=str(hyphy_dir), outdir=str(outdir))
        )
    except OSError as exc:
        console.print(f"Error: could not parse HyPhy reference: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA HyPhy Reference Parse",
        summary,
        ["status", "outdir"],
    )


@app.command("write-reference-results-template")
def write_reference_results_template_command(
    family_id: str = typer.Option(..., "--family-id"),
    foreground: str = typer.Option(..., "--foreground"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Write a pending codeml/HyPhy reference-results TSV template."""
    try:
        summary = write_reference_results_template(
            ReferenceResultsTemplateConfig(
                family_id=family_id,
                foreground=foreground,
                outdir=str(outdir),
            )
        )
    except OSError as exc:
        console.print(f"Error: could not write reference-results template: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Reference Results Template",
        summary,
        ["status", "path", "rows"],
    )


@app.command("build-reference-results-table")
def build_reference_results_table_command(
    panel_id: str = typer.Option(..., "--panel-id"),
    codeml_parsed: Path = typer.Option(..., "--codeml-parsed"),
    hyphy_parsed: Path = typer.Option(..., "--hyphy-parsed"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Build the reference_results.tsv table from parsed codeml/HyPhy outputs."""
    try:
        summary = build_reference_results_table(
            ReferenceResultsTableConfig(
                panel_id=panel_id,
                codeml_parsed=str(codeml_parsed),
                hyphy_parsed=str(hyphy_parsed),
                outdir=str(outdir),
            )
        )
    except OSError as exc:
        console.print(f"Error: could not build reference-results table: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Reference Results Table",
        summary,
        ["status", "path", "json"],
    )


@app.command("run-simulation-matched-null-calibration")
def run_simulation_matched_null_calibration_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
    deployable_model_package: Path = typer.Option(..., "--deployable-model-package"),
    outdir: Path = typer.Option(..., "--outdir"),
    n_replicates: int = typer.Option(100, "--n-replicates"),
    device: str = typer.Option("auto", "--device"),
    seed: int = typer.Option(42, "--seed"),
    fast_null_mode: bool = typer.Option(False, "--fast-null-mode"),
) -> None:
    """Run the safe staged one-family simulation-matched null calibration pilot."""
    try:
        summary = run_simulation_matched_null_calibration(
            SimulationMatchedNullCalibrationConfig(
                plan_dir=str(plan_dir),
                deployable_model_package=str(deployable_model_package),
                outdir=str(outdir),
                n_replicates=n_replicates,
                device=device,
                seed=seed,
                fast_null_mode=fast_null_mode,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not run matched-null calibration: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Simulation-Matched Null Calibration",
        summary,
        ["status", "outdir", "n_replicates_requested", "n_replicates_completed", "observed_max_gene_support", "observed_called_rows"],
    )


@app.command("validate-simulation-matched-null-calibration")
def validate_simulation_matched_null_calibration_command(
    calibration_dir: Path = typer.Option(..., "--calibration-dir"),
) -> None:
    """Validate staged or completed matched-null calibration outputs."""
    try:
        summary = validate_simulation_matched_null_calibration(
            SimulationMatchedNullCalibrationValidationConfig(calibration_dir=str(calibration_dir))
        )
    except OSError as exc:
        console.print(f"Error: could not validate matched-null calibration: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Matched Null Calibration Validation",
        summary,
        ["status", "json"],
    )
    if summary.get("warnings"):
        _print_warnings(summary["warnings"])
    if summary.get("status") == "fail":
        _print_warnings(summary.get("failures", []))
        raise typer.Exit(code=1)


@app.command("write-wrky-matched-null-script")
def write_wrky_matched_null_script_command(
    plan_dir: Path = typer.Option(..., "--plan-dir"),
    output_root: Path = typer.Option(..., "--output-root"),
) -> None:
    """Write the USER-RUN small matched-null calibration scaffold script."""
    try:
        path = write_wrky_matched_null_script(str(plan_dir), str(output_root))
    except OSError as exc:
        console.print(f"Error: could not write matched-null script: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA WRKY Matched Null Script",
        {"status": "written", "script": str(path), "executed": False},
        ["status", "script", "executed"],
    )


@app.command("make-wrky-interpretation-status")
def make_wrky_interpretation_status_command(
    family_id: str = typer.Option(..., "--family-id"),
    babappa_panel_run: Path = typer.Option(..., "--babappa-panel-run"),
    evidence_pack: Path = typer.Option(..., "--evidence-pack"),
    calibration_summary: Path = typer.Option(..., "--calibration-summary"),
    reference_results: Path = typer.Option(..., "--reference-results"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Create the guarded WRKY interpretation-status report."""
    try:
        summary = make_wrky_interpretation_status(
            WRKYInterpretationStatusConfig(
                family_id=family_id,
                babappa_panel_run=str(babappa_panel_run),
                evidence_pack=str(evidence_pack),
                calibration_summary=str(calibration_summary),
                reference_results=str(reference_results),
                outdir=str(outdir),
            )
        )
    except OSError as exc:
        console.print(f"Error: could not make WRKY interpretation status: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA WRKY Interpretation Status",
        summary,
        ["status", "decision", "manuscript_ready", "json"],
    )


@app.command("make-wrky-reference-calibration-report")
def make_wrky_reference_calibration_report_command(
    evidence_pack: Path = typer.Option(..., "--evidence-pack"),
    babappa_panel_run: Path = typer.Option(..., "--babappa-panel-run"),
    reference_results: Path = typer.Option(..., "--reference-results"),
    comparison_dir: Path = typer.Option(..., "--comparison-dir"),
    matched_null_calibration: Path = typer.Option(..., "--matched-null-calibration"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Create the integrated WRKY reference/calibration interpretation report."""
    try:
        summary = make_wrky_reference_calibration_report(
            WRKYReferenceCalibrationReportConfig(
                evidence_pack=str(evidence_pack),
                babappa_panel_run=str(babappa_panel_run),
                reference_results=str(reference_results),
                comparison_dir=str(comparison_dir),
                matched_null_calibration=str(matched_null_calibration),
                outdir=str(outdir),
            )
        )
    except OSError as exc:
        console.print(f"Error: could not make WRKY reference/calibration report: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA WRKY Reference/Calibration Report",
        summary,
        ["status", "decision_category", "manuscript_ready", "json"],
    )


@app.command("interpret-babappa-only-signal")
def interpret_babappa_only_signal_command(
    babappa_report: Path = typer.Option(..., "--babappa-report"),
    matched_null: Path = typer.Option(..., "--matched-null"),
    reference_results: Path = typer.Option(..., "--reference-results"),
    outdir: Path = typer.Option(..., "--outdir"),
) -> None:
    """Interpret a BABAPPA-only signal after reference and null-calibration results."""
    try:
        summary = interpret_babappa_only_signal(
            BabappaOnlySignalInterpretationConfig(
                babappa_report=str(babappa_report),
                matched_null=str(matched_null),
                reference_results=str(reference_results),
                outdir=str(outdir),
            )
        )
    except OSError as exc:
        console.print(f"Error: could not interpret BABAPPA-only signal: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA-Only Signal Interpretation",
        summary,
        ["status", "decision", "manuscript_ready", "json"],
    )


@app.command("audit-babappa-only-result")
def audit_babappa_only_result_command(
    family: str = typer.Option(..., "--family"),
    babappa_run: Path = typer.Option(..., "--babappa-run"),
    reference_results: Path = typer.Option(..., "--reference-results"),
    outdir: Path = typer.Option(..., "--outdir"),
    matched_null: Optional[Path] = typer.Option(None, "--matched-null"),
) -> None:
    """Audit a BABAPPA-only diagnostic signal for concentration and artifact risks."""
    try:
        summary = audit_babappa_only_result(
            BabappaOnlyResultAuditConfig(
                family=family,
                babappa_run=str(babappa_run),
                reference_results=str(reference_results),
                outdir=str(outdir),
                matched_null=str(matched_null) if matched_null else "",
            )
        )
    except OSError as exc:
        console.print(f"Error: could not audit BABAPPA-only result: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA-Only Result Audit",
        summary,
        ["status", "outdir", "json"],
    )
    if summary.get("warnings"):
        _print_warnings(summary["warnings"])


@app.command("plan-close-taxa-control-family")
def plan_close_taxa_control_family_command(
    control_id: str = typer.Option(..., "--control-id"),
    query_species: str = typer.Option(..., "--query-species"),
    query_gene_or_locus: str = typer.Option(..., "--query-gene-or-locus"),
    target_taxa_file: Path = typer.Option(..., "--target-taxa-file"),
    outdir: Path = typer.Option(..., "--outdir"),
    max_mean_pdistance: float = typer.Option(0.25, "--max-mean-pdistance"),
    min_taxa: int = typer.Option(6, "--min-taxa"),
    min_codons: int = typer.Option(100, "--min-codons"),
) -> None:
    """Plan a USER-RUN close-taxa conserved control family workflow."""
    try:
        summary = plan_close_taxa_control_family(
            CloseTaxaControlFamilyPlanConfig(
                control_id=control_id,
                query_species=query_species,
                query_gene_or_locus=query_gene_or_locus,
                target_taxa_file=str(target_taxa_file),
                outdir=str(outdir),
                max_mean_pdistance=max_mean_pdistance,
                min_taxa=min_taxa,
                min_codons=min_codons,
            )
        )
    except OSError as exc:
        console.print(f"Error: could not plan close-taxa control family: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table(
        "BABAPPA Close-Taxa Control Family Plan",
        summary,
        ["status", "control", "outdir", "scripts", "executed"],
    )


@app.command("compare-site-calibrations")
def compare_site_calibrations_command(
    calibration_dirs: str = typer.Option(..., "--calibration-dirs"),
    outdir: Path = typer.Option(..., "--outdir"),
    names: Optional[str] = typer.Option(None, "--names"),
) -> None:
    """Compare site calibration outputs."""
    try:
        summary = compare_site_calibrations(
            SiteCalibrationCompareConfig(
                calibration_dirs=_parse_csv(calibration_dirs),
                outdir=str(outdir),
                names=None if names is None else _parse_csv(names),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not compare site calibrations: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    _print_summary_table("BABAPPA Site Calibration Comparison", summary, ["outdir", "json", "recommendation"])


@app.command("validate-site-calibration-comparison")
def validate_site_calibration_comparison(
    compare_dir: Path = typer.Option(..., "--compare-dir")
) -> None:
    """Validate site calibration comparison artifacts."""
    summary = validate_site_calibration_comparison_dir(compare_dir)
    _print_validation_table("BABAPPA Site Calibration Comparison Validation", summary)
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("diagnose-predictions")
def diagnose_predictions_command(
    predictions: Path = typer.Option(
        ...,
        "--predictions",
        help="Prediction TSV to diagnose.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for prediction diagnostics.",
    ),
    metrics: Optional[Path] = typer.Option(
        None,
        "--metrics",
        help="Optional metrics JSON associated with the predictions.",
    ),
    calibration: Optional[Path] = typer.Option(
        None,
        "--calibration",
        help="Optional calibration JSON associated with the predictions.",
    ),
    probability_column: str = typer.Option(
        "prob_positive",
        "--probability-column",
        help="Prediction probability column to diagnose.",
    ),
    label_column: str = typer.Option(
        "gene_label",
        "--label-column",
        help="Ground-truth label column.",
    ),
    split_column: str = typer.Option(
        "split",
        "--split-column",
        help="Split column.",
    ),
    model_name: str = typer.Option(
        "model",
        "--model-name",
        help="Model name for diagnostics outputs.",
    ),
) -> None:
    """Diagnose prediction score distributions and threshold behavior."""
    try:
        config = PredictionDiagnosticsConfig(
            predictions_tsv=str(predictions),
            metrics_json=_optional_path_to_str(metrics),
            calibration_json=_optional_path_to_str(calibration),
            outdir=str(outdir),
            probability_column=probability_column,
            label_column=label_column,
            split_column=split_column,
            model_name=model_name,
        )
        summary = diagnose_predictions(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not diagnose predictions: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Prediction Diagnostics")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("JSON", summary["json"])
    table.add_row("Score summary", summary["score_summary"])
    table.add_row("Threshold curve", summary["threshold_curve"])
    table.add_row("Markdown", summary["markdown"])
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")


@app.command("validate-prediction-diagnostics")
def validate_prediction_diagnostics(
    diag_dir: Path = typer.Option(
        ...,
        "--diag-dir",
        help="Prediction diagnostics directory to validate.",
    )
) -> None:
    """Validate a BABAPPA prediction diagnostics directory."""
    summary = validate_prediction_diagnostics_dir(diag_dir)
    table = Table(title="BABAPPA Prediction Diagnostics Validation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("plan-large-run")
def plan_large_run_command(
    scale: int = typer.Option(..., "--scale"),
    families_per_tier: int = typer.Option(..., "--families-per-tier"),
    outdir: Path = typer.Option(..., "--outdir"),
    negative_downsample_ratio: float = typer.Option(..., "--negative-downsample-ratio"),
    methods: str = typer.Option(
        "identity,codon_dropout",
        "--methods",
        help="Comma-separated default internal methods for the validated large-run path.",
    ),
    external_methods: str = typer.Option(
        "",
        "--external-methods",
        help="Optional comma-separated external methods for external_aligner_run_commands.sh.",
    ),
    require_aligners: str = typer.Option(
        "false",
        "--require-aligners",
        help="Require external aligners in generated external script: true or false.",
    ),
    with_site_maps: str = typer.Option(
        "false",
        "--with-site-maps",
        help="Record whether site maps are expected in this large-run plan.",
    ),
) -> None:
    """Write command templates and expected outputs for a large validation run."""
    try:
        summary = plan_large_run(
            LargeRunPlanConfig(
                scale=scale,
                families_per_tier=families_per_tier,
                outdir=str(outdir),
                negative_downsample_ratio=negative_downsample_ratio,
                methods=_parse_methods(methods),
                external_methods=_parse_methods(external_methods),
                require_aligners=_parse_bool(require_aligners),
                with_site_maps=_parse_bool(with_site_maps),
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan large run: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    table = Table(title="BABAPPA Large-Run Plan")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Commands", summary["commands"])
    table.add_row("External aligner commands", summary["external_aligner_commands"])
    table.add_row("Expected outputs", summary["expected_outputs"])
    table.add_row("Expected raw site rows", str(summary["expected_raw_site_rows"]))
    table.add_row(
        "Approximate downsampled rows",
        str(summary["approximate_downsampled_site_rows"]),
    )
    console.print(table)


@app.command("validate-large-run-plan")
def validate_large_run_plan(
    plan_dir: Path = typer.Option(..., "--plan-dir")
) -> None:
    """Validate large-run planning artifacts without executing them."""
    summary = validate_large_run_plan_dir(plan_dir)
    _print_validation_table("BABAPPA Large-Run Plan Validation", summary)
    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command("plan-external-aligner-validation")
def plan_external_aligner_validation_command(
    panel_dir: Path = typer.Option(..., "--panel-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    methods: str = typer.Option(
        "identity,mafft,babappalign,muscle",
        "--methods",
        help="Comma-separated methods for mapped external-aligner validation.",
    ),
    optional_methods: str = typer.Option(
        "",
        "--optional-methods",
        help="Optional diagnostic methods to include only when explicitly requested.",
    ),
    exclude_methods: str = typer.Option(
        "prank",
        "--exclude-methods",
        help="Comma-separated methods to exclude from the generated fast workflow.",
    ),
    tiers: str = typer.Option(
        "low,moderate,high,extreme",
        "--tiers",
        help="Comma-separated saturation tiers to plan.",
    ),
    negative_downsample_ratio: float = typer.Option(10.0, "--negative-downsample-ratio"),
    max_method_failure_fraction: float = typer.Option(0.01, "--max-method-failure-fraction"),
    timeout_seconds: int = typer.Option(300, "--timeout-seconds"),
    conda_sh: str = typer.Option(
        "/home/rajamosai/miniconda3/etc/profile.d/conda.sh",
        "--conda-sh",
        help="Conda profile script sourced by the generated user-run shell script.",
    ),
    conda_env: str = typer.Option(
        "molevo",
        "--conda-env",
        help="Conda environment activated by the generated user-run shell script.",
    ),
) -> None:
    """Write user-run commands for clean mapped external-aligner validation."""
    try:
        summary = plan_external_aligner_validation(
            ExternalAlignerValidationPlanConfig(
                panel_dir=str(panel_dir),
                outdir=str(outdir),
                methods=_parse_methods(methods),
                optional_methods=_parse_methods(optional_methods),
                exclude_methods=_parse_methods(exclude_methods),
                tiers=_parse_methods(tiers),
                negative_downsample_ratio=negative_downsample_ratio,
                conda_sh=conda_sh,
                conda_env=conda_env,
                max_method_failure_fraction=max_method_failure_fraction,
                timeout_seconds=timeout_seconds,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(
            f"Error: could not plan external-aligner validation: {exc}", style="red"
        )
        raise typer.Exit(code=1) from exc
    table = Table(title="BABAPPA External-Aligner Validation Plan")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Methods", _format_methods(summary["methods"]))
    table.add_row("Optional methods", _format_methods(summary["optional_methods"]))
    table.add_row("Excluded methods", _format_methods(summary["exclude_methods"]))
    table.add_row("Tiers", _format_methods(summary["tiers"]))
    table.add_row("Commands", summary["commands"])
    table.add_row("Expected outputs", summary["expected_outputs"])
    console.print(table)


@app.command("plan-complete-external-tier-reports")
def plan_complete_external_tier_reports_command(
    tiers: str = typer.Option(..., "--tiers"),
    outdir: Path = typer.Option(..., "--outdir"),
    conda_sh: str = typer.Option(
        "/home/rajamosai/miniconda3/etc/profile.d/conda.sh",
        "--conda-sh",
    ),
    conda_env: str = typer.Option("molevo", "--conda-env"),
) -> None:
    """Write user-run commands to complete calibration/policies for existing external tiers."""
    try:
        summary = plan_complete_external_tier_reports(
            ExternalCompletedTierReportPlanConfig(
                tiers=_parse_methods(tiers),
                outdir=str(outdir),
                conda_sh=conda_sh,
                conda_env=conda_env,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan completed external tier reports: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    table = Table(title="BABAPPA Completed External Tier Report Plan")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Tiers", _format_methods(summary["tiers"]))
    table.add_row("Commands", summary["commands"])
    table.add_row("Expected outputs", summary["expected_outputs"])
    console.print(table)


@app.command("plan-external-extreme-recovery")
def plan_external_extreme_recovery_command(
    panel_dir: Path = typer.Option(..., "--panel-dir"),
    outdir: Path = typer.Option(..., "--outdir"),
    methods: str = typer.Option("identity,mafft,babappalign,muscle", "--methods"),
    negative_downsample_ratio: float = typer.Option(10.0, "--negative-downsample-ratio"),
    timeout_seconds: int = typer.Option(300, "--timeout-seconds"),
    max_method_failure_fraction: float = typer.Option(0.01, "--max-method-failure-fraction"),
    conda_sh: str = typer.Option(
        "/home/rajamosai/miniconda3/etc/profile.d/conda.sh",
        "--conda-sh",
    ),
    conda_env: str = typer.Option("molevo", "--conda-env"),
) -> None:
    """Write a user-run fast recovery plan for the missing external extreme tier."""
    try:
        summary = plan_external_extreme_recovery(
            ExternalExtremeRecoveryPlanConfig(
                panel_dir=str(panel_dir),
                outdir=str(outdir),
                methods=_parse_methods(methods),
                negative_downsample_ratio=negative_downsample_ratio,
                timeout_seconds=timeout_seconds,
                max_method_failure_fraction=max_method_failure_fraction,
                conda_sh=conda_sh,
                conda_env=conda_env,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan external extreme recovery: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    table = Table(title="BABAPPA External Extreme Recovery Plan")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Methods", _format_methods(summary["methods"]))
    table.add_row("Commands", summary["commands"])
    table.add_row("Expected outputs", summary["expected_outputs"])
    console.print(table)


@app.command("plan-fast-external-10k")
def plan_fast_external_10k_command(
    outdir: Path = typer.Option(..., "--outdir"),
    panel_outdir: Path = typer.Option(..., "--panel-outdir"),
    families_per_tier: int = typer.Option(2500, "--families-per-tier"),
    tiers: str = typer.Option("low,moderate,high,extreme", "--tiers"),
    methods: str = typer.Option("identity,mafft,babappalign,muscle", "--methods"),
    negative_downsample_ratio: float = typer.Option(10.0, "--negative-downsample-ratio"),
    conda_sh: str = typer.Option(
        "/home/rajamosai/miniconda3/etc/profile.d/conda.sh",
        "--conda-sh",
    ),
    conda_env: str = typer.Option("molevo", "--conda-env"),
    timeout_seconds: int = typer.Option(300, "--timeout-seconds"),
    max_method_failure_fraction: float = typer.Option(0.01, "--max-method-failure-fraction"),
) -> None:
    """Write a user-run 10K fast external-aligner validation plan."""
    try:
        summary = plan_fast_external_10k(
            FastExternal10kPlanConfig(
                outdir=str(outdir),
                panel_outdir=str(panel_outdir),
                families_per_tier=families_per_tier,
                tiers=_parse_methods(tiers),
                methods=_parse_methods(methods),
                negative_downsample_ratio=negative_downsample_ratio,
                conda_sh=conda_sh,
                conda_env=conda_env,
                timeout_seconds=timeout_seconds,
                max_method_failure_fraction=max_method_failure_fraction,
            )
        )
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not plan fast external 10K: {exc}", style="red")
        raise typer.Exit(code=1) from exc
    table = Table(title="BABAPPA Fast External 10K Plan")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("Methods", _format_methods(summary["methods"]))
    table.add_row("Tiers", _format_methods(summary["tiers"]))
    table.add_row("Run script", summary["run"])
    table.add_row("Monitor script", summary["monitor"])
    table.add_row("Validate script", summary["validate"])
    table.add_row("Summarize script", summary["summarize"])
    table.add_row("Expected outputs", summary["expected_outputs"])
    console.print(table)


@app.command("diagnose-neural")
def diagnose_neural_command(
    model_dir: Path = typer.Option(
        ...,
        "--model-dir",
        help="Neural model artifact directory to diagnose.",
    ),
    outdir: Path = typer.Option(
        ...,
        "--outdir",
        help="Output directory for neural diagnostics.",
    ),
    predictions: Optional[Path] = typer.Option(
        None,
        "--predictions",
        help="Optional prediction TSV override.",
    ),
    history: Optional[Path] = typer.Option(
        None,
        "--history",
        help="Optional training history TSV override.",
    ),
    metadata: Optional[Path] = typer.Option(
        None,
        "--metadata",
        help="Optional model metadata JSON override.",
    ),
    model_name: str = typer.Option(
        "neural_model",
        "--model-name",
        help="Model name for neural diagnostics outputs.",
    ),
) -> None:
    """Diagnose neural probability distributions, history, and metadata."""
    try:
        config = NeuralDiagnosticsConfig(
            model_dir=str(model_dir),
            outdir=str(outdir),
            predictions_tsv=_optional_path_to_str(predictions),
            history_tsv=_optional_path_to_str(history),
            metadata_json=_optional_path_to_str(metadata),
            model_name=model_name,
        )
        summary = diagnose_neural_run(config)
    except (OSError, ValueError) as exc:
        console.print(f"Error: could not diagnose neural run: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title="BABAPPA Neural Diagnostics")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", summary["outdir"])
    table.add_row("JSON", summary["json"])
    table.add_row("Probability summary", summary["summary_tsv"])
    table.add_row("Markdown", summary["markdown"])
    table.add_row("Warnings", str(len(summary["warnings"])))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")


@app.command("validate-neural-diagnostics")
def validate_neural_diagnostics(
    diag_dir: Path = typer.Option(
        ...,
        "--diag-dir",
        help="Neural diagnostics directory to validate.",
    )
) -> None:
    """Validate a BABAPPA neural diagnostics directory."""
    summary = validate_neural_diagnostics_dir(diag_dir)
    table = Table(title="BABAPPA Neural Diagnostics Validation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", summary["status"])
    table.add_row("Fail", str(summary["n_fail"]))
    table.add_row("Warning", str(summary["n_warning"]))
    console.print(table)

    if summary["warnings"]:
        console.print("Warnings:")
        for warning in summary["warnings"]:
            console.print(f"- {warning}")
    if summary["failures"]:
        console.print("Failures:", style="red")
        for failure in summary["failures"]:
            console.print(f"- {failure}", style="red")

    if summary["status"] == "fail":
        raise typer.Exit(code=1)


@app.command()
def validate(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input FASTA or alignment file to validate.",
    )
) -> None:
    """Validate basic input path requirements."""
    if not input_path.exists():
        console.print(f"Error: input path does not exist: {input_path}", style="red")
        raise typer.Exit(code=1)

    if not input_path.is_file():
        console.print(f"Error: input path is not a file: {input_path}", style="red")
        raise typer.Exit(code=1)

    if input_path.stat().st_size == 0:
        console.print(f"Error: input file is empty: {input_path}", style="red")
        raise typer.Exit(code=1)

    console.print(f"Input file is present and non-empty: {input_path}")


def _parse_methods(methods: str) -> list[str]:
    return [method.strip() for method in methods.split(",") if method.strip()]


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in _parse_csv(value)]


def _parse_optional_methods(methods: Optional[str]) -> Optional[list[str]]:
    if methods is None:
        return None
    parsed = _parse_methods(methods)
    return parsed or None


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"expected boolean value, got: {value}")


def _mapped_oracle_usability(method: str) -> str:
    if method in {"identity"}:
        return "yes"
    if method == "codon_dropout":
        return "no_unmappable_noise_control"
    if method in {"mafft", "babappalign", "muscle"}:
        return "requires_site_map_qc"
    if method in {"prank", "tcoffee"}:
        return "diagnostic_requires_site_map_qc"
    return "unknown"


def _print_warnings(warnings: list[str]) -> None:
    if warnings:
        console.print("Warnings:")
        for warning in warnings:
            console.print(f"- {warning}")


def _print_failures(failures: list[str]) -> None:
    if failures:
        console.print("Failures:", style="red")
        for failure in failures:
            console.print(f"- {failure}", style="red")


def _print_validation_table(title: str, summary: dict, count_key: Optional[str] = None) -> None:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Status", str(summary.get("status")))
    if count_key is not None and count_key in summary:
        table.add_row(count_key, str(summary[count_key]))
    table.add_row("Fail", str(summary.get("n_fail", 0)))
    table.add_row("Warning", str(summary.get("n_warning", 0)))
    console.print(table)
    _print_warnings(summary.get("warnings", []))
    _print_failures(summary.get("failures", []))


def _print_summary_table(title: str, summary: dict, keys: list[str]) -> None:
    table = Table(title=title)
    table.add_column("Field")
    table.add_column("Value")
    for key in keys:
        value = summary.get(key)
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        table.add_row(key, str(value))
    console.print(table)
    _print_warnings(summary.get("warnings", []))


def _format_methods(methods: list) -> str:
    return ",".join(str(method) for method in methods)


def _format_mapping(mapping: dict) -> str:
    if not mapping:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(mapping.items()))


def _optional_path_to_str(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    return str(path)


def _print_metric_summary(metrics_by_split: dict) -> None:
    table = Table(title="Prediction Metrics")
    table.add_column("Split")
    table.add_column("n")
    table.add_column("Accuracy")
    table.add_column("AUROC")
    for split in ["train", "val", "calib", "test", "all"]:
        metrics = metrics_by_split.get(split)
        if not metrics:
            continue
        table.add_row(
            split,
            str(metrics.get("n")),
            _format_optional_float(metrics.get("accuracy")),
            _format_optional_float(metrics.get("auroc")),
        )
    console.print(table)


def _format_optional_float(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _validate_simulation_directory(
    sim_dir: Path,
    require_branch_truth: bool = False,
    warnings: Optional[list[str]] = None,
) -> list[str]:
    errors: list[str] = []
    validation_warnings = warnings if warnings is not None else []
    manifest_path = sim_dir / "manifest.json"
    if not manifest_path.exists():
        return [f"missing manifest.json: {manifest_path}"]

    manifest = _read_json_safely(manifest_path, errors)
    if not isinstance(manifest, dict):
        if not errors:
            errors.append("manifest.json is not a JSON object")
        return errors

    family_ids = manifest.get("family_ids")
    if not isinstance(family_ids, list):
        errors.append("manifest.json does not contain a family_ids list")
        return errors

    families_dir = sim_dir / "families"
    if not families_dir.exists():
        errors.append(f"missing families directory: {families_dir}")
        return errors

    n_branch_truth_files = 0
    n_branch_site_truth_rows = 0
    n_branch_positive_rows = 0
    for family_id in family_ids:
        if not isinstance(family_id, str):
            errors.append("manifest family_ids contains a non-string entry")
            continue

        family_dir = families_dir / family_id
        if not family_dir.exists():
            errors.append(f"missing family directory: {family_dir}")
            continue

        required_files = {
            key: family_dir / f"{family_id}{suffix}"
            for key, suffix in REQUIRED_SIMULATION_SUFFIXES.items()
        }
        for key, path in required_files.items():
            if not path.exists():
                errors.append(f"missing {key} file: {path}")

        fasta_path = required_files["fasta"]
        if fasta_path.exists() and fasta_path.stat().st_size == 0:
            errors.append(f"FASTA file is empty: {fasta_path}")

        truth_path = required_files["truth"]
        if truth_path.exists():
            _read_json_safely(truth_path, errors)

        events_path = required_files["events"]
        if events_path.exists():
            _validate_tsv_header(events_path, EVENTS_HEADER, errors)

        homology_path = required_files["homology"]
        if homology_path.exists():
            _validate_tsv_header(homology_path, HOMOLOGY_HEADER, errors)

        branch_truth_path = family_dir / f"{family_id}.branch_truth.json"
        if branch_truth_path.exists():
            branch_truth = _read_json_safely(branch_truth_path, errors)
            if isinstance(branch_truth, dict):
                n_branch_truth_files += 1
                records = branch_truth.get("branch_site_records")
                if not isinstance(records, list):
                    errors.append(f"branch truth missing branch_site_records list: {branch_truth_path}")
                else:
                    n_branch_site_truth_rows += len(records)
                    n_branch_positive_rows += sum(
                        1 for record in records
                        if isinstance(record, dict)
                        and str(record.get("y_branch_site", "")).strip() in {"1", "1.0", "true", "True"}
                    )
                if branch_truth.get("truth_source") not in {"explicit_simulator_branch_truth", None}:
                    errors.append(f"branch truth source is not explicit_simulator_branch_truth: {branch_truth_path}")
        elif require_branch_truth:
            errors.append(f"missing branch truth file: {branch_truth_path}")
        else:
            validation_warnings.append(f"missing optional branch truth file: {branch_truth_path}")

    branch_manifest_path = sim_dir / "branch_truth_manifest.json"
    branch_site_truth_path = sim_dir / "branch_site_truth.tsv"
    if branch_manifest_path.exists():
        _read_json_safely(branch_manifest_path, errors)
    elif require_branch_truth:
        errors.append(f"missing branch_truth_manifest.json: {branch_manifest_path}")
    else:
        validation_warnings.append(f"missing optional branch_truth_manifest.json: {branch_manifest_path}")

    if branch_site_truth_path.exists():
        _validate_tsv_header(branch_site_truth_path, BRANCH_SITE_TRUTH_HEADER, errors)
    elif require_branch_truth:
        errors.append(f"missing branch_site_truth.tsv: {branch_site_truth_path}")
    else:
        validation_warnings.append(f"missing optional branch_site_truth.tsv: {branch_site_truth_path}")

    if require_branch_truth and n_branch_truth_files == 0:
        errors.append("no branch truth files found")
    if require_branch_truth and n_branch_site_truth_rows == 0:
        errors.append("no branch-site truth rows found")

    manifest_status = manifest.get("branch_truth_status")
    if manifest_status not in (None, "explicit_truth_ok"):
        validation_warnings.append(f"manifest branch_truth_status is {manifest_status}")
    if manifest.get("n_branch_truth_files") not in (None, n_branch_truth_files):
        validation_warnings.append("manifest n_branch_truth_files does not match files on disk")
    if manifest.get("n_branch_site_truth_rows") not in (None, n_branch_site_truth_rows):
        validation_warnings.append("manifest n_branch_site_truth_rows does not match family branch truth files")
    if manifest.get("n_branch_positive_rows") not in (None, n_branch_positive_rows):
        validation_warnings.append("manifest n_branch_positive_rows does not match family branch truth files")

    return errors


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_json_safely(path: Path, errors: list[str]) -> object:
    try:
        return _read_json(path)
    except json.JSONDecodeError as exc:
        errors.append(f"invalid JSON in {path}: {exc}")
    except OSError as exc:
        errors.append(f"could not read {path}: {exc}")
    return None


def _validate_tsv_header(
    path: Path, expected_header: list[str], errors: list[str]
) -> None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n").split("\t")
    except OSError as exc:
        errors.append(f"could not read {path}: {exc}")
        return

    if header != expected_header:
        errors.append(f"unexpected TSV header in {path}")


def main() -> None:
    """Run the BABAPPA CLI."""
    app()


if __name__ == "__main__":
    main()
