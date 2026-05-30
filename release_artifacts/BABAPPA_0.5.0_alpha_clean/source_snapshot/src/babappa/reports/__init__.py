"""Reporting utilities for BABAPPA."""

from babappa.reports.audit import validate_report_dir
from babappa.reports.ablation_compare import (
    AblationCompareConfig,
    compare_neural_ablations,
)
from babappa.reports.ablation_compare_audit import (
    validate_ablation_comparison_dir,
)
from babappa.reports.compare import ModelCompareConfig, compare_models
from babappa.reports.compare_audit import validate_model_comparison_dir
from babappa.reports.label_signal_audit import (
    LabelSignalAuditConfig,
    audit_label_signal,
)
from babappa.reports.label_signal_audit_audit import validate_label_signal_audit_dir
from babappa.reports.leakage_audit import LeakageAuditConfig, audit_leakage
from babappa.reports.leakage_audit_audit import validate_leakage_audit_dir
from babappa.reports.neural_diagnostics import (
    NeuralDiagnosticsConfig,
    diagnose_neural_run,
)
from babappa.reports.neural_diagnostics_audit import (
    validate_neural_diagnostics_dir,
)
from babappa.reports.prediction_diagnostics import (
    PredictionDiagnosticsConfig,
    diagnose_predictions,
)
from babappa.reports.prediction_diagnostics_audit import (
    validate_prediction_diagnostics_dir,
)
from babappa.reports.run_summary import RunSummaryConfig, summarize_run
from babappa.reports.run_summary_audit import validate_run_summary_dir
from babappa.reports.external_tier_summary import (
    ExternalTierSummaryConfig,
    summarize_external_tiers,
    validate_external_tier_summary_dir,
)
from babappa.reports.stratified_eval import (
    StratifiedEvalConfig,
    stratified_evaluate_predictions,
)
from babappa.reports.stratified_eval_audit import validate_stratified_eval_dir
from babappa.reports.summary import (
    ReportConfig,
    build_report,
    generate_report,
    load_json_if_exists,
    summarize_tsv,
)

__all__ = [
    "ModelCompareConfig",
    "AblationCompareConfig",
    "LabelSignalAuditConfig",
    "LeakageAuditConfig",
    "NeuralDiagnosticsConfig",
    "PredictionDiagnosticsConfig",
    "ReportConfig",
    "RunSummaryConfig",
    "ExternalTierSummaryConfig",
    "StratifiedEvalConfig",
    "build_report",
    "compare_models",
    "compare_neural_ablations",
    "audit_label_signal",
    "audit_leakage",
    "diagnose_predictions",
    "diagnose_neural_run",
    "generate_report",
    "load_json_if_exists",
    "summarize_run",
    "summarize_external_tiers",
    "summarize_tsv",
    "stratified_evaluate_predictions",
    "validate_ablation_comparison_dir",
    "validate_label_signal_audit_dir",
    "validate_leakage_audit_dir",
    "validate_model_comparison_dir",
    "validate_neural_diagnostics_dir",
    "validate_prediction_diagnostics_dir",
    "validate_report_dir",
    "validate_run_summary_dir",
    "validate_external_tier_summary_dir",
    "validate_stratified_eval_dir",
]
