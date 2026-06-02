"""Known-truth simulation benchmark framework for BABAPPA."""

from .compare_methods import (
    KnownTruthMethodComparisonConfig,
    KnownTruthReferenceComparisonPlanConfig,
    compare_methods_known_truth,
    plan_known_truth_reference_comparison,
)
from .design import KnownTruthBenchmarkDesignConfig, design_known_truth_benchmark
from .metrics import (
    KnownTruthCalibrationEvaluationConfig,
    KnownTruthEvaluationConfig,
    bh_qvalues,
    evaluate_known_truth_benchmark,
    evaluate_known_truth_calibration,
)
from .report import KnownTruthBenchmarkReportConfig, make_known_truth_benchmark_report
from .run_plan import KnownTruthBenchmarkPlanConfig, plan_known_truth_benchmark
from .simulate import (
    KnownTruthAlignmentConfig,
    KnownTruthScoringConfig,
    KnownTruthSimulationConfig,
    run_known_truth_alignments,
    score_known_truth_benchmark,
    simulate_known_truth_benchmark,
)
from .validate import KnownTruthValidationConfig, validate_known_truth_benchmark

__all__ = [
    "KnownTruthAlignmentConfig",
    "KnownTruthBenchmarkDesignConfig",
    "KnownTruthBenchmarkPlanConfig",
    "KnownTruthBenchmarkReportConfig",
    "KnownTruthCalibrationEvaluationConfig",
    "KnownTruthEvaluationConfig",
    "KnownTruthMethodComparisonConfig",
    "KnownTruthReferenceComparisonPlanConfig",
    "KnownTruthScoringConfig",
    "KnownTruthSimulationConfig",
    "KnownTruthValidationConfig",
    "bh_qvalues",
    "compare_methods_known_truth",
    "design_known_truth_benchmark",
    "evaluate_known_truth_benchmark",
    "evaluate_known_truth_calibration",
    "make_known_truth_benchmark_report",
    "plan_known_truth_benchmark",
    "plan_known_truth_reference_comparison",
    "run_known_truth_alignments",
    "score_known_truth_benchmark",
    "simulate_known_truth_benchmark",
    "validate_known_truth_benchmark",
]

