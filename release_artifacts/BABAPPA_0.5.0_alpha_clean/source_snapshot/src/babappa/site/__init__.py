"""Site-level oracle datasets and baselines for BABAPPA."""

from babappa.site.baseline import SiteBaselineConfig, train_site_baseline
from babappa.site.baseline_audit import validate_site_baseline_dir
from babappa.site.aggregate import SiteAggregationConfig, aggregate_site_predictions
from babappa.site.aggregate_audit import validate_site_aggregation_dir
from babappa.site.aggregation_controls import (
    SiteAggregationControlConfig,
    run_site_aggregation_controls,
)
from babappa.site.aggregation_controls_audit import validate_site_aggregation_controls_dir
from babappa.site.aggregation_threshold_policy import (
    AggregationThresholdPolicyConfig,
    build_aggregation_threshold_policy,
)
from babappa.site.aggregation_threshold_policy_audit import (
    validate_aggregation_threshold_policy_dir,
)
from babappa.site.calibration import SiteCalibrationConfig, calibrate_site_model
from babappa.site.calibration_audit import validate_site_calibration_dir
from babappa.site.calibration_compare import (
    SiteCalibrationCompareConfig,
    compare_site_calibrations,
)
from babappa.site.calibration_compare_audit import (
    validate_site_calibration_comparison_dir,
)
from babappa.site.compare import SiteModelCompareConfig, compare_site_models
from babappa.site.compare_audit import validate_site_model_comparison_dir
from babappa.site.dataset import SiteDatasetConfig, build_site_dataset
from babappa.site.dataset_audit import validate_site_dataset_dir
from babappa.site.leakage import audit_site_dataset_leakage
from babappa.site.neural_audit import validate_site_neural_dir
from babappa.site.neural_data import (
    SiteFeatureTorchDataset,
    SiteNeuralDatasetConfig,
    collate_site_feature_batch,
    load_site_feature_arrays,
)
from babappa.site.neural_model import SiteMLPClassifier, count_parameters
from babappa.site.neural_train import SiteNeuralTrainConfig, train_site_neural_model
from babappa.site.oracle_labels import (
    OracleSiteLabelConfig,
    extract_oracle_site_labels,
    normalize_site_indices,
)
from babappa.site.oracle_labels_audit import validate_site_label_dir
from babappa.site.stratified_eval import (
    SiteStratifiedEvalConfig,
    site_stratified_evaluate,
)
from babappa.site.stratified_eval_audit import validate_site_stratified_eval_dir
from babappa.site.stability import SiteStabilityConfig, run_site_stability_benchmark
from babappa.site.stability_audit import validate_site_stability_dir
from babappa.site.threshold_policy import (
    SiteThresholdPolicyConfig,
    build_site_threshold_policy,
)
from babappa.site.threshold_policy_audit import validate_site_threshold_policy_dir

__all__ = [
    "OracleSiteLabelConfig",
    "AggregationThresholdPolicyConfig",
    "SiteAggregationControlConfig",
    "SiteAggregationConfig",
    "SiteBaselineConfig",
    "SiteCalibrationConfig",
    "SiteCalibrationCompareConfig",
    "SiteDatasetConfig",
    "SiteFeatureTorchDataset",
    "SiteModelCompareConfig",
    "SiteMLPClassifier",
    "SiteNeuralDatasetConfig",
    "SiteNeuralTrainConfig",
    "SiteStabilityConfig",
    "SiteStratifiedEvalConfig",
    "SiteThresholdPolicyConfig",
    "aggregate_site_predictions",
    "audit_site_dataset_leakage",
    "build_aggregation_threshold_policy",
    "build_site_dataset",
    "build_site_threshold_policy",
    "calibrate_site_model",
    "collate_site_feature_batch",
    "compare_site_calibrations",
    "compare_site_models",
    "count_parameters",
    "extract_oracle_site_labels",
    "load_site_feature_arrays",
    "normalize_site_indices",
    "run_site_aggregation_controls",
    "run_site_stability_benchmark",
    "site_stratified_evaluate",
    "train_site_baseline",
    "train_site_neural_model",
    "validate_aggregation_threshold_policy_dir",
    "validate_site_aggregation_controls_dir",
    "validate_site_aggregation_dir",
    "validate_site_baseline_dir",
    "validate_site_calibration_dir",
    "validate_site_calibration_comparison_dir",
    "validate_site_dataset_dir",
    "validate_site_label_dir",
    "validate_site_model_comparison_dir",
    "validate_site_neural_dir",
    "validate_site_stability_dir",
    "validate_site_stratified_eval_dir",
    "validate_site_threshold_policy_dir",
]
