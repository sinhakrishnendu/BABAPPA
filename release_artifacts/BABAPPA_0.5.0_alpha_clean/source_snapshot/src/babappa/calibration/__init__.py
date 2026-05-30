"""Calibration utilities for BABAPPA."""

from babappa.calibration.audit import validate_baseline_calibration_dir
from babappa.calibration.baseline import (
    BaselineCalibrationConfig,
    binary_nll,
    brier_score,
    calibrate_baseline_model,
    expected_calibration_error,
    fit_temperature_grid,
    select_threshold_by_fdr,
    temperature_scale_probs,
    threshold_metrics,
)
from babappa.calibration.neural import (
    NeuralCalibrationConfig,
    calibrate_neural_model,
)
from babappa.calibration.neural_audit import validate_neural_calibration_dir
from babappa.calibration.stratified_calibration import (
    StratifiedCalibrationConfig,
    calibrate_by_group,
)
from babappa.calibration.stratified_calibration_audit import (
    validate_stratified_calibration_dir,
)
from babappa.calibration.threshold_policy import (
    ThresholdPolicyConfig,
    build_threshold_policy,
)
from babappa.calibration.threshold_policy_audit import validate_threshold_policy_dir

__all__ = [
    "BaselineCalibrationConfig",
    "NeuralCalibrationConfig",
    "StratifiedCalibrationConfig",
    "ThresholdPolicyConfig",
    "binary_nll",
    "brier_score",
    "build_threshold_policy",
    "calibrate_baseline_model",
    "calibrate_by_group",
    "calibrate_neural_model",
    "expected_calibration_error",
    "fit_temperature_grid",
    "select_threshold_by_fdr",
    "temperature_scale_probs",
    "threshold_metrics",
    "validate_baseline_calibration_dir",
    "validate_neural_calibration_dir",
    "validate_stratified_calibration_dir",
    "validate_threshold_policy_dir",
]
