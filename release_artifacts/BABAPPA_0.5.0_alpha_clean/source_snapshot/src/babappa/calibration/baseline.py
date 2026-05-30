"""Baseline probability calibration and empirical threshold selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv
from babappa.models.baseline import VALID_SPLITS, compute_binary_metrics

CALIBRATION_VERSION = __version__
ALLOWED_CALIBRATION_METHODS = {"none", "temperature"}
CALIBRATED_PREDICTION_FIELDNAMES = [
    "family_id",
    "method",
    "split",
    "tensor_file",
    "gene_label",
    "saturation_tier",
    "prob_positive_raw",
    "prob_positive_calibrated",
    "pred_label_calibrated",
    "correct_calibrated",
]


@dataclass(frozen=True)
class BaselineCalibrationConfig:
    """Configuration for calibrating Cycle 7 baseline predictions."""

    model_dir: str
    outdir: str
    target_fdr: float = 0.10
    threshold_grid_size: int = 181
    min_threshold: float = 0.05
    max_threshold: float = 0.95
    calibration_method: str = "temperature"

    def __post_init__(self) -> None:
        model_path = Path(self.model_dir)
        out_path = Path(self.outdir)
        if not model_path.exists():
            raise ValueError(f"model_dir does not exist: {model_path}")
        for required in ("baseline_predictions.tsv", "baseline_metrics.json"):
            if not (model_path / required).exists():
                raise ValueError(f"model_dir is missing {required}: {model_path}")
        if not 0 <= self.target_fdr <= 1:
            raise ValueError("target_fdr must be between 0 and 1")
        if self.threshold_grid_size < 2:
            raise ValueError("threshold_grid_size must be >= 2")
        if self.min_threshold < 0:
            raise ValueError("min_threshold must be >= 0")
        if self.max_threshold > 1:
            raise ValueError("max_threshold must be <= 1")
        if self.min_threshold >= self.max_threshold:
            raise ValueError("min_threshold must be < max_threshold")
        if self.calibration_method not in ALLOWED_CALIBRATION_METHODS:
            allowed = ", ".join(sorted(ALLOWED_CALIBRATION_METHODS))
            raise ValueError(
                f"calibration_method must be one of: {allowed}"
            )
        out_path.mkdir(parents=True, exist_ok=True)


def clip_prob(p: object, eps: float = 1e-7) -> object:
    """Clip probabilities into a numerically safe open interval."""
    clipped = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    if clipped.ndim == 0:
        return float(clipped)
    return clipped


def prob_to_logit(p: object) -> object:
    """Convert probability values to logits with clipping."""
    clipped = clip_prob(p)
    logits = np.log(np.asarray(clipped, dtype=np.float64) / (1.0 - clipped))
    if logits.ndim == 0:
        return float(logits)
    return logits


def logit_to_prob(z: object) -> object:
    """Convert logits to probabilities with stable clipping."""
    logits = np.clip(np.asarray(z, dtype=np.float64), -35.0, 35.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    if probs.ndim == 0:
        return float(probs)
    return probs


def temperature_scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar temperature scaling to raw positive-class probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = np.asarray(prob_to_logit(probs), dtype=np.float64)
    return np.asarray(logit_to_prob(logits / temperature), dtype=np.float64)


def binary_nll(y_true: np.ndarray, probs: np.ndarray) -> Optional[float]:
    """Binary negative log-likelihood."""
    y_true = np.asarray(y_true, dtype=np.float64)
    probs = np.asarray(clip_prob(probs), dtype=np.float64)
    if y_true.size == 0:
        return None
    loss = -(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs))
    return float(loss.mean())


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> Optional[float]:
    """Mean squared probability error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    if y_true.size == 0:
        return None
    return float(np.mean((probs - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> Optional[float]:
    """Compute equal-width expected calibration error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    if y_true.size == 0 or n_bins < 1:
        return None

    ece = 0.0
    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if bin_index == n_bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        if not mask.any():
            continue
        confidence = float(probs[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += (int(mask.sum()) / y_true.size) * abs(confidence - accuracy)
    return float(ece)


def fit_temperature_grid(
    y_true: np.ndarray,
    probs: np.ndarray,
    temperatures: Optional[np.ndarray] = None,
) -> dict:
    """Fit scalar temperature by deterministic grid search on calibration data."""
    y_true = np.asarray(y_true, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    warnings: List[str] = []
    if y_true.size == 0:
        return {
            "temperature": 1.0,
            "objective": None,
            "method": "temperature_grid",
            "warnings": ["empty_calibration_split"],
        }
    if np.unique(y_true.astype(np.int32)).size < 2:
        return {
            "temperature": 1.0,
            "objective": binary_nll(y_true, probs),
            "method": "temperature_grid",
            "warnings": ["single_class_calibration_split"],
        }

    if temperatures is None:
        temperatures = np.linspace(0.25, 5.0, 96)

    best_temperature = 1.0
    best_objective = float("inf")
    for temperature in temperatures:
        scaled = temperature_scale_probs(probs, float(temperature))
        objective = binary_nll(y_true, scaled)
        if objective is None:
            continue
        if objective < best_objective:
            best_objective = objective
            best_temperature = float(temperature)

    if not np.isfinite(best_objective):
        warnings.append("temperature_grid_no_valid_objective")
        best_objective_value: Optional[float] = None
        best_temperature = 1.0
    else:
        best_objective_value = float(best_objective)

    return {
        "temperature": best_temperature,
        "objective": best_objective_value,
        "method": "temperature_grid",
        "warnings": warnings,
    }


def threshold_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Compute threshold-dependent confusion and call metrics."""
    y_true = np.asarray(y_true, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float64)
    y_pred = (probs >= threshold).astype(np.int32)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    called_positive = tp + fp
    called_negative = tn + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    accuracy = None if y_true.size == 0 else (tp + tn) / y_true.size
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "called_positive": called_positive,
        "called_negative": called_negative,
        "empirical_fdr": fp / max(1, called_positive),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
    }


def select_threshold_by_fdr(
    y_true: np.ndarray,
    probs: np.ndarray,
    target_fdr: float,
    min_threshold: float,
    max_threshold: float,
    threshold_grid_size: int,
) -> dict:
    """Select a threshold satisfying empirical FDR on calibration data."""
    y_true = np.asarray(y_true, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float64)
    warnings: List[str] = []

    if y_true.size == 0:
        warnings.append("empty_calibration_split")
        selected = threshold_metrics(y_true, probs, 0.5)
        return {
            "selected_threshold": 0.5,
            "metrics": selected,
            "warnings": warnings,
        }
    if np.unique(y_true).size < 2:
        warnings.append("single_class_calibration_split")
        selected = threshold_metrics(y_true, probs, 0.5)
        return {
            "selected_threshold": 0.5,
            "metrics": selected,
            "warnings": warnings,
        }

    candidates = []
    for threshold in np.linspace(min_threshold, max_threshold, threshold_grid_size):
        metrics = threshold_metrics(y_true, probs, float(threshold))
        if metrics["called_positive"] < 1:
            continue
        if metrics["empirical_fdr"] <= target_fdr:
            candidates.append(metrics)

    if not candidates:
        warnings.append("no_threshold_met_target_fdr")
        selected = threshold_metrics(y_true, probs, 0.5)
        return {
            "selected_threshold": 0.5,
            "metrics": selected,
            "warnings": warnings,
        }

    selected = sorted(
        candidates,
        key=lambda row: (
            -(row["recall"] if row["recall"] is not None else -1.0),
            row["empirical_fdr"],
            row["threshold"],
        ),
    )[0]
    return {
        "selected_threshold": float(selected["threshold"]),
        "metrics": selected,
        "warnings": warnings,
    }


def calibrate_baseline_model(config: BaselineCalibrationConfig) -> dict:
    """Calibrate baseline predictions and write calibrated artifacts."""
    model_dir = Path(config.model_dir)
    outdir = Path(config.outdir)
    rows = read_tsv(model_dir / "baseline_predictions.tsv")
    if not rows:
        raise ValueError("baseline_predictions.tsv contains no rows")

    y_true = _labels_from_rows(rows)
    raw_probs = _raw_probs_from_rows(rows)
    split_masks = _split_masks(rows)
    calib_mask = split_masks["calib"]
    warnings: List[str] = []

    if config.calibration_method == "temperature":
        fit = fit_temperature_grid(y_true[calib_mask], raw_probs[calib_mask])
        temperature = float(fit["temperature"])
        warnings.extend(fit["warnings"])
    else:
        fit = {
            "temperature": 1.0,
            "objective": binary_nll(y_true[calib_mask], raw_probs[calib_mask]),
            "method": "none",
            "warnings": [],
        }
        temperature = 1.0

    calibrated_probs = temperature_scale_probs(raw_probs, temperature)
    threshold_selection = select_threshold_by_fdr(
        y_true=y_true[calib_mask],
        probs=calibrated_probs[calib_mask],
        target_fdr=config.target_fdr,
        min_threshold=config.min_threshold,
        max_threshold=config.max_threshold,
        threshold_grid_size=config.threshold_grid_size,
    )
    warnings.extend(threshold_selection["warnings"])
    selected_threshold = float(threshold_selection["selected_threshold"])
    calibrated_pred = (calibrated_probs >= selected_threshold).astype(np.int32)

    calibration_path = outdir / "baseline_calibration.json"
    predictions_path = outdir / "baseline_calibrated_predictions.tsv"
    metrics_path = outdir / "baseline_calibrated_metrics.json"

    prediction_rows = _build_calibrated_prediction_rows(
        rows=rows,
        raw_probs=raw_probs,
        calibrated_probs=calibrated_probs,
        calibrated_pred=calibrated_pred,
    )
    metrics_by_split_raw = _metrics_by_split(rows, y_true, raw_probs, 0.5)
    metrics_by_split_calibrated = _metrics_by_split(
        rows, y_true, calibrated_probs, selected_threshold
    )

    raw_calib_metrics = _calibration_scores(y_true[calib_mask], raw_probs[calib_mask])
    calibrated_calib_metrics = _calibration_scores(
        y_true[calib_mask], calibrated_probs[calib_mask]
    )
    sorted_warnings = sorted(set(warnings))

    _write_json(
        calibration_path,
        {
            "calibration_version": CALIBRATION_VERSION,
            "source_model_dir": str(model_dir),
            "calibration_method": config.calibration_method,
            "temperature": temperature,
            "target_fdr": config.target_fdr,
            "selected_threshold": selected_threshold,
            "threshold_selection": threshold_selection,
            "warnings": sorted_warnings,
            "calibration_split_size": int(calib_mask.sum()),
            "calibration_split_positive_count": int(y_true[calib_mask].sum()),
            "raw_calibration_metrics": raw_calib_metrics,
            "calibrated_calibration_metrics": calibrated_calib_metrics,
        },
    )
    write_tsv(predictions_path, prediction_rows, CALIBRATED_PREDICTION_FIELDNAMES)
    _write_json(
        metrics_path,
        {
            "metrics_by_split_raw": metrics_by_split_raw,
            "metrics_by_split_calibrated": metrics_by_split_calibrated,
            "selected_threshold": selected_threshold,
            "temperature": temperature,
            "target_fdr": config.target_fdr,
            "files": {
                "calibration": str(calibration_path),
                "predictions": str(predictions_path),
                "metrics": str(metrics_path),
            },
        },
    )

    return {
        "status": "ok",
        "outdir": str(outdir),
        "calibration": str(calibration_path),
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "temperature": temperature,
        "selected_threshold": selected_threshold,
        "warnings": sorted_warnings,
    }


def _labels_from_rows(rows: List[dict]) -> np.ndarray:
    labels = []
    for row in rows:
        try:
            labels.append(int(float(row.get("gene_label", "0"))))
        except ValueError as exc:
            raise ValueError(f"gene_label is not numeric: {row.get('gene_label')}") from exc
    return np.asarray(labels, dtype=np.int32)


def _raw_probs_from_rows(rows: List[dict]) -> np.ndarray:
    probs = []
    for row in rows:
        try:
            prob = float(row.get("prob_positive", ""))
        except ValueError as exc:
            raise ValueError(
                f"prob_positive is not numeric: {row.get('prob_positive')}"
            ) from exc
        if not 0 <= prob <= 1:
            raise ValueError(f"prob_positive out of range: {prob}")
        probs.append(prob)
    return np.asarray(probs, dtype=np.float64)


def _split_masks(rows: List[dict]) -> Dict[str, np.ndarray]:
    return {
        split: np.asarray([row.get("split") == split for row in rows], dtype=bool)
        for split in VALID_SPLITS
    }


def _build_calibrated_prediction_rows(
    rows: List[dict],
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    calibrated_pred: np.ndarray,
) -> List[dict]:
    prediction_rows = []
    for index, row in enumerate(rows):
        gene_label = int(float(row["gene_label"]))
        pred_label = int(calibrated_pred[index])
        prediction_rows.append(
            {
                "family_id": row.get("family_id", ""),
                "method": row.get("method", ""),
                "split": row.get("split", ""),
                "tensor_file": row.get("tensor_file", ""),
                "gene_label": gene_label,
                "saturation_tier": row.get("saturation_tier", ""),
                "prob_positive_raw": float(raw_probs[index]),
                "prob_positive_calibrated": float(calibrated_probs[index]),
                "pred_label_calibrated": pred_label,
                "correct_calibrated": int(pred_label == gene_label),
            }
        )
    return prediction_rows


def _metrics_by_split(
    rows: List[dict], y_true: np.ndarray, probs: np.ndarray, threshold: float
) -> Dict[str, dict]:
    metrics = {}
    for split in VALID_SPLITS:
        mask = np.asarray([row.get("split") == split for row in rows], dtype=bool)
        metrics[split] = compute_binary_metrics(y_true[mask], probs[mask], threshold)
    metrics["all"] = compute_binary_metrics(y_true, probs, threshold)
    return metrics


def _calibration_scores(y_true: np.ndarray, probs: np.ndarray) -> dict:
    return {
        "n": int(y_true.size),
        "positives": int(y_true.sum()) if y_true.size else 0,
        "nll": binary_nll(y_true, probs),
        "brier_score": brier_score(y_true, probs),
        "expected_calibration_error": expected_calibration_error(y_true, probs),
    }


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
