"""Calibration for scale-ready BABAPPA neural predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from babappa import __version__
from babappa.calibration.baseline import (
    ALLOWED_CALIBRATION_METHODS,
    binary_nll,
    brier_score,
    expected_calibration_error,
    fit_temperature_grid,
    select_threshold_by_fdr,
    temperature_scale_probs,
)
from babappa.datasets.index import read_tsv, write_tsv
from babappa.models.baseline import compute_binary_metrics

NEURAL_CALIBRATION_VERSION = __version__
VALID_SPLITS = ["train", "val", "calib", "test"]
NEURAL_CALIBRATED_PREDICTION_FIELDNAMES = [
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
class NeuralCalibrationConfig:
    """Configuration for calibrating scale-ready neural predictions."""

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
        required_files = [
            model_path / "predictions" / "neural_predictions.tsv",
            model_path / "neural_metrics.json",
            model_path / "neural_model_meta.json",
        ]
        for path in required_files:
            if not path.exists():
                raise ValueError(f"model_dir is missing required file: {path}")
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
            raise ValueError(f"calibration_method must be one of: {allowed}")
        out_path.mkdir(parents=True, exist_ok=True)


def calibrate_neural_model(config: NeuralCalibrationConfig) -> dict:
    """Calibrate neural probabilities and write calibrated neural artifacts."""
    model_dir = Path(config.model_dir)
    outdir = Path(config.outdir)
    rows = read_tsv(model_dir / "predictions" / "neural_predictions.tsv")
    if not rows:
        raise ValueError("neural_predictions.tsv contains no rows")

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
    sorted_warnings = sorted(set(warnings))

    calibration_path = outdir / "neural_calibration.json"
    predictions_path = outdir / "neural_calibrated_predictions.tsv"
    metrics_path = outdir / "neural_calibrated_metrics.json"
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

    _write_json(
        calibration_path,
        {
            "neural_calibration_version": NEURAL_CALIBRATION_VERSION,
            "source_model_dir": str(model_dir),
            "calibration_method": config.calibration_method,
            "temperature": temperature,
            "target_fdr": config.target_fdr,
            "selected_threshold": selected_threshold,
            "threshold_selection": threshold_selection,
            "warnings": sorted_warnings,
            "calibration_split_size": int(calib_mask.sum()),
            "calibration_split_positive_count": int(y_true[calib_mask].sum()),
            "raw_calibration_metrics": _calibration_scores(
                y_true[calib_mask], raw_probs[calib_mask]
            ),
            "calibrated_calibration_metrics": _calibration_scores(
                y_true[calib_mask], calibrated_probs[calib_mask]
            ),
            "note": "Calibration for scale-ready gene-level neural trainer; not final branch-site BABAPPA calibration.",
        },
    )
    write_tsv(
        predictions_path,
        prediction_rows,
        NEURAL_CALIBRATED_PREDICTION_FIELDNAMES,
    )
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
            "note": "Calibrated metrics for scale-ready gene-level neural trainer; not final branch-site BABAPPA calibration.",
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


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
