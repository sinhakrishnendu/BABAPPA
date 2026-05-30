"""Stratified probability calibration for BABAPPA predictions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
from babappa.models.baseline import VALID_SPLITS, compute_binary_metrics

STRATIFIED_CALIBRATION_VERSION = __version__
STRATIFIED_CALIBRATED_PREDICTION_FIELDNAMES = [
    "family_id",
    "method",
    "split",
    "tensor_file",
    "gene_label",
    "saturation_tier",
    "prob_positive_raw",
    "prob_positive_group_calibrated",
    "group_selected_threshold",
    "pred_label_group_calibrated",
    "correct_group_calibrated",
]


@dataclass(frozen=True)
class StratifiedCalibrationConfig:
    """Configuration for group-wise probability calibration."""

    predictions_tsv: str
    outdir: str
    group_column: str = "saturation_tier"
    probability_column: str = "prob_positive"
    label_column: str = "gene_label"
    split_column: str = "split"
    target_fdr: float = 0.10
    calibration_method: str = "temperature"
    min_group_calib_n: int = 20
    threshold_grid_size: int = 181
    min_threshold: float = 0.05
    max_threshold: float = 0.95

    def __post_init__(self) -> None:
        predictions_path = Path(self.predictions_tsv)
        out_path = Path(self.outdir)
        if not predictions_path.exists():
            raise ValueError(f"predictions_tsv does not exist: {predictions_path}")
        if not 0 <= self.target_fdr <= 1:
            raise ValueError("target_fdr must be between 0 and 1")
        if self.calibration_method not in ALLOWED_CALIBRATION_METHODS:
            allowed = ", ".join(sorted(ALLOWED_CALIBRATION_METHODS))
            raise ValueError(f"calibration_method must be one of: {allowed}")
        if self.min_group_calib_n < 1:
            raise ValueError("min_group_calib_n must be >= 1")
        if self.threshold_grid_size < 2:
            raise ValueError("threshold_grid_size must be >= 2")
        if self.min_threshold < 0:
            raise ValueError("min_threshold must be >= 0")
        if self.max_threshold > 1:
            raise ValueError("max_threshold must be <= 1")
        if self.min_threshold >= self.max_threshold:
            raise ValueError("min_threshold must be < max_threshold")
        out_path.mkdir(parents=True, exist_ok=True)


def calibrate_by_group(config: StratifiedCalibrationConfig) -> dict:
    """Calibrate prediction probabilities independently by group when possible."""
    rows = read_tsv(Path(config.predictions_tsv))
    if not rows:
        raise ValueError("predictions_tsv contains no rows")
    _validate_required_columns(rows[0], config)

    outdir = Path(config.outdir)
    y_true = _labels_from_rows(rows, config.label_column)
    raw_probs = _probs_from_rows(rows, config.probability_column)
    groups = np.asarray(
        [row.get(config.group_column, "") or "unknown" for row in rows],
        dtype=object,
    )
    splits = np.asarray(
        [row.get(config.split_column, "") for row in rows],
        dtype=object,
    )
    warnings: List[str] = []

    global_fit, global_temperature, global_threshold_selection = _fit_one_group(
        y_true=y_true,
        raw_probs=raw_probs,
        calib_mask=splits == "calib",
        config=config,
        warnings=warnings,
        warning_prefix="global",
        allow_fallback=False,
    )
    global_threshold = float(global_threshold_selection["selected_threshold"])
    global_fallback = {
        "temperature": global_temperature,
        "fit": global_fit,
        "selected_threshold": global_threshold,
        "threshold_selection": global_threshold_selection,
    }

    group_payload: Dict[str, dict] = {}
    calibrated_probs = np.zeros_like(raw_probs, dtype=np.float64)
    selected_thresholds = np.zeros_like(raw_probs, dtype=np.float64)
    for group in sorted({str(group) for group in groups}):
        group_mask = groups == group
        calib_group_mask = group_mask & (splits == "calib")
        use_fallback = (
            int(calib_group_mask.sum()) < config.min_group_calib_n
            or np.unique(y_true[calib_group_mask]).size < 2
        )
        if use_fallback:
            reason = (
                "insufficient_group_calibration_rows"
                if int(calib_group_mask.sum()) < config.min_group_calib_n
                else "single_class_group_calibration_split"
            )
            warnings.append(f"{group}:{reason}_using_global_fallback")
            temperature = global_temperature
            threshold_selection = global_threshold_selection
            fit = global_fit
            selected_threshold = global_threshold
            used_global_fallback = True
        else:
            fit, temperature, threshold_selection = _fit_one_group(
                y_true=y_true,
                raw_probs=raw_probs,
                calib_mask=calib_group_mask,
                config=config,
                warnings=warnings,
                warning_prefix=group,
                allow_fallback=True,
            )
            selected_threshold = float(threshold_selection["selected_threshold"])
            used_global_fallback = False

        calibrated_probs[group_mask] = temperature_scale_probs(
            raw_probs[group_mask], temperature
        )
        selected_thresholds[group_mask] = selected_threshold
        group_payload[group] = {
            "calibration_split_size": int(calib_group_mask.sum()),
            "calibration_split_positive_count": int(y_true[calib_group_mask].sum()),
            "temperature": temperature,
            "selected_threshold": selected_threshold,
            "fit": fit,
            "threshold_selection": threshold_selection,
            "used_global_fallback": used_global_fallback,
        }

    pred_labels = (calibrated_probs >= selected_thresholds).astype(np.int32)
    prediction_rows = _prediction_rows(
        rows=rows,
        config=config,
        raw_probs=raw_probs,
        calibrated_probs=calibrated_probs,
        selected_thresholds=selected_thresholds,
        pred_labels=pred_labels,
    )
    metrics_by_group = _metrics_by_group(groups, y_true, calibrated_probs, pred_labels)
    metrics_by_split = _metrics_by_split(splits, y_true, calibrated_probs, pred_labels)
    metrics_by_group_and_split = _metrics_by_group_and_split(
        groups, splits, y_true, calibrated_probs, pred_labels
    )

    calibration_path = outdir / "stratified_calibration.json"
    predictions_path = outdir / "stratified_calibrated_predictions.tsv"
    metrics_path = outdir / "stratified_calibrated_metrics.json"
    markdown_path = outdir / "stratified_calibration.md"
    sorted_warnings = sorted(set(warnings))
    payload = {
        "stratified_calibration_version": STRATIFIED_CALIBRATION_VERSION,
        "group_column": config.group_column,
        "probability_column": config.probability_column,
        "target_fdr": config.target_fdr,
        "calibration_method": config.calibration_method,
        "min_group_calib_n": config.min_group_calib_n,
        "groups": group_payload,
        "global_fallback": global_fallback,
        "warnings": sorted_warnings,
        "generated_files": {
            "calibration": str(calibration_path),
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "markdown": str(markdown_path),
        },
        "note": "Stratified gene-level calibration; not final branch-site BABAPPA calibration.",
    }
    _write_json(calibration_path, payload)
    write_tsv(
        predictions_path,
        prediction_rows,
        STRATIFIED_CALIBRATED_PREDICTION_FIELDNAMES,
    )
    _write_json(
        metrics_path,
        {
            "metrics_by_group": metrics_by_group,
            "metrics_by_split": metrics_by_split,
            "metrics_by_group_and_split": metrics_by_group_and_split,
            "generated_files": payload["generated_files"],
        },
    )
    markdown_path.write_text(_render_markdown(payload, metrics_by_group), encoding="utf-8")

    return {
        "status": "ok",
        "outdir": str(outdir),
        "calibration": str(calibration_path),
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "markdown": str(markdown_path),
        "warnings": sorted_warnings,
    }


def _fit_one_group(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    calib_mask: np.ndarray,
    config: StratifiedCalibrationConfig,
    warnings: List[str],
    warning_prefix: str,
    allow_fallback: bool,
) -> tuple[dict, float, dict]:
    if config.calibration_method == "temperature":
        fit = fit_temperature_grid(y_true[calib_mask], raw_probs[calib_mask])
        temperature = float(fit["temperature"])
        warnings.extend(f"{warning_prefix}:{warning}" for warning in fit["warnings"])
    else:
        fit = {
            "temperature": 1.0,
            "objective": binary_nll(y_true[calib_mask], raw_probs[calib_mask]),
            "method": "none",
            "warnings": [],
        }
        temperature = 1.0
    calibrated = temperature_scale_probs(raw_probs[calib_mask], temperature)
    threshold_selection = select_threshold_by_fdr(
        y_true=y_true[calib_mask],
        probs=calibrated,
        target_fdr=config.target_fdr,
        min_threshold=config.min_threshold,
        max_threshold=config.max_threshold,
        threshold_grid_size=config.threshold_grid_size,
    )
    warnings.extend(
        f"{warning_prefix}:{warning}"
        for warning in threshold_selection["warnings"]
        if allow_fallback or warning_prefix == "global"
    )
    return fit, temperature, threshold_selection


def _validate_required_columns(row: dict, config: StratifiedCalibrationConfig) -> None:
    required = [config.probability_column, config.label_column, config.split_column]
    missing = [column for column in required if column not in row]
    if missing:
        raise ValueError("predictions_tsv missing columns: " + ", ".join(missing))


def _labels_from_rows(rows: List[dict], label_column: str) -> np.ndarray:
    labels = []
    for row in rows:
        try:
            labels.append(int(float(row.get(label_column, "0"))))
        except ValueError as exc:
            raise ValueError(f"{label_column} is not numeric: {row.get(label_column)}") from exc
    return np.asarray(labels, dtype=np.int32)


def _probs_from_rows(rows: List[dict], probability_column: str) -> np.ndarray:
    probs = []
    for row in rows:
        try:
            prob = float(row.get(probability_column, ""))
        except ValueError as exc:
            raise ValueError(
                f"{probability_column} is not numeric: {row.get(probability_column)}"
            ) from exc
        if not 0 <= prob <= 1:
            raise ValueError(f"{probability_column} out of range: {prob}")
        probs.append(prob)
    return np.asarray(probs, dtype=np.float64)


def _prediction_rows(
    rows: List[dict],
    config: StratifiedCalibrationConfig,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    selected_thresholds: np.ndarray,
    pred_labels: np.ndarray,
) -> List[dict]:
    output = []
    for index, row in enumerate(rows):
        gene_label = int(float(row.get(config.label_column, "0")))
        pred_label = int(pred_labels[index])
        output.append(
            {
                "family_id": row.get("family_id", ""),
                "method": row.get("method", ""),
                "split": row.get(config.split_column, ""),
                "tensor_file": row.get("tensor_file", ""),
                "gene_label": gene_label,
                "saturation_tier": row.get(config.group_column, "unknown") or "unknown",
                "prob_positive_raw": float(raw_probs[index]),
                "prob_positive_group_calibrated": float(calibrated_probs[index]),
                "group_selected_threshold": float(selected_thresholds[index]),
                "pred_label_group_calibrated": pred_label,
                "correct_group_calibrated": int(pred_label == gene_label),
            }
        )
    return output


def _metrics_by_group(
    groups: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    pred_labels: np.ndarray,
) -> Dict[str, dict]:
    return {
        str(group): _metrics_from_predictions(
            y_true[groups == group], probs[groups == group], pred_labels[groups == group]
        )
        for group in sorted({str(group) for group in groups})
    }


def _metrics_by_split(
    splits: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    pred_labels: np.ndarray,
) -> Dict[str, dict]:
    metrics = {}
    for split in VALID_SPLITS:
        mask = splits == split
        metrics[split] = _metrics_from_predictions(
            y_true[mask], probs[mask], pred_labels[mask]
        )
    metrics["all"] = _metrics_from_predictions(y_true, probs, pred_labels)
    return metrics


def _metrics_by_group_and_split(
    groups: np.ndarray,
    splits: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    pred_labels: np.ndarray,
) -> Dict[str, dict]:
    metrics = {}
    for group in sorted({str(group) for group in groups}):
        for split in VALID_SPLITS:
            mask = (groups == group) & (splits == split)
            metrics[f"{group}|{split}"] = _metrics_from_predictions(
                y_true[mask], probs[mask], pred_labels[mask]
            )
    return metrics


def _metrics_from_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    pred_labels: np.ndarray,
) -> dict:
    base = compute_binary_metrics(y_true, probs, 0.5)
    if y_true.size == 0:
        return base
    y_true = np.asarray(y_true, dtype=np.int32)
    pred_labels = np.asarray(pred_labels, dtype=np.int32)
    tp = int(((y_true == 1) & (pred_labels == 1)).sum())
    fp = int(((y_true == 0) & (pred_labels == 1)).sum())
    tn = int(((y_true == 0) & (pred_labels == 0)).sum())
    fn = int(((y_true == 1) & (pred_labels == 0)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = None
    if denominator > 0:
        denom_sqrt = math.sqrt(float(denominator))
        if denom_sqrt > 0:
            mcc = ((tp * tn) - (fp * fn)) / denom_sqrt
    base.update(
        {
            "accuracy": (tp + tn) / y_true.size,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "mcc": mcc,
        }
    )
    return base


def _calibration_scores(y_true: np.ndarray, probs: np.ndarray) -> dict:
    return {
        "n": int(y_true.size),
        "positives": int(y_true.sum()) if y_true.size else 0,
        "nll": binary_nll(y_true, probs),
        "brier_score": brier_score(y_true, probs),
        "expected_calibration_error": expected_calibration_error(y_true, probs),
    }


def _render_markdown(payload: dict, metrics_by_group: Dict[str, dict]) -> str:
    lines = [
        "# Stratified calibration report",
        "",
        "## Input",
        "",
        f"- Predictions: `{payload['generated_files']['predictions']}`",
        f"- Group column: `{payload['group_column']}`",
        f"- Probability column: `{payload['probability_column']}`",
        "",
        "## Groups",
        "",
        "| Group | Calib n | Temperature | Threshold | Fallback | AUROC | F1 |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for group, details in payload["groups"].items():
        metrics = metrics_by_group.get(group, {})
        lines.append(
            "| {group} | {n} | {temperature:.4f} | {threshold:.4f} | {fallback} | {auroc} | {f1} |".format(
                group=group,
                n=details["calibration_split_size"],
                temperature=float(details["temperature"]),
                threshold=float(details["selected_threshold"]),
                fallback=str(details["used_global_fallback"]),
                auroc=_format_optional(metrics.get("auroc")),
                f1=_format_optional(metrics.get("f1")),
            )
        )
    lines.extend(["", "## Warnings", ""])
    if payload["warnings"]:
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "- Per-group calibration is useful when probability scale shifts across saturation tiers.",
            "- Groups with small or single-class calibration splits fall back to global calibration.",
            "- This is gene-level calibration and is not final branch-site BABAPPA calibration.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _format_optional(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
