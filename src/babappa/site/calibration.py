"""Calibration for site-level neural predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from babappa import __version__
from babappa.calibration.baseline import (
    brier_score,
    expected_calibration_error,
    fit_temperature_grid,
    select_threshold_by_fdr,
    temperature_scale_probs,
)
from babappa.datasets.index import read_tsv, write_tsv
from babappa.site.baseline import _compute_binary_metrics

SITE_CALIBRATION_VERSION = __version__
ALLOWED_CALIBRATION_METHODS = {"none", "temperature", "quantile"}
CALIBRATED_FIELDNAMES = [
    "site_id",
    "family_id",
    "method",
    "saturation_tier",
    "split",
    "site_index_zero",
    "y_site",
    "prob_positive_raw",
    "prob_positive_calibrated",
    "pred_label_calibrated",
    "correct_calibrated",
]


@dataclass(frozen=True)
class SiteCalibrationConfig:
    """Configuration for site-level neural calibration."""

    model_dir: str
    outdir: str
    target_fdr: float = 0.10
    calibration_method: str = "temperature"
    threshold_grid_size: int = 501
    min_threshold: float = 0.0
    max_threshold: float = 1.0
    n_bins: int = 20

    def __post_init__(self) -> None:
        model_path = Path(self.model_dir)
        out_path = Path(self.outdir)
        if not model_path.exists():
            raise ValueError(f"model_dir does not exist: {model_path}")
        if not (model_path / "site_neural_predictions.tsv").exists():
            raise ValueError(f"model_dir is missing site_neural_predictions.tsv: {model_path}")
        if not 0 <= self.target_fdr <= 1:
            raise ValueError("target_fdr must be between 0 and 1")
        if self.calibration_method not in ALLOWED_CALIBRATION_METHODS:
            allowed = ", ".join(sorted(ALLOWED_CALIBRATION_METHODS))
            raise ValueError(f"calibration_method must be one of: {allowed}")
        if self.threshold_grid_size < 2:
            raise ValueError("threshold_grid_size must be >= 2")
        if self.min_threshold < 0:
            raise ValueError("min_threshold must be >= 0")
        if self.max_threshold > 1:
            raise ValueError("max_threshold must be <= 1")
        if self.min_threshold >= self.max_threshold:
            raise ValueError("min_threshold must be < max_threshold")
        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        out_path.mkdir(parents=True, exist_ok=True)


def calibrate_site_model(config: SiteCalibrationConfig) -> dict:
    """Calibrate site neural probabilities and select empirical FDR threshold."""
    outdir = Path(config.outdir)
    rows = read_tsv(Path(config.model_dir) / "site_neural_predictions.tsv")
    if not rows:
        raise ValueError("site_neural_predictions.tsv contains no rows")
    y = np.array([int(float(row["y_site"])) for row in rows], dtype=np.int32)
    probs = np.array([float(row["prob_positive"]) for row in rows], dtype=np.float64)
    calib_mask = np.array([row.get("split") == "calib" for row in rows])
    warnings: List[str] = []
    quantile_mapping = None
    if config.calibration_method == "none":
        temperature = 1.0
        calibrated = probs.copy()
    elif config.calibration_method == "temperature":
        fit = fit_temperature_grid(y[calib_mask], probs[calib_mask])
        temperature = float(fit["temperature"])
        warnings.extend(fit.get("warnings", []))
        calibrated = temperature_scale_probs(probs, temperature)
    else:
        temperature = 1.0
        quantile_mapping, quantile_warnings = _fit_quantile_mapping(
            y[calib_mask], probs[calib_mask], config.n_bins
        )
        warnings.extend(quantile_warnings)
        calibrated = _apply_quantile_mapping(probs, quantile_mapping)
    threshold_selection = select_threshold_by_fdr(
        y_true=y[calib_mask],
        probs=calibrated[calib_mask],
        target_fdr=config.target_fdr,
        min_threshold=config.min_threshold,
        max_threshold=config.max_threshold,
        threshold_grid_size=config.threshold_grid_size,
    )
    warnings.extend(threshold_selection.get("warnings", []))
    selected_threshold = float(threshold_selection["selected_threshold"])
    pred = (calibrated >= selected_threshold).astype(np.int32)
    output_rows = []
    for index, row in enumerate(rows):
        y_site = int(y[index])
        pred_label = int(pred[index])
        output_rows.append(
            {
                "site_id": row.get("site_id", ""),
                "family_id": row.get("family_id", ""),
                "method": row.get("method", ""),
                "saturation_tier": row.get("saturation_tier", "unknown") or "unknown",
                "split": row.get("split", ""),
                "site_index_zero": row.get("site_index_zero", ""),
                "y_site": y_site,
                "prob_positive_raw": float(probs[index]),
                "prob_positive_calibrated": float(calibrated[index]),
                "pred_label_calibrated": pred_label,
                "correct_calibrated": int(pred_label == y_site),
            }
        )

    calibration_path = outdir / "site_calibration.json"
    predictions_path = outdir / "site_calibrated_predictions.tsv"
    metrics_path = outdir / "site_calibrated_metrics.json"
    markdown_path = outdir / "site_calibration.md"
    metrics = {
        "site_calibration_version": SITE_CALIBRATION_VERSION,
        "metrics_by_split_raw": _metrics_by_field(rows, y, probs, 0.5, "split", True),
        "metrics_by_split_calibrated": _metrics_by_field(
            rows, y, calibrated, selected_threshold, "split", True
        ),
        "metrics_by_saturation_tier_calibrated": _metrics_by_field(
            rows, y, calibrated, selected_threshold, "saturation_tier", True
        ),
        "metrics_by_method_calibrated": _metrics_by_field(
            rows, y, calibrated, selected_threshold, "method", True
        ),
        "selected_threshold": selected_threshold,
        "temperature": temperature,
        "quantile_mapping": quantile_mapping,
        "target_fdr": config.target_fdr,
    }
    payload = {
        "site_calibration_version": SITE_CALIBRATION_VERSION,
        "source_model_dir": str(Path(config.model_dir)),
        "calibration_method": config.calibration_method,
        "temperature": temperature,
        "n_bins": config.n_bins,
        "quantile_mapping": quantile_mapping,
        "target_fdr": config.target_fdr,
        "selected_threshold": selected_threshold,
        "threshold_selection": threshold_selection,
        "warnings": sorted(set(warnings)),
        "calibration_split_size": int(calib_mask.sum()),
        "calibration_split_positive_count": int(y[calib_mask].sum()),
        "raw_calibration_metrics": _calibration_metrics(y[calib_mask], probs[calib_mask]),
        "calibrated_calibration_metrics": _calibration_metrics(
            y[calib_mask], calibrated[calib_mask]
        ),
        "generated_files": {
            "calibration": str(calibration_path),
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "markdown": str(markdown_path),
        },
        "note": "Site-level calibration for oracle-supervised classifier; not empirical branch-site inference.",
    }
    _write_json(calibration_path, payload)
    write_tsv(predictions_path, output_rows, CALIBRATED_FIELDNAMES)
    _write_json(metrics_path, metrics)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "calibration": str(calibration_path),
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "markdown": str(markdown_path),
        "temperature": temperature,
        "selected_threshold": selected_threshold,
        "warnings": payload["warnings"],
    }


def _calibration_metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    return {
        "n": int(y.size),
        "positives": int(y.sum()) if y.size else 0,
        "brier": brier_score(y, prob),
        "ece": expected_calibration_error(y, prob),
    }


def _fit_quantile_mapping(y: np.ndarray, prob: np.ndarray, n_bins: int) -> tuple[dict, List[str]]:
    warnings: List[str] = []
    y = np.asarray(y, dtype=np.int32)
    prob = np.asarray(prob, dtype=np.float64)
    if y.size == 0:
        return {"bin_edges": [0.0, 1.0], "bin_rates": [0.5], "n_bins": 1}, ["empty_calibration_split"]
    order = np.argsort(prob, kind="mergesort")
    sorted_prob = prob[order]
    sorted_y = y[order]
    bins = min(n_bins, max(1, y.size))
    index_bins = np.array_split(np.arange(y.size), bins)
    edges = []
    rates = []
    for indices in index_bins:
        if indices.size == 0:
            continue
        right_edge = float(sorted_prob[indices[-1]])
        positives = int(sorted_y[indices].sum())
        total = int(indices.size)
        rates.append(float((positives + 1) / (total + 2)))
        edges.append(right_edge)
    if not edges:
        return {"bin_edges": [0.0, 1.0], "bin_rates": [0.5], "n_bins": 1}, ["empty_quantile_bins"]
    edges[-1] = 1.0
    unique_pairs = []
    for edge, rate in zip(edges, rates):
        if not unique_pairs or edge > unique_pairs[-1][0]:
            unique_pairs.append((edge, rate))
        else:
            unique_pairs[-1] = (edge, rate)
    if len(unique_pairs) < bins:
        warnings.append("quantile_bins_collapsed_due_to_ties")
    return {
        "bin_edges": [float(edge) for edge, _rate in unique_pairs],
        "bin_rates": [float(rate) for _edge, rate in unique_pairs],
        "n_bins": len(unique_pairs),
    }, warnings


def _apply_quantile_mapping(prob: np.ndarray, mapping: dict) -> np.ndarray:
    edges = np.asarray(mapping.get("bin_edges", [1.0]), dtype=np.float64)
    rates = np.asarray(mapping.get("bin_rates", [0.5]), dtype=np.float64)
    indices = np.searchsorted(edges, prob, side="left")
    indices = np.clip(indices, 0, rates.size - 1)
    return rates[indices].astype(np.float64)


def _metrics_by_field(rows: List[dict], y: np.ndarray, prob: np.ndarray, threshold: float, field: str, include_all: bool) -> Dict[str, dict]:
    result = {}
    values = sorted({row.get(field, "") for row in rows})
    for value in values:
        mask = np.array([row.get(field, "") == value for row in rows])
        result[value or "unknown"] = _compute_binary_metrics(y[mask], prob[mask], threshold)
    if include_all:
        result["all"] = _compute_binary_metrics(y, prob, threshold)
    return result


def _render_markdown(payload: dict) -> str:
    warnings = payload.get("warnings") or []
    lines = [
        "# Site calibration",
        "",
        f"- Method: {payload.get('calibration_method')}",
        f"- Temperature: {payload.get('temperature')}",
        f"- Quantile bins: {(payload.get('quantile_mapping') or {}).get('n_bins')}",
        f"- Selected threshold: {payload.get('selected_threshold')}",
        f"- Target FDR: {payload.get('target_fdr')}",
        f"- Calibration split size: {payload.get('calibration_split_size')}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- none"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Calibration is based on oracle-supervised site predictions and should be interpreted as a simulation-development diagnostic.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
