"""Prediction score diagnostics for BABAPPA model outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from babappa import __version__

PREDICTION_DIAGNOSTICS_VERSION = __version__
SCORE_FIELDNAMES = [
    "model_name",
    "split",
    "n",
    "positives",
    "negatives",
    "prob_min",
    "prob_q01",
    "prob_q05",
    "prob_q25",
    "prob_median",
    "prob_q75",
    "prob_q95",
    "prob_q99",
    "prob_max",
    "prob_mean",
    "prob_std",
    "positive_prob_mean",
    "negative_prob_mean",
    "positive_prob_median",
    "negative_prob_median",
    "fraction_prob_ge_0_5",
    "fraction_prob_ge_0_3",
    "fraction_prob_ge_0_1",
    "threshold_0_5_called_positive",
    "threshold_0_5_recall",
    "threshold_0_5_specificity",
    "warnings",
]
THRESHOLD_FIELDNAMES = [
    "model_name",
    "split",
    "threshold",
    "tp",
    "fp",
    "tn",
    "fn",
    "called_positive",
    "precision",
    "recall",
    "specificity",
    "f1",
    "empirical_fdr",
    "accuracy",
]


@dataclass(frozen=True)
class PredictionDiagnosticsConfig:
    """Configuration for prediction score diagnostics."""

    predictions_tsv: str
    outdir: str
    metrics_json: Optional[str] = None
    calibration_json: Optional[str] = None
    probability_column: str = "prob_positive"
    label_column: str = "gene_label"
    split_column: str = "split"
    model_name: str = "model"

    def __post_init__(self) -> None:
        predictions_path = Path(self.predictions_tsv)
        if not predictions_path.exists():
            raise ValueError(f"predictions_tsv does not exist: {predictions_path}")
        if not predictions_path.is_file():
            raise ValueError(f"predictions_tsv is not a file: {predictions_path}")
        for label, value in [
            ("metrics_json", self.metrics_json),
            ("calibration_json", self.calibration_json),
        ]:
            if value is None:
                continue
            path = Path(value)
            if not path.exists():
                raise ValueError(f"{label} does not exist: {path}")
            if not path.is_file():
                raise ValueError(f"{label} is not a file: {path}")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def read_tsv(path: Path) -> List[dict]:
    """Read a TSV file as dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    """Write dictionaries to a TSV file."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_json_if_exists(path: Optional[Path]) -> Optional[dict]:
    """Load a JSON object if a path is supplied and exists."""
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def safe_float(value: object, default: Optional[float] = None) -> Optional[float]:
    """Convert values to float without raising."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def quantile(values: np.ndarray, q: float) -> Optional[float]:
    """Compute a quantile for non-empty arrays."""
    if values.size == 0:
        return None
    return float(np.quantile(values, q))


def diagnose_predictions(config: PredictionDiagnosticsConfig) -> dict:
    """Write score-distribution and threshold diagnostics for predictions."""
    rows = read_tsv(Path(config.predictions_tsv))
    if not rows:
        raise ValueError("predictions_tsv contains no rows")
    _validate_prediction_columns(rows[0], config)

    metrics = load_json_if_exists(
        Path(config.metrics_json) if config.metrics_json is not None else None
    )
    calibration = load_json_if_exists(
        Path(config.calibration_json) if config.calibration_json is not None else None
    )
    splits = sorted({row.get(config.split_column, "") for row in rows if row})
    splits = [split for split in splits if split]
    ordered_splits = [split for split in ["train", "val", "calib", "test"] if split in splits]
    ordered_splits.extend(split for split in splits if split not in ordered_splits)
    ordered_splits.append("all")

    score_rows = []
    threshold_rows = []
    warnings: List[str] = []
    diagnostics_by_split: Dict[str, dict] = {}
    threshold_grid = np.round(np.linspace(0.0, 1.0, 101), 2)

    for split in ordered_splits:
        split_rows = rows if split == "all" else [
            row for row in rows if row.get(config.split_column) == split
        ]
        labels = _labels(split_rows, config.label_column)
        probs = _probabilities(split_rows, config.probability_column)
        split_warnings = _detect_warnings(labels, probs)
        warnings.extend(split_warnings)
        score_row = _score_summary(config.model_name, split, labels, probs, split_warnings)
        score_rows.append(score_row)
        diagnostics_by_split[split] = dict(score_row)
        for threshold in threshold_grid:
            threshold_rows.append(
                _threshold_row(config.model_name, split, labels, probs, float(threshold))
            )

    warnings = sorted(set(warnings))
    outdir = Path(config.outdir)
    json_path = outdir / "prediction_diagnostics.json"
    score_path = outdir / "prediction_score_summary.tsv"
    threshold_path = outdir / "threshold_curve.tsv"
    markdown_path = outdir / "prediction_diagnostics.md"

    payload = {
        "prediction_diagnostics_version": PREDICTION_DIAGNOSTICS_VERSION,
        "model_name": config.model_name,
        "inputs": {
            "predictions_tsv": str(config.predictions_tsv),
            "metrics_json": str(config.metrics_json) if config.metrics_json else None,
            "calibration_json": str(config.calibration_json)
            if config.calibration_json
            else None,
            "probability_column": config.probability_column,
            "label_column": config.label_column,
            "split_column": config.split_column,
        },
        "diagnostics_by_split": diagnostics_by_split,
        "metrics_json": metrics,
        "calibration_json": calibration,
        "warnings": warnings,
        "generated_files": {
            "json": str(json_path),
            "score_summary": str(score_path),
            "threshold_curve": str(threshold_path),
            "markdown": str(markdown_path),
        },
    }
    _write_json(json_path, payload)
    write_tsv(score_path, score_rows, SCORE_FIELDNAMES)
    write_tsv(threshold_path, threshold_rows, THRESHOLD_FIELDNAMES)
    markdown_path.write_text(_render_markdown(payload, score_rows), encoding="utf-8")

    return {
        "status": "ok",
        "outdir": str(outdir),
        "json": str(json_path),
        "score_summary": str(score_path),
        "threshold_curve": str(threshold_path),
        "markdown": str(markdown_path),
        "warnings": warnings,
    }


def _validate_prediction_columns(row: dict, config: PredictionDiagnosticsConfig) -> None:
    missing = [
        column
        for column in [config.probability_column, config.label_column, config.split_column]
        if column not in row
    ]
    if missing:
        raise ValueError(f"predictions_tsv missing columns: {', '.join(missing)}")


def _labels(rows: List[dict], label_column: str) -> np.ndarray:
    return np.asarray([int(float(row[label_column])) for row in rows], dtype=np.int32)


def _probabilities(rows: List[dict], probability_column: str) -> np.ndarray:
    probs = [safe_float(row.get(probability_column)) for row in rows]
    if any(prob is None for prob in probs):
        raise ValueError(f"{probability_column} contains non-numeric values")
    return np.asarray(probs, dtype=np.float64)


def _score_summary(
    model_name: str,
    split: str,
    labels: np.ndarray,
    probs: np.ndarray,
    warnings: List[str],
) -> dict:
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]
    threshold_05 = _threshold_metrics(labels, probs, 0.5)
    return {
        "model_name": model_name,
        "split": split,
        "n": int(labels.size),
        "positives": positives,
        "negatives": negatives,
        "prob_min": _array_stat(probs, "min"),
        "prob_q01": quantile(probs, 0.01),
        "prob_q05": quantile(probs, 0.05),
        "prob_q25": quantile(probs, 0.25),
        "prob_median": quantile(probs, 0.50),
        "prob_q75": quantile(probs, 0.75),
        "prob_q95": quantile(probs, 0.95),
        "prob_q99": quantile(probs, 0.99),
        "prob_max": _array_stat(probs, "max"),
        "prob_mean": _array_stat(probs, "mean"),
        "prob_std": _array_stat(probs, "std"),
        "positive_prob_mean": _array_stat(pos_probs, "mean"),
        "negative_prob_mean": _array_stat(neg_probs, "mean"),
        "positive_prob_median": quantile(pos_probs, 0.50),
        "negative_prob_median": quantile(neg_probs, 0.50),
        "fraction_prob_ge_0_5": _fraction_ge(probs, 0.5),
        "fraction_prob_ge_0_3": _fraction_ge(probs, 0.3),
        "fraction_prob_ge_0_1": _fraction_ge(probs, 0.1),
        "threshold_0_5_called_positive": threshold_05["called_positive"],
        "threshold_0_5_recall": threshold_05["recall"],
        "threshold_0_5_specificity": threshold_05["specificity"],
        "warnings": ",".join(warnings),
    }


def _detect_warnings(labels: np.ndarray, probs: np.ndarray) -> List[str]:
    warnings = []
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    threshold_05 = _threshold_metrics(labels, probs, 0.5)
    called_negative = int(labels.size) - int(threshold_05["called_positive"])
    if positives == 0:
        warnings.append("missing_positive_class")
    if negatives == 0:
        warnings.append("missing_negative_class")
    if threshold_05["called_positive"] == 0 and positives > 0:
        warnings.append("all_negative_at_0_5")
    if called_negative == 0 and negatives > 0:
        warnings.append("all_positive_at_0_5")
    prob_std = _array_stat(probs, "std")
    if prob_std is not None and prob_std < 0.02:
        warnings.append("probability_collapse")
    if positives > 0 and negatives > 0:
        positive_mean = float(probs[labels == 1].mean())
        negative_mean = float(probs[labels == 0].mean())
        if abs(positive_mean - negative_mean) < 0.02:
            warnings.append("weak_separation")
        if positive_mean < negative_mean:
            warnings.append("inverted_signal_possible")
    return sorted(set(warnings))


def _threshold_row(
    model_name: str,
    split: str,
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> dict:
    metrics = _threshold_metrics(labels, probs, threshold)
    return {
        "model_name": model_name,
        "split": split,
        "threshold": threshold,
        **metrics,
    }


def _threshold_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    pred = (probs >= threshold).astype(np.int32)
    tp = int(((labels == 1) & (pred == 1)).sum())
    fp = int(((labels == 0) & (pred == 1)).sum())
    tn = int(((labels == 0) & (pred == 0)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum())
    called_positive = tp + fp
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "called_positive": called_positive,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "empirical_fdr": fp / max(1, called_positive),
        "accuracy": None if labels.size == 0 else (tp + tn) / labels.size,
    }


def _array_stat(values: np.ndarray, stat: str) -> Optional[float]:
    if values.size == 0:
        return None
    if stat == "min":
        return float(values.min())
    if stat == "max":
        return float(values.max())
    if stat == "mean":
        return float(values.mean())
    if stat == "std":
        return float(values.std())
    raise ValueError(f"unknown stat: {stat}")


def _fraction_ge(values: np.ndarray, threshold: float) -> Optional[float]:
    if values.size == 0:
        return None
    return float((values >= threshold).mean())


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _render_markdown(payload: dict, score_rows: List[dict]) -> str:
    lines = ["# Prediction diagnostics", ""]
    lines.extend(["## Input", ""])
    for key, value in payload["inputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Probability distribution by split", ""])
    lines.extend(
        [
            "| Split | n | Positives | Mean | Std | Pos mean | Neg mean | >=0.5 | Warnings |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in score_rows:
        lines.append(
            "| {split} | {n} | {positives} | {mean} | {std} | {pos_mean} | {neg_mean} | {ge_05} | {warnings} |".format(
                split=row["split"],
                n=row["n"],
                positives=row["positives"],
                mean=_format_float(row["prob_mean"]),
                std=_format_float(row["prob_std"]),
                pos_mean=_format_float(row["positive_prob_mean"]),
                neg_mean=_format_float(row["negative_prob_mean"]),
                ge_05=_format_float(row["fraction_prob_ge_0_5"]),
                warnings=row["warnings"] or "none",
            )
        )
    lines.extend(["", "## Threshold behavior", ""])
    lines.append(
        "Threshold curve data were written to `threshold_curve.tsv` for thresholds 0.00 through 1.00."
    )
    lines.extend(["", "## Warnings", ""])
    warnings = payload.get("warnings") or []
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- All-negative prediction at threshold 0.5 indicates threshold/model collapse.",
            "- AUROC can remain non-null even if thresholded recall is zero.",
            "- Calibration cannot rescue a model whose score ordering is weak or inverted.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _format_float(value: object) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.4f}"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
