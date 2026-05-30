"""Minimal NumPy logistic-regression baseline for site-level BABAPPA datasets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv
SITE_BASELINE_VERSION = __version__
VALID_POSITIVE_CLASS_WEIGHT = {"auto", "none"}
SITE_METADATA_COLUMNS = {
    "site_id",
    "family_id",
    "original_family_id",
    "source_dataset",
    "method",
    "split",
    "saturation_tier",
    "tensor_file",
    "labels_file",
    "aligned_site_index_one",
    "original_site_index_zero",
    "original_site_index_one",
    "mapping_status",
    "mapping_confidence",
    "mappable_site",
    "y_site",
}
LEAKAGE_NAME_TOKENS = ("selected", "truth", "label", "positive")
PREDICTION_FIELDNAMES = [
    "site_id",
    "family_id",
    "method",
    "saturation_tier",
    "split",
    "site_index_zero",
    "y_site",
    "prob_positive",
    "pred_label",
    "correct",
]


@dataclass(frozen=True)
class SiteBaselineConfig:
    """Configuration for the site-level NumPy baseline."""

    site_dataset_dir: str
    outdir: str
    seed: int = 42
    epochs: int = 300
    learning_rate: float = 0.05
    l2: float = 0.001
    positive_class_weight: str = "auto"
    threshold: float = 0.5

    def __post_init__(self) -> None:
        dataset_path = Path(self.site_dataset_dir)
        out_path = Path(self.outdir)
        if not dataset_path.exists():
            raise ValueError(f"site_dataset_dir does not exist: {dataset_path}")
        for filename in ("site_dataset_index.json", "site_features.tsv", "site_splits.tsv"):
            if not (dataset_path / filename).exists():
                raise ValueError(f"site_dataset_dir is missing {filename}: {dataset_path}")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.l2 < 0:
            raise ValueError("l2 must be >= 0")
        if self.positive_class_weight not in VALID_POSITIVE_CLASS_WEIGHT:
            allowed = ", ".join(sorted(VALID_POSITIVE_CLASS_WEIGHT))
            raise ValueError(f"positive_class_weight must be one of: {allowed}")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        out_path.mkdir(parents=True, exist_ok=True)


def train_site_baseline(config: SiteBaselineConfig) -> dict:
    """Train a site-level logistic baseline and write artifacts."""
    dataset_dir = Path(config.site_dataset_dir)
    outdir = Path(config.outdir)
    rows = read_tsv(dataset_dir / "site_features.tsv")
    feature_columns = get_site_feature_columns(rows)
    warnings: List[str] = []
    if not feature_columns:
        raise ValueError("no usable numeric feature columns found")
    X, y, metadata = _make_matrix(rows, feature_columns, warnings)
    train_mask = np.array([row["split"] == "train" for row in metadata])
    if not train_mask.any():
        raise ValueError("train split has no rows")

    feature_mean = X[train_mask].mean(axis=0)
    feature_std = X[train_mask].std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    X_standardized = (X - feature_mean) / feature_std
    rng = np.random.default_rng(config.seed)
    fit = _fit_logistic_regression(
        X_train=X_standardized[train_mask],
        y_train=y[train_mask],
        config=config,
        rng=rng,
    )
    warnings.extend(fit["warnings"])
    y_prob = _sigmoid(X_standardized @ fit["weights"] + fit["bias"])
    y_pred = (y_prob >= config.threshold).astype(np.int32)
    predictions = _prediction_rows(metadata, y_prob, y_pred)
    metrics = _all_metrics(metadata, y, y_prob, config.threshold)

    model_path = outdir / "site_baseline_model.npz"
    meta_path = outdir / "site_baseline_model_meta.json"
    predictions_path = outdir / "site_baseline_predictions.tsv"
    metrics_path = outdir / "site_baseline_metrics.json"
    np.savez_compressed(
        model_path,
        weights=fit["weights"],
        bias=np.array(fit["bias"], dtype=np.float64),
        feature_mean=feature_mean,
        feature_std=feature_std,
        threshold=np.array(config.threshold, dtype=np.float64),
    )
    _write_json(
        meta_path,
        {
            "site_baseline_version": SITE_BASELINE_VERSION,
            "site_dataset_dir": str(dataset_dir),
            "feature_columns": feature_columns,
            "excluded_feature_columns": _excluded_columns(rows),
            "seed": config.seed,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "l2": config.l2,
            "positive_class_weight": config.positive_class_weight,
            "threshold": config.threshold,
            "train_rows": int(train_mask.sum()),
            "warnings": sorted(set(warnings)),
            "training_history": fit["training_history"],
            "note": "Minimal site-level NumPy baseline; not final branch-site BABAPPA.",
        },
    )
    write_tsv(predictions_path, predictions, PREDICTION_FIELDNAMES)
    _write_json(
        metrics_path,
        {
            "metrics_by_split": metrics["by_split"],
            "metrics_by_saturation_tier": metrics["by_saturation_tier"],
            "metrics_by_method": metrics["by_method"],
            "metrics_by_split_saturation_tier": metrics["by_split_saturation_tier"],
            "metrics_by_split_method": metrics["by_split_method"],
            "feature_columns": feature_columns,
            "model_files": {
                "model": str(model_path),
                "meta": str(meta_path),
                "predictions": str(predictions_path),
                "metrics": str(metrics_path),
            },
        },
    )
    return {
        "status": "ok",
        "outdir": str(outdir),
        "model": str(model_path),
        "meta": str(meta_path),
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "metrics_by_split": metrics["by_split"],
        "warnings": sorted(set(warnings)),
    }


def get_site_feature_columns(rows: List[dict]) -> List[str]:
    """Return numeric non-leaking site feature columns."""
    if not rows:
        return []
    columns = list(rows[0].keys())
    selected = []
    for column in columns:
        if _exclude_column(column):
            continue
        if all(_to_float_or_none(row.get(column)) is not None for row in rows):
            selected.append(column)
    return selected


def _fit_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: SiteBaselineConfig,
    rng: np.random.Generator,
) -> dict:
    n_rows, n_features = X_train.shape
    weights = rng.normal(loc=0.0, scale=0.01, size=n_features)
    bias = 0.0
    sample_weights, warnings = _class_weights(y_train, config)
    history = []
    for epoch in range(1, config.epochs + 1):
        logits = X_train @ weights + bias
        prob = _sigmoid(logits)
        errors = (prob - y_train) * sample_weights
        grad_w = (X_train.T @ errors) / n_rows + config.l2 * weights
        grad_b = float(errors.mean())
        weights -= config.learning_rate * grad_w
        bias -= config.learning_rate * grad_b
        if epoch % 10 == 0 or epoch == config.epochs:
            history.append(
                {
                    "epoch": epoch,
                    "loss": _weighted_bce(y_train, prob, sample_weights, weights, config.l2),
                }
            )
    return {"weights": weights, "bias": float(bias), "warnings": warnings, "training_history": history}


def _make_matrix(
    rows: List[dict], feature_columns: List[str], warnings: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    X = np.zeros((len(rows), len(feature_columns)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.float64)
    metadata: List[dict] = []
    for row_index, row in enumerate(rows):
        y[row_index] = _to_float(row.get("y_site"), warnings, "y_site")
        for column_index, column in enumerate(feature_columns):
            X[row_index, column_index] = _to_float(row.get(column), warnings, column)
        metadata.append(
            {
                "site_id": row.get("site_id", ""),
                "family_id": row.get("family_id", ""),
                "method": row.get("method", ""),
                "saturation_tier": row.get("saturation_tier", "unknown") or "unknown",
                "split": row.get("split", ""),
                "site_index_zero": row.get("site_index_zero", ""),
                "y_site": int(y[row_index]),
            }
        )
    return X, y, metadata


def _all_metrics(
    metadata: List[dict], y: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict:
    return {
        "by_split": _metrics_by_field(metadata, y, y_prob, threshold, "split", include_all=True),
        "by_saturation_tier": _metrics_by_field(
            metadata, y, y_prob, threshold, "saturation_tier", include_all=True
        ),
        "by_method": _metrics_by_field(metadata, y, y_prob, threshold, "method", include_all=True),
        "by_split_saturation_tier": _metrics_by_fields(
            metadata, y, y_prob, threshold, ["split", "saturation_tier"]
        ),
        "by_split_method": _metrics_by_fields(
            metadata, y, y_prob, threshold, ["split", "method"]
        ),
    }


def _metrics_by_field(
    metadata: List[dict],
    y: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    field: str,
    include_all: bool,
) -> Dict[str, dict]:
    values = sorted({row.get(field, "") for row in metadata})
    metrics = {}
    for value in values:
        mask = np.array([row.get(field, "") == value for row in metadata])
        metrics[value or "unknown"] = _compute_binary_metrics(y[mask], y_prob[mask], threshold)
    if include_all:
        metrics["all"] = _compute_binary_metrics(y, y_prob, threshold)
    return metrics


def _metrics_by_fields(
    metadata: List[dict],
    y: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    fields: List[str],
) -> Dict[str, dict]:
    keys = sorted({tuple(row.get(field, "") for field in fields) for row in metadata})
    metrics = {}
    for key in keys:
        mask = np.array(
            [tuple(row.get(field, "") for field in fields) == key for row in metadata]
        )
        metrics["::".join(value or "unknown" for value in key)] = _compute_binary_metrics(
            y[mask], y_prob[mask], threshold
        )
    return metrics


def _compute_binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict:
    y_true = y_true.astype(np.int32)
    y_prob = y_prob.astype(np.float64)
    n = int(y_true.size)
    positives = int(y_true.sum())
    negatives = n - positives
    if n == 0:
        return {
            "n": 0,
            "positives": 0,
            "negatives": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "specificity": None,
            "f1": None,
            "mcc": None,
            "auroc": None,
        }
    y_pred = (y_prob >= threshold).astype(np.int32)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "n": n,
        "positives": positives,
        "negatives": negatives,
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": _mcc(tp, tn, fp, fn),
        "auroc": _auroc_rank(y_true, y_prob),
    }


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _mcc(tp: int, tn: int, fp: int, fn: int) -> Optional[float]:
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator == 0:
        return None
    return (tp * tn - fp * fn) / math.sqrt(float(denominator))


def _auroc_rank(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    cursor = 0
    while cursor < scores.size:
        end = cursor + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[cursor]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        ranks[order[cursor:end]] = average_rank
        cursor = end
    positive_rank_sum = float(ranks[y_true == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _prediction_rows(metadata: List[dict], y_prob: np.ndarray, y_pred: np.ndarray) -> List[dict]:
    rows = []
    for index, row in enumerate(metadata):
        y_site = int(row["y_site"])
        pred = int(y_pred[index])
        rows.append(
            {
                "site_id": row["site_id"],
                "family_id": row["family_id"],
                "method": row["method"],
                "saturation_tier": row["saturation_tier"],
                "split": row["split"],
                "site_index_zero": row["site_index_zero"],
                "y_site": y_site,
                "prob_positive": float(y_prob[index]),
                "pred_label": pred,
                "correct": int(pred == y_site),
            }
        )
    return rows


def _exclude_column(column: str) -> bool:
    if column in SITE_METADATA_COLUMNS:
        return True
    lowered = column.lower()
    if any(token in lowered for token in LEAKAGE_NAME_TOKENS):
        return True
    return False


def _excluded_columns(rows: List[dict]) -> List[str]:
    if not rows:
        return []
    return sorted(column for column in rows[0] if _exclude_column(column))


def _class_weights(y_train: np.ndarray, config: SiteBaselineConfig) -> Tuple[np.ndarray, List[str]]:
    if config.positive_class_weight == "none":
        return np.ones_like(y_train, dtype=np.float64), []
    positives = int(y_train.sum())
    negatives = int(y_train.size - positives)
    if positives == 0 or negatives == 0:
        return np.ones_like(y_train, dtype=np.float64), ["single_class_train_split"]
    weights = np.ones_like(y_train, dtype=np.float64)
    weights[y_train == 1] = y_train.size / (2 * positives)
    weights[y_train == 0] = y_train.size / (2 * negatives)
    return weights, []


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35)))


def _weighted_bce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weights: np.ndarray,
    weights: np.ndarray,
    l2: float,
) -> float:
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1 - eps)
    loss = -(
        sample_weights * (y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    ).mean()
    return float(loss + 0.5 * l2 * np.sum(weights * weights))


def _to_float(value: object, warnings: List[str], column: str) -> float:
    parsed = _to_float_or_none(value)
    if parsed is None:
        warnings.append(f"non_numeric_value_converted_to_zero:{column}")
        return 0.0
    return parsed


def _to_float_or_none(value: object) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
