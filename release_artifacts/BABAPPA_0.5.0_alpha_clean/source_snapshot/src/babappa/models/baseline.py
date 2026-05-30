"""NumPy logistic-regression baseline for BABAPPA dataset indexes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv
from babappa.leakage_policy import should_exclude_from_feature_model

BASELINE_MODEL_VERSION = __version__
DEFAULT_FEATURE_CANDIDATES = [
    "n_taxa",
    "n_codons",
    "n_channels",
    "gap_codon_count",
    "gap_codon_fraction",
    "codon_id_mean",
    "codon_id_std",
    "codon_id_min",
    "codon_id_max",
    "codon_id_nonzero_fraction",
    "unique_codon_id_count",
    "unique_codon_id_fraction",
    "mean_taxon_codon_id_std",
    "mean_site_codon_id_std",
]
PREDICTION_FIELDNAMES = [
    "family_id",
    "method",
    "split",
    "tensor_file",
    "gene_label",
    "saturation_tier",
    "prob_positive",
    "pred_label",
    "correct",
]
VALID_SPLITS = ["train", "val", "calib", "test"]


@dataclass(frozen=True)
class BaselineTrainConfig:
    """Configuration for the NumPy baseline logistic-regression model."""

    dataset_dir: str
    outdir: str
    seed: int = 42
    learning_rate: float = 0.05
    epochs: int = 300
    l2: float = 0.001
    threshold: float = 0.5

    def __post_init__(self) -> None:
        dataset_path = Path(self.dataset_dir)
        out_path = Path(self.outdir)
        if not dataset_path.exists():
            raise ValueError(f"dataset_dir does not exist: {dataset_path}")
        for required in ("dataset_index.json", "features.tsv", "splits.tsv"):
            if not (dataset_path / required).exists():
                raise ValueError(f"dataset_dir is missing {required}: {dataset_path}")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.l2 < 0:
            raise ValueError("l2 must be >= 0")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        out_path.mkdir(parents=True, exist_ok=True)


def get_default_feature_columns(rows: List[dict]) -> List[str]:
    """Return available numeric, non-leaking baseline feature columns."""
    if not rows:
        return []
    available_columns = set()
    for row in rows:
        available_columns.update(row.keys())
    return [
        column
        for column in DEFAULT_FEATURE_CANDIDATES
        if column in available_columns and not should_exclude_from_feature_model(column)
    ]


def make_matrix(
    rows: List[dict], feature_columns: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Build feature matrix, labels, and row metadata from feature rows."""
    warnings: List[str] = []
    return _make_matrix(rows, feature_columns, warnings)


def fit_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: BaselineTrainConfig,
    rng: np.random.Generator,
) -> dict:
    """Fit full-batch logistic regression with L2 regularization."""
    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D matrix")
    if X_train.shape[0] == 0:
        raise ValueError("train split has no rows")

    n_rows, n_features = X_train.shape
    weights = rng.normal(loc=0.0, scale=0.01, size=n_features)
    bias = 0.0
    sample_weights, warnings = _class_weights(y_train)
    training_history = []

    for epoch in range(1, config.epochs + 1):
        logits = X_train @ weights + bias
        prob = _sigmoid(logits)
        errors = (prob - y_train) * sample_weights
        grad_w = (X_train.T @ errors) / n_rows + config.l2 * weights
        grad_b = float(errors.mean())
        weights -= config.learning_rate * grad_w
        bias -= config.learning_rate * grad_b

        if epoch % 10 == 0 or epoch == config.epochs:
            loss = _weighted_binary_cross_entropy(
                y_true=y_train,
                y_prob=prob,
                sample_weights=sample_weights,
                weights=weights,
                l2=config.l2,
            )
            training_history.append({"epoch": epoch, "loss": loss})

    return {
        "weights": weights,
        "bias": bias,
        "training_history": training_history,
        "warnings": warnings,
    }


def predict_proba(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """Predict positive-class probability."""
    return _sigmoid(X @ weights + bias)


def predict_labels(prob: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probabilities into binary labels."""
    return (prob >= threshold).astype(np.int32)


def compute_binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict:
    """Compute small binary-classification metrics without external libraries."""
    y_true = y_true.astype(np.int32)
    y_prob = y_prob.astype(np.float64)
    n = int(y_true.size)
    positives = int(y_true.sum())
    negatives = int(n - positives)
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

    y_pred = predict_labels(y_prob, threshold)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    mcc = _mcc(tp=tp, tn=tn, fp=fp, fn=fn)
    return {
        "n": n,
        "positives": positives,
        "negatives": negatives,
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        "auroc": _auroc_pairwise(y_true, y_prob),
    }


def train_baseline_model(config: BaselineTrainConfig) -> dict:
    """Train the NumPy baseline and write model, predictions, and metrics."""
    dataset_dir = Path(config.dataset_dir)
    outdir = Path(config.outdir)
    feature_rows = read_tsv(dataset_dir / "features.tsv")
    split_rows = read_tsv(dataset_dir / "splits.tsv")
    merged_rows = _merge_split_rows(feature_rows, split_rows)
    feature_columns = get_default_feature_columns(merged_rows)
    warnings: List[str] = []
    excluded_columns = _leakage_excluded_columns(merged_rows)
    if excluded_columns:
        warnings.append(
            "excluded_leakage_like_feature_columns:" + ",".join(excluded_columns)
        )
    X, y, row_metadata = _make_matrix(merged_rows, feature_columns, warnings)
    train_mask = np.array([row["split"] == "train" for row in row_metadata])
    if not train_mask.any():
        raise ValueError("train split has no rows")

    feature_mean = X[train_mask].mean(axis=0)
    feature_std = X[train_mask].std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    X_standardized = (X - feature_mean) / feature_std

    rng = np.random.default_rng(config.seed)
    fit = fit_logistic_regression(
        X_train=X_standardized[train_mask],
        y_train=y[train_mask],
        config=config,
        rng=rng,
    )
    warnings.extend(fit["warnings"])
    weights = fit["weights"]
    bias = float(fit["bias"])
    y_prob = predict_proba(X_standardized, weights, bias)
    y_pred = predict_labels(y_prob, config.threshold)

    predictions = _build_prediction_rows(
        row_metadata=row_metadata,
        y_prob=y_prob,
        y_pred=y_pred,
    )
    metrics_by_split = _metrics_by_split(row_metadata, y, y_prob, config.threshold)

    model_path = outdir / "baseline_model.npz"
    meta_path = outdir / "baseline_model_meta.json"
    predictions_path = outdir / "baseline_predictions.tsv"
    metrics_path = outdir / "baseline_metrics.json"

    np.savez_compressed(
        model_path,
        weights=weights,
        bias=np.array(bias, dtype=np.float64),
        feature_mean=feature_mean,
        feature_std=feature_std,
        threshold=np.array(config.threshold, dtype=np.float64),
    )
    _write_json(
        meta_path,
        {
            "baseline_model_version": BASELINE_MODEL_VERSION,
            "dataset_dir": str(dataset_dir),
            "feature_columns": feature_columns,
            "excluded_feature_columns": excluded_columns,
            "seed": config.seed,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "l2": config.l2,
            "threshold": config.threshold,
            "train_rows": int(train_mask.sum()),
            "warnings": sorted(set(warnings)),
            "training_history": fit["training_history"],
        },
    )
    write_tsv(predictions_path, predictions, PREDICTION_FIELDNAMES)
    _write_json(
        metrics_path,
        {
            "metrics_by_split": metrics_by_split,
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
        "metrics_by_split": metrics_by_split,
}


def _leakage_excluded_columns(rows: List[dict]) -> List[str]:
    columns = sorted({column for row in rows for column in row.keys()})
    return [
        column
        for column in columns
        if column not in DEFAULT_FEATURE_CANDIDATES
        and should_exclude_from_feature_model(column)
    ] + [
        column
        for column in columns
        if column in DEFAULT_FEATURE_CANDIDATES
        and should_exclude_from_feature_model(column)
    ]


def _merge_split_rows(feature_rows: List[dict], split_rows: List[dict]) -> List[dict]:
    split_lookup = {
        (row["family_id"], row["method"], row["tensor_file"]): row["split"]
        for row in split_rows
    }
    merged_rows = []
    for row in feature_rows:
        key = (row.get("family_id"), row.get("method"), row.get("tensor_file"))
        merged = dict(row)
        merged["split"] = split_lookup.get(key, "")
        if not merged["split"]:
            raise ValueError(f"feature row has no split assignment: {key}")
        merged_rows.append(merged)
    return merged_rows


def _make_matrix(
    rows: List[dict], feature_columns: List[str], warnings: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    X = np.zeros((len(rows), len(feature_columns)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.float64)
    metadata = []
    for row_index, row in enumerate(rows):
        y[row_index] = _to_float(row.get("gene_label", 0), warnings, "gene_label")
        for column_index, column in enumerate(feature_columns):
            X[row_index, column_index] = _to_float(row.get(column, 0), warnings, column)
        metadata.append(
            {
                "family_id": row.get("family_id", ""),
                "method": row.get("method", ""),
                "split": row.get("split", ""),
                "tensor_file": row.get("tensor_file", ""),
                "gene_label": int(y[row_index]),
                "saturation_tier": row.get("saturation_tier", ""),
            }
        )
    return X, y, metadata


def _to_float(value: object, warnings: List[str], column: str) -> float:
    try:
        if value in ("", None):
            raise ValueError
        return float(value)
    except (TypeError, ValueError):
        warnings.append(f"non_numeric_value_converted_to_zero:{column}")
        return 0.0


def _class_weights(y_train: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    unique_classes = np.unique(y_train.astype(np.int32))
    if unique_classes.size < 2:
        return np.ones_like(y_train, dtype=np.float64), ["single_class_train_split"]
    n_rows = y_train.size
    positives = int(y_train.sum())
    negatives = int(n_rows - positives)
    weights = np.ones_like(y_train, dtype=np.float64)
    weights[y_train == 1] = n_rows / (2 * positives)
    weights[y_train == 0] = n_rows / (2 * negatives)
    return weights, []


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35)))


def _weighted_binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weights: np.ndarray,
    weights: np.ndarray,
    l2: float,
) -> float:
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1 - eps)
    loss = -(
        sample_weights
        * (y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    ).mean()
    return float(loss + 0.5 * l2 * np.sum(weights * weights))


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _mcc(tp: int, tn: int, fp: int, fn: int) -> Optional[float]:
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator == 0:
        return None
    denom_sqrt = math.sqrt(float(denominator))
    if denom_sqrt == 0:
        return None
    return (tp * tn - fp * fn) / denom_sqrt


def _auroc_pairwise(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    positive_scores = y_prob[y_true == 1]
    negative_scores = y_prob[y_true == 0]
    if positive_scores.size == 0 or negative_scores.size == 0:
        return None
    wins = 0.0
    total = positive_scores.size * negative_scores.size
    for positive_score in positive_scores:
        wins += float((positive_score > negative_scores).sum())
        wins += 0.5 * float((positive_score == negative_scores).sum())
    return wins / total


def _build_prediction_rows(
    row_metadata: List[dict], y_prob: np.ndarray, y_pred: np.ndarray
) -> List[dict]:
    rows = []
    for index, row in enumerate(row_metadata):
        gene_label = int(row["gene_label"])
        pred_label = int(y_pred[index])
        rows.append(
            {
                "family_id": row["family_id"],
                "method": row["method"],
                "split": row["split"],
                "tensor_file": row["tensor_file"],
                "gene_label": gene_label,
                "saturation_tier": row["saturation_tier"],
                "prob_positive": float(y_prob[index]),
                "pred_label": pred_label,
                "correct": int(pred_label == gene_label),
            }
        )
    return rows


def _metrics_by_split(
    row_metadata: List[dict],
    y: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, dict]:
    metrics = {}
    for split in VALID_SPLITS:
        mask = np.array([row["split"] == split for row in row_metadata])
        metrics[split] = compute_binary_metrics(y[mask], y_prob[mask], threshold)
    metrics["all"] = compute_binary_metrics(y, y_prob, threshold)
    return metrics


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
