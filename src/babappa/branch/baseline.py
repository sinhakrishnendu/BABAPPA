"""Minimal NumPy branch-site baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from babappa import __version__
from babappa.branch.dataset import get_branch_site_feature_columns
from babappa.datasets.index import read_tsv, write_tsv
from babappa.site.baseline import _class_weights, _compute_binary_metrics, _sigmoid, _weighted_bce

BRANCH_BASELINE_VERSION = __version__
VALID_POSITIVE_CLASS_WEIGHT = {"auto", "none"}
PREDICTION_FIELDNAMES = [
    "branch_site_id",
    "family_id",
    "method",
    "saturation_tier",
    "split",
    "branch_id",
    "site_index_zero",
    "y_branch_site",
    "y_site",
    "gene_label",
    "prob_positive",
    "pred_label",
    "correct",
]


@dataclass(frozen=True)
class BranchSiteBaselineConfig:
    """Configuration for branch-site logistic baseline."""

    branch_site_dataset_dir: str
    outdir: str
    seed: int = 42
    epochs: int = 300
    learning_rate: float = 0.05
    l2: float = 0.001
    positive_class_weight: str = "auto"
    threshold: float = 0.5

    def __post_init__(self) -> None:
        dataset_path = Path(self.branch_site_dataset_dir)
        if not dataset_path.exists():
            raise ValueError(f"branch_site_dataset_dir does not exist: {dataset_path}")
        for filename in ("branch_site_dataset_index.json", "branch_site_features.tsv", "branch_site_splits.tsv"):
            if not (dataset_path / filename).exists():
                raise ValueError(f"branch_site_dataset_dir is missing {filename}: {dataset_path}")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.l2 < 0:
            raise ValueError("l2 must be >= 0")
        if self.positive_class_weight not in VALID_POSITIVE_CLASS_WEIGHT:
            raise ValueError("positive_class_weight must be auto or none")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def train_branch_site_baseline(config: BranchSiteBaselineConfig) -> dict:
    """Train a branch-site logistic baseline and write artifacts."""
    dataset_dir = Path(config.branch_site_dataset_dir)
    outdir = Path(config.outdir)
    rows = read_tsv(dataset_dir / "branch_site_features.tsv")
    feature_columns = _feature_columns_from_index(dataset_dir) or get_branch_site_feature_columns(rows)
    if not feature_columns:
        raise ValueError("no usable numeric branch-site feature columns found")
    warnings: List[str] = []
    X, y, metadata = _make_matrix(rows, feature_columns, warnings)
    train_mask = np.array([row["split"] == "train" for row in metadata])
    if not train_mask.any():
        raise ValueError("train split has no rows")
    feature_mean = X[train_mask].mean(axis=0)
    feature_std = X[train_mask].std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    X_std = (X - feature_mean) / feature_std
    rng = np.random.default_rng(config.seed)
    fit = _fit_logistic_regression(X_std[train_mask], y[train_mask], config, rng)
    warnings.extend(fit["warnings"])
    probs = _sigmoid(X_std @ fit["weights"] + fit["bias"])
    pred = (probs >= config.threshold).astype(np.int32)
    predictions = _prediction_rows(metadata, probs, pred)
    metrics = _all_metrics(metadata, y, probs, config.threshold)

    model_path = outdir / "branch_site_baseline_model.npz"
    meta_path = outdir / "branch_site_baseline_model_meta.json"
    predictions_path = outdir / "branch_site_baseline_predictions.tsv"
    metrics_path = outdir / "branch_site_baseline_metrics.json"
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
            "branch_site_baseline_version": BRANCH_BASELINE_VERSION,
            "branch_site_dataset_dir": str(dataset_dir),
            "feature_columns": feature_columns,
            "seed": config.seed,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "l2": config.l2,
            "positive_class_weight": config.positive_class_weight,
            "threshold": config.threshold,
            "train_rows": int(train_mask.sum()),
            "warnings": sorted(set(warnings)),
            "training_history": fit["training_history"],
            "note": "Branch-site NumPy baseline for research-alpha validation; not empirical inference.",
        },
    )
    write_tsv(predictions_path, predictions, PREDICTION_FIELDNAMES)
    _write_json(metrics_path, metrics | {"feature_columns": feature_columns})
    return {
        "status": "ok",
        "outdir": str(outdir),
        "model": str(model_path),
        "meta": str(meta_path),
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "metrics_by_split": metrics["metrics_by_split"],
        "warnings": sorted(set(warnings)),
    }


def validate_branch_site_baseline_dir(model_dir: str | Path) -> dict:
    """Validate branch-site baseline artifacts."""
    path = Path(model_dir)
    failures: List[str] = []
    warnings: List[str] = []
    for filename in (
        "branch_site_baseline_model.npz",
        "branch_site_baseline_model_meta.json",
        "branch_site_baseline_predictions.tsv",
        "branch_site_baseline_metrics.json",
    ):
        if not (path / filename).exists():
            failures.append(f"missing_file:{path / filename}")
    rows = []
    predictions_path = path / "branch_site_baseline_predictions.tsv"
    if predictions_path.exists():
        rows = read_tsv(predictions_path)
        for row in rows:
            try:
                prob = float(row.get("prob_positive", "nan"))
            except ValueError:
                failures.append(f"invalid_probability:{row.get('branch_site_id')}")
                continue
            if not 0 <= prob <= 1:
                failures.append(f"probability_out_of_range:{row.get('branch_site_id')}:{prob}")
            if row.get("y_branch_site") not in {"0", "1"}:
                failures.append(f"invalid_y_branch_site:{row.get('branch_site_id')}:{row.get('y_branch_site')}")
    if not rows:
        failures.append("no_predictions")
    return {
        "status": "fail" if failures else "ok",
        "n_predictions": len(rows),
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _feature_columns_from_index(dataset_dir: Path) -> List[str]:
    try:
        payload = json.loads((dataset_dir / "branch_site_dataset_index.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    columns = payload.get("feature_columns")
    return list(columns) if isinstance(columns, list) else []


def _make_matrix(rows: List[dict], feature_columns: List[str], warnings: List[str]) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    X = np.zeros((len(rows), len(feature_columns)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.float64)
    metadata: List[dict] = []
    for row_index, row in enumerate(rows):
        y[row_index] = _to_float(row.get("y_branch_site"), warnings, "y_branch_site")
        for column_index, column in enumerate(feature_columns):
            X[row_index, column_index] = _to_float(row.get(column), warnings, column)
        metadata.append(
            {
                "branch_site_id": row.get("branch_site_id", ""),
                "family_id": row.get("family_id", ""),
                "method": row.get("method", ""),
                "saturation_tier": row.get("saturation_tier", "unknown") or "unknown",
                "split": row.get("split", ""),
                "branch_id": row.get("branch_id", ""),
                "site_index_zero": row.get("site_index_zero", ""),
                "y_branch_site": int(y[row_index]),
                "y_site": int(_to_float(row.get("y_site"), warnings, "y_site")),
                "gene_label": int(_to_float(row.get("gene_label"), warnings, "gene_label")),
            }
        )
    return X, y, metadata


def _fit_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, config: BranchSiteBaselineConfig, rng: np.random.Generator) -> dict:
    n_rows, n_features = X_train.shape
    weights = rng.normal(loc=0.0, scale=0.01, size=n_features)
    bias = 0.0
    sample_weights, warnings = _class_weights(y_train, config)  # type: ignore[arg-type]
    history = []
    for epoch in range(1, config.epochs + 1):
        prob = _sigmoid(X_train @ weights + bias)
        errors = (prob - y_train) * sample_weights
        grad_w = (X_train.T @ errors) / n_rows + config.l2 * weights
        grad_b = float(errors.mean())
        weights -= config.learning_rate * grad_w
        bias -= config.learning_rate * grad_b
        if epoch % 10 == 0 or epoch == config.epochs:
            history.append({"epoch": epoch, "loss": _weighted_bce(y_train, prob, sample_weights, weights, config.l2)})
    return {"weights": weights, "bias": float(bias), "warnings": warnings, "training_history": history}


def _prediction_rows(metadata: List[dict], probs: np.ndarray, pred: np.ndarray) -> List[dict]:
    rows = []
    for index, row in enumerate(metadata):
        label = int(row["y_branch_site"])
        predicted = int(pred[index])
        rows.append({**row, "prob_positive": float(probs[index]), "pred_label": predicted, "correct": int(predicted == label)})
    return rows


def _all_metrics(metadata: List[dict], y: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    return {
        "branch_site_baseline_version": BRANCH_BASELINE_VERSION,
        "metrics_by_split": _metrics_by_field(metadata, y, probs, threshold, "split", True),
        "metrics_by_saturation_tier": _metrics_by_field(metadata, y, probs, threshold, "saturation_tier", True),
        "metrics_by_method": _metrics_by_field(metadata, y, probs, threshold, "method", True),
        "metrics_by_branch_id": _metrics_by_field(metadata, y, probs, threshold, "branch_id", False),
        "metrics_by_split_method": _metrics_by_fields(metadata, y, probs, threshold, ["split", "method"]),
    }


def _metrics_by_field(metadata: List[dict], y: np.ndarray, probs: np.ndarray, threshold: float, field: str, include_all: bool) -> Dict[str, dict]:
    result = {}
    for value in sorted({row.get(field, "") for row in metadata}):
        mask = np.array([row.get(field, "") == value for row in metadata])
        result[value or "unknown"] = _compute_binary_metrics(y[mask], probs[mask], threshold)
    if include_all:
        result["all"] = _compute_binary_metrics(y, probs, threshold)
    return result


def _metrics_by_fields(metadata: List[dict], y: np.ndarray, probs: np.ndarray, threshold: float, fields: List[str]) -> Dict[str, dict]:
    result = {}
    keys = sorted({tuple(row.get(field, "") for field in fields) for row in metadata})
    for key in keys:
        mask = np.array([tuple(row.get(field, "") for field in fields) == key for row in metadata])
        result["::".join(value or "unknown" for value in key)] = _compute_binary_metrics(y[mask], probs[mask], threshold)
    return result


def _to_float(value: object, warnings: List[str], column: str) -> float:
    try:
        if value in ("", None):
            raise ValueError
        return float(value)
    except (TypeError, ValueError):
        warnings.append(f"non_numeric_value_converted_to_zero:{column}")
        return 0.0


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
