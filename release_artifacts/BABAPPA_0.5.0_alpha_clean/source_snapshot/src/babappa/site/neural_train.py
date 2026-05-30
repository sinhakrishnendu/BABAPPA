"""Training loop for site-level neural classifiers."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from babappa import __version__
from babappa.datasets.index import write_tsv
from babappa.site.baseline import _compute_binary_metrics
from babappa.site.neural_data import SiteNeuralDatasetConfig, load_site_feature_arrays
from babappa.site.neural_model import SiteMLPClassifier, count_parameters
from babappa.training.neural_env import VALID_DEVICES, mps_runtime_guidance, resolve_torch_device, safe_import_torch

SITE_NEURAL_VERSION = __version__
VALID_POSITIVE_CLASS_WEIGHT = {"auto", "none"}
VALID_MONITOR_METRICS = {"val_loss", "val_auroc"}
VALID_SPLITS = ["train", "val", "calib", "test"]
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
HISTORY_FIELDNAMES = ["epoch", "train_loss", "val_loss", "val_auroc"]


@dataclass(frozen=True)
class SiteNeuralTrainConfig:
    """Configuration for site-level neural training."""

    site_dataset_dir: str
    outdir: str
    seed: int = 42
    device: str = "auto"
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    hidden_dim: int = 64
    dropout: float = 0.1
    positive_class_weight: str = "auto"
    threshold: float = 0.5
    early_stopping_patience: int = 8
    monitor_metric: str = "val_auroc"
    max_train_items: Optional[int] = None
    max_val_items: Optional[int] = None
    max_calib_items: Optional[int] = None
    max_test_items: Optional[int] = None

    def __post_init__(self) -> None:
        dataset_dir = Path(self.site_dataset_dir)
        out_path = Path(self.outdir)
        if not dataset_dir.exists():
            raise ValueError(f"site_dataset_dir does not exist: {dataset_dir}")
        for filename in ("site_dataset_index.json", "site_features.tsv", "site_splits.tsv"):
            if not (dataset_dir / filename).exists():
                raise ValueError(f"site_dataset_dir is missing {filename}: {dataset_dir}")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if self.positive_class_weight not in VALID_POSITIVE_CLASS_WEIGHT:
            allowed = ", ".join(sorted(VALID_POSITIVE_CLASS_WEIGHT))
            raise ValueError(f"positive_class_weight must be one of: {allowed}")
        if self.monitor_metric not in VALID_MONITOR_METRICS:
            allowed = ", ".join(sorted(VALID_MONITOR_METRICS))
            raise ValueError(f"monitor_metric must be one of: {allowed}")
        if self.device not in VALID_DEVICES:
            allowed = ", ".join(sorted(VALID_DEVICES))
            raise ValueError(f"device must be one of: {allowed}")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")
        out_path.mkdir(parents=True, exist_ok=True)


def train_site_neural_model(config: SiteNeuralTrainConfig) -> dict:
    """Train a site-level neural classifier and write artifacts."""
    torch, error = safe_import_torch()
    if torch is None:
        raise RuntimeError(
            "PyTorch is not available. Install torch or use an environment containing torch."
        ) from error

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _resolve_device(torch, config.device)
    outdir = Path(config.outdir)

    split_data, feature_columns, feature_mean, feature_std = _load_split_data(config)
    X_train, y_train, _train_meta = split_data["train"]
    if X_train.shape[0] == 0:
        raise ValueError("train split has no rows")
    if not feature_columns:
        raise ValueError("site dataset has no usable feature columns")

    try:
        model = SiteMLPClassifier(
            input_dim=len(feature_columns),
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        ).to(device)
        pos_weight = _pos_weight(y_train, config)
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=None if pos_weight is None else torch.tensor(pos_weight, device=device)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        train_loader = _loader(torch, X_train, y_train, config.batch_size, shuffle=True)

        history: List[dict] = []
        best_epoch = 0
        best_value = None
        best_state = None
        epochs_without_improvement = 0
        stopped_early = False
        for epoch in range(1, config.epochs + 1):
            model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))
            train_loss = float(np.mean(train_losses)) if train_losses else None
            val_loss, val_auroc = _validation_metrics(torch, model, split_data["val"], loss_fn, device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auroc": val_auroc,
            }
            history.append(row)
            monitor_value = row[config.monitor_metric]
            improved = _is_improved(config.monitor_metric, monitor_value, best_value)
            if improved:
                best_value = monitor_value
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                stopped_early = True
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            best_epoch = len(history)

        predictions, y_all, prob_all, metadata_all = _predict_all_splits(
            torch, model, split_data, config, device
        )
    except RuntimeError as exc:
        if str(device) == "mps":
            raise RuntimeError(mps_runtime_guidance(exc)) from exc
        raise
    metrics = _metrics(metadata_all, y_all, prob_all, config.threshold)

    checkpoint_path = outdir / "site_neural_checkpoint.pt"
    meta_path = outdir / "site_neural_model_meta.json"
    history_path = outdir / "site_neural_history.tsv"
    predictions_path = outdir / "site_neural_predictions.tsv"
    metrics_path = outdir / "site_neural_metrics.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "feature_columns": feature_columns,
            "config": asdict(config),
        },
        checkpoint_path,
    )
    _write_json(
        meta_path,
        {
            "site_neural_version": SITE_NEURAL_VERSION,
            "site_dataset_dir": str(Path(config.site_dataset_dir)),
            "feature_columns": feature_columns,
            "n_features": len(feature_columns),
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "positive_class_weight": config.positive_class_weight,
            "resolved_pos_weight": pos_weight,
            "seed": config.seed,
            "device": str(device),
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "monitor_metric": config.monitor_metric,
            "parameter_count": count_parameters(model),
            "split_rows": {
                split: int(split_data[split][1].shape[0])
                for split in VALID_SPLITS
            },
            "note": "Site-level oracle-supervised classifier; not empirical branch-site inference.",
        },
    )
    write_tsv(history_path, history, HISTORY_FIELDNAMES)
    write_tsv(predictions_path, predictions, PREDICTION_FIELDNAMES)
    _write_json(metrics_path, metrics)
    return {
        "status": "ok",
        "outdir": str(outdir),
        "checkpoint": str(checkpoint_path),
        "meta": str(meta_path),
        "history": str(history_path),
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "metrics_by_split": metrics["metrics_by_split"],
        "best_epoch": best_epoch,
    }


def _load_split_data(config: SiteNeuralTrainConfig):
    max_items = {
        "train": config.max_train_items,
        "val": config.max_val_items,
        "calib": config.max_calib_items,
        "test": config.max_test_items,
    }
    train_config = SiteNeuralDatasetConfig(
        site_dataset_dir=config.site_dataset_dir,
        split="train",
        max_items=config.max_train_items,
        seed=config.seed,
    )
    X_train, y_train, train_meta, feature_columns = load_site_feature_arrays(train_config)
    feature_mean = X_train.mean(axis=0) if X_train.size else np.zeros(len(feature_columns), dtype=np.float32)
    feature_std = X_train.std(axis=0) if X_train.size else np.ones(len(feature_columns), dtype=np.float32)
    feature_std = np.where(feature_std == 0, 1.0, feature_std).astype(np.float32)
    feature_mean = feature_mean.astype(np.float32)
    data = {
        "train": (
            _standardize(X_train, feature_mean, feature_std),
            y_train,
            train_meta,
        )
    }
    for split in ["val", "calib", "test"]:
        X, y, meta, _columns = load_site_feature_arrays(
            SiteNeuralDatasetConfig(
                site_dataset_dir=config.site_dataset_dir,
                split=split,
                max_items=max_items[split],
                seed=config.seed,
            ),
            feature_columns=feature_columns,
        )
        data[split] = (_standardize(X, feature_mean, feature_std), y, meta)
    return data, feature_columns, feature_mean, feature_std


def _standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X.astype(np.float32)
    return ((X - mean) / std).astype(np.float32)


def _resolve_device(torch, requested: str):
    return torch.device(resolve_torch_device(torch, requested))


def _pos_weight(y_train: np.ndarray, config: SiteNeuralTrainConfig) -> Optional[float]:
    if config.positive_class_weight == "none":
        return None
    positives = int(y_train.sum())
    negatives = int(y_train.size - positives)
    if positives == 0 or negatives == 0:
        return None
    return float(negatives / positives)


def _loader(torch, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _validation_metrics(torch, model, data, loss_fn, device) -> Tuple[Optional[float], Optional[float]]:
    X, y, _metadata = data
    if X.shape[0] == 0:
        return None, None
    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=device)
        logits = model(x_tensor)
        loss = float(loss_fn(logits, y_tensor).detach().cpu().item())
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    metrics = _compute_binary_metrics(y, probs, threshold=0.5)
    return loss, metrics.get("auroc")


def _is_improved(metric_name: str, value: Optional[float], best: Optional[float]) -> bool:
    if value is None:
        return best is None
    if best is None:
        return True
    if metric_name == "val_loss":
        return value < best
    return value > best


def _predict_all_splits(torch, model, split_data, config, device):
    rows: List[dict] = []
    y_all_parts = []
    prob_all_parts = []
    metadata_all: List[dict] = []
    model.eval()
    for split in VALID_SPLITS:
        X, y, metadata = split_data[split]
        if X.shape[0] == 0:
            continue
        probs = _predict_probs(torch, model, X, config.batch_size, device)
        preds = (probs >= config.threshold).astype(np.int32)
        for index, meta in enumerate(metadata):
            y_value = int(y[index])
            pred = int(preds[index])
            rows.append(
                {
                    "site_id": meta["site_id"],
                    "family_id": meta["family_id"],
                    "method": meta["method"],
                    "saturation_tier": meta["saturation_tier"],
                    "split": meta["split"],
                    "site_index_zero": meta["site_index_zero"],
                    "y_site": y_value,
                    "prob_positive": float(probs[index]),
                    "pred_label": pred,
                    "correct": int(pred == y_value),
                }
            )
        y_all_parts.append(y)
        prob_all_parts.append(probs)
        metadata_all.extend(metadata)
    y_all = np.concatenate(y_all_parts) if y_all_parts else np.array([], dtype=np.float32)
    prob_all = np.concatenate(prob_all_parts) if prob_all_parts else np.array([], dtype=np.float32)
    return rows, y_all, prob_all, metadata_all


def _predict_probs(torch, model, X: np.ndarray, batch_size: int, device) -> np.ndarray:
    probs = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            x_tensor = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            logits = model(x_tensor)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs).astype(np.float64) if probs else np.array([], dtype=np.float64)


def _metrics(metadata: List[dict], y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    return {
        "site_neural_version": SITE_NEURAL_VERSION,
        "metrics_by_split": _metrics_by_field(metadata, y, prob, threshold, "split", include_all=True),
        "metrics_by_saturation_tier": _metrics_by_field(metadata, y, prob, threshold, "saturation_tier", include_all=True),
        "metrics_by_method": _metrics_by_field(metadata, y, prob, threshold, "method", include_all=True),
        "metrics_by_split_saturation_tier": _metrics_by_fields(metadata, y, prob, threshold, ["split", "saturation_tier"]),
        "metrics_by_split_method": _metrics_by_fields(metadata, y, prob, threshold, ["split", "method"]),
    }


def _metrics_by_field(
    metadata: List[dict],
    y: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    field: str,
    include_all: bool,
) -> Dict[str, dict]:
    result = {}
    for value in sorted({row.get(field, "") for row in metadata}):
        mask = np.array([row.get(field, "") == value for row in metadata])
        result[value or "unknown"] = _compute_binary_metrics(y[mask], prob[mask], threshold)
    if include_all:
        result["all"] = _compute_binary_metrics(y, prob, threshold)
    return result


def _metrics_by_fields(
    metadata: List[dict], y: np.ndarray, prob: np.ndarray, threshold: float, fields: List[str]
) -> Dict[str, dict]:
    result = {}
    keys = sorted({tuple(row.get(field, "") for field in fields) for row in metadata})
    for key in keys:
        mask = np.array([tuple(row.get(field, "") for field in fields) == key for row in metadata])
        result["::".join(value or "unknown" for value in key)] = _compute_binary_metrics(y[mask], prob[mask], threshold)
    return result


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
