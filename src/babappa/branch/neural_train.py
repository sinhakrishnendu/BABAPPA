"""Lightweight branch-site neural trainer."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from babappa import __version__
from babappa.branch.baseline import PREDICTION_FIELDNAMES, _all_metrics, _feature_columns_from_index, _make_matrix
from babappa.branch.feature_policy import columns_for_policy, get_branch_feature_policy
from babappa.datasets.index import read_tsv, write_tsv
from babappa.site.baseline import _compute_binary_metrics
from babappa.site.neural_model import SiteMLPClassifier, count_parameters
from babappa.training.neural_env import VALID_DEVICES, mps_runtime_guidance, resolve_torch_device, safe_import_torch

BRANCH_NEURAL_VERSION = __version__
VALID_POSITIVE_CLASS_WEIGHT = {"auto", "none"}
VALID_MONITOR_METRICS = {"val_loss", "val_auroc"}
VALID_SPLITS = ["train", "val", "calib", "test"]
HISTORY_FIELDNAMES = ["epoch", "train_loss", "val_loss", "val_auroc"]


@dataclass(frozen=True)
class BranchSiteNeuralTrainConfig:
    branch_site_dataset_dir: str
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
    feature_policy: str = "full_context"
    threads: int = 0

    def __post_init__(self) -> None:
        dataset_dir = Path(self.branch_site_dataset_dir)
        if not dataset_dir.exists():
            raise ValueError(f"branch_site_dataset_dir does not exist: {dataset_dir}")
        for filename in ("branch_site_dataset_index.json", "branch_site_features.tsv", "branch_site_splits.tsv"):
            if not (dataset_dir / filename).exists():
                raise ValueError(f"branch_site_dataset_dir is missing {filename}: {dataset_dir}")
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
            raise ValueError("positive_class_weight must be auto or none")
        if self.monitor_metric not in VALID_MONITOR_METRICS:
            raise ValueError("monitor_metric must be val_loss or val_auroc")
        if self.device not in VALID_DEVICES:
            allowed = ", ".join(sorted(VALID_DEVICES))
            raise ValueError(f"device must be one of: {allowed}")
        if self.threads < 0:
            raise ValueError("threads must be >= 0")
        policy = get_branch_feature_policy(self.feature_policy)
        object.__setattr__(self, "feature_policy", policy.name)
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def train_branch_site_neural_model(config: BranchSiteNeuralTrainConfig) -> dict:
    """Train a small branch-site MLP classifier."""
    torch, error = safe_import_torch()
    if torch is None:
        raise RuntimeError("PyTorch is not available. Install torch or use an environment containing torch.") from error

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    thread_report = _configure_torch_threads(torch, config.threads)
    device = _resolve_device(torch, config.device)
    outdir = Path(config.outdir)
    split_data, feature_columns, feature_mean, feature_std = _load_split_data(config)
    X_train, y_train, _ = split_data["train"]
    if X_train.shape[0] == 0:
        raise ValueError("train split has no rows")
    if not feature_columns:
        raise ValueError("branch-site dataset has no usable feature columns")

    try:
        model = SiteMLPClassifier(input_dim=len(feature_columns), hidden_dim=config.hidden_dim, dropout=config.dropout).to(device)
        pos_weight = _pos_weight(y_train, config)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=None if pos_weight is None else torch.tensor(pos_weight, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        train_loader = _loader(torch, X_train, y_train, config.batch_size, shuffle=True)

        history: List[dict] = []
        best_epoch = 0
        best_value = None
        best_state = None
        stopped_early = False
        epochs_without_improvement = 0
        for epoch in range(1, config.epochs + 1):
            model.train()
            losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
            val_loss, val_auroc = _validation_metrics(torch, model, split_data["val"], loss_fn, device)
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None, "val_loss": val_loss, "val_auroc": val_auroc}
            history.append(row)
            monitor_value = row[config.monitor_metric]
            if _is_improved(config.monitor_metric, monitor_value, best_value):
                best_value = monitor_value
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                stopped_early = True
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        elif history:
            best_epoch = len(history)

        predictions, y_all, prob_all, metadata_all = _predict_all_splits(torch, model, split_data, config, device)
    except RuntimeError as exc:
        if str(device) == "mps":
            raise RuntimeError(mps_runtime_guidance(exc)) from exc
        raise
    metrics = _all_metrics(metadata_all, y_all, prob_all, config.threshold)
    checkpoint_path = outdir / "branch_site_neural_checkpoint.pt"
    meta_path = outdir / "branch_site_neural_model_meta.json"
    predictions_path = outdir / "branch_site_neural_predictions.tsv"
    metrics_path = outdir / "branch_site_neural_metrics.json"
    history_path = outdir / "branch_site_neural_history.tsv"
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
            "branch_site_neural_version": BRANCH_NEURAL_VERSION,
            "branch_site_dataset_dir": str(Path(config.branch_site_dataset_dir)),
            "feature_columns": feature_columns,
            "n_features": len(feature_columns),
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "positive_class_weight": config.positive_class_weight,
            "resolved_pos_weight": pos_weight,
            "feature_policy": config.feature_policy,
            "requested_threads": config.threads,
            "torch_threads": thread_report,
            "seed": config.seed,
            "device": str(device),
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "monitor_metric": config.monitor_metric,
            "parameter_count": count_parameters(model),
            "split_rows": {split: int(split_data[split][1].shape[0]) for split in VALID_SPLITS},
            "note": "Lightweight branch-context MLP for research-alpha validation; not empirical branch-site inference.",
        },
    )
    write_tsv(history_path, history, HISTORY_FIELDNAMES)
    write_tsv(predictions_path, predictions, PREDICTION_FIELDNAMES)
    _write_json(metrics_path, metrics)
    return {"status": "ok", "outdir": str(outdir), "checkpoint": str(checkpoint_path), "meta": str(meta_path), "history": str(history_path), "predictions": str(predictions_path), "metrics": str(metrics_path), "metrics_by_split": metrics["metrics_by_split"], "best_epoch": best_epoch, "threads": thread_report.get("num_threads")}


def validate_branch_site_neural_dir(model_dir: str | Path) -> dict:
    path = Path(model_dir)
    failures: List[str] = []
    warnings: List[str] = []
    for filename in ("branch_site_neural_checkpoint.pt", "branch_site_neural_model_meta.json", "branch_site_neural_predictions.tsv", "branch_site_neural_metrics.json", "branch_site_neural_history.tsv"):
        if not (path / filename).exists():
            failures.append(f"missing_file:{path / filename}")
    rows = []
    predictions_path = path / "branch_site_neural_predictions.tsv"
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
    return {"status": "fail" if failures else "ok", "n_predictions": len(rows), "n_fail": len(failures), "n_warning": len(warnings), "failures": failures, "warnings": warnings}


def load_branch_site_feature_arrays(
    branch_site_dataset_dir: str | Path,
    split: str,
    max_items: Optional[int] = None,
    seed: int = 42,
    feature_policy: str = "full_context",
):
    dataset_dir = Path(branch_site_dataset_dir)
    rows = [row for row in read_tsv(dataset_dir / "branch_site_features.tsv") if row.get("split") == split]
    if max_items is not None and len(rows) > max_items:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_items]
    feature_columns = columns_for_policy(_feature_columns_from_index(dataset_dir), feature_policy)
    warnings: List[str] = []
    X, y, metadata = _make_matrix(rows, feature_columns, warnings)
    return X.astype(np.float32), y.astype(np.float32), metadata, feature_columns


def _load_split_data(config: BranchSiteNeuralTrainConfig):
    max_items = {"train": config.max_train_items, "val": config.max_val_items, "calib": config.max_calib_items, "test": config.max_test_items}
    X_train, y_train, train_meta, feature_columns = load_branch_site_feature_arrays(
        config.branch_site_dataset_dir,
        "train",
        config.max_train_items,
        config.seed,
        config.feature_policy,
    )
    feature_mean = X_train.mean(axis=0) if X_train.size else np.zeros(len(feature_columns), dtype=np.float32)
    feature_std = X_train.std(axis=0) if X_train.size else np.ones(len(feature_columns), dtype=np.float32)
    feature_std = np.where(feature_std == 0, 1.0, feature_std).astype(np.float32)
    feature_mean = feature_mean.astype(np.float32)
    data = {"train": (_standardize(X_train, feature_mean, feature_std), y_train, train_meta)}
    for split in ["val", "calib", "test"]:
        X, y, meta, _columns = load_branch_site_feature_arrays(
            config.branch_site_dataset_dir,
            split,
            max_items[split],
            config.seed,
            config.feature_policy,
        )
        data[split] = (_standardize(X, feature_mean, feature_std), y, meta)
    return data, feature_columns, feature_mean, feature_std


def _standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32) if X.size else X.astype(np.float32)


def _loader(torch, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _configure_torch_threads(torch, requested_threads: int) -> dict:
    report = {
        "requested_threads": int(requested_threads),
        "num_threads": None,
        "num_interop_threads": None,
        "warnings": [],
    }
    if requested_threads > 0:
        try:
            torch.set_num_threads(int(requested_threads))
        except Exception as exc:  # pragma: no cover - torch build specific
            report["warnings"].append(f"set_num_threads_failed:{exc}")
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(max(1, min(4, int(requested_threads))))
            except Exception as exc:  # pragma: no cover - torch only allows this once
                report["warnings"].append(f"set_num_interop_threads_failed:{exc}")
    try:
        report["num_threads"] = int(torch.get_num_threads())
    except Exception as exc:  # pragma: no cover - torch build specific
        report["warnings"].append(f"get_num_threads_failed:{exc}")
    if hasattr(torch, "get_num_interop_threads"):
        try:
            report["num_interop_threads"] = int(torch.get_num_interop_threads())
        except Exception as exc:  # pragma: no cover - torch build specific
            report["warnings"].append(f"get_num_interop_threads_failed:{exc}")
    return report


def _validation_metrics(torch, model, split_tuple, loss_fn, device) -> Tuple[Optional[float], Optional[float]]:
    X, y, _ = split_tuple
    if X.shape[0] == 0:
        return None, None
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X).to(device)
        target = torch.from_numpy(y).to(device)
        logits = model(x)
        loss = float(loss_fn(logits, target).detach().cpu().item())
        prob = torch.sigmoid(logits).detach().cpu().numpy()
    return loss, _compute_binary_metrics(y.astype(np.int32), prob, threshold=0.5).get("auroc")


def _predict_all_splits(torch, model, split_data: dict, config: BranchSiteNeuralTrainConfig, device):
    predictions = []
    y_all = []
    prob_all = []
    metadata_all = []
    model.eval()
    with torch.no_grad():
        for split in VALID_SPLITS:
            X, y, metadata = split_data[split]
            if X.shape[0] == 0:
                continue
            logits = model(torch.from_numpy(X).to(device))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            pred = (probs >= config.threshold).astype(np.int32)
            for index, row in enumerate(metadata):
                label = int(y[index])
                predicted = int(pred[index])
                predictions.append({**row, "prob_positive": float(probs[index]), "pred_label": predicted, "correct": int(predicted == label)})
            y_all.extend(y.tolist())
            prob_all.extend(probs.tolist())
            metadata_all.extend(metadata)
    return predictions, np.array(y_all, dtype=np.float64), np.array(prob_all, dtype=np.float64), metadata_all


def _resolve_device(torch, requested: str):
    return torch.device(resolve_torch_device(torch, requested))


def _pos_weight(y_train: np.ndarray, config: BranchSiteNeuralTrainConfig) -> Optional[float]:
    if config.positive_class_weight == "none":
        return None
    positives = float(y_train.sum())
    negatives = float(y_train.size - positives)
    if positives == 0 or negatives == 0:
        return None
    return negatives / positives


def _is_improved(metric: str, value, best) -> bool:
    if value is None:
        return best is None
    if best is None:
        return True
    if metric == "val_loss":
        return value < best
    return value > best


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
