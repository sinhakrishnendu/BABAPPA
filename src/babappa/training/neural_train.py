"""Small PyTorch smoke-training pipeline for BABAPPA tensors."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from babappa import __version__
from babappa.datasets.index import write_tsv
from babappa.models.baseline import compute_binary_metrics
from babappa.training.neural_data import (
    BabappaTensorDataset,
    NeuralDatasetConfig,
    collate_babappa_batch,
)
from babappa.training.neural_env import is_mps_available, mps_runtime_guidance, resolve_torch_device, safe_import_torch
from babappa.training.neural_model import (
    build_small_gene_classifier,
    count_parameters,
)

NEURAL_SMOKE_VERSION = __version__
NEURAL_PREDICTION_FIELDNAMES = [
    "family_id",
    "method",
    "split",
    "tensor_file",
    "gene_label",
    "prob_positive",
    "pred_label",
    "correct",
]
NEURAL_HISTORY_FIELDNAMES = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "val_auroc",
]
VALID_DEVICES = {"auto", "cpu", "cuda", "mps"}


@dataclass(frozen=True)
class NeuralTrainConfig:
    """Configuration for the minimal neural smoke-training loop."""

    dataset_dir: str
    outdir: str
    seed: int = 42
    device: str = "auto"
    methods: Optional[List[str]] = None
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    embedding_dim: int = 16
    hidden_dim: int = 32
    dropout: float = 0.1
    max_train_items: Optional[int] = None
    max_val_items: Optional[int] = None
    threshold: float = 0.5

    def __post_init__(self) -> None:
        dataset_path = Path(self.dataset_dir)
        out_path = Path(self.outdir)
        if not dataset_path.exists():
            raise ValueError(f"dataset_dir does not exist: {dataset_path}")
        if not (dataset_path / "dataset_index.json").exists():
            raise ValueError(f"dataset_dir is missing dataset_index.json: {dataset_path}")
        if not (dataset_path / "splits.tsv").exists():
            raise ValueError(f"dataset_dir is missing splits.tsv: {dataset_path}")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if self.device not in VALID_DEVICES:
            allowed = ", ".join(sorted(VALID_DEVICES))
            raise ValueError(f"device must be one of: {allowed}")
        if self.max_train_items is not None and self.max_train_items <= 0:
            raise ValueError("max_train_items must be positive when provided")
        if self.max_val_items is not None and self.max_val_items <= 0:
            raise ValueError("max_val_items must be positive when provided")
        if self.methods is not None:
            resolved_methods = [method for method in self.methods if method]
            object.__setattr__(self, "methods", resolved_methods or None)
        out_path.mkdir(parents=True, exist_ok=True)


def set_random_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds when PyTorch is available."""
    random.seed(seed)
    np.random.seed(seed)
    torch, _error = safe_import_torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    """Resolve a requested device string to a usable PyTorch device name."""
    torch, _error = safe_import_torch()
    if torch is None:
        raise RuntimeError(
            "PyTorch is not available. Install torch or use an environment containing torch."
        ) from None
    return resolve_torch_device(torch, device)


def train_neural_smoke_model(config: NeuralTrainConfig) -> dict:
    """Train the minimal gene-level neural smoke classifier."""
    torch, _error = safe_import_torch()
    if torch is None:
        raise RuntimeError(
            "PyTorch is not available. Install torch or use an environment containing torch."
        ) from None

    set_random_seeds(config.seed)
    device_used = resolve_device(config.device)
    outdir = Path(config.outdir)
    warnings: List[str] = []

    train_dataset = BabappaTensorDataset(
        NeuralDatasetConfig(
            dataset_dir=config.dataset_dir,
            split="train",
            methods=config.methods,
            max_items=config.max_train_items,
            require_torch=True,
        )
    )
    val_dataset = BabappaTensorDataset(
        NeuralDatasetConfig(
            dataset_dir=config.dataset_dir,
            split="val",
            methods=config.methods,
            max_items=config.max_val_items,
            require_torch=True,
        )
    )
    if len(train_dataset) == 0:
        raise ValueError("train split has no rows")
    if len(val_dataset) == 0:
        warnings.append("empty_val_split")

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_babappa_batch,
        generator=generator,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_babappa_batch,
    )
    vocab_size = _infer_vocab_size(train_dataset)
    try:
        model = build_small_gene_classifier(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        ).to(device_used)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        history = []

        for epoch in range(1, config.epochs + 1):
            model.train()
            train_losses = []
            for batch in train_loader:
                X = batch["X"].to(device_used)
                y = batch["y"].to(device_used)
                optimizer.zero_grad(set_to_none=True)
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))

            val_loss = _evaluate_loss(model, val_loader, criterion, device_used)
            val_predictions = predict_neural_dataset(
                model, val_loader, device_used, threshold=config.threshold
            )
            val_metrics = _metrics_from_prediction_rows(val_predictions, config.threshold)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(train_losses)) if train_losses else None,
                    "val_loss": val_loss,
                    "val_accuracy": val_metrics.get("accuracy"),
                    "val_auroc": val_metrics.get("auroc"),
                }
            )
    except RuntimeError as exc:
        if device_used == "mps":
            raise RuntimeError(mps_runtime_guidance(exc)) from exc
        raise

    train_eval_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_babappa_batch,
    )
    val_eval_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_babappa_batch,
    )
    prediction_rows = []
    prediction_rows.extend(
        predict_neural_dataset(
            model, train_eval_loader, device_used, threshold=config.threshold
        )
    )
    prediction_rows.extend(
        predict_neural_dataset(
            model, val_eval_loader, device_used, threshold=config.threshold
        )
    )
    metrics_by_split = _metrics_by_split(prediction_rows, config.threshold)

    checkpoint_path = outdir / "neural_smoke_checkpoint.pt"
    meta_path = outdir / "neural_smoke_model_meta.json"
    history_path = outdir / "neural_smoke_history.tsv"
    predictions_path = outdir / "neural_smoke_predictions.tsv"
    metrics_path = outdir / "neural_smoke_metrics.json"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "vocab_size": vocab_size,
        "model_class": "SmallGeneClassifier",
        "note": "Gene-level neural smoke model, not final BABAPPA branch-site model",
    }
    torch.save(checkpoint, checkpoint_path)
    meta = {
        "neural_smoke_version": NEURAL_SMOKE_VERSION,
        "dataset_dir": config.dataset_dir,
        "seed": config.seed,
        "device_requested": config.device,
        "device_used": device_used,
        "methods": list(config.methods or []),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "threshold": config.threshold,
        "vocab_size": vocab_size,
        "parameter_count": count_parameters(model),
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "warnings": sorted(set(warnings)),
        "note": "Gene-level neural smoke model, not final BABAPPA branch-site model",
    }
    _write_json(meta_path, meta)
    write_tsv(history_path, history, NEURAL_HISTORY_FIELDNAMES)
    write_tsv(predictions_path, prediction_rows, NEURAL_PREDICTION_FIELDNAMES)
    _write_json(
        metrics_path,
        {
            "metrics_by_split": metrics_by_split,
            "model_files": {
                "checkpoint": str(checkpoint_path),
                "meta": str(meta_path),
                "history": str(history_path),
                "predictions": str(predictions_path),
                "metrics": str(metrics_path),
            },
            "note": "Gene-level neural smoke metrics, not final BABAPPA branch-site performance",
        },
    )

    return {
        "status": "ok",
        "outdir": str(outdir),
        "checkpoint": str(checkpoint_path),
        "meta": str(meta_path),
        "history": str(history_path),
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "device_used": device_used,
        "warnings": sorted(set(warnings)),
    }


def predict_neural_dataset(model, dataloader, device: str, threshold: float = 0.5) -> List[dict]:
    """Predict probabilities and labels for a neural dataloader."""
    torch, _error = safe_import_torch()
    if torch is None:
        raise RuntimeError(
            "PyTorch is not available. Install torch or use an environment containing torch."
        ) from None

    rows = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            logits = model(X)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels = y.detach().cpu().numpy().astype(np.int32)
            pred_labels = (probs >= threshold).astype(np.int32)
            for index, prob in enumerate(probs):
                gene_label = int(labels[index])
                pred_label = int(pred_labels[index])
                rows.append(
                    {
                        "family_id": batch["family_id"][index],
                        "method": batch["method"][index],
                        "split": batch["split"][index],
                        "tensor_file": batch["tensor_file"][index],
                        "gene_label": gene_label,
                        "prob_positive": float(prob),
                        "pred_label": pred_label,
                        "correct": int(pred_label == gene_label),
                    }
                )
    return rows


def _infer_vocab_size(dataset: BabappaTensorDataset, default: int = 128) -> int:
    max_codon_id = 0
    for index in range(min(len(dataset), 32)):
        item = dataset[index]
        max_codon_id = max(max_codon_id, int(item["X"][..., 0].max().item()))
    return max(default, max_codon_id + 1)


def _evaluate_loss(model, dataloader, criterion, device: str) -> Optional[float]:
    torch, _error = safe_import_torch()
    if torch is None:
        return None
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            logits = model(X)
            losses.append(float(criterion(logits, y).detach().cpu().item()))
    if not losses:
        return None
    return float(np.mean(losses))


def _metrics_by_split(prediction_rows: List[dict], threshold: float) -> Dict[str, dict]:
    metrics = {}
    for split in ["train", "val"]:
        split_rows = [row for row in prediction_rows if row["split"] == split]
        metrics[split] = _metrics_from_prediction_rows(split_rows, threshold)
    metrics["all"] = _metrics_from_prediction_rows(prediction_rows, threshold)
    return metrics


def _metrics_from_prediction_rows(rows: List[dict], threshold: float) -> dict:
    y_true = np.asarray([int(row["gene_label"]) for row in rows], dtype=np.int32)
    y_prob = np.asarray([float(row["prob_positive"]) for row in rows], dtype=np.float64)
    return compute_binary_metrics(y_true, y_prob, threshold)


def _mps_available(torch) -> bool:
    return is_mps_available(torch)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
