"""Scale-ready gene-level neural training pipeline for BABAPPA."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from babappa import __version__
from babappa.datasets.index import write_tsv
from babappa.models.baseline import compute_binary_metrics
from babappa.training.neural_data import (
    BabappaTensorDataset,
    NeuralDatasetConfig,
    SATURATION_TIER_TO_ID,
    collate_babappa_batch,
    saturation_tier_to_id,
)
from babappa.training.neural_env import safe_import_torch
from babappa.training.losses import VALID_LOSS_MODES, combined_loss
from babappa.training.neural_model import build_gene_classifier, count_parameters
from babappa.training.neural_train import resolve_device, set_random_seeds

NEURAL_TRAINER_VERSION = __version__
NEURAL_FULL_PREDICTION_FIELDNAMES = [
    "family_id",
    "method",
    "split",
    "tensor_file",
    "gene_label",
    "saturation_tier",
    "saturation_id",
    "prob_positive",
    "pred_label",
    "correct",
]
NEURAL_FULL_HISTORY_FIELDNAMES = [
    "epoch",
    "train_loss",
    "train_accuracy",
    "train_auroc",
    "val_loss",
    "val_accuracy",
    "val_auroc",
    "monitor_metric",
    "monitor_value",
    "is_best",
    "seconds_elapsed",
]
VALID_DEVICES = {"auto", "cpu", "cuda", "mps"}
VALID_MONITOR_METRICS = {"val_loss", "val_auroc", "val_accuracy"}
VALID_ARCHITECTURES = {
    "small",
    "contrastive",
    "saturation_aware",
    "site_attention",
    "site_attention_saturation",
}
VALID_POSITIVE_CLASS_WEIGHT = {"none", "auto"}
VALID_GROUP_WEIGHTING = {"none", "saturation_inverse_frequency"}
VALID_SAMPLERS = {"none", "saturation_balanced"}
PRESET_OVERRIDE_DEFAULTS = {
    "architecture": "saturation_aware",
    "positive_class_weight": "auto",
    "group_weighting": "none",
    "sampler": "none",
    "loss_mode": "bce_rank",
    "rank_weight": 0.2,
    "focal_gamma": 2.0,
}
TRAINING_PRESETS = {
    "contrastive_v2": {
        "architecture": "contrastive",
        "positive_class_weight": "auto",
        "group_weighting": "none",
        "sampler": "none",
        "loss_mode": "bce",
        "rank_weight": 0.0,
        "focal_gamma": 2.0,
    },
    "saturation_embed_only": {
        "architecture": "saturation_aware",
        "positive_class_weight": "auto",
        "group_weighting": "none",
        "sampler": "none",
        "loss_mode": "bce",
        "rank_weight": 0.0,
        "focal_gamma": 2.0,
    },
    "saturation_group_weight_only": {
        "architecture": "contrastive",
        "positive_class_weight": "auto",
        "group_weighting": "saturation_inverse_frequency",
        "sampler": "none",
        "loss_mode": "bce",
        "rank_weight": 0.0,
        "focal_gamma": 2.0,
    },
    "saturation_sampler_only": {
        "architecture": "contrastive",
        "positive_class_weight": "auto",
        "group_weighting": "none",
        "sampler": "saturation_balanced",
        "loss_mode": "bce",
        "rank_weight": 0.0,
        "focal_gamma": 2.0,
    },
    "saturation_full_v3": {
        "architecture": "saturation_aware",
        "positive_class_weight": "auto",
        "group_weighting": "saturation_inverse_frequency",
        "sampler": "saturation_balanced",
        "loss_mode": "bce",
        "rank_weight": 0.0,
        "focal_gamma": 2.0,
    },
    "contrastive_class_weighted": {
        "architecture": "contrastive",
        "positive_class_weight": "auto",
        "group_weighting": "none",
        "sampler": "none",
        "loss_mode": "bce",
        "rank_weight": 0.0,
        "focal_gamma": 2.0,
    },
    "contrastive_unweighted": {
        "architecture": "contrastive",
        "positive_class_weight": "none",
        "group_weighting": "none",
        "sampler": "none",
        "loss_mode": "bce",
        "rank_weight": 0.0,
        "focal_gamma": 2.0,
    },
    "site_attention_ranked": {
        "architecture": "site_attention",
        "positive_class_weight": "auto",
        "group_weighting": "none",
        "sampler": "none",
        "loss_mode": "bce_rank",
        "rank_weight": 0.2,
        "focal_gamma": 2.0,
    },
    "site_attention_focal_ranked": {
        "architecture": "site_attention",
        "positive_class_weight": "auto",
        "group_weighting": "none",
        "sampler": "none",
        "loss_mode": "focal_rank",
        "rank_weight": 0.2,
        "focal_gamma": 2.0,
    },
    "site_attention_saturation_ranked": {
        "architecture": "site_attention_saturation",
        "positive_class_weight": "auto",
        "group_weighting": "none",
        "sampler": "none",
        "loss_mode": "bce_rank",
        "rank_weight": 0.2,
        "focal_gamma": 2.0,
    },
    "contrastive_ranked": {
        "architecture": "contrastive",
        "positive_class_weight": "auto",
        "group_weighting": "none",
        "sampler": "none",
        "loss_mode": "bce_rank",
        "rank_weight": 0.2,
        "focal_gamma": 2.0,
    },
}


@dataclass(frozen=True)
class NeuralFullTrainConfig:
    """Configuration for scale-ready gene-level neural training."""

    dataset_dir: str
    outdir: str
    seed: int = 42
    device: str = "auto"
    methods: Optional[List[str]] = None
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    embedding_dim: int = 32
    hidden_dim: int = 64
    dropout: float = 0.1
    architecture: str = "saturation_aware"
    saturation_embedding_dim: int = 8
    positive_class_weight: str = "auto"
    group_weighting: str = "none"
    sampler: str = "none"
    training_preset: Optional[str] = None
    loss_mode: str = "bce_rank"
    rank_weight: float = 0.2
    focal_gamma: float = 2.0
    min_delta: float = 0.0
    threshold: float = 0.5
    max_train_items: Optional[int] = None
    max_val_items: Optional[int] = None
    max_calib_items: Optional[int] = None
    max_test_items: Optional[int] = None
    early_stopping_patience: int = 8
    monitor_metric: str = "val_loss"
    save_every_epoch: bool = False

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
        if self.training_preset is not None:
            preset_values = _training_preset_values(self.training_preset)
            for key, value in preset_values.items():
                default_value = PRESET_OVERRIDE_DEFAULTS.get(key)
                if getattr(self, key) == default_value:
                    object.__setattr__(self, key, value)
        if self.architecture not in VALID_ARCHITECTURES:
            allowed = ", ".join(sorted(VALID_ARCHITECTURES))
            raise ValueError(f"architecture must be one of: {allowed}")
        if self.saturation_embedding_dim < 1:
            raise ValueError("saturation_embedding_dim must be >= 1")
        if self.positive_class_weight not in VALID_POSITIVE_CLASS_WEIGHT:
            allowed = ", ".join(sorted(VALID_POSITIVE_CLASS_WEIGHT))
            raise ValueError(f"positive_class_weight must be one of: {allowed}")
        if self.group_weighting not in VALID_GROUP_WEIGHTING:
            allowed = ", ".join(sorted(VALID_GROUP_WEIGHTING))
            raise ValueError(f"group_weighting must be one of: {allowed}")
        if self.sampler not in VALID_SAMPLERS:
            allowed = ", ".join(sorted(VALID_SAMPLERS))
            raise ValueError(f"sampler must be one of: {allowed}")
        if self.loss_mode not in VALID_LOSS_MODES:
            allowed = ", ".join(sorted(VALID_LOSS_MODES))
            raise ValueError(f"loss_mode must be one of: {allowed}")
        if self.rank_weight < 0:
            raise ValueError("rank_weight must be >= 0")
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be >= 0")
        if self.min_delta < 0:
            raise ValueError("min_delta must be >= 0")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if self.device not in VALID_DEVICES:
            allowed = ", ".join(sorted(VALID_DEVICES))
            raise ValueError(f"device must be one of: {allowed}")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")
        if self.monitor_metric not in VALID_MONITOR_METRICS:
            allowed = ", ".join(sorted(VALID_MONITOR_METRICS))
            raise ValueError(f"monitor_metric must be one of: {allowed}")
        for field_name in [
            "max_train_items",
            "max_val_items",
            "max_calib_items",
            "max_test_items",
        ]:
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be positive when provided")
        if self.methods is not None:
            resolved_methods = [method for method in self.methods if method]
            object.__setattr__(self, "methods", resolved_methods or None)
        out_path.mkdir(parents=True, exist_ok=True)


def apply_training_preset(
    config: NeuralFullTrainConfig, preset: str
) -> NeuralFullTrainConfig:
    """Return a copy of a neural training config with a named preset applied."""
    _training_preset_values(preset)
    return replace(config, training_preset=preset)


def train_neural_model(config: NeuralFullTrainConfig) -> dict:
    """Train a scale-ready gene-level neural model with checkpointing."""
    torch, _error = safe_import_torch()
    if torch is None:
        raise RuntimeError(
            "PyTorch is not available. Install torch or use an environment containing torch."
        ) from None

    set_random_seeds(config.seed)
    device_used = resolve_device(config.device)
    outdir = Path(config.outdir)
    checkpoints_dir = outdir / "checkpoints"
    predictions_dir = outdir / "predictions"
    logs_dir = outdir / "logs"
    for directory in [checkpoints_dir, predictions_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    datasets = _build_split_datasets(config)
    if len(datasets["train"]) == 0:
        raise ValueError("train split has no rows")
    for split in ["val", "calib", "test"]:
        if len(datasets[split]) == 0:
            warnings.append(f"empty_{split}_split")

    loaders = _build_dataloaders(datasets, config, torch)
    vocab_size = _infer_vocab_size_safely(datasets["train"], warnings)
    model = build_gene_classifier(
        architecture=config.architecture,
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        saturation_embedding_dim=config.saturation_embedding_dim,
    ).to(device_used)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    resolved_pos_weight = _resolve_pos_weight(
        torch=torch,
        dataset=datasets["train"],
        positive_class_weight=config.positive_class_weight,
        device=device_used,
        warnings=warnings,
    )
    resolved_pos_weight_value = (
        None
        if resolved_pos_weight is None
        else float(resolved_pos_weight.detach().cpu().item())
    )
    saturation_tier_counts_train = _saturation_tier_counts(datasets["train"])
    resolved_saturation_group_weights, group_weight_tensor = _resolve_group_weights(
        torch=torch,
        dataset=datasets["train"],
        group_weighting=config.group_weighting,
        device=device_used,
        warnings=warnings,
    )

    best_epoch: Optional[int] = None
    best_monitor_value: Optional[float] = None
    epochs_without_improvement = 0
    stopped_early = False
    stop_reason = "completed_epochs"
    history = []
    start_time = time.perf_counter()

    best_checkpoint_path = checkpoints_dir / "best_model.pt"
    last_checkpoint_path = checkpoints_dir / "last_model.pt"

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_one_epoch(
            model=model,
            dataloader=loaders["train"],
            optimizer=optimizer,
            pos_weight=resolved_pos_weight,
            group_weight_tensor=group_weight_tensor,
            loss_mode=config.loss_mode,
            rank_weight=config.rank_weight,
            focal_gamma=config.focal_gamma,
            device=device_used,
        )
        train_predictions = predict_neural_split(
            model, loaders["train_eval"], device_used, config.threshold
        )
        val_loss = _evaluate_loss(
            model,
            loaders["val"],
            resolved_pos_weight,
            group_weight_tensor,
            config.loss_mode,
            config.rank_weight,
            config.focal_gamma,
            device_used,
        )
        val_predictions = predict_neural_split(
            model, loaders["val"], device_used, config.threshold
        )
        train_metrics = _metrics_from_rows(train_predictions, config.threshold)
        val_metrics = _metrics_from_rows(val_predictions, config.threshold)
        monitor_value = _monitor_value(config.monitor_metric, val_loss, val_metrics)
        is_best = _is_improvement(
            metric=config.monitor_metric,
            value=monitor_value,
            best_value=best_monitor_value,
            min_delta=config.min_delta,
        )

        if best_epoch is None:
            is_best = True
            if monitor_value is None:
                warnings.append("best_checkpoint_saved_without_valid_monitor_value")

        if is_best:
            best_epoch = epoch
            best_monitor_value = monitor_value
            epochs_without_improvement = 0
            _save_checkpoint(
                torch=torch,
                path=best_checkpoint_path,
                model=model,
                config=config,
                vocab_size=vocab_size,
                epoch=epoch,
                monitor_value=monitor_value,
            )
        else:
            epochs_without_improvement += 1

        _save_checkpoint(
            torch=torch,
            path=last_checkpoint_path,
            model=model,
            config=config,
            vocab_size=vocab_size,
            epoch=epoch,
            monitor_value=monitor_value,
        )
        if config.save_every_epoch:
            _save_checkpoint(
                torch=torch,
                path=checkpoints_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                config=config,
                vocab_size=vocab_size,
                epoch=epoch,
                monitor_value=monitor_value,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics.get("accuracy"),
                "train_auroc": train_metrics.get("auroc"),
                "val_loss": val_loss,
                "val_accuracy": val_metrics.get("accuracy"),
                "val_auroc": val_metrics.get("auroc"),
                "monitor_metric": config.monitor_metric,
                "monitor_value": monitor_value,
                "is_best": int(is_best),
                "seconds_elapsed": time.perf_counter() - start_time,
            }
        )

        if (
            best_epoch is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_early = True
            stop_reason = f"no_improvement_for_{config.early_stopping_patience}_epochs"
            break

    epochs_completed = len(history)
    if best_epoch is None:
        best_epoch = epochs_completed
        stop_reason = "no_best_epoch_recorded"
        _save_checkpoint(
            torch=torch,
            path=best_checkpoint_path,
            model=model,
            config=config,
            vocab_size=vocab_size,
            epoch=best_epoch,
            monitor_value=best_monitor_value,
        )

    checkpoint = torch.load(best_checkpoint_path, map_location=device_used, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_used)

    prediction_rows = []
    for split in ["train", "val", "calib", "test"]:
        prediction_rows.extend(
            predict_neural_split(model, loaders[split], device_used, config.threshold)
        )
    metrics_by_split = _metrics_by_split(prediction_rows, config.threshold)

    meta_path = outdir / "neural_model_meta.json"
    history_path = logs_dir / "neural_training_history.tsv"
    predictions_path = predictions_dir / "neural_predictions.tsv"
    metrics_path = outdir / "neural_metrics.json"

    write_tsv(history_path, history, NEURAL_FULL_HISTORY_FIELDNAMES)
    write_tsv(predictions_path, prediction_rows, NEURAL_FULL_PREDICTION_FIELDNAMES)
    _write_json(
        metrics_path,
        {
            "metrics_by_split": metrics_by_split,
            "best_epoch": best_epoch,
            "best_monitor_value": best_monitor_value,
            "monitor_metric": config.monitor_metric,
            "threshold": config.threshold,
            "files": {
                "best_checkpoint": str(best_checkpoint_path),
                "last_checkpoint": str(last_checkpoint_path),
                "meta": str(meta_path),
                "history": str(history_path),
                "predictions": str(predictions_path),
                "metrics": str(metrics_path),
            },
            "note": "Scale-ready gene-level neural metrics, not final BABAPPA branch-site performance",
        },
    )

    meta = {
        "neural_trainer_version": NEURAL_TRAINER_VERSION,
        "model_class": _model_class_name(config.architecture),
        "architecture": config.architecture,
        "dataset_dir": config.dataset_dir,
        "seed": config.seed,
        "device_requested": config.device,
        "device_used": device_used,
        "methods": list(config.methods or []),
        "epochs_requested": config.epochs,
        "epochs_completed": epochs_completed,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "training_preset": config.training_preset,
        "loss_mode": config.loss_mode,
        "rank_weight": config.rank_weight,
        "focal_gamma": config.focal_gamma,
        "saturation_embedding_dim": config.saturation_embedding_dim,
        "positive_class_weight": config.positive_class_weight,
        "resolved_pos_weight": resolved_pos_weight_value,
        "group_weighting": config.group_weighting,
        "resolved_saturation_group_weights": resolved_saturation_group_weights,
        "sampler": config.sampler,
        "saturation_tier_counts_train": saturation_tier_counts_train,
        "min_delta": config.min_delta,
        "threshold": config.threshold,
        "vocab_size": vocab_size,
        "parameter_count": count_parameters(model),
        "train_rows": len(datasets["train"]),
        "val_rows": len(datasets["val"]),
        "calib_rows": len(datasets["calib"]),
        "test_rows": len(datasets["test"]),
        "monitor_metric": config.monitor_metric,
        "best_epoch": best_epoch,
        "best_monitor_value": best_monitor_value,
        "early_stopping_patience": config.early_stopping_patience,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "warnings": sorted(set(warnings)),
        "note": "Saturation-aware gene-level neural trainer; not final branch-site BABAPPA.",
    }
    _write_json(meta_path, meta)

    return {
        "status": "ok",
        "outdir": str(outdir),
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "meta": str(meta_path),
        "history": str(history_path),
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "device_used": device_used,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "warnings": sorted(set(warnings)),
    }


def predict_neural_split(model, dataloader, device: str, threshold: float) -> List[dict]:
    """Predict one split for the full trainer."""
    torch, _error = safe_import_torch()
    if torch is None:
        raise RuntimeError(
            "PyTorch is not available. Install torch or use an environment containing torch."
        ) from None
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            y = batch["y"].to(device)
            probs = torch.sigmoid(_model_logits(model, batch, device)).detach().cpu().numpy()
            labels = y.detach().cpu().numpy().astype(np.int32)
            pred_labels = (probs >= threshold).astype(np.int32)
            saturation_ids = batch.get("saturation_id")
            if saturation_ids is not None:
                saturation_ids_np = saturation_ids.detach().cpu().numpy().astype(np.int32)
            else:
                saturation_ids_np = np.zeros(len(labels), dtype=np.int32)
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
                        "saturation_tier": batch["saturation_tier"][index],
                        "saturation_id": int(saturation_ids_np[index]),
                        "prob_positive": float(prob),
                        "pred_label": pred_label,
                        "correct": int(pred_label == gene_label),
                    }
                )
    return rows


def _build_split_datasets(config: NeuralFullTrainConfig) -> Dict[str, BabappaTensorDataset]:
    max_items_by_split = {
        "train": config.max_train_items,
        "val": config.max_val_items,
        "calib": config.max_calib_items,
        "test": config.max_test_items,
    }
    return {
        split: BabappaTensorDataset(
            NeuralDatasetConfig(
                dataset_dir=config.dataset_dir,
                split=split,
                methods=config.methods,
                max_items=max_items_by_split[split],
                require_torch=True,
            )
        )
        for split in ["train", "val", "calib", "test"]
    }


def _build_dataloaders(datasets: Dict[str, BabappaTensorDataset], config, torch) -> dict:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_sampler = None
    train_shuffle = True
    if config.sampler == "saturation_balanced":
        train_sampler = _build_saturation_balanced_sampler(datasets["train"], torch)
        train_shuffle = False
    loaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            collate_fn=collate_babappa_batch,
            generator=generator,
        ),
        "train_eval": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_babappa_batch,
        ),
    }
    for split in ["val", "calib", "test"]:
        loaders[split] = torch.utils.data.DataLoader(
            datasets[split],
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_babappa_batch,
        )
    return loaders


def _train_one_epoch(
    model,
    dataloader,
    optimizer,
    pos_weight,
    group_weight_tensor,
    loss_mode: str,
    rank_weight: float,
    focal_gamma: float,
    device: str,
) -> Optional[float]:
    losses = []
    model.train()
    for batch in dataloader:
        y = batch["y"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = _model_logits(model, batch, device)
        loss = combined_loss(
            logits,
            y,
            loss_mode=loss_mode,
            pos_weight=pos_weight,
            sample_weight=_sample_weights(batch, group_weight_tensor, device),
            rank_weight=rank_weight,
            focal_gamma=focal_gamma,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    if not losses:
        return None
    return float(np.mean(losses))


def _evaluate_loss(
    model,
    dataloader,
    pos_weight,
    group_weight_tensor,
    loss_mode: str,
    rank_weight: float,
    focal_gamma: float,
    device: str,
) -> Optional[float]:
    torch, _error = safe_import_torch()
    if torch is None:
        return None
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            y = batch["y"].to(device)
            logits = _model_logits(model, batch, device)
            loss = combined_loss(
                logits,
                y,
                loss_mode=loss_mode,
                pos_weight=pos_weight,
                sample_weight=_sample_weights(batch, group_weight_tensor, device),
                rank_weight=rank_weight,
                focal_gamma=focal_gamma,
            )
            losses.append(float(loss.detach().cpu().item()))
    if not losses:
        return None
    return float(np.mean(losses))


def _infer_vocab_size_safely(dataset: BabappaTensorDataset, warnings: List[str]) -> int:
    try:
        max_codon_id = 0
        for index in range(min(len(dataset), 64)):
            item = dataset[index]
            max_codon_id = max(max_codon_id, int(item["X"][..., 0].max().item()))
        return max(128, max_codon_id + 1)
    except Exception as exc:  # pragma: no cover - defensive fallback
        warnings.append(f"vocab_size_inference_failed:{exc}")
        return 128


def _resolve_pos_weight(
    torch,
    dataset: BabappaTensorDataset,
    positive_class_weight: str,
    device: str,
    warnings: List[str],
):
    if positive_class_weight == "none":
        return None
    positives = 0
    negatives = 0
    for row in dataset.rows:
        label = int(float(row["gene_label"]))
        if label == 1:
            positives += 1
        else:
            negatives += 1
    if positives == 0 or negatives == 0:
        warnings.append("pos_weight_not_applied_single_class_train_split")
        return None
    return torch.tensor([negatives / positives], dtype=torch.float32, device=device)


def _model_logits(model, batch: dict, device: str):
    X = batch["X"].to(device)
    if getattr(model, "uses_saturation_id", False):
        saturation_id = batch.get("saturation_id")
        if saturation_id is not None:
            return model(X, saturation_id=saturation_id.to(device))
    return model(X)


def _sample_weights(batch: dict, group_weight_tensor, device: str):
    if group_weight_tensor is not None:
        saturation_id = batch.get("saturation_id")
        if saturation_id is not None:
            return group_weight_tensor[saturation_id.to(device).long()]
    return None


def _saturation_tier_counts(dataset: BabappaTensorDataset) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in dataset.rows:
        tier = row.get("saturation_tier") or "unknown"
        counts[tier] = counts.get(tier, 0) + 1
    return dict(sorted(counts.items()))


def _resolve_group_weights(
    torch,
    dataset: BabappaTensorDataset,
    group_weighting: str,
    device: str,
    warnings: List[str],
) -> tuple[Dict[str, float], object]:
    if group_weighting == "none":
        return {}, None
    id_counts: Dict[int, int] = {}
    for row in dataset.rows:
        tier_id = saturation_tier_to_id(row.get("saturation_tier", "unknown"))
        id_counts[tier_id] = id_counts.get(tier_id, 0) + 1
    if not id_counts:
        warnings.append("saturation_group_weights_not_applied_empty_train_split")
        return {}, None
    total = sum(id_counts.values())
    n_groups = len(id_counts)
    weights_by_id = {
        tier_id: total / (n_groups * count)
        for tier_id, count in id_counts.items()
        if count > 0
    }
    tensor_values = [1.0] * len(SATURATION_TIER_TO_ID)
    for tier_id, weight in weights_by_id.items():
        if 0 <= tier_id < len(tensor_values):
            tensor_values[tier_id] = float(weight)
    group_weight_tensor = torch.tensor(tensor_values, dtype=torch.float32, device=device)
    id_to_tier = {value: key for key, value in SATURATION_TIER_TO_ID.items()}
    resolved = {
        id_to_tier.get(tier_id, f"id_{tier_id}"): float(weight)
        for tier_id, weight in sorted(weights_by_id.items())
    }
    return resolved, group_weight_tensor


def _build_saturation_balanced_sampler(dataset: BabappaTensorDataset, torch):
    id_counts: Dict[int, int] = {}
    tier_ids = []
    for row in dataset.rows:
        tier_id = saturation_tier_to_id(row.get("saturation_tier", "unknown"))
        tier_ids.append(tier_id)
        id_counts[tier_id] = id_counts.get(tier_id, 0) + 1
    sample_weights = [
        1.0 / max(1, id_counts.get(tier_id, 0))
        for tier_id in tier_ids
    ]
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def _monitor_value(metric: str, val_loss: Optional[float], val_metrics: dict) -> Optional[float]:
    if metric == "val_loss":
        return val_loss
    if metric == "val_auroc":
        return val_metrics.get("auroc")
    if metric == "val_accuracy":
        return val_metrics.get("accuracy")
    return None


def _is_improvement(
    metric: str,
    value: Optional[float],
    best_value: Optional[float],
    min_delta: float = 0.0,
) -> bool:
    if value is None:
        return False
    if best_value is None:
        return True
    if metric == "val_loss":
        return value < (best_value - min_delta)
    return value > (best_value + min_delta)


def _save_checkpoint(torch, path: Path, model, config, vocab_size: int, epoch: int, monitor_value) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "vocab_size": vocab_size,
            "epoch": epoch,
            "monitor_value": monitor_value,
            "model_class": _model_class_name(config.architecture),
            "architecture": config.architecture,
            "note": "Scale-ready gene-level neural trainer; not final branch-site BABAPPA architecture.",
        },
        path,
    )


def _metrics_by_split(prediction_rows: List[dict], threshold: float) -> Dict[str, dict]:
    metrics = {}
    for split in ["train", "val", "calib", "test"]:
        rows = [row for row in prediction_rows if row["split"] == split]
        metrics[split] = _metrics_from_rows(rows, threshold)
    metrics["all"] = _metrics_from_rows(prediction_rows, threshold)
    return metrics


def _metrics_from_rows(rows: List[dict], threshold: float) -> dict:
    y_true = np.asarray([int(row["gene_label"]) for row in rows], dtype=np.int32)
    y_prob = np.asarray([float(row["prob_positive"]) for row in rows], dtype=np.float64)
    return compute_binary_metrics(y_true, y_prob, threshold)


def _model_class_name(architecture: str) -> str:
    if architecture == "saturation_aware":
        return "SaturationAwareGeneClassifier"
    if architecture in {"site_attention", "site_attention_saturation"}:
        return "SiteAttentionGeneClassifier"
    if architecture == "contrastive":
        return "ContrastiveGeneClassifier"
    return "SmallGeneClassifier"


def _training_preset_values(preset: str) -> dict:
    if preset not in TRAINING_PRESETS:
        allowed = ", ".join(sorted(TRAINING_PRESETS))
        raise ValueError(f"training_preset must be one of: {allowed}")
    return dict(TRAINING_PRESETS[preset])


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
