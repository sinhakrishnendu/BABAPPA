"""Validation utilities for BABAPPA neural smoke-training artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

from babappa.datasets.index import read_tsv
from babappa.training.neural_env import safe_import_torch
from babappa.training.neural_train import NEURAL_PREDICTION_FIELDNAMES


def validate_neural_smoke_dir(model_dir: Union[str, Path]) -> dict:
    """Validate neural smoke-training artifacts."""
    model_path = Path(model_dir)
    failures: List[str] = []
    warnings: List[str] = []
    checkpoint_path = model_path / "neural_smoke_checkpoint.pt"
    meta_path = model_path / "neural_smoke_model_meta.json"
    history_path = model_path / "neural_smoke_history.tsv"
    predictions_path = model_path / "neural_smoke_predictions.tsv"
    metrics_path = model_path / "neural_smoke_metrics.json"

    _validate_checkpoint(checkpoint_path, failures, warnings)
    _validate_json(meta_path, "neural_smoke_model_meta.json", failures)
    metrics = _validate_json(metrics_path, "neural_smoke_metrics.json", failures)
    _validate_history(history_path, failures)
    prediction_rows = _validate_predictions(predictions_path, failures)

    if isinstance(metrics, dict) and "metrics_by_split" not in metrics:
        failures.append("neural_smoke_metrics.json missing metrics_by_split")

    return {
        "status": "fail" if failures else "ok",
        "n_predictions": len(prediction_rows),
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _validate_checkpoint(path: Path, failures: List[str], warnings: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing neural_smoke_checkpoint.pt: {path}")
        return
    torch, _error = safe_import_torch()
    if torch is None:
        warnings.append("torch_unavailable_checkpoint_not_loaded")
        return
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        failures.append(f"unreadable neural_smoke_checkpoint.pt: {exc}")
        return
    if not isinstance(checkpoint, dict):
        failures.append("neural_smoke_checkpoint.pt is not a dictionary checkpoint")
        return
    for key in ["model_state_dict", "config", "vocab_size", "model_class"]:
        if key not in checkpoint:
            failures.append(f"neural_smoke_checkpoint.pt missing {key}")


def _validate_json(path: Path, label: str, failures: List[str]) -> object:
    if not path.exists():
        failures.append(f"missing {label}: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            failures.append(f"{label} is not a JSON object")
            return None
        return payload
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"unreadable {label}: {exc}")
        return None


def _validate_history(path: Path, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing neural_smoke_history.tsv: {path}")
        return
    try:
        rows = read_tsv(path)
    except OSError as exc:
        failures.append(f"unreadable neural_smoke_history.tsv: {exc}")
        return
    if not rows:
        failures.append("neural_smoke_history.tsv has no rows")


def _validate_predictions(path: Path, failures: List[str]) -> List[dict]:
    if not path.exists():
        failures.append(f"missing neural_smoke_predictions.tsv: {path}")
        return []
    try:
        rows = read_tsv(path)
    except OSError as exc:
        failures.append(f"unreadable neural_smoke_predictions.tsv: {exc}")
        return []
    if rows:
        missing = [column for column in NEURAL_PREDICTION_FIELDNAMES if column not in rows[0]]
        if missing:
            failures.append(
                "neural_smoke_predictions.tsv missing columns: "
                + ", ".join(missing)
            )
    for row in rows:
        try:
            prob = float(row.get("prob_positive", ""))
            if not 0 <= prob <= 1:
                failures.append(f"prob_positive out of range: {prob}")
        except ValueError:
            failures.append(f"prob_positive is not numeric: {row.get('prob_positive')}")
        if row.get("pred_label") not in {"0", "1", 0, 1}:
            failures.append(f"invalid pred_label: {row.get('pred_label')}")
    return rows
