"""Validation utilities for BABAPPA baseline model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

import numpy as np

from babappa.datasets.index import read_tsv
from babappa.models.baseline import PREDICTION_FIELDNAMES, VALID_SPLITS


def validate_baseline_model_dir(model_dir: Union[str, Path]) -> dict:
    """Validate NumPy baseline model artifact directory."""
    model_path = Path(model_dir)
    failures: List[str] = []
    warnings: List[str] = []
    model_npz = model_path / "baseline_model.npz"
    meta_json = model_path / "baseline_model_meta.json"
    predictions_tsv = model_path / "baseline_predictions.tsv"
    metrics_json = model_path / "baseline_metrics.json"

    _validate_model_npz(model_npz, failures)
    _validate_json(meta_json, "baseline_model_meta.json", failures)
    metrics = _validate_json(metrics_json, "baseline_metrics.json", failures)
    prediction_rows = _validate_predictions(predictions_tsv, failures)

    if isinstance(metrics, dict) and "metrics_by_split" not in metrics:
        failures.append("baseline_metrics.json missing metrics_by_split")

    return {
        "status": "fail" if failures else "ok",
        "n_predictions": len(prediction_rows),
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _validate_model_npz(path: Path, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing baseline_model.npz: {path}")
        return
    try:
        with np.load(path, allow_pickle=False) as model:
            required = {"weights", "bias", "feature_mean", "feature_std", "threshold"}
            missing = required - set(model.files)
            if missing:
                failures.append(f"baseline_model.npz missing arrays: {sorted(missing)}")
                return
            weights = model["weights"]
            feature_mean = model["feature_mean"]
            feature_std = model["feature_std"]
            if weights.ndim != 1:
                failures.append("weights array is not 1D")
            if feature_mean.shape != weights.shape:
                failures.append("feature_mean length does not match weights length")
            if feature_std.shape != weights.shape:
                failures.append("feature_std length does not match weights length")
    except (OSError, ValueError) as exc:
        failures.append(f"unreadable baseline_model.npz: {exc}")


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


def _validate_predictions(path: Path, failures: List[str]) -> List[dict]:
    if not path.exists():
        failures.append(f"missing baseline_predictions.tsv: {path}")
        return []
    try:
        rows = read_tsv(path)
    except OSError as exc:
        failures.append(f"unreadable baseline_predictions.tsv: {exc}")
        return []

    if rows:
        missing_columns = [
            column for column in PREDICTION_FIELDNAMES if column not in rows[0]
        ]
        if missing_columns:
            failures.append(
                f"baseline_predictions.tsv missing columns: {', '.join(missing_columns)}"
            )

    valid_splits = set(VALID_SPLITS)
    for row in rows:
        try:
            prob_positive = float(row.get("prob_positive", ""))
            if not 0 <= prob_positive <= 1:
                failures.append(f"prob_positive out of range: {prob_positive}")
        except ValueError:
            failures.append(f"prob_positive is not numeric: {row.get('prob_positive')}")

        if row.get("split") not in valid_splits:
            failures.append(f"invalid split value: {row.get('split')}")

    return rows
