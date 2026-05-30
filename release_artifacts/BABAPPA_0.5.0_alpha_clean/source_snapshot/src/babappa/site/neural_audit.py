"""Validation for site-level neural classifier outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import numpy as np

LEAKAGE_TOKENS = ("selected", "truth", "label", "positive")


def validate_site_neural_dir(model_dir: str | Path) -> dict:
    """Validate site-level neural artifacts."""
    path = Path(model_dir)
    failures: List[str] = []
    warnings: List[str] = []
    required = {
        "checkpoint": path / "site_neural_checkpoint.pt",
        "meta": path / "site_neural_model_meta.json",
        "history": path / "site_neural_history.tsv",
        "predictions": path / "site_neural_predictions.tsv",
        "metrics": path / "site_neural_metrics.json",
    }
    for label, file_path in required.items():
        if not file_path.exists():
            failures.append(f"missing_{label}:{file_path}")

    meta = _load_json(required["meta"], failures)
    _load_json(required["metrics"], failures)
    feature_columns = meta.get("feature_columns") or []
    bad_features = [
        column
        for column in feature_columns
        if any(token in str(column).lower() for token in LEAKAGE_TOKENS)
    ]
    if bad_features:
        failures.append("leakage_like_feature_columns:" + ",".join(sorted(bad_features)))

    probs = []
    pred_positive = 0
    n_predictions = 0
    if required["predictions"].exists():
        try:
            with required["predictions"].open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                missing = sorted(
                    {
                        "site_id",
                        "y_site",
                        "prob_positive",
                        "pred_label",
                        "correct",
                    }
                    - set(reader.fieldnames or [])
                )
                if missing:
                    failures.append("missing_prediction_columns:" + ",".join(missing))
                for row in reader:
                    n_predictions += 1
                    if row.get("y_site") not in {"0", "1"}:
                        failures.append(f"invalid_y_site:{row.get('site_id')}:{row.get('y_site')}")
                    if row.get("pred_label") == "1":
                        pred_positive += 1
                    try:
                        prob = float(row.get("prob_positive", "nan"))
                    except ValueError:
                        failures.append(f"non_numeric_probability:{row.get('site_id')}")
                        continue
                    probs.append(prob)
                    if not 0 <= prob <= 1:
                        failures.append(f"probability_out_of_range:{row.get('site_id')}:{prob}")
        except OSError as exc:
            failures.append(f"could_not_read_predictions:{required['predictions']}:{exc}")

    if n_predictions == 0:
        failures.append("no_predictions")
    if probs:
        prob_std = float(np.std(np.asarray(probs, dtype=np.float64)))
        if prob_std < 0.02:
            warnings.append("probability_collapse_std_below_0_02")
    if n_predictions > 0 and pred_positive == 0:
        warnings.append("no_positive_predictions_at_threshold_0_5")

    return {
        "status": "fail" if failures else "ok",
        "n_predictions": n_predictions,
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _load_json(path: Path, failures: List[str]) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"could_not_parse_json:{path}:{exc}")
        return {}
    if not isinstance(payload, dict):
        failures.append(f"json_not_object:{path}")
        return {}
    return payload
