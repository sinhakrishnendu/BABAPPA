"""Validation for site-level baseline model outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

FORBIDDEN_FEATURE_TOKENS = ("selected", "truth", "label", "positive")


def validate_site_baseline_dir(model_dir: str | Path) -> dict:
    """Validate site-level baseline artifacts."""
    path = Path(model_dir)
    failures: List[str] = []
    warnings: List[str] = []
    model_path = path / "site_baseline_model.npz"
    meta_path = path / "site_baseline_model_meta.json"
    predictions_path = path / "site_baseline_predictions.tsv"
    metrics_path = path / "site_baseline_metrics.json"
    n_predictions = 0

    for required in (model_path, meta_path, predictions_path, metrics_path):
        if not required.exists():
            failures.append(f"missing_file:{required}")

    meta = _load_json(meta_path, failures)
    _load_json(metrics_path, failures)
    feature_columns = meta.get("feature_columns") or []
    bad_features = [
        column
        for column in feature_columns
        if any(token in str(column).lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if bad_features:
        failures.append("leakage_like_feature_columns:" + ",".join(sorted(bad_features)))

    if predictions_path.exists():
        try:
            with predictions_path.open("r", encoding="utf-8", newline="") as handle:
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
                    try:
                        prob = float(row.get("prob_positive", "nan"))
                    except ValueError:
                        failures.append(f"non_numeric_probability:{row.get('site_id')}")
                        continue
                    if not 0 <= prob <= 1:
                        failures.append(f"probability_out_of_range:{row.get('site_id')}:{prob}")
                    if row.get("y_site") not in {"0", "1"}:
                        failures.append(f"invalid_y_site:{row.get('site_id')}:{row.get('y_site')}")
        except OSError as exc:
            failures.append(f"could_not_read_predictions:{predictions_path}:{exc}")

    if n_predictions == 0:
        failures.append("no_predictions")

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
