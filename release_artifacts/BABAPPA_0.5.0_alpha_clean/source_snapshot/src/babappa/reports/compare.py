"""Baseline-versus-neural metric comparison reports for BABAPPA."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from babappa import __version__

COMPARISON_VERSION = __version__
SPLITS = ["train", "val", "calib", "test", "all"]
METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "mcc", "auroc"]
MODEL_LABELS = [
    "baseline_raw",
    "baseline_calibrated",
    "neural_raw",
    "neural_calibrated",
]
FIELDNAMES = [
    "split",
    "metric",
    "baseline_raw",
    "baseline_calibrated",
    "neural_raw",
    "neural_calibrated",
    "best_model",
    "best_value",
]


@dataclass(frozen=True)
class ModelCompareConfig:
    """Configuration for comparing BABAPPA model metrics."""

    outdir: str
    baseline_metrics: Optional[str] = None
    baseline_calibrated_metrics: Optional[str] = None
    neural_metrics: Optional[str] = None
    neural_calibrated_metrics: Optional[str] = None
    title: str = "BABAPPA model comparison"

    def __post_init__(self) -> None:
        supplied = _supplied_inputs(self)
        if len(supplied) < 2:
            raise ValueError("at least two metric files must be supplied")
        for label, filename in supplied.items():
            path = Path(filename)
            if not path.exists():
                raise ValueError(f"{label} does not exist: {path}")
            if not path.is_file():
                raise ValueError(f"{label} is not a file: {path}")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def compare_models(config: ModelCompareConfig) -> dict:
    """Compare supplied baseline and neural metric files."""
    outdir = Path(config.outdir)
    json_path = outdir / "model_comparison.json"
    tsv_path = outdir / "model_comparison.tsv"
    markdown_path = outdir / "model_comparison.md"
    warnings: List[str] = []

    metrics_by_model = _load_metric_sources(config, warnings)
    rows = _comparison_rows(metrics_by_model)
    comparison_by_split = _comparison_by_split(rows)
    best_by_metric = _best_by_metric(rows)

    payload = {
        "comparison_version": COMPARISON_VERSION,
        "title": config.title,
        "inputs": _supplied_inputs(config),
        "available_models": sorted(metrics_by_model.keys()),
        "comparison_by_split": comparison_by_split,
        "best_by_metric": best_by_metric,
        "warnings": sorted(set(warnings)),
        "generated_files": {
            "json": str(json_path),
            "tsv": str(tsv_path),
            "markdown": str(markdown_path),
        },
    }
    _write_json(json_path, payload)
    _write_tsv(tsv_path, rows)
    markdown_path.write_text(_render_markdown(payload, rows), encoding="utf-8")

    return {
        "status": "ok",
        "outdir": str(outdir),
        "json": str(json_path),
        "tsv": str(tsv_path),
        "markdown": str(markdown_path),
        "warnings": payload["warnings"],
    }


def _load_metric_sources(
    config: ModelCompareConfig, warnings: List[str]
) -> Dict[str, dict]:
    sources: Dict[str, dict] = {}
    for label, filename in _supplied_inputs(config).items():
        payload = _load_json(Path(filename), warnings)
        if payload is None:
            continue
        metrics = _extract_metrics(label, payload)
        if not metrics:
            warnings.append(f"no_metrics_found:{label}:{filename}")
        sources[label] = metrics
    return sources


def _extract_metrics(label: str, payload: dict) -> dict:
    if label in {"baseline_raw", "neural_raw"}:
        return payload.get("metrics_by_split") or {}
    if label in {"baseline_calibrated", "neural_calibrated"}:
        return payload.get("metrics_by_split_calibrated") or {}
    return {}


def _comparison_rows(metrics_by_model: Dict[str, dict]) -> List[dict]:
    rows = []
    for split in SPLITS:
        for metric in METRICS:
            row = {"split": split, "metric": metric}
            values: Dict[str, Optional[float]] = {}
            for label in MODEL_LABELS:
                value = _metric_value(metrics_by_model.get(label, {}), split, metric)
                values[label] = value
                row[label] = "" if value is None else value
            best_model, best_value = _best_model(values)
            row["best_model"] = best_model or ""
            row["best_value"] = "" if best_value is None else best_value
            rows.append(row)
    return rows


def _metric_value(metrics_by_split: dict, split: str, metric: str) -> Optional[float]:
    metrics = metrics_by_split.get(split)
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_model(values: Dict[str, Optional[float]]) -> tuple:
    valid = [(label, value) for label, value in values.items() if value is not None]
    if not valid:
        return None, None
    valid.sort(key=lambda item: item[1], reverse=True)
    return valid[0][0], valid[0][1]


def _comparison_by_split(rows: List[dict]) -> dict:
    by_split: Dict[str, dict] = {split: {} for split in SPLITS}
    for row in rows:
        split = row["split"]
        metric = row["metric"]
        by_split.setdefault(split, {})[metric] = {
            "baseline_raw": row["baseline_raw"],
            "baseline_calibrated": row["baseline_calibrated"],
            "neural_raw": row["neural_raw"],
            "neural_calibrated": row["neural_calibrated"],
            "best_model": row["best_model"],
            "best_value": row["best_value"],
        }
    return by_split


def _best_by_metric(rows: List[dict]) -> dict:
    result: Dict[str, dict] = {metric: {} for metric in METRICS}
    for row in rows:
        result[row["metric"]][row["split"]] = {
            "best_model": row["best_model"],
            "best_value": row["best_value"],
        }
    return result


def _load_json(path: Path, warnings: List[str]) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"could_not_load_json:{path}:{exc}")
        return None
    if not isinstance(payload, dict):
        warnings.append(f"json_not_object:{path}")
        return None
    return payload


def _render_markdown(payload: dict, rows: List[dict]) -> str:
    lines: List[str] = [f"# {payload['title']}", ""]
    lines.extend(["## Inputs", ""])
    for label, path in payload.get("inputs", {}).items():
        lines.append(f"- `{label}`: `{path}`")
    lines.extend(["", "## Summary", ""])
    lines.append(f"- Available models: {_format_list(payload.get('available_models'))}")
    lines.append(f"- Warnings: {_format_list(payload.get('warnings'))}")
    lines.extend(["", "## Comparison by split", ""])
    lines.extend(
        [
            "| Split | Metric | Baseline raw | Baseline calibrated | Neural raw | Neural calibrated | Best model | Best value |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {split} | {metric} | {baseline_raw} | {baseline_calibrated} | {neural_raw} | {neural_calibrated} | {best_model} | {best_value} |".format(
                split=row["split"],
                metric=row["metric"],
                baseline_raw=_format_float(row["baseline_raw"]),
                baseline_calibrated=_format_float(row["baseline_calibrated"]),
                neural_raw=_format_float(row["neural_raw"]),
                neural_calibrated=_format_float(row["neural_calibrated"]),
                best_model=row["best_model"] or "NA",
                best_value=_format_float(row["best_value"]),
            )
        )
    lines.extend(["", "## Warnings", ""])
    warnings = payload.get("warnings") or []
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation caveats",
            "",
            "- Comparisons are only meaningful when models were trained/evaluated on the same dataset split.",
            "- Calibration warnings must be considered before interpreting threshold-dependent metrics.",
            "- Current neural model is gene-level and scaffold-alignment based.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _format_float(value: Any) -> str:
    if value in ("", None):
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_list(value: Any) -> str:
    if not value:
        return "none"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _supplied_inputs(config: ModelCompareConfig) -> Dict[str, str]:
    raw = {
        "baseline_raw": config.baseline_metrics,
        "baseline_calibrated": config.baseline_calibrated_metrics,
        "neural_raw": config.neural_metrics,
        "neural_calibrated": config.neural_calibrated_metrics,
    }
    return {label: str(path) for label, path in raw.items() if path is not None}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_tsv(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
