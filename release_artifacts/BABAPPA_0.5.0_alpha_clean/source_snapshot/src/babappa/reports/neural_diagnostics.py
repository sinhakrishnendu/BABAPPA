"""Diagnostics for BABAPPA neural training runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv

NEURAL_DIAGNOSTICS_VERSION = __version__
SUMMARY_FIELDNAMES = [
    "model_name",
    "split",
    "n",
    "positives",
    "negatives",
    "prob_min",
    "prob_q01",
    "prob_q05",
    "prob_q25",
    "prob_median",
    "prob_q75",
    "prob_q95",
    "prob_q99",
    "prob_max",
    "prob_mean",
    "prob_std",
    "positive_prob_mean",
    "negative_prob_mean",
    "separation",
    "fraction_ge_0_1",
    "fraction_ge_0_3",
    "fraction_ge_0_5",
    "fraction_ge_0_7",
    "fraction_ge_0_9",
]


@dataclass(frozen=True)
class NeuralDiagnosticsConfig:
    """Configuration for neural run diagnostics."""

    model_dir: str
    outdir: str
    predictions_tsv: Optional[str] = None
    history_tsv: Optional[str] = None
    metadata_json: Optional[str] = None
    model_name: str = "neural_model"

    def __post_init__(self) -> None:
        model_path = Path(self.model_dir)
        if not model_path.exists():
            raise ValueError(f"model_dir does not exist: {model_path}")
        if not model_path.is_dir():
            raise ValueError(f"model_dir is not a directory: {model_path}")
        for label, path in self._resolved_paths().items():
            if not path.exists():
                raise ValueError(f"{label} does not exist: {path}")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)

    def _resolved_paths(self) -> Dict[str, Path]:
        model_path = Path(self.model_dir)
        return {
            "predictions_tsv": Path(self.predictions_tsv)
            if self.predictions_tsv
            else model_path / "predictions" / "neural_predictions.tsv",
            "history_tsv": Path(self.history_tsv)
            if self.history_tsv
            else model_path / "logs" / "neural_training_history.tsv",
            "metadata_json": Path(self.metadata_json)
            if self.metadata_json
            else model_path / "neural_model_meta.json",
        }


def diagnose_neural_run(config: NeuralDiagnosticsConfig) -> dict:
    """Summarize probability, threshold, history, and metadata diagnostics."""
    paths = config._resolved_paths()
    outdir = Path(config.outdir)
    rows = read_tsv(paths["predictions_tsv"])
    history_rows = read_tsv(paths["history_tsv"])
    metadata = _load_json(paths["metadata_json"])
    if not rows:
        raise ValueError("neural prediction TSV contains no rows")

    warnings: List[str] = []
    summaries = _probability_summaries(config.model_name, rows, warnings)
    history_summary = _history_summary(history_rows, metadata, warnings)
    metadata_summary = _metadata_summary(metadata)

    json_path = outdir / "neural_diagnostics.json"
    tsv_path = outdir / "neural_probability_summary.tsv"
    markdown_path = outdir / "neural_diagnostics.md"
    sorted_warnings = sorted(set(warnings))
    payload = {
        "neural_diagnostics_version": NEURAL_DIAGNOSTICS_VERSION,
        "model_name": config.model_name,
        "model_dir": str(Path(config.model_dir)),
        "inputs": {
            "predictions_tsv": str(paths["predictions_tsv"]),
            "history_tsv": str(paths["history_tsv"]),
            "metadata_json": str(paths["metadata_json"]),
        },
        "metadata_summary": metadata_summary,
        "history_summary": history_summary,
        "probability_summary_by_split": {
            row["split"]: row for row in summaries
        },
        "warnings": sorted_warnings,
        "generated_files": {
            "json": str(json_path),
            "summary_tsv": str(tsv_path),
            "markdown": str(markdown_path),
        },
    }
    _write_json(json_path, payload)
    write_tsv(tsv_path, summaries, SUMMARY_FIELDNAMES)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "json": str(json_path),
        "summary_tsv": str(tsv_path),
        "markdown": str(markdown_path),
        "warnings": sorted_warnings,
    }


def _probability_summaries(
    model_name: str, rows: List[dict], warnings: List[str]
) -> List[dict]:
    splits = ["train", "val", "calib", "test"]
    summaries = []
    for split in splits + ["all"]:
        split_rows = rows if split == "all" else [row for row in rows if row.get("split") == split]
        summary = _summarize_split(model_name, split, split_rows, warnings)
        summaries.append(summary)
    return summaries


def _summarize_split(
    model_name: str, split: str, rows: List[dict], warnings: List[str]
) -> dict:
    probs = np.asarray([_safe_float(row.get("prob_positive")) for row in rows], dtype=float)
    labels = np.asarray([int(float(row.get("gene_label", 0))) for row in rows], dtype=int)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if probs.size == 0:
        return {
            "model_name": model_name,
            "split": split,
            "n": 0,
            "positives": 0,
            "negatives": 0,
            **{name: None for name in SUMMARY_FIELDNAMES if name.startswith("prob_")},
            "positive_prob_mean": None,
            "negative_prob_mean": None,
            "separation": None,
            "fraction_ge_0_1": None,
            "fraction_ge_0_3": None,
            "fraction_ge_0_5": None,
            "fraction_ge_0_7": None,
            "fraction_ge_0_9": None,
        }
    positive_probs = probs[labels == 1]
    negative_probs = probs[labels == 0]
    positive_mean = _mean_or_none(positive_probs)
    negative_mean = _mean_or_none(negative_probs)
    separation = (
        None
        if positive_mean is None or negative_mean is None
        else positive_mean - negative_mean
    )
    fraction_ge_0_5 = float((probs >= 0.5).mean())
    if positives > 0 and int((probs >= 0.5).sum()) == 0:
        warnings.append(f"{split}:all_negative_at_0_5")
    if negatives > 0 and int((probs < 0.5).sum()) == 0:
        warnings.append(f"{split}:all_positive_at_0_5")
    if float(probs.std()) < 0.02:
        warnings.append(f"{split}:probability_collapse")
    if separation is not None and separation < 0:
        warnings.append(f"{split}:inverted_signal")
    return {
        "model_name": model_name,
        "split": split,
        "n": int(probs.size),
        "positives": positives,
        "negatives": negatives,
        "prob_min": float(probs.min()),
        "prob_q01": _quantile(probs, 0.01),
        "prob_q05": _quantile(probs, 0.05),
        "prob_q25": _quantile(probs, 0.25),
        "prob_median": _quantile(probs, 0.50),
        "prob_q75": _quantile(probs, 0.75),
        "prob_q95": _quantile(probs, 0.95),
        "prob_q99": _quantile(probs, 0.99),
        "prob_max": float(probs.max()),
        "prob_mean": float(probs.mean()),
        "prob_std": float(probs.std()),
        "positive_prob_mean": positive_mean,
        "negative_prob_mean": negative_mean,
        "separation": separation,
        "fraction_ge_0_1": float((probs >= 0.1).mean()),
        "fraction_ge_0_3": float((probs >= 0.3).mean()),
        "fraction_ge_0_5": fraction_ge_0_5,
        "fraction_ge_0_7": float((probs >= 0.7).mean()),
        "fraction_ge_0_9": float((probs >= 0.9).mean()),
    }


def _history_summary(
    history_rows: List[dict], metadata: dict, warnings: List[str]
) -> dict:
    train_losses = [_safe_float(row.get("train_loss")) for row in history_rows]
    val_losses = [_safe_float(row.get("val_loss")) for row in history_rows]
    val_aurocs = [_safe_float(row.get("val_auroc")) for row in history_rows]
    train_losses = [value for value in train_losses if value is not None]
    val_losses = [value for value in val_losses if value is not None]
    val_aurocs = [value for value in val_aurocs if value is not None]
    overfitting_possible = False
    if len(train_losses) >= 2 and len(val_losses) >= 2:
        overfitting_possible = train_losses[-1] < train_losses[0] and val_losses[-1] > val_losses[0]
        if overfitting_possible:
            warnings.append("overfitting_possible")
    return {
        "epochs_completed": len(history_rows),
        "best_epoch": metadata.get("best_epoch"),
        "train_loss_first": train_losses[0] if train_losses else None,
        "train_loss_last": train_losses[-1] if train_losses else None,
        "val_loss_first": val_losses[0] if val_losses else None,
        "val_loss_last": val_losses[-1] if val_losses else None,
        "val_auroc_first": val_aurocs[0] if val_aurocs else None,
        "val_auroc_last": val_aurocs[-1] if val_aurocs else None,
        "overfitting_possible": overfitting_possible,
    }


def _metadata_summary(metadata: dict) -> dict:
    keys = [
        "architecture",
        "training_preset",
        "group_weighting",
        "sampler",
        "positive_class_weight",
        "monitor_metric",
        "best_epoch",
        "stopped_early",
    ]
    return {key: metadata.get(key) for key in keys}


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Neural diagnostics",
        "",
        "## Model metadata",
        "",
    ]
    for key, value in payload["metadata_summary"].items():
        lines.append(f"- {key}: {_format_value(value)}")
    lines.extend(["", "## Training history", ""])
    for key, value in payload["history_summary"].items():
        lines.append(f"- {key}: {_format_value(value)}")
    lines.extend(
        [
            "",
            "## Probability distribution",
            "",
            "| Split | n | mean | std | median | frac >= 0.5 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, summary in payload["probability_summary_by_split"].items():
        lines.append(
            "| {split} | {n} | {mean} | {std} | {median} | {frac} |".format(
                split=split,
                n=summary.get("n", 0),
                mean=_format_float(summary.get("prob_mean")),
                std=_format_float(summary.get("prob_std")),
                median=_format_float(summary.get("prob_median")),
                frac=_format_float(summary.get("fraction_ge_0_5")),
            )
        )
    lines.extend(["", "## Class separation", ""])
    for split, summary in payload["probability_summary_by_split"].items():
        lines.append(
            f"- {split}: positive_mean={_format_float(summary.get('positive_prob_mean'))}, "
            f"negative_mean={_format_float(summary.get('negative_prob_mean'))}, "
            f"separation={_format_float(summary.get('separation'))}"
        )
    lines.extend(["", "## Warnings", ""])
    if payload["warnings"]:
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- All-negative predictions at threshold 0.5 indicate threshold/model collapse.",
            "- AUROC can remain non-null even if thresholded recall is zero.",
            "- Calibration cannot rescue a model whose score ordering is weak or inverted.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _safe_float(value: object) -> Optional[float]:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_or_none(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    return float(values.mean())


def _quantile(values: np.ndarray, q: float) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.quantile(values, q))


def _format_float(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _format_value(value: object) -> str:
    if value is None:
        return "NA"
    return str(value)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
