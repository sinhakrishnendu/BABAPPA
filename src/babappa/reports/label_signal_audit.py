"""Audit whether dataset-level features carry learnable label signal."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv

LABEL_SIGNAL_AUDIT_VERSION = __version__
SUMMARY_FIELDNAMES = [
    "scope",
    "saturation_tier",
    "feature",
    "n",
    "positives",
    "negatives",
    "positive_mean",
    "negative_mean",
    "difference",
    "pooled_std",
    "standardized_difference",
    "auroc",
    "auroc_distance_from_0_5",
]


@dataclass(frozen=True)
class LabelSignalAuditConfig:
    """Configuration for label-signal auditing."""

    dataset_dir: str
    outdir: str
    label_column: str = "gene_label"
    saturation_column: str = "saturation_tier"
    method_column: str = "method"
    split_column: str = "split"

    def __post_init__(self) -> None:
        dataset_path = Path(self.dataset_dir)
        if not dataset_path.exists():
            raise ValueError(f"dataset_dir does not exist: {dataset_path}")
        if not (dataset_path / "features.tsv").exists():
            raise ValueError(f"dataset_dir is missing features.tsv: {dataset_path}")
        if not (dataset_path / "splits.tsv").exists():
            raise ValueError(f"dataset_dir is missing splits.tsv: {dataset_path}")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def audit_label_signal(config: LabelSignalAuditConfig) -> dict:
    """Audit univariate feature-label signal in a BABAPPA dataset directory."""
    dataset_dir = Path(config.dataset_dir)
    outdir = Path(config.outdir)
    features = read_tsv(dataset_dir / "features.tsv")
    splits = read_tsv(dataset_dir / "splits.tsv")
    split_lookup = _split_lookup(splits, config)
    rows = [_merge_split(row, split_lookup, config) for row in features]
    numeric_features = _numeric_feature_columns(rows, config)
    warnings: List[str] = []
    if not numeric_features:
        warnings.append("no_numeric_features_detected")

    summary_rows: List[dict] = []
    for feature in numeric_features:
        summary = _feature_summary(rows, feature, "all", "all", config)
        if summary is not None:
            summary_rows.append(summary)
        tiers = sorted({row.get(config.saturation_column) or "unknown" for row in rows})
        for tier in tiers:
            tier_rows = [
                row
                for row in rows
                if (row.get(config.saturation_column) or "unknown") == tier
            ]
            summary = _feature_summary(tier_rows, feature, "saturation_tier", tier, config)
            if summary is not None:
                summary_rows.append(summary)

    top_standardized = sorted(
        [row for row in summary_rows if row["scope"] == "all"],
        key=lambda row: _abs_or_negative(row.get("standardized_difference")),
        reverse=True,
    )[:20]
    top_auroc = sorted(
        [row for row in summary_rows if row["scope"] == "all"],
        key=lambda row: _float_or_negative(row.get("auroc_distance_from_0_5")),
        reverse=True,
    )[:20]
    class_balance = {
        "by_split": _class_balance(rows, config.split_column, config),
        "by_saturation_tier": _class_balance(rows, config.saturation_column, config),
        "by_method": _class_balance(rows, config.method_column, config),
    }
    interpretation = _interpretation(top_auroc, summary_rows)
    payload = {
        "label_signal_audit_version": LABEL_SIGNAL_AUDIT_VERSION,
        "dataset_dir": str(dataset_dir),
        "n_rows": len(rows),
        "n_numeric_features": len(numeric_features),
        "class_balance": class_balance,
        "top_features_by_abs_standardized_difference": top_standardized,
        "top_features_by_auroc_distance": top_auroc,
        "warnings": sorted(set(warnings)),
        "interpretation": interpretation,
        "generated_files": {
            "json": str(outdir / "label_signal_audit.json"),
            "features": str(outdir / "label_signal_features.tsv"),
            "markdown": str(outdir / "label_signal_audit.md"),
        },
    }
    _write_json(outdir / "label_signal_audit.json", payload)
    write_tsv(outdir / "label_signal_features.tsv", summary_rows, SUMMARY_FIELDNAMES)
    (outdir / "label_signal_audit.md").write_text(
        _render_markdown(payload),
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "outdir": str(outdir),
        "json": str(outdir / "label_signal_audit.json"),
        "features": str(outdir / "label_signal_features.tsv"),
        "markdown": str(outdir / "label_signal_audit.md"),
        "warnings": payload["warnings"],
        "interpretation": interpretation,
    }


def _split_lookup(splits: List[dict], config: LabelSignalAuditConfig) -> Dict[tuple, str]:
    lookup = {}
    for row in splits:
        key = (
            row.get("family_id", ""),
            row.get(config.method_column, ""),
            row.get("tensor_file", ""),
        )
        lookup[key] = row.get(config.split_column, "")
        lookup[(key[0], key[1], "")] = row.get(config.split_column, "")
    return lookup


def _merge_split(row: dict, split_lookup: Dict[tuple, str], config: LabelSignalAuditConfig) -> dict:
    merged = dict(row)
    key = (
        row.get("family_id", ""),
        row.get(config.method_column, ""),
        row.get("tensor_file", ""),
    )
    split = split_lookup.get(key) or split_lookup.get((key[0], key[1], "")) or ""
    merged[config.split_column] = split
    return merged


def _numeric_feature_columns(rows: List[dict], config: LabelSignalAuditConfig) -> List[str]:
    excluded = {
        "family_id",
        "merged_family_id",
        "original_family_id",
        "source_dataset",
        "tensor_file",
        "tensor_meta_file",
        "labels_file",
        config.label_column,
        config.saturation_column,
        config.method_column,
        config.split_column,
        "foreground_taxon",
    }
    columns = sorted({key for row in rows for key in row.keys() if key not in excluded})
    numeric = []
    for column in columns:
        values = [_safe_float(row.get(column)) for row in rows]
        if any(value is not None and math.isfinite(value) for value in values):
            numeric.append(column)
    return numeric


def _feature_summary(
    rows: List[dict],
    feature: str,
    scope: str,
    tier: str,
    config: LabelSignalAuditConfig,
) -> Optional[dict]:
    values = []
    labels = []
    for row in rows:
        value = _safe_float(row.get(feature))
        label = _safe_int(row.get(config.label_column))
        if value is None or label is None:
            continue
        values.append(value)
        labels.append(label)
    if not values:
        return None
    y = np.asarray(labels, dtype=np.int32)
    x = np.asarray(values, dtype=np.float64)
    positive = x[y == 1]
    negative = x[y == 0]
    if positive.size == 0 or negative.size == 0:
        auroc = None
        positive_mean = float(positive.mean()) if positive.size else None
        negative_mean = float(negative.mean()) if negative.size else None
        difference = None
        pooled_std = None
        standardized = None
    else:
        positive_mean = float(positive.mean())
        negative_mean = float(negative.mean())
        difference = positive_mean - negative_mean
        pooled_std = float(np.sqrt((positive.var() + negative.var()) / 2.0))
        standardized = None if pooled_std == 0 else difference / pooled_std
        auroc = _univariate_auroc(y, x)
    return {
        "scope": scope,
        "saturation_tier": tier,
        "feature": feature,
        "n": int(len(values)),
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
        "positive_mean": positive_mean,
        "negative_mean": negative_mean,
        "difference": difference,
        "pooled_std": pooled_std,
        "standardized_difference": standardized,
        "auroc": auroc,
        "auroc_distance_from_0_5": None if auroc is None else abs(auroc - 0.5),
    }


def _univariate_auroc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum_pos = float(ranks[y_true == 1].sum())
    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def _class_balance(rows: List[dict], column: str, config: LabelSignalAuditConfig) -> dict:
    counts: Dict[str, dict] = {}
    for row in rows:
        key = row.get(column) or "unknown"
        label = _safe_int(row.get(config.label_column))
        entry = counts.setdefault(key, {"n": 0, "positives": 0, "negatives": 0})
        entry["n"] += 1
        if label == 1:
            entry["positives"] += 1
        elif label == 0:
            entry["negatives"] += 1
    return dict(sorted(counts.items()))


def _interpretation(top_auroc: List[dict], rows: List[dict]) -> str:
    best = top_auroc[0] if top_auroc else {}
    best_distance = _float_or_negative(best.get("auroc_distance_from_0_5"))
    tier_rows = [row for row in rows if row.get("scope") == "saturation_tier"]
    tier_signal = any(
        _float_or_negative(row.get("auroc_distance_from_0_5")) >= 0.1
        for row in tier_rows
    )
    if best_distance < 0.1:
        text = (
            "No simple tensor-derived feature has AUROC far from 0.5; the current "
            "global feature table weakly encodes the label."
        )
    else:
        text = (
            "At least one simple tensor-derived feature shows univariate signal; "
            "inspect the top features before interpreting neural failures as data "
            "unlearnability."
        )
    text += (
        " Weak feature signal does not prove tensors lack signal, but suggests "
        "sparse or deep representations may be required."
    )
    if tier_signal:
        text += " Signal varies by saturation tier, consistent with saturation as a confounder."
    return text


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Label-signal audit",
        "",
        "## Dataset",
        "",
        f"- Dataset directory: {payload['dataset_dir']}",
        f"- Rows: {payload['n_rows']}",
        f"- Numeric features: {payload['n_numeric_features']}",
        "",
        "## Class balance",
        "",
    ]
    for group_name, counts in payload["class_balance"].items():
        lines.append(f"- {group_name}: {counts}")
    lines.extend(["", "## Top univariate signals", ""])
    for row in payload["top_features_by_auroc_distance"][:10]:
        lines.append(
            "- {feature}: AUROC={auroc}, standardized_difference={std}".format(
                feature=row["feature"],
                auroc=_format_float(row.get("auroc")),
                std=_format_float(row.get("standardized_difference")),
            )
        )
    lines.extend(["", "## Saturation-tier signal", ""])
    tier_rows = [
        row
        for row in payload["top_features_by_abs_standardized_difference"]
        if row.get("scope") == "saturation_tier"
    ]
    if not tier_rows:
        lines.append("- See label_signal_features.tsv for per-tier feature summaries.")
    else:
        for row in tier_rows[:10]:
            lines.append(f"- {row['saturation_tier']} / {row['feature']}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "## Limitations",
            "",
            "- This audit uses global tensor-derived features, not full tensor sequence information.",
            "- Weak feature signal does not prove the tensors are unlearnable.",
            "- Saturation-tier labels come from the current simulator and are not final observed diagnostics.",
            "",
        ]
    )
    return "\n".join(lines)


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> Optional[int]:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _float_or_negative(value: object) -> float:
    parsed = _safe_float(value)
    return -1.0 if parsed is None else parsed


def _abs_or_negative(value: object) -> float:
    parsed = _safe_float(value)
    return -1.0 if parsed is None else abs(parsed)


def _format_float(value: object) -> str:
    parsed = _safe_float(value)
    return "NA" if parsed is None else f"{parsed:.4f}"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
