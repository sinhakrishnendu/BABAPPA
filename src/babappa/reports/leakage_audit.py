"""Truth/leakage audit for BABAPPA dataset feature tables."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv
from babappa.leakage_policy import (
    METADATA_NOT_FEATURE_COLUMNS,
    STRICT_LEAKAGE_COLUMNS,
    is_metadata_column,
    is_strict_leakage_column,
    is_suspicious_feature_name,
    should_exclude_from_feature_model,
)

LEAKAGE_AUDIT_VERSION = __version__
LEAKAGE_TSV_FIELDNAMES = ["source", "column", "category", "detail"]


@dataclass(frozen=True)
class LeakageAuditConfig:
    """Configuration for feature-table leakage auditing."""

    dataset_dir: str
    outdir: str
    label_column: str = "gene_label"
    split_column: str = "split"
    method_column: str = "method"
    saturation_column: str = "saturation_tier"

    def __post_init__(self) -> None:
        dataset_path = Path(self.dataset_dir)
        if not dataset_path.exists():
            raise ValueError(f"dataset_dir does not exist: {dataset_path}")
        if not (dataset_path / "features.tsv").exists():
            raise ValueError(f"dataset_dir is missing features.tsv: {dataset_path}")
        if not (dataset_path / "splits.tsv").exists():
            raise ValueError(f"dataset_dir is missing splits.tsv: {dataset_path}")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def audit_leakage(config: LeakageAuditConfig) -> dict:
    """Audit a dataset directory for leakage-prone columns."""
    dataset_dir = Path(config.dataset_dir)
    outdir = Path(config.outdir)
    feature_rows = read_tsv(dataset_dir / "features.tsv")
    split_rows = read_tsv(dataset_dir / "splits.tsv")
    feature_columns = _columns(feature_rows)
    split_columns = _columns(split_rows)
    all_columns = sorted(set(feature_columns) | set(split_columns))

    strict_features = _matching(feature_columns, is_strict_leakage_column)
    strict_splits = _matching(split_columns, is_strict_leakage_column)
    suspicious = _matching(all_columns, is_suspicious_feature_name)
    metadata = _matching(all_columns, is_metadata_column)
    near_perfect = _near_perfect_univariate_columns(feature_rows, config.label_column)
    safe_candidates = [
        column
        for column in _numeric_columns(feature_rows)
        if not should_exclude_from_feature_model(column)
        and column not in {row["column"] for row in near_perfect}
    ]
    recommended_excluded = sorted(
        set(strict_features)
        | set(strict_splits)
        | set(suspicious)
        | {row["column"] for row in near_perfect}
        | {
            column
            for column in all_columns
            if should_exclude_from_feature_model(column)
        }
    )
    warnings: List[str] = []
    if strict_features or strict_splits:
        warnings.append("strict_leakage_columns_present")
    if near_perfect:
        warnings.append("near_perfect_univariate_predictors_present")
    suspicious_near_perfect = [
        row
        for row in near_perfect
        if is_suspicious_feature_name(row["column"])
        or is_strict_leakage_column(row["column"])
    ]
    status = "warning" if (strict_features or strict_splits or suspicious_near_perfect) else "ok"

    leakage_rows = _leakage_rows(
        strict_features,
        strict_splits,
        suspicious,
        metadata,
        near_perfect,
    )
    payload = {
        "leakage_audit_version": LEAKAGE_AUDIT_VERSION,
        "dataset_dir": str(dataset_dir),
        "strict_leakage_columns_present": {
            "features": strict_features,
            "splits": strict_splits,
        },
        "suspicious_columns_present": suspicious,
        "metadata_columns_present": metadata,
        "near_perfect_univariate_columns": near_perfect,
        "safe_numeric_feature_candidates": safe_candidates,
        "recommended_excluded_columns": recommended_excluded,
        "status": status,
        "warnings": sorted(set(warnings)),
        "generated_files": {
            "json": str(outdir / "leakage_audit.json"),
            "columns": str(outdir / "leakage_columns.tsv"),
            "markdown": str(outdir / "leakage_audit.md"),
        },
    }
    _write_json(outdir / "leakage_audit.json", payload)
    write_tsv(outdir / "leakage_columns.tsv", leakage_rows, LEAKAGE_TSV_FIELDNAMES)
    (outdir / "leakage_audit.md").write_text(
        _render_markdown(payload),
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "leakage_status": status,
        "outdir": str(outdir),
        "json": str(outdir / "leakage_audit.json"),
        "columns": str(outdir / "leakage_columns.tsv"),
        "markdown": str(outdir / "leakage_audit.md"),
        "warnings": payload["warnings"],
        "recommended_excluded_columns": recommended_excluded,
    }


def _columns(rows: List[dict]) -> List[str]:
    if not rows:
        return []
    columns = []
    for row in rows:
        for column in row.keys():
            if column not in columns:
                columns.append(column)
    return columns


def _matching(columns: List[str], predicate) -> List[str]:
    return sorted(column for column in columns if predicate(column))


def _numeric_columns(rows: List[dict]) -> List[str]:
    columns = _columns(rows)
    numeric = []
    for column in columns:
        values = [_safe_float(row.get(column)) for row in rows]
        if any(value is not None for value in values):
            numeric.append(column)
    return numeric


def _near_perfect_univariate_columns(rows: List[dict], label_column: str) -> List[dict]:
    labels = [_safe_int(row.get(label_column)) for row in rows]
    if not labels or any(label is None for label in labels):
        return []
    y = np.asarray(labels, dtype=np.int32)
    if int(y.sum()) == 0 or int(y.sum()) == len(y):
        return []
    near_perfect = []
    for column in _numeric_columns(rows):
        if column == label_column:
            continue
        values = [_safe_float(row.get(column)) for row in rows]
        if any(value is None for value in values):
            continue
        auroc = _auroc_pairwise(y, np.asarray(values, dtype=np.float64))
        if auroc is not None and (auroc >= 0.99 or auroc <= 0.01):
            near_perfect.append(
                {
                    "column": column,
                    "auroc": auroc,
                    "suspicious_name": is_suspicious_feature_name(column),
                }
            )
    return near_perfect


def _auroc_pairwise(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    positive_scores = scores[y_true == 1]
    negative_scores = scores[y_true == 0]
    if positive_scores.size == 0 or negative_scores.size == 0:
        return None
    wins = 0.0
    total = positive_scores.size * negative_scores.size
    for positive_score in positive_scores:
        wins += float((positive_score > negative_scores).sum())
        wins += 0.5 * float((positive_score == negative_scores).sum())
    return wins / total


def _leakage_rows(
    strict_features: List[str],
    strict_splits: List[str],
    suspicious: List[str],
    metadata: List[str],
    near_perfect: List[dict],
) -> List[dict]:
    rows = []
    for column in strict_features:
        rows.append(
            {
                "source": "features.tsv",
                "column": column,
                "category": "strict_leakage",
                "detail": "truth-derived or label-like column",
            }
        )
    for column in strict_splits:
        rows.append(
            {
                "source": "splits.tsv",
                "column": column,
                "category": "strict_leakage",
                "detail": "truth-derived or label-like column",
            }
        )
    for column in suspicious:
        rows.append(
            {
                "source": "features_or_splits",
                "column": column,
                "category": "suspicious_name",
                "detail": "name contains selected/truth/label/positive/foreground",
            }
        )
    for column in metadata:
        rows.append(
            {
                "source": "features_or_splits",
                "column": column,
                "category": "metadata_not_feature",
                "detail": "metadata column should not be used by feature baselines",
            }
        )
    for row in near_perfect:
        rows.append(
            {
                "source": "features.tsv",
                "column": row["column"],
                "category": "near_perfect_univariate",
                "detail": f"AUROC={row['auroc']:.6f}",
            }
        )
    rows.sort(key=lambda row: (row["column"], row["category"], row["source"]))
    return rows


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Leakage audit",
        "",
        "## Dataset",
        "",
        f"- Dataset directory: {payload['dataset_dir']}",
        f"- Status: {payload['status']}",
        "",
        "## Strict leakage columns",
        "",
        f"- features.tsv: {_format_list(payload['strict_leakage_columns_present']['features'])}",
        f"- splits.tsv: {_format_list(payload['strict_leakage_columns_present']['splits'])}",
        "",
        "## Suspicious columns",
        "",
        f"- {_format_list(payload['suspicious_columns_present'])}",
        "",
        "## Near-perfect univariate predictors",
        "",
    ]
    near_perfect = payload.get("near_perfect_univariate_columns") or []
    if near_perfect:
        for row in near_perfect:
            lines.append(f"- {row['column']}: AUROC={row['auroc']:.4f}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Recommended excluded columns",
            "",
            f"- {_format_list(payload['recommended_excluded_columns'])}",
            "",
            "## Safe feature candidates",
            "",
            f"- {_format_list(payload['safe_numeric_feature_candidates'])}",
            "",
            "## Interpretation",
            "",
            "- n_selected_sites and selected_sites are truth-derived and must not be used for predictive training.",
            "- Neural tensor models may not use these features directly, but baselines or feature-based models must exclude them.",
            "- Near-perfect univariate predictors should be treated as possible leakage until proven otherwise.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_list(values: List[str]) -> str:
    return ", ".join(str(value) for value in values) if values else "none"


def _safe_float(value: object) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> Optional[int]:
    parsed = _safe_float(value)
    return None if parsed is None else int(parsed)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
