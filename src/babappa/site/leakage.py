"""Leakage audit for site-level BABAPPA datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv

SITE_LEAKAGE_AUDIT_VERSION = __version__
FORBIDDEN_COLUMNS = {
    "selected_sites",
    "n_selected_sites",
    "positive_sites",
    "oracle_selected_sites",
    "site_labels",
    "gene_label",
    "truth_label",
    "true_label",
    "positive_family",
    "is_positive",
    "y_gene",
    "y",
    "label",
}
SUSPICIOUS_BUT_ALLOWED = {
    "foreground_taxon",
    "foreground_taxon_present",
    "foreground_taxon_index",
}
NAME_TOKENS = ("selected", "truth", "positive", "label")
OUTPUT_COLUMNS = ["column", "category", "detail"]


def audit_site_dataset_leakage(site_dataset_dir: str | Path, outdir: str | Path) -> dict:
    """Audit a site-level dataset for target leakage risks."""
    dataset_dir = Path(site_dataset_dir)
    out_path = Path(outdir)
    if not dataset_dir.exists():
        raise ValueError(f"site_dataset_dir does not exist: {dataset_dir}")
    features_path = dataset_dir / "site_features.tsv"
    if not features_path.exists():
        raise ValueError(f"site_dataset_dir is missing site_features.tsv: {dataset_dir}")
    out_path.mkdir(parents=True, exist_ok=True)
    rows = read_tsv(features_path)
    fieldnames = _fieldnames(features_path)
    warnings: List[str] = []
    audit_rows: List[dict] = []

    forbidden_present = sorted(column for column in fieldnames if column in FORBIDDEN_COLUMNS)
    for column in forbidden_present:
        audit_rows.append(
            {"column": column, "category": "forbidden_feature_column", "detail": "strict"}
        )

    suspicious_names = []
    for column in fieldnames:
        lowered = column.lower()
        if column == "y_site" or column in SUSPICIOUS_BUT_ALLOWED:
            continue
        if any(token in lowered for token in NAME_TOKENS):
            suspicious_names.append(column)
            audit_rows.append(
                {
                    "column": column,
                    "category": "suspicious_name",
                    "detail": "contains selected/truth/positive/label token",
                }
            )

    near_perfect = _near_perfect_columns(rows)
    for item in near_perfect:
        audit_rows.append(
            {
                "column": item["column"],
                "category": "near_perfect_univariate",
                "detail": f"auroc={item['auroc']}",
            }
        )

    if forbidden_present:
        warnings.append("forbidden_site_feature_columns_present")
    if near_perfect:
        warnings.append("near_perfect_site_feature_predictor_present")

    json_path = out_path / "site_leakage_audit.json"
    columns_path = out_path / "site_leakage_columns.tsv"
    markdown_path = out_path / "site_leakage_audit.md"
    payload = {
        "site_leakage_audit_version": SITE_LEAKAGE_AUDIT_VERSION,
        "site_dataset_dir": str(dataset_dir),
        "forbidden_columns_present": forbidden_present,
        "suspicious_columns_present": sorted(suspicious_names),
        "suspicious_but_allowed_columns_present": sorted(
            column for column in fieldnames if column in SUSPICIOUS_BUT_ALLOWED
        ),
        "near_perfect_univariate_columns": near_perfect,
        "status": "warning" if warnings else "ok",
        "warnings": sorted(set(warnings)),
        "generated_files": {
            "json": str(json_path),
            "columns_tsv": str(columns_path),
            "markdown": str(markdown_path),
        },
        "interpretation": (
            "Site-level oracle labels are valid targets, but selected/truth/label-like "
            "columns must not be used as predictive inputs."
        ),
    }
    _write_json(json_path, payload)
    write_tsv(columns_path, audit_rows, OUTPUT_COLUMNS)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(out_path),
        "json": str(json_path),
        "columns_tsv": str(columns_path),
        "markdown": str(markdown_path),
        "leakage_status": payload["status"],
        "warnings": payload["warnings"],
    }


def _fieldnames(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or [])


def _near_perfect_columns(rows: List[dict]) -> List[dict]:
    if not rows or "y_site" not in rows[0]:
        return []
    y = np.array([_to_float(row.get("y_site")) for row in rows], dtype=np.float64)
    result = []
    for column in rows[0]:
        if column == "y_site" or column in SUSPICIOUS_BUT_ALLOWED:
            continue
        values = [_to_float_or_none(row.get(column)) for row in rows]
        if any(value is None for value in values):
            continue
        x = np.array([float(value) for value in values], dtype=np.float64)
        auroc = _auroc_pairwise(y, x)
        if auroc is None:
            continue
        if auroc >= 0.99 or auroc <= 0.01:
            result.append({"column": column, "auroc": auroc})
    return sorted(result, key=lambda item: abs(float(item["auroc"]) - 0.5), reverse=True)


def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float:
    parsed = _to_float_or_none(value)
    return 0.0 if parsed is None else parsed


def _auroc_pairwise(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """Compute AUROC from average ranks in O(n log n)."""
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    cursor = 0
    while cursor < scores.size:
        end = cursor + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[cursor]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        ranks[order[cursor:end]] = average_rank
        cursor = end
    positive_rank_sum = float(ranks[y_true == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Site leakage audit",
        "",
        f"- Dataset: `{payload.get('site_dataset_dir')}`",
        f"- Status: {payload.get('status')}",
        "",
        "## Forbidden columns",
        "",
    ]
    forbidden = payload.get("forbidden_columns_present") or []
    lines.extend([f"- {column}" for column in forbidden] if forbidden else ["- none"])
    lines.extend(["", "## Suspicious columns", ""])
    suspicious = payload.get("suspicious_columns_present") or []
    lines.extend([f"- {column}" for column in suspicious] if suspicious else ["- none"])
    lines.extend(["", "## Near-perfect univariate predictors", ""])
    near = payload.get("near_perfect_univariate_columns") or []
    if near:
        lines.extend(f"- {item['column']}: AUROC {item['auroc']}" for item in near)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Oracle site labels are valid supervised targets, not inference inputs.",
            "- Columns containing selected/truth/positive/label tokens should be treated as leakage unless explicitly audited.",
            "- Foreground context columns are suspicious but allowed metadata/context, not target labels.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
