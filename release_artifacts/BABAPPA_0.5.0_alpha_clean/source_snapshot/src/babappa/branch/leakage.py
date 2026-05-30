"""Leakage audit for branch-conditioned site datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from babappa import __version__
from babappa.branch.dataset import FORBIDDEN_FEATURE_COLUMNS, SENSITIVE_CONTEXT_COLUMNS
from babappa.datasets.index import read_tsv, write_tsv

BRANCH_LEAKAGE_AUDIT_VERSION = __version__
OUTPUT_COLUMNS = ["column", "category", "detail"]
NAME_TOKENS = ("selected", "truth", "positive", "label")


def audit_branch_site_leakage(branch_site_dataset_dir: str | Path, outdir: str | Path) -> dict:
    """Audit branch-site features for target leakage and sensitive context."""
    dataset_dir = Path(branch_site_dataset_dir)
    if not dataset_dir.exists():
        raise ValueError(f"branch_site_dataset_dir does not exist: {dataset_dir}")
    features_path = dataset_dir / "branch_site_features.tsv"
    index_path = dataset_dir / "branch_site_dataset_index.json"
    if not features_path.exists():
        raise ValueError(f"branch_site_dataset_dir is missing branch_site_features.tsv: {dataset_dir}")
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    rows = read_tsv(features_path)
    fieldnames = _fieldnames(features_path)
    index_payload = _load_json(index_path)
    feature_columns = list(index_payload.get("feature_columns") or [])
    if not feature_columns:
        feature_columns = _infer_numeric_columns(rows)

    audit_rows: List[dict] = []
    warnings: List[str] = []
    forbidden = sorted(set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS)
    for column in forbidden:
        audit_rows.append({"column": column, "category": "forbidden_feature_column", "detail": "strict"})
    suspicious = []
    for column in feature_columns:
        lowered = column.lower()
        if column in SENSITIVE_CONTEXT_COLUMNS:
            continue
        if any(token in lowered for token in NAME_TOKENS):
            suspicious.append(column)
            audit_rows.append(
                {
                    "column": column,
                    "category": "suspicious_feature_name",
                    "detail": "contains selected/truth/positive/label token",
                }
            )
    sensitive = sorted(column for column in feature_columns if column in SENSITIVE_CONTEXT_COLUMNS)
    for column in sensitive:
        audit_rows.append(
            {
                "column": column,
                "category": "biologically_sensitive_context",
                "detail": "allowed branch/foreground context; interpret cautiously",
            }
        )
    near_perfect = _near_perfect_columns(rows, feature_columns)
    for item in near_perfect:
        audit_rows.append(
            {
                "column": item["column"],
                "category": "near_perfect_univariate",
                "detail": f"auroc={item['auroc']}",
            }
        )

    if forbidden:
        warnings.append("forbidden_branch_feature_columns_present")
    if near_perfect:
        warnings.append("near_perfect_branch_feature_predictor_present")
    if sensitive:
        warnings.append("foreground_context_columns_present")

    json_path = out_path / "branch_site_leakage_audit.json"
    columns_path = out_path / "branch_site_leakage_columns.tsv"
    markdown_path = out_path / "branch_site_leakage_audit.md"
    payload = {
        "branch_site_leakage_audit_version": BRANCH_LEAKAGE_AUDIT_VERSION,
        "branch_site_dataset_dir": str(dataset_dir),
        "feature_columns": feature_columns,
        "forbidden_columns_present": forbidden,
        "suspicious_columns_present": sorted(suspicious),
        "sensitive_context_columns_present": sensitive,
        "near_perfect_univariate_columns": near_perfect,
        "status": "warning" if warnings else "ok",
        "warnings": sorted(set(warnings)),
        "generated_files": {
            "json": str(json_path),
            "columns_tsv": str(columns_path),
            "markdown": str(markdown_path),
        },
        "interpretation": "Branch-site labels are valid targets only; site/gene labels must not enter predictive features.",
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


def validate_branch_site_leakage_dir(leakage_dir: str | Path) -> dict:
    """Validate branch-site leakage audit artifacts."""
    path = Path(leakage_dir)
    failures: List[str] = []
    warnings: List[str] = []
    payload = _load_json(path / "branch_site_leakage_audit.json", failures)
    rows = _read_tsv(path / "branch_site_leakage_columns.tsv", failures)
    markdown = path / "branch_site_leakage_audit.md"
    if not markdown.exists():
        failures.append(f"missing_file:{markdown}")
    elif not markdown.read_text(encoding="utf-8").strip():
        failures.append("empty_markdown")
    if payload.get("status") == "warning":
        warnings.extend(payload.get("warnings", []))
    return {
        "status": "fail" if failures else "ok",
        "n_rows": len(rows),
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _fieldnames(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t").fieldnames or [])


def _infer_numeric_columns(rows: List[dict]) -> List[str]:
    if not rows:
        return []
    selected = []
    for column in rows[0]:
        if column in {"y_branch_site", "branch_site_id", "family_id", "method", "split", "saturation_tier"}:
            continue
        if all(_to_float_or_none(row.get(column)) is not None for row in rows):
            selected.append(column)
    return selected


def _near_perfect_columns(rows: List[dict], feature_columns: List[str]) -> List[dict]:
    if not rows or "y_branch_site" not in rows[0]:
        return []
    y = np.array([_to_float(row.get("y_branch_site")) for row in rows], dtype=np.float64)
    result = []
    for column in feature_columns:
        if column in {"y_branch_site"}:
            continue
        values = [_to_float_or_none(row.get(column)) for row in rows]
        if any(value is None for value in values):
            continue
        auroc = _auroc_pairwise(y, np.array(values, dtype=np.float64))
        if auroc is not None and (auroc >= 0.99 or auroc <= 0.01):
            result.append({"column": column, "auroc": auroc})
    return sorted(result, key=lambda item: abs(float(item["auroc"]) - 0.5), reverse=True)


def _auroc_pairwise(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
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
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    positive_rank_sum = float(ranks[y_true == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


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


def _load_json(path: Path, failures: Optional[List[str]] = None) -> dict:
    if not path.exists():
        if failures is not None:
            failures.append(f"missing_file:{path}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        if failures is not None:
            failures.append(f"could_not_parse_json:{path}:{exc}")
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_tsv(path: Path, failures: List[str]) -> List[dict]:
    if not path.exists():
        failures.append(f"missing_file:{path}")
        return []
    try:
        return read_tsv(path)
    except OSError as exc:
        failures.append(f"could_not_read_tsv:{path}:{exc}")
        return []


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Branch-site leakage audit",
        "",
        f"- Dataset: `{payload.get('branch_site_dataset_dir')}`",
        f"- Status: {payload.get('status')}",
        "",
        "## Forbidden feature columns",
    ]
    forbidden = payload.get("forbidden_columns_present") or []
    lines.extend([f"- {column}" for column in forbidden] if forbidden else ["- none"])
    lines.extend(["", "## Sensitive foreground context", ""])
    sensitive = payload.get("sensitive_context_columns_present") or []
    lines.extend([f"- {column}" for column in sensitive] if sensitive else ["- none"])
    lines.extend(["", "## Near-perfect univariate predictors", ""])
    near = payload.get("near_perfect_univariate_columns") or []
    lines.extend([f"- {item['column']}: AUROC {item['auroc']}" for item in near] if near else ["- none"])
    lines.extend(["", "Branch/foreground context is allowed but biologically sensitive and must be reported.", ""])
    return "\n".join(lines)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
