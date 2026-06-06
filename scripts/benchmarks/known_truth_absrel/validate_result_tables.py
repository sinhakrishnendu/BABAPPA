#!/usr/bin/env python
"""Validate simplified known-truth benchmark result TSV files."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_config, read_tsv, resolve_outdir, safe_float, write_json, write_tsv


METHOD_COMPARISON_COLUMNS = [
    "family_id",
    "method",
    "truth_class",
    "truth_positive",
    "expected_applicability",
    "status",
    "status_class",
    "score",
    "call",
    "result_class",
]
BABAPPA_RESULT_COLUMNS = [
    "family_id",
    "method",
    "truth_class",
    "truth_positive",
    "expected_applicability",
    "status",
    "status_class",
    "score",
    "call",
    "result_class",
    "diagnostic_only",
    "applicability",
    "failure_reason",
]
VALID_METHODS = {"BABAPPA", "aBSREL"}
VALID_TRUTH = {"null", "positive", "ood_null", "ood_positive", "ambiguous"}
VALID_CALLS = {"0", "1", "NA"}
VALID_RESULT_CLASSES = {
    "diagnostic_positive",
    "diagnostic_negative",
    "diagnostic_only",
    "positive",
    "negative",
    "failed",
    "no_call",
    "inconclusive",
}
ABSREL_RESULT_COLUMNS = [
    "family_id",
    "status",
    "positive_count",
    "call",
    "p_value",
    "notes",
]
PENDING_OR_FAILED_STATUSES = {
    "failed",
    "missing",
    "pending_not_run",
    "parse_failed",
    "tool_missing",
    "warning_missing_official_field",
}


def _raw_shape_errors(path: Path) -> List[str]:
    if not path.exists():
        return [f"missing file: {path}"]
    errors: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows:
        return [f"empty file: {path}"]
    width = len(rows[0])
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != width:
            errors.append(f"{path}: line {line_number} has {len(row)} columns, expected {width}")
    return errors


def _status_requires_na(status: str) -> bool:
    return status in PENDING_OR_FAILED_STATUSES or status.startswith("warning_")


def _validate_rows(path: Path, required: List[str], method_required: bool, exact_header: bool = True) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    errors.extend({"file": str(path), "row": "", "reason": reason} for reason in _raw_shape_errors(path))
    header: List[str] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            try:
                header = next(csv.reader(handle, delimiter="\t"))
            except StopIteration:
                header = []
    if required and exact_header and header != required:
        errors.append({"file": str(path), "row": "", "reason": "header does not match expected schema"})
    rows = read_tsv(path)
    if not rows:
        return errors
    missing = [column for column in required if column not in header]
    if missing:
        errors.append({"file": str(path), "row": "", "reason": "missing required columns: " + ",".join(missing)})
        return errors
    for index, row in enumerate(rows, start=2):
        if method_required and row.get("method") not in VALID_METHODS:
            errors.append({"file": str(path), "row": index, "reason": f"invalid method: {row.get('method')}"})
        if row.get("truth_class") and row.get("truth_class") not in VALID_TRUTH:
            errors.append({"file": str(path), "row": index, "reason": f"invalid truth_class: {row.get('truth_class')}"})
        if "call" in required:
            call = str(row.get("call") if row.get("call") is not None else "")
            if call in VALID_RESULT_CLASSES:
                errors.append({"file": str(path), "row": index, "reason": "result_class appears shifted into call column"})
            if call not in VALID_CALLS:
                errors.append({"file": str(path), "row": index, "reason": f"call is not one of 0,1,NA: {call}"})
        if "result_class" in required:
            result_class = str(row.get("result_class") if row.get("result_class") is not None else "")
            if not result_class:
                errors.append({"file": str(path), "row": index, "reason": "empty result_class"})
            elif result_class not in VALID_RESULT_CLASSES:
                errors.append({"file": str(path), "row": index, "reason": f"invalid result_class: {result_class}"})
        status = str(row.get("status") or "")
        if row.get("status") == "ok":
            if "score" in required and safe_float(row.get("score"), None) is None:  # type: ignore[arg-type]
                errors.append({"file": str(path), "row": index, "reason": "ok row score is not numeric"})
            if row.get("call") not in {"0", "1"}:
                errors.append({"file": str(path), "row": index, "reason": "ok row call is not 0/1"})
            if "positive_count" in required and row.get("positive_count") not in {"NA", ""} and safe_float(row.get("positive_count"), None) is None:  # type: ignore[arg-type]
                errors.append({"file": str(path), "row": index, "reason": "positive_count is not numeric/NA"})
        elif _status_requires_na(status):
            if "score" in required and row.get("score") != "NA":
                errors.append({"file": str(path), "row": index, "reason": "failed/pending row score must be NA"})
            if "call" in required and row.get("call") != "NA":
                errors.append({"file": str(path), "row": index, "reason": "failed/pending row call must be NA"})
    return errors


def _family_coverage_errors(path: Path, rows: List[Dict[str, str]], expected_family_ids: Sequence[str] | None) -> List[Dict[str, Any]]:
    if not expected_family_ids or not rows:
        return []
    expected = set(expected_family_ids)
    errors: List[Dict[str, Any]] = []
    name = path.name
    if name in {"babappa_results.tsv", "absrel_results.tsv"}:
        seen: Dict[str, int] = {}
        for row in rows:
            family_id = row.get("family_id", "")
            seen[family_id] = seen.get(family_id, 0) + 1
        missing = sorted(expected - set(seen))
        extra = sorted(set(seen) - expected)
        duplicates = sorted(family_id for family_id, count in seen.items() if count > 1)
        for family_id in missing:
            errors.append({"file": str(path), "row": "", "reason": f"missing family row: {family_id}"})
        for family_id in extra:
            errors.append({"file": str(path), "row": "", "reason": f"unexpected family row: {family_id}"})
        for family_id in duplicates:
            errors.append({"file": str(path), "row": "", "reason": f"duplicate family row: {family_id}"})
    elif name == "method_comparison.tsv":
        seen: Dict[tuple[str, str], int] = {}
        by_family: Dict[str, set[str]] = {}
        for row in rows:
            family_id = row.get("family_id", "")
            method = row.get("method", "")
            seen[(family_id, method)] = seen.get((family_id, method), 0) + 1
            by_family.setdefault(family_id, set()).add(method)
        for family_id in sorted(expected - set(by_family)):
            errors.append({"file": str(path), "row": "", "reason": f"missing family rows: {family_id}"})
        for family_id in sorted(set(by_family) - expected):
            errors.append({"file": str(path), "row": "", "reason": f"unexpected family rows: {family_id}"})
        for family_id, method in sorted(key for key, count in seen.items() if count > 1):
            errors.append({"file": str(path), "row": "", "reason": f"duplicate family/method row: {family_id}/{method}"})
        for family_id in sorted(expected & set(by_family)):
            missing_methods = sorted(VALID_METHODS - by_family.get(family_id, set()))
            for method in missing_methods:
                errors.append({"file": str(path), "row": "", "reason": f"missing method row: {family_id}/{method}"})
    return errors


def validate_tables(tables: List[Path], expected_family_ids: Sequence[str] | None = None) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    by_table: Dict[str, int] = {}
    for table in tables:
        name = table.name
        if name == "babappa_results.tsv":
            table_errors = _validate_rows(table, BABAPPA_RESULT_COLUMNS, method_required=True)
        elif name == "absrel_results.tsv":
            table_errors = _validate_rows(table, ABSREL_RESULT_COLUMNS, method_required=False)
        elif name == "method_comparison.tsv":
            table_errors = _validate_rows(table, METHOD_COMPARISON_COLUMNS, method_required=True)
        else:
            table_errors = _validate_rows(table, [], method_required=False, exact_header=False)
        if table.name in {"babappa_results.tsv", "absrel_results.tsv", "method_comparison.tsv"}:
            table_errors.extend(_family_coverage_errors(table, read_tsv(table), expected_family_ids))
        errors.extend(table_errors)
        by_table[str(table)] = len(table_errors)
    return {"status": "fail" if errors else "pass", "n_errors": len(errors), "errors": errors, "errors_by_table": by_table}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--table", type=Path, action="append", default=[])
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    tables: List[Path] = []
    outdir: Path
    if args.config:
        config = read_config(args.config)
        benchmark_dir = resolve_outdir(config, args.benchmark_dir)
        tables.extend([benchmark_dir / "babappa_results.tsv", benchmark_dir / "absrel_results.tsv", benchmark_dir / "method_comparison.tsv"])
        outdir = args.outdir or benchmark_dir
        manifest_rows = read_tsv(benchmark_dir / "manifest.tsv")
        expected_family_ids = [row["family_id"] for row in manifest_rows if row.get("family_id")]
    else:
        if not args.table:
            raise SystemExit("provide --config or at least one --table")
        tables.extend(args.table)
        outdir = args.outdir or Path(".")
        expected_family_ids = None
    payload = validate_tables(tables, expected_family_ids)
    write_json(outdir / "result_table_validation.json", payload)
    write_tsv(outdir / "result_table_validation.tsv", payload["errors"], ["file", "row", "reason"])
    print(f"result_table_validation={payload['status']} errors={payload['n_errors']}")
    return 1 if payload["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
