#!/usr/bin/env python
"""Compare BABAPPA and aBSREL benchmark outputs against simulator truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import average_precision, classification_metrics, read_config, read_tsv, resolve_outdir, roc_auc, safe_float, write_json, write_tsv
from validate_result_tables import validate_tables


def _call_value(row: Dict[str, str]) -> int | None:
    value = row.get("call", "")
    if value in {"0", "1"}:
        return int(value)
    return None


def _status_class(result: Dict[str, str], call: int | None) -> str:
    status = result.get("status", "missing")
    explicit = result.get("status_class", "")
    if explicit in {"method_positive", "method_negative", "method_pending", "method_failed", "diagnostic_only", "no_call", "inconclusive"}:
        return explicit
    if status in {"pending_not_run", "missing"}:
        return "pending_not_run"
    if status in {"failed", "parse_failed", "tool_missing"} or status.startswith("warning_"):
        return "method_failed" if status == "failed" else "inconclusive"
    if call == 1:
        return "method_positive"
    if call == 0:
        return "method_negative"
    return "inconclusive"


def _result_class(result: Dict[str, str], call: int | None, status_class: str) -> str:
    explicit = result.get("result_class", "")
    if explicit:
        return explicit
    if call == 1:
        return "positive"
    if call == 0:
        return "negative"
    if status_class == "diagnostic_only":
        return "diagnostic_only"
    if status_class == "method_failed":
        return "failed"
    if status_class in {"pending_not_run", "method_pending", "no_call"}:
        return "no_call"
    return "inconclusive"


def _method_rows(truth_rows: List[Dict[str, str]], method_name: str, result_rows: List[Dict[str, str]], score_field: str = "score") -> List[Dict[str, Any]]:
    by_id = {row["family_id"]: row for row in result_rows}
    rows: List[Dict[str, Any]] = []
    for truth in truth_rows:
        family_id = truth["family_id"]
        result = by_id.get(family_id, {})
        truth_positive = int(truth["truth_class"] in {"positive", "ood_positive"})
        call = _call_value(result)
        status = result.get("status", "missing")
        status_class = _status_class(result, call)
        raw_score = result.get(score_field, "")
        score = "NA" if call is None else safe_float(raw_score, 0.0)
        result_class = _result_class(result, call, status_class)
        rows.append(
            {
                "family_id": family_id,
                "method": method_name,
                "truth_class": truth["truth_class"],
                "truth_positive": truth_positive,
                "expected_applicability": truth["expected_applicability"],
                "status": status,
                "status_class": status_class,
                "score": score,
                "call": "NA" if call is None else call,
                "result_class": result_class,
            }
        )
    return rows


def _summary_for(method: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluable = [row for row in rows if row["call"] in {0, 1}]
    labels = [int(row["truth_positive"]) for row in evaluable]
    calls = [int(row["call"]) for row in evaluable]
    scores = [float(row["score"]) for row in evaluable]
    score_metrics_available = len(set(scores)) > 1 and len(set(labels)) > 1 if evaluable else False
    metrics = classification_metrics(labels, calls) if evaluable else classification_metrics([], [])
    ood_null = [row for row in evaluable if row["expected_applicability"] == "out_of_domain" and row["truth_class"] == "ood_null"]
    ood_positive = [row for row in evaluable if row["expected_applicability"] == "out_of_domain" and row["truth_class"] == "ood_positive"]
    ood_false_calls = sum(1 for row in ood_null if int(row["call"]) == 1)
    ood_positive_calls = sum(1 for row in ood_positive if int(row["call"]) == 1)
    status_counts = {
        "pending_not_run": sum(1 for row in rows if row["status_class"] in {"pending_not_run", "method_pending"}),
        "failed": sum(1 for row in rows if row["status_class"] == "method_failed"),
        "positive": sum(1 for row in rows if row["status_class"] == "method_positive"),
        "negative": sum(1 for row in rows if row["status_class"] == "method_negative"),
        "diagnostic_only": sum(1 for row in rows if row["status_class"] == "diagnostic_only"),
        "no_call": sum(1 for row in rows if row["status_class"] == "no_call"),
        "inconclusive": sum(1 for row in rows if row["status_class"] == "inconclusive"),
    }
    if status_counts["positive"] == 0 and len(evaluable):
        no_positive_note = "no positive calls made"
    elif status_counts["positive"] == 0:
        no_positive_note = "no evaluated positive calls; method pending/failed"
    else:
        no_positive_note = ""
    payload = {
        "method": method,
        "families_total": len(rows),
        "families_evaluable": len(evaluable),
        "failure_rate": (len(rows) - len(evaluable)) / len(rows) if rows else 0.0,
        "score_metrics_available": score_metrics_available,
        "auroc": roc_auc(labels, scores) if score_metrics_available else "unavailable",
        "auprc": average_precision(labels, scores) if score_metrics_available else "unavailable",
        "warnings": "constant_or_missing_scores" if not score_metrics_available else "",
        "ood_null_denominator": len(ood_null),
        "ood_null_false_calls": ood_false_calls,
        "ood_false_call_rate": ood_false_calls / len(ood_null) if ood_null else 0.0,
        "ood_positive_denominator": len(ood_positive),
        "ood_positive_calls": ood_positive_calls,
        "no_positive_call_note": no_positive_note,
    }
    payload.update(metrics)
    payload.update(status_counts)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--benchmark-dir", default=None)
    args = parser.parse_args()

    config = read_config(args.config)
    benchmark_dir = resolve_outdir(config, args.benchmark_dir)
    truth = read_tsv(benchmark_dir / "truth" / "family_truth.tsv")
    if not truth:
        raise SystemExit("missing truth/family_truth.tsv")
    babappa = read_tsv(benchmark_dir / "babappa_results.tsv")
    absrel = read_tsv(benchmark_dir / "absrel_results.tsv")
    absrel_warning = ""
    if not absrel:
        absrel_warning = "aBSREL smoke results are absent. Run: bash benchmarks/known_truth_absrel/run_absrel_smoke.sh"
        print(absrel_warning)
        absrel = [{"family_id": row["family_id"], "status": "pending_not_run", "status_class": "method_pending", "call": "NA", "score": "0"} for row in truth]
    method_rows = _method_rows(truth, "BABAPPA", babappa) + _method_rows(truth, "aBSREL", absrel, score_field="call")
    summary_rows = [_summary_for("BABAPPA", [row for row in method_rows if row["method"] == "BABAPPA"]), _summary_for("aBSREL", [row for row in method_rows if row["method"] == "aBSREL"])]
    write_tsv(benchmark_dir / "method_comparison.tsv", method_rows, ["family_id", "method", "truth_class", "truth_positive", "expected_applicability", "status", "status_class", "score", "call", "result_class"])
    write_tsv(
        benchmark_dir / "manuscript_table_babappa_vs_absrel.tsv",
        summary_rows,
        [
            "method",
            "families_total",
            "families_evaluable",
            "pending_not_run",
            "failed",
            "positive",
            "negative",
            "diagnostic_only",
            "no_call",
            "inconclusive",
            "failure_rate",
            "score_metrics_available",
            "auroc",
            "auprc",
            "warnings",
            "precision",
            "recall_power",
            "specificity",
            "f1",
            "mcc",
            "fpr",
            "fnr",
            "empirical_fdr",
            "ood_null_denominator",
            "ood_null_false_calls",
            "ood_false_call_rate",
            "ood_positive_denominator",
            "ood_positive_calls",
            "no_positive_call_note",
        ],
    )
    validation_payload = validate_tables([benchmark_dir / "babappa_results.tsv", benchmark_dir / "absrel_results.tsv", benchmark_dir / "method_comparison.tsv"])
    write_json(benchmark_dir / "result_table_validation.json", validation_payload)
    write_tsv(benchmark_dir / "result_table_validation.tsv", validation_payload["errors"], ["file", "row", "reason"])
    smoke_status = _smoke_status(benchmark_dir, summary_rows, absrel, absrel_warning, validation_payload)
    write_json(benchmark_dir / "benchmark_summary.json", {"status": "ok", "smoke_status": smoke_status, "methods": summary_rows, "warnings": [absrel_warning] if absrel_warning else []})
    write_json(benchmark_dir / "smoke_status.json", smoke_status)
    (benchmark_dir / "smoke_status.md").write_text(_render_smoke_status_md(smoke_status), encoding="utf-8")
    print(f"Wrote method comparison: {benchmark_dir / 'method_comparison.tsv'}")
    print(f"smoke_status={smoke_status['status']}")
    return 0


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _smoke_status(benchmark_dir: Path, summary_rows: List[Dict[str, Any]], absrel_rows: List[Dict[str, str]], absrel_warning: str, validation_payload: Dict[str, Any]) -> Dict[str, Any]:
    audit = _read_json(benchmark_dir / "babappa_score_audit.json")
    if audit.get("status") != "pass":
        reasons = audit.get("reasons") or []
        if "schema_invalid" in reasons:
            status = "smoke_fail_babappa_schema"
        elif "scores_constant" in reasons or "scores_all_zero" in reasons or "scores_missing" in reasons:
            status = "smoke_fail_babappa_constant_scores"
        else:
            status = "smoke_fail_babappa_schema"
        return {"status": status, "ready_for_pilot": False, "reason": ",".join(reasons), "babappa_audit": str(benchmark_dir / "babappa_score_audit.json")}
    if validation_payload.get("status") != "pass":
        return {
            "status": "smoke_fail_result_table_schema",
            "ready_for_pilot": False,
            "reason": f"result_table_validation_errors:{validation_payload.get('n_errors')}",
            "result_table_validation": str(benchmark_dir / "result_table_validation.json"),
        }
    if not absrel_rows:
        return {"status": "smoke_fail_absrel_missing", "ready_for_pilot": False, "reason": absrel_warning}
    if all(row.get("status") == "pending_not_run" for row in absrel_rows):
        return {"status": "smoke_pass_absrel_pending_allowed", "ready_for_pilot": True, "reason": "BABAPPA passed; aBSREL pending is explicit"}
    return {"status": "smoke_pass", "ready_for_pilot": True, "reason": "BABAPPA passed and aBSREL rows are present"}


def _render_smoke_status_md(payload: Dict[str, Any]) -> str:
    return "\n".join([
        "# Known-Truth Smoke Status",
        "",
        f"- status: `{payload.get('status')}`",
        f"- ready for pilot: `{payload.get('ready_for_pilot')}`",
        f"- reason: `{payload.get('reason', '')}`",
        "",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
