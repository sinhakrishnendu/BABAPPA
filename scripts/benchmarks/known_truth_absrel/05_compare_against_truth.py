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


def _call_value(row: Dict[str, str]) -> int | None:
    value = row.get("call", "")
    if value in {"0", "1"}:
        return int(value)
    return None


def _method_rows(truth_rows: List[Dict[str, str]], method_name: str, result_rows: List[Dict[str, str]], score_field: str = "score") -> List[Dict[str, Any]]:
    by_id = {row["family_id"]: row for row in result_rows}
    rows: List[Dict[str, Any]] = []
    for truth in truth_rows:
        family_id = truth["family_id"]
        result = by_id.get(family_id, {})
        truth_positive = int(truth["truth_class"] in {"positive", "ood_positive"})
        call = _call_value(result)
        status = result.get("status", "missing")
        rows.append(
            {
                "family_id": family_id,
                "method": method_name,
                "truth_class": truth["truth_class"],
                "truth_positive": truth_positive,
                "expected_applicability": truth["expected_applicability"],
                "status": status,
                "score": safe_float(result.get(score_field), 0.0),
                "call": "NA" if call is None else call,
                "result_class": result.get("result_class", ""),
            }
        )
    return rows


def _summary_for(method: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluable = [row for row in rows if row["call"] in {0, 1}]
    labels = [int(row["truth_positive"]) for row in evaluable]
    calls = [int(row["call"]) for row in evaluable]
    scores = [float(row["score"]) for row in evaluable]
    metrics = classification_metrics(labels, calls) if evaluable else classification_metrics([], [])
    ood = [row for row in evaluable if row["expected_applicability"] == "out_of_domain"]
    ood_false_calls = sum(1 for row in ood if int(row["call"]) == 1 and row["truth_positive"] == 0)
    payload = {
        "method": method,
        "families_total": len(rows),
        "families_evaluable": len(evaluable),
        "failure_rate": (len(rows) - len(evaluable)) / len(rows) if rows else 0.0,
        "auroc": roc_auc(labels, scores) if evaluable else None,
        "auprc": average_precision(labels, scores) if evaluable else None,
        "ood_false_call_rate": ood_false_calls / len(ood) if ood else 0.0,
    }
    payload.update(metrics)
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
    if not absrel:
        absrel = [{"family_id": row["family_id"], "status": "pending_not_run", "call": "NA", "score": "0"} for row in truth]
    method_rows = _method_rows(truth, "BABAPPA", babappa) + _method_rows(truth, "aBSREL", absrel, score_field="call")
    summary_rows = [_summary_for("BABAPPA", [row for row in method_rows if row["method"] == "BABAPPA"]), _summary_for("aBSREL", [row for row in method_rows if row["method"] == "aBSREL"])]
    write_tsv(benchmark_dir / "method_comparison.tsv", method_rows, ["family_id", "method", "truth_class", "truth_positive", "expected_applicability", "status", "score", "call", "result_class"])
    write_tsv(benchmark_dir / "manuscript_table_babappa_vs_absrel.tsv", summary_rows, ["method", "families_total", "families_evaluable", "failure_rate", "auroc", "auprc", "precision", "recall_power", "specificity", "f1", "mcc", "fpr", "fnr", "empirical_fdr", "ood_false_call_rate"])
    write_json(benchmark_dir / "benchmark_summary.json", {"status": "ok", "methods": summary_rows})
    print(f"Wrote method comparison: {benchmark_dir / 'method_comparison.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
