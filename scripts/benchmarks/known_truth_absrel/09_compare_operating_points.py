#!/usr/bin/env python
"""Compare BABAPPA threshold operating points against aBSREL default."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import classification_metrics, read_config, read_tsv, resolve_outdir, safe_float, write_tsv


FIELDS = [
    "method",
    "operating_point",
    "status",
    "threshold",
    "positive_calls",
    "precision",
    "recall_power",
    "specificity",
    "f1",
    "mcc",
    "empirical_fdr",
    "fpr",
    "fnr",
    "ood_null_false_call_rate",
    "notes",
]


def _metric_row(method: str, operating_point: str, status: str, threshold: str, rows: List[Dict[str, Any]], calls: List[int], notes: str) -> Dict[str, Any]:
    labels = [int(row["truth_positive"]) for row in rows]
    metrics = classification_metrics(labels, calls)
    ood_null = [(row, call) for row, call in zip(rows, calls) if row["truth_class"] == "ood_null"]
    ood_false = sum(1 for _row, call in ood_null if call == 1)
    payload = {
        "method": method,
        "operating_point": operating_point,
        "status": status,
        "threshold": threshold,
        "positive_calls": sum(calls),
        "ood_null_false_call_rate": ood_false / len(ood_null) if ood_null else 0.0,
        "notes": notes,
    }
    payload.update(metrics)
    return payload


def _method_rows(benchmark_dir: Path, method: str) -> List[Dict[str, Any]]:
    rows = []
    for row in read_tsv(benchmark_dir / "method_comparison.tsv"):
        if row.get("method") != method:
            continue
        if row.get("call") not in {"0", "1"}:
            continue
        score = safe_float(row.get("score"), 0.0)
        rows.append(
            {
                "family_id": row["family_id"],
                "truth_class": row["truth_class"],
                "truth_positive": int(row["truth_positive"]),
                "expected_applicability": row["expected_applicability"],
                "score": float(score),
                "call": int(row["call"]),
            }
        )
    return rows


def _babappa_calls(rows: List[Dict[str, Any]], threshold: str) -> List[int]:
    value = safe_float(threshold, None)  # type: ignore[arg-type]
    if value is None:
        return [0 for _row in rows]
    calls = []
    for row in rows:
        is_ood = row["expected_applicability"] == "out_of_domain" or str(row["truth_class"]).startswith("ood")
        calls.append(int(row["score"] >= value and not is_ood))
    return calls


def _current_calls(rows: List[Dict[str, Any]]) -> List[int]:
    return [int(row["call"]) for row in rows]


def build_operating_points(benchmark_dir: Path) -> List[Dict[str, Any]]:
    babappa_rows = _method_rows(benchmark_dir, "BABAPPA")
    absrel_rows = _method_rows(benchmark_dir, "aBSREL")
    policies = {row["policy"]: row for row in read_tsv(benchmark_dir / "threshold_policy_recommendation.tsv")}
    rows: List[Dict[str, Any]] = []
    if babappa_rows:
        rows.append(_metric_row("BABAPPA", "current_default", "current_policy", "package_default", babappa_rows, _current_calls(babappa_rows), "current ultra-conservative package calls"))
        for policy_name in ["FDR_0.05_policy", "FDR_0.10_policy", "balanced_MCC_policy", "OOD_safe_policy"]:
            policy = policies.get(policy_name, {})
            status = policy.get("status", "missing")
            threshold = policy.get("threshold", "NA")
            calls = _babappa_calls(babappa_rows, threshold) if status == "selected" else [0 for _row in babappa_rows]
            rows.append(_metric_row("BABAPPA", policy_name, status, threshold, babappa_rows, calls, policy.get("notes", "")))
    if absrel_rows:
        rows.append(_metric_row("aBSREL", "default", "external_comparator", "default", absrel_rows, _current_calls(absrel_rows), "aBSREL default calls against simulator truth; comparator, not truth source"))
    return rows


def _render_md(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Operating Point Comparison",
        "",
        "Simulator labels are the ground truth. aBSREL is an external comparator, not a truth source.",
        "",
        "| method | operating point | status | threshold | positives | precision | recall | FDR | OOD false-call rate | notes |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['operating_point']} | {row['status']} | {row['threshold']} | {row['positive_calls']} | {row['precision']} | {row['recall_power']} | {row['empirical_fdr']} | {row['ood_null_false_call_rate']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "BABAPPA should not be claimed as a superior positive caller from this table alone. Threshold policies must be calibrated and confirmed on independent simulation regimes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--benchmark-dir", default=None)
    args = parser.parse_args()
    if args.config:
        config = read_config(args.config)
        benchmark_dir = resolve_outdir(config, args.benchmark_dir)
    elif args.benchmark_dir:
        benchmark_dir = Path(args.benchmark_dir)
        if not benchmark_dir.is_absolute():
            benchmark_dir = Path.cwd() / benchmark_dir
    else:
        raise SystemExit("provide --config or --benchmark-dir")
    rows = build_operating_points(benchmark_dir)
    write_tsv(benchmark_dir / "operating_point_comparison.tsv", rows, FIELDS)
    (benchmark_dir / "operating_point_comparison.md").write_text(_render_md(rows), encoding="utf-8")
    print(f"Wrote operating point comparison: {benchmark_dir / 'operating_point_comparison.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
