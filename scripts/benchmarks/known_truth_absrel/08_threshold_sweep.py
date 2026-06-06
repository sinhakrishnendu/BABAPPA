#!/usr/bin/env python
"""Sweep BABAPPA score thresholds on a known-truth BABAPPA/aBSREL benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import average_precision, classification_metrics, read_config, read_tsv, resolve_outdir, roc_auc, safe_float, write_json, write_tsv


SWEEP_FIELDS = [
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
    "ood_diagnostic_no_call_rate",
    "in_domain_positive_calls",
    "ood_false_calls",
    "tp",
    "tn",
    "fp",
    "fn",
]
POLICY_FIELDS = [
    "policy",
    "status",
    "threshold",
    "positive_calls",
    "precision",
    "recall_power",
    "specificity",
    "f1",
    "mcc",
    "empirical_fdr",
    "ood_null_false_call_rate",
    "notes",
]


def _load_babappa_rows(benchmark_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for row in read_tsv(benchmark_dir / "method_comparison.tsv"):
        if row.get("method") != "BABAPPA":
            continue
        score = safe_float(row.get("score"), None)  # type: ignore[arg-type]
        if score is None or row.get("call") not in {"0", "1"}:
            continue
        rows.append(
            {
                "family_id": row["family_id"],
                "truth_class": row["truth_class"],
                "truth_positive": int(row["truth_positive"]),
                "expected_applicability": row["expected_applicability"],
                "score": float(score),
                "current_call": int(row["call"]),
                "current_status_class": row.get("status_class", ""),
            }
        )
    return rows


def _thresholds(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    unique = sorted(set(float(score) for score in scores), reverse=True)
    return [max(unique) + 1e-12] + unique + [min(unique) - 1e-12]


def _calls_for_threshold(rows: Sequence[Dict[str, Any]], threshold: float) -> List[int]:
    calls = []
    for row in rows:
        is_ood = row["expected_applicability"] == "out_of_domain" or str(row["truth_class"]).startswith("ood")
        calls.append(int((row["score"] >= threshold) and not is_ood))
    return calls


def _metrics_for_calls(rows: Sequence[Dict[str, Any]], calls: Sequence[int], threshold: float | str) -> Dict[str, Any]:
    labels = [int(row["truth_positive"]) for row in rows]
    metrics = classification_metrics(labels, list(calls))
    ood_null = [(row, call) for row, call in zip(rows, calls) if row["truth_class"] == "ood_null"]
    ood_all = [(row, call) for row, call in zip(rows, calls) if row["expected_applicability"] == "out_of_domain" or str(row["truth_class"]).startswith("ood")]
    ood_false_calls = sum(1 for _row, call in ood_null if call == 1)
    ood_no_calls = sum(1 for _row, call in ood_all if call == 0)
    in_domain_positive_calls = sum(1 for row, call in zip(rows, calls) if call == 1 and row["truth_positive"] == 1 and row["expected_applicability"] != "out_of_domain")
    payload: Dict[str, Any] = {
        "threshold": threshold,
        "positive_calls": sum(calls),
        "ood_null_false_call_rate": ood_false_calls / len(ood_null) if ood_null else 0.0,
        "ood_diagnostic_no_call_rate": ood_no_calls / len(ood_all) if ood_all else 0.0,
        "in_domain_positive_calls": in_domain_positive_calls,
        "ood_false_calls": ood_false_calls,
    }
    payload.update(metrics)
    return payload


def compute_threshold_sweep(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_metrics_for_calls(rows, _calls_for_threshold(rows, threshold), threshold) for threshold in _thresholds([row["score"] for row in rows])]


def _current_policy(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return _policy_row("ultra_conservative_current", "current_policy", _metrics_for_calls(rows, [row["current_call"] for row in rows], "package_default"), "current package call policy")


def _policy_row(policy: str, status: str, metric_row: Dict[str, Any] | None, notes: str) -> Dict[str, Any]:
    if metric_row is None:
        return {
            "policy": policy,
            "status": status,
            "threshold": "NA",
            "positive_calls": 0,
            "precision": 0.0,
            "recall_power": 0.0,
            "specificity": 0.0,
            "f1": 0.0,
            "mcc": 0.0,
            "empirical_fdr": 0.0,
            "ood_null_false_call_rate": 0.0,
            "notes": notes,
        }
    row = {field: metric_row.get(field, "") for field in POLICY_FIELDS if field not in {"policy", "status", "notes"}}
    row.update({"policy": policy, "status": status, "notes": notes})
    return row


def _select_fdr_policy(sweep: Sequence[Dict[str, Any]], target: float) -> Dict[str, Any] | None:
    candidates = [row for row in sweep if float(row["empirical_fdr"]) <= target and float(row["recall_power"]) > 0.0 and int(row["positive_calls"]) > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (float(row["recall_power"]), float(row["mcc"]), -float(row["empirical_fdr"]), int(row["positive_calls"])))


def _select_balanced_mcc(sweep: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not sweep:
        return None
    return max(sweep, key=lambda row: (float(row["mcc"]), float(row["recall_power"]), -float(row["empirical_fdr"])))


def _select_ood_safe(sweep: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    candidates = [row for row in sweep if float(row["ood_null_false_call_rate"]) == 0.0]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (float(row["recall_power"]), float(row["mcc"]), -float(row["empirical_fdr"])))


def recommend_policies(rows: Sequence[Dict[str, Any]], sweep: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    fdr_005 = _select_fdr_policy(sweep, 0.05)
    fdr_010 = _select_fdr_policy(sweep, 0.10)
    balanced = _select_balanced_mcc(sweep)
    ood_safe = _select_ood_safe(sweep)
    policies = [
        _current_policy(rows),
        _policy_row(
            "FDR_0.05_policy",
            "selected" if fdr_005 else "no_valid_threshold",
            fdr_005,
            "highest recall threshold with empirical FDR <= 0.05" if fdr_005 else "no threshold achieved empirical FDR <= 0.05 with nonzero recall",
        ),
        _policy_row(
            "FDR_0.10_policy",
            "selected" if fdr_010 else "no_valid_threshold",
            fdr_010,
            "highest recall threshold with empirical FDR <= 0.10" if fdr_010 else "no threshold achieved empirical FDR <= 0.10 with nonzero recall",
        ),
        _policy_row("balanced_MCC_policy", "selected" if balanced else "unavailable", balanced, "threshold maximizing MCC"),
        _policy_row("OOD_safe_policy", "selected" if ood_safe else "unavailable", ood_safe, "best recall among thresholds with OOD null false-call rate = 0"),
        {
            "policy": "diagnostic_score_only",
            "status": "available",
            "threshold": "NA",
            "positive_calls": "NA",
            "precision": "NA",
            "recall_power": "NA",
            "specificity": "NA",
            "f1": "NA",
            "mcc": "NA",
            "empirical_fdr": "NA",
            "ood_null_false_call_rate": "NA",
            "notes": "report AUROC/AUPRC and OOD abstention without binary positive calls",
        },
    ]
    labels = [row["truth_positive"] for row in rows]
    scores = [row["score"] for row in rows]
    return {
        "status": "ok",
        "n_rows": len(rows),
        "score_auroc": roc_auc(labels, scores),
        "score_auprc": average_precision(labels, scores),
        "policies": policies,
    }


def _render_sweep_md(sweep: Sequence[Dict[str, Any]], recommendations: Dict[str, Any]) -> str:
    lines = [
        "# BABAPPA Threshold Sweep",
        "",
        f"- rows: `{recommendations['n_rows']}`",
        f"- score AUROC: `{recommendations['score_auroc']}`",
        f"- score AUPRC: `{recommendations['score_auprc']}`",
        f"- thresholds tested: `{len(sweep)}`",
        "",
        "BABAPPA threshold sweeps preserve the OOD abstention gate. Out-of-domain families remain diagnostic/no-call even if their raw score is high.",
        "",
        "## Policy Recommendations",
        "",
        "| policy | status | threshold | positive calls | recall | FDR | OOD null false-call rate | notes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in recommendations["policies"]:
        lines.append(
            f"| {row['policy']} | {row['status']} | {row['threshold']} | {row['positive_calls']} | {row['recall_power']} | {row['empirical_fdr']} | {row['ood_null_false_call_rate']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "If no FDR-controlled threshold has nonzero recall, BABAPPA should be framed as a score-ranking/OOD-screening tool until additional calibration is available.",
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
    rows = _load_babappa_rows(benchmark_dir)
    if not rows:
        raise SystemExit(f"no BABAPPA rows found in {benchmark_dir / 'method_comparison.tsv'}")
    sweep = compute_threshold_sweep(rows)
    recommendations = recommend_policies(rows, sweep)
    write_tsv(benchmark_dir / "threshold_sweep_babappa.tsv", sweep, SWEEP_FIELDS)
    write_json(benchmark_dir / "threshold_policy_recommendation.json", recommendations)
    write_tsv(benchmark_dir / "threshold_policy_recommendation.tsv", recommendations["policies"], POLICY_FIELDS)
    markdown = _render_sweep_md(sweep, recommendations)
    (benchmark_dir / "threshold_sweep_babappa.md").write_text(markdown, encoding="utf-8")
    (benchmark_dir / "threshold_policy_recommendation.md").write_text(markdown, encoding="utf-8")
    print(f"Wrote threshold sweep: {benchmark_dir / 'threshold_sweep_babappa.tsv'}")
    print(f"thresholds_tested={len(sweep)} score_auroc={recommendations['score_auroc']} score_auprc={recommendations['score_auprc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
