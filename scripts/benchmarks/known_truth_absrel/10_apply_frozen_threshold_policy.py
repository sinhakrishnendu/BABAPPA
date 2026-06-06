#!/usr/bin/env python
"""Apply a frozen BABAPPA threshold policy to a known-truth benchmark run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import average_precision, classification_metrics, read_config, read_tsv, repo_root, resolve_outdir, roc_auc, safe_float, write_tsv


RESULT_FIELDS = [
    "method",
    "policy_id",
    "policy_name",
    "policy_type",
    "threshold_source",
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
    "ood_null_denominator",
    "ood_null_false_calls",
    "ood_null_false_call_rate",
    "ood_positive_denominator",
    "ood_positive_calls",
    "ood_positive_call_rate",
    "score_auroc",
    "score_auprc",
    "notes",
]


def _coerce_value(value: str) -> Any:
    cleaned = value.strip()
    if cleaned.lower() == "true":
        return True
    if cleaned.lower() == "false":
        return False
    return cleaned


def read_threshold_policy(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"threshold policy missing: {path}")
    payload: Dict[str, Any] = {"policies": {}}
    current_policy: str | None = None
    in_policies = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if indent == 0 and line == "policies:":
            in_policies = True
            current_policy = None
            continue
        if indent == 0 and ":" in line:
            key, value = line.split(":", 1)
            payload[key.strip()] = _coerce_value(value.strip())
            continue
        if in_policies and indent == 2 and line.endswith(":"):
            current_policy = line[:-1].strip()
            payload["policies"][current_policy] = {"policy_id": current_policy}
            continue
        if in_policies and indent >= 4 and current_policy and ":" in line:
            key, value = line.split(":", 1)
            payload["policies"][current_policy][key.strip()] = _coerce_value(value.strip())
    return payload


def _load_babappa_rows(benchmark_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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
            }
        )
    return rows


def _calls_for_policy(rows: List[Dict[str, Any]], policy: Dict[str, Any]) -> List[int] | None:
    policy_type = policy.get("policy_type")
    if policy_type == "score_ranking":
        return None
    if policy_type == "current_default":
        return [int(row["current_call"]) for row in rows]
    threshold = safe_float(policy.get("threshold"), None)  # type: ignore[arg-type]
    if threshold is None:
        return None
    calls: List[int] = []
    for row in rows:
        is_ood = row["expected_applicability"] == "out_of_domain" or str(row["truth_class"]).startswith("ood")
        calls.append(int(row["score"] >= threshold and not is_ood))
    return calls


def _metrics_for_policy(rows: List[Dict[str, Any]], policy_id: str, policy: Dict[str, Any]) -> Dict[str, Any]:
    labels = [int(row["truth_positive"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    calls = _calls_for_policy(rows, policy)
    base = {
        "method": "BABAPPA",
        "policy_id": policy_id,
        "policy_name": policy.get("policy_name", policy_id),
        "policy_type": policy.get("policy_type", ""),
        "threshold_source": policy.get("threshold_source", ""),
        "threshold": policy.get("threshold", "NA"),
        "score_auroc": roc_auc(labels, scores),
        "score_auprc": average_precision(labels, scores),
        "notes": policy.get("notes", ""),
    }
    if calls is None:
        base.update(
            {
                "positive_calls": "NA",
                "precision": "NA",
                "recall_power": "NA",
                "specificity": "NA",
                "f1": "NA",
                "mcc": "NA",
                "empirical_fdr": "NA",
                "fpr": "NA",
                "fnr": "NA",
                "ood_null_denominator": "NA",
                "ood_null_false_calls": "NA",
                "ood_null_false_call_rate": "NA",
                "ood_positive_denominator": "NA",
                "ood_positive_calls": "NA",
                "ood_positive_call_rate": "NA",
            }
        )
        return base
    metrics = classification_metrics(labels, calls)
    ood_null = [(row, call) for row, call in zip(rows, calls) if row["truth_class"] == "ood_null"]
    ood_positive = [(row, call) for row, call in zip(rows, calls) if row["truth_class"] == "ood_positive"]
    ood_null_false_calls = sum(1 for _row, call in ood_null if call == 1)
    ood_positive_calls = sum(1 for _row, call in ood_positive if call == 1)
    base.update(metrics)
    base.update(
        {
            "positive_calls": sum(calls),
            "ood_null_denominator": len(ood_null),
            "ood_null_false_calls": ood_null_false_calls,
            "ood_null_false_call_rate": ood_null_false_calls / len(ood_null) if ood_null else 0.0,
            "ood_positive_denominator": len(ood_positive),
            "ood_positive_calls": ood_positive_calls,
            "ood_positive_call_rate": ood_positive_calls / len(ood_positive) if ood_positive else 0.0,
        }
    )
    return base


def apply_frozen_policy(benchmark_dir: Path, policy_path: Path) -> List[Dict[str, Any]]:
    rows = _load_babappa_rows(benchmark_dir)
    if not rows:
        raise ValueError(f"no BABAPPA rows found in {benchmark_dir / 'method_comparison.tsv'}")
    policy = read_threshold_policy(policy_path)
    policy_rows = []
    for policy_id, item in policy.get("policies", {}).items():
        policy_rows.append(_metrics_for_policy(rows, policy_id, item))
    return policy_rows


def _render_md(rows: List[Dict[str, Any]], policy_path: Path) -> str:
    sources = {str(row.get("threshold_source", "")).lower() for row in rows}
    if any("posthoc paper" in source or "post-hoc paper" in source for source in sources):
        threshold_source = "post-hoc paper threshold sweep"
        rule = "apply these candidate thresholds unchanged to an independent validation run; do not recalibrate on validation results"
        closing = "These candidate thresholds were selected after inspecting a paper-profile run. They remain validation hypotheses until an independent validation profile confirms their behavior."
    else:
        threshold_source = "pilot threshold sweep"
        rule = "apply these thresholds unchanged; do not recalibrate on paper results"
        closing = "The calibrated policy was selected on the pilot profile. The paper profile is an evaluation set, not a threshold-tuning set."
    lines = [
        "# Frozen BABAPPA Threshold Policy Results",
        "",
        f"- policy file: `{policy_path}`",
        f"- threshold source: {threshold_source}",
        f"- rule: {rule}",
        "",
        "| policy | threshold | positives | precision | recall | FDR | MCC | OOD false-call rate | notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy_name']} | {row['threshold']} | {row['positive_calls']} | {row['precision']} | {row['recall_power']} | {row['empirical_fdr']} | {row['mcc']} | {row['ood_null_false_call_rate']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            closing,
            "",
        ]
    )
    return "\n".join(lines)


def _default_policy_path() -> Path:
    return repo_root() / "benchmarks" / "known_truth_absrel" / "threshold_policy.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--threshold-policy", type=Path, default=None)
    args = parser.parse_args()
    if args.config:
        benchmark_dir = resolve_outdir(read_config(args.config), args.benchmark_dir)
    elif args.benchmark_dir:
        benchmark_dir = Path(args.benchmark_dir)
        if not benchmark_dir.is_absolute():
            benchmark_dir = Path.cwd() / benchmark_dir
    else:
        raise SystemExit("provide --config or --benchmark-dir")
    policy_path = args.threshold_policy or _default_policy_path()
    rows = apply_frozen_policy(benchmark_dir, policy_path)
    write_tsv(benchmark_dir / "frozen_policy_results.tsv", rows, RESULT_FIELDS)
    write_tsv(benchmark_dir / "manuscript_table_frozen_policy.tsv", rows, RESULT_FIELDS)
    (benchmark_dir / "frozen_policy_results.md").write_text(_render_md(rows, policy_path), encoding="utf-8")
    print(f"Wrote frozen policy results: {benchmark_dir / 'frozen_policy_results.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
