#!/usr/bin/env python
"""Render Markdown reports for the simplified known-truth benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_config, read_tsv, resolve_outdir


def _frozen_policy_section_intro(frozen_policies):
    sources = {str(row.get("threshold_source", "")).lower() for row in frozen_policies}
    if any("posthoc paper" in source or "post-hoc paper" in source for source in sources):
        return [
            "",
            "## Frozen Paper-Derived Validation-Candidate Threshold Policy",
            "",
            "These candidate thresholds were selected after inspecting a previous paper-profile run. They must be applied unchanged to an independent validation profile before they can be claimed as final operating points. The validation profile is therefore an evaluation set, not a threshold-tuning set.",
            "",
            "| policy | type | threshold | positives | precision | recall | FDR | MCC | OOD false-call rate |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    return [
        "",
        "## Frozen Pilot-Selected Threshold Policy",
        "",
        "The calibrated policy was selected on the pilot profile. The paper profile must use this policy unchanged. The paper profile is therefore an evaluation set, not a threshold-tuning set.",
        "",
        "| policy | type | threshold | positives | precision | recall | FDR | MCC | OOD false-call rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--benchmark-dir", default=None)
    args = parser.parse_args()
    config = read_config(args.config)
    benchmark_dir = resolve_outdir(config, args.benchmark_dir)
    table = read_tsv(benchmark_dir / "manuscript_table_babappa_vs_absrel.tsv")
    threshold_policies = read_tsv(benchmark_dir / "threshold_policy_recommendation.tsv")
    operating_points = read_tsv(benchmark_dir / "operating_point_comparison.tsv")
    frozen_policies = read_tsv(benchmark_dir / "frozen_policy_results.tsv")
    summary_path = benchmark_dir / "benchmark_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"methods": []}
    lines = [
        "# Known-Truth BABAPPA/aBSREL Benchmark Summary",
        "",
        "Simulator labels are the ground truth in this benchmark. aBSREL is an external comparator against the same labels, not the truth source.",
        "",
        "BABAPPA is evaluated as a conservative, OOD-gated, simulation-trained support framework. The benchmark does not frame BABAPPA as a replacement for aBSREL.",
        "",
        "## Method Summary",
        "",
        "| method | evaluable | AUROC | AUPRC | precision | recall/power | specificity | empirical FDR | OOD false-call rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            "| {method} | {families_evaluable} | {auroc} | {auprc} | {precision} | {recall_power} | {specificity} | {empirical_fdr} | {ood_false_call_rate} |".format(**row)
        )
    lines.extend(["", "## Status Counts", "", "| method | pending | failed | positive | negative | diagnostic only | no call | inconclusive | note |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for row in table:
        lines.append(
            "| {method} | {pending_not_run} | {failed} | {positive} | {negative} | {diagnostic_only} | {no_call} | {inconclusive} | {no_positive_call_note} |".format(**row)
        )
    babappa_default = next((row for row in table if row.get("method") == "BABAPPA"), {})
    absrel_default = next((row for row in table if row.get("method") == "aBSREL"), {})
    if babappa_default:
        lines.extend(
            [
                "",
                "## BABAPPA Score Ranking Versus Default Calls",
                "",
                f"- BABAPPA score AUROC: `{babappa_default.get('auroc')}`",
                f"- BABAPPA score AUPRC: `{babappa_default.get('auprc')}`",
                f"- BABAPPA default positive calls: `{babappa_default.get('positive')}`",
                "",
            ]
        )
        if str(babappa_default.get("positive")) == "0":
            lines.extend(
                [
                    "BABAPPA current default is ultra-conservative. BABAPPA scores carry signal, but the default threshold produced zero positive calls in the pilot.",
                    "",
                ]
            )
    if absrel_default:
        lines.extend(
            [
                "aBSREL is more sensitive in this pilot, but its default calls must be interpreted against simulator truth rather than treated as truth.",
                f"- aBSREL empirical FDR: `{absrel_default.get('empirical_fdr')}`",
                f"- aBSREL OOD false-call rate: `{absrel_default.get('ood_false_call_rate')}`",
                "",
            ]
        )
    if threshold_policies:
        lines.extend(
            [
                "",
                "## Threshold Sweep",
                "",
                "BABAPPA threshold policies preserve the OOD abstention gate. Out-of-domain families remain diagnostic/no-call even when their raw score is high.",
                "",
                "| policy | status | threshold | positives | recall | FDR | OOD false-call rate | notes |",
                "|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in threshold_policies:
            lines.append(
                f"| {row['policy']} | {row['status']} | {row['threshold']} | {row['positive_calls']} | {row['recall_power']} | {row['empirical_fdr']} | {row['ood_null_false_call_rate']} | {row['notes']} |"
            )
    if frozen_policies:
        lines.extend(_frozen_policy_section_intro(frozen_policies))
        for row in frozen_policies:
            lines.append(
                f"| {row['policy_name']} | {row['policy_type']} | {row['threshold']} | {row['positive_calls']} | {row['precision']} | {row['recall_power']} | {row['empirical_fdr']} | {row['mcc']} | {row['ood_null_false_call_rate']} |"
            )
    if operating_points:
        lines.extend(
            [
                "",
                "## Operating Point Comparison",
                "",
                "| method | operating point | status | threshold | positives | precision | recall | FDR | OOD false-call rate |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in operating_points:
            lines.append(
                f"| {row['method']} | {row['operating_point']} | {row['status']} | {row['threshold']} | {row['positive_calls']} | {row['precision']} | {row['recall_power']} | {row['empirical_fdr']} | {row['ood_null_false_call_rate']} |"
            )
    lines.extend(["", "## OOD False-Call Definition", "", "OOD false-call rate is computed only among OOD null/stress-null families. OOD positive families are reported separately and are not included in the false-call denominator.", ""])
    warnings = summary.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(
        [
            "",
            "## Metric Interpretation",
            "",
            "AUROC and AUPRC are score-based metrics and are unavailable when scores are missing, constant, or contain only one truth class. Precision, recall/power, specificity, F1, MCC, FPR, FNR, and empirical FDR are call-based metrics.",
            "",
            "## Claim Boundary",
            "",
            "This report supports known-truth simulation benchmarking. It does not make empirical discovery claims.",
            "",
            "BABAPPA should not be claimed as a superior caller until threshold policy is calibrated on independent simulation regimes. If no threshold gives useful recall without high FDR, BABAPPA should be framed mainly as a score-ranking and OOD-screening tool.",
            "",
        ]
    )
    (benchmark_dir / "method_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (benchmark_dir / "benchmark_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote benchmark report: {benchmark_dir / 'benchmark_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
