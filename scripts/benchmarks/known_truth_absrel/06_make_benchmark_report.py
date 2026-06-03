#!/usr/bin/env python
"""Render Markdown reports for the simplified known-truth benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_config, read_tsv, resolve_outdir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--benchmark-dir", default=None)
    args = parser.parse_args()
    config = read_config(args.config)
    benchmark_dir = resolve_outdir(config, args.benchmark_dir)
    table = read_tsv(benchmark_dir / "manuscript_table_babappa_vs_absrel.tsv")
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
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This report supports known-truth simulation benchmarking. It does not make empirical discovery claims.",
            "",
        ]
    )
    (benchmark_dir / "method_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (benchmark_dir / "benchmark_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote benchmark report: {benchmark_dir / 'benchmark_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
