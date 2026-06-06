#!/usr/bin/env python
"""Print a read-only diagnosis for a simplified known-truth benchmark run."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_config, read_tsv, resolve_outdir, safe_float
from validate_result_tables import validate_tables


def _median(values: list[float]) -> str:
    if not values:
        return "NA"
    values = sorted(values)
    n = len(values)
    if n % 2:
        return f"{values[n // 2]:.6g}"
    return f"{((values[n // 2 - 1] + values[n // 2]) / 2):.6g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--benchmark-dir", default=None)
    args = parser.parse_args()
    config = read_config(args.config)
    benchmark_dir = resolve_outdir(config, args.benchmark_dir)
    truth = read_tsv(benchmark_dir / "truth" / "family_truth.tsv")
    babappa = read_tsv(benchmark_dir / "babappa_results.tsv")
    babappa_failures = read_tsv(benchmark_dir / "babappa_failures.tsv")
    absrel = read_tsv(benchmark_dir / "absrel_results.tsv")
    absrel_failures = read_tsv(benchmark_dir / "absrel_failures.tsv")
    scores = [safe_float(row.get("score"), 0.0) for row in babappa if row.get("score") not in {"", "NA", None}]
    validation = validate_tables([benchmark_dir / "babappa_results.tsv", benchmark_dir / "absrel_results.tsv", benchmark_dir / "method_comparison.tsv"])
    errors_by_table = validation.get("errors_by_table", {})
    babappa_malformed = int(errors_by_table.get(str(benchmark_dir / "babappa_results.tsv"), 0))
    absrel_malformed = int(errors_by_table.get(str(benchmark_dir / "absrel_results.tsv"), 0))
    method_malformed = int(errors_by_table.get(str(benchmark_dir / "method_comparison.tsv"), 0))
    truth_counts = Counter(row.get("truth_class", "") for row in truth)
    absrel_json_count = len(set((benchmark_dir / "absrel_json").glob("*.json")) | set((benchmark_dir / "absrel_json").glob("*.absrel.json")))
    truth_by_id = {row.get("family_id", ""): row for row in truth}
    absrel_positive = sum(1 for row in absrel if row.get("call") == "1")
    babappa_ood_null_false_calls = sum(1 for row in babappa if row.get("call") == "1" and truth_by_id.get(row.get("family_id", ""), {}).get("truth_class") == "ood_null")
    absrel_ood_null_false_calls = sum(1 for row in absrel if row.get("call") == "1" and truth_by_id.get(row.get("family_id", ""), {}).get("truth_class") == "ood_null")
    positive_truth = sum(1 for row in truth if row.get("truth_class") in {"positive", "ood_positive"})
    print("BABAPPA/aBSREL smoke diagnosis")
    print(f"benchmark_dir\t{benchmark_dir}")
    print(f"truth_family_count\t{len(truth)}")
    print("truth_class_counts\t" + ",".join(f"{key}:{truth_counts[key]}" for key in sorted(truth_counts)))
    print(f"babappa_result_rows\t{len(babappa)}")
    print(f"babappa_malformed_rows\t{babappa_malformed}")
    print(f"absrel_malformed_rows\t{absrel_malformed}")
    print(f"method_comparison_malformed_rows\t{method_malformed}")
    print(f"total_result_table_malformed_rows\t{validation.get('n_errors', 0)}")
    result_failure_count = sum(1 for row in babappa if row.get("status") == "failed")
    print(f"babappa_failures\t{max(len(babappa_failures), result_failure_count)}")
    print(f"babappa_positive_truth_count\t{positive_truth}")
    print(f"babappa_positive_calls\t{sum(1 for row in babappa if row.get('call') == '1')}")
    print(f"babappa_score_min\t{min(scores):.6g}" if scores else "babappa_score_min\tNA")
    print(f"babappa_score_median\t{_median(scores)}")
    print(f"babappa_score_max\t{max(scores):.6g}" if scores else "babappa_score_max\tNA")
    print(f"babappa_score_unique_count\t{len(set(scores)) if scores else 0}")
    print(f"babappa_score_audit_path\t{benchmark_dir / 'babappa_score_audit.json'}")
    print(f"absrel_json_count\t{absrel_json_count}")
    print(f"absrel_parsed_rows\t{len(absrel)}")
    print(f"absrel_positive_calls\t{absrel_positive}")
    print(f"absrel_failed_rows\t{len(absrel_failures)}")
    print(f"babappa_ood_null_false_call_count\t{babappa_ood_null_false_calls}")
    print(f"absrel_ood_null_false_call_count\t{absrel_ood_null_false_calls}")
    print(f"comparison_summary_path\t{benchmark_dir / 'benchmark_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
