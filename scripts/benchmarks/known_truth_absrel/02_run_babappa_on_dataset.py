#!/usr/bin/env python
"""Run BABAPPA direct prediction on a known-truth benchmark dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import deterministic_score, read_config, read_tsv, repo_root, resolve_outdir, run_command, safe_float, write_tsv


def _read_gene_summary(path: Path) -> Dict[str, str]:
    rows = read_tsv(path)
    return rows[0] if rows else {}


def _mock_row(row: Dict[str, str], seed: int) -> Dict[str, Any]:
    truth_positive = row["truth_class"] in {"positive", "ood_positive"}
    ood = row["expected_applicability"] == "out_of_domain"
    score = deterministic_score(seed, row["family_id"], truth_positive, ood)
    call = int(score >= 0.5 and not ood)
    return {
        "family_id": row["family_id"],
        "status": "ok",
        "applicability": row["expected_applicability"],
        "score": f"{score:.6f}",
        "call": call,
        "result_class": "positive" if call else "negative_or_abstained",
        "outdir": "mock_for_tests",
        "notes": "test-only mock backend",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--model-package", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--mock-for-tests", action="store_true")
    args = parser.parse_args()

    config = read_config(args.config)
    benchmark_dir = resolve_outdir(config, args.benchmark_dir)
    manifest = read_tsv(benchmark_dir / "manifest.tsv")
    if not manifest:
        raise SystemExit(f"missing or empty manifest: {benchmark_dir / 'manifest.tsv'}")
    model_package = args.model_package or config.get("model_package", "deployable_model_conservative_branch_site_100k_mps")
    device = args.device or config.get("device", "auto")
    null_replicates = config.get("babappa_null_replicates", "0")
    seed = int(config.get("seed", "42"))
    results_dir = benchmark_dir / "babappa_scores"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []

    for row in manifest:
        family_id = row["family_id"]
        if args.mock_for_tests:
            result_rows.append(_mock_row(row, seed))
            continue
        family_outdir = results_dir / family_id
        cmd = [
            "babappa",
            "predict-branch-sites",
            "--msa",
            str(benchmark_dir / row["codon_fasta"]),
            "--tree",
            str(benchmark_dir / row["tree"]),
            "--foreground",
            row["foreground"],
            "--outdir",
            str(family_outdir),
            "--model-package",
            model_package,
            "--device",
            device,
            "--allow-missing-start-codon",
            "--null-replicates",
            str(null_replicates),
        ]
        completed = run_command(cmd, cwd=repo_root())
        (family_outdir / "command_stdout.txt").parent.mkdir(parents=True, exist_ok=True)
        (family_outdir / "command_stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (family_outdir / "command_stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            error_text = " ".join((completed.stderr.strip() or completed.stdout.strip()).split())[-500:]
            failure = {"family_id": family_id, "status": "failed", "returncode": completed.returncode, "error": error_text}
            failure_rows.append(failure)
            result_rows.append({"family_id": family_id, "status": "failed", "applicability": "", "score": "NA", "call": "NA", "result_class": "failed", "outdir": str(family_outdir), "notes": failure["error"]})
            if not args.continue_on_failure:
                break
            continue
        summary = _read_gene_summary(family_outdir / "gene_summary.tsv")
        score = safe_float(summary.get("max_gene_support"), 0.0)
        native_class = summary.get("babappa_native_result_class") or summary.get("result_class") or "not_classified"
        call = int("positive" in native_class.lower() or safe_float(summary.get("n_called_positive"), 0.0) > 0)
        result_rows.append(
            {
                "family_id": family_id,
                "status": "ok",
                "applicability": summary.get("applicability", ""),
                "score": score,
                "call": call,
                "result_class": native_class,
                "outdir": str(family_outdir),
                "notes": "",
            }
        )

    fields = ["family_id", "status", "applicability", "score", "call", "result_class", "outdir", "notes"]
    write_tsv(benchmark_dir / "babappa_results.tsv", result_rows, fields)
    write_tsv(benchmark_dir / "babappa_failures.tsv", failure_rows, ["family_id", "status", "returncode", "error"])
    (benchmark_dir / "babappa_run_manifest.json").write_text(json.dumps({"status": "ok", "families": len(result_rows), "failures": len(failure_rows)}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote BABAPPA benchmark results: {benchmark_dir / 'babappa_results.tsv'}")
    return 0 if not failure_rows or args.continue_on_failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
