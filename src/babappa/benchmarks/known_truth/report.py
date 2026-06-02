"""Report generation for BABAPPA known-truth benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv

from .truth_schema import write_json


@dataclass(frozen=True)
class KnownTruthBenchmarkReportConfig:
    benchmark_dir: str
    outdir: str


def make_known_truth_benchmark_report(config: KnownTruthBenchmarkReportConfig) -> Dict[str, Any]:
    benchmark_dir = Path(config.benchmark_dir)
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    truth_manifest = benchmark_dir / "simulated_families" / "benchmark_truth_manifest.tsv"
    evaluation = _read_json(benchmark_dir / "evaluation" / "evaluation_summary.json")
    calibration = _read_json(benchmark_dir / "calibration_evaluation" / "calibration_evaluation.json")
    truth_rows = read_tsv(truth_manifest) if truth_manifest.exists() else []
    counts: Dict[str, int] = {}
    for row in truth_rows:
        counts[row["truth_class"]] = counts.get(row["truth_class"], 0) + 1
    payload = {
        "known_truth_benchmark_report_version": __version__,
        "status": "ok" if evaluation else "partial",
        "benchmark_dir": str(benchmark_dir),
        "n_families": len(truth_rows),
        "truth_class_counts": counts,
        "evaluation": evaluation.get("gene_level", {}),
        "branch_site_level": evaluation.get("branch_site_level", {}),
        "calibration": calibration.get("fdr_power", []),
        "claim_boundary": (
            "This known-truth simulation benchmark supports simulation validation and conservative "
            "method claims. It does not support empirical discovery claims by itself."
        ),
    }
    write_json(outdir / "known_truth_benchmark_report.json", payload)
    (outdir / "known_truth_benchmark_report.md").write_text(_render_report_md(payload), encoding="utf-8")
    _write_manuscript_tables(outdir, payload)
    return {"status": payload["status"], "outdir": str(outdir), "n_families": len(truth_rows)}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manuscript_tables(outdir: Path, payload: Dict[str, Any]) -> None:
    eval_row = {
        "n_families": payload["n_families"],
        "gene_auroc": payload["evaluation"].get("auroc", ""),
        "gene_auprc": payload["evaluation"].get("auprc", ""),
        "precision": payload["evaluation"].get("precision", ""),
        "recall": payload["evaluation"].get("recall", ""),
        "empirical_fdr": payload["evaluation"].get("empirical_fdr", ""),
    }
    write_tsv(outdir / "manuscript_table_simulation_truth.tsv", [eval_row], list(eval_row))
    ood_row = {
        "ood_total": payload["evaluation"].get("ood_total", ""),
        "ood_abstention_rate": payload["evaluation"].get("ood_abstention_rate", ""),
        "ood_false_call_rate": payload["evaluation"].get("ood_false_call_rate", ""),
        "false_positives_in_ood_null_families": payload["evaluation"].get("false_positives_in_ood_null_families", ""),
    }
    write_tsv(outdir / "manuscript_table_ood_abstention.tsv", [ood_row], list(ood_row))
    power_rows = payload.get("calibration", []) or [{"q_threshold": "", "called": "", "tp": "", "fp": "", "empirical_fdr": "", "power": ""}]
    write_tsv(outdir / "manuscript_table_power.tsv", power_rows, ["q_threshold", "called", "tp", "fp", "empirical_fdr", "power"])


def _render_report_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# Known-Truth Benchmark Report",
        "",
        "## Benchmark Purpose",
        "",
        "This benchmark evaluates BABAPPA against explicit simulated truth labels.",
        "",
        "## Known-Truth Design",
        "",
        f"Families: {payload['n_families']}",
        f"Truth classes: {payload['truth_class_counts']}",
        "",
        "## BABAPPA Performance Against Truth",
        "",
        f"Gene AUROC: {payload['evaluation'].get('auroc', 'NA')}",
        f"Gene AUPRC: {payload['evaluation'].get('auprc', 'NA')}",
        "",
        "## OOD Abstention Performance",
        "",
        f"OOD abstention rate: {payload['evaluation'].get('ood_abstention_rate', 'NA')}",
        f"OOD false-call rate: {payload['evaluation'].get('ood_false_call_rate', 'NA')}",
        "",
        "## Calibration/FDR",
        "",
        "See `calibration_evaluation/` and `manuscript_table_power.tsv`.",
        "",
        "## Reference-Comparison Status",
        "",
        "Reference methods should be evaluated against the same simulation truth, not treated as truth.",
        "",
        "## Claim Boundary",
        "",
        payload["claim_boundary"],
        "",
        "## Recommended Manuscript Tables",
        "",
        "- `manuscript_table_simulation_truth.tsv`",
        "- `manuscript_table_ood_abstention.tsv`",
        "- `manuscript_table_power.tsv`",
        "",
    ]
    return "\n".join(lines)

