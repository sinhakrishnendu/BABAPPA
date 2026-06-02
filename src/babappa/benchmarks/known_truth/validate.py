"""Validation for known-truth benchmark outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from babappa import __version__
from babappa.datasets.index import write_tsv

from .truth_schema import validate_truth_manifest, write_json


@dataclass(frozen=True)
class KnownTruthValidationConfig:
    benchmark_dir: str
    outdir: str = ""


def validate_known_truth_benchmark(config: KnownTruthValidationConfig) -> Dict[str, Any]:
    benchmark_dir = Path(config.benchmark_dir)
    outdir = Path(config.outdir) if config.outdir else benchmark_dir / "validation"
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = benchmark_dir / "simulated_families" / "benchmark_truth_manifest.tsv"
    if not manifest.exists():
        manifest = benchmark_dir / "benchmark_truth_manifest.tsv"
    summary = validate_truth_manifest(manifest)
    rows: List[Dict[str, Any]] = [{"kind": "failure", "message": item} for item in summary["failures"]]
    rows.extend({"kind": "warning", "message": item} for item in summary["warnings"])
    write_json(outdir / "known_truth_validation.json", {"known_truth_validation_version": __version__, **summary})
    write_tsv(outdir / "known_truth_validation.tsv", rows, ["kind", "message"])
    (outdir / "known_truth_validation.md").write_text(_render_validation_md(summary), encoding="utf-8")
    return {"status": summary["status"], "outdir": str(outdir), "n_families": summary["n_families"], "n_failures": len(summary["failures"])}


def _render_validation_md(summary: Dict[str, Any]) -> str:
    lines = [
        "# Known-Truth Benchmark Validation",
        "",
        f"Status: `{summary['status']}`",
        f"Families: {summary['n_families']}",
        "",
        "Truth labels are restricted to benchmark evaluation.",
        "",
    ]
    if summary["failures"]:
        lines.append("## Failures")
        lines.extend(f"- {item}" for item in summary["failures"])
        lines.append("")
    if summary["warnings"]:
        lines.append("## Warnings")
        lines.extend(f"- {item}" for item in summary["warnings"])
        lines.append("")
    return "\n".join(lines)

