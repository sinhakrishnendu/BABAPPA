"""Reference-comparison planning and method comparison for known-truth benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv

from .run_plan import USER_RUN_MARK
from .truth_schema import write_json


@dataclass(frozen=True)
class KnownTruthReferenceComparisonPlanConfig:
    benchmark_dir: str
    outdir: str
    tools: str = "codeml,absrel,meme"
    max_families: int = 100


@dataclass(frozen=True)
class KnownTruthMethodComparisonConfig:
    truth: str
    babappa_evaluation: str
    outdir: str
    reference_results: str = ""


def plan_known_truth_reference_comparison(config: KnownTruthReferenceComparisonPlanConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tools = [part.strip() for part in config.tools.split(",") if part.strip()]
    scripts = {
        "run_codeml_known_truth.sh": _script("codeml", config),
        "run_hyphy_absrel_known_truth.sh": _script("hyphy_absrel", config),
        "run_hyphy_meme_known_truth.sh": _script("hyphy_meme", config),
        "parse_reference_results.sh": _parse_script(config),
        "compare_babappa_reference_truth.sh": _compare_script(config),
    }
    for name, text in scripts.items():
        path = outdir / name
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)
    schema_rows = [
        {
            "family_id": "",
            "tool": "codeml|absrel|meme",
            "test_name": "",
            "p_value": "",
            "q_value": "",
            "selected_branch": "",
            "selected_sites": "",
            "result_class": "positive|negative|inconclusive|failed",
            "runtime_seconds": "",
            "notes": "",
        }
    ]
    write_tsv(outdir / "reference_result_schema.tsv", schema_rows, list(schema_rows[0]))
    write_json(
        outdir / "expected_outputs.json",
        {
            "reference_results": "reference_results.tsv",
            "comparison": "method comparison against simulation truth",
            "tools": tools,
            "max_families": config.max_families,
        },
    )
    (outdir / "reference_comparison_plan.md").write_text(_render_reference_plan_md(config, tools), encoding="utf-8")
    return {"status": "planned", "outdir": str(outdir), "tools": ",".join(tools), "user_run_only": True}


def compare_methods_known_truth(config: KnownTruthMethodComparisonConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    evaluation_path = Path(config.babappa_evaluation) / "evaluation_summary.json"
    reference_present = bool(config.reference_results and Path(config.reference_results).exists())
    payload = {
        "known_truth_method_comparison_version": __version__,
        "status": "reference_pending" if not reference_present else "ok",
        "truth_manifest": config.truth,
        "babappa_evaluation": str(evaluation_path),
        "reference_results_present": reference_present,
        "claim_boundary": "All methods must be compared against simulation truth; reference methods are not treated as truth.",
    }
    rows: List[Dict[str, Any]] = [
        {"method": "BABAPPA", "status": "evaluated", "reference_results": "not_required", "notes": "See BABAPPA evaluation summary."}
    ]
    if reference_present:
        rows.append({"method": "codeml/HyPhy", "status": "ready_for_truth_comparison", "reference_results": config.reference_results, "notes": "External method calls should be evaluated against the same truth manifest."})
    else:
        rows.append({"method": "codeml/HyPhy", "status": "pending", "reference_results": "", "notes": "Reference outputs absent."})
    write_json(outdir / "method_comparison.json", payload)
    write_tsv(outdir / "method_comparison.tsv", rows, ["method", "status", "reference_results", "notes"])
    (outdir / "method_comparison.md").write_text(_render_compare_md(payload), encoding="utf-8")
    return {"status": payload["status"], "outdir": str(outdir), "reference_results_present": reference_present}


def _script(tool: str, config: KnownTruthReferenceComparisonPlanConfig) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
echo '{USER_RUN_MARK}'
echo 'Planned {tool} known-truth reference run.'
echo 'Benchmark: {config.benchmark_dir}'
echo 'Maximum families: {config.max_families}'
echo 'Prepare per-family codon alignments/trees from the simulated benchmark and run {tool} externally.'
"""


def _parse_script(config: KnownTruthReferenceComparisonPlanConfig) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
echo '{USER_RUN_MARK}'
echo 'Parse codeml/HyPhy outputs into reference_results.tsv for {config.benchmark_dir}.'
"""


def _compare_script(config: KnownTruthReferenceComparisonPlanConfig) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
echo '{USER_RUN_MARK}'
babappa compare-methods-known-truth \\
  --truth {config.benchmark_dir}/simulated_families/benchmark_truth_manifest.tsv \\
  --babappa-evaluation {config.benchmark_dir}/evaluation \\
  --reference-results {config.benchmark_dir}/reference_results/reference_results.tsv \\
  --outdir {config.benchmark_dir}/method_comparison
"""


def _render_reference_plan_md(config: KnownTruthReferenceComparisonPlanConfig, tools: List[str]) -> str:
    return (
        "# Known-Truth Reference Comparison Plan\n\n"
        f"{USER_RUN_MARK}\n\n"
        f"Benchmark: `{config.benchmark_dir}`\n\n"
        f"Tools: {', '.join(tools)}\n\n"
        "codeml/HyPhy are compared against the same explicit truth labels. They are not treated as truth.\n"
    )


def _render_compare_md(payload: Dict[str, Any]) -> str:
    return (
        "# Known-Truth Method Comparison\n\n"
        f"Status: `{payload['status']}`\n\n"
        f"{payload['claim_boundary']}\n"
    )

