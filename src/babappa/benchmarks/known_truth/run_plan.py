"""Run-plan generation for BABAPPA known-truth benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

from babappa import __version__

from .design import PROFILE_SIZES
from .truth_schema import write_json

USER_RUN_MARK = "USER-RUN ONLY — DO NOT EXECUTE IN " + "CODE" + "X"


@dataclass(frozen=True)
class KnownTruthBenchmarkPlanConfig:
    profile: str
    design_dir: str
    outdir: str
    methods: Sequence[str] | str = ("identity", "mafft", "babappalign", "muscle")
    device: str = "auto"
    threads: int = 8
    max_workers: int = 4


def plan_known_truth_benchmark(config: KnownTruthBenchmarkPlanConfig) -> Dict[str, Any]:
    if config.profile not in PROFILE_SIZES:
        raise ValueError(f"unknown benchmark profile: {config.profile}")
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    methods = ",".join(config.methods) if not isinstance(config.methods, str) else config.methods
    benchmark_dir = f"known_truth_benchmark_{config.profile}"
    score_backend = "smoke_surrogate" if config.profile == "smoke" else "direct"
    long_run_mark = "" if config.profile == "smoke" else f"echo '{USER_RUN_MARK}'\n"
    scripts = {
        "run_known_truth_benchmark.sh": _run_script(config, benchmark_dir, methods, score_backend, long_run_mark),
        "monitor_known_truth_benchmark.sh": _monitor_script(benchmark_dir, long_run_mark),
        "validate_known_truth_benchmark.sh": _validate_script(benchmark_dir, long_run_mark),
        "summarize_known_truth_benchmark.sh": _summarize_script(benchmark_dir, long_run_mark),
        "compare_known_truth_methods.sh": _reference_plan_script(benchmark_dir, long_run_mark),
    }
    for name, text in scripts.items():
        path = outdir / name
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)
    payload = {
        "known_truth_benchmark_plan_version": __version__,
        "profile": config.profile,
        "profile_size": PROFILE_SIZES[config.profile],
        "benchmark_dir": benchmark_dir,
        "methods": methods,
        "device": config.device,
        "threads": config.threads,
        "max_workers": config.max_workers,
        "score_backend": score_backend,
        "long_run_handoff": config.profile != "smoke",
        "user_run_mark": USER_RUN_MARK if config.profile != "smoke" else "",
    }
    write_json(outdir / "expected_outputs.json", _expected_outputs(benchmark_dir))
    write_json(outdir / "known_truth_benchmark_plan.json", payload)
    (outdir / "known_truth_benchmark_plan.md").write_text(_render_plan_md(payload), encoding="utf-8")
    return {
        "status": "planned",
        "outdir": str(outdir),
        "profile": config.profile,
        "families": PROFILE_SIZES[config.profile],
        "score_backend": score_backend,
        "user_run_only": config.profile != "smoke",
    }


def _run_script(config: KnownTruthBenchmarkPlanConfig, benchmark_dir: str, methods: str, score_backend: str, mark: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd "$(pwd)"
{mark}mkdir -p {benchmark_dir}
babappa simulate-known-truth-benchmark \\
  --design-dir {config.design_dir} \\
  --profile {config.profile} \\
  --outdir {benchmark_dir}/simulated_families \\
  --seed 42
babappa run-known-truth-alignments \\
  --sim-dir {benchmark_dir}/simulated_families \\
  --outdir {benchmark_dir}/alignments \\
  --methods {methods} \\
  --threads {config.threads} \\
  --max-workers {config.max_workers}
babappa score-known-truth-benchmark \\
  --sim-dir {benchmark_dir}/simulated_families \\
  --alignment-dir {benchmark_dir}/alignments \\
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \\
  --outdir {benchmark_dir}/babappa_scores \\
  --device {config.device} \\
  --score-backend {score_backend}
babappa evaluate-known-truth-benchmark \\
  --truth {benchmark_dir}/simulated_families/benchmark_truth_manifest.tsv \\
  --scores {benchmark_dir}/babappa_scores \\
  --outdir {benchmark_dir}/evaluation
babappa evaluate-known-truth-calibration \\
  --truth {benchmark_dir}/simulated_families/benchmark_truth_manifest.tsv \\
  --scores {benchmark_dir}/babappa_scores \\
  --outdir {benchmark_dir}/calibration_evaluation
babappa make-known-truth-benchmark-report \\
  --benchmark-dir {benchmark_dir} \\
  --outdir {benchmark_dir}/report
"""


def _monitor_script(benchmark_dir: str, mark: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
{mark}du -sh {benchmark_dir} 2>/dev/null || true
find {benchmark_dir} -maxdepth 3 -type f | wc -l
find {benchmark_dir} -maxdepth 3 -name '*summary*.json' -o -name '*manifest*.json' 2>/dev/null | sort
"""


def _validate_script(benchmark_dir: str, mark: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
{mark}babappa validate-known-truth-benchmark \\
  --benchmark-dir {benchmark_dir} \\
  --outdir {benchmark_dir}/validation
"""


def _summarize_script(benchmark_dir: str, mark: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
{mark}babappa make-known-truth-benchmark-report \\
  --benchmark-dir {benchmark_dir} \\
  --outdir {benchmark_dir}/report
"""


def _reference_plan_script(benchmark_dir: str, mark: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
{mark}babappa plan-known-truth-reference-comparison \\
  --benchmark-dir {benchmark_dir} \\
  --outdir {benchmark_dir}/reference_comparison_plan \\
  --tools absrel \\
  --max-families 100
"""


def _expected_outputs(benchmark_dir: str) -> Dict[str, Any]:
    return {
        "simulated_families": f"{benchmark_dir}/simulated_families",
        "truth_manifest": f"{benchmark_dir}/simulated_families/benchmark_truth_manifest.tsv",
        "alignments": f"{benchmark_dir}/alignments",
        "babappa_scores": f"{benchmark_dir}/babappa_scores",
        "evaluation": f"{benchmark_dir}/evaluation",
        "calibration_evaluation": f"{benchmark_dir}/calibration_evaluation",
        "report": f"{benchmark_dir}/report",
    }


def _render_plan_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# Known-Truth Benchmark Plan",
        "",
        f"Profile: `{payload['profile']}`",
        f"Families: {payload['profile_size']}",
        f"Score backend: `{payload['score_backend']}`",
        "",
    ]
    if payload["long_run_handoff"]:
        lines.extend([USER_RUN_MARK, "", "This profile is intended for offline execution by the user.", ""])
    else:
        lines.append("Smoke profile is small enough for local validation.")
    return "\n".join(lines)
