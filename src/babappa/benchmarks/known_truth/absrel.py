"""aBSREL comparator layer for BABAPPA known-truth benchmarks."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv

from .design import PROFILE_SIZES
from .design import KnownTruthBenchmarkDesignConfig, design_known_truth_benchmark
from .metrics import _auprc, _auroc, _confusion
from .metrics import KnownTruthCalibrationEvaluationConfig, KnownTruthEvaluationConfig, evaluate_known_truth_benchmark, evaluate_known_truth_calibration
from .report import KnownTruthBenchmarkReportConfig, make_known_truth_benchmark_report
from .run_plan import USER_RUN_MARK
from .simulate import KnownTruthAlignmentConfig, KnownTruthScoringConfig, KnownTruthSimulationConfig, run_known_truth_alignments, score_known_truth_benchmark, simulate_known_truth_benchmark
from .truth_schema import write_json


@dataclass(frozen=True)
class KnownTruthAbsrelSubsetConfig:
    benchmark_dir: str
    outdir: str
    max_families: int = 12
    stratify_by: str = "regime,truth_class,ood_status,saturation_tier"


@dataclass(frozen=True)
class KnownTruthBenchmarkRunConfig:
    profile: str
    outdir: str
    deployable_model_package: str = "deployable_model_conservative_branch_site_100k_mps"
    methods: Sequence[str] | str = ("identity", "mafft", "babappalign", "muscle")
    device: str = "auto"
    seed: int = 42


@dataclass(frozen=True)
class KnownTruthAbsrelPlanConfig:
    subset: str
    outdir: str
    alignment_source: str = "mafft_codon"
    user_run_only: bool = True


@dataclass(frozen=True)
class KnownTruthAbsrelParseConfig:
    absrel_run_dir: str
    truth_dir: str
    outdir: str


@dataclass(frozen=True)
class KnownTruthBabappaAbsrelComparisonConfig:
    babappa_report: str
    absrel_results: str
    truth_dir: str
    outdir: str


@dataclass(frozen=True)
class KnownTruthBenchmarkSuitePlanConfig:
    profile: str
    outdir: str
    deployable_model_package: str = "deployable_model_conservative_branch_site_100k_mps"
    include_absrel: bool = True
    absrel_max_families: int = 300
    device: str = "auto"
    conda_env: str = "molevo"


@dataclass(frozen=True)
class KnownTruthBenchmarkPlanValidationConfig:
    plan_dir: str


@dataclass(frozen=True)
class KnownTruthBenchmarkIntakeConfig:
    benchmark_dir: str
    outdir: str


@dataclass(frozen=True)
class KnownTruthBabappaAbsrelDecisionConfig:
    babappa_report: str
    absrel_results: str
    comparison_dir: str
    outdir: str


ABSREL_SUBSET_FIELDS = [
    "family_id",
    "regime",
    "truth_class",
    "foreground_branch",
    "codon_alignment_path",
    "tree_path",
    "expected_class",
    "ood_status",
    "saturation_tier",
    "selected_sites",
    "selected_branches",
]

ABSREL_RESULT_FIELDS = [
    "family_id",
    "regime",
    "truth_class",
    "foreground_branch",
    "absrel_status",
    "p_value",
    "q_value",
    "result_class",
    "notes",
]


def select_known_truth_absrel_subset(config: KnownTruthAbsrelSubsetConfig) -> Dict[str, Any]:
    benchmark_dir = Path(config.benchmark_dir)
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    truth_manifest = _truth_manifest_path(benchmark_dir)
    rows = read_tsv(truth_manifest)
    strata = [field.strip() for field in config.stratify_by.split(",") if field.strip()]
    groups: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        augmented = dict(row)
        augmented["ood_status"] = "ood" if row.get("expected_applicability") == "out_of_domain" or row.get("truth_class", "").startswith("ood") else "in_domain_or_borderline"
        key = "|".join(str(augmented.get(field, "")) for field in strata)
        groups.setdefault(key, []).append(augmented)
    selected: List[Dict[str, str]] = []
    while len(selected) < config.max_families and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected) < config.max_families:
                selected.append(groups[key].pop(0))
    subset_rows = [_subset_row(row, benchmark_dir) for row in selected]
    write_tsv(outdir / "absrel_subset.tsv", subset_rows, ABSREL_SUBSET_FIELDS)
    write_json(
        outdir / "absrel_subset.json",
        {
            "known_truth_absrel_subset_version": __version__,
            "status": "ok",
            "benchmark_dir": str(benchmark_dir),
            "max_families": config.max_families,
            "n_selected": len(subset_rows),
            "stratify_by": strata,
            "simulated_truth_is_ground_truth": True,
        },
    )
    (outdir / "absrel_subset.md").write_text(_render_subset_md(config, subset_rows), encoding="utf-8")
    return {"status": "ok", "outdir": str(outdir), "n_selected": len(subset_rows), "subset": str(outdir / "absrel_subset.tsv")}


def run_known_truth_benchmark(config: KnownTruthBenchmarkRunConfig) -> Dict[str, Any]:
    if config.profile not in PROFILE_SIZES:
        raise ValueError(f"unknown profile: {config.profile}")
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if config.profile != "smoke":
        plan_dir = outdir / "user_run_plan"
        plan_known_truth_benchmark_suite(
            KnownTruthBenchmarkSuitePlanConfig(
                profile=config.profile,
                outdir=str(plan_dir),
                deployable_model_package=config.deployable_model_package,
                include_absrel=False,
                device=config.device,
            )
        )
        return {"status": "planned_user_run_only", "profile": config.profile, "outdir": str(outdir), "plan_dir": str(plan_dir)}
    design_dir = outdir / "design"
    design_known_truth_benchmark(KnownTruthBenchmarkDesignConfig(outdir=str(design_dir), benchmark_name="BABAPPA-BENCH-SIM-v1", seed=config.seed))
    sim_dir = outdir / "simulated_families"
    simulate_known_truth_benchmark(KnownTruthSimulationConfig(design_dir=str(design_dir), profile=config.profile, outdir=str(sim_dir), seed=config.seed))
    align_dir = outdir / "alignments"
    run_known_truth_alignments(KnownTruthAlignmentConfig(sim_dir=str(sim_dir), outdir=str(align_dir), methods=config.methods))
    scores_dir = outdir / "babappa_scores"
    score_known_truth_benchmark(
        KnownTruthScoringConfig(
            sim_dir=str(sim_dir),
            alignment_dir=str(align_dir),
            deployable_model_package=config.deployable_model_package,
            outdir=str(scores_dir),
            device=config.device,
            score_backend="smoke_surrogate",
        )
    )
    evaluation_dir = outdir / "evaluation"
    evaluate_known_truth_benchmark(KnownTruthEvaluationConfig(truth=str(sim_dir / "benchmark_truth_manifest.tsv"), scores=str(scores_dir), outdir=str(evaluation_dir)))
    calibration_dir = outdir / "calibration_evaluation"
    evaluate_known_truth_calibration(KnownTruthCalibrationEvaluationConfig(truth=str(sim_dir / "benchmark_truth_manifest.tsv"), scores=str(scores_dir), outdir=str(calibration_dir)))
    report_dir = outdir / "report"
    report = make_known_truth_benchmark_report(KnownTruthBenchmarkReportConfig(benchmark_dir=str(outdir), outdir=str(report_dir)))
    return {"status": "ok", "profile": config.profile, "outdir": str(outdir), "n_families": report["n_families"], "report": str(report_dir / "known_truth_benchmark_report.md")}


def plan_known_truth_absrel_comparison(config: KnownTruthAbsrelPlanConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    subset_path = Path(config.subset)
    if not subset_path.exists():
        raise FileNotFoundError(f"subset does not exist: {subset_path}")
    shutil.copyfile(subset_path, outdir / "absrel_subset.tsv")
    scripts = {
        "run_absrel_comparator.sh": _absrel_run_script(config),
        "monitor_absrel_comparator.sh": _absrel_monitor_script(),
        "validate_absrel_comparator.sh": _absrel_validate_script(),
        "summarize_absrel_comparator.sh": _absrel_summarize_script(),
    }
    for name, text in scripts.items():
        path = outdir / name
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)
    write_json(
        outdir / "expected_outputs.json",
        {
            "run_dir": "absrel_run/",
            "results": "known_truth_absrel_results_PROFILE/absrel_results.tsv",
            "alignment_source": config.alignment_source,
            "tools": ["hyphy_absrel"],
            "codeml_included": False,
            "meme_included": False,
        },
    )
    (outdir / "absrel_comparison_plan.md").write_text(_render_absrel_plan_md(config), encoding="utf-8")
    return {"status": "planned", "outdir": str(outdir), "scripts": len(scripts), "codeml_included": False, "meme_included": False}


def parse_known_truth_absrel_results(config: KnownTruthAbsrelParseConfig) -> Dict[str, Any]:
    run_dir = Path(config.absrel_run_dir)
    truth_dir = Path(config.truth_dir)
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    subset = _subset_for_parse(run_dir, truth_dir)
    rows: List[Dict[str, Any]] = []
    for item in subset:
        family_id = item["family_id"]
        family_dir = run_dir / "absrel_run" / family_id
        if not family_dir.exists():
            family_dir = run_dir / family_id
        json_path = family_dir / "absrel.json"
        log_path = family_dir / "absrel.log"
        if not json_path.exists():
            status = "failed" if log_path.exists() and "error" in log_path.read_text(encoding="utf-8", errors="ignore").lower() else "pending_not_run"
            rows.append(_absrel_result_row(item, status, "", "", status, "aBSREL JSON absent"))
            continue
        try:
            parsed = json.loads(json_path.read_text(encoding="utf-8"))
            positive_count = int(((parsed.get("test results") or {}).get("positive test results")) or 0)
            p_value = _extract_min_p_value(parsed)
            result_class = "positive" if positive_count > 0 else "negative"
            rows.append(_absrel_result_row(item, "parsed", _fmt(p_value), "", result_class, f"official_positive_test_results={positive_count}"))
        except Exception as exc:  # noqa: BLE001
            rows.append(_absrel_result_row(item, "failed", "", "", "failed", f"parse_error:{exc}"))
    write_tsv(outdir / "absrel_results.tsv", rows, ABSREL_RESULT_FIELDS)
    write_json(
        outdir / "absrel_results.json",
        {
            "known_truth_absrel_parse_version": __version__,
            "status": "ok",
            "n_rows": len(rows),
            "n_pending": sum(1 for row in rows if row["result_class"] == "pending_not_run"),
            "simulated_truth_is_ground_truth": True,
        },
    )
    (outdir / "absrel_results.md").write_text(_render_absrel_results_md(rows), encoding="utf-8")
    return {"status": "ok", "outdir": str(outdir), "rows": len(rows), "pending": sum(1 for row in rows if row["result_class"] == "pending_not_run")}


def compare_known_truth_babappa_absrel(config: KnownTruthBabappaAbsrelComparisonConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    benchmark_dir = Path(config.truth_dir)
    truth_rows = read_tsv(_truth_manifest_path(benchmark_dir))
    labels = {row["family_id"]: 1 if row["truth_class"] == "positive" else 0 for row in truth_rows}
    babappa_scores = _load_babappa_gene_scores(Path(config.babappa_report), benchmark_dir)
    absrel_rows = read_tsv(Path(config.absrel_results))
    absrel_scores: Dict[str, float] = {}
    for row in absrel_rows:
        if row.get("result_class") == "positive":
            absrel_scores[row["family_id"]] = 1.0 - _as_float(row.get("p_value"), 0.0)
        elif row.get("result_class") == "negative":
            absrel_scores[row["family_id"]] = 1.0 - _as_float(row.get("p_value"), 1.0)
        else:
            absrel_scores[row["family_id"]] = 0.0
    method_rows = [
        _method_metric_row("BABAPPA", labels, babappa_scores, truth_rows),
        _method_metric_row("aBSREL", labels, absrel_scores, truth_rows, failures={row["family_id"] for row in absrel_rows if row.get("result_class") in {"failed", "pending_not_run"}}),
    ]
    payload = {
        "known_truth_babappa_absrel_comparison_version": __version__,
        "status": "ok",
        "simulated_truth_is_ground_truth": True,
        "absrel_is_comparator_not_ground_truth": True,
        "babappa_not_likelihood_replacement": True,
        "methods": method_rows,
        "interpretation": (
            "BABAPPA and aBSREL are compared against simulator truth. aBSREL is a comparator, "
            "not ground truth. BABAPPA is evaluated as a complementary, alignment-aware, OOD-gated, "
            "simulation-trained branch-site support framework."
        ),
    }
    write_tsv(outdir / "babappa_absrel_method_comparison.tsv", method_rows, list(method_rows[0]))
    write_tsv(outdir / "manuscript_table_babappa_absrel_comparison.tsv", method_rows, list(method_rows[0]))
    runtime_rows = [{"method": "BABAPPA", "failure_rate": method_rows[0]["failure_rate"], "runtime_seconds": ""}, {"method": "aBSREL", "failure_rate": method_rows[1]["failure_rate"], "runtime_seconds": ""}]
    write_tsv(outdir / "manuscript_table_runtime_failure.tsv", runtime_rows, ["method", "failure_rate", "runtime_seconds"])
    write_json(outdir / "babappa_absrel_method_comparison.json", payload)
    (outdir / "babappa_absrel_method_comparison.md").write_text(_render_method_comparison_md(payload), encoding="utf-8")
    return {"status": "ok", "outdir": str(outdir), "methods": "BABAPPA,aBSREL", "simulated_truth_ground_truth": True}


def plan_known_truth_benchmark_suite(config: KnownTruthBenchmarkSuitePlanConfig) -> Dict[str, Any]:
    if config.profile not in PROFILE_SIZES:
        raise ValueError(f"unknown profile: {config.profile}")
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    benchmark_dir = f"known_truth_benchmark_{config.profile}"
    subset_dir = f"known_truth_absrel_subset_{config.profile}"
    absrel_plan_dir = f"known_truth_absrel_comparison_plan_{config.profile}"
    absrel_results_dir = f"known_truth_absrel_results_{config.profile}"
    comparison_dir = f"known_truth_babappa_absrel_comparison_{config.profile}"
    scripts = {
        "run_known_truth_benchmark.sh": _suite_run_babappa(config, benchmark_dir),
        "monitor_known_truth_benchmark.sh": _suite_monitor_benchmark(benchmark_dir),
        "validate_known_truth_benchmark.sh": _suite_validate_benchmark(benchmark_dir),
        "summarize_known_truth_benchmark.sh": _suite_summarize_benchmark(benchmark_dir),
        "run_absrel_comparator.sh": _suite_run_absrel(config, benchmark_dir, subset_dir, absrel_plan_dir),
        "monitor_absrel_comparator.sh": _suite_monitor_absrel(absrel_plan_dir),
        "validate_absrel_comparator.sh": _suite_validate_absrel(absrel_plan_dir),
        "summarize_absrel_comparator.sh": _suite_summarize_absrel(absrel_plan_dir, benchmark_dir, absrel_results_dir),
        "compare_babappa_absrel.sh": _suite_compare(benchmark_dir, absrel_results_dir, comparison_dir),
    }
    for name, text in scripts.items():
        path = outdir / name
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)
    write_json(
        outdir / "expected_outputs.json",
        {
            "profile": config.profile,
            "benchmark_dir": benchmark_dir,
            "absrel_subset": subset_dir,
            "absrel_plan": absrel_plan_dir,
            "absrel_results": absrel_results_dir,
            "comparison": comparison_dir,
            "include_absrel": config.include_absrel,
            "codeml_included": False,
            "meme_included": False,
        },
    )
    (outdir / "benchmark_suite_plan.md").write_text(_render_suite_plan_md(config), encoding="utf-8")
    return {"status": "planned", "outdir": str(outdir), "profile": config.profile, "absrel_included": config.include_absrel, "codeml_included": False, "meme_included": False}


def validate_known_truth_benchmark_plan(config: KnownTruthBenchmarkPlanValidationConfig) -> Dict[str, Any]:
    plan_dir = Path(config.plan_dir)
    outdir = plan_dir
    failures: List[str] = []
    warnings: List[str] = []
    required = [
        "run_known_truth_benchmark.sh",
        "monitor_known_truth_benchmark.sh",
        "validate_known_truth_benchmark.sh",
        "summarize_known_truth_benchmark.sh",
        "run_absrel_comparator.sh",
        "monitor_absrel_comparator.sh",
        "validate_absrel_comparator.sh",
        "summarize_absrel_comparator.sh",
        "compare_babappa_absrel.sh",
    ]
    for name in required:
        path = plan_dir / name
        if not path.exists():
            failures.append(f"missing_script:{name}")
            continue
        text = path.read_text(encoding="utf-8")
        if USER_RUN_MARK not in text:
            failures.append(f"missing_user_run_marker:{name}")
        lowered = text.lower()
        if "codeml" in lowered:
            failures.append(f"forbidden_codeml_command:{name}")
        if "meme" in lowered:
            failures.append(f"forbidden_meme_command:{name}")
        if any(cmd in lowered for cmd in ["rm -rf", "git reset", "mkfs", "shutdown"]):
            failures.append(f"destructive_command:{name}")
        try:
            subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            failures.append(f"bash_syntax:{name}:{exc.stderr.strip()}")
    payload = {"known_truth_plan_validation_version": __version__, "status": "fail" if failures else "ok", "failures": failures, "warnings": warnings}
    write_json(outdir / "plan_validation.json", payload)
    write_tsv(outdir / "plan_validation.tsv", [{"kind": "failure", "message": item} for item in failures] + [{"kind": "warning", "message": item} for item in warnings], ["kind", "message"])
    (outdir / "plan_validation.md").write_text(_render_plan_validation_md(payload), encoding="utf-8")
    return {"status": payload["status"], "outdir": str(outdir), "n_failures": len(failures)}


def intake_known_truth_benchmark_results(config: KnownTruthBenchmarkIntakeConfig) -> Dict[str, Any]:
    benchmark_dir = Path(config.benchmark_dir)
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checks = {
        "truth_manifest": _truth_manifest_path(benchmark_dir).exists(),
        "report": (benchmark_dir / "report" / "known_truth_benchmark_report.json").exists(),
        "evaluation": (benchmark_dir / "evaluation" / "evaluation_summary.json").exists(),
        "calibration": (benchmark_dir / "calibration_evaluation" / "calibration_evaluation.json").exists(),
        "scores": (benchmark_dir / "babappa_scores" / "gene_support.tsv").exists(),
    }
    status = "ok" if all(checks.values()) else "incomplete"
    write_json(outdir / "known_truth_benchmark_intake.json", {"status": status, "checks": checks, "benchmark_dir": str(benchmark_dir)})
    write_tsv(outdir / "known_truth_benchmark_intake.tsv", [{"check": key, "present": value} for key, value in checks.items()], ["check", "present"])
    (outdir / "known_truth_benchmark_intake.md").write_text(_render_intake_md(status, checks), encoding="utf-8")
    return {"status": status, "outdir": str(outdir)}


def make_known_truth_babappa_absrel_decision_report(config: KnownTruthBabappaAbsrelDecisionConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    comparison_path = Path(config.comparison_dir) / "babappa_absrel_method_comparison.json"
    absrel_path = Path(config.absrel_results)
    report_path = Path(config.babappa_report) / "known_truth_benchmark_report.json"
    decision = "pilot_incomplete"
    reasons: List[str] = []
    if not report_path.exists():
        reasons.append("missing_babappa_report")
    if not absrel_path.exists():
        reasons.append("missing_absrel_results")
    if not comparison_path.exists():
        reasons.append("missing_comparison")
    if not reasons:
        absrel_rows = read_tsv(absrel_path)
        pending_absrel = sum(1 for row in absrel_rows if row.get("result_class") == "pending_not_run")
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        methods = {row["method"]: row for row in comparison.get("methods", [])}
        babappa = methods.get("BABAPPA", {})
        ood_false_call = _as_float(babappa.get("ood_false_call_rate"), 0.0)
        fdr = _as_float(babappa.get("empirical_fdr"), 0.0)
        if pending_absrel == len(absrel_rows) and absrel_rows:
            decision = "pilot_conditional_pass_fix_warnings"
            reasons.append("absrel_pending_not_run")
        elif ood_false_call > 0.10 or fdr > 0.20:
            decision = "pilot_conditional_pass_fix_warnings"
            reasons.append("high_fdr_or_ood_false_call")
        else:
            decision = "pilot_pass_ready_for_paper_profile"
            reasons.append("pilot_interpretable")
    payload = {
        "known_truth_babappa_absrel_decision_version": __version__,
        "status": "ok",
        "decision": decision,
        "reasons": reasons,
        "claim_boundary": "Paper profile should scale only after pilot outputs are interpretable. BABAPPA is not positioned as an aBSREL replacement.",
    }
    write_json(outdir / "known_truth_babappa_absrel_decision_report.json", payload)
    (outdir / "known_truth_babappa_absrel_decision_report.md").write_text(_render_decision_md(payload), encoding="utf-8")
    return {"status": "ok", "decision": decision, "outdir": str(outdir)}


def _truth_manifest_path(benchmark_dir: Path) -> Path:
    candidate = benchmark_dir / "simulated_families" / "benchmark_truth_manifest.tsv"
    if candidate.exists():
        return candidate
    return benchmark_dir / "benchmark_truth_manifest.tsv"


def _subset_row(row: Dict[str, str], benchmark_dir: Path) -> Dict[str, str]:
    foreground = (row.get("foreground_branches") or "").split(",")[0] or "leaves"
    family_id = row["family_id"]
    mafft = benchmark_dir / "alignments" / family_id / "mafft.aln.fasta"
    alignment = mafft if mafft.exists() else Path(row["cds_fasta"])
    return {
        "family_id": family_id,
        "regime": row["regime"],
        "truth_class": row["truth_class"],
        "foreground_branch": foreground,
        "codon_alignment_path": str(alignment),
        "tree_path": row["tree_file"],
        "expected_class": "positive" if row["truth_class"] == "positive" else "negative_or_abstain",
        "ood_status": "ood" if row.get("expected_applicability") == "out_of_domain" or row.get("truth_class", "").startswith("ood") else "in_domain_or_borderline",
        "saturation_tier": row.get("saturation_tier", ""),
        "selected_sites": row.get("n_selected_sites", "0"),
        "selected_branches": row.get("positive_branches", ""),
    }


def _subset_for_parse(run_dir: Path, truth_dir: Path) -> List[Dict[str, str]]:
    for candidate in [run_dir / "absrel_subset.tsv", run_dir.parent / "absrel_subset.tsv"]:
        if candidate.exists():
            return read_tsv(candidate)
    return [_subset_row(row, truth_dir) for row in read_tsv(_truth_manifest_path(truth_dir))]


def _absrel_result_row(item: Dict[str, str], status: str, p_value: str, q_value: str, result_class: str, notes: str) -> Dict[str, str]:
    return {
        "family_id": item["family_id"],
        "regime": item.get("regime", ""),
        "truth_class": item.get("truth_class", ""),
        "foreground_branch": item.get("foreground_branch", ""),
        "absrel_status": status,
        "p_value": p_value,
        "q_value": q_value,
        "result_class": result_class,
        "notes": notes,
    }


def _extract_min_p_value(payload: Any) -> float:
    values: List[float] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                lowered = str(key).lower()
                if "p-value" in lowered or lowered in {"p", "p_value", "pvalue"}:
                    try:
                        values.append(float(value))
                    except (TypeError, ValueError):
                        pass
                visit(value)
        elif isinstance(obj, list):
            for value in obj:
                visit(value)

    visit(payload)
    return min(values) if values else 1.0


def _load_babappa_gene_scores(report_dir: Path, benchmark_dir: Path) -> Dict[str, float]:
    for candidate in [benchmark_dir / "babappa_scores" / "gene_support.tsv", report_dir.parent / "babappa_scores" / "gene_support.tsv"]:
        if candidate.exists():
            return {row["family_id"]: _as_float(row.get("score", row.get("gene_support")), 0.0) for row in read_tsv(candidate)}
    return {}


def _method_metric_row(method: str, labels_by_family: Dict[str, int], scores_by_family: Dict[str, float], truth_rows: List[Dict[str, str]], failures: Iterable[str] = ()) -> Dict[str, Any]:
    failure_set = set(failures)
    ids = [row["family_id"] for row in truth_rows]
    labels = [labels_by_family.get(family_id, 0) for family_id in ids]
    scores = [scores_by_family.get(family_id, 0.0) for family_id in ids]
    c = _confusion(labels, scores, 0.5)
    ood_ids = {row["family_id"] for row in truth_rows if row.get("expected_applicability") == "out_of_domain" or row.get("truth_class", "").startswith("ood")}
    ood_false = sum(1 for family_id in ood_ids if labels_by_family.get(family_id, 0) == 0 and scores_by_family.get(family_id, 0.0) >= 0.5)
    return {
        "method": method,
        "n_families": len(ids),
        "auroc": _auroc(labels, scores),
        "auprc": _auprc(labels, scores),
        "precision": c["precision"],
        "recall_power": c["recall"],
        "specificity": c["specificity"],
        "f1": c["f1"],
        "mcc": c["mcc"],
        "empirical_fdr": c["empirical_fdr"],
        "fpr": c["fpr"],
        "fnr": c["fnr"],
        "ood_false_call_rate": 0.0 if not ood_ids else ood_false / len(ood_ids),
        "failure_rate": 0.0 if not ids else len(failure_set) / len(ids),
    }


def _absrel_run_script(config: KnownTruthAbsrelPlanConfig) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
echo '{USER_RUN_MARK}'
SUBSET="$(cd "$(dirname "$0")" && pwd)/absrel_subset.tsv"
RUN_DIR="$(cd "$(dirname "$0")" && pwd)/absrel_run"
mkdir -p "$RUN_DIR"
if ! command -v hyphy >/dev/null 2>&1; then
  echo "HyPhy is not available on PATH" >&2
  exit 1
fi
tail -n +2 "$SUBSET" | while IFS=$'\\t' read -r family_id regime truth_class foreground alignment tree expected ood saturation selected_sites selected_branches; do
  family_out="$RUN_DIR/$family_id"
  mkdir -p "$family_out"
  if [ -f "$family_out/absrel.json" ] && [ "${{BABAPPA_FORCE:-0}}" != "1" ]; then
    echo "Skipping completed $family_id"
    continue
  fi
  cp "$alignment" "$family_out/alignment.fasta"
  cp "$tree" "$family_out/tree.nwk"
  (
    cd "$family_out"
    hyphy absrel --alignment alignment.fasta --tree tree.nwk --branches "$foreground" --output absrel.json > absrel.log 2> absrel.err
  ) || {{
    echo "aBSREL failed for $family_id" >&2
    if [ "${{BABAPPA_FAIL_FAST:-0}}" = "1" ]; then exit 1; fi
  }}
done
"""


def _absrel_monitor_script() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
echo '{USER_RUN_MARK}'
ROOT="$(cd "$(dirname "$0")" && pwd)"
find "$ROOT/absrel_run" -name absrel.json 2>/dev/null | wc -l
find "$ROOT/absrel_run" -name absrel.err 2>/dev/null | wc -l
"""


def _absrel_validate_script() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
echo '{USER_RUN_MARK}'
ROOT="$(cd "$(dirname "$0")" && pwd)"
test -f "$ROOT/absrel_subset.tsv"
bash -n "$ROOT/run_absrel_comparator.sh"
"""


def _absrel_summarize_script() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
echo '{USER_RUN_MARK}'
ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "aBSREL JSON files:"
find "$ROOT/absrel_run" -name absrel.json 2>/dev/null | wc -l
"""


def _render_subset_md(config: KnownTruthAbsrelSubsetConfig, rows: List[Dict[str, str]]) -> str:
    return "\n".join([
        "# Known-Truth aBSREL Subset",
        "",
        f"Benchmark: `{config.benchmark_dir}`",
        f"Selected families: `{len(rows)}`",
        "",
        "aBSREL will be evaluated against simulator truth, not treated as truth.",
        "",
    ])


def _render_absrel_plan_md(config: KnownTruthAbsrelPlanConfig) -> str:
    return "\n".join([
        "# Known-Truth aBSREL Comparator Plan",
        "",
        USER_RUN_MARK,
        "",
        "This plan runs only HyPhy aBSREL. It does not include codeml or MEME.",
        "",
        f"Subset: `{config.subset}`",
        f"Alignment source: `{config.alignment_source}`",
        "",
    ])


def _render_absrel_results_md(rows: List[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["result_class"]] = counts.get(row["result_class"], 0) + 1
    return "\n".join([
        "# Known-Truth aBSREL Results",
        "",
        f"Rows: `{len(rows)}`",
        f"Classes: `{counts}`",
        "",
        "Simulator truth is the ground truth for evaluation.",
        "",
    ])


def _render_method_comparison_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# BABAPPA vs aBSREL Known-Truth Comparison",
        "",
        "Simulated truth is the ground truth. aBSREL is a comparator, not ground truth.",
        "",
        payload["interpretation"],
        "",
    ]
    for row in payload["methods"]:
        lines.append(f"- {row['method']}: AUROC `{row['auroc']}`, FDR `{row['empirical_fdr']}`, OOD false-call `{row['ood_false_call_rate']}`")
    lines.append("")
    return "\n".join(lines)


def _suite_run_babappa(config: KnownTruthBenchmarkSuitePlanConfig, benchmark_dir: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
echo '{USER_RUN_MARK}'
conda activate {config.conda_env} 2>/dev/null || true
babappa run-known-truth-benchmark \\
  --profile {config.profile} \\
  --outdir {benchmark_dir} \\
  --deployable-model-package {config.deployable_model_package} \\
  --methods identity,mafft,babappalign,muscle \\
  --device {config.device} \\
  --seed 42
"""


def _suite_monitor_benchmark(benchmark_dir: str) -> str:
    return f"#!/usr/bin/env bash\nset -euo pipefail\necho '{USER_RUN_MARK}'\ndu -sh {benchmark_dir} 2>/dev/null || true\nfind {benchmark_dir} -maxdepth 3 -type f 2>/dev/null | wc -l\n"


def _suite_validate_benchmark(benchmark_dir: str) -> str:
    return f"#!/usr/bin/env bash\nset -euo pipefail\necho '{USER_RUN_MARK}'\nbabappa validate-known-truth-benchmark --benchmark-dir {benchmark_dir}\n"


def _suite_summarize_benchmark(benchmark_dir: str) -> str:
    return f"#!/usr/bin/env bash\nset -euo pipefail\necho '{USER_RUN_MARK}'\nbabappa make-known-truth-benchmark-report --benchmark-dir {benchmark_dir} --outdir {benchmark_dir}/report\n"


def _suite_run_absrel(config: KnownTruthBenchmarkSuitePlanConfig, benchmark_dir: str, subset_dir: str, absrel_plan_dir: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
echo '{USER_RUN_MARK}'
conda activate {config.conda_env} 2>/dev/null || true
babappa select-known-truth-absrel-subset \\
  --benchmark-dir {benchmark_dir} \\
  --outdir {subset_dir} \\
  --max-families {config.absrel_max_families} \\
  --stratify-by regime,truth_class,ood_status,saturation_tier
babappa plan-known-truth-absrel-comparison \\
  --subset {subset_dir}/absrel_subset.tsv \\
  --outdir {absrel_plan_dir} \\
  --alignment-source mafft_codon \\
  --user-run-only true
bash {absrel_plan_dir}/run_absrel_comparator.sh
"""


def _suite_monitor_absrel(absrel_plan_dir: str) -> str:
    return f"#!/usr/bin/env bash\nset -euo pipefail\necho '{USER_RUN_MARK}'\nbash {absrel_plan_dir}/monitor_absrel_comparator.sh\n"


def _suite_validate_absrel(absrel_plan_dir: str) -> str:
    return f"#!/usr/bin/env bash\nset -euo pipefail\necho '{USER_RUN_MARK}'\nbash {absrel_plan_dir}/validate_absrel_comparator.sh\n"


def _suite_summarize_absrel(absrel_plan_dir: str, benchmark_dir: str, absrel_results_dir: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
echo '{USER_RUN_MARK}'
bash {absrel_plan_dir}/summarize_absrel_comparator.sh
babappa parse-known-truth-absrel-results \\
  --absrel-run-dir {absrel_plan_dir} \\
  --truth-dir {benchmark_dir} \\
  --outdir {absrel_results_dir}
"""


def _suite_compare(benchmark_dir: str, absrel_results_dir: str, comparison_dir: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
echo '{USER_RUN_MARK}'
babappa compare-known-truth-babappa-absrel \\
  --babappa-report {benchmark_dir}/report \\
  --absrel-results {absrel_results_dir}/absrel_results.tsv \\
  --truth-dir {benchmark_dir} \\
  --outdir {comparison_dir}
"""


def _render_suite_plan_md(config: KnownTruthBenchmarkSuitePlanConfig) -> str:
    return "\n".join([
        "# Known-Truth Benchmark Suite Plan",
        "",
        USER_RUN_MARK,
        "",
        f"Profile: `{config.profile}`",
        f"Include aBSREL: `{config.include_absrel}`",
        "codeml included: `False`",
        "MEME included: `False`",
        "",
        "Run pilot before scaling to paper profile.",
        "",
    ])


def _render_plan_validation_md(payload: Dict[str, Any]) -> str:
    lines = ["# Known-Truth Benchmark Plan Validation", "", f"Status: `{payload['status']}`", ""]
    if payload["failures"]:
        lines.append("## Failures")
        lines.extend(f"- {item}" for item in payload["failures"])
    return "\n".join(lines) + "\n"


def _render_intake_md(status: str, checks: Dict[str, bool]) -> str:
    lines = ["# Known-Truth Benchmark Intake", "", f"Status: `{status}`", ""]
    lines.extend(f"- {key}: `{value}`" for key, value in checks.items())
    return "\n".join(lines) + "\n"


def _render_decision_md(payload: Dict[str, Any]) -> str:
    return "\n".join([
        "# Known-Truth BABAPPA/aBSREL Decision Report",
        "",
        f"Decision: `{payload['decision']}`",
        "",
        "BABAPPA is evaluated as a complementary framework, not an aBSREL replacement.",
        "",
        "## Reasons",
        "",
        *[f"- {item}" for item in payload["reasons"]],
        "",
    ])


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        numeric = float(value)
        if math.isnan(numeric):
            return default
        return numeric
    except (TypeError, ValueError):
        return default


def _fmt(value: float) -> str:
    return "" if value is None else f"{value:.6g}"
