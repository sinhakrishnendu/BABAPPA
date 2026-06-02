import json
from pathlib import Path

from babappa.benchmarks.known_truth import (
    KnownTruthAbsrelParseConfig,
    KnownTruthAbsrelPlanConfig,
    KnownTruthAbsrelSubsetConfig,
    KnownTruthBabappaAbsrelComparisonConfig,
    KnownTruthBabappaAbsrelDecisionConfig,
    KnownTruthBenchmarkPlanValidationConfig,
    KnownTruthBenchmarkRunConfig,
    KnownTruthBenchmarkSuitePlanConfig,
    compare_known_truth_babappa_absrel,
    make_known_truth_babappa_absrel_decision_report,
    parse_known_truth_absrel_results,
    plan_known_truth_absrel_comparison,
    plan_known_truth_benchmark_suite,
    run_known_truth_benchmark,
    select_known_truth_absrel_subset,
    validate_known_truth_benchmark_plan,
)


def test_run_known_truth_smoke_creates_truth_schema(tmp_path):
    benchmark = tmp_path / "known_truth_benchmark_smoke"
    summary = run_known_truth_benchmark(KnownTruthBenchmarkRunConfig(profile="smoke", outdir=str(benchmark)))
    assert summary["status"] == "ok"
    manifest = benchmark / "simulated_families" / "benchmark_truth_manifest.tsv"
    assert manifest.exists()
    first = next(path for path in (benchmark / "simulated_families").iterdir() if path.is_dir())
    assert (first / "family_truth.json").exists()
    assert (first / "branch_site_truth.tsv").exists()
    assert (first / "selected_sites.tsv").exists()
    assert (first / "selected_branches.tsv").exists()


def test_absrel_subset_selector_stratifies_smoke(tmp_path):
    benchmark = tmp_path / "known_truth_benchmark_smoke"
    run_known_truth_benchmark(KnownTruthBenchmarkRunConfig(profile="smoke", outdir=str(benchmark)))
    subset_dir = tmp_path / "known_truth_absrel_subset_smoke"
    summary = select_known_truth_absrel_subset(
        KnownTruthAbsrelSubsetConfig(
            benchmark_dir=str(benchmark),
            outdir=str(subset_dir),
            max_families=12,
            stratify_by="regime,truth_class,ood_status,saturation_tier",
        )
    )
    assert summary["n_selected"] == 12
    text = (subset_dir / "absrel_subset.tsv").read_text()
    assert "foreground_branch" in text
    assert "ood_status" in text


def test_absrel_planner_writes_only_absrel_user_run_scripts(tmp_path):
    benchmark = tmp_path / "known_truth_benchmark_smoke"
    run_known_truth_benchmark(KnownTruthBenchmarkRunConfig(profile="smoke", outdir=str(benchmark)))
    subset_dir = tmp_path / "subset"
    select_known_truth_absrel_subset(KnownTruthAbsrelSubsetConfig(benchmark_dir=str(benchmark), outdir=str(subset_dir), max_families=4))
    plan_dir = tmp_path / "absrel_plan"
    summary = plan_known_truth_absrel_comparison(
        KnownTruthAbsrelPlanConfig(subset=str(subset_dir / "absrel_subset.tsv"), outdir=str(plan_dir))
    )
    assert summary["codeml_included"] is False
    assert summary["meme_included"] is False
    scripts = "\n".join(path.read_text() for path in plan_dir.glob("*.sh"))
    assert "USER-RUN ONLY" in scripts
    assert "codeml" not in scripts.lower()
    assert "meme" not in scripts.lower()


def test_absrel_parser_marks_missing_outputs_pending(tmp_path):
    benchmark = tmp_path / "known_truth_benchmark_smoke"
    run_known_truth_benchmark(KnownTruthBenchmarkRunConfig(profile="smoke", outdir=str(benchmark)))
    subset_dir = tmp_path / "subset"
    select_known_truth_absrel_subset(KnownTruthAbsrelSubsetConfig(benchmark_dir=str(benchmark), outdir=str(subset_dir), max_families=3))
    plan_dir = tmp_path / "absrel_plan"
    plan_known_truth_absrel_comparison(KnownTruthAbsrelPlanConfig(subset=str(subset_dir / "absrel_subset.tsv"), outdir=str(plan_dir)))
    outdir = tmp_path / "absrel_results"
    summary = parse_known_truth_absrel_results(
        KnownTruthAbsrelParseConfig(absrel_run_dir=str(plan_dir), truth_dir=str(benchmark), outdir=str(outdir))
    )
    assert summary["pending"] == 3
    assert "pending_not_run" in (outdir / "absrel_results.tsv").read_text()


def test_babappa_vs_absrel_comparison_uses_simulated_truth(tmp_path):
    benchmark = tmp_path / "known_truth_benchmark_smoke"
    run_known_truth_benchmark(KnownTruthBenchmarkRunConfig(profile="smoke", outdir=str(benchmark)))
    subset_dir = tmp_path / "subset"
    select_known_truth_absrel_subset(KnownTruthAbsrelSubsetConfig(benchmark_dir=str(benchmark), outdir=str(subset_dir), max_families=6))
    plan_dir = tmp_path / "absrel_plan"
    plan_known_truth_absrel_comparison(KnownTruthAbsrelPlanConfig(subset=str(subset_dir / "absrel_subset.tsv"), outdir=str(plan_dir)))
    results_dir = tmp_path / "absrel_results"
    parse_known_truth_absrel_results(KnownTruthAbsrelParseConfig(absrel_run_dir=str(plan_dir), truth_dir=str(benchmark), outdir=str(results_dir)))
    comparison_dir = tmp_path / "comparison"
    summary = compare_known_truth_babappa_absrel(
        KnownTruthBabappaAbsrelComparisonConfig(
            babappa_report=str(benchmark / "report"),
            absrel_results=str(results_dir / "absrel_results.tsv"),
            truth_dir=str(benchmark),
            outdir=str(comparison_dir),
        )
    )
    assert summary["simulated_truth_ground_truth"] is True
    payload = json.loads((comparison_dir / "babappa_absrel_method_comparison.json").read_text())
    assert payload["absrel_is_comparator_not_ground_truth"] is True


def test_plan_validator_detects_missing_user_run_marker(tmp_path):
    plan = tmp_path / "plan"
    plan.mkdir()
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
        (plan / name).write_text("#!/usr/bin/env bash\nset -euo pipefail\necho ok\n")
    summary = validate_known_truth_benchmark_plan(KnownTruthBenchmarkPlanValidationConfig(plan_dir=str(plan)))
    assert summary["status"] == "fail"
    assert "missing_user_run_marker" in (plan / "plan_validation.tsv").read_text()


def test_decision_report_incomplete_when_outputs_missing(tmp_path):
    outdir = tmp_path / "decision"
    summary = make_known_truth_babappa_absrel_decision_report(
        KnownTruthBabappaAbsrelDecisionConfig(
            babappa_report=str(tmp_path / "missing_report"),
            absrel_results=str(tmp_path / "missing.tsv"),
            comparison_dir=str(tmp_path / "missing_comparison"),
            outdir=str(outdir),
        )
    )
    assert summary["decision"] == "pilot_incomplete"


def test_suite_plan_validates_and_excludes_codeml_meme(tmp_path):
    plan_dir = tmp_path / "suite_plan"
    plan_known_truth_benchmark_suite(
        KnownTruthBenchmarkSuitePlanConfig(
            profile="pilot",
            outdir=str(plan_dir),
            include_absrel=True,
            absrel_max_families=300,
            conda_env="molevo",
        )
    )
    summary = validate_known_truth_benchmark_plan(KnownTruthBenchmarkPlanValidationConfig(plan_dir=str(plan_dir)))
    assert summary["status"] == "ok"
    scripts = "\n".join(path.read_text() for path in plan_dir.glob("*.sh"))
    assert "codeml" not in scripts.lower()
    assert "meme" not in scripts.lower()


def test_docs_state_not_replacement():
    text = Path("docs/BABAPPA_KNOWN_TRUTH_BENCHMARK.md").read_text()
    assert "not a likelihood-method replacement" in text
