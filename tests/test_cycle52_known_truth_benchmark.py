import json
from pathlib import Path

from babappa.benchmarks.known_truth import (
    KnownTruthAlignmentConfig,
    KnownTruthBenchmarkDesignConfig,
    KnownTruthBenchmarkPlanConfig,
    KnownTruthBenchmarkReportConfig,
    KnownTruthCalibrationEvaluationConfig,
    KnownTruthEvaluationConfig,
    KnownTruthReferenceComparisonPlanConfig,
    KnownTruthScoringConfig,
    KnownTruthSimulationConfig,
    KnownTruthValidationConfig,
    bh_qvalues,
    design_known_truth_benchmark,
    evaluate_known_truth_benchmark,
    evaluate_known_truth_calibration,
    make_known_truth_benchmark_report,
    plan_known_truth_benchmark,
    plan_known_truth_reference_comparison,
    run_known_truth_alignments,
    score_known_truth_benchmark,
    simulate_known_truth_benchmark,
    validate_known_truth_benchmark,
)
from babappa.empirical.reference_eval import MethodClaimReadinessConfig, validate_method_claim_readiness


def test_known_truth_design_writes_required_regimes(tmp_path):
    outdir = tmp_path / "design"
    summary = design_known_truth_benchmark(KnownTruthBenchmarkDesignConfig(outdir=str(outdir)))
    assert summary["status"] == "ok"
    regimes = (outdir / "regime_manifest.tsv").read_text()
    assert "null_low_divergence" in regimes
    assert "positive_moderate_branch_site" in regimes
    assert "ood_extreme_saturation" in regimes


def test_smoke_simulation_creates_explicit_truth_files(tmp_path):
    design = tmp_path / "design"
    design_known_truth_benchmark(KnownTruthBenchmarkDesignConfig(outdir=str(design)))
    sim = tmp_path / "simulated"
    summary = simulate_known_truth_benchmark(
        KnownTruthSimulationConfig(design_dir=str(design), profile="smoke", outdir=str(sim))
    )
    assert summary["n_families"] == 12
    manifest = sim / "benchmark_truth_manifest.tsv"
    assert manifest.exists()
    first_family = next(path for path in sim.iterdir() if path.is_dir())
    assert (first_family / "family_truth.json").exists()
    assert (first_family / "branch_site_truth.tsv").exists()
    validation = validate_known_truth_benchmark(KnownTruthValidationConfig(benchmark_dir=str(sim), outdir=str(tmp_path / "validation")))
    assert validation["status"] == "ok"


def test_smoke_scoring_and_evaluation_pipeline(tmp_path):
    design = tmp_path / "design"
    design_known_truth_benchmark(KnownTruthBenchmarkDesignConfig(outdir=str(design)))
    sim = tmp_path / "benchmark" / "simulated_families"
    simulate_known_truth_benchmark(KnownTruthSimulationConfig(design_dir=str(design), profile="smoke", outdir=str(sim)))
    align = tmp_path / "benchmark" / "alignments"
    run_known_truth_alignments(KnownTruthAlignmentConfig(sim_dir=str(sim), outdir=str(align)))
    scores = tmp_path / "benchmark" / "babappa_scores"
    score_known_truth_benchmark(
        KnownTruthScoringConfig(
            sim_dir=str(sim),
            alignment_dir=str(align),
            deployable_model_package="deployable_model_conservative_branch_site_100k_mps",
            outdir=str(scores),
            score_backend="smoke_surrogate",
        )
    )
    evaluation = tmp_path / "benchmark" / "evaluation"
    summary = evaluate_known_truth_benchmark(
        KnownTruthEvaluationConfig(
            truth=str(sim / "benchmark_truth_manifest.tsv"),
            scores=str(scores),
            outdir=str(evaluation),
        )
    )
    assert summary["status"] == "ok"
    payload = json.loads((evaluation / "evaluation_summary.json").read_text())
    assert "ood_abstention_rate" in payload["gene_level"]
    calibration = tmp_path / "benchmark" / "calibration_evaluation"
    cal_summary = evaluate_known_truth_calibration(
        KnownTruthCalibrationEvaluationConfig(
            truth=str(sim / "benchmark_truth_manifest.tsv"),
            scores=str(scores),
            outdir=str(calibration),
        )
    )
    assert cal_summary["status"] == "ok"


def test_bh_qvalues_are_monotone_for_synthetic_scores():
    q = bh_qvalues([0.001, 0.01, 0.2, 0.5])
    assert q[0] <= q[1] <= q[2] <= q[3]
    assert q[0] <= 0.01


def test_reference_plan_scripts_are_user_run_only(tmp_path):
    outdir = tmp_path / "reference_plan"
    summary = plan_known_truth_reference_comparison(
        KnownTruthReferenceComparisonPlanConfig(
            benchmark_dir="known_truth_benchmark_paper",
            outdir=str(outdir),
            tools="codeml,absrel,meme",
            max_families=100,
        )
    )
    assert summary["user_run_only"] is True
    assert "USER-RUN ONLY" in (outdir / "run_hyphy_absrel_known_truth.sh").read_text()


def test_known_truth_report_contains_claim_boundary(tmp_path):
    design = tmp_path / "design"
    design_known_truth_benchmark(KnownTruthBenchmarkDesignConfig(outdir=str(design)))
    benchmark = tmp_path / "benchmark"
    sim = benchmark / "simulated_families"
    simulate_known_truth_benchmark(KnownTruthSimulationConfig(design_dir=str(design), profile="smoke", outdir=str(sim)))
    align = benchmark / "alignments"
    scores = benchmark / "babappa_scores"
    run_known_truth_alignments(KnownTruthAlignmentConfig(sim_dir=str(sim), outdir=str(align)))
    score_known_truth_benchmark(
        KnownTruthScoringConfig(
            sim_dir=str(sim),
            alignment_dir=str(align),
            deployable_model_package="deployable_model_conservative_branch_site_100k_mps",
            outdir=str(scores),
            score_backend="smoke_surrogate",
        )
    )
    evaluate_known_truth_benchmark(KnownTruthEvaluationConfig(truth=str(sim / "benchmark_truth_manifest.tsv"), scores=str(scores), outdir=str(benchmark / "evaluation")))
    evaluate_known_truth_calibration(KnownTruthCalibrationEvaluationConfig(truth=str(sim / "benchmark_truth_manifest.tsv"), scores=str(scores), outdir=str(benchmark / "calibration_evaluation")))
    report = benchmark / "report"
    make_known_truth_benchmark_report(KnownTruthBenchmarkReportConfig(benchmark_dir=str(benchmark), outdir=str(report)))
    assert "does not support empirical discovery claims" in (report / "known_truth_benchmark_report.md").read_text()


def test_plan_known_truth_paper_marks_long_run_handoff(tmp_path):
    outdir = tmp_path / "plan"
    summary = plan_known_truth_benchmark(
        KnownTruthBenchmarkPlanConfig(
            profile="paper",
            design_dir="known_truth_benchmark_design_v1",
            outdir=str(outdir),
        )
    )
    assert summary["user_run_only"] is True
    assert "USER-RUN ONLY" in (outdir / "run_known_truth_benchmark.sh").read_text()


def test_method_claim_readiness_recognizes_known_truth_layer(tmp_path):
    known_truth = tmp_path / "known_truth.json"
    known_truth.write_text(json.dumps({"status": "ok"}))
    simulation = tmp_path / "simulation.json"
    simulation.write_text(json.dumps({"status": "ok"}))
    drosophila = tmp_path / "drosophila.json"
    drosophila.write_text(
        json.dumps(
            {
                "status": "ok",
                "n_families": 140,
                "hyphy_absrel_positive_families": 73,
                "babappa_native_calibrated_support": 14,
                "positive_agreement_against_hyphy": 0.04,
                "concordance_counts": {"concordant_positive": 3, "concordant_negative": 56},
            }
        )
    )
    wrky = tmp_path / "wrky.json"
    wrky.write_text(json.dumps({"decision_category": "diagnostic_positive_calibration_pending"}))
    summary = validate_method_claim_readiness(
        MethodClaimReadinessConfig(
            simulation_summary=str(simulation),
            drosophila_summary=str(drosophila),
            wrky_report=str(wrky),
            known_truth_summary=str(known_truth),
            outdir=str(tmp_path / "claim"),
        )
    )
    assert summary["ready_as_simulation_validated_framework"] is True
    assert summary["not_ready_as_likelihood_replacement"] is True
