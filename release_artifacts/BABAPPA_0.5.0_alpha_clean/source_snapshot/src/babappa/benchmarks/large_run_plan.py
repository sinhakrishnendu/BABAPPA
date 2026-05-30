"""Planning artifacts for large BABAPPA site-level validation runs.

This module intentionally writes user-run command scripts only. It must never
execute large benchmark commands.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from babappa import __version__

LARGE_RUN_PLAN_VERSION = __version__
TIERS = ["low", "moderate", "high", "extreme"]
METHODS = ["identity", "codon_dropout"]
N_CODONS = 300
N_TAXA = 16
EXPECTED_POSITIVE_SITE_FRACTION = 0.02465


@dataclass(frozen=True)
class LargeRunPlanConfig:
    """Configuration for large-run planning artifact generation."""

    scale: int
    families_per_tier: int
    outdir: str
    negative_downsample_ratio: float
    methods: List[str] = field(default_factory=lambda: ["identity", "codon_dropout"])
    external_methods: List[str] = field(default_factory=list)
    require_aligners: bool = False
    with_site_maps: bool = False

    def __post_init__(self) -> None:
        if self.scale < 1:
            raise ValueError("scale must be >= 1")
        if self.families_per_tier < 1:
            raise ValueError("families_per_tier must be >= 1")
        if self.families_per_tier * len(TIERS) != self.scale:
            raise ValueError("scale must equal families_per_tier * 4 tiers")
        if self.negative_downsample_ratio <= 0:
            raise ValueError("negative_downsample_ratio must be > 0")
        if not self.methods:
            raise ValueError("methods must be non-empty")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def plan_large_run(config: LargeRunPlanConfig) -> dict:
    """Write non-executing large-run planning artifacts."""
    outdir = Path(config.outdir)
    names = _names(config.scale)
    expected = _expected_outputs(config, names)
    commands_path = outdir / "large_run_commands.sh"
    reference_path = outdir / "large_run_commands_commented_reference.sh"
    monitor_path = outdir / "monitor_commands.sh"
    external_path = outdir / "external_aligner_run_commands.sh"
    expected_path = outdir / "expected_outputs.json"
    markdown_path = outdir / "large_run_plan.md"

    commands = _large_run_commands(config, names)
    commands_path.write_text(_render_executable_script(commands), encoding="utf-8")
    commands_path.chmod(0o755)
    reference_path.write_text(_render_commented_reference(commands), encoding="utf-8")
    monitor_path.write_text(_render_monitor_commands(names), encoding="utf-8")
    external_path.write_text(_render_external_aligner_commands(config, names), encoding="utf-8")
    external_path.chmod(0o755)
    _write_json(expected_path, expected)
    markdown_path.write_text(_render_markdown(config, expected), encoding="utf-8")

    return {
        "status": "ok",
        "outdir": str(outdir),
        "commands": str(commands_path),
        "commented_reference": str(reference_path),
        "monitor_commands": str(monitor_path),
        "external_aligner_commands": str(external_path),
        "expected_outputs": str(expected_path),
        "markdown": str(markdown_path),
        "expected_raw_site_rows": expected["expected_raw_site_rows"],
        "approximate_downsampled_site_rows": expected["approximate_downsampled_site_rows"],
    }


def _names(scale: int) -> Dict[str, str]:
    suffix = str(scale)
    return {
        "panel": f"saturation_panel_{suffix}",
        "dataset": f"dataset_saturation_{suffix}",
        "site_oracle": f"site_oracle_saturation_{suffix}",
        "site_dataset": f"site_dataset_saturation_{suffix}",
        "site_leakage": f"site_leakage_saturation_{suffix}",
        "site_baseline": f"site_baseline_saturation_{suffix}",
        "site_neural": f"site_neural_saturation_{suffix}",
        "calibration_temperature": f"site_neural_calibration_saturation_{suffix}",
        "calibration_quantile": f"site_neural_calibration_quantile_saturation_{suffix}",
        "calibration_compare": f"site_calibration_compare_saturation_{suffix}",
        "site_policy": f"site_neural_policy_saturation_{suffix}",
        "site_stratified": f"site_neural_stratified_saturation_{suffix}",
        "site_to_gene": f"site_to_gene_saturation_{suffix}",
        "aggregation_controls": f"site_aggregation_controls_saturation_{suffix}",
        "aggregation_policy": f"site_to_gene_policy_saturation_{suffix}",
        "site_model_compare": f"site_model_compare_saturation_{suffix}",
        "site_stability": f"site_stability_saturation_{suffix}",
        "report": f"report_site_robustness_saturation_{suffix}",
        "summary": f"run_summary_site_robustness_saturation_{suffix}",
    }


def _expected_outputs(config: LargeRunPlanConfig, names: Dict[str, str]) -> dict:
    expected_gene_family_count = config.scale
    expected_family_method_rows = config.scale * len(config.methods)
    expected_raw_site_rows = expected_family_method_rows * N_CODONS
    expected_positive_site_rows = int(round(expected_raw_site_rows * EXPECTED_POSITIVE_SITE_FRACTION))
    approximate_downsampled_site_rows = int(
        min(
            expected_raw_site_rows,
            round(expected_positive_site_rows * (1.0 + config.negative_downsample_ratio)),
        )
    )
    return {
        "large_run_plan_version": LARGE_RUN_PLAN_VERSION,
        "scale": config.scale,
        "families_per_tier": config.families_per_tier,
        "tiers": TIERS,
        "methods": list(config.methods),
        "external_methods": list(config.external_methods),
        "require_aligners": config.require_aligners,
        "with_site_maps": config.with_site_maps,
        "n_taxa": N_TAXA,
        "n_codons": N_CODONS,
        "expected_gene_family_count": expected_gene_family_count,
        "expected_family_method_rows": expected_family_method_rows,
        "expected_raw_site_rows": expected_raw_site_rows,
        "expected_positive_site_fraction_assumption": EXPECTED_POSITIVE_SITE_FRACTION,
        "expected_positive_site_rows": expected_positive_site_rows,
        "negative_downsample_ratio": config.negative_downsample_ratio,
        "approximate_downsampled_site_rows": approximate_downsampled_site_rows,
        "output_directories": names,
        "recommended_disk_warning": (
            "Large site datasets can require substantial disk space; inspect output growth "
            "before starting neural training."
        ),
        "recommended_runtime_warning": (
            "Large-run simulation, tensorization, site dataset construction, and neural training "
            "must be run by the user outside Codex."
        ),
        "planner_executed_commands": [],
    }


def _large_run_commands(config: LargeRunPlanConfig, names: Dict[str, str]) -> List[str]:
    dataset_dirs = ",".join(f"{names['panel']}/tiers/{tier}/dataset" for tier in TIERS)
    tier_names = ",".join(TIERS)
    method_names = ",".join(config.methods)
    return [
        "conda activate molevo",
        f"babappa make-saturation-panel --outdir {names['panel']} --n-families-per-tier {config.families_per_tier} --tiers {tier_names} --n-taxa {N_TAXA} --n-codons {N_CODONS} --seed 42 --positive-rate 0.5 --methods {method_names} --dropout-rate 0.02",
        f"babappa validate-saturation-panel --panel-dir {names['panel']}",
        f"babappa merge-datasets --dataset-dirs {dataset_dirs} --names {tier_names} --outdir {names['dataset']} --seed 42 --resplit",
        f"babappa validate-merged-dataset --dataset-dir {names['dataset']}",
        f"babappa extract-site-labels --dataset-dir {names['dataset']} --outdir {names['site_oracle']}",
        f"babappa build-site-dataset --dataset-dir {names['dataset']} --oracle-labels {names['site_oracle']}/site_oracle_labels.tsv --outdir {names['site_dataset']} --negative-downsample-ratio {config.negative_downsample_ratio:g} --seed 42",
        f"babappa validate-site-dataset --site-dataset-dir {names['site_dataset']}",
        f"babappa audit-site-leakage --site-dataset-dir {names['site_dataset']} --outdir {names['site_leakage']}",
        f"babappa train-site-baseline --site-dataset-dir {names['site_dataset']} --outdir {names['site_baseline']} --seed 42 --epochs 300 --learning-rate 0.05 --l2 0.001",
        f"babappa train-site-neural --site-dataset-dir {names['site_dataset']} --outdir {names['site_neural']} --device auto --epochs 30 --batch-size 256 --learning-rate 0.001 --weight-decay 0.0001 --hidden-dim 64 --dropout 0.1 --positive-class-weight auto --monitor-metric val_auroc",
        f"babappa validate-site-neural --model-dir {names['site_neural']}",
        f"babappa calibrate-site-neural --model-dir {names['site_neural']} --outdir {names['calibration_temperature']} --target-fdr 0.10 --calibration-method temperature",
        f"babappa calibrate-site-neural --model-dir {names['site_neural']} --outdir {names['calibration_quantile']} --target-fdr 0.10 --calibration-method quantile --n-bins 20",
        f"babappa compare-site-calibrations --calibration-dirs {names['calibration_temperature']},{names['calibration_quantile']} --names temperature,quantile --outdir {names['calibration_compare']}",
        f"babappa site-threshold-policy --predictions {names['site_neural']}/site_neural_predictions.tsv --outdir {names['site_policy']} --probability-column prob_positive --label-column y_site --split-column split --selection-split calib --target-fdr 0.10 --precision-floor 0.80 --recall-floor 0.80",
        f"babappa site-stratified-eval --predictions {names['site_neural']}/site_neural_predictions.tsv --outdir {names['site_stratified']} --probability-column prob_positive --label-column y_site --threshold-policy-dir {names['site_policy']}",
        f"babappa aggregate-sites --predictions {names['site_neural']}/site_neural_predictions.tsv --gene-dataset-dir {names['dataset']} --outdir {names['site_to_gene']}",
        f"babappa validate-site-aggregation --aggregation-dir {names['site_to_gene']}",
        f"babappa aggregation-controls --predictions {names['site_neural']}/site_neural_predictions.tsv --gene-dataset-dir {names['dataset']} --outdir {names['aggregation_controls']} --n-permutations 50 --seed 42",
        f"babappa aggregation-threshold-policy --aggregation-dir {names['site_to_gene']} --outdir {names['aggregation_policy']} --score-column max_site_probability --label-column gene_label --split-column split --selection-split calib --target-fdr 0.10",
        f"babappa compare-site-models --site-baseline-dir {names['site_baseline']} --site-neural-dir {names['site_neural']} --site-stratified-eval-dir {names['site_stratified']} --site-aggregation-dir {names['site_to_gene']} --outdir {names['site_model_compare']}",
        f"babappa site-stability-benchmark --site-dataset-dir {names['site_dataset']} --outdir {names['site_stability']} --seeds 42,43,44 --device auto --epochs 10 --batch-size 256 --learning-rate 0.001 --max-train-items 50000 --max-val-items 10000 --max-calib-items 10000 --max-test-items 10000",
        f"babappa make-report --outdir {names['report']} --title \"BABAPPA site-level robustness {config.scale}-family report\" --merged-dataset-dir {names['dataset']} --site-label-dir {names['site_oracle']} --site-dataset-dir {names['site_dataset']} --site-leakage-audit-dir {names['site_leakage']} --site-baseline-dir {names['site_baseline']} --site-neural-dir {names['site_neural']} --site-calibration-dir {names['calibration_temperature']} --site-threshold-policy-dir {names['site_policy']} --site-stratified-eval-dir {names['site_stratified']} --site-aggregation-dir {names['site_to_gene']} --site-stability-dir {names['site_stability']} --site-model-comparison-dir {names['site_model_compare']} --site-aggregation-controls-dir {names['aggregation_controls']} --site-aggregation-threshold-policy-dir {names['aggregation_policy']} --site-calibration-comparison-dir {names['calibration_compare']}",
        f"babappa summarize-run --outdir {names['summary']} --title \"BABAPPA site-level robustness {config.scale}-family summary\" --merged-dataset-dir {names['dataset']} --site-label-dir {names['site_oracle']} --site-dataset-dir {names['site_dataset']} --site-leakage-audit-dir {names['site_leakage']} --site-baseline-dir {names['site_baseline']} --site-neural-dir {names['site_neural']} --site-calibration-dir {names['calibration_temperature']} --site-threshold-policy-dir {names['site_policy']} --site-stratified-eval-dir {names['site_stratified']} --site-aggregation-dir {names['site_to_gene']} --site-stability-dir {names['site_stability']} --site-model-comparison-dir {names['site_model_compare']} --site-aggregation-controls-dir {names['aggregation_controls']} --site-aggregation-threshold-policy-dir {names['aggregation_policy']} --site-calibration-comparison-dir {names['calibration_compare']} --report-dir {names['report']}",
    ]


def _render_executable_script(commands: List[str]) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "# USER-RUN ONLY — DO NOT EXECUTE IN CODEX",
            "# This script launches a large BABAPPA validation run for offline user execution.",
            "",
            *commands,
            "",
        ]
    )


def _render_commented_reference(commands: List[str]) -> str:
    return "\n".join(
        [
            "# USER-RUN ONLY — DO NOT EXECUTE IN CODEX",
            "# Fully commented reference copy of large_run_commands.sh.",
            "#",
            *[f"# {command}" for command in commands],
            "",
        ]
    )


def _render_monitor_commands(names: Dict[str, str]) -> str:
    dirs = " ".join(names.values())
    return "\n".join(
        [
            "# USER-RUN ONLY — DO NOT EXECUTE IN CODEX",
            "# Monitoring command templates for manually launched large runs.",
            "# ps -eo pid,etime,pcpu,pmem,args | grep '[p]ython'",
            "# Use platform-specific accelerator monitoring outside this portable template.",
            f"# du -sh {dirs} 2>/dev/null | sort -h",
            f"# find {names['panel']} {names['site_dataset']} {names['site_neural']} -type f -printf '%TY-%Tm-%Td %TH:%TM %p\\n' 2>/dev/null | sort | tail -n 30",
            f"# find {names['site_neural']} -path '*/logs/*' -type f -print -exec tail -n 20 {{}} \\;",
            f"# wc -l {names['site_dataset']}/site_features.tsv {names['site_neural']}/site_neural_predictions.tsv {names['site_to_gene']}/site_to_gene_predictions.tsv 2>/dev/null",
            f"# babappa validate-saturation-panel --panel-dir {names['panel']}",
            f"# babappa validate-merged-dataset --dataset-dir {names['dataset']}",
            f"# babappa validate-site-dataset --site-dataset-dir {names['site_dataset']}",
            f"# babappa validate-site-neural --model-dir {names['site_neural']}",
            f"# babappa validate-site-aggregation --aggregation-dir {names['site_to_gene']}",
            "",
        ]
    )


def _render_external_aligner_commands(config: LargeRunPlanConfig, names: Dict[str, str]) -> str:
    methods = list(config.methods)
    for method in config.external_methods:
        if method not in methods:
            methods.append(method)
    method_names = ",".join(methods or METHODS)
    require_available = "true" if config.require_aligners else "false"
    mapped_oracle = f"{names['site_oracle']}_mapped"
    mapped_site_dataset = f"{names['site_dataset']}_mapped"
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "# USER-RUN ONLY — DO NOT EXECUTE IN CODEX",
            "# Optional external-aligner validation path. Start at 1K before considering larger scales.",
            "",
            "conda activate molevo",
            "babappa check-aligners --json-out aligner_status_external.json",
            f"babappa align-external --sim-dir {names['panel']}/tiers/moderate/sim --outdir align_external_{config.scale}_moderate --methods {method_names} --require-available {require_available} --threads 1",
            f"babappa build-site-map --sim-dir {names['panel']}/tiers/moderate/sim --align-dir align_external_{config.scale}_moderate --outdir site_map_external_{config.scale}_moderate",
            f"babappa validate-site-map --site-map-dir site_map_external_{config.scale}_moderate",
            f"babappa build-tensors --sim-dir {names['panel']}/tiers/moderate/sim --align-dir align_external_{config.scale}_moderate --outdir tensors_external_{config.scale}_moderate",
            f"babappa index-dataset --tensor-dir tensors_external_{config.scale}_moderate --outdir dataset_external_{config.scale}_moderate --seed 42",
            f"babappa extract-site-labels --dataset-dir dataset_external_{config.scale}_moderate --outdir {mapped_oracle} --site-map-dir site_map_external_{config.scale}_moderate --aligned-site-mode mapped",
            f"babappa build-site-dataset --dataset-dir dataset_external_{config.scale}_moderate --oracle-labels {mapped_oracle}/site_oracle_labels.tsv --outdir {mapped_site_dataset} --negative-downsample-ratio {config.negative_downsample_ratio:g} --seed 42 --require-mappable-sites",
            f"babappa validate-site-dataset --site-dataset-dir {mapped_site_dataset}",
            "",
        ]
    )


def _render_markdown(config: LargeRunPlanConfig, expected: dict) -> str:
    outputs = expected["output_directories"]
    lines = [
        "# BABAPPA large-run plan",
        "",
        "## Purpose",
        "",
        "Prepare user-run commands for staged site-level validation without executing heavy work in Codex.",
        "",
        "## Scientific rationale",
        "",
        "The site-level BABAPPA framework is stable at 1K scale and must now be validated at larger scales with null controls, aggregation thresholds, and calibration comparison.",
        "",
        "## Exact scale",
        "",
        f"- Families: {config.scale}",
        f"- Families per tier: {config.families_per_tier}",
        f"- Tiers: {', '.join(TIERS)}",
        f"- Methods: {', '.join(config.methods)}",
        f"- Optional external methods: {', '.join(config.external_methods) if config.external_methods else 'none'}",
        "",
        "## Expected row counts",
        "",
        f"- Expected family-method rows: {expected['expected_family_method_rows']}",
        f"- Expected raw site rows: {expected['expected_raw_site_rows']}",
        f"- Expected positive site rows: {expected['expected_positive_site_rows']}",
        f"- Approximate downsampled site rows: {expected['approximate_downsampled_site_rows']}",
        "",
        "## Output directories",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in outputs.items())
    lines.extend(
        [
            "",
            "## When to stop",
            "",
            "- Stop if leakage audit reports forbidden columns.",
            "- Stop if validation commands fail.",
            "- Stop if disk usage grows beyond available capacity.",
            "- Stop if site-neural validation AUROC collapses or probabilities degenerate.",
            "",
            "## When to continue",
            "",
            "- Continue after each validation command reports ok.",
            "- Continue from 10K to larger scale only after aggregation controls beat null/decoy controls.",
            "",
            "## Why Codex must not execute the job",
            "",
            "10K, 50K, and 100K validation runs create large datasets and long neural training jobs. Codex only prepares reproducible commands and monitoring templates.",
            "",
            "## After-run summary commands",
            "",
            f"- `babappa validate-report --report-dir {outputs['report']}`",
            f"- `babappa validate-run-summary --summary-dir {outputs['summary']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
