"""Planner for branch-conditioned 10K validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

BRANCH_METHODS = ["identity", "mafft", "babappalign", "muscle-with-quarantine"]


@dataclass(frozen=True)
class BranchConditioned10kPlanConfig:
    outdir: str = "branch_conditioned_10k_plan"
    tiers: List[str] | None = None
    negative_downsample_ratio: float = 5.0
    max_output_rows_per_tier: int = 1_000_000
    output_suffix: str = "streamed"
    conda_sh: str = "/home/rajamosai/miniconda3/etc/profile.d/conda.sh"
    conda_env: str = "molevo"
    neural_epochs: int = 10
    batch_size: int = 256
    max_train_items: int = 50000
    max_val_items: int = 10000
    max_calib_items: int = 10000
    max_test_items: int = 10000
    n_control_permutations: int = 20

    def __post_init__(self) -> None:
        if self.tiers is None:
            object.__setattr__(self, "tiers", ["low", "moderate", "high", "extreme"])
        if not self.tiers:
            raise ValueError("tiers must not be empty")
        if self.negative_downsample_ratio <= 0:
            raise ValueError("negative_downsample_ratio must be > 0")
        if self.max_output_rows_per_tier <= 0:
            raise ValueError("max_output_rows_per_tier must be > 0")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def plan_branch_conditioned_10k(config: BranchConditioned10kPlanConfig) -> dict:
    """Write user-run branch-conditioned 10K validation scripts and metadata."""
    outdir = Path(config.outdir)
    run_path = outdir / "run_branch_conditioned_10k.sh"
    monitor_path = outdir / "monitor_branch_conditioned_10k.sh"
    validate_path = outdir / "validate_branch_conditioned_10k.sh"
    summarize_path = outdir / "summarize_branch_conditioned_10k.sh"
    expected_path = outdir / "expected_outputs.json"
    markdown_path = outdir / "branch_conditioned_10k_plan.md"
    expected = _expected_outputs(config)
    run_path.write_text(_run_script(config), encoding="utf-8")
    monitor_path.write_text(_monitor_script(config), encoding="utf-8")
    validate_path.write_text(_validate_script(config), encoding="utf-8")
    summarize_path.write_text(_summarize_script(config), encoding="utf-8")
    expected_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(config, expected), encoding="utf-8")
    for script in [run_path, monitor_path, validate_path, summarize_path]:
        os.chmod(script, 0o755)
    return {
        "status": "ok",
        "outdir": str(outdir),
        "run": str(run_path),
        "monitor": str(monitor_path),
        "validate": str(validate_path),
        "summarize": str(summarize_path),
        "expected_outputs": str(expected_path),
        "markdown": str(markdown_path),
        "tiers": config.tiers,
        "methods": BRANCH_METHODS,
    }


def _header(config: BranchConditioned10kPlanConfig) -> str:
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"source {config.conda_sh}",
        f"conda activate {config.conda_env}",
        "",
        "# USER-RUN ONLY — DO NOT EXECUTE IN CODEX",
        "",
    ])


def _tier_dirs(tier: str) -> dict:
    suffix = ""
    return _tier_dirs_with_suffix(tier, suffix)


def _tier_dirs_with_suffix(tier: str, suffix: str) -> dict:
    suffix_part = f"_{suffix}" if suffix else ""
    return {
        "dataset": f"dataset_fast_external_10k_{tier}",
        "site_map": f"site_map_fast_external_10k_{tier}",
        "labels": f"branch_site_oracle_fast_external_10k_{tier}",
        "dataset_out": f"branch_site_dataset_fast_external_10k_{tier}{suffix_part}",
        "leakage": f"branch_site_leakage_fast_external_10k_{tier}{suffix_part}",
        "baseline": f"branch_site_baseline_fast_external_10k_{tier}{suffix_part}",
        "neural": f"branch_site_neural_fast_external_10k_{tier}{suffix_part}",
        "calibration": f"branch_site_calibration_fast_external_10k_{tier}{suffix_part}",
        "threshold": f"branch_site_threshold_policy_fast_external_10k_{tier}{suffix_part}",
        "aggregation": f"branch_aggregation_fast_external_10k_{tier}{suffix_part}",
        "controls": f"branch_aggregation_controls_fast_external_10k_{tier}{suffix_part}",
        "aggregation_policy": f"branch_aggregation_policy_fast_external_10k_{tier}{suffix_part}",
        "summary": f"branch_site_run_summary_fast_external_10k_{tier}{suffix_part}",
    }


def _run_script(config: BranchConditioned10kPlanConfig) -> str:
    lines = [_header(config), "mkdir -p logs", "ts=$(date +%Y%m%d_%H%M%S)", "log=logs/branch_conditioned_10k_${ts}.log", "exec > >(tee -a \"$log\") 2>&1", "", "echo \"BABAPPA branch-conditioned 10K run started: $(date)\"", "echo \"This script reuses completed fast external 10K tensors/datasets; it does not run alignments.\"", ""]
    for tier in config.tiers or []:
        d = _tier_dirs_with_suffix(tier, config.output_suffix)
        lines.extend([
            f"echo \"=== Branch-conditioned tier: {tier} ===\"",
            f"test -d {d['dataset']} || (echo 'Missing required dataset directory: {d['dataset']}' >&2; exit 1)",
            f"test -d {d['site_map']} || (echo 'Missing required site-map directory: {d['site_map']}' >&2; exit 1)",
            f"if [ -s {d['labels']}/branch_site_oracle_labels.tsv ]; then echo 'Reusing existing branch-site labels: {d['labels']}'; else babappa extract-branch-site-labels --dataset-dir {d['dataset']} --site-map-dir {d['site_map']} --outdir {d['labels']} --aligned-site-mode mapped --foreground-source auto --streaming-output; fi",
            f"babappa validate-branch-site-labels --label-dir {d['labels']}",
            f"test ! -e {d['dataset_out']} || (echo 'Refusing to overwrite existing streamed dataset output: {d['dataset_out']}' >&2; exit 1)",
            f"babappa build-branch-site-dataset --dataset-dir {d['dataset']} --branch-site-labels {d['labels']}/branch_site_oracle_labels.tsv --outdir {d['dataset_out']} --negative-downsample-ratio {config.negative_downsample_ratio:g} --seed 42 --require-mappable-sites --streaming --max-output-rows {config.max_output_rows_per_tier}",
            f"babappa validate-branch-site-dataset --branch-site-dataset-dir {d['dataset_out']}",
            f"babappa audit-branch-site-leakage --branch-site-dataset-dir {d['dataset_out']} --outdir {d['leakage']}",
            f"babappa validate-branch-site-leakage --leakage-dir {d['leakage']}",
            f"babappa train-branch-site-baseline --branch-site-dataset-dir {d['dataset_out']} --outdir {d['baseline']} --epochs 300 --learning-rate 0.05",
            f"babappa validate-branch-site-baseline --model-dir {d['baseline']}",
            f"babappa train-branch-site-neural --branch-site-dataset-dir {d['dataset_out']} --outdir {d['neural']} --device auto --epochs {config.neural_epochs} --batch-size {config.batch_size} --max-train-items {config.max_train_items} --max-val-items {config.max_val_items} --max-calib-items {config.max_calib_items} --max-test-items {config.max_test_items}",
            f"babappa validate-branch-site-neural --model-dir {d['neural']}",
            f"babappa calibrate-branch-site-neural --model-dir {d['neural']} --outdir {d['calibration']}",
            f"babappa validate-branch-site-calibration --calibration-dir {d['calibration']}",
            f"babappa branch-site-threshold-policy --predictions {d['calibration']}/branch_site_calibrated_predictions.tsv --outdir {d['threshold']} --probability-column prob_positive_raw --calibrated-probability-column prob_positive_calibrated",
            f"babappa validate-branch-site-threshold-policy --policy-dir {d['threshold']}",
            f"babappa aggregate-branch-sites --predictions {d['neural']}/branch_site_neural_predictions.tsv --outdir {d['aggregation']}",
            f"babappa validate-branch-aggregation --aggregation-dir {d['aggregation']}",
            f"babappa branch-aggregation-controls --predictions {d['neural']}/branch_site_neural_predictions.tsv --outdir {d['controls']} --n-permutations {config.n_control_permutations}",
            f"babappa validate-branch-aggregation-controls --controls-dir {d['controls']}",
            f"babappa branch-aggregation-threshold-policy --aggregation-dir {d['aggregation']} --outdir {d['aggregation_policy']}",
            f"babappa validate-branch-aggregation-threshold-policy --policy-dir {d['aggregation_policy']}",
            f"babappa summarize-branch-site-run --outdir {d['summary']} --title 'BABAPPA branch-conditioned fast external 10K {tier} summary' --branch-site-label-dir {d['labels']} --branch-site-dataset-dir {d['dataset_out']} --branch-site-leakage-dir {d['leakage']} --branch-site-baseline-dir {d['baseline']} --branch-site-neural-dir {d['neural']} --branch-site-calibration-dir {d['calibration']} --branch-aggregation-dir {d['aggregation']} --branch-aggregation-controls-dir {d['controls']} --branch-site-threshold-policy-dir {d['threshold']} --branch-aggregation-threshold-policy-dir {d['aggregation_policy']}",
            f"babappa validate-branch-site-run-summary --summary-dir {d['summary']}",
            "",
        ])
    lines.append("echo \"BABAPPA branch-conditioned 10K run completed: $(date)\"")
    return "\n".join(lines) + "\n"


def _monitor_script(config: BranchConditioned10kPlanConfig) -> str:
    lines = [_header(config), "echo 'Active BABAPPA / Python / aligner processes:'", "pgrep -af 'babappa|python|mafft|muscle|babappalign' || true", "", "echo 'Latest branch-conditioned log:'", "ls -t logs/branch_conditioned_10k_*.log 2>/dev/null | head -1 | xargs -r tail -60", ""]
    for tier in config.tiers or []:
        d = _tier_dirs_with_suffix(tier, config.output_suffix)
        lines.extend([
            f"echo '--- {tier} directory presence ---'",
            f"for dir in {d['labels']} {d['dataset_out']} {d['leakage']} {d['baseline']} {d['neural']} {d['calibration']} {d['aggregation']} {d['summary']}; do [ -d \"$dir\" ] && echo present:$dir || echo missing:$dir; done",
            f"[ -f {d['labels']}/branch_site_oracle_labels.tsv ] && echo labels_rows=$(($(wc -l < {d['labels']}/branch_site_oracle_labels.tsv)-1)) || true",
            f"[ -f {d['neural']}/branch_site_neural_predictions.tsv ] && echo neural_predictions=$(($(wc -l < {d['neural']}/branch_site_neural_predictions.tsv)-1)) || true",
            "",
        ])
    return "\n".join(lines) + "\n"


def _validate_script(config: BranchConditioned10kPlanConfig) -> str:
    lines = [_header(config), "run_or_missing() { if [ -d \"$2\" ]; then babappa \"$1\" \"$3\" \"$2\"; else echo \"MISSING: $2\"; fi; }", ""]
    for tier in config.tiers or []:
        d = _tier_dirs_with_suffix(tier, config.output_suffix)
        lines.extend([
            f"echo '=== Validate branch-conditioned tier {tier} ==='",
            f"run_or_missing validate-branch-site-labels {d['labels']} --label-dir",
            f"run_or_missing validate-branch-site-dataset {d['dataset_out']} --branch-site-dataset-dir",
            f"run_or_missing validate-branch-site-leakage {d['leakage']} --leakage-dir",
            f"run_or_missing validate-branch-site-baseline {d['baseline']} --model-dir",
            f"run_or_missing validate-branch-site-neural {d['neural']} --model-dir",
            f"run_or_missing validate-branch-site-calibration {d['calibration']} --calibration-dir",
            f"run_or_missing validate-branch-site-threshold-policy {d['threshold']} --policy-dir",
            f"run_or_missing validate-branch-aggregation {d['aggregation']} --aggregation-dir",
            f"run_or_missing validate-branch-aggregation-controls {d['controls']} --controls-dir",
            f"run_or_missing validate-branch-aggregation-threshold-policy {d['aggregation_policy']} --policy-dir",
            f"run_or_missing validate-branch-site-run-summary {d['summary']} --summary-dir",
            "",
        ])
    return "\n".join(lines) + "\n"


def _summarize_script(config: BranchConditioned10kPlanConfig) -> str:
    lines = [_header(config)]
    for tier in config.tiers or []:
        d = _tier_dirs_with_suffix(tier, config.output_suffix)
        lines.extend([
            f"babappa summarize-branch-site-run --outdir {d['summary']} --title 'BABAPPA branch-conditioned fast external 10K {tier} summary' --branch-site-label-dir {d['labels']} --branch-site-dataset-dir {d['dataset_out']} --branch-site-leakage-dir {d['leakage']} --branch-site-baseline-dir {d['baseline']} --branch-site-neural-dir {d['neural']} --branch-site-calibration-dir {d['calibration']} --branch-aggregation-dir {d['aggregation']} --branch-aggregation-controls-dir {d['controls']} --branch-site-threshold-policy-dir {d['threshold']} --branch-aggregation-threshold-policy-dir {d['aggregation_policy']}",
            f"babappa validate-branch-site-run-summary --summary-dir {d['summary']}",
        ])
    return "\n".join(lines) + "\n"


def _expected_outputs(config: BranchConditioned10kPlanConfig) -> dict:
    tiers = config.tiers or []
    return {
        "scale": "fast_external_10k_reuse",
        "tiers": tiers,
        "methods_preserved": BRANCH_METHODS,
        "diagnostic_methods_excluded": ["prank", "tcoffee"],
        "does_not_regenerate_alignments": True,
        "does_not_regenerate_site_neural": True,
        "branch_truth_status": "proxy_from_foreground_taxon_if_explicit_branch_site_truth_absent",
        "negative_downsample_ratio": config.negative_downsample_ratio,
        "max_output_rows_per_tier": config.max_output_rows_per_tier,
        "output_suffix": config.output_suffix,
        "expected_output_directories": {tier: _tier_dirs_with_suffix(tier, config.output_suffix) for tier in tiers},
        "runtime_warning": "Branch-conditioned dataset expansion is family x method x branch x site; streamed row caps are used for research-alpha validation.",
        "disk_warning": "Branch-site feature TSVs can be larger than site-level TSVs by approximately the number of taxa/branches.",
        "prototype_note": "Branch-conditioned 10K is a prototype validation; memory-safe row caps are used until explicit branch-site simulator truth is implemented.",
        "scientific_boundary": "Research-alpha branch-conditioned validation; not empirical branch-site inference.",
    }


def _markdown(config: BranchConditioned10kPlanConfig, expected: dict) -> str:
    return "\n".join([
        "# BABAPPA branch-conditioned 10K validation plan",
        "",
        "## Why this is next",
        "",
        "Fast external 10K validated site-level oracle evidence and site-to-gene aggregation. The next gap is branch-conditioned branch-site inference.",
        "",
        "## What this plan does",
        "",
        "- Reuses completed fast external 10K datasets and site maps.",
        "- Extracts branch-conditioned oracle labels using explicit branch labels when available, otherwise a foreground-taxon proxy is reported.",
        f"- Builds branch-aware features with streaming enabled, negative downsample ratio {config.negative_downsample_ratio:g}, and max {config.max_output_rows_per_tier} output rows per tier.",
        "- Does not run alignments, site-level neural training, 10K generation, 50K, or 100K.",
        "- Uses suffixed output directories so failed partial non-streamed outputs are not overwritten.",
        "",
        "## Prototype cap note",
        "",
        "Branch-conditioned 10K is a prototype validation; memory-safe row caps are used until explicit branch-site simulator truth is implemented.",
        "",
        "## Method policy",
        "",
        "- Production-fast methods remain identity, MAFFT, BABAPPAlign, and MUSCLE with quarantine.",
        "- PRANK and T-Coffee remain diagnostic only and excluded from this plan.",
        "",
        "## How to run",
        "",
        "```bash",
        "cd /home/rajamosai/Desktop/BABAPPA && bash branch_conditioned_10k_plan/run_branch_conditioned_10k.sh",
        "```",
        "",
        "## How to monitor",
        "",
        "```bash",
        "cd /home/rajamosai/Desktop/BABAPPA && bash branch_conditioned_10k_plan/monitor_branch_conditioned_10k.sh",
        "```",
        "",
        "## Recovery",
        "",
        "A failed tier can be resumed by rerunning commands for that tier after validating the last completed directory. The scripts are USER-RUN ONLY and were not executed by Codex.",
        "",
        "## Why 100K waits",
        "",
        "100K should wait until branch-conditioned 10K passes because branch-site supervision, leakage controls, calibration, and decoys are now the scientific bottleneck.",
        "",
        "## Expected outputs",
        "",
        json.dumps(expected["expected_output_directories"], indent=2, sort_keys=True),
        "",
    ])
