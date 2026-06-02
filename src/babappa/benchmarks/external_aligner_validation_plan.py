"""Planning artifacts for clean external-aligner validation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from babappa import __version__

EXTERNAL_ALIGNER_VALIDATION_PLAN_VERSION = __version__
DEFAULT_TIERS = ["low", "moderate", "high", "extreme"]
DEFAULT_METHODS = ["identity", "mafft", "babappalign", "muscle"]
DEFAULT_EXCLUDE_METHODS = ["prank"]
DEFAULT_CONDA_SH = "/home/rajamosai/miniconda3/etc/profile.d/conda.sh"
DEFAULT_CONDA_ENV = "molevo"


@dataclass(frozen=True)
class ExternalAlignerValidationPlanConfig:
    """Configuration for external-aligner validation planning."""

    panel_dir: str
    outdir: str
    methods: List[str] = field(default_factory=lambda: list(DEFAULT_METHODS))
    optional_methods: List[str] = field(default_factory=list)
    exclude_methods: List[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_METHODS))
    tiers: List[str] = field(default_factory=lambda: list(DEFAULT_TIERS))
    negative_downsample_ratio: float = 10.0
    conda_sh: str = DEFAULT_CONDA_SH
    conda_env: str = DEFAULT_CONDA_ENV
    max_method_failure_fraction: float = 0.01
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if not self.methods:
            raise ValueError("methods must be non-empty")
        if not self.tiers:
            raise ValueError("tiers must be non-empty")
        if not str(self.conda_sh).strip():
            raise ValueError("conda_sh must be non-empty")
        if not str(self.conda_env).strip():
            raise ValueError("conda_env must be non-empty")
        if self.negative_downsample_ratio <= 0:
            raise ValueError("negative_downsample_ratio must be > 0")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.max_method_failure_fraction < 0 or self.max_method_failure_fraction > 1:
            raise ValueError("max_method_failure_fraction must be between 0 and 1")
        effective = _effective_methods(self.methods, self.optional_methods, self.exclude_methods)
        if not effective:
            raise ValueError("effective methods must be non-empty after exclusions")
        if "codon_dropout" in effective:
            raise ValueError(
                "codon_dropout is quarantined for mapped oracle-label validation; "
                "omit it unless adding a dedicated unmappable-noise-control workflow"
            )
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def plan_external_aligner_validation(config: ExternalAlignerValidationPlanConfig) -> dict:
    """Write manual execution scripts for external-aligner validation without executing jobs."""
    outdir = Path(config.outdir)
    commands_path = outdir / "external_aligner_validation_commands.sh"
    monitor_path = outdir / "monitor_external_aligner_validation.sh"
    expected_path = outdir / "expected_external_outputs.json"
    markdown_path = outdir / "external_aligner_validation_plan.md"
    commands = _commands(config)
    expected = _expected(config)
    commands_path.write_text(_render_script(commands, config), encoding="utf-8")
    commands_path.chmod(0o755)
    monitor_path.write_text(_render_monitor(config), encoding="utf-8")
    _write_json(expected_path, expected)
    markdown_path.write_text(_render_markdown(config, expected), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "commands": str(commands_path),
        "monitor": str(monitor_path),
        "expected_outputs": str(expected_path),
        "markdown": str(markdown_path),
        "methods": expected["effective_methods"],
        "optional_methods": list(config.optional_methods),
        "exclude_methods": list(config.exclude_methods),
        "tiers": list(config.tiers),
    }


@dataclass(frozen=True)
class ExternalCompletedTierReportPlanConfig:
    """Configuration for completing reports for already generated external tiers."""

    tiers: List[str]
    outdir: str
    conda_sh: str = DEFAULT_CONDA_SH
    conda_env: str = DEFAULT_CONDA_ENV

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError("tiers must be non-empty")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def plan_complete_external_tier_reports(config: ExternalCompletedTierReportPlanConfig) -> dict:
    """Write a manual execution script to add calibration/policies to completed external tiers."""
    outdir = Path(config.outdir)
    commands_path = outdir / "complete_external_tier_reports_commands.sh"
    expected_path = outdir / "expected_outputs.json"
    markdown_path = outdir / "complete_external_tier_reports_plan.md"
    commands = _complete_tier_commands(config.tiers)
    commands_path.write_text(_render_script(commands, config), encoding="utf-8")
    commands_path.chmod(0o755)
    expected = {
        "plan_version": EXTERNAL_ALIGNER_VALIDATION_PLAN_VERSION,
        "tiers": list(config.tiers),
        "planner_executed_commands": [],
        "generated_script": str(commands_path),
    }
    _write_json(expected_path, expected)
    markdown_path.write_text(_complete_tier_markdown(config), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "commands": str(commands_path),
        "expected_outputs": str(expected_path),
        "markdown": str(markdown_path),
        "tiers": list(config.tiers),
    }


@dataclass(frozen=True)
class ExternalExtremeRecoveryPlanConfig:
    """Configuration for planning the missing external extreme tier."""

    panel_dir: str
    outdir: str
    methods: List[str] = field(default_factory=lambda: list(DEFAULT_METHODS))
    negative_downsample_ratio: float = 10.0
    timeout_seconds: int = 300
    conda_sh: str = DEFAULT_CONDA_SH
    conda_env: str = DEFAULT_CONDA_ENV
    max_method_failure_fraction: float = 0.01

    def __post_init__(self) -> None:
        if not self.methods:
            raise ValueError("methods must be non-empty")
        if "prank" in self.methods:
            raise ValueError("PRANK is diagnostic-only and excluded from extreme fast recovery")
        if self.negative_downsample_ratio <= 0:
            raise ValueError("negative_downsample_ratio must be > 0")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def plan_external_extreme_recovery(config: ExternalExtremeRecoveryPlanConfig) -> dict:
    """Write a manual execution recovery plan for the missing external extreme tier."""
    outdir = Path(config.outdir)
    commands_path = outdir / "external_extreme_recovery_commands.sh"
    expected_path = outdir / "expected_outputs.json"
    markdown_path = outdir / "external_extreme_recovery_plan.md"
    plan_config = ExternalAlignerValidationPlanConfig(
        panel_dir=config.panel_dir,
        outdir=config.outdir,
        methods=config.methods,
        optional_methods=[],
        exclude_methods=["prank"],
        tiers=["extreme"],
        negative_downsample_ratio=config.negative_downsample_ratio,
        conda_sh=config.conda_sh,
        conda_env=config.conda_env,
        max_method_failure_fraction=config.max_method_failure_fraction,
        timeout_seconds=config.timeout_seconds,
    )
    commands = _commands(plan_config)
    commands_path.write_text(_render_script(commands, config), encoding="utf-8")
    commands_path.chmod(0o755)
    expected = _expected(plan_config) | {"recovery_scope": "external_extreme_only"}
    _write_json(expected_path, expected)
    markdown_path.write_text(_extreme_markdown(config, expected), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "commands": str(commands_path),
        "expected_outputs": str(expected_path),
        "markdown": str(markdown_path),
        "methods": list(config.methods),
    }


@dataclass(frozen=True)
class FastExternal10kPlanConfig:
    """Configuration for a manual execution fast external-aligner 10K validation plan."""

    outdir: str
    panel_outdir: str
    families_per_tier: int = 2500
    tiers: List[str] = field(default_factory=lambda: list(DEFAULT_TIERS))
    methods: List[str] = field(default_factory=lambda: list(DEFAULT_METHODS))
    negative_downsample_ratio: float = 10.0
    conda_sh: str = DEFAULT_CONDA_SH
    conda_env: str = DEFAULT_CONDA_ENV
    timeout_seconds: int = 300
    max_method_failure_fraction: float = 0.01
    neural_epochs: int = 10
    aggregation_control_permutations: int = 20
    n_codons: int = 300

    def __post_init__(self) -> None:
        if self.families_per_tier <= 0:
            raise ValueError("families_per_tier must be > 0")
        if not self.tiers:
            raise ValueError("tiers must be non-empty")
        if not self.methods:
            raise ValueError("methods must be non-empty")
        banned = {"prank", "tcoffee", "t_coffee"}
        selected_banned = sorted(banned.intersection(set(self.methods)))
        if selected_banned:
            raise ValueError(
                "diagnostic-only methods are excluded from the fast 10K plan: "
                + ",".join(selected_banned)
            )
        if self.negative_downsample_ratio <= 0:
            raise ValueError("negative_downsample_ratio must be > 0")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.max_method_failure_fraction < 0 or self.max_method_failure_fraction > 1:
            raise ValueError("max_method_failure_fraction must be between 0 and 1")
        if self.neural_epochs <= 0:
            raise ValueError("neural_epochs must be > 0")
        if self.aggregation_control_permutations <= 0:
            raise ValueError("aggregation_control_permutations must be > 0")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def plan_fast_external_10k(config: FastExternal10kPlanConfig) -> dict:
    """Write a manual execution fast external-aligner 10K plan without executing jobs."""
    outdir = Path(config.outdir)
    run_path = outdir / "run_fast_external_10k.sh"
    monitor_path = outdir / "monitor_fast_external_10k.sh"
    validate_path = outdir / "validate_fast_external_10k.sh"
    summarize_path = outdir / "summarize_fast_external_10k.sh"
    expected_path = outdir / "expected_outputs.json"
    markdown_path = outdir / "fast_external_10k_plan.md"

    expected = _fast10k_expected(config)
    run_path.write_text(_fast10k_run_script(config), encoding="utf-8")
    monitor_path.write_text(_fast10k_monitor_script(config), encoding="utf-8")
    validate_path.write_text(_fast10k_validate_script(config), encoding="utf-8")
    summarize_path.write_text(_fast10k_summarize_script(config), encoding="utf-8")
    markdown_path.write_text(_fast10k_markdown(config, expected), encoding="utf-8")
    _write_json(expected_path, expected)
    for path in [run_path, monitor_path, validate_path, summarize_path]:
        path.chmod(0o755)

    return {
        "status": "ok",
        "outdir": str(outdir),
        "run": str(run_path),
        "monitor": str(monitor_path),
        "validate": str(validate_path),
        "summarize": str(summarize_path),
        "expected_outputs": str(expected_path),
        "markdown": str(markdown_path),
        "tiers": list(config.tiers),
        "methods": list(config.methods),
        "planner_executed_commands": [],
    }


def _commands(config: ExternalAlignerValidationPlanConfig) -> List[str]:
    commands = ["babappa check-aligners --json-out external_aligner_status.json"]
    method_csv = ",".join(_effective_methods(config.methods, config.optional_methods, config.exclude_methods))
    for tier in config.tiers:
        names = _tier_names(tier)
        sim_dir = f"{config.panel_dir}/tiers/{tier}/sim"
        commands.extend(
            [
                f"# Tier: {tier}",
                f"babappa align-external --sim-dir {sim_dir} --outdir {names['align']} --methods {method_csv} --require-available false --threads 1 --timeout-seconds {config.timeout_seconds} --max-method-failure-fraction {config.max_method_failure_fraction:g}",
                f"babappa validate-align --align-dir {names['align']}",
                f"babappa build-site-map --sim-dir {sim_dir} --align-dir {names['align']} --outdir {names['site_map']}",
                f"babappa validate-site-map --site-map-dir {names['site_map']}",
                f"babappa aligner-method-policy --align-dir {names['align']} --site-map-dir {names['site_map']} --outdir {names['method_policy']} --max-conflict-fraction 0.03 --max-frame-error-fraction 0.0 --max-method-failure-fraction {config.max_method_failure_fraction:g}",
                f"babappa validate-aligner-method-policy --policy-dir {names['method_policy']}",
                f"USABLE_METHODS_{tier}=$(python -c \"import csv; rows=list(csv.DictReader(open('{names['method_policy']}/method_policy.tsv'), delimiter='\\t')); print(','.join(r['method'] for r in rows if r['recommendation'] in ('usable','caution')))\")",
                f"test -n \"$USABLE_METHODS_{tier}\" || (echo 'No usable methods for tier {tier}; inspect {names['method_policy']}/method_policy.tsv' >&2; exit 1)",
                f"babappa build-tensors --sim-dir {sim_dir} --align-dir {names['align']} --outdir {names['tensors']} --methods \"$USABLE_METHODS_{tier}\"",
                f"babappa index-dataset --tensor-dir {names['tensors']} --outdir {names['dataset']} --seed 42",
                f"babappa extract-site-labels --dataset-dir {names['dataset']} --outdir {names['site_oracle']} --site-map-dir {names['site_map']} --aligned-site-mode mapped",
                f"babappa build-site-dataset --dataset-dir {names['dataset']} --oracle-labels {names['site_oracle']}/site_oracle_labels.tsv --outdir {names['site_dataset']} --negative-downsample-ratio {config.negative_downsample_ratio:g} --seed 42 --require-mappable-sites",
                f"babappa validate-site-dataset --site-dataset-dir {names['site_dataset']}",
                f"babappa audit-site-leakage --site-dataset-dir {names['site_dataset']} --outdir {names['site_leakage']}",
                f"babappa train-site-baseline --site-dataset-dir {names['site_dataset']} --outdir {names['site_baseline']} --seed 42 --epochs 300 --learning-rate 0.05 --l2 0.001",
                f"babappa train-site-neural --site-dataset-dir {names['site_dataset']} --outdir {names['site_neural']} --device auto --epochs 10 --batch-size 256 --learning-rate 0.001 --weight-decay 0.0001 --hidden-dim 64 --dropout 0.1 --positive-class-weight auto --monitor-metric val_auroc --max-train-items 50000 --max-val-items 10000 --max-calib-items 10000 --max-test-items 10000",
                f"babappa validate-site-neural --model-dir {names['site_neural']}",
                f"babappa aggregate-sites --predictions {names['site_neural']}/site_neural_predictions.tsv --gene-dataset-dir {names['dataset']} --outdir {names['site_to_gene']}",
                f"babappa validate-site-aggregation --aggregation-dir {names['site_to_gene']}",
                f"babappa aggregation-controls --predictions {names['site_neural']}/site_neural_predictions.tsv --gene-dataset-dir {names['dataset']} --outdir {names['aggregation_controls']} --n-permutations 20 --seed 42",
                f"babappa make-report --outdir {names['report']} --title \"BABAPPA external-aligner {tier} validation report\" --merged-dataset-dir {names['dataset']} --site-label-dir {names['site_oracle']} --site-dataset-dir {names['site_dataset']} --site-leakage-audit-dir {names['site_leakage']} --site-baseline-dir {names['site_baseline']} --site-neural-dir {names['site_neural']} --site-aggregation-dir {names['site_to_gene']} --site-aggregation-controls-dir {names['aggregation_controls']}",
                f"babappa summarize-run --outdir {names['summary']} --title \"BABAPPA external-aligner {tier} validation summary\" --merged-dataset-dir {names['dataset']} --site-label-dir {names['site_oracle']} --site-dataset-dir {names['site_dataset']} --site-leakage-audit-dir {names['site_leakage']} --site-baseline-dir {names['site_baseline']} --site-neural-dir {names['site_neural']} --site-aggregation-dir {names['site_to_gene']} --site-aggregation-controls-dir {names['aggregation_controls']} --report-dir {names['report']}",
            ]
        )
    return commands


def _complete_tier_commands(tiers: List[str]) -> List[str]:
    commands: List[str] = []
    for tier in tiers:
        names = _tier_names(tier)
        calibration = f"site_neural_calibration_external_aligner_validation_{tier}"
        aggregation_policy = f"aggregation_policy_external_aligner_validation_{tier}"
        report = f"report_external_aligner_validation_{tier}"
        summary = f"run_summary_external_aligner_validation_{tier}"
        commands.extend(
            [
                f"# Complete tier: {tier}",
                f"babappa calibrate-site-neural --model-dir {names['site_neural']} --outdir {calibration} --target-fdr 0.10",
                f"babappa validate-site-calibration --calibration-dir {calibration}",
                f"babappa aggregation-threshold-policy --aggregation-dir {names['site_to_gene']} --outdir {aggregation_policy} --score-column max_site_probability --label-column gene_label --split-column split --selection-split calib --target-fdr 0.10",
                f"babappa validate-aggregation-threshold-policy --policy-dir {aggregation_policy}",
                f"babappa make-report --outdir {report} --title \"BABAPPA external-aligner {tier} validation report\" --merged-dataset-dir {names['dataset']} --site-label-dir {names['site_oracle']} --site-dataset-dir {names['site_dataset']} --site-leakage-audit-dir {names['site_leakage']} --site-baseline-dir {names['site_baseline']} --site-neural-dir {names['site_neural']} --site-calibration-dir {calibration} --site-aggregation-dir {names['site_to_gene']} --site-aggregation-controls-dir {names['aggregation_controls']} --site-aggregation-threshold-policy-dir {aggregation_policy}",
                f"babappa summarize-run --outdir {summary} --title \"BABAPPA external-aligner {tier} validation summary\" --merged-dataset-dir {names['dataset']} --site-label-dir {names['site_oracle']} --site-dataset-dir {names['site_dataset']} --site-leakage-audit-dir {names['site_leakage']} --site-baseline-dir {names['site_baseline']} --site-neural-dir {names['site_neural']} --site-calibration-dir {calibration} --site-aggregation-dir {names['site_to_gene']} --site-aggregation-controls-dir {names['aggregation_controls']} --site-aggregation-threshold-policy-dir {aggregation_policy} --report-dir {report}",
            ]
        )
    return commands


def _tier_names(tier: str) -> Dict[str, str]:
    prefix = f"external_aligner_validation_{tier}"
    return {
        "align": f"align_{prefix}",
        "site_map": f"site_map_{prefix}",
        "method_policy": f"method_policy_{prefix}",
        "tensors": f"tensors_{prefix}",
        "dataset": f"dataset_{prefix}",
        "site_oracle": f"site_oracle_{prefix}",
        "site_dataset": f"site_dataset_{prefix}",
        "site_leakage": f"site_leakage_{prefix}",
        "site_baseline": f"site_baseline_{prefix}",
        "site_neural": f"site_neural_{prefix}",
        "site_to_gene": f"site_to_gene_{prefix}",
        "aggregation_controls": f"aggregation_controls_{prefix}",
        "report": f"report_{prefix}",
        "summary": f"run_summary_{prefix}",
    }


def _expected(config: ExternalAlignerValidationPlanConfig) -> dict:
    effective_methods = _effective_methods(config.methods, config.optional_methods, config.exclude_methods)
    return {
        "external_aligner_validation_plan_version": EXTERNAL_ALIGNER_VALIDATION_PLAN_VERSION,
        "panel_dir": config.panel_dir,
        "methods": list(config.methods),
        "optional_methods": list(config.optional_methods),
        "exclude_methods": list(config.exclude_methods),
        "effective_methods": effective_methods,
        "tiers": list(config.tiers),
        "negative_downsample_ratio": config.negative_downsample_ratio,
        "conda_sh": config.conda_sh,
        "conda_env": config.conda_env,
        "timeout_seconds": config.timeout_seconds,
        "max_method_failure_fraction": config.max_method_failure_fraction,
        "prank_policy": "diagnostic_only_excluded_from_default_fast_ensemble",
        "tcoffee_policy": "optional_diagnostic_excluded_unless_requested",
        "codon_dropout_policy": "excluded_by_default_quarantined_unmappable_noise_control",
        "planner_executed_commands": [],
        "output_directories_by_tier": {tier: _tier_names(tier) for tier in config.tiers},
    }


def _render_script(commands: List[str], config) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"source {config.conda_sh}",
            f"conda activate {config.conda_env}",
            "",
            "# MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING",
            "# Fast external-aligner mapped-site validation with method-policy quarantine.",
            "",
            *commands,
            "",
        ]
    )


def _render_monitor(config: ExternalAlignerValidationPlanConfig) -> str:
    tier_dirs = []
    for tier in config.tiers:
        tier_dirs.extend(_tier_names(tier).values())
    dirs = " ".join(tier_dirs)
    return "\n".join(
        [
            "# MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING",
            "# Monitoring command templates for external-aligner validation.",
            "# ps -eo pid,etime,pcpu,pmem,args | grep '[p]ython'",
            "# Use platform-specific accelerator monitoring outside this portable template.",
            f"# du -sh {dirs} 2>/dev/null | sort -h",
            f"# find {dirs} -type f -printf '%TY-%Tm-%Td %TH:%TM %p\\n' 2>/dev/null | sort | tail -n 50",
            f"# wc -l {' '.join(f'site_dataset_external_aligner_validation_{tier}/site_features.tsv' for tier in config.tiers)} 2>/dev/null",
            "",
        ]
    )


def _render_markdown(config: ExternalAlignerValidationPlanConfig, expected: dict) -> str:
    lines = [
        "# BABAPPA external-aligner validation plan",
        "",
        "## Purpose",
        "",
        "Prepare a fast mapped-site validation using identity, MAFFT, BABAPPAlign, and MUSCLE by default without executing jobs automatically.",
        "",
        "## Configuration",
        "",
        f"- Panel directory: `{config.panel_dir}`",
        f"- Requested methods: {', '.join(config.methods)}",
        f"- Optional diagnostic methods: {', '.join(config.optional_methods) if config.optional_methods else 'none'}",
        f"- Excluded methods: {', '.join(config.exclude_methods) if config.exclude_methods else 'none'}",
        f"- Effective methods: {', '.join(expected['effective_methods'])}",
        f"- Tiers: {', '.join(config.tiers)}",
        f"- Negative downsample ratio: {config.negative_downsample_ratio:g}",
        f"- Timeout seconds: {config.timeout_seconds}",
        f"- Max method failure fraction: {config.max_method_failure_fraction:g}",
        "",
        "## Method policy",
        "",
        "PRANK is diagnostic-only and excluded from the default fast ensemble. T-Coffee is optional diagnostic and excluded unless requested. Generated scripts build `method_policy.tsv` per tier and tensorize only methods marked usable or caution.",
        "",
        "## Generated files",
        "",
        "- `external_aligner_validation_commands.sh`",
        "- `monitor_external_aligner_validation.sh`",
        "- `expected_external_outputs.json`",
        "",
    ]
    return "\n".join(lines)


def _complete_tier_markdown(config: ExternalCompletedTierReportPlanConfig) -> str:
    return "\n".join(
        [
            "# BABAPPA completed external-tier report completion plan",
            "",
            "This plan writes manual execution commands to add site calibration, aggregation threshold policies, and refreshed reports for already completed external tiers.",
            "",
            f"- Tiers: {', '.join(config.tiers)}",
            "- The script is MANUAL EXECUTION SCRIPT and was not executed automatically.",
            "",
        ]
    )


def _extreme_markdown(config: ExternalExtremeRecoveryPlanConfig, expected: dict) -> str:
    return "\n".join(
        [
            "# BABAPPA external extreme-tier fast recovery plan",
            "",
            "This plan recovers only the missing extreme external-aligner tier with the fast/default ensemble.",
            "",
            f"- Panel directory: `{config.panel_dir}`",
            f"- Methods: {', '.join(config.methods)}",
            "- PRANK is excluded.",
            "- MUSCLE is optional and skipped gracefully if unavailable.",
            f"- Timeout seconds: {config.timeout_seconds}",
            "- The script is MANUAL EXECUTION SCRIPT and was not executed automatically.",
            "",
        ]
    )


def _fast10k_names(tier: str) -> Dict[str, str]:
    prefix = f"fast_external_10k_{tier}"
    return {
        "align": f"align_{prefix}",
        "site_map": f"site_map_{prefix}",
        "method_policy": f"method_policy_{prefix}",
        "tensors": f"tensors_{prefix}",
        "dataset": f"dataset_{prefix}",
        "site_oracle": f"site_oracle_{prefix}",
        "site_dataset": f"site_dataset_{prefix}",
        "site_leakage": f"site_leakage_{prefix}",
        "site_baseline": f"site_baseline_{prefix}",
        "site_neural": f"site_neural_{prefix}",
        "site_calibration": f"site_calibration_{prefix}",
        "site_threshold_policy": f"site_threshold_policy_{prefix}",
        "site_to_gene": f"site_to_gene_{prefix}",
        "aggregation_controls": f"aggregation_controls_{prefix}",
        "aggregation_policy": f"aggregation_policy_{prefix}",
        "report": f"report_{prefix}",
        "summary": f"run_summary_{prefix}",
    }


def _fast10k_header(config: FastExternal10kPlanConfig) -> List[str]:
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"source {config.conda_sh}",
        f"conda activate {config.conda_env}",
        "",
        "# MANUAL EXECUTION SCRIPT — Review before running",
        "",
    ]


def _fast10k_run_script(config: FastExternal10kPlanConfig) -> str:
    method_csv = ",".join(config.methods)
    tier_csv = ",".join(config.tiers)
    expected_dirs = [config.panel_outdir, "fast_external_10k_cross_tier_summary"]
    for tier in config.tiers:
        expected_dirs.extend(
            value
            for key, value in _fast10k_names(tier).items()
            if key != "sim"
        )
    lines = [
        *_fast10k_header(config),
        'LOCK_FILE="/tmp/babappa_fast_external_10k.lock"',
        'if [[ "${BABAPPA_FAST_EXTERNAL_10K_LOGGING:-0}" != "1" ]]; then',
        "  mkdir -p logs",
        '  LOG="logs/fast_external_10k_$(date +%Y%m%d_%H%M%S).log"',
        '  echo "Logging to ${LOG}"',
        '  BABAPPA_FAST_EXTERNAL_10K_LOGGING=1 exec "$0" "$@" > >(tee -a "${LOG}") 2>&1',
        "fi",
        'exec 9>"${LOCK_FILE}"',
        'flock -n 9 || { echo "Another fast external 10K run is already active: ${LOCK_FILE}" >&2; exit 1; }',
        "",
        "EXPECTED_DIRS=(",
        *[f'  "{directory}"' for directory in expected_dirs],
        ")",
        'if [[ "${BABAPPA_FAST_EXTERNAL_10K_ALLOW_EXISTING:-0}" != "1" ]]; then',
        '  for d in "${EXPECTED_DIRS[@]}"; do',
        '    if [[ -e "${d}" ]]; then',
        '      echo "Refusing to overwrite existing output: ${d}" >&2',
        '      echo "Set BABAPPA_FAST_EXTERNAL_10K_ALLOW_EXISTING=1 only for deliberate resume/recovery." >&2',
        "      exit 1",
        "    fi",
        "  done",
        "fi",
        "",
        'echo "BABAPPA fast external 10K started at $(date)"',
        "babappa check-aligners --json-out fast_external_10k_aligner_status.json",
        (
            f"babappa make-saturation-panel --outdir {config.panel_outdir} "
            f"--n-families-per-tier {config.families_per_tier} --tiers {tier_csv} "
            f"--n-codons {config.n_codons} --seed 42 --positive-rate 0.5 "
            "--methods identity --no-build-tensors --no-index-datasets"
        ),
    ]
    for tier in config.tiers:
        names = _fast10k_names(tier)
        sim_dir = f"{config.panel_outdir}/tiers/{tier}/sim"
        usable_var = f"USABLE_METHODS_{tier}"
        lines.extend(
            [
                "",
                f'echo "===== Tier: {tier} ====="',
                f"babappa align-external --sim-dir {sim_dir} --outdir {names['align']} --methods {method_csv} --require-available false --threads 1 --timeout-seconds {config.timeout_seconds} --max-method-failure-fraction {config.max_method_failure_fraction:g}",
                f"babappa validate-align --align-dir {names['align']}",
                f"babappa build-site-map --sim-dir {sim_dir} --align-dir {names['align']} --outdir {names['site_map']}",
                f"if ! babappa validate-site-map --site-map-dir {names['site_map']}; then",
                f"  echo \"WARNING: strict site-map validation flagged tier {tier}; continuing to method policy for quarantine.\" >&2",
                "fi",
                f"babappa aligner-method-policy --align-dir {names['align']} --site-map-dir {names['site_map']} --outdir {names['method_policy']} --max-conflict-fraction 0.03 --max-frame-error-fraction 0.0 --max-method-failure-fraction {config.max_method_failure_fraction:g}",
                f"babappa validate-aligner-method-policy --policy-dir {names['method_policy']}",
                f"{usable_var}=$(python -c \"import csv; rows=list(csv.DictReader(open('{names['method_policy']}/method_policy.tsv'), delimiter='\\t')); print(','.join(r['method'] for r in rows if r['recommendation'] in ('usable','caution')))\")",
                f"test -n \"${usable_var}\" || (echo 'No usable methods for tier {tier}; inspect {names['method_policy']}/method_policy.tsv' >&2; exit 1)",
                f'echo "Tier {tier} usable methods: ${usable_var}"',
                f"babappa build-tensors --sim-dir {sim_dir} --align-dir {names['align']} --outdir {names['tensors']} --methods \"${usable_var}\"",
                f"babappa validate-tensors --tensor-dir {names['tensors']}",
                f"babappa index-dataset --tensor-dir {names['tensors']} --outdir {names['dataset']} --seed 42",
                f"babappa validate-index --index-dir {names['dataset']}",
                f"babappa extract-site-labels --dataset-dir {names['dataset']} --outdir {names['site_oracle']} --site-map-dir {names['site_map']} --aligned-site-mode mapped",
                f"babappa validate-site-labels --site-label-dir {names['site_oracle']}",
                f"babappa build-site-dataset --dataset-dir {names['dataset']} --oracle-labels {names['site_oracle']}/site_oracle_labels.tsv --outdir {names['site_dataset']} --negative-downsample-ratio {config.negative_downsample_ratio:g} --seed 42 --require-mappable-sites",
                f"babappa validate-site-dataset --site-dataset-dir {names['site_dataset']}",
                f"babappa audit-site-leakage --site-dataset-dir {names['site_dataset']} --outdir {names['site_leakage']}",
                f"babappa train-site-baseline --site-dataset-dir {names['site_dataset']} --outdir {names['site_baseline']} --seed 42 --epochs 300 --learning-rate 0.05 --l2 0.001",
                f"babappa validate-site-baseline --model-dir {names['site_baseline']}",
                f"babappa train-site-neural --site-dataset-dir {names['site_dataset']} --outdir {names['site_neural']} --device auto --epochs {config.neural_epochs} --batch-size 256 --learning-rate 0.001 --weight-decay 0.0001 --hidden-dim 64 --dropout 0.1 --positive-class-weight auto --monitor-metric val_auroc --max-train-items 50000 --max-val-items 10000 --max-calib-items 10000 --max-test-items 10000",
                f"babappa validate-site-neural --model-dir {names['site_neural']}",
                f"babappa calibrate-site-neural --model-dir {names['site_neural']} --outdir {names['site_calibration']} --target-fdr 0.10",
                f"babappa validate-site-calibration --calibration-dir {names['site_calibration']}",
                f"babappa site-threshold-policy --predictions {names['site_calibration']}/site_calibrated_predictions.tsv --outdir {names['site_threshold_policy']} --probability-column prob_positive_raw --calibrated-probability-column prob_positive_calibrated --target-fdr 0.10",
                f"babappa validate-site-threshold-policy --policy-dir {names['site_threshold_policy']}",
                f"babappa aggregate-sites --predictions {names['site_neural']}/site_neural_predictions.tsv --gene-dataset-dir {names['dataset']} --outdir {names['site_to_gene']}",
                f"babappa validate-site-aggregation --aggregation-dir {names['site_to_gene']}",
                f"babappa aggregation-controls --predictions {names['site_neural']}/site_neural_predictions.tsv --gene-dataset-dir {names['dataset']} --outdir {names['aggregation_controls']} --n-permutations {config.aggregation_control_permutations} --seed 42",
                f"babappa validate-aggregation-controls --controls-dir {names['aggregation_controls']}",
                f"babappa aggregation-threshold-policy --aggregation-dir {names['site_to_gene']} --outdir {names['aggregation_policy']} --score-column max_site_probability --label-column gene_label --split-column split --selection-split calib --target-fdr 0.10",
                f"babappa validate-aggregation-threshold-policy --policy-dir {names['aggregation_policy']}",
                f"babappa make-report --outdir {names['report']} --title \"BABAPPA fast external 10K {tier} validation report\" --merged-dataset-dir {names['dataset']} --site-label-dir {names['site_oracle']} --site-dataset-dir {names['site_dataset']} --site-leakage-audit-dir {names['site_leakage']} --site-baseline-dir {names['site_baseline']} --site-neural-dir {names['site_neural']} --site-calibration-dir {names['site_calibration']} --site-threshold-policy-dir {names['site_threshold_policy']} --site-aggregation-dir {names['site_to_gene']} --site-aggregation-controls-dir {names['aggregation_controls']} --site-aggregation-threshold-policy-dir {names['aggregation_policy']}",
                f"babappa validate-report --report-dir {names['report']}",
                f"babappa summarize-run --outdir {names['summary']} --title \"BABAPPA fast external 10K {tier} validation summary\" --merged-dataset-dir {names['dataset']} --site-label-dir {names['site_oracle']} --site-dataset-dir {names['site_dataset']} --site-leakage-audit-dir {names['site_leakage']} --site-baseline-dir {names['site_baseline']} --site-neural-dir {names['site_neural']} --site-calibration-dir {names['site_calibration']} --site-threshold-policy-dir {names['site_threshold_policy']} --site-aggregation-dir {names['site_to_gene']} --site-aggregation-controls-dir {names['aggregation_controls']} --site-aggregation-threshold-policy-dir {names['aggregation_policy']} --report-dir {names['report']}",
                f"babappa validate-run-summary --summary-dir {names['summary']}",
            ]
        )
    lines.extend(
        [
            "",
            "babappa summarize-external-tiers --tiers "
            + tier_csv
            + " --run-name fast_external_10k --outdir fast_external_10k_cross_tier_summary",
            "babappa validate-external-tier-summary --summary-dir fast_external_10k_cross_tier_summary",
            'echo "BABAPPA fast external 10K completed at $(date)"',
            "",
        ]
    )
    return "\n".join(lines)


def _fast10k_monitor_script(config: FastExternal10kPlanConfig) -> str:
    lines = [
        *_fast10k_header(config),
        'echo "== Active BABAPPA / aligner processes =="',
        "ps -eo pid,ppid,etimes,stat,pcpu,pmem,cmd | grep -E 'babappa|mafft|muscle|babappalign|python' | grep -v grep || true",
        "",
        'echo "== Accelerator status =="',
        "echo 'Use platform-specific accelerator monitoring outside this portable script.'",
        "",
        'echo "== Output directory sizes =="',
        "du -sh saturation_panel_external_fast_10k fast_external_10k_cross_tier_summary *_fast_external_10k_* 2>/dev/null | sort -h || true",
        "",
        'echo "== Latest fast external 10K log tail =="',
        "latest_log=$(ls -1t logs/fast_external_10k_*.log 2>/dev/null | head -1 || true)",
        'if [[ -n "${latest_log}" ]]; then echo "Log: ${latest_log}"; tail -80 "${latest_log}"; else echo "No fast external 10K log found"; fi',
        "",
        'echo "== Tier-stage directory presence =="',
    ]
    for tier in config.tiers:
        names = _fast10k_names(tier)
        lines.append(f'echo "-- {tier} --"')
        for key, directory in names.items():
            if key == "sim":
                directory = f"{config.panel_outdir}/tiers/{tier}/sim"
            lines.append(f'[[ -d "{directory}" ]] && echo "present {directory}" || echo "missing {directory}"')
    lines.extend(["", 'echo "== Aligned FASTA counts by tier/method =="'])
    for tier in config.tiers:
        for method in config.methods:
            align_dir = _fast10k_names(tier)["align"]
            lines.append(
                f'printf "{tier}\\t{method}\\t"; find {align_dir}/families -type f -name "*.{method}.codon.fasta" 2>/dev/null | wc -l'
            )
    lines.extend(["", 'echo "== Site neural prediction row counts =="'])
    for tier in config.tiers:
        predictions = f"{_fast10k_names(tier)['site_neural']}/site_neural_predictions.tsv"
        lines.append(
            f'if [[ -f "{predictions}" ]]; then printf "{tier}\\t"; wc -l < "{predictions}"; else echo "{tier}\\tmissing"; fi'
        )
    lines.append("")
    return "\n".join(lines)


def _fast10k_validate_script(config: FastExternal10kPlanConfig) -> str:
    lines = [
        *_fast10k_header(config),
        "status=0",
        "run_or_missing() {",
        "  local label=\"$1\"",
        "  local path=\"$2\"",
        "  shift 2",
        "  if [[ -e \"${path}\" ]]; then",
        "    echo \"== ${label} ==\"",
        "    \"$@\" || status=1",
        "  else",
        "    echo \"MISSING ${label}: ${path}\"",
        "    status=1",
        "  fi",
        "}",
        "",
    ]
    for tier in config.tiers:
        names = _fast10k_names(tier)
        lines.extend(
            [
                f'echo "===== Validate tier: {tier} ====="',
                f"run_or_missing 'align {tier}' '{names['align']}' babappa validate-align --align-dir {names['align']}",
                f"run_or_missing 'site-map {tier}' '{names['site_map']}' babappa validate-site-map --site-map-dir {names['site_map']}",
                f"run_or_missing 'method-policy {tier}' '{names['method_policy']}' babappa validate-aligner-method-policy --policy-dir {names['method_policy']}",
                f"run_or_missing 'tensors {tier}' '{names['tensors']}' babappa validate-tensors --tensor-dir {names['tensors']}",
                f"run_or_missing 'dataset {tier}' '{names['dataset']}' babappa validate-index --index-dir {names['dataset']}",
                f"run_or_missing 'site-labels {tier}' '{names['site_oracle']}' babappa validate-site-labels --site-label-dir {names['site_oracle']}",
                f"run_or_missing 'site-dataset {tier}' '{names['site_dataset']}' babappa validate-site-dataset --site-dataset-dir {names['site_dataset']}",
                f"run_or_missing 'site-baseline {tier}' '{names['site_baseline']}' babappa validate-site-baseline --model-dir {names['site_baseline']}",
                f"run_or_missing 'site-neural {tier}' '{names['site_neural']}' babappa validate-site-neural --model-dir {names['site_neural']}",
                f"run_or_missing 'site-calibration {tier}' '{names['site_calibration']}' babappa validate-site-calibration --calibration-dir {names['site_calibration']}",
                f"run_or_missing 'site-threshold-policy {tier}' '{names['site_threshold_policy']}' babappa validate-site-threshold-policy --policy-dir {names['site_threshold_policy']}",
                f"run_or_missing 'site-aggregation {tier}' '{names['site_to_gene']}' babappa validate-site-aggregation --aggregation-dir {names['site_to_gene']}",
                f"run_or_missing 'aggregation-controls {tier}' '{names['aggregation_controls']}' babappa validate-aggregation-controls --controls-dir {names['aggregation_controls']}",
                f"run_or_missing 'aggregation-threshold-policy {tier}' '{names['aggregation_policy']}' babappa validate-aggregation-threshold-policy --policy-dir {names['aggregation_policy']}",
                f"run_or_missing 'report {tier}' '{names['report']}' babappa validate-report --report-dir {names['report']}",
                f"run_or_missing 'run-summary {tier}' '{names['summary']}' babappa validate-run-summary --summary-dir {names['summary']}",
                "",
            ]
        )
    lines.extend(
        [
            "run_or_missing 'external-tier-summary' 'fast_external_10k_cross_tier_summary' babappa validate-external-tier-summary --summary-dir fast_external_10k_cross_tier_summary",
            "exit ${status}",
            "",
        ]
    )
    return "\n".join(lines)


def _fast10k_summarize_script(config: FastExternal10kPlanConfig) -> str:
    return "\n".join(
        [
            *_fast10k_header(config),
            "babappa summarize-external-tiers --tiers "
            + ",".join(config.tiers)
            + " --run-name fast_external_10k --outdir fast_external_10k_cross_tier_summary",
            "babappa validate-external-tier-summary --summary-dir fast_external_10k_cross_tier_summary",
            "",
        ]
    )


def _fast10k_expected(config: FastExternal10kPlanConfig) -> dict:
    n_tiers = len(config.tiers)
    n_methods = len(config.methods)
    total_families = config.families_per_tier * n_tiers
    expected_family_method_attempts = total_families * n_methods
    expected_raw_site_rows = expected_family_method_attempts * config.n_codons
    positive_family_fraction = 0.5
    selected_site_fraction = 0.05
    expected_positive_site_rows = int(
        expected_raw_site_rows * positive_family_fraction * selected_site_fraction
    )
    expected_site_dataset_rows = int(
        expected_positive_site_rows * (1 + config.negative_downsample_ratio)
    )
    return {
        "plan_version": EXTERNAL_ALIGNER_VALIDATION_PLAN_VERSION,
        "scale": 10000,
        "families_per_tier": config.families_per_tier,
        "tiers": list(config.tiers),
        "methods_requested": list(config.methods),
        "expected_family_method_attempts": expected_family_method_attempts,
        "expected_raw_site_rows_assuming_300_codons": expected_raw_site_rows,
        "expected_positive_fraction_assumption": {
            "positive_family_fraction": positive_family_fraction,
            "selected_site_fraction_in_positive_families": selected_site_fraction,
            "raw_site_positive_fraction": positive_family_fraction * selected_site_fraction,
        },
        "negative_downsample_ratio": config.negative_downsample_ratio,
        "expected_site_dataset_rows_under_downsample_ratio_10": expected_site_dataset_rows,
        "expected_output_directories": {
            "panel": config.panel_outdir,
            "cross_tier_summary": "fast_external_10k_cross_tier_summary",
            "by_tier": {
                tier: {
                    "sim": f"{config.panel_outdir}/tiers/{tier}/sim",
                    **_fast10k_names(tier),
                }
                for tier in config.tiers
            },
        },
        "estimated_runtime_warning": (
            "10K external alignment is a long manual execution job. BABAPPAlign and MUSCLE "
            "can dominate runtime; monitor logs and per-tier method policy."
        ),
        "estimated_disk_warning": (
            "Expect substantially more disk use than 1K because tensors, site datasets, "
            "predictions, and reports are produced for 40,000 family-method attempts before quarantine."
        ),
        "method_policy_note": "Downstream tensorization uses methods marked usable or caution in method_policy.tsv.",
        "diagnostic_exclusions": {
            "prank": "diagnostic only, excluded from default",
            "tcoffee": "optional diagnostic only, excluded from default",
        },
        "planner_executed_commands": [],
    }


def _fast10k_markdown(config: FastExternal10kPlanConfig, expected: dict) -> str:
    return "\n".join(
        [
            "# BABAPPA fast external 10K validation plan",
            "",
            "## Why 10K is next",
            "",
            "The completed 1K cross-tier summary shows strong site-neural performance and perfect oracle-simulation site-to-gene aggregation across low, moderate, high, and extreme tiers. A 10K fast external-aligner run is the next feasibility scale before considering 100K.",
            "",
            "## Why PRANK is excluded",
            "",
            "PRANK is diagnostic-only because prior external-tier runs showed slow runtime and high-tier codon-frame problems. It is not part of the production-fast default.",
            "",
            "## Why MUSCLE is quarantine-controlled",
            "",
            "MUSCLE is retained in the fast method set, but every tier runs method-policy quarantine before tensorization. If MUSCLE produces frame errors or exceeds failure thresholds, downstream tensors use only usable methods.",
            "",
            "## Why BABAPPAlign is retained",
            "",
            "BABAPPAlign remains part of the fast production ensemble because it provides an independent alignment backend and has acceptable low-failure behavior when timeout/failure handling and method quarantine are enabled.",
            "",
            "## Why 100K should wait",
            "",
            "100K should wait until this 10K fast external plan completes, validates, and produces cross-tier summaries without unresolved method-policy or calibration failures.",
            "",
            "## How to run",
            "",
            "```bash",
            f"bash {config.outdir}/run_fast_external_10k.sh",
            "```",
            "",
            "The run script uses `/tmp/babappa_fast_external_10k.lock`, logs to `logs/fast_external_10k_<timestamp>.log`, and refuses output collisions unless `BABAPPA_FAST_EXTERNAL_10K_ALLOW_EXISTING=1` is set deliberately.",
            "",
            "## How to monitor",
            "",
            "```bash",
            f"bash {config.outdir}/monitor_fast_external_10k.sh",
            "```",
            "",
            "## How to summarize outputs",
            "",
            "```bash",
            f"bash {config.outdir}/summarize_fast_external_10k.sh",
            "```",
            "",
            "## How to recover from a failed tier",
            "",
            "Inspect the tier log and `method_policy_fast_external_10k_<tier>/method_policy.tsv`. If alignment and site-map completed but a method was quarantined, resume downstream with only methods marked `usable` or `caution` rather than rerunning alignment.",
            "",
            "## Expected outputs",
            "",
            f"- Families per tier: {config.families_per_tier}",
            f"- Tiers: {', '.join(config.tiers)}",
            f"- Methods requested: {', '.join(config.methods)}",
            f"- Expected family-method attempts: {expected['expected_family_method_attempts']}",
            f"- Expected raw site rows assuming 300 codons: {expected['expected_raw_site_rows_assuming_300_codons']}",
            f"- Approximate site dataset rows under 10:1 downsampling: {expected['expected_site_dataset_rows_under_downsample_ratio_10']}",
            "",
        ]
    )


def _effective_methods(methods: List[str], optional_methods: List[str], exclude_methods: List[str]) -> List[str]:
    excluded = {str(method).strip() for method in exclude_methods if str(method).strip()}
    result: List[str] = []
    for method in [*methods, *optional_methods]:
        method = str(method).strip()
        if not method or method in excluded or method in result:
            continue
        result.append(method)
    return result


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
