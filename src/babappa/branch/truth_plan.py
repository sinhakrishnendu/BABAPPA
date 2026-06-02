"""Planner for explicit branch-site truth prototype validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from babappa.branch.feature_policy import get_branch_feature_policy


@dataclass(frozen=True)
class ExplicitBranchTruthPrototypePlanConfig:
    """Configuration for explicit branch-truth prototype planning."""

    outdir: str = "explicit_branch_truth_prototype_plan"
    n_families: int = 1000
    tiers: Union[str, Sequence[str]] = "low,moderate,high,extreme"
    methods: Union[str, Sequence[str]] = "identity,mafft,babappalign,muscle"

    def __post_init__(self) -> None:
        if self.n_families < 1:
            raise ValueError("n_families must be >= 1")
        tiers = _parse_csv(self.tiers)
        methods = _parse_csv(self.methods)
        if not tiers:
            raise ValueError("tiers must not be empty")
        if not methods:
            raise ValueError("methods must not be empty")
        object.__setattr__(self, "tiers", tiers)
        object.__setattr__(self, "methods", methods)
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ExplicitBranchTruth1kPlanConfig:
    """Configuration for explicit branch-truth 1K manual execution planning."""

    outdir: str = "explicit_branch_truth_1k_plan"
    n_families_per_tier: int = 250
    tiers: Union[str, Sequence[str]] = "low,moderate,high,extreme"
    methods: Union[str, Sequence[str]] = "identity,mafft,babappalign,muscle"
    negative_downsample_ratio: float = 5.0
    conda_sh: str = "/home/rajamosai/miniconda3/etc/profile.d/conda.sh"
    conda_env: str = "molevo"

    def __post_init__(self) -> None:
        if self.n_families_per_tier < 1:
            raise ValueError("n_families_per_tier must be >= 1")
        if self.negative_downsample_ratio <= 0:
            raise ValueError("negative_downsample_ratio must be > 0")
        tiers = _parse_csv(self.tiers)
        methods = _parse_csv(self.methods)
        if not tiers:
            raise ValueError("tiers must not be empty")
        if not methods:
            raise ValueError("methods must not be empty")
        object.__setattr__(self, "tiers", tiers)
        object.__setattr__(self, "methods", methods)
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ExplicitBranchTruth10kPlanConfig:
    """Configuration for conservative explicit branch-truth 10K manual execution planning."""

    outdir: str = "explicit_branch_truth_10k_plan"
    n_families_per_tier: int = 2500
    tiers: Union[str, Sequence[str]] = "low,moderate,high,extreme"
    methods: Union[str, Sequence[str]] = "identity,mafft,babappalign,muscle"
    feature_policy: str = "conservative_branch_site"
    negative_downsample_ratio: float = 5.0
    max_output_rows_per_tier: int = 1_000_000
    conda_sh: str = "/home/rajamosai/miniconda3/etc/profile.d/conda.sh"
    conda_env: str = "molevo"

    def __post_init__(self) -> None:
        if self.n_families_per_tier < 1:
            raise ValueError("n_families_per_tier must be >= 1")
        if self.negative_downsample_ratio <= 0:
            raise ValueError("negative_downsample_ratio must be > 0")
        if self.max_output_rows_per_tier < 1:
            raise ValueError("max_output_rows_per_tier must be >= 1")
        tiers = _parse_csv(self.tiers)
        methods = _parse_csv(self.methods)
        if not tiers:
            raise ValueError("tiers must not be empty")
        if not methods:
            raise ValueError("methods must not be empty")
        policy = get_branch_feature_policy(self.feature_policy)
        if self.feature_policy == "full_context":
            raise ValueError("full_context is upper-bound diagnostic only; use conservative_branch_site for 10K planning")
        object.__setattr__(self, "tiers", tiers)
        object.__setattr__(self, "methods", methods)
        object.__setattr__(self, "feature_policy", policy.name)
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ExplicitBranchTruth10kMacPlanConfig:
    """Configuration for Apple Silicon/MPS explicit branch-truth 10K planning."""

    outdir: str = "explicit_branch_truth_10k_mps_plan"
    n_families_per_tier: int = 2500
    tiers: Union[str, Sequence[str]] = "low,moderate,high,extreme"
    methods: Union[str, Sequence[str]] = "identity,mafft,babappalign,muscle"
    feature_policy: str = "conservative_branch_site"
    truth_mode: str = "explicit"
    negative_downsample_ratio: float = 5.0
    max_output_rows_per_tier: int = 1_000_000
    device: str = "mps"
    batch_size: int = 128
    threads: int = 8
    conda_env: str = "molevo"
    mps_fallback: bool = True
    mps_high_watermark_ratio: Optional[float] = None
    allow_missing_babappalign: bool = False

    def __post_init__(self) -> None:
        _validate_mac_plan_config(self)


@dataclass(frozen=True)
class ExplicitBranchTruth100kMacPlanConfig:
    """Configuration for Apple Silicon/MPS explicit branch-truth 100K planning."""

    outdir: str = "explicit_branch_truth_100k_mps_plan"
    n_families_per_tier: int = 25000
    tiers: Union[str, Sequence[str]] = "low,moderate,high,extreme"
    methods: Union[str, Sequence[str]] = "identity,mafft,babappalign,muscle"
    feature_policy: str = "conservative_branch_site"
    truth_mode: str = "explicit"
    negative_downsample_ratio: float = 5.0
    max_output_rows_per_tier: int = 2_000_000
    device: str = "mps"
    batch_size: int = 64
    threads: int = 8
    conda_env: str = "molevo"
    mps_fallback: bool = True
    mps_high_watermark_ratio: Optional[float] = None
    allow_missing_babappalign: bool = False

    def __post_init__(self) -> None:
        _validate_mac_plan_config(self)


def plan_explicit_branch_truth_prototype(config: ExplicitBranchTruthPrototypePlanConfig) -> Dict[str, object]:
    """Write future-facing scripts for an explicit branch-site truth prototype."""

    outdir = Path(config.outdir)
    run_path = outdir / "run_explicit_branch_truth_prototype.sh"
    monitor_path = outdir / "monitor_explicit_branch_truth_prototype.sh"
    validate_path = outdir / "validate_explicit_branch_truth_prototype.sh"
    expected_path = outdir / "expected_outputs.json"
    markdown_path = outdir / "explicit_branch_truth_prototype_plan.md"
    expected = _expected_outputs(config)

    run_path.write_text(_run_script(config), encoding="utf-8")
    monitor_path.write_text(_monitor_script(config), encoding="utf-8")
    validate_path.write_text(_validate_script(config), encoding="utf-8")
    expected_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(config, expected), encoding="utf-8")
    for script in [run_path, monitor_path, validate_path]:
        os.chmod(script, 0o755)
    return {
        "status": "ok",
        "outdir": str(outdir),
        "run": str(run_path),
        "monitor": str(monitor_path),
        "validate": str(validate_path),
        "expected_outputs": str(expected_path),
        "markdown": str(markdown_path),
        "n_families": config.n_families,
        "tiers": config.tiers,
        "methods": config.methods,
        "does_not_run_jobs": True,
    }


def plan_explicit_branch_truth_1k(config: ExplicitBranchTruth1kPlanConfig) -> Dict[str, object]:
    """Write scripts for the explicit branch-truth 1K prototype without executing them."""

    outdir = Path(config.outdir)
    run_path = outdir / "run_explicit_branch_truth_1k.sh"
    monitor_path = outdir / "monitor_explicit_branch_truth_1k.sh"
    validate_path = outdir / "validate_explicit_branch_truth_1k.sh"
    summarize_path = outdir / "summarize_explicit_branch_truth_1k.sh"
    expected_path = outdir / "expected_outputs.json"
    markdown_path = outdir / "explicit_branch_truth_1k_plan.md"
    expected = _expected_1k_outputs(config)

    run_path.write_text(_run_1k_script(config), encoding="utf-8")
    monitor_path.write_text(_monitor_1k_script(config), encoding="utf-8")
    validate_path.write_text(_validate_1k_script(config), encoding="utf-8")
    summarize_path.write_text(_summarize_1k_script(config), encoding="utf-8")
    expected_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_1k(config), encoding="utf-8")
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
        "does_not_run_jobs": True,
        "tiers": config.tiers,
        "methods": config.methods,
    }


def plan_explicit_branch_truth_10k(config: ExplicitBranchTruth10kPlanConfig) -> Dict[str, object]:
    """Write scripts for conservative explicit branch-truth 10K validation without executing them."""

    outdir = Path(config.outdir)
    run_path = outdir / "run_explicit_branch_truth_10k.sh"
    monitor_path = outdir / "monitor_explicit_branch_truth_10k.sh"
    validate_path = outdir / "validate_explicit_branch_truth_10k.sh"
    summarize_path = outdir / "summarize_explicit_branch_truth_10k.sh"
    expected_path = outdir / "expected_outputs.json"
    markdown_path = outdir / "explicit_branch_truth_10k_plan.md"
    expected = _expected_10k_outputs(config)

    run_path.write_text(_run_10k_script(config), encoding="utf-8")
    monitor_path.write_text(_monitor_10k_script(config), encoding="utf-8")
    validate_path.write_text(_validate_10k_script(config), encoding="utf-8")
    summarize_path.write_text(_summarize_10k_script(config), encoding="utf-8")
    expected_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_10k(config), encoding="utf-8")
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
        "does_not_run_jobs": True,
        "tiers": config.tiers,
        "methods": config.methods,
        "feature_policy": config.feature_policy,
        "truth_mode": "explicit",
    }


def plan_explicit_branch_truth_10k_mac(config: ExplicitBranchTruth10kMacPlanConfig) -> Dict[str, object]:
    """Write Apple Silicon/MPS scripts for conservative explicit branch-truth 10K."""

    return _write_mac_plan(config, scale="10k", block_until_env=False)


def plan_explicit_branch_truth_100k_mac(config: ExplicitBranchTruth100kMacPlanConfig) -> Dict[str, object]:
    """Write Apple Silicon/MPS scripts for explicit branch-truth 100K, gated behind 10K."""

    return _write_mac_plan(config, scale="100k", block_until_env=True)


def _write_mac_plan(config, scale: str, block_until_env: bool) -> Dict[str, object]:
    outdir = Path(config.outdir)
    stem = f"explicit_branch_truth_{scale}_mps"
    run_path = outdir / f"run_{stem}.sh"
    monitor_path = outdir / f"monitor_{stem}.sh"
    validate_path = outdir / f"validate_{stem}.sh"
    summarize_path = outdir / f"summarize_{stem}.sh"
    expected_path = outdir / "expected_outputs.json"
    markdown_path = outdir / f"{stem}_plan.md"
    expected = _expected_mac_outputs(config, scale)

    run_path.write_text(_mac_run_script(config, scale, block_until_env), encoding="utf-8")
    monitor_path.write_text(_mac_monitor_script(config, scale), encoding="utf-8")
    validate_path.write_text(_mac_validate_script(config, scale), encoding="utf-8")
    summarize_path.write_text(_mac_summarize_script(config, scale), encoding="utf-8")
    expected_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_mac_markdown(config, scale), encoding="utf-8")
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
        "does_not_run_jobs": True,
        "tiers": config.tiers,
        "methods": config.methods,
        "feature_policy": config.feature_policy,
        "truth_mode": config.truth_mode,
        "device": config.device,
        "batch_size": config.batch_size,
        "allow_missing_babappalign": config.allow_missing_babappalign,
        "blocked_until_10k_passes": scale == "100k",
    }


def _validate_mac_plan_config(config) -> None:
    if config.n_families_per_tier < 1:
        raise ValueError("n_families_per_tier must be >= 1")
    if config.negative_downsample_ratio <= 0:
        raise ValueError("negative_downsample_ratio must be > 0")
    if config.max_output_rows_per_tier < 1:
        raise ValueError("max_output_rows_per_tier must be >= 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.threads < 1:
        raise ValueError("threads must be >= 1")
    if config.truth_mode != "explicit":
        raise ValueError("truth_mode must be explicit for Apple Silicon explicit branch-truth plans")
    if config.device not in {"auto", "cpu", "mps"}:
        raise ValueError("Mac planners accept device auto, cpu, or mps")
    if config.mps_high_watermark_ratio is not None and config.mps_high_watermark_ratio < 0:
        raise ValueError("mps_high_watermark_ratio must be >= 0 when supplied")
    tiers = _parse_csv(config.tiers)
    methods = _parse_csv(config.methods)
    if not tiers:
        raise ValueError("tiers must not be empty")
    if not methods:
        raise ValueError("methods must not be empty")
    policy = get_branch_feature_policy(config.feature_policy)
    if policy.name != "conservative_branch_site":
        raise ValueError("Mac explicit branch-truth plans require conservative_branch_site")
    object.__setattr__(config, "tiers", tiers)
    object.__setattr__(config, "methods", methods)
    object.__setattr__(config, "feature_policy", policy.name)
    Path(config.outdir).mkdir(parents=True, exist_ok=True)


def _mac_run_script(config, scale: str, block_until_env: bool) -> str:
    lines = _mac_header(config)
    lines.extend(_mac_preflight_lines(config, scale))
    lines.extend(_mac_lock_lines(scale))
    if block_until_env:
        lines.extend(
            [
                "if [ \"${BABAPPA_ALLOW_100K_AFTER_10K:-0}\" != \"1\" ]; then",
                "  echo 'DO NOT RUN 100K until the conservative 10K MPS plan completes and validates.' >&2",
                "  echo 'After 10K passes, rerun with BABAPPA_ALLOW_100K_AFTER_10K=1.' >&2",
                "  exit 2",
                "fi",
                "",
            ]
        )
    lines.extend(
        [
            "mkdir -p logs",
            "ts=$(date +%Y%m%d_%H%M%S)",
            f"log=logs/explicit_branch_truth_{scale}_mps_${{ts}}.log",
            "exec > >(tee -a \"$log\") 2>&1",
            f"echo 'BABAPPA explicit branch-truth {scale.upper()} Apple Silicon/MPS plan started.'",
            "echo 'MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING'",
                f"echo 'Device: {config.device}; batch size: '\"$BABAPPA_MPS_BATCH_SIZE\"'; workers: '\"$BABAPPA_PERF_WORKERS\"'; torch threads: '\"$BABAPPA_TORCH_THREADS\"'; babappalign device: '\"$BABAPPA_BABAPPALIGN_DEVICE\"'; babappalign backend: '\"$BABAPPA_BABAPPALIGN_BACKEND\"'; babappalign workers: '\"$BABAPPA_BABAPPALIGN_WORKERS\"'; babappalign max workers: '\"$BABAPPA_BABAPPALIGN_MAX_WORKERS\"'; aligner child threads: '\"$BABAPPA_ALIGNER_SUBPROCESS_THREADS\"'; min free GB: '\"$BABAPPA_MIN_FREE_GB\"'; feature policy: {config.feature_policy}; truth mode: {config.truth_mode}'",
            "",
        ]
    )
    lines.extend(_mac_babappalign_preflight(config))
    lines.extend(_mac_stage_helpers())
    methods = ",".join(config.methods)
    max_train = min(config.max_output_rows_per_tier, 200_000)
    max_eval = min(config.max_output_rows_per_tier, 50_000)
    marker_dir = f"{config.outdir}/stage_markers"

    def marker(tier: str, stage: str) -> str:
        return f"{marker_dir}/.stage_complete_{tier}_{stage}"

    for tier in config.tiers:
        names = _mac_tier_names(scale, tier, config.feature_policy)
        lines.extend(
            [
                f"echo '=== explicit branch-truth {scale.upper()} MPS tier: {tier} ==='",
                f"run_stage_dir {marker(tier, 'simulate')} {names['sim']} babappa simulate --outdir {names['sim']} --n-families {config.n_families_per_tier} --n-taxa 8 --n-codons 300 --seed 42 --positive-rate 0.5 --saturation-tier {tier} --workers \"$BABAPPA_PERF_WORKERS\"",
                f"run_stage {marker(tier, 'validate_sim')} babappa validate-sim --sim-dir {names['sim']} --require-branch-truth",
                f"run_stage_dir {marker(tier, 'audit_sim')} {names['sim']}/audit babappa audit-sim --sim-dir {names['sim']} --outdir {names['sim']}/audit",
                f"run_stage_dir {marker(tier, 'align')} {names['align']} babappa align-external --sim-dir {names['sim']} --outdir {names['align']} --methods {methods} --threads \"$BABAPPA_PERF_WORKERS\" --aligner-subprocess-threads \"$BABAPPA_ALIGNER_SUBPROCESS_THREADS\" --babappalign-device \"$BABAPPA_BABAPPALIGN_DEVICE\" --babappalign-backend \"$BABAPPA_BABAPPALIGN_BACKEND\" --babappalign-workers \"$BABAPPA_BABAPPALIGN_WORKERS\"{_mac_allow_missing_babappalign_arg(config)}",
                f"run_stage_dir {marker(tier, 'site_map')} {names['site_map']} babappa build-site-map --sim-dir {names['sim']} --align-dir {names['align']} --outdir {names['site_map']} --methods {methods} --workers \"$BABAPPA_PERF_WORKERS\"",
                f"run_stage_dir {marker(tier, 'method_policy')} {names['method_policy']} babappa aligner-method-policy --align-dir {names['align']} --site-map-dir {names['site_map']} --outdir {names['method_policy']} --max-conflict-fraction 0.03 --max-frame-error-fraction 0.0 --max-method-failure-fraction 0.01",
                f"run_stage {marker(tier, 'validate_method_policy')} babappa validate-aligner-method-policy --policy-dir {names['method_policy']}",
                f"run_stage_dir {marker(tier, 'tensors')} {names['tensors']} babappa build-tensors --sim-dir {names['sim']} --align-dir {names['align']} --outdir {names['tensors']} --methods {methods} --workers \"$BABAPPA_PERF_WORKERS\"",
                f"run_stage {marker(tier, 'validate_tensors')} babappa validate-tensors --tensor-dir {names['tensors']}",
                f"run_stage_dir {marker(tier, 'index')} {names['dataset']} babappa index-dataset --tensor-dir {names['tensors']} --outdir {names['dataset']} --methods {methods} --workers \"$BABAPPA_PERF_WORKERS\"",
                f"run_stage_dir {marker(tier, 'labels')} {names['labels']} babappa extract-branch-site-labels --dataset-dir {names['dataset']} --site-map-dir {names['site_map']} --outdir {names['labels']} --truth-mode {config.truth_mode} --aligned-site-mode mapped --foreground-source truth --streaming-output",
                f"run_stage {marker(tier, 'validate_labels')} babappa validate-branch-site-labels --label-dir {names['labels']}",
                f"run_stage_dir {marker(tier, 'branch_dataset')} {names['branch_dataset']} babappa build-branch-site-dataset --dataset-dir {names['dataset']} --branch-site-labels {names['labels']}/branch_site_oracle_labels.tsv --outdir {names['branch_dataset']} --negative-downsample-ratio {config.negative_downsample_ratio:g} --seed 42 --require-mappable-sites --streaming --max-output-rows {config.max_output_rows_per_tier}",
                f"run_stage {marker(tier, 'validate_branch_dataset')} babappa validate-branch-site-dataset --branch-site-dataset-dir {names['branch_dataset']}",
                f"run_stage_dir {marker(tier, 'leakage')} {names['leakage']} babappa audit-branch-site-leakage --branch-site-dataset-dir {names['branch_dataset']} --outdir {names['leakage']}",
                f"run_stage_dir {marker(tier, 'branch_neural')} {names['branch_neural']} babappa train-branch-site-neural --branch-site-dataset-dir {names['branch_dataset']} --outdir {names['branch_neural']} --device {config.device} --batch-size \"$BABAPPA_MPS_BATCH_SIZE\" --threads \"$BABAPPA_TORCH_THREADS\" --feature-policy {config.feature_policy} --epochs 10 --learning-rate 0.001 --weight-decay 0.0001 --hidden-dim 64 --dropout 0.1 --positive-class-weight auto --monitor-metric val_auroc --max-train-items {max_train} --max-val-items {max_eval} --max-calib-items {max_eval} --max-test-items {max_eval}",
                f"run_stage {marker(tier, 'validate_branch_neural')} babappa validate-branch-site-neural --model-dir {names['branch_neural']}",
                f"run_stage_dir {marker(tier, 'calibration')} {names['calibration']} babappa calibrate-branch-site-neural --model-dir {names['branch_neural']} --outdir {names['calibration']}",
                f"run_stage_dir {marker(tier, 'aggregation')} {names['aggregation']} babappa aggregate-branch-sites --predictions {names['branch_neural']}/branch_site_neural_predictions.tsv --outdir {names['aggregation']}",
                f"run_stage {marker(tier, 'validate_aggregation')} babappa validate-branch-aggregation --aggregation-dir {names['aggregation']}",
                f"run_stage_dir {marker(tier, 'controls')} {names['controls']} babappa branch-aggregation-controls --predictions {names['branch_neural']}/branch_site_neural_predictions.tsv --outdir {names['controls']} --n-permutations 100 --seed 42 --workers \"$BABAPPA_PERF_WORKERS\"",
                f"run_stage {marker(tier, 'validate_controls')} babappa validate-branch-aggregation-controls --controls-dir {names['controls']}",
                f"run_stage_dir {marker(tier, 'threshold')} {names['threshold']} babappa branch-site-threshold-policy --predictions {names['calibration']}/branch_site_calibrated_predictions.tsv --outdir {names['threshold']} --probability-column prob_positive_raw --calibrated-probability-column prob_positive_calibrated",
                f"run_stage_dir {marker(tier, 'aggregation_policy')} {names['aggregation_policy']} babappa branch-aggregation-threshold-policy --aggregation-dir {names['aggregation']} --outdir {names['aggregation_policy']}",
                f"run_stage_dir {marker(tier, 'summary')} {names['summary']} babappa summarize-branch-site-run --outdir {names['summary']} --title 'BABAPPA explicit branch-truth {scale.upper()} MPS {tier} summary' --branch-site-label-dir {names['labels']} --branch-site-dataset-dir {names['branch_dataset']} --branch-site-leakage-dir {names['leakage']} --branch-site-neural-dir {names['branch_neural']} --branch-site-calibration-dir {names['calibration']} --branch-aggregation-dir {names['aggregation']} --branch-aggregation-controls-dir {names['controls']} --branch-site-threshold-policy-dir {names['threshold']} --branch-aggregation-threshold-policy-dir {names['aggregation_policy']}",
                f"run_stage_dir {marker(tier, 'truth_audit')} {names['truth_audit']} babappa audit-branch-truth-status --tiers {tier} --run-name explicit_branch_truth_{scale}_mps --output-suffix _streamed --outdir {names['truth_audit']}",
                "",
            ]
        )
    if scale == "100k":
        lines.append("echo '100K finished only if every tier and stage marker completed. Validate carefully before interpreting.'")
    else:
        lines.append("echo '10K MPS plan completed. Validate and summarize before considering 100K.'")
    return "\n".join(lines) + "\n"


def _mac_monitor_script(config, scale: str) -> str:
    lines = _mac_header(config)
    key_dirs = []
    for tier in config.tiers:
        names = _mac_tier_names(scale, tier, config.feature_policy)
        key_dirs.extend([names["sim"], names["align"], names["branch_dataset"], names["branch_neural"], names["aggregation"]])
    lines.extend(
        [
            "echo 'Active BABAPPA / Python / aligner processes:'",
            "ps aux | egrep 'babappa|python|mafft|muscle|babappalign' | grep -v egrep || true",
            "",
            "echo 'macOS memory pressure:'",
            "vm_stat || true",
            "if command -v memory_pressure >/dev/null 2>&1; then memory_pressure || true; else echo 'memory_pressure unavailable'; fi",
            "top -l 1 -o mem -n 20 || true",
            "",
            "echo 'Disk usage:'",
            "df -h . || true",
            "du -sh " + " ".join(key_dirs) + " 2>/dev/null || true",
            "",
            f"echo 'Latest explicit branch-truth {scale.upper()} MPS log:'",
            f"ls -t logs/explicit_branch_truth_{scale}_mps_*.log 2>/dev/null | head -1 | xargs tail -80 2>/dev/null || true",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _mac_validate_script(config, scale: str) -> str:
    lines = _mac_header(config)
    lines.extend(
        [
            "validate_branch_site_labels_or_pruned() {",
            "  local label_dir=\"$1\"",
            "  local marker=\"$2\"",
            "  if [ -s \"$label_dir/branch_site_oracle_labels.tsv\" ]; then",
            "    babappa validate-branch-site-labels --label-dir \"$label_dir\"",
            "    return 0",
            "  fi",
            "  if [ -f \"$marker\" ] && [ -s \"$label_dir/branch_site_oracle_summary.json\" ]; then",
            "    echo \"Raw label TSV was pruned after downstream validation; retaining compact summary for $label_dir\"",
            "    return 0",
            "  fi",
            "  babappa validate-branch-site-labels --label-dir \"$label_dir\"",
            "}",
            "",
        ]
    )
    marker_dir = f"{config.outdir}/stage_markers"
    for tier in config.tiers:
        names = _mac_tier_names(scale, tier, config.feature_policy)
        lines.extend(
            [
                f"echo '=== validate explicit branch-truth {scale.upper()} MPS tier {tier} ==='",
                f"babappa validate-sim --sim-dir {names['sim']} --require-branch-truth",
                f"babappa validate-align --align-dir {names['align']}",
                f"babappa validate-site-map --site-map-dir {names['site_map']}",
                f"babappa validate-aligner-method-policy --policy-dir {names['method_policy']}",
                f"babappa validate-tensors --tensor-dir {names['tensors']}",
                f"babappa validate-index --index-dir {names['dataset']}",
                f"validate_branch_site_labels_or_pruned {names['labels']} {marker_dir}/.stage_complete_{tier}_labels",
                f"babappa validate-branch-site-dataset --branch-site-dataset-dir {names['branch_dataset']}",
                f"babappa validate-branch-site-leakage --leakage-dir {names['leakage']}",
                f"babappa validate-branch-site-neural --model-dir {names['branch_neural']}",
                f"babappa validate-branch-site-calibration --calibration-dir {names['calibration']}",
                f"babappa validate-branch-aggregation --aggregation-dir {names['aggregation']}",
                f"babappa validate-branch-aggregation-controls --controls-dir {names['controls']}",
                f"babappa validate-branch-site-threshold-policy --policy-dir {names['threshold']}",
                f"babappa validate-branch-aggregation-threshold-policy --policy-dir {names['aggregation_policy']}",
                f"babappa validate-branch-site-run-summary --summary-dir {names['summary']}",
                f"babappa validate-branch-truth-status-audit --audit-dir {names['truth_audit']}",
                "",
            ]
        )
    if scale == "100k":
        lines.append("echo 'DO NOT interpret 100K unless 10K MPS completed and validated first.'")
    return "\n".join(lines) + "\n"


def _mac_summarize_script(config, scale: str) -> str:
    lines = _mac_header(config)
    tier_csv = ",".join(config.tiers)
    lines.extend(
        [
            f"babappa summarize-branch-conditioned-tiers --tiers {tier_csv} --run-name explicit_branch_truth_{scale}_mps --output-suffix _streamed --outdir explicit_branch_truth_{scale}_mps_cross_tier_summary",
            f"babappa validate-branch-conditioned-tier-summary --summary-dir explicit_branch_truth_{scale}_mps_cross_tier_summary",
            f"babappa audit-branch-truth-status --tiers {tier_csv} --run-name explicit_branch_truth_{scale}_mps --output-suffix _streamed --outdir explicit_branch_truth_{scale}_mps_truth_status_audit",
            f"babappa validate-branch-truth-status-audit --audit-dir explicit_branch_truth_{scale}_mps_truth_status_audit",
        ]
    )
    if scale == "100k":
        lines.append("echo '100K summary is blocked scientifically unless 10K MPS passed first.'")
    return "\n".join(lines) + "\n"


def _mac_header(config, activate: bool = True) -> List[str]:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING",
        "",
    ]
    if activate:
        lines.extend(
            [
                "set +u",
                "if [ -f \"$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then",
                "  source \"$HOME/miniconda3/etc/profile.d/conda.sh\"",
                "elif [ -f \"$HOME/miniforge3/etc/profile.d/conda.sh\" ]; then",
                "  source \"$HOME/miniforge3/etc/profile.d/conda.sh\"",
                "elif [ -f \"$HOME/anaconda3/etc/profile.d/conda.sh\" ]; then",
                "  source \"$HOME/anaconda3/etc/profile.d/conda.sh\"",
                "else",
                "  echo 'Could not find conda.sh. Install Miniforge/Miniconda or source conda before running.' >&2",
                "  exit 1",
                "fi",
                f"conda activate {config.conda_env}",
                "set -u",
                f"export BABAPPA_PERF_WORKERS=\"${{BABAPPA_PERF_WORKERS:-$(sysctl -n hw.ncpu 2>/dev/null || sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo {config.threads})}}\"",
                f"export BABAPPA_MPS_BATCH_SIZE=\"${{BABAPPA_MPS_BATCH_SIZE:-{config.batch_size}}}\"",
                "export BABAPPA_TORCH_THREADS=\"${BABAPPA_TORCH_THREADS:-$BABAPPA_PERF_WORKERS}\"",
                "export BABAPPA_ALIGNER_SUBPROCESS_THREADS=\"${BABAPPA_ALIGNER_SUBPROCESS_THREADS:-1}\"",
                "export BABAPPA_BABAPPALIGN_DEVICE=\"${BABAPPA_BABAPPALIGN_DEVICE:-mps}\"",
                "export BABAPPA_BABAPPALIGN_BACKEND=\"${BABAPPA_BABAPPALIGN_BACKEND:-embedded}\"",
                "_babappa_mem_bytes=\"$(sysctl -n hw.memsize 2>/dev/null || true)\"",
                "if [ -z \"$_babappa_mem_bytes\" ]; then _babappa_mem_bytes=38654705664; fi",
                "_babappa_mem_gb=$(( _babappa_mem_bytes / 1024 / 1024 / 1024 ))",
                "_babappa_babappalign_default_workers=$(( _babappa_mem_gb / 9 ))",
                "if [ \"$_babappa_babappalign_default_workers\" -lt 2 ]; then _babappa_babappalign_default_workers=2; fi",
                "if [ \"$_babappa_babappalign_default_workers\" -gt 4 ]; then _babappa_babappalign_default_workers=4; fi",
                "if [ \"$_babappa_babappalign_default_workers\" -gt \"$BABAPPA_PERF_WORKERS\" ]; then _babappa_babappalign_default_workers=\"$BABAPPA_PERF_WORKERS\"; fi",
                "export BABAPPA_BABAPPALIGN_WORKERS=\"${BABAPPA_BABAPPALIGN_WORKERS:-$_babappa_babappalign_default_workers}\"",
                "export BABAPPA_BABAPPALIGN_MAX_WORKERS=\"${BABAPPA_BABAPPALIGN_MAX_WORKERS:-$BABAPPA_BABAPPALIGN_WORKERS}\"",
                "export BABAPPA_MIN_FREE_GB=\"${BABAPPA_MIN_FREE_GB:-8}\"",
                f"export PYTORCH_ENABLE_MPS_FALLBACK={'1' if config.mps_fallback else '0'}",
                "export OMP_NUM_THREADS=\"${OMP_NUM_THREADS:-$BABAPPA_PERF_WORKERS}\"",
                "export MKL_NUM_THREADS=\"${MKL_NUM_THREADS:-$BABAPPA_PERF_WORKERS}\"",
                "export OPENBLAS_NUM_THREADS=\"${OPENBLAS_NUM_THREADS:-$BABAPPA_PERF_WORKERS}\"",
                "export NUMEXPR_NUM_THREADS=\"${NUMEXPR_NUM_THREADS:-$BABAPPA_PERF_WORKERS}\"",
                "export VECLIB_MAXIMUM_THREADS=\"${VECLIB_MAXIMUM_THREADS:-$BABAPPA_PERF_WORKERS}\"",
            ]
        )
        if config.mps_high_watermark_ratio is not None:
            lines.append(f"export PYTORCH_MPS_HIGH_WATERMARK_RATIO={config.mps_high_watermark_ratio:g}")
        lines.append("")
    return lines


def _mac_preflight_lines(config, scale: str) -> List[str]:
    require_babappalign = "true" if "babappalign" in config.methods and not config.allow_missing_babappalign else "false"
    require_mps = "true" if config.device in {"mps", "auto"} else "false"
    return [
        "babappa preflight-explicit-branch-truth-mps-plan \\",
        f"  --plan-dir {config.outdir} \\",
        f"  --scale {scale} \\",
        f"  --require-babappalign {require_babappalign} \\",
        f"  --require-mps {require_mps} \\",
        f"  --conda-env {config.conda_env}",
        "",
    ]


def _mac_lock_lines(scale: str) -> List[str]:
    lock = f"/tmp/babappa_explicit_branch_truth_{scale}_mps.lock"
    return [
        f"lock_dir={lock}",
        "cleanup_lock() {",
        "  if [ -d \"$lock_dir\" ] && [ \"$(cat \"$lock_dir/pid\" 2>/dev/null || true)\" = \"$$\" ]; then",
        "    rm -rf \"$lock_dir\"",
        "  fi",
        "}",
        "if mkdir \"$lock_dir\" 2>/dev/null; then",
        "  echo \"$$\" > \"$lock_dir/pid\"",
        "  trap cleanup_lock EXIT",
        "else",
        "  echo \"Another BABAPPA MPS run may already be using $lock_dir\" >&2",
        "  if [ -f \"$lock_dir/pid\" ]; then echo \"Recorded PID: $(cat \"$lock_dir/pid\")\" >&2; fi",
        "  echo 'If no BABAPPA run is active, remove the stale lock safely with:' >&2",
        "  echo \"rm -rf \\\"$lock_dir\\\"\" >&2",
        "  exit 1",
        "fi",
        "",
    ]


def _mac_allow_missing_babappalign_arg(config) -> str:
    return " --allow-missing-babappalign" if config.allow_missing_babappalign else ""


def _mac_babappalign_preflight(config) -> List[str]:
    if "babappalign" not in config.methods or config.allow_missing_babappalign:
        return []
    return [
        "babappalign_model=\"$HOME/.cache/babappalign/models/babappascore.pt\"",
        "if [ ! -s \"$babappalign_model\" ]; then",
        "  echo 'babappalign_model_missing: required BABAPPAScore model is missing.' >&2",
        "  echo \"Expected file: $babappalign_model\" >&2",
        "  echo 'Install with:' >&2",
        "  echo 'mkdir -p \"$HOME/.cache/babappalign/models\"' >&2",
        "  echo 'curl -L \"https://zenodo.org/record/18053201/files/babappascore.pt\" -o \"$HOME/.cache/babappalign/models/babappascore.pt\"' >&2",
        "  exit 1",
        "fi",
        "",
    ]


def _mac_stage_helpers() -> List[str]:
    return [
        "validate_existing_output() {",
        "  local marker=\"$1\"",
        "  local outdir=\"$2\"",
        "  case \"$marker\" in",
        "    *_simulate) echo \"Validation command: babappa validate-sim --sim-dir $outdir --require-branch-truth\"; babappa validate-sim --sim-dir \"$outdir\" --require-branch-truth ;;",
        "    *_align) echo \"Validation command: babappa validate-align --align-dir $outdir\"; babappa validate-align --align-dir \"$outdir\" ;;",
        "    *_site_map) echo \"Validation command: babappa validate-site-map --site-map-dir $outdir\"; babappa validate-site-map --site-map-dir \"$outdir\" ;;",
        "    *_method_policy) echo \"Validation command: babappa validate-aligner-method-policy --policy-dir $outdir\"; babappa validate-aligner-method-policy --policy-dir \"$outdir\" ;;",
        "    *_branch_dataset) echo \"Validation command: babappa validate-branch-site-dataset --branch-site-dataset-dir $outdir\"; babappa validate-branch-site-dataset --branch-site-dataset-dir \"$outdir\" ;;",
        "    *_index) echo \"Validation command: babappa validate-index --index-dir $outdir\"; babappa validate-index --index-dir \"$outdir\" ;;",
        "    *_labels) echo \"Validation command: babappa validate-branch-site-labels --label-dir $outdir\"; babappa validate-branch-site-labels --label-dir \"$outdir\" ;;",
        "    *_leakage) echo \"Validation command: babappa validate-branch-site-leakage --leakage-dir $outdir\"; babappa validate-branch-site-leakage --leakage-dir \"$outdir\" ;;",
        "    *_branch_neural) echo \"Validation command: babappa validate-branch-site-neural --model-dir $outdir\"; babappa validate-branch-site-neural --model-dir \"$outdir\" ;;",
        "    *_calibration) echo \"Validation command: babappa validate-branch-site-calibration --calibration-dir $outdir\"; babappa validate-branch-site-calibration --calibration-dir \"$outdir\" ;;",
        "    *_aggregation_policy) echo \"Validation command: babappa validate-branch-aggregation-threshold-policy --policy-dir $outdir\"; babappa validate-branch-aggregation-threshold-policy --policy-dir \"$outdir\" ;;",
        "    *_aggregation) echo \"Validation command: babappa validate-branch-aggregation --aggregation-dir $outdir\"; babappa validate-branch-aggregation --aggregation-dir \"$outdir\" ;;",
        "    *_controls) echo \"Validation command: babappa validate-branch-aggregation-controls --controls-dir $outdir\"; babappa validate-branch-aggregation-controls --controls-dir \"$outdir\" ;;",
        "    *_threshold) echo \"Validation command: babappa validate-branch-site-threshold-policy --policy-dir $outdir\"; babappa validate-branch-site-threshold-policy --policy-dir \"$outdir\" ;;",
        "    *_summary) echo \"Validation command: babappa validate-branch-site-run-summary --summary-dir $outdir\"; babappa validate-branch-site-run-summary --summary-dir \"$outdir\" ;;",
        "    *_truth_audit) echo \"Validation command: babappa validate-branch-truth-status-audit --audit-dir $outdir\"; babappa validate-branch-truth-status-audit --audit-dir \"$outdir\" ;;",
        "    *_tensors) echo \"Validation command: babappa validate-tensors --tensor-dir $outdir\"; babappa validate-tensors --tensor-dir \"$outdir\" ;;",
        "    *) echo \"No validator is registered for existing output $outdir from $marker\" >&2; return 2 ;;",
        "  esac",
        "}",
        "",
        "babappa_free_memory_gb() {",
        "  if ! command -v vm_stat >/dev/null 2>&1; then echo 999; return 0; fi",
        "  vm_stat | awk '",
        "    function lastnum(    i, t) { for (i = NF; i >= 1; i--) { t = $i; gsub(/[^0-9]/, \"\", t); if (t != \"\") return t + 0 } return 0 }",
        "    /page size of/ { page_size = lastnum() }",
        "    /Pages free/ || /Pages speculative/ || /Pages inactive/ || /Pages purgeable/ { pages += lastnum() }",
        "    END { if (page_size > 0) printf \"%d\\n\", pages * page_size / 1073741824; else print 999 }",
        "  '",
        "}",
        "",
        "memory_guard() {",
        "  local stage=\"$1\"",
        "  local min_gb=\"${BABAPPA_MIN_FREE_GB:-8}\"",
        "  local tries=0",
        "  local free_gb",
        "  while [ \"$tries\" -lt 40 ]; do",
        "    free_gb=\"$(babappa_free_memory_gb)\"",
        "    if [ \"${free_gb:-999}\" -ge \"$min_gb\" ]; then return 0; fi",
        "    echo \"Memory guard: waiting before $stage; estimated available memory ${free_gb}GB is below ${min_gb}GB.\"",
        "    sleep 30",
        "    tries=$((tries + 1))",
        "  done",
        "  echo \"Memory guard: refusing to start $stage because estimated available memory stayed below ${min_gb}GB.\" >&2",
        "  exit 1",
        "}",
        "",
        "run_stage() {",
        "  local marker=\"$1\"",
        "  shift",
        "  mkdir -p \"$(dirname \"$marker\")\"",
        "  if [ -f \"$marker\" ] && [ \"${BABAPPA_FORCE:-0}\" != \"1\" ]; then",
        "    echo \"Skipping completed stage: $marker\"",
        "    return 0",
        "  fi",
        "  if [ -f \"${marker}.partial\" ] && [ \"${BABAPPA_FORCE:-0}\" != \"1\" ]; then",
        "    echo \"Refusing to resume partial stage without BABAPPA_FORCE=1: $marker\" >&2",
        "    exit 1",
        "  fi",
        "  memory_guard \"$marker\"",
        "  touch \"${marker}.partial\"",
        "  echo \"Running stage: $marker\"",
        "  echo \"Command: $*\"",
        "  if ! \"$@\"; then",
        "    echo \"Stage failed: $marker\" >&2",
        "    echo \"Command failed: $*\" >&2",
        "    echo 'Resume recommendation: fix the reported issue, leave validated completed outputs in place, then rerun this script. Completed markers will be skipped.' >&2",
        "    exit 1",
        "  fi",
        "  rm -f \"${marker}.partial\"",
        "  touch \"$marker\"",
        "}",
        "",
        "run_stage_dir() {",
        "  local marker=\"$1\"",
        "  local outdir=\"$2\"",
        "  shift 2",
        "  mkdir -p \"$(dirname \"$marker\")\"",
        "  if [ -f \"$marker\" ] && [ \"${BABAPPA_FORCE:-0}\" != \"1\" ]; then",
        "    if [ ! -e \"$outdir\" ]; then",
        "      echo \"Completed marker exists but expected output is missing: $marker -> $outdir\" >&2",
        "      echo 'Resume recommendation: remove the stale marker after confirming the output is absent, then rerun.' >&2",
        "      exit 1",
        "    fi",
        "    echo \"Skipping completed stage: $marker\"",
        "    return 0",
        "  fi",
        "  if [ -e \"$outdir\" ] && [ \"${BABAPPA_FORCE:-0}\" != \"1\" ]; then",
        "    echo \"Existing output without marker: $outdir\"",
        "    if validate_existing_output \"$marker\" \"$outdir\"; then",
        "      echo \"Existing output validates; marking reusable stage complete: $marker\"",
        "      touch \"$marker\"",
        "      return 0",
        "    fi",
        "    echo \"Unsafe partial output collision for stage: $marker\" >&2",
        "    echo \"Expected output: $outdir\" >&2",
        "    echo 'Resume recommendation: validate or move the partial output, or set BABAPPA_FORCE=1 only when intentional.' >&2",
        "    exit 1",
        "  fi",
        "  run_stage \"$marker\" \"$@\"",
        "}",
        "",
    ]


def _mac_tier_names(scale: str, tier: str, feature_policy: str) -> Dict[str, str]:
    prefix = f"explicit_branch_truth_{scale}_mps_{tier}"
    return {
        "sim": f"sim_{prefix}",
        "align": f"align_{prefix}",
        "site_map": f"site_map_{prefix}",
        "method_policy": f"method_policy_{prefix}",
        "tensors": f"tensors_{prefix}",
        "dataset": f"dataset_{prefix}",
        "labels": f"branch_site_oracle_{prefix}",
        "branch_dataset": f"branch_site_dataset_{prefix}_streamed",
        "leakage": f"branch_site_leakage_{prefix}_streamed",
        "branch_neural": f"branch_site_neural_{prefix}_streamed",
        "calibration": f"branch_site_calibration_{prefix}_streamed",
        "aggregation": f"branch_aggregation_{prefix}_streamed",
        "controls": f"branch_aggregation_controls_{prefix}_streamed",
        "threshold": f"branch_site_threshold_policy_{prefix}_streamed",
        "aggregation_policy": f"branch_aggregation_policy_{prefix}_streamed",
        "summary": f"branch_site_run_summary_{prefix}_streamed",
        "truth_audit": f"branch_truth_status_audit_{prefix}",
        "feature_policy": feature_policy,
    }


def _expected_mac_outputs(config, scale: str) -> Dict[str, object]:
    return {
        "plan_only": True,
        "does_not_execute_jobs": True,
        "apple_silicon_mps": True,
        "scale": f"explicit_branch_truth_{scale}_mps",
        "n_families_per_tier": config.n_families_per_tier,
        "tiers": config.tiers,
        "methods": config.methods,
        "feature_policy": config.feature_policy,
        "truth_mode": config.truth_mode,
        "negative_downsample_ratio": config.negative_downsample_ratio,
        "max_output_rows_per_tier": config.max_output_rows_per_tier,
        "device": config.device,
        "batch_size": config.batch_size,
        "threads": config.threads,
        "mps_fallback": config.mps_fallback,
        "mps_high_watermark_ratio": config.mps_high_watermark_ratio,
        "allow_missing_babappalign": config.allow_missing_babappalign,
        "babappalign_model_expected_path": "$HOME/.cache/babappalign/models/babappascore.pt",
        "blocked_until_10k_passes": scale == "100k",
        "memory_policy": "36 GB unified memory requires streamed outputs, capped datasets, and resume checkpoints.",
        "expected_output_directories": {
            tier: _mac_tier_names(scale, tier, config.feature_policy)
            for tier in config.tiers
        },
    }


def _mac_markdown(config, scale: str) -> str:
    heading_scale = "100K" if scale == "100k" else "10K"
    lines = [
        f"# Explicit branch-truth {heading_scale} Apple Silicon/MPS plan",
        "",
        "## Scope",
        "",
        f"- Families per tier: {config.n_families_per_tier}",
        f"- Tiers: {', '.join(config.tiers)}",
        f"- Methods: {', '.join(config.methods)}",
        f"- Feature policy: `{config.feature_policy}`",
        f"- Truth mode: `{config.truth_mode}`",
        f"- Device: `{config.device}`",
        f"- Batch size: {config.batch_size}",
        f"- Threads: {config.threads}",
        "",
        "## Apple Silicon safety",
        "",
        "- Scripts export `PYTORCH_ENABLE_MPS_FALLBACK=1` before Python starts.",
        "- Scripts set OMP/MKL/OpenBLAS/NumExpr thread caps.",
        "- Scripts do not set CUDA_VISIBLE_DEVICES and do not call NVIDIA monitoring tools.",
        "- Monitor scripts use `vm_stat`, `memory_pressure`, `top`, `df`, and `du`.",
        "- Each tier and stage has marker files and the run script uses a flock lock.",
        "- If `babappalign` is requested, the run script checks `$HOME/.cache/babappalign/models/babappascore.pt` before starting.",
        "",
        "## Memory policy",
        "",
        "36 GB unified memory requires streamed outputs, capped branch-site datasets, conservative batch sizes, and resumable tier-by-tier execution.",
        "",
    ]
    if scale == "100k":
        lines.extend(
            [
                "## 100K gate",
                "",
                "DO NOT RUN 100K until the 10K MPS plan completes and validates.",
                "100K may require multiple days and large disk space. It should remain tier-resumable and never monolithic.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 100K policy",
                "",
                "Do not run 100K until conservative explicit branch-truth 10K MPS validates and controls pass.",
                "",
            ]
        )
    return "\n".join(lines)


def _run_script(config: ExplicitBranchTruthPrototypePlanConfig) -> str:
    lines = _header()
    lines.extend([
        "echo 'Explicit branch-site truth prototype plan only.'",
        "echo 'Current simulator may need branch_truth.json support before these TODO commands can run.'",
        "",
    ])
    for tier in config.tiers:
        methods = ",".join(config.methods)
        lines.extend([
            f"echo 'TODO explicit branch-truth tier: {tier}'",
            f"# TODO after simulator support exists:",
            f"# babappa simulate --outdir explicit_branch_truth_sim_{tier}_1k --n-families {config.n_families} --saturation-tier {tier} --emit-branch-truth",
            f"# babappa validate-sim --sim-dir explicit_branch_truth_sim_{tier}_1k",
            f"# babappa align-external --sim-dir explicit_branch_truth_sim_{tier}_1k --outdir explicit_branch_truth_align_{tier}_1k --methods {methods}",
            f"# babappa build-site-map --align-dir explicit_branch_truth_align_{tier}_1k --outdir explicit_branch_truth_site_map_{tier}_1k",
            f"# babappa extract-branch-site-labels --dataset-dir explicit_branch_truth_dataset_{tier}_1k --site-map-dir explicit_branch_truth_site_map_{tier}_1k --outdir explicit_branch_truth_labels_{tier}_1k --foreground-source truth",
            f"# babappa audit-branch-truth-status --tiers {tier} --outdir explicit_branch_truth_audit_{tier}_1k",
            "",
        ])
    lines.append("echo 'Do not run final 100K until this prototype passes.'")
    return "\n".join(lines) + "\n"


def _run_1k_script(config: ExplicitBranchTruth1kPlanConfig) -> str:
    lines = _conda_header(config.conda_sh, config.conda_env)
    lines.extend([
        "mkdir -p logs",
        "ts=$(date +%Y%m%d_%H%M%S)",
        "log=logs/explicit_branch_truth_1k_${ts}.log",
        "exec > >(tee -a \"$log\") 2>&1",
        "echo 'BABAPPA explicit branch-truth 1K prototype started.'",
        "",
    ])
    methods = ",".join(config.methods)
    for tier in config.tiers:
        prefix = f"explicit_branch_truth_1k_{tier}"
        lines.extend([
            f"echo '=== explicit branch-truth 1K tier: {tier} ==='",
            f"test ! -e sim_{prefix} || (echo 'Refusing to overwrite sim_{prefix}' >&2; exit 1)",
            f"babappa simulate --outdir sim_{prefix} --n-families {config.n_families_per_tier} --n-taxa 8 --n-codons 300 --seed 42 --positive-rate 0.5 --saturation-tier {tier}",
            f"babappa validate-sim --sim-dir sim_{prefix} --require-branch-truth",
            f"babappa audit-sim --sim-dir sim_{prefix} --outdir sim_{prefix}/audit",
            f"babappa align-external --sim-dir sim_{prefix} --outdir align_{prefix} --methods {methods}",
            f"babappa build-site-map --sim-dir sim_{prefix} --align-dir align_{prefix} --outdir site_map_{prefix} --methods {methods}",
            f"babappa aligner-method-policy --align-dir align_{prefix} --site-map-dir site_map_{prefix} --outdir method_policy_{prefix} --max-conflict-fraction 0.03 --max-frame-error-fraction 0.0 --max-method-failure-fraction 0.01",
            f"babappa validate-aligner-method-policy --policy-dir method_policy_{prefix}",
            f"babappa build-tensors --sim-dir sim_{prefix} --align-dir align_{prefix} --outdir tensors_{prefix} --methods {methods}",
            f"babappa validate-tensors --tensor-dir tensors_{prefix}",
            f"babappa index-dataset --tensor-dir tensors_{prefix} --outdir dataset_{prefix} --methods {methods}",
            f"babappa extract-branch-site-labels --dataset-dir dataset_{prefix} --site-map-dir site_map_{prefix} --outdir branch_site_oracle_{prefix} --truth-mode explicit --aligned-site-mode mapped --foreground-source truth --streaming-output",
            f"babappa validate-branch-site-labels --label-dir branch_site_oracle_{prefix}",
            f"babappa build-branch-site-dataset --dataset-dir dataset_{prefix} --branch-site-labels branch_site_oracle_{prefix}/branch_site_oracle_labels.tsv --outdir branch_site_dataset_{prefix}_streamed --negative-downsample-ratio {config.negative_downsample_ratio:g} --seed 42 --require-mappable-sites --streaming --max-output-rows 1000000",
            f"babappa audit-branch-site-leakage --branch-site-dataset-dir branch_site_dataset_{prefix}_streamed --outdir branch_site_leakage_{prefix}_streamed",
            f"babappa train-branch-site-baseline --branch-site-dataset-dir branch_site_dataset_{prefix}_streamed --outdir branch_site_baseline_{prefix}_streamed --epochs 300 --learning-rate 0.05",
            f"babappa train-branch-site-neural --branch-site-dataset-dir branch_site_dataset_{prefix}_streamed --outdir branch_site_neural_{prefix}_streamed --device auto --epochs 10 --batch-size 256 --max-train-items 50000 --max-val-items 10000 --max-calib-items 10000 --max-test-items 10000",
            f"babappa calibrate-branch-site-neural --model-dir branch_site_neural_{prefix}_streamed --outdir branch_site_calibration_{prefix}_streamed",
            f"babappa aggregate-branch-sites --predictions branch_site_neural_{prefix}_streamed/branch_site_neural_predictions.tsv --outdir branch_aggregation_{prefix}_streamed",
            f"babappa branch-aggregation-controls --predictions branch_site_neural_{prefix}_streamed/branch_site_neural_predictions.tsv --outdir branch_aggregation_controls_{prefix}_streamed --n-permutations 20",
            f"babappa branch-site-threshold-policy --predictions branch_site_calibration_{prefix}_streamed/branch_site_calibrated_predictions.tsv --outdir branch_site_threshold_policy_{prefix}_streamed --probability-column prob_positive_raw --calibrated-probability-column prob_positive_calibrated",
            f"babappa branch-aggregation-threshold-policy --aggregation-dir branch_aggregation_{prefix}_streamed --outdir branch_aggregation_policy_{prefix}_streamed",
            f"babappa summarize-branch-site-run --outdir branch_site_run_summary_{prefix}_streamed --title 'BABAPPA explicit branch-truth 1K {tier} summary' --branch-site-label-dir branch_site_oracle_{prefix} --branch-site-dataset-dir branch_site_dataset_{prefix}_streamed --branch-site-leakage-dir branch_site_leakage_{prefix}_streamed --branch-site-baseline-dir branch_site_baseline_{prefix}_streamed --branch-site-neural-dir branch_site_neural_{prefix}_streamed --branch-site-calibration-dir branch_site_calibration_{prefix}_streamed --branch-aggregation-dir branch_aggregation_{prefix}_streamed --branch-aggregation-controls-dir branch_aggregation_controls_{prefix}_streamed --branch-site-threshold-policy-dir branch_site_threshold_policy_{prefix}_streamed --branch-aggregation-threshold-policy-dir branch_aggregation_policy_{prefix}_streamed",
            f"babappa audit-branch-truth-status --tiers {tier} --run-name explicit_branch_truth_1k_streamed --outdir branch_truth_status_audit_{prefix}",
            "",
        ])
    lines.append("echo 'Do not run final 100K until explicit branch-truth 1K and 10K validation pass.'")
    return "\n".join(lines) + "\n"


def _run_10k_script(config: ExplicitBranchTruth10kPlanConfig) -> str:
    lines = _conda_header(config.conda_sh, config.conda_env)
    lines.extend([
        "mkdir -p logs",
        "ts=$(date +%Y%m%d_%H%M%S)",
        "log=logs/explicit_branch_truth_10k_${ts}.log",
        "exec > >(tee -a \"$log\") 2>&1",
        "echo 'BABAPPA conservative explicit branch-truth 10K validation started.'",
        "echo 'Feature policy: " + config.feature_policy + "'",
        "echo 'Truth mode: explicit'",
        "echo 'Full_context is an optional upper-bound diagnostic only.'",
        "",
    ])
    methods = ",".join(config.methods)
    ablation_root = "branch_context_ablation_explicit_branch_truth_10k"
    for tier in config.tiers:
        prefix = f"explicit_branch_truth_10k_{tier}"
        ablation_tier_dir = f"{ablation_root}/{tier}"
        conservative_predictions = f"{ablation_tier_dir}/{config.feature_policy}/baseline/branch_site_baseline_predictions.tsv"
        lines.extend([
            f"echo '=== conservative explicit branch-truth 10K tier: {tier} ==='",
            f"test ! -e sim_{prefix} || (echo 'Refusing to overwrite sim_{prefix}' >&2; exit 1)",
            f"babappa simulate --outdir sim_{prefix} --n-families {config.n_families_per_tier} --n-taxa 8 --n-codons 300 --seed 42 --positive-rate 0.5 --saturation-tier {tier}",
            f"babappa validate-sim --sim-dir sim_{prefix} --require-branch-truth",
            f"babappa audit-sim --sim-dir sim_{prefix} --outdir sim_{prefix}/audit",
            f"babappa align-external --sim-dir sim_{prefix} --outdir align_{prefix} --methods {methods}",
            f"babappa build-site-map --sim-dir sim_{prefix} --align-dir align_{prefix} --outdir site_map_{prefix} --methods {methods}",
            f"babappa aligner-method-policy --align-dir align_{prefix} --site-map-dir site_map_{prefix} --outdir method_policy_{prefix} --max-conflict-fraction 0.03 --max-frame-error-fraction 0.0 --max-method-failure-fraction 0.01",
            f"babappa validate-aligner-method-policy --policy-dir method_policy_{prefix}",
            f"babappa build-tensors --sim-dir sim_{prefix} --align-dir align_{prefix} --outdir tensors_{prefix} --methods {methods}",
            f"babappa validate-tensors --tensor-dir tensors_{prefix}",
            f"babappa index-dataset --tensor-dir tensors_{prefix} --outdir dataset_{prefix} --methods {methods}",
            f"babappa extract-branch-site-labels --dataset-dir dataset_{prefix} --site-map-dir site_map_{prefix} --outdir branch_site_oracle_{prefix} --truth-mode explicit --aligned-site-mode mapped --foreground-source truth --streaming-output",
            f"babappa validate-branch-site-labels --label-dir branch_site_oracle_{prefix}",
            f"babappa build-branch-site-dataset --dataset-dir dataset_{prefix} --branch-site-labels branch_site_oracle_{prefix}/branch_site_oracle_labels.tsv --outdir branch_site_dataset_{prefix}_streamed --negative-downsample-ratio {config.negative_downsample_ratio:g} --seed 42 --require-mappable-sites --streaming --max-output-rows {config.max_output_rows_per_tier}",
            f"babappa audit-branch-site-leakage --branch-site-dataset-dir branch_site_dataset_{prefix}_streamed --outdir branch_site_leakage_{prefix}_streamed",
            f"babappa run-branch-context-ablation --branch-site-dataset-dir branch_site_dataset_{prefix}_streamed --outdir {ablation_tier_dir} --profiles {config.feature_policy} --model baseline --seed 42 --epochs 300 --learning-rate 0.05",
            f"babappa aggregate-branch-sites --predictions {conservative_predictions} --outdir branch_aggregation_{prefix}_{config.feature_policy}_streamed",
            f"babappa branch-aggregation-controls --predictions {conservative_predictions} --outdir branch_aggregation_controls_{prefix}_{config.feature_policy}_streamed --n-permutations 100 --seed 42",
            f"babappa validate-branch-aggregation-controls --controls-dir branch_aggregation_controls_{prefix}_{config.feature_policy}_streamed",
            "",
        ])
    lines.append("echo 'Do not run final 100K until conservative explicit branch-truth 10K validation passes and controls are interpreted.'")
    return "\n".join(lines) + "\n"


def _monitor_script(config: ExplicitBranchTruthPrototypePlanConfig) -> str:
    lines = _header()
    lines.extend([
        "echo 'Plan-only monitor. No jobs are launched by the planner.'",
        "echo 'Expected prototype directories, if a user later implements and runs the TODO commands:'",
    ])
    for tier in config.tiers:
        lines.append(f"for dir in explicit_branch_truth_sim_{tier}_1k explicit_branch_truth_labels_{tier}_1k explicit_branch_truth_audit_{tier}_1k; do [ -d \"$dir\" ] && echo present:$dir || echo missing:$dir; done")
    return "\n".join(lines) + "\n"


def _monitor_1k_script(config: ExplicitBranchTruth1kPlanConfig) -> str:
    lines = _conda_header(config.conda_sh, config.conda_env)
    lines.extend([
        "echo 'Active BABAPPA / Python / aligner processes:'",
        "pgrep -af 'babappa|python|mafft|muscle|babappalign' || true",
        "echo 'Latest explicit branch-truth 1K log:'",
        "ls -t logs/explicit_branch_truth_1k_*.log 2>/dev/null | head -1 | xargs -r tail -80",
        "",
    ])
    for tier in config.tiers:
        prefix = f"explicit_branch_truth_1k_{tier}"
        lines.append(f"for dir in sim_{prefix} dataset_{prefix} branch_site_oracle_{prefix} branch_site_neural_{prefix}_streamed branch_truth_status_audit_{prefix}; do [ -d \"$dir\" ] && echo present:$dir || echo missing:$dir; done")
    return "\n".join(lines) + "\n"


def _monitor_10k_script(config: ExplicitBranchTruth10kPlanConfig) -> str:
    lines = _conda_header(config.conda_sh, config.conda_env)
    lines.extend([
        "echo 'Active BABAPPA / Python / aligner processes:'",
        "pgrep -af 'babappa|python|mafft|muscle|babappalign' || true",
        "echo 'Latest explicit branch-truth 10K log:'",
        "ls -t logs/explicit_branch_truth_10k_*.log 2>/dev/null | head -1 | xargs -r tail -80",
        "",
    ])
    for tier in config.tiers:
        prefix = f"explicit_branch_truth_10k_{tier}"
        lines.append(
            f"for dir in sim_{prefix} dataset_{prefix} branch_site_oracle_{prefix} "
            f"branch_context_ablation_explicit_branch_truth_10k/{tier}/{config.feature_policy} "
            f"branch_aggregation_controls_{prefix}_{config.feature_policy}_streamed; "
            "do [ -d \"$dir\" ] && echo present:$dir || echo missing:$dir; done"
        )
    return "\n".join(lines) + "\n"


def _validate_script(config: ExplicitBranchTruthPrototypePlanConfig) -> str:
    lines = _header()
    lines.extend([
        "echo 'Plan-only validation scaffold. It checks future outputs only if they exist.'",
    ])
    for tier in config.tiers:
        lines.extend([
            f"if [ -d explicit_branch_truth_audit_{tier}_1k ]; then babappa validate-branch-truth-status-audit --audit-dir explicit_branch_truth_audit_{tier}_1k; else echo 'MISSING future audit dir: explicit_branch_truth_audit_{tier}_1k'; fi",
        ])
    lines.append("echo 'Do not run final 100K until explicit branch-site truth validation passes.'")
    return "\n".join(lines) + "\n"


def _validate_1k_script(config: ExplicitBranchTruth1kPlanConfig) -> str:
    lines = _conda_header(config.conda_sh, config.conda_env)
    for tier in config.tiers:
        prefix = f"explicit_branch_truth_1k_{tier}"
        lines.extend([
            f"echo '=== validate explicit branch-truth 1K tier {tier} ==='",
            f"babappa validate-sim --sim-dir sim_{prefix} --require-branch-truth",
            f"babappa validate-index --index-dir dataset_{prefix}",
            f"babappa validate-branch-site-labels --label-dir branch_site_oracle_{prefix}",
            f"babappa validate-branch-site-dataset --branch-site-dataset-dir branch_site_dataset_{prefix}_streamed",
            f"babappa validate-branch-site-leakage --leakage-dir branch_site_leakage_{prefix}_streamed",
            f"babappa validate-branch-site-baseline --model-dir branch_site_baseline_{prefix}_streamed",
            f"babappa validate-branch-site-neural --model-dir branch_site_neural_{prefix}_streamed",
            f"babappa validate-branch-site-calibration --calibration-dir branch_site_calibration_{prefix}_streamed",
            f"babappa validate-branch-aggregation --aggregation-dir branch_aggregation_{prefix}_streamed",
            f"babappa validate-branch-aggregation-controls --controls-dir branch_aggregation_controls_{prefix}_streamed",
            f"babappa validate-branch-site-threshold-policy --policy-dir branch_site_threshold_policy_{prefix}_streamed",
            f"babappa validate-branch-aggregation-threshold-policy --policy-dir branch_aggregation_policy_{prefix}_streamed",
            f"babappa validate-branch-site-run-summary --summary-dir branch_site_run_summary_{prefix}_streamed",
            f"babappa validate-branch-truth-status-audit --audit-dir branch_truth_status_audit_{prefix}",
            "",
        ])
    return "\n".join(lines) + "\n"


def _validate_10k_script(config: ExplicitBranchTruth10kPlanConfig) -> str:
    lines = _conda_header(config.conda_sh, config.conda_env)
    for tier in config.tiers:
        prefix = f"explicit_branch_truth_10k_{tier}"
        lines.extend([
            f"echo '=== validate conservative explicit branch-truth 10K tier {tier} ==='",
            f"babappa validate-sim --sim-dir sim_{prefix} --require-branch-truth",
            f"babappa validate-index --index-dir dataset_{prefix}",
            f"babappa validate-branch-site-labels --label-dir branch_site_oracle_{prefix}",
            f"babappa validate-branch-site-dataset --branch-site-dataset-dir branch_site_dataset_{prefix}_streamed",
            f"test -f branch_context_ablation_explicit_branch_truth_10k/{tier}/{config.feature_policy}/profile_metrics.json",
            f"babappa validate-branch-aggregation --aggregation-dir branch_aggregation_{prefix}_{config.feature_policy}_streamed",
            f"babappa validate-branch-aggregation-controls --controls-dir branch_aggregation_controls_{prefix}_{config.feature_policy}_streamed",
            "",
        ])
    return "\n".join(lines) + "\n"


def _summarize_1k_script(config: ExplicitBranchTruth1kPlanConfig) -> str:
    lines = _conda_header(config.conda_sh, config.conda_env)
    lines.extend([
        "babappa summarize-branch-conditioned-tiers --tiers "
        + ",".join(config.tiers)
        + " --run-name explicit_branch_truth_1k_streamed --outdir explicit_branch_truth_1k_cross_tier_summary",
        "babappa validate-branch-conditioned-tier-summary --summary-dir explicit_branch_truth_1k_cross_tier_summary",
        "babappa audit-branch-truth-status --tiers "
        + ",".join(config.tiers)
        + " --run-name explicit_branch_truth_1k_streamed --outdir explicit_branch_truth_1k_truth_status_audit",
        "babappa validate-branch-truth-status-audit --audit-dir explicit_branch_truth_1k_truth_status_audit",
    ])
    return "\n".join(lines) + "\n"


def _summarize_10k_script(config: ExplicitBranchTruth10kPlanConfig) -> str:
    lines = _conda_header(config.conda_sh, config.conda_env)
    lines.extend([
        "babappa summarize-branch-context-ablation "
        "--ablation-dir branch_context_ablation_explicit_branch_truth_10k "
        "--outdir explicit_branch_truth_10k_context_ablation_summary",
        "babappa audit-branch-truth-status --tiers "
        + ",".join(config.tiers)
        + " --run-name explicit_branch_truth_10k --output-suffix _streamed --outdir explicit_branch_truth_10k_truth_status_audit",
        "babappa validate-branch-truth-status-audit --audit-dir explicit_branch_truth_10k_truth_status_audit",
        "echo 'Summarize conservative 10K aggregation/control outputs after all tiers complete.'",
    ])
    return "\n".join(lines) + "\n"


def _expected_outputs(config: ExplicitBranchTruthPrototypePlanConfig) -> Dict[str, object]:
    return {
        "plan_only": True,
        "does_not_execute_jobs": True,
        "n_families": config.n_families,
        "tiers": config.tiers,
        "methods": config.methods,
        "simulator_support_required": "current simulator may need branch_truth.json support before this plan can run",
        "defer_100k_until_passes": True,
        "required_truth_fields_per_family": [
            "family_id",
            "tree",
            "foreground_branch_id",
            "foreground_taxon",
            "branch_length",
            "selected_sites",
            "selected_site_by_branch",
            "y_branch_site matrix",
            "selection_event_id",
            "omega/background/foreground parameters if available",
            "saturation tier",
            "alignment method after mapping",
        ],
        "required_files": [
            "family_XXXX.branch_truth.json",
            "branch_truth_manifest.json",
            "branch_site_truth.tsv",
        ],
        "expected_output_directories": {
            tier: {
                "simulation": f"explicit_branch_truth_sim_{tier}_1k",
                "labels": f"explicit_branch_truth_labels_{tier}_1k",
                "truth_audit": f"explicit_branch_truth_audit_{tier}_1k",
            }
            for tier in config.tiers
        },
    }


def _expected_1k_outputs(config: ExplicitBranchTruth1kPlanConfig) -> Dict[str, object]:
    return {
        "plan_only": True,
        "does_not_execute_jobs": True,
        "scale": "explicit_branch_truth_1k",
        "n_families_per_tier": config.n_families_per_tier,
        "tiers": config.tiers,
        "methods": config.methods,
        "negative_downsample_ratio": config.negative_downsample_ratio,
        "truth_mode": "explicit",
        "required_truth_source": "explicit_simulator_branch_truth",
        "defer_100k_until_explicit_1k_and_10k_pass": True,
        "expected_output_directories": {
            tier: {
                "simulation": f"sim_explicit_branch_truth_1k_{tier}",
                "labels": f"branch_site_oracle_explicit_branch_truth_1k_{tier}",
                "summary": f"branch_site_run_summary_explicit_branch_truth_1k_{tier}_streamed",
                "truth_audit": f"branch_truth_status_audit_explicit_branch_truth_1k_{tier}",
            }
            for tier in config.tiers
        },
    }


def _expected_10k_outputs(config: ExplicitBranchTruth10kPlanConfig) -> Dict[str, object]:
    return {
        "plan_only": True,
        "does_not_execute_jobs": True,
        "scale": "explicit_branch_truth_10k",
        "n_families_per_tier": config.n_families_per_tier,
        "tiers": config.tiers,
        "methods": config.methods,
        "diagnostic_methods_excluded": {
            "prank": "diagnostic only, excluded from default",
            "tcoffee": "diagnostic only, excluded from default",
        },
        "feature_policy": config.feature_policy,
        "full_context_role": "optional context-aware upper-bound diagnostic only",
        "negative_downsample_ratio": config.negative_downsample_ratio,
        "max_output_rows_per_tier": config.max_output_rows_per_tier,
        "truth_mode": "explicit",
        "required_truth_source": "explicit_simulator_branch_truth",
        "defer_100k_until_conservative_10k_controls_pass": True,
        "expected_output_directories": {
            tier: {
                "simulation": f"sim_explicit_branch_truth_10k_{tier}",
                "labels": f"branch_site_oracle_explicit_branch_truth_10k_{tier}",
                "conservative_ablation": f"branch_context_ablation_explicit_branch_truth_10k/{tier}/{config.feature_policy}",
                "aggregation_controls": (
                    f"branch_aggregation_controls_explicit_branch_truth_10k_{tier}_{config.feature_policy}_streamed"
                ),
            }
            for tier in config.tiers
        },
    }


def _markdown(config: ExplicitBranchTruthPrototypePlanConfig, expected: Dict[str, object]) -> str:
    return "\n".join([
        "# Explicit branch-site truth prototype plan",
        "",
        "## Purpose",
        "",
        "This is a future-facing planner for validating explicit simulator branch-site truth before any final 100K run.",
        "",
        "## Run boundary",
        "",
        "The planner does not run jobs. The current simulator may need branch_truth.json support before this plan can run.",
        "",
        "## Prototype scope",
        "",
        f"- Families per tier: {config.n_families}",
        f"- Tiers: {', '.join(config.tiers)}",
        f"- Methods: {', '.join(config.methods)}",
        "",
        "## Required outputs",
        "",
        "- `family_XXXX.branch_truth.json`",
        "- `branch_truth_manifest.json`",
        "- `branch_site_truth.tsv`",
        "",
        "## 100K policy",
        "",
        "Do not run 100K until this explicit branch-site truth prototype passes.",
        "",
    ])


def _markdown_1k(config: ExplicitBranchTruth1kPlanConfig) -> str:
    return "\n".join([
        "# Explicit branch-truth 1K validation plan",
        "",
        "## Scope",
        "",
        f"- Families per tier: {config.n_families_per_tier}",
        f"- Tiers: {', '.join(config.tiers)}",
        f"- Methods: {', '.join(config.methods)}",
        "- Truth mode: explicit",
        "",
        "## Boundary",
        "",
        "This planner writes scripts only. It does not execute simulation, alignment, tensorization, training, or summary jobs.",
        "",
        "## 100K policy",
        "",
        "Do not run final 100K until explicit branch-truth 1K and 10K validation pass.",
        "",
    ])


def _markdown_10k(config: ExplicitBranchTruth10kPlanConfig) -> str:
    return "\n".join([
        "# Conservative explicit branch-truth 10K validation plan",
        "",
        "## Scope",
        "",
        f"- Families per tier: {config.n_families_per_tier}",
        f"- Tiers: {', '.join(config.tiers)}",
        f"- Methods: {', '.join(config.methods)}",
        f"- Feature policy: `{config.feature_policy}`",
        "- Truth mode: explicit",
        f"- Negative downsample ratio: {config.negative_downsample_ratio:g}",
        f"- Max output rows per tier: {config.max_output_rows_per_tier}",
        "",
        "## Feature Policy",
        "",
        "`conservative_branch_site` is the planned default because explicit 1K ablation showed context-only features are highly predictive. `full_context` is only an optional context-aware upper-bound diagnostic.",
        "",
        "## Method Policy",
        "",
        "Default methods are identity, MAFFT, BABAPPAlign, and MUSCLE with quarantine. PRANK and T-Coffee remain diagnostic-only and excluded from this plan.",
        "",
        "## Boundary",
        "",
        "This planner writes scripts only. It does not execute simulation, alignment, tensorization, training, controls, or summary jobs.",
        "",
        "## 100K policy",
        "",
        "Do not run final 100K until conservative explicit branch-truth 10K validation and strengthened aggregation controls pass.",
        "",
    ])


def _header() -> List[str]:
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# MANUAL EXECUTION SCRIPT - PLAN SCAFFOLD",
        "",
    ]


def _conda_header(conda_sh: str, conda_env: str) -> List[str]:
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"source {conda_sh}",
        f"conda activate {conda_env}",
        "",
        "# MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING",
        "",
    ]


def _parse_csv(value: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]
