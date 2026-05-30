"""Foreground-context ablation utilities for branch-site models."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from babappa import __version__
from babappa.branch.baseline import BranchSiteBaselineConfig, train_branch_site_baseline
from babappa.branch.summary import _normalize_output_suffix, _parse_tiers, _tier_prefix
from babappa.datasets.index import read_tsv, write_tsv

BRANCH_CONTEXT_ABLATION_VERSION = __version__

FOREGROUND_IDENTITY_COLUMNS = [
    "foreground_taxon_index",
    "branch_query_id_numeric",
    "foreground_taxon_present",
    "foreground_branch_present",
]
FOREGROUND_CODON_CONTEXT_COLUMNS = [
    "foreground_codon_id",
    "foreground_gap",
    "foreground_background_codon_delta",
]
FOREGROUND_ALL_COLUMNS = sorted(set(FOREGROUND_IDENTITY_COLUMNS + FOREGROUND_CODON_CONTEXT_COLUMNS))
CONTEXT_ONLY_COLUMNS = list(FOREGROUND_ALL_COLUMNS)
DEFAULT_ABLATION_PROFILES = [
    "full_model",
    "no_foreground_identity",
    "no_foreground_codon_context",
    "no_foreground_all",
    "context_only",
]
SUMMARY_FIELDS = [
    "tier",
    "profile",
    "model",
    "n_features",
    "excluded_columns",
    "test_n",
    "test_auroc",
    "test_f1",
    "test_mcc",
    "all_auroc",
    "all_f1",
    "all_mcc",
    "metrics_json",
]
INTERPRETATION_FIELDS = [
    "tier",
    "full_model_test_auroc",
    "context_only_test_auroc",
    "no_foreground_all_test_auroc",
    "full_minus_no_foreground_all_auroc",
    "warnings",
]


@dataclass(frozen=True)
class BranchContextAblationPlanConfig:
    """Configuration for a branch-context ablation plan."""

    run_name: str
    tiers: Union[str, Sequence[str]]
    outdir: str
    output_suffix: str = "_streamed"
    profiles: Union[str, Sequence[str]] = tuple(DEFAULT_ABLATION_PROFILES)
    ablation_root: str = "branch_context_ablation_explicit_1k"
    model: str = "baseline"
    seed: int = 42
    epochs: int = 300
    learning_rate: float = 0.05

    def __post_init__(self) -> None:
        if self.model != "baseline":
            raise ValueError("only baseline ablation planning is currently supported")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BranchContextAblationRunConfig:
    """Configuration for running baseline branch-context ablations on one tier."""

    branch_site_dataset_dir: str
    outdir: str
    profiles: Union[str, Sequence[str]] = tuple(DEFAULT_ABLATION_PROFILES)
    model: str = "baseline"
    seed: int = 42
    epochs: int = 300
    learning_rate: float = 0.05

    def __post_init__(self) -> None:
        dataset_dir = Path(self.branch_site_dataset_dir)
        if not dataset_dir.exists():
            raise ValueError(f"branch_site_dataset_dir does not exist: {dataset_dir}")
        for filename in ("branch_site_dataset_index.json", "branch_site_features.tsv", "branch_site_splits.tsv"):
            if not (dataset_dir / filename).exists():
                raise ValueError(f"branch_site_dataset_dir is missing {filename}: {dataset_dir}")
        if self.model != "baseline":
            raise ValueError("only baseline branch-context ablation is currently supported")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BranchContextAblationSummaryConfig:
    """Configuration for summarizing branch-context ablation outputs."""

    ablation_dir: str
    outdir: str

    def __post_init__(self) -> None:
        if not Path(self.ablation_dir).exists():
            raise ValueError(f"ablation_dir does not exist: {self.ablation_dir}")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BranchContextAblationInterpretationConfig:
    """Configuration for interpreting branch-context ablation summary outputs."""

    summary_dir: str
    outdir: str
    context_only_shortcut_threshold: float = 0.95
    foreground_drop_threshold: float = 0.10
    non_context_signal_threshold: float = 0.80

    def __post_init__(self) -> None:
        if not Path(self.summary_dir).exists():
            raise ValueError(f"summary_dir does not exist: {self.summary_dir}")
        if not (Path(self.summary_dir) / "branch_context_ablation_summary.tsv").exists():
            raise ValueError(f"summary_dir is missing branch_context_ablation_summary.tsv: {self.summary_dir}")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def plan_branch_context_ablation(config: BranchContextAblationPlanConfig) -> Dict[str, Any]:
    """Write scripts for foreground-context ablation without running jobs."""

    outdir = Path(config.outdir)
    tiers = _parse_tiers(config.tiers)
    profiles = _parse_profiles(config.profiles)
    suffix = _normalize_output_suffix(config.output_suffix)
    commands = []
    expected_outputs: Dict[str, Dict[str, str]] = {}
    for tier in tiers:
        prefix = _tier_prefix(config.run_name, tier, output_suffix=suffix)
        dataset_dir = f"branch_site_dataset_{prefix}"
        tier_outdir = f"{config.ablation_root}/{tier}"
        expected_outputs[tier] = {
            "branch_site_dataset_dir": dataset_dir,
            "outdir": tier_outdir,
        }
        commands.append(
            " ".join(
                [
                    "babappa run-branch-context-ablation",
                    f"--branch-site-dataset-dir {dataset_dir}",
                    f"--outdir {tier_outdir}",
                    f"--profiles {','.join(profiles)}",
                    "--model baseline",
                    f"--seed {config.seed}",
                    f"--epochs {config.epochs}",
                    f"--learning-rate {config.learning_rate:g}",
                ]
            )
        )
    run_path = outdir / "run_branch_context_ablation.sh"
    script = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "echo 'BABAPPA branch-context ablation started: '\"$(date)\"",
        "echo 'This script uses existing explicit branch-site datasets and does not run alignments or 10K/100K jobs.'",
        f"mkdir -p {config.ablation_root}",
        "",
        *commands,
        "",
        "babappa summarize-branch-context-ablation "
        f"--ablation-dir {config.ablation_root} --outdir {config.ablation_root}_summary",
        "echo 'BABAPPA branch-context ablation completed: '\"$(date)\"",
        "",
    ]
    run_path.write_text("\n".join(script), encoding="utf-8")
    run_path.chmod(0o755)

    expected_path = outdir / "expected_outputs.json"
    expected = {
        "branch_context_ablation_version": BRANCH_CONTEXT_ABLATION_VERSION,
        "plan_only": True,
        "does_not_run_jobs": True,
        "run_name": config.run_name,
        "tiers": tiers,
        "output_suffix": suffix,
        "profiles": profiles,
        "model": config.model,
        "ablation_root": config.ablation_root,
        "expected_outputs": expected_outputs,
    }
    _write_json(expected_path, expected)
    markdown_path = outdir / "branch_context_ablation_plan.md"
    markdown_path.write_text(_render_plan_markdown(expected, run_path), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "run": str(run_path),
        "expected_outputs": str(expected_path),
        "markdown": str(markdown_path),
        "does_not_run_jobs": True,
        "profiles": profiles,
    }


def run_branch_context_ablation(config: BranchContextAblationRunConfig) -> Dict[str, Any]:
    """Run baseline foreground-context ablation profiles on one branch-site dataset."""

    dataset_dir = Path(config.branch_site_dataset_dir)
    outdir = Path(config.outdir)
    profiles = _parse_profiles(config.profiles)
    source_index = _load_index(dataset_dir)
    source_features = list(source_index.get("feature_columns") or [])
    if not source_features:
        raise ValueError("source branch-site dataset has no feature_columns")

    profile_summaries = []
    for profile in profiles:
        profile_dir = outdir / profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        selected_features = branch_context_profile_columns(source_features, profile)
        excluded = [column for column in source_features if column not in selected_features]
        filtered_dataset = profile_dir / "filtered_dataset"
        _write_filtered_dataset(dataset_dir, filtered_dataset, source_index, profile, selected_features, excluded)
        model_dir = profile_dir / "baseline"
        train_summary = train_branch_site_baseline(
            BranchSiteBaselineConfig(
                branch_site_dataset_dir=str(filtered_dataset),
                outdir=str(model_dir),
                seed=config.seed,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
            )
        )
        metrics = _load_json(model_dir / "branch_site_baseline_metrics.json")
        profile_payload = {
            "branch_context_ablation_version": BRANCH_CONTEXT_ABLATION_VERSION,
            "profile": profile,
            "model": config.model,
            "branch_site_dataset_dir": str(dataset_dir),
            "filtered_dataset_dir": str(filtered_dataset),
            "model_dir": str(model_dir),
            "feature_columns": selected_features,
            "excluded_columns": excluded,
            "n_features": len(selected_features),
            "metrics": metrics,
            "warnings": train_summary.get("warnings", []),
        }
        metrics_path = profile_dir / "profile_metrics.json"
        tsv_path = profile_dir / "profile_metrics.tsv"
        _write_json(metrics_path, profile_payload)
        write_tsv(tsv_path, [_summary_row("", profile_payload, metrics_path)], SUMMARY_FIELDS)
        profile_summaries.append(profile_payload)

    summary_path = outdir / "branch_context_ablation_run.json"
    _write_json(
        summary_path,
        {
            "branch_context_ablation_version": BRANCH_CONTEXT_ABLATION_VERSION,
            "branch_site_dataset_dir": str(dataset_dir),
            "profiles": [
                {
                    "profile": payload["profile"],
                    "n_features": payload["n_features"],
                    "excluded_columns": payload["excluded_columns"],
                }
                for payload in profile_summaries
            ],
            "model": config.model,
        },
    )
    return {
        "status": "ok",
        "outdir": str(outdir),
        "profiles": [payload["profile"] for payload in profile_summaries],
        "n_profiles": len(profile_summaries),
        "summary": str(summary_path),
    }


def summarize_branch_context_ablation(config: BranchContextAblationSummaryConfig) -> Dict[str, Any]:
    """Summarize branch-context ablation profile metrics."""

    ablation_dir = Path(config.ablation_dir)
    outdir = Path(config.outdir)
    rows: List[Dict[str, Any]] = []
    profile_payloads = []
    for metrics_path in sorted(ablation_dir.glob("*/*/profile_metrics.json")):
        tier = metrics_path.parent.parent.name
        payload = _load_json(metrics_path)
        row = _summary_row(tier, payload, metrics_path)
        rows.append(row)
        profile_payloads.append(payload)
    if not rows:
        for metrics_path in sorted(ablation_dir.glob("*/profile_metrics.json")):
            payload = _load_json(metrics_path)
            row = _summary_row("", payload, metrics_path)
            rows.append(row)
            profile_payloads.append(payload)
    if not rows:
        raise ValueError(f"no profile_metrics.json files found under {ablation_dir}")

    json_path = outdir / "branch_context_ablation_summary.json"
    tsv_path = outdir / "branch_context_ablation_summary.tsv"
    markdown_path = outdir / "branch_context_ablation_summary.md"
    summary = {
        "branch_context_ablation_version": BRANCH_CONTEXT_ABLATION_VERSION,
        "ablation_dir": str(ablation_dir),
        "profiles": sorted({row["profile"] for row in rows}),
        "tiers": sorted({row["tier"] for row in rows if row["tier"]}),
        "rows": rows,
        "interpretation": {
            "no_foreground_all": "If no_foreground_all remains strong, branch-site signal is not solely foreground-context shortcut.",
            "context_only": "If context_only is very high, foreground-context may be too predictive and should be treated cautiously.",
            "collapse": "If performance collapses without foreground context, the model may depend heavily on context and needs redesign or clearer claims.",
        },
        "generated_files": {
            "json": str(json_path),
            "tsv": str(tsv_path),
            "markdown": str(markdown_path),
        },
    }
    _write_json(json_path, summary)
    write_tsv(tsv_path, rows, SUMMARY_FIELDS)
    markdown_path.write_text(_render_summary_markdown(summary), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "json": str(json_path),
        "tsv": str(tsv_path),
        "markdown": str(markdown_path),
        "n_rows": len(rows),
    }


def interpret_branch_context_ablation(config: BranchContextAblationInterpretationConfig) -> Dict[str, Any]:
    """Interpret explicit branch-context ablation metrics as a feature-policy decision."""

    summary_dir = Path(config.summary_dir)
    outdir = Path(config.outdir)
    rows = read_tsv(summary_dir / "branch_context_ablation_summary.tsv")
    by_tier: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in rows:
        tier = row.get("tier", "")
        profile = row.get("profile", "")
        if tier and profile:
            by_tier.setdefault(tier, {})[profile] = row
    if not by_tier:
        raise ValueError("ablation summary has no tier/profile rows")

    tier_rows = []
    global_warnings = set()
    no_foreground_all_values = []
    for tier in sorted(by_tier):
        profiles = by_tier[tier]
        full = _float_or_none(profiles.get("full_model", {}).get("test_auroc"))
        context_only = _float_or_none(profiles.get("context_only", {}).get("test_auroc"))
        no_foreground_all = _float_or_none(profiles.get("no_foreground_all", {}).get("test_auroc"))
        warnings = []
        if context_only is not None and context_only >= config.context_only_shortcut_threshold:
            warnings.append("context_only_shortcut_high")
            global_warnings.add("context_only_shortcut_high")
        drop = None
        if full is not None and no_foreground_all is not None:
            drop = full - no_foreground_all
            no_foreground_all_values.append(no_foreground_all)
            if drop > config.foreground_drop_threshold:
                warnings.append("foreground_context_dependence")
                global_warnings.add("foreground_context_dependence")
        tier_rows.append(
            {
                "tier": tier,
                "full_model_test_auroc": full,
                "context_only_test_auroc": context_only,
                "no_foreground_all_test_auroc": no_foreground_all,
                "full_minus_no_foreground_all_auroc": drop,
                "warnings": ";".join(warnings),
            }
        )

    non_context_signal = bool(no_foreground_all_values) and all(
        value >= config.non_context_signal_threshold for value in no_foreground_all_values
    )
    conclusions = []
    if non_context_signal:
        conclusions.append("non_context_sequence_signal_present")

    json_path = outdir / "branch_context_ablation_interpretation.json"
    tsv_path = outdir / "branch_context_ablation_interpretation.tsv"
    markdown_path = outdir / "branch_context_ablation_interpretation.md"
    payload = {
        "branch_context_ablation_version": BRANCH_CONTEXT_ABLATION_VERSION,
        "summary_dir": str(summary_dir),
        "recommended_next_default": "conservative_branch_site",
        "warnings": sorted(global_warnings),
        "conclusions": conclusions,
        "non_context_sequence_signal_present": non_context_signal,
        "ten_k_readiness": {
            "full_context_only_10k": "not_ready",
            "conservative_profile_10k": (
                "ready_only_after_conservative_profile_plan_generated_and_strengthened_controls_rerun"
            ),
        },
        "tier_rows": tier_rows,
        "generated_files": {
            "json": str(json_path),
            "tsv": str(tsv_path),
            "markdown": str(markdown_path),
        },
    }
    _write_json(json_path, payload)
    write_tsv(tsv_path, tier_rows, INTERPRETATION_FIELDS)
    markdown_path.write_text(_render_interpretation_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "json": str(json_path),
        "tsv": str(tsv_path),
        "markdown": str(markdown_path),
        "warnings": sorted(global_warnings),
        "conclusions": conclusions,
        "recommended_next_default": "conservative_branch_site",
        "ten_k_readiness": payload["ten_k_readiness"],
    }


def branch_context_profile_columns(feature_columns: Sequence[str], profile: str) -> List[str]:
    """Return feature columns for a foreground-context ablation profile."""

    columns = list(feature_columns)
    profile = profile.strip()
    if profile in {"full_model", "full_context"}:
        selected = columns
    elif profile in {"no_foreground_identity", "conservative_branch_site"}:
        selected = [column for column in columns if column not in FOREGROUND_IDENTITY_COLUMNS]
    elif profile == "no_foreground_codon_context":
        selected = [column for column in columns if column not in FOREGROUND_CODON_CONTEXT_COLUMNS]
    elif profile == "no_foreground_all":
        selected = [column for column in columns if column not in FOREGROUND_ALL_COLUMNS]
    elif profile == "context_only":
        selected = [column for column in columns if column in CONTEXT_ONLY_COLUMNS]
    else:
        raise ValueError(f"unknown branch context ablation profile: {profile}")
    if not selected:
        raise ValueError(f"profile {profile} selects no feature columns")
    return selected


def _parse_profiles(value: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(value, str):
        profiles = [item.strip() for item in value.split(",") if item.strip()]
    else:
        profiles = [str(item).strip() for item in value if str(item).strip()]
    return profiles or list(DEFAULT_ABLATION_PROFILES)


def _write_filtered_dataset(
    source_dir: Path,
    filtered_dir: Path,
    source_index: Dict[str, Any],
    profile: str,
    selected_features: List[str],
    excluded: List[str],
) -> None:
    filtered_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "branch_site_features.tsv", filtered_dir / "branch_site_features.tsv")
    shutil.copy2(source_dir / "branch_site_splits.tsv", filtered_dir / "branch_site_splits.tsv")
    index_payload = dict(source_index)
    index_payload["feature_columns"] = selected_features
    index_payload["ablation_profile"] = profile
    index_payload["ablation_excluded_columns"] = excluded
    index_payload["source_branch_site_dataset_dir"] = str(source_dir)
    _write_json(filtered_dir / "branch_site_dataset_index.json", index_payload)


def _summary_row(tier: str, payload: Dict[str, Any], metrics_path: Path) -> Dict[str, Any]:
    metrics = payload.get("metrics") or {}
    test = ((metrics.get("metrics_by_split") or {}).get("test") or {})
    all_metrics = ((metrics.get("metrics_by_split") or {}).get("all") or {})
    return {
        "tier": tier,
        "profile": payload.get("profile"),
        "model": payload.get("model", "baseline"),
        "n_features": payload.get("n_features"),
        "excluded_columns": ";".join(payload.get("excluded_columns") or []),
        "test_n": test.get("n"),
        "test_auroc": test.get("auroc"),
        "test_f1": test.get("f1"),
        "test_mcc": test.get("mcc"),
        "all_auroc": all_metrics.get("auroc"),
        "all_f1": all_metrics.get("f1"),
        "all_mcc": all_metrics.get("mcc"),
        "metrics_json": str(metrics_path),
    }


def _render_plan_markdown(payload: Dict[str, Any], run_path: Path) -> str:
    lines = [
        "# BABAPPA branch-context ablation plan",
        "",
        "This is a plan-only artifact. It does not execute ablation jobs automatically.",
        "",
        f"- Run script: `{run_path}`",
        f"- Model: `{payload.get('model')}`",
        f"- Profiles: `{', '.join(payload.get('profiles', []))}`",
        "",
        "## Interpretation",
        "",
        "- If no_foreground_all remains strong, branch-site signal is not solely foreground-context shortcut.",
        "- If context_only is very high, foreground-context may be too predictive and should be treated cautiously.",
        "- If performance collapses without foreground context, the model may depend heavily on context and needs redesign or clearer claims.",
        "",
    ]
    return "\n".join(lines)


def _render_summary_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Branch context ablation summary",
        "",
        "## Interpretation",
        "",
        "- If no_foreground_all remains strong, branch-site signal is not solely foreground-context shortcut.",
        "- If context_only is very high, foreground-context may be too predictive and should be treated cautiously.",
        "- If performance collapses without foreground context, the model may depend heavily on context and needs redesign or clearer claims.",
        "",
        "## Results",
        "",
        "| Tier | Profile | Test AUROC | Test F1 | Test MCC | Features |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("rows", []):
        lines.append(
            f"| {row.get('tier')} | {row.get('profile')} | {row.get('test_auroc')} | "
            f"{row.get('test_f1')} | {row.get('test_mcc')} | {row.get('n_features')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_interpretation_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Branch context ablation interpretation",
        "",
        "## Decision",
        "",
        "- Recommended next default: `conservative_branch_site`",
        "- Full-context model: context-aware upper-bound, not the main conservative branch-site claim.",
        "- 10K readiness: not ready for full_context-only 10K.",
        "",
        "## Warnings",
        "",
    ]
    warnings = payload.get("warnings") or []
    if warnings:
        lines.extend(f"- `{warning}`" for warning in warnings)
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Conclusions",
        "",
    ])
    conclusions = payload.get("conclusions") or []
    if conclusions:
        lines.extend(f"- `{conclusion}`" for conclusion in conclusions)
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Tier Metrics",
        "",
        "| Tier | full_model AUROC | context_only AUROC | no_foreground_all AUROC | Full minus no_foreground_all | Warnings |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in payload.get("tier_rows", []):
        lines.append(
            f"| {row.get('tier')} | {row.get('full_model_test_auroc')} | "
            f"{row.get('context_only_test_auroc')} | {row.get('no_foreground_all_test_auroc')} | "
            f"{row.get('full_minus_no_foreground_all_auroc')} | {row.get('warnings')} |"
        )
    lines.extend([
        "",
        "## 10K Readiness",
        "",
        "- Full-context-only 10K: `not_ready`",
        "- Conservative profile 10K: `ready_only_after_conservative_profile_plan_generated_and_strengthened_controls_rerun`",
        "",
    ])
    return "\n".join(lines)


def _float_or_none(value: object) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_index(dataset_dir: Path) -> Dict[str, Any]:
    return _load_json(dataset_dir / "branch_site_dataset_index.json")


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
