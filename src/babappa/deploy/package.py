"""Deployable model packaging for conservative branch-site models."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from babappa import __version__
from babappa.datasets.index import read_tsv
from babappa.training.neural_env import resolve_torch_device, safe_import_torch

TIERS = ["low", "moderate", "high", "extreme"]
METHODS = ["identity", "mafft", "babappalign", "muscle"]
FORBIDDEN_EMPIRICAL_INPUT_COLUMNS = [
    "branch_site_truth.tsv",
    "selected_sites",
    "truth.json",
    "branch_truth.json",
    "oracle",
    "y_branch_site",
    "y_site",
    "gene_label",
]
KNOWN_WARNINGS = [
    "context_only_shortcut_high",
    "foreground_context_columns_present",
    "simulation_supervised_only",
    "conditional_pass_due_pruned_raw_intermediates",
]


@dataclass(frozen=True)
class DeployableModelPackageConfig:
    """Configuration for packaging a retained simulation-trained model bundle."""

    run_name: str
    model_dirs: Sequence[str] | str
    calibration_dirs: Sequence[str] | str
    truth_audit_dir: str
    validation_report: str
    feature_policy: str
    truth_mode: str
    methods: Sequence[str] | str
    outdir: str


@dataclass(frozen=True)
class DeployableModelPackageValidationConfig:
    """Configuration for deployable package validation."""

    package_dir: str


@dataclass(frozen=True)
class DeployableModelSmokeConfig:
    """Configuration for deployable package loading smoke tests."""

    package_dir: str
    device: str = "auto"
    outdir: str = "deployable_model_load_smoke"


def package_deployable_model(config: DeployableModelPackageConfig) -> Dict[str, Any]:
    """Package retained checkpoint and calibration artifacts into a deployable bundle."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_dirs = _tier_map(_parse_csv(config.model_dirs), "model_dirs")
    calibration_dirs = _tier_map(_parse_csv(config.calibration_dirs), "calibration_dirs")
    methods = _parse_csv(config.methods)
    blockers = _package_blockers(config, model_dirs, calibration_dirs, methods)
    if blockers:
        payload = _blocker_payload(config, blockers)
        _write_json(outdir / "packaging_blocker_report.json", payload)
        (outdir / "packaging_blocker_report.md").write_text(_render_blocker_md(payload), encoding="utf-8")
        return {
            "status": "blocked",
            "outdir": str(outdir),
            "blockers": blockers,
            "blocker_report": str(outdir / "packaging_blocker_report.json"),
        }

    tier_model_dir = outdir / "tier_models"
    tier_calibration_dir = outdir / "tier_calibrations"
    tier_model_dir.mkdir(parents=True, exist_ok=True)
    tier_calibration_dir.mkdir(parents=True, exist_ok=True)

    validation_report = _read_json(Path(config.validation_report))
    truth_audit_rows = read_tsv(Path(config.truth_audit_dir) / "branch_truth_status_audit.tsv")
    first_meta = _read_json(model_dirs["low"] / "branch_site_neural_model_meta.json")
    feature_columns = [str(column) for column in first_meta.get("feature_columns", [])]
    copied_artifacts: List[Dict[str, Any]] = []
    tier_models: Dict[str, Any] = {}
    tier_calibrations: Dict[str, Any] = {}

    for tier in TIERS:
        model_target = tier_model_dir / tier
        calibration_target = tier_calibration_dir / tier
        model_target.mkdir(parents=True, exist_ok=True)
        calibration_target.mkdir(parents=True, exist_ok=True)
        for filename in [
            "branch_site_neural_checkpoint.pt",
            "branch_site_neural_model_meta.json",
            "branch_site_neural_metrics.json",
            "branch_site_neural_history.tsv",
        ]:
            source = model_dirs[tier] / filename
            if source.exists():
                copied_artifacts.append(_copy_artifact(source, model_target / filename, outdir, tier, "model"))
        for filename in [
            "branch_site_calibration.json",
            "branch_site_calibrated_metrics.json",
            "branch_site_calibration.md",
        ]:
            source = calibration_dirs[tier] / filename
            if source.exists():
                copied_artifacts.append(_copy_artifact(source, calibration_target / filename, outdir, tier, "calibration"))
        model_meta = _read_json(model_dirs[tier] / "branch_site_neural_model_meta.json")
        metrics = _read_json(model_dirs[tier] / "branch_site_neural_metrics.json")
        calibration = _read_json(calibration_dirs[tier] / "branch_site_calibration.json")
        tier_models[tier] = {
            "source_dir": str(model_dirs[tier]),
            "package_dir": str((model_target).relative_to(outdir)),
            "checkpoint": str((model_target / "branch_site_neural_checkpoint.pt").relative_to(outdir)),
            "model_meta": str((model_target / "branch_site_neural_model_meta.json").relative_to(outdir)),
            "metrics": str((model_target / "branch_site_neural_metrics.json").relative_to(outdir)),
            "history": str((model_target / "branch_site_neural_history.tsv").relative_to(outdir)),
            "n_features": model_meta.get("n_features"),
            "hidden_dim": model_meta.get("hidden_dim"),
            "dropout": model_meta.get("dropout"),
            "device_used": model_meta.get("device"),
            "metrics_by_split": metrics.get("metrics_by_split", {}),
        }
        tier_calibrations[tier] = {
            "source_dir": str(calibration_dirs[tier]),
            "package_dir": str((calibration_target).relative_to(outdir)),
            "calibration": str((calibration_target / "branch_site_calibration.json").relative_to(outdir)),
            "metrics": str((calibration_target / "branch_site_calibrated_metrics.json").relative_to(outdir)),
            "temperature": calibration.get("temperature"),
            "selected_threshold": calibration.get("selected_threshold"),
            "target_fdr": calibration.get("target_fdr"),
            "warnings": calibration.get("warnings", []),
        }

    manifest = _build_manifest(
        config,
        methods,
        validation_report,
        truth_audit_rows,
        feature_columns,
        tier_models,
        tier_calibrations,
        copied_artifacts,
    )
    feature_schema = _build_feature_schema(config.feature_policy, feature_columns)
    calibration_schema = _build_calibration_schema(tier_calibrations)
    training_envelope = _build_training_envelope(validation_report, tier_models)
    validation_summary = _build_validation_summary(validation_report, truth_audit_rows)

    _write_json(outdir / "model_manifest.json", manifest)
    _write_json(outdir / "feature_schema.json", feature_schema)
    _write_json(outdir / "calibration_schema.json", calibration_schema)
    _write_json(outdir / "training_envelope.json", training_envelope)
    _write_json(outdir / "validation_summary.json", validation_summary)
    (outdir / "model_card.md").write_text(_render_model_card(manifest), encoding="utf-8")
    (outdir / "limitations.md").write_text(_render_limitations(), encoding="utf-8")
    (outdir / "README.md").write_text(_render_readme(manifest), encoding="utf-8")
    _write_checksums(outdir / "checksums.sha256", copied_artifacts + _self_artifacts(outdir))

    return {
        "status": "ok",
        "outdir": str(outdir),
        "manifest": str(outdir / "model_manifest.json"),
        "model_card": str(outdir / "model_card.md"),
        "feature_schema": str(outdir / "feature_schema.json"),
        "calibration_schema": str(outdir / "calibration_schema.json"),
        "training_envelope": str(outdir / "training_envelope.json"),
        "n_model_files": sum(1 for item in copied_artifacts if item["kind"] == "model"),
        "n_calibration_files": sum(1 for item in copied_artifacts if item["kind"] == "calibration"),
        "checksums": str(outdir / "checksums.sha256"),
    }


def validate_deployable_model_package(config: DeployableModelPackageValidationConfig) -> Dict[str, Any]:
    """Validate a packaged deployable model bundle."""

    package_dir = Path(config.package_dir)
    failures: List[str] = []
    warnings: List[str] = []
    required = [
        "model_manifest.json",
        "model_card.md",
        "feature_schema.json",
        "calibration_schema.json",
        "training_envelope.json",
        "checksums.sha256",
        "validation_summary.json",
        "limitations.md",
        "README.md",
    ]
    for filename in required:
        if not (package_dir / filename).exists():
            failures.append(f"missing_file:{filename}")
    manifest = _read_optional_json(package_dir / "model_manifest.json", failures)
    feature_schema = _read_optional_json(package_dir / "feature_schema.json", failures)
    card_text = (package_dir / "model_card.md").read_text(encoding="utf-8") if (package_dir / "model_card.md").exists() else ""

    if manifest:
        if manifest.get("feature_policy") != "conservative_branch_site":
            failures.append("feature_policy_not_conservative_branch_site")
        if manifest.get("truth_mode") != "explicit":
            failures.append("truth_mode_not_explicit")
        if manifest.get("empirical_claim_status") != "not_final_empirical_inference":
            failures.append("empirical_claim_status_not_blocked")
        methods = set(manifest.get("methods_supported") or [])
        if methods != set(METHODS):
            failures.append("unexpected_methods:" + ",".join(sorted(methods)))
        if {"prank", "tcoffee", "t-coffee"} & {method.lower() for method in methods}:
            failures.append("diagnostic_aligner_in_production_defaults")
        known_warnings = set(manifest.get("known_warnings") or [])
        for warning in ["context_only_shortcut_high", "foreground_context_columns_present"]:
            if warning not in known_warnings:
                failures.append(f"missing_warning:{warning}")
        if manifest.get("truth_audit", {}).get("proxy_label_tiers"):
            failures.append("truth_audit_contains_proxy_label_tiers")
        for artifact in manifest.get("artifacts", []):
            rel = artifact.get("relative_path")
            if not rel:
                failures.append("artifact_missing_relative_path")
                continue
            candidate = package_dir / rel
            if not candidate.exists():
                failures.append(f"missing_artifact:{rel}")
            elif _sha256(candidate) != artifact.get("sha256"):
                failures.append(f"checksum_mismatch:{rel}")

    if feature_schema:
        blocked = set(feature_schema.get("blocked_empirical_input_columns") or [])
        for column in ["y_branch_site", "y_site", "gene_label"]:
            if column not in blocked:
                failures.append(f"feature_schema_does_not_block:{column}")
    if "simulation-supervised" not in card_text.lower():
        failures.append("model_card_missing_simulation_supervised_limitation")
    if "not final empirical" not in card_text.lower():
        failures.append("model_card_missing_empirical_claim_boundary")
    forbidden_files = _find_forbidden_package_files(package_dir)
    if forbidden_files:
        failures.extend(f"forbidden_raw_truth_file:{path}" for path in forbidden_files)

    report = {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "package_dir": str(package_dir),
    }
    _write_json(package_dir / "deployable_model_package_validation.json", report)
    (package_dir / "deployable_model_package_validation.md").write_text(
        _render_validation_md(report),
        encoding="utf-8",
    )
    return report


def smoke_load_deployable_model(config: DeployableModelSmokeConfig) -> Dict[str, Any]:
    """Smoke-load a deployable model package and run tiny forward passes when possible."""

    package_dir = Path(config.package_dir)
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    failures: List[str] = []
    warnings: List[str] = []
    manifest = _read_optional_json(package_dir / "model_manifest.json", failures)
    feature_schema = _read_optional_json(package_dir / "feature_schema.json", failures)
    calibration_schema = _read_optional_json(package_dir / "calibration_schema.json", failures)
    if failures:
        payload = _smoke_payload("fail", config, None, False, True, [], failures, warnings)
        _write_smoke_reports(outdir, payload)
        return payload

    torch, torch_error = safe_import_torch()
    if torch is None:
        warnings.append(f"torch_unavailable_metadata_only:{torch_error}")
        payload = _smoke_payload("ok", config, "metadata-only", False, True, TIERS, failures, warnings)
        _write_smoke_reports(outdir, payload)
        return payload

    try:
        device = resolve_torch_device(torch, config.device)
    except Exception as exc:
        warnings.append(f"device_resolution_failed_metadata_only:{exc}")
        payload = _smoke_payload("ok", config, "metadata-only", False, True, TIERS, failures, warnings)
        _write_smoke_reports(outdir, payload)
        return payload

    loaded: List[str] = []
    forward_pass = False
    n_features = len(feature_schema.get("expected_feature_columns") or [])
    try:
        from babappa.site.neural_model import SiteMLPClassifier
    except Exception as exc:  # pragma: no cover - environment dependent
        warnings.append(f"model_class_import_failed_metadata_only:{exc}")
        payload = _smoke_payload("ok", config, "metadata-only", False, True, TIERS, failures, warnings)
        _write_smoke_reports(outdir, payload)
        return payload

    for tier, model_info in (manifest.get("tier_models") or {}).items():
        try:
            checkpoint_path = package_dir / model_info["checkpoint"]
            checkpoint = _torch_load(torch, checkpoint_path)
            model = SiteMLPClassifier(
                input_dim=n_features,
                hidden_dim=int(model_info.get("hidden_dim") or 64),
                dropout=0.0,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
            with torch.no_grad():
                x = torch.zeros((2, n_features), dtype=torch.float32, device=device)
                logits = model(x)
                _ = torch.sigmoid(logits).detach().cpu().tolist()
            loaded.append(str(tier))
            forward_pass = True
        except Exception as exc:
            warnings.append(f"{tier}:model_load_or_forward_failed:{exc}")

    metadata_only = not forward_pass
    status = "ok" if not failures else "fail"
    payload = _smoke_payload(status, config, device, forward_pass, metadata_only, loaded, failures, warnings)
    payload["calibration_tiers_loaded"] = sorted((calibration_schema.get("tiers") or {}).keys())
    _write_smoke_reports(outdir, payload)
    return payload


def _package_blockers(
    config: DeployableModelPackageConfig,
    model_dirs: Dict[str, Path],
    calibration_dirs: Dict[str, Path],
    methods: List[str],
) -> List[str]:
    blockers: List[str] = []
    if config.feature_policy != "conservative_branch_site":
        blockers.append("feature_policy_must_be_conservative_branch_site")
    if config.truth_mode != "explicit":
        blockers.append("truth_mode_must_be_explicit")
    if methods != METHODS:
        blockers.append("methods_must_be_identity_mafft_babappalign_muscle")
    truth_tsv = Path(config.truth_audit_dir) / "branch_truth_status_audit.tsv"
    if not truth_tsv.exists():
        blockers.append(f"missing_truth_audit:{truth_tsv}")
    else:
        rows = read_tsv(truth_tsv)
        proxy = [row.get("tier", "") for row in rows if row.get("proxy_from_foreground_taxon") != "False"]
        if proxy:
            blockers.append("truth_audit_has_proxy_labels:" + ",".join(proxy))
    if not Path(config.validation_report).exists():
        blockers.append(f"missing_validation_report:{config.validation_report}")
    feature_columns: Optional[List[str]] = None
    for tier in TIERS:
        model_dir = model_dirs[tier]
        calibration_dir = calibration_dirs[tier]
        for filename in [
            "branch_site_neural_checkpoint.pt",
            "branch_site_neural_model_meta.json",
            "branch_site_neural_metrics.json",
        ]:
            if not (model_dir / filename).exists():
                blockers.append(f"missing_model_artifact:{tier}:{filename}")
        for filename in ["branch_site_calibration.json", "branch_site_calibrated_metrics.json"]:
            if not (calibration_dir / filename).exists():
                blockers.append(f"missing_calibration_artifact:{tier}:{filename}")
        meta_path = model_dir / "branch_site_neural_model_meta.json"
        if meta_path.exists():
            meta = _read_json(meta_path)
            columns = [str(column) for column in meta.get("feature_columns", [])]
            if not columns:
                blockers.append(f"missing_feature_columns:{tier}")
            elif feature_columns is None:
                feature_columns = columns
            elif columns != feature_columns:
                blockers.append(f"feature_columns_differ:{tier}")
            if meta.get("feature_policy") != "conservative_branch_site":
                blockers.append(f"model_feature_policy_mismatch:{tier}:{meta.get('feature_policy')}")
    return blockers


def _build_manifest(
    config: DeployableModelPackageConfig,
    methods: List[str],
    validation_report: Dict[str, Any],
    truth_audit_rows: List[Dict[str, str]],
    feature_columns: List[str],
    tier_models: Dict[str, Any],
    tier_calibrations: Dict[str, Any],
    artifacts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    report_identity = validation_report.get("run_identity", {})
    return {
        "package_name": "babappa_conservative_branch_site_100k_mps",
        "package_version": __version__,
        "babappa_version": __version__,
        "source_run": config.run_name,
        "model_type": "tier_aware_branch_site_mlp",
        "feature_policy": config.feature_policy,
        "truth_mode": config.truth_mode,
        "truth_source": "explicit_simulator_branch_truth",
        "methods_supported": methods,
        "saturation_tiers": TIERS,
        "tier_models": tier_models,
        "tier_calibrations": tier_calibrations,
        "model_dirs": [str(path) for path in _parse_csv(config.model_dirs)],
        "calibration_dirs": [str(path) for path in _parse_csv(config.calibration_dirs)],
        "artifacts": artifacts,
        "model_file_checksums": {
            item["relative_path"]: item["sha256"] for item in artifacts if item["kind"] == "model"
        },
        "calibration_file_checksums": {
            item["relative_path"]: item["sha256"] for item in artifacts if item["kind"] == "calibration"
        },
        "expected_feature_columns": feature_columns,
        "excluded_features": FORBIDDEN_EMPIRICAL_INPUT_COLUMNS,
        "input_requirements": {
            "requires_aligned_codon_features": True,
            "requires_branch_context_features": True,
            "requires_truth_columns_at_inference": False,
            "blocked_empirical_input_columns": FORBIDDEN_EMPIRICAL_INPUT_COLUMNS,
        },
        "ood_applicability_thresholds": {
            "status": "initial_scaffold_requires_empirical_calibration",
            "gap_fraction_warn_above": 0.30,
            "n_taxa_warn_below": 4,
            "n_codons_warn_below": 60,
            "saturation_tier_route": "match empirical saturation proxy to low/moderate/high/extreme tier model",
        },
        "calibration_thresholds_by_tier": {
            tier: {
                "temperature": tier_calibrations[tier].get("temperature"),
                "selected_threshold": tier_calibrations[tier].get("selected_threshold"),
                "target_fdr": tier_calibrations[tier].get("target_fdr"),
            }
            for tier in TIERS
        },
        "neural_metrics_by_tier": _metrics_by_tier(validation_report.get("neural_rows", []), "test"),
        "aggregation_metrics_by_tier": {
            "branch": _metrics_by_tier(validation_report.get("branch_aggregation_rows", []), "all"),
            "gene": _metrics_by_tier(validation_report.get("gene_aggregation_rows", []), "all"),
        },
        "control_metrics_by_tier": _controls_by_tier(validation_report.get("controls_rows", [])),
        "known_warnings": KNOWN_WARNINGS,
        "empirical_claim_status": "not_final_empirical_inference",
        "validation_decision": validation_report.get("decision", {}),
        "truth_audit": {
            "explicit_truth_available": all(row.get("explicit_branch_site_truth_available") == "True" for row in truth_audit_rows),
            "proxy_label_tiers": [
                row.get("tier", "") for row in truth_audit_rows if row.get("proxy_from_foreground_taxon") != "False"
            ],
            "rows": truth_audit_rows,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "source_run_identity": report_identity,
        "reproducibility_notes": [
            "Package is built from retained Cycle 39 artifacts after raw/intermediate 100K files were pruned.",
            "No new simulation, alignment, or neural training is performed by packaging.",
            "Empirical inference requires QC, OOD gates, simulation-matched calibration, and external benchmarks.",
        ],
    }


def _build_feature_schema(feature_policy: str, columns: List[str]) -> Dict[str, Any]:
    return {
        "schema_version": __version__,
        "feature_policy": feature_policy,
        "expected_feature_columns": columns,
        "columns": [{"name": column, "type": "float32"} for column in columns],
        "blocked_empirical_input_columns": FORBIDDEN_EMPIRICAL_INPUT_COLUMNS,
        "notes": [
            "Empirical inference inputs must be derived from alignment/taxon/branch context only.",
            "Oracle labels and simulator truth are forbidden as empirical inference inputs.",
        ],
    }


def _build_calibration_schema(tier_calibrations: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": __version__,
        "calibration_kind": "simulation_trained_temperature_and_threshold",
        "tiers": {
            tier: {
                "temperature": tier_calibrations[tier].get("temperature"),
                "selected_threshold": tier_calibrations[tier].get("selected_threshold"),
                "target_fdr": tier_calibrations[tier].get("target_fdr"),
            }
            for tier in TIERS
        },
        "empirical_status": "requires_simulation_matched_recalibration_before_claims",
    }


def _build_training_envelope(validation_report: Dict[str, Any], tier_models: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": __version__,
        "run_identity": validation_report.get("run_identity", {}),
        "stage_markers": validation_report.get("stage_markers", {}),
        "tier_model_metadata": {
            tier: {
                "n_features": info.get("n_features"),
                "hidden_dim": info.get("hidden_dim"),
                "dropout": info.get("dropout"),
                "device_used": info.get("device_used"),
            }
            for tier, info in tier_models.items()
        },
        "scientific_boundary": "simulation-supervised research-alpha; not final empirical branch-site inference",
    }


def _build_validation_summary(validation_report: Dict[str, Any], truth_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "schema_version": __version__,
        "decision": validation_report.get("decision", {}),
        "truth_audit": truth_rows,
        "tier_summary": validation_report.get("tier_summary", []),
        "scientific_cautions": validation_report.get("scientific_cautions", []),
    }


def _render_model_card(manifest: Dict[str, Any]) -> str:
    lines = [
        "# BABAPPA conservative_branch_site 100K MPS model card",
        "",
        "## Model name",
        "",
        "`babappa_conservative_branch_site_100k_mps`",
        "",
        "## Intended use",
        "",
        "Research-alpha simulation-supervised scoring of branch-site candidates after BABAPPA input QC, alignment ensemble, conservative feature extraction, OOD/applicability checks, and simulation-matched calibration.",
        "",
        "## Not intended use",
        "",
        "Not final empirical branch-site inference. Not a replacement for empirical calibration, external benchmark panels, or domain review.",
        "",
        "## Training data",
        "",
        "Simulation-trained on the completed conservative explicit branch-truth 100K Apple Silicon/MPS validation run.",
        "",
        "## Validation scale",
        "",
        "100,000 simulated families across low, moderate, high, and extreme saturation tiers.",
        "",
        "## Explicit branch-truth status",
        "",
        "Validation used explicit simulator branch-site truth. No simulator truth is used during empirical inference.",
        "",
        "## Feature policy",
        "",
        f"`{manifest['feature_policy']}`",
        "",
        "## Supported aligners",
        "",
        ", ".join(f"`{method}`" for method in manifest["methods_supported"]),
        "",
        "## Saturation-tier behavior",
        "",
        "Tier-aware package with low, moderate, high, and extreme checkpoints. Extreme remains strong but is the hardest tier.",
        "",
        "## Performance table",
        "",
        "| tier | AUROC | F1 | MCC | precision | recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tier in TIERS:
        row = manifest["neural_metrics_by_tier"].get(tier, {})
        lines.append(
            f"| {tier} | {_fmt(row.get('auroc'))} | {_fmt(row.get('f1'))} | {_fmt(row.get('mcc'))} | "
            f"{_fmt(row.get('precision'))} | {_fmt(row.get('recall'))} |"
        )
    lines.extend([
        "",
        "## Calibration table",
        "",
        "| tier | temperature | selected threshold | target FDR |",
        "| --- | ---: | ---: | ---: |",
    ])
    for tier in TIERS:
        row = manifest["calibration_thresholds_by_tier"].get(tier, {})
        lines.append(
            f"| {tier} | {_fmt(row.get('temperature'))} | {_fmt(row.get('selected_threshold'))} | {_fmt(row.get('target_fdr'))} |"
        )
    lines.extend([
        "",
        "## Controls interpretation",
        "",
        "Destructive controls support that branch-label randomization degrades signal, but controls are simulation-supervised and do not by themselves establish empirical validity.",
        "",
        "## Known limitations",
        "",
    ])
    for warning in manifest["known_warnings"]:
        lines.append(f"- {warning}")
    lines.extend([
        "",
        "## Empirical-use warning",
        "",
        "This package is simulation-trained and simulation-supervised. It is not final empirical branch-site inference.",
        "",
        "## Required empirical workflow",
        "",
        "input QC -> alignment ensemble -> feature extraction -> applicability/OOD check -> score -> simulation-matched calibration -> report",
        "",
        "## Citation/manuscript placeholder",
        "",
        "BABAPPA manuscript citation to be added after empirical benchmark validation.",
        "",
        "## Version and checksums",
        "",
        f"- BABAPPA version: `{manifest['babappa_version']}`",
        "- Checksums: `checksums.sha256`",
        "",
    ])
    return "\n".join(lines)


def _render_limitations() -> str:
    return "\n".join([
        "# Limitations",
        "",
        "- Simulation-trained and simulation-supervised only.",
        "- Explicit simulator branch-site truth was used for validation, not empirical inference.",
        "- Conditional pass reflects pruned raw/intermediate artifacts after retained validation outputs were preserved.",
        "- Foreground context columns remain a known leakage/OOD caution.",
        "- Context-only shortcut risk remains a required empirical calibration caution.",
        "- Final empirical branch-site inference claims are not supported by this package alone.",
        "",
    ])


def _render_readme(manifest: Dict[str, Any]) -> str:
    return "\n".join([
        "# BABAPPA deployable conservative_branch_site model package",
        "",
        "This package contains the retained tier checkpoints and calibration metadata from the completed 100K MPS validation.",
        "",
        "Status: simulation-trained research-alpha; not final empirical branch-site inference.",
        "",
        "Validate with:",
        "",
        "```bash",
        "babappa validate-deployable-model-package --package-dir deployable_model_conservative_branch_site_100k_mps",
        "```",
        "",
        "Smoke-load with:",
        "",
        "```bash",
        "babappa smoke-load-deployable-model --package-dir deployable_model_conservative_branch_site_100k_mps --device auto --outdir deployable_model_load_smoke",
        "```",
        "",
        f"Source run: `{manifest['source_run']}`",
        "",
    ])


def _render_validation_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Deployable model package validation",
        "",
        f"- status: `{report['status']}`",
        f"- failures: `{report['n_fail']}`",
        f"- warnings: `{report['n_warning']}`",
        "",
    ]
    if report["failures"]:
        lines.append("## Failures")
        lines.extend(f"- {failure}" for failure in report["failures"])
        lines.append("")
    if report["warnings"]:
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in report["warnings"])
        lines.append("")
    return "\n".join(lines)


def _smoke_payload(
    status: str,
    config: DeployableModelSmokeConfig,
    device: Optional[str],
    forward_pass: bool,
    metadata_only: bool,
    loaded_tiers: List[str],
    failures: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    return {
        "status": status,
        "package_dir": config.package_dir,
        "device": device,
        "forward_pass": forward_pass,
        "metadata_only": metadata_only,
        "loaded_tiers": loaded_tiers,
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _write_smoke_reports(outdir: Path, payload: Dict[str, Any]) -> None:
    _write_json(outdir / "deployable_model_load_smoke.json", payload)
    lines = [
        "# Deployable model load smoke",
        "",
        f"- status: `{payload['status']}`",
        f"- device: `{payload.get('device')}`",
        f"- forward pass: `{payload.get('forward_pass')}`",
        f"- metadata-only: `{payload.get('metadata_only')}`",
        f"- loaded tiers: `{','.join(payload.get('loaded_tiers') or [])}`",
        "",
    ]
    if payload.get("warnings"):
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in payload["warnings"])
        lines.append("")
    if payload.get("failures"):
        lines.append("## Failures")
        lines.extend(f"- {failure}" for failure in payload["failures"])
        lines.append("")
    (outdir / "deployable_model_load_smoke.md").write_text("\n".join(lines), encoding="utf-8")


def _parse_csv(value: Sequence[str] | str) -> List[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    parsed: List[str] = []
    for item in value:
        parsed.extend(part.strip() for part in str(item).split(",") if part.strip())
    return parsed


def _tier_map(paths: List[str], label: str) -> Dict[str, Path]:
    if len(paths) != 4:
        raise ValueError(f"{label} must contain exactly four tier directories")
    mapped: Dict[str, Path] = {}
    for index, raw in enumerate(paths):
        path = Path(raw)
        tier = _infer_tier(path) or TIERS[index]
        if tier in mapped:
            raise ValueError(f"duplicate tier in {label}: {tier}")
        mapped[tier] = path
    missing = sorted(set(TIERS) - set(mapped))
    if missing:
        raise ValueError(f"{label} missing tiers: {','.join(missing)}")
    return mapped


def _infer_tier(path: Path) -> Optional[str]:
    text = path.name.lower()
    for tier in TIERS:
        if f"_{tier}_" in text or text.endswith(f"_{tier}") or text.endswith(f"_{tier}_streamed"):
            return tier
    return None


def _copy_artifact(source: Path, target: Path, outdir: Path, tier: str, kind: str) -> Dict[str, Any]:
    shutil.copy2(source, target)
    rel = target.relative_to(outdir)
    return {
        "tier": tier,
        "kind": kind,
        "source_path": str(source),
        "relative_path": str(rel),
        "size_bytes": target.stat().st_size,
        "sha256": _sha256(target),
    }


def _self_artifacts(outdir: Path) -> List[Dict[str, Any]]:
    artifacts = []
    for filename in [
        "model_manifest.json",
        "model_card.md",
        "feature_schema.json",
        "calibration_schema.json",
        "training_envelope.json",
        "validation_summary.json",
        "limitations.md",
        "README.md",
    ]:
        path = outdir / filename
        artifacts.append({
            "tier": "",
            "kind": "package_metadata",
            "source_path": str(path),
            "relative_path": filename,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return artifacts


def _write_checksums(path: Path, artifacts: List[Dict[str, Any]]) -> None:
    lines = [f"{artifact['sha256']}  {artifact['relative_path']}" for artifact in artifacts]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metrics_by_tier(rows: Iterable[Dict[str, Any]], split: str) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("tier")): row
        for row in rows
        if str(row.get("tier")) in TIERS and str(row.get("split")) == split
    }


def _controls_by_tier(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_tier: Dict[str, List[Dict[str, Any]]] = {tier: [] for tier in TIERS}
    for row in rows:
        tier = str(row.get("tier"))
        if tier in by_tier:
            by_tier[tier].append(row)
    return by_tier


def _blocker_payload(config: DeployableModelPackageConfig, blockers: List[str]) -> Dict[str, Any]:
    return {
        "status": "blocked",
        "version": __version__,
        "source_run": config.run_name,
        "blockers": blockers,
        "scientific_boundary": "Packaging stopped before creating a deployable bundle.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _render_blocker_md(payload: Dict[str, Any]) -> str:
    lines = ["# Deployable model packaging blocker report", "", f"- status: `{payload['status']}`", ""]
    for blocker in payload["blockers"]:
        lines.append(f"- {blocker}")
    lines.append("")
    return "\n".join(lines)


def _find_forbidden_package_files(package_dir: Path) -> List[str]:
    forbidden: List[str] = []
    for path in package_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(package_dir)).lower()
        basename = path.name.lower()
        if basename in {"branch_site_truth.tsv", "branch_truth.json", "truth.json"}:
            forbidden.append(str(path.relative_to(package_dir)))
        elif "selected_sites" in rel or "/oracle" in rel or rel.startswith("oracle"):
            forbidden.append(str(path.relative_to(package_dir)))
    return forbidden


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return data


def _read_optional_json(path: Path, failures: List[str]) -> Dict[str, Any]:
    if not path.exists():
        failures.append(f"missing_file:{path.name}")
        return {}
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        failures.append(f"invalid_json:{path.name}:{exc}")
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(torch: Any, path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""
