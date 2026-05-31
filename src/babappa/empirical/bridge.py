"""Empirical validation, feature extraction, scoring, and report scaffolds."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
from importlib import resources
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from babappa import __version__
from babappa.align.backends import babappalign_model_status, detect_aligner_backends
from babappa.align.ensemble import write_fasta
from babappa.align.site_map import build_site_map_for_alignment
from babappa.datasets.index import read_tsv, write_tsv
from babappa.simulate.audit import read_fasta
from babappa.training.neural_env import resolve_torch_device, safe_import_torch

TIERS = ["low", "moderate", "high", "extreme"]
METHODS = ["identity", "mafft", "babappalign", "muscle"]
STOP_CODONS = {"TAA", "TAG", "TGA"}
FEATURE_OUTPUT_FIELDS = [
    "family_id",
    "method",
    "branch_id",
    "foreground_taxon",
    "aligned_site_index_zero",
    "original_site_index_zero",
    "mapping_status",
]
FORBIDDEN_EMPIRICAL_INPUT_COLUMNS = [
    "branch_site_truth",
    "selected_sites",
    "truth",
    "branch_truth",
    "oracle",
    "y_branch_site",
    "y_site",
    "gene_label",
    "n_selected_sites",
    "positive_label",
    "simulated_label",
]


@dataclass(frozen=True)
class EmpiricalInputValidationConfig:
    """Configuration for empirical CDS/tree input validation."""

    cds_fasta: str
    tree: str
    foreground: str
    outdir: str
    allow_stop_codons: bool = False
    min_taxa: int = 3
    min_codons: int = 3


@dataclass(frozen=True)
class EmpiricalAlignmentEnsembleConfig:
    """Configuration for tiny empirical alignment ensemble runs."""

    cds_fasta: str
    tree: str
    foreground: str
    outdir: str
    methods: Sequence[str] | str = tuple(METHODS)
    require_babappalign: bool = True
    threads: int = 4
    timeout_seconds: int = 120


@dataclass(frozen=True)
class EmpiricalFeatureExtractionConfig:
    """Configuration for empirical branch-site feature extraction."""

    empirical_validation_dir: str
    alignment_dir: str
    deployable_model_package: str
    outdir: str
    foreground: str


@dataclass(frozen=True)
class EmpiricalFeatureAuditConfig:
    """Configuration for empirical feature safety audit."""

    features: str
    deployable_model_package: str
    outdir: str


@dataclass(frozen=True)
class EmpiricalApplicabilityConfig:
    """Configuration for empirical applicability/OOD scoring."""

    empirical_validation_dir: str
    empirical_feature_dir: str
    deployable_model_package: str
    outdir: str


@dataclass(frozen=True)
class EmpiricalBranchSiteScoringConfig:
    """Configuration for deployable model empirical scoring."""

    features: str
    deployable_model_package: str
    applicability_dir: str
    outdir: str
    device: str = "auto"


@dataclass(frozen=True)
class EmpiricalBranchSiteReportConfig:
    """Configuration for empirical branch-site report generation."""

    outdir: str
    empirical_validation_dir: str
    alignment_dir: str
    feature_dir: str
    feature_audit_dir: str
    applicability_dir: str
    scoring_dir: str
    simulation_matched_calibration_plan: str
    deployable_model_package: str


@dataclass(frozen=True)
class DirectBranchSitePredictionConfig:
    """Configuration for direct user MSA/tree branch-site prediction."""

    msa: str
    tree: str
    outdir: str
    foreground: str = "all"
    model_package: str = "deployable_model_conservative_branch_site_100k_mps"
    device: str = "auto"
    allow_stop_codons: bool = False
    min_taxa: int = 3
    min_codons: int = 3
    dry_run: bool = False


@dataclass(frozen=True)
class ExternalBenchmarkPanelPlanConfig:
    """Configuration for external benchmark panel planning."""

    panel_manifest: str
    deployable_model_package: str
    outdir: str
    methods: Sequence[str] | str = tuple(METHODS)
    classical_tools: Sequence[str] | str = ("codeml", "hyphy")


def validate_empirical_input(config: EmpiricalInputValidationConfig) -> Dict[str, Any]:
    """Validate a tiny empirical CDS FASTA/tree pair and emit QC summaries."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fasta_path = Path(config.cds_fasta)
    tree_path = Path(config.tree)
    failures: List[str] = []
    warnings: List[str] = []
    records, duplicate_ids = _read_fasta_with_duplicates(fasta_path)
    if not records:
        failures.append("no_fasta_records")
    if duplicate_ids:
        failures.append("duplicate_ids:" + ",".join(sorted(duplicate_ids)))
    tree_text = tree_path.read_text(encoding="utf-8") if tree_path.exists() else ""
    if not tree_text:
        failures.append(f"missing_or_empty_tree:{tree_path}")
    tree_tips = _parse_newick_tips(tree_text)
    fasta_ids = set(records)
    missing_in_tree = sorted(fasta_ids - tree_tips)
    missing_in_fasta = sorted(tree_tips - fasta_ids)
    if missing_in_tree:
        failures.append("fasta_ids_missing_from_tree:" + ",".join(missing_in_tree))
    if missing_in_fasta:
        failures.append("tree_tips_missing_from_fasta:" + ",".join(missing_in_fasta))
    if config.foreground not in fasta_ids:
        failures.append(f"foreground_missing_from_fasta:{config.foreground}")
    if config.foreground not in tree_tips and tree_tips:
        failures.append(f"foreground_missing_from_tree:{config.foreground}")
    if len(records) < config.min_taxa:
        failures.append(f"too_few_taxa:{len(records)}<{config.min_taxa}")

    lengths = {taxon: len(seq) for taxon, seq in records.items()}
    if len(set(lengths.values())) > 1:
        warnings.append("unequal_sequence_lengths_unaligned_input")
    for taxon, sequence in records.items():
        if len(sequence) % 3 != 0:
            failures.append(f"frameshift_length_not_divisible_by_3:{taxon}:{len(sequence)}")
        codons = _codons(sequence)
        if len(codons) < config.min_codons:
            failures.append(f"too_few_codons:{taxon}:{len(codons)}<{config.min_codons}")
        stops = [
            index for index, codon in enumerate(codons[:-1])
            if codon.upper().replace("U", "T") in STOP_CODONS
        ]
        if stops and not config.allow_stop_codons:
            failures.append(f"premature_stop_codon:{taxon}:{','.join(str(i) for i in stops)}")

    n_taxa = len(records)
    n_codons = min((len(seq) // 3 for seq in records.values()), default=0)
    ambiguous_fraction = _ambiguous_fraction(records)
    gap_fraction = _gap_fraction(records)
    p_distance = _mean_pairwise_p_distance(records)
    saturation_proxy = _saturation_proxy(p_distance)
    if ambiguous_fraction > 0.05:
        warnings.append(f"high_ambiguous_base_fraction:{ambiguous_fraction:.6g}")
    if gap_fraction > 0.20:
        warnings.append(f"high_gap_fraction:{gap_fraction:.6g}")
    if p_distance > 0.30:
        warnings.append(f"high_pairwise_p_distance:{p_distance:.6g}")

    status = "fail" if failures else ("warning" if warnings else "pass")
    payload = {
        "empirical_input_validation_version": __version__,
        "status": status,
        "empirical_validation_status": status,
        "cds_fasta": str(fasta_path),
        "tree": str(tree_path),
        "foreground": config.foreground,
        "n_taxa": n_taxa,
        "n_codons": n_codons,
        "tree_tips": sorted(tree_tips),
        "fasta_ids": sorted(fasta_ids),
        "ambiguous_base_fraction": ambiguous_fraction,
        "gap_fraction": gap_fraction,
        "mean_pairwise_p_distance": p_distance,
        "saturation_proxy": saturation_proxy,
        "foreground_taxon": config.foreground,
        "tree_shape_summary": _tree_shape_summary(tree_text, tree_tips),
        "failures": failures,
        "warnings": warnings,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(outdir / "empirical_input_validation.json", payload)
    write_tsv(
        outdir / "empirical_input_validation.tsv",
        [_flat_validation_row(payload)],
        [
            "status",
            "foreground",
            "n_taxa",
            "n_codons",
            "ambiguous_base_fraction",
            "gap_fraction",
            "mean_pairwise_p_distance",
            "saturation_proxy",
            "failures",
            "warnings",
        ],
    )
    (outdir / "empirical_input_validation.md").write_text(_render_empirical_input_md(payload), encoding="utf-8")
    return {
        "status": status,
        "outdir": str(outdir),
        "json": str(outdir / "empirical_input_validation.json"),
        "tsv": str(outdir / "empirical_input_validation.tsv"),
        "markdown": str(outdir / "empirical_input_validation.md"),
        "n_taxa": n_taxa,
        "n_codons": n_codons,
        "failures": failures,
        "warnings": warnings,
    }


def run_empirical_alignment_ensemble(config: EmpiricalAlignmentEnsembleConfig) -> Dict[str, Any]:
    """Run a tiny empirical alignment ensemble and build site maps/policy."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    methods = _parse_csv(config.methods)
    source_fasta = Path(config.cds_fasta).resolve()
    records = read_fasta(source_fasta)
    expected_taxa = set(records)
    backends = detect_aligner_backends()
    babappalign_status = babappalign_model_status()
    warnings: List[str] = []
    failures: List[str] = []
    if "babappalign" in methods and config.require_babappalign and not babappalign_status["model_present"]:
        failures.append("babappalign_model_missing:" + str(babappalign_status["model_expected_path"]))

    method_rows: List[Dict[str, Any]] = []
    created_files: Dict[str, Dict[str, str]] = {}
    site_map_dir = outdir / "site_map"
    policy_dir = outdir / "method_policy"
    site_map_dir.mkdir(parents=True, exist_ok=True)
    policy_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        method_outdir = outdir / "methods" / method
        method_outdir.mkdir(parents=True, exist_ok=True)
        aligned_fasta = method_outdir / f"empirical.{method}.codon.fasta"
        qc_path = method_outdir / f"empirical.{method}.qc.json"
        started = time.monotonic()
        status, reason, command = _run_empirical_method(
            method,
            source_fasta,
            aligned_fasta,
            config.timeout_seconds,
        )
        runtime = time.monotonic() - started
        validation = _validate_aligned_fasta(aligned_fasta, expected_taxa)
        if status == "ok" and validation["status"] != "ok":
            status = "fail"
            reason = ";".join(validation["failures"])
        if status == "ok":
            site_rows = build_site_map_for_alignment(source_fasta, aligned_fasta, family_id="empirical", method=method)
            site_map_path = site_map_dir / f"{method}.site_map.tsv"
            write_tsv(site_map_path, site_rows, _site_map_fields(site_rows))
            created_files[method] = {
                "codon_fasta": str(aligned_fasta),
                "qc": str(qc_path),
                "site_map": str(site_map_path),
            }
        elif method == "babappalign" and config.require_babappalign:
            failures.append(f"method_failed:{method}:{reason}")
        else:
            warnings.append(f"method_failed:{method}:{reason}")
        qc = {
            "method": method,
            "status": status,
            "reason": reason,
            "command": command,
            "runtime_seconds": runtime,
            "validation": validation,
        }
        _write_json(qc_path, qc)
        method_rows.append({
            "method": method,
            "status": status,
            "reason": reason,
            "runtime_seconds": f"{runtime:.6f}",
            "aligned_fasta": str(aligned_fasta) if aligned_fasta.exists() else "",
            "site_map": created_files.get(method, {}).get("site_map", ""),
        })

    policy_payload = _build_empirical_method_policy(method_rows, site_map_dir, policy_dir)
    status = "fail" if failures else ("warning" if warnings else "ok")
    manifest = {
        "empirical_alignment_version": __version__,
        "status": status,
        "cds_fasta": config.cds_fasta,
        "tree": config.tree,
        "foreground": config.foreground,
        "methods_requested": methods,
        "methods_run": [row["method"] for row in method_rows if row["status"] == "ok"],
        "method_rows": method_rows,
        "created_files": created_files,
        "site_map_dir": str(site_map_dir),
        "method_policy_dir": str(policy_dir),
        "method_policy": policy_payload,
        "babappalign_model_status": babappalign_status,
        "backend_status": {name: backend.as_dict() for name, backend in backends.items() if name in methods},
        "failures": failures,
        "warnings": warnings,
    }
    _write_json(outdir / "empirical_alignment_manifest.json", manifest)
    write_tsv(outdir / "empirical_alignment_summary.tsv", method_rows, ["method", "status", "reason", "runtime_seconds", "aligned_fasta", "site_map"])
    (outdir / "empirical_alignment_report.md").write_text(_render_alignment_report(manifest), encoding="utf-8")
    return {
        "status": status,
        "outdir": str(outdir),
        "methods": methods,
        "methods_run": manifest["methods_run"],
        "manifest": str(outdir / "empirical_alignment_manifest.json"),
        "report": str(outdir / "empirical_alignment_report.md"),
        "failures": failures,
        "warnings": warnings,
    }


def extract_empirical_branch_site_features(config: EmpiricalFeatureExtractionConfig) -> Dict[str, Any]:
    """Extract empirical branch-site feature rows matching the deployable schema."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    package_dir = Path(config.deployable_model_package)
    feature_schema = _read_json(package_dir / "feature_schema.json")
    expected_features = [str(item) for item in feature_schema.get("expected_feature_columns", [])]
    if not expected_features:
        raise ValueError("deployable package feature_schema.json has no expected_feature_columns")
    validation = _read_json(Path(config.empirical_validation_dir) / "empirical_input_validation.json")
    align_manifest = _read_json(Path(config.alignment_dir) / "empirical_alignment_manifest.json")
    rows: List[Dict[str, Any]] = []
    missing_features: List[str] = []
    for method in align_manifest.get("methods_run", []):
        files = align_manifest.get("created_files", {}).get(method, {})
        aligned_fasta = files.get("codon_fasta")
        site_map_file = files.get("site_map")
        if not aligned_fasta or not site_map_file:
            continue
        records = read_fasta(Path(aligned_fasta))
        site_rows = read_tsv(Path(site_map_file))
        for site_row in site_rows:
            if site_row.get("mapping_status") not in {"unique", "conflict"}:
                continue
            for branch_id in records:
                feature_values = _feature_row_from_site(
                    records,
                    site_row,
                    branch_id,
                    config.foreground,
                    validation,
                    expected_features,
                )
                row = {
                    "family_id": "empirical",
                    "method": method,
                    "branch_id": branch_id,
                    "foreground_taxon": config.foreground,
                    "aligned_site_index_zero": site_row.get("aligned_site_index_zero", ""),
                    "original_site_index_zero": site_row.get("original_site_index_zero", ""),
                    "mapping_status": site_row.get("mapping_status", ""),
                }
                row.update(feature_values)
                missing_features.extend([name for name in expected_features if row.get(name) in {None, ""}])
                rows.append(row)
    forbidden = _forbidden_columns(FEATURE_OUTPUT_FIELDS + expected_features)
    schema_match = "pass" if rows and not missing_features and not forbidden else "fail"
    fields = FEATURE_OUTPUT_FIELDS + expected_features
    write_tsv(outdir / "empirical_branch_site_features.tsv", rows, fields)
    schema_check = {
        "status": schema_match,
        "feature_schema_match": schema_match,
        "expected_feature_columns": expected_features,
        "n_rows": len(rows),
        "missing_features": sorted(set(missing_features)),
        "forbidden_columns": forbidden,
    }
    _write_json(outdir / "empirical_feature_schema_check.json", schema_check)
    manifest = {
        "empirical_feature_version": __version__,
        "status": schema_match,
        "feature_policy": feature_schema.get("feature_policy"),
        "n_rows": len(rows),
        "foreground": config.foreground,
        "alignment_dir": config.alignment_dir,
        "deployable_model_package": config.deployable_model_package,
        "feature_schema_check": schema_check,
        "truth_derived_inputs_excluded": True,
        "generated_files": {
            "features": str(outdir / "empirical_branch_site_features.tsv"),
            "schema_check": str(outdir / "empirical_feature_schema_check.json"),
            "report": str(outdir / "empirical_feature_report.md"),
        },
    }
    _write_json(outdir / "empirical_feature_manifest.json", manifest)
    (outdir / "empirical_feature_report.md").write_text(_render_feature_report(manifest), encoding="utf-8")
    if schema_match != "pass":
        raise ValueError("empirical feature schema check failed: " + ",".join(schema_check["missing_features"] + forbidden))
    return {
        "status": "ok",
        "outdir": str(outdir),
        "features": str(outdir / "empirical_branch_site_features.tsv"),
        "rows": len(rows),
        "schema_match": schema_match,
        "forbidden_columns": forbidden,
    }


def audit_empirical_features(config: EmpiricalFeatureAuditConfig) -> Dict[str, Any]:
    """Audit empirical features for forbidden truth-derived columns."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = read_tsv(Path(config.features))
    columns = list(rows[0].keys()) if rows else _read_header(Path(config.features))
    forbidden = _forbidden_columns(columns)
    status = "fail" if forbidden else "ok"
    audit_rows = [{"check": "forbidden_columns", "status": status, "value": ",".join(forbidden)}]
    payload = {
        "empirical_feature_audit_version": __version__,
        "status": status,
        "features": config.features,
        "n_rows": len(rows),
        "columns": columns,
        "forbidden_columns": forbidden,
        "deployable_model_package": config.deployable_model_package,
    }
    _write_json(outdir / "empirical_feature_audit.json", payload)
    write_tsv(outdir / "empirical_feature_audit.tsv", audit_rows, ["check", "status", "value"])
    (outdir / "empirical_feature_audit.md").write_text(_render_feature_audit_md(payload), encoding="utf-8")
    return {
        "status": status,
        "outdir": str(outdir),
        "json": str(outdir / "empirical_feature_audit.json"),
        "forbidden_columns": forbidden,
    }


def run_empirical_applicability(config: EmpiricalApplicabilityConfig) -> Dict[str, Any]:
    """Run first-pass rule-based empirical applicability/OOD classification."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    validation = _read_json(Path(config.empirical_validation_dir) / "empirical_input_validation.json")
    feature_check = _read_json(Path(config.empirical_feature_dir) / "empirical_feature_schema_check.json")
    package_dir = Path(config.deployable_model_package)
    training_envelope = _read_json(package_dir / "training_envelope.json")
    reasons: List[str] = []
    level = "in_domain"
    n_taxa = _as_float(validation.get("n_taxa"))
    n_codons = _as_float(validation.get("n_codons"))
    p_distance = _as_float(validation.get("mean_pairwise_p_distance"))
    p_distance_source = "input_cds_positional"
    alignment_p_distance = _alignment_mean_pairwise_p_distance(Path(config.empirical_feature_dir))
    if "unequal_sequence_lengths_unaligned_input" in set(validation.get("warnings") or []) and alignment_p_distance is not None:
        p_distance = alignment_p_distance
        p_distance_source = "alignment_ensemble_mean"
    gap_fraction = _as_float(validation.get("gap_fraction"))
    ambiguous_fraction = _as_float(validation.get("ambiguous_base_fraction"))
    if n_taxa is not None and n_taxa < 4:
        level = _max_level(level, "borderline")
        reasons.append(f"low_taxa:{int(n_taxa)}")
    if n_codons is not None and n_codons < 60:
        level = _max_level(level, "borderline")
        reasons.append(f"short_alignment:{int(n_codons)}")
    if p_distance is not None and p_distance > 0.35:
        level = "out_of_domain"
        reasons.append(f"very_high_p_distance:{p_distance:.6g}")
    elif p_distance is not None and p_distance > 0.25:
        level = _max_level(level, "borderline")
        reasons.append(f"high_p_distance:{p_distance:.6g}")
    if gap_fraction is not None and gap_fraction > 0.40:
        level = "out_of_domain"
        reasons.append(f"very_high_gap_fraction:{gap_fraction:.6g}")
    elif gap_fraction is not None and gap_fraction > 0.20:
        level = _max_level(level, "borderline")
        reasons.append(f"high_gap_fraction:{gap_fraction:.6g}")
    if ambiguous_fraction is not None and ambiguous_fraction > 0.10:
        level = "out_of_domain"
        reasons.append(f"high_ambiguous_fraction:{ambiguous_fraction:.6g}")
    if feature_check.get("feature_schema_match") != "pass":
        level = "out_of_domain"
        reasons.append("feature_schema_mismatch")
    if not reasons:
        reasons.append("all_rule_based_checks_passed")
    recommended_tier = _recommended_tier(p_distance)
    payload = {
        "empirical_applicability_version": __version__,
        "status": level,
        "applicability_status": level,
        "recommended_tier": recommended_tier,
        "reasons": reasons,
        "validation": {
            "n_taxa": validation.get("n_taxa"),
            "n_codons": validation.get("n_codons"),
            "mean_pairwise_p_distance": validation.get("mean_pairwise_p_distance"),
            "alignment_mean_pairwise_p_distance": alignment_p_distance,
            "p_distance_used": p_distance,
            "p_distance_source": p_distance_source,
            "saturation_proxy": validation.get("saturation_proxy"),
            "gap_fraction": validation.get("gap_fraction"),
            "ambiguous_base_fraction": validation.get("ambiguous_base_fraction"),
        },
        "feature_rows": feature_check.get("n_rows"),
        "training_envelope_available": bool(training_envelope),
        "feature_distribution_range_check": "not_available_in_package_v1",
        "diagnostic_only_if_scored": level == "out_of_domain",
    }
    _write_json(outdir / "empirical_applicability.json", payload)
    write_tsv(outdir / "empirical_applicability.tsv", [_flat_applicability_row(payload)], ["status", "recommended_tier", "reasons", "feature_rows"])
    (outdir / "empirical_applicability.md").write_text(_render_applicability_md(payload), encoding="utf-8")
    return {
        "status": level,
        "outdir": str(outdir),
        "json": str(outdir / "empirical_applicability.json"),
        "reasons": reasons,
        "recommended_tier": recommended_tier,
    }


def score_empirical_branch_sites(config: EmpiricalBranchSiteScoringConfig) -> Dict[str, Any]:
    """Score empirical branch-site rows with the packaged deployable model."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch, error = safe_import_torch()
    if torch is None:
        payload = {
            "status": "fail",
            "reason": f"torch_unavailable:{error}",
            "message": "PyTorch is required for empirical scoring; metadata-only scoring is not allowed.",
        }
        _write_json(outdir / "empirical_scoring_manifest.json", payload)
        (outdir / "empirical_scoring_report.md").write_text(_render_scoring_failure_md(payload), encoding="utf-8")
        raise RuntimeError(payload["message"] + " " + payload["reason"])
    package_dir = _resolve_deployable_package_path(config.deployable_model_package)
    manifest = _read_json(package_dir / "model_manifest.json")
    feature_schema = _read_json(package_dir / "feature_schema.json")
    applicability = _read_json(Path(config.applicability_dir) / "empirical_applicability.json")
    rows = read_tsv(Path(config.features))
    if not rows:
        raise ValueError("empirical feature table has no rows")
    feature_columns = [str(column) for column in feature_schema.get("expected_feature_columns", [])]
    tier = str(applicability.get("recommended_tier") or "low")
    if tier not in TIERS:
        tier = "low"
    device = resolve_torch_device(torch, config.device)
    model_info = manifest["tier_models"][tier]
    calibration = manifest["calibration_thresholds_by_tier"][tier]
    from babappa.site.neural_model import SiteMLPClassifier

    checkpoint = _torch_load(torch, package_dir / model_info["checkpoint"])
    model = SiteMLPClassifier(
        input_dim=len(feature_columns),
        hidden_dim=int(model_info.get("hidden_dim") or 64),
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    X = np.array([[_as_float(row.get(column), 0.0) or 0.0 for column in feature_columns] for row in rows], dtype=np.float32)
    mean = np.asarray(checkpoint.get("feature_mean", np.zeros(len(feature_columns))), dtype=np.float32)
    std = np.asarray(checkpoint.get("feature_std", np.ones(len(feature_columns))), dtype=np.float32)
    std = np.where(std == 0, 1.0, std)
    X = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        tensor = torch.from_numpy(X).to(device)
        logits = model(tensor)
        temperature = float(calibration.get("temperature") or 1.0)
        probs = torch.sigmoid(logits / max(temperature, 1e-6)).detach().cpu().numpy()
    threshold = float(calibration.get("selected_threshold") or 0.5)
    diagnostic_only = applicability.get("applicability_status") == "out_of_domain"
    score_rows: List[Dict[str, Any]] = []
    for row, prob in zip(rows, probs):
        score_rows.append({
            "family_id": row.get("family_id", "empirical"),
            "method": row.get("method", ""),
            "branch_id": row.get("branch_id", ""),
            "foreground_taxon": row.get("foreground_taxon", ""),
            "aligned_site_index_zero": row.get("aligned_site_index_zero", ""),
            "original_site_index_zero": row.get("original_site_index_zero", ""),
            "prob_positive": float(prob),
            "called_positive": int(float(prob) >= threshold),
            "tier_model": tier,
            "calibrated_threshold": threshold,
            "diagnostic_only": diagnostic_only,
        })
    branch_rows = _aggregate_scores(score_rows, ["family_id", "method", "branch_id"])
    gene_rows = _aggregate_scores(score_rows, ["family_id", "method"])
    write_tsv(outdir / "empirical_branch_site_scores.tsv", score_rows, list(score_rows[0]))
    write_tsv(outdir / "empirical_branch_scores.tsv", branch_rows, list(branch_rows[0]) if branch_rows else ["family_id"])
    write_tsv(outdir / "empirical_gene_support.tsv", gene_rows, list(gene_rows[0]) if gene_rows else ["family_id"])
    payload = {
        "empirical_scoring_version": __version__,
        "status": "ok",
        "device": str(device),
        "tier_model": tier,
        "n_rows": len(score_rows),
        "diagnostic_only": diagnostic_only,
        "applicability_status": applicability.get("applicability_status"),
        "calibration": calibration,
        "outputs": {
            "branch_site_scores": str(outdir / "empirical_branch_site_scores.tsv"),
            "branch_scores": str(outdir / "empirical_branch_scores.tsv"),
            "gene_support": str(outdir / "empirical_gene_support.tsv"),
        },
    }
    _write_json(outdir / "empirical_scoring_manifest.json", payload)
    (outdir / "empirical_scoring_report.md").write_text(_render_scoring_report(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "device": str(device),
        "tier_model": tier,
        "diagnostic_only": diagnostic_only,
        "n_rows": len(score_rows),
    }


def make_empirical_branch_site_report(config: EmpiricalBranchSiteReportConfig) -> Dict[str, Any]:
    """Assemble a guarded empirical branch-site smoke report."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    validation = _read_json(Path(config.empirical_validation_dir) / "empirical_input_validation.json")
    alignment = _read_json(Path(config.alignment_dir) / "empirical_alignment_manifest.json")
    feature_manifest = _read_json(Path(config.feature_dir) / "empirical_feature_manifest.json")
    feature_audit = _read_json(Path(config.feature_audit_dir) / "empirical_feature_audit.json")
    applicability = _read_json(Path(config.applicability_dir) / "empirical_applicability.json")
    scoring = _read_optional_json(Path(config.scoring_dir) / "empirical_scoring_manifest.json")
    calibration_plan = _read_json(Path(config.simulation_matched_calibration_plan) / "simulation_matched_calibration_plan.json")
    package_manifest = _read_json(Path(config.deployable_model_package) / "model_manifest.json")
    payload = {
        "empirical_report_version": __version__,
        "status": "ok",
        "no_simulator_truth_used": True,
        "model_is_simulation_trained": True,
        "not_final_empirical_inference": True,
        "external_validation_recommended": "codeml/HyPhy-style workflows",
        "deployable_model_package": config.deployable_model_package,
        "package_name": package_manifest.get("package_name"),
        "input_validation": validation,
        "alignment": {
            "status": alignment.get("status"),
            "methods_run": alignment.get("methods_run"),
            "method_policy": alignment.get("method_policy"),
        },
        "feature_extraction": {
            "status": feature_manifest.get("status"),
            "n_rows": feature_manifest.get("n_rows"),
            "audit_status": feature_audit.get("status"),
            "forbidden_columns": feature_audit.get("forbidden_columns"),
        },
        "applicability": applicability,
        "scoring": scoring,
        "simulation_matched_calibration_plan": calibration_plan,
        "limitations": [
            "scores are not final empirical claims unless calibration and OOD checks pass",
            "external validation with codeml/HyPhy-style workflows is recommended",
            "tiny smoke data are for infrastructure validation only",
        ],
    }
    _write_json(outdir / "empirical_branch_site_report.json", payload)
    write_tsv(
        outdir / "empirical_branch_site_report.tsv",
        [_flat_report_row(payload)],
        ["status", "no_simulator_truth_used", "applicability_status", "scoring_status", "not_final_empirical_inference"],
    )
    (outdir / "empirical_branch_site_report.md").write_text(_render_empirical_report_md(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "json": str(outdir / "empirical_branch_site_report.json"),
        "markdown": str(outdir / "empirical_branch_site_report.md"),
        "no_simulator_truth_used": True,
    }


def predict_branch_sites(config: DirectBranchSitePredictionConfig) -> Dict[str, Any]:
    """Score a user-supplied codon MSA/tree without realigning it.

    This is the simple end-user path: the supplied MSA is treated as the
    authoritative alignment, site maps are identity maps on that MSA, and the
    deployable branch-site model scores requested foreground branches.
    """

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    msa_path = Path(config.msa)
    tree_path = Path(config.tree)
    records, duplicate_ids = _read_fasta_with_duplicates(msa_path)
    if not records:
        raise ValueError(f"MSA FASTA has no records: {msa_path}")
    if duplicate_ids:
        raise ValueError("MSA FASTA contains duplicate IDs: " + ",".join(sorted(duplicate_ids)))
    tree_text = tree_path.read_text(encoding="utf-8") if tree_path.exists() else ""
    if not tree_text:
        raise ValueError(f"tree file is missing or empty: {tree_path}")
    tree_tips = _parse_newick_tips(tree_text)
    _validate_direct_msa(records, tree_tips, config.min_taxa, config.min_codons)
    foregrounds = _resolve_direct_foregrounds(config.foreground, records, tree_tips)
    validation_foreground = foregrounds[0]
    package_dir = _resolve_deployable_package_path(config.model_package)

    input_dir = outdir / "input_validation"
    validation_summary = validate_empirical_input(
        EmpiricalInputValidationConfig(
            cds_fasta=str(msa_path),
            tree=str(tree_path),
            foreground=validation_foreground,
            outdir=str(input_dir),
            allow_stop_codons=config.allow_stop_codons,
            min_taxa=config.min_taxa,
            min_codons=config.min_codons,
        )
    )
    if validation_summary.get("status") == "fail":
        raise ValueError("input validation failed: " + ";".join(validation_summary.get("failures") or []))

    alignment_dir = outdir / "user_msa"
    _write_user_msa_alignment_artifacts(
        msa_path=msa_path,
        tree_path=tree_path,
        records=records,
        foreground_requested=config.foreground,
        foregrounds=foregrounds,
        outdir=alignment_dir,
    )

    feature_dir = outdir / "features"
    _extract_direct_branch_site_features(
        validation_dir=input_dir,
        alignment_dir=alignment_dir,
        deployable_model_package=package_dir,
        outdir=feature_dir,
        foregrounds=foregrounds,
    )
    audit_dir = outdir / "feature_audit"
    audit_summary = audit_empirical_features(
        EmpiricalFeatureAuditConfig(
            features=str(feature_dir / "empirical_branch_site_features.tsv"),
            deployable_model_package=str(package_dir),
            outdir=str(audit_dir),
        )
    )
    if audit_summary.get("status") != "ok":
        raise ValueError("empirical feature safety audit failed")
    applicability_dir = outdir / "applicability"
    applicability_summary = run_empirical_applicability(
        EmpiricalApplicabilityConfig(
            empirical_validation_dir=str(input_dir),
            empirical_feature_dir=str(feature_dir),
            deployable_model_package=str(package_dir),
            outdir=str(applicability_dir),
        )
    )

    status = "dry_run" if config.dry_run else "ok"
    scoring_summary: Dict[str, Any] = {}
    if not config.dry_run:
        scores_dir = outdir / "scores"
        scoring_summary = score_empirical_branch_sites(
            EmpiricalBranchSiteScoringConfig(
                features=str(feature_dir / "empirical_branch_site_features.tsv"),
                deployable_model_package=str(package_dir),
                applicability_dir=str(applicability_dir),
                outdir=str(scores_dir),
                device=config.device,
            )
        )
        _write_direct_prediction_outputs(outdir, scores_dir, applicability_dir)

    manifest = _direct_prediction_manifest(
        config=config,
        outdir=outdir,
        foregrounds=foregrounds,
        records=records,
        validation_summary=validation_summary,
        applicability_summary=applicability_summary,
        scoring_summary=scoring_summary,
        status=status,
    )
    _write_json(outdir / "prediction_manifest.json", manifest)
    (outdir / "qc_report.md").write_text(_render_direct_qc_report(manifest), encoding="utf-8")
    (outdir / "prediction_report.md").write_text(_render_direct_prediction_report(manifest, outdir), encoding="utf-8")
    return {
        "status": status,
        "outdir": str(outdir),
        "foreground": config.foreground,
        "n_foregrounds": len(foregrounds),
        "n_taxa": len(records),
        "n_codons": len(next(iter(records.values()))) // 3,
        "applicability": applicability_summary.get("status"),
        "device": scoring_summary.get("device", config.device if not config.dry_run else "not_run"),
        "branch_site_predictions": str(outdir / "branch_site_predictions.tsv") if not config.dry_run else "",
        "report": str(outdir / "prediction_report.md"),
    }


def plan_external_benchmark_panel(config: ExternalBenchmarkPanelPlanConfig) -> Dict[str, Any]:
    """Plan a codeml/HyPhy external benchmark panel without executing tools."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    methods = _parse_csv(config.methods)
    classical_tools = _parse_csv(config.classical_tools)
    panel_path = Path(config.panel_manifest)
    panel_rows = read_tsv(panel_path) if panel_path.exists() else []
    expected_rows = panel_rows or _default_panel_rows()
    write_tsv(outdir / "expected_inputs.tsv", expected_rows, list(expected_rows[0]))
    payload = {
        "external_benchmark_panel_plan_version": __version__,
        "status": "planned",
        "panel_manifest": str(panel_path),
        "panel_manifest_exists": panel_path.exists(),
        "deployable_model_package": config.deployable_model_package,
        "methods": methods,
        "classical_tools": classical_tools,
        "panel_categories": [
            "known positives",
            "likely negatives",
            "alignment-sensitive families",
            "saturated families",
            "short/low-information families",
            "paralogy-risk families",
        ],
        "heavy_jobs_executed": False,
    }
    _write_json(outdir / "comparison_schema.json", _comparison_schema())
    (outdir / "benchmark_panel_plan.md").write_text(_render_benchmark_plan_md(payload, expected_rows), encoding="utf-8")
    (outdir / "proposed_babappa_commands.sh").write_text(_render_benchmark_babappa_commands(expected_rows, config, methods), encoding="utf-8")
    (outdir / "proposed_codeml_commands.sh").write_text(_render_classical_commands(expected_rows, "codeml"), encoding="utf-8")
    (outdir / "proposed_hyphy_commands.sh").write_text(_render_classical_commands(expected_rows, "hyphy"), encoding="utf-8")
    for filename in ["proposed_babappa_commands.sh", "proposed_codeml_commands.sh", "proposed_hyphy_commands.sh"]:
        (outdir / filename).chmod(0o755)
    return {
        "status": "planned",
        "outdir": str(outdir),
        "codeml_template": str(outdir / "proposed_codeml_commands.sh"),
        "hyphy_template": str(outdir / "proposed_hyphy_commands.sh"),
        "heavy_jobs_executed": False,
    }


def _run_empirical_method(method: str, source_fasta: Path, aligned_fasta: Path, timeout: int) -> Tuple[str, str, List[str]]:
    if method == "identity":
        shutil.copyfile(source_fasta, aligned_fasta)
        return "ok", "identity_copy", ["internal_identity"]
    executable = shutil.which(method if method != "muscle" else "muscle") or (shutil.which("muscle5") if method == "muscle" else None)
    if executable is None:
        return "fail", "executable_unavailable", [method]
    local_source = aligned_fasta.parent / source_fasta.name
    if source_fasta.resolve() != local_source.resolve():
        shutil.copyfile(source_fasta, local_source)
    local_source_arg = str(local_source.resolve())
    aligned_fasta_arg = str(aligned_fasta.resolve())
    try:
        if method == "mafft":
            command = [executable, "--auto", local_source_arg]
            with aligned_fasta.open("w", encoding="utf-8") as stdout:
                proc = subprocess.run(command, check=False, stdout=stdout, stderr=subprocess.PIPE, text=True, timeout=timeout)
        elif method == "muscle":
            command = [executable, "-align", local_source_arg, "-output", aligned_fasta_arg]
            proc = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
            if proc.returncode != 0 or not aligned_fasta.exists() or aligned_fasta.stat().st_size == 0:
                command = [executable, "-in", local_source_arg, "-out", aligned_fasta_arg]
                proc = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        elif method == "babappalign":
            device = os.environ.get("BABAPPA_BABAPPALIGN_DEVICE", "cpu")
            command = [executable, "--mode", "codon", "--device", device, local_source_arg]
            proc = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, cwd=str(aligned_fasta.parent))
            if proc.stdout.lstrip().startswith(">"):
                aligned_fasta.write_text(proc.stdout, encoding="utf-8")
            elif not aligned_fasta.exists() or aligned_fasta.stat().st_size == 0:
                announced = _babappalign_announced_codon_alignment(proc.stdout)
                if announced is not None and announced.exists():
                    shutil.copyfile(announced, aligned_fasta)
            if not aligned_fasta.exists() or aligned_fasta.stat().st_size == 0:
                candidate = _find_sidecar_fasta(aligned_fasta.parent, local_source)
                if candidate:
                    shutil.copyfile(candidate, aligned_fasta)
        else:
            return "fail", "unsupported_method", [method]
        if proc.returncode == 0 and aligned_fasta.exists() and aligned_fasta.stat().st_size > 0:
            return "ok", "completed", command
        stderr = (proc.stderr or proc.stdout or "").strip()[:300]
        if method == "babappalign" and "babappascore.pt" in stderr.lower():
            return "fail", "babappalign_model_missing", command
        return "fail", stderr or f"return_code:{proc.returncode}", command
    except subprocess.TimeoutExpired:
        return "fail", "timeout", [method]
    except OSError as exc:
        return "fail", f"os_error:{exc}", [method]


def _find_sidecar_fasta(directory: Path, source_fasta: Path) -> Optional[Path]:
    candidates = sorted(directory.glob("*.fasta")) + sorted(directory.glob("*.fa"))
    for candidate in candidates:
        if candidate.name != source_fasta.name and candidate.stat().st_size > 0:
            return candidate
    return None


def _babappalign_announced_codon_alignment(stdout: str) -> Optional[Path]:
    for line in stdout.splitlines():
        if "Codon alignment written:" in line:
            raw = line.split("Codon alignment written:", 1)[1].strip()
            return Path(raw)
    return None


def _validate_aligned_fasta(path: Path, expected_taxa: set[str]) -> Dict[str, Any]:
    failures: List[str] = []
    if not path.exists() or path.stat().st_size == 0:
        return {"status": "fail", "failures": ["missing_or_empty_alignment"]}
    records = read_fasta(path)
    missing = sorted(expected_taxa - set(records))
    if missing:
        failures.append("missing_taxa:" + ",".join(missing))
    lengths = [len(seq) for seq in records.values()]
    if len(set(lengths)) > 1:
        failures.append("unaligned_lengths_not_equal")
    if any(length % 3 != 0 for length in lengths):
        failures.append("alignment_length_not_divisible_by_3")
    return {"status": "fail" if failures else "ok", "failures": failures, "n_taxa": len(records)}


def _build_empirical_method_policy(method_rows: List[Dict[str, Any]], site_map_dir: Path, outdir: Path) -> Dict[str, Any]:
    policy_rows: List[Dict[str, Any]] = []
    usable: List[str] = []
    quarantined: List[str] = []
    for row in method_rows:
        method = str(row["method"])
        if row["status"] != "ok":
            recommendation = "quarantine"
            reason = row["reason"]
            unique = conflict = frame = 0.0
        else:
            site_rows = read_tsv(site_map_dir / f"{method}.site_map.tsv")
            total = max(1, len(site_rows))
            counts = Counter(site_row.get("mapping_status") for site_row in site_rows)
            unique = counts["unique"] / total
            conflict = counts["conflict"] / total
            frame = counts["frame_error"] / total
            recommendation = "usable" if frame == 0 and conflict <= 0.03 else "quarantine"
            reason = "passes_policy_thresholds" if recommendation == "usable" else "site_map_quality"
        if recommendation == "usable":
            usable.append(method)
        else:
            quarantined.append(method)
        policy_rows.append({
            "method": method,
            "attempted_families": 1,
            "successful_families": 1 if row["status"] == "ok" else 0,
            "failed_families": 0 if row["status"] == "ok" else 1,
            "failure_fraction": "0" if row["status"] == "ok" else "1",
            "site_map_unique_fraction": f"{unique:.6g}",
            "site_map_conflict_fraction": f"{conflict:.6g}",
            "site_map_frame_error_fraction": f"{frame:.6g}",
            "recommendation": recommendation,
            "reason": reason,
        })
    payload = {"usable_methods": usable, "quarantined_methods": quarantined, "methods": policy_rows}
    _write_json(outdir / "method_policy.json", payload)
    write_tsv(outdir / "method_policy.tsv", policy_rows, list(policy_rows[0]) if policy_rows else ["method"])
    (outdir / "method_policy.md").write_text(_render_policy_md(payload), encoding="utf-8")
    return payload


def _feature_row_from_site(
    records: Dict[str, str],
    site_row: Dict[str, str],
    branch_id: str,
    foreground: str,
    validation: Dict[str, Any],
    expected_features: List[str],
) -> Dict[str, Any]:
    aligned_site = int(float(site_row.get("aligned_site_index_zero") or 0))
    original_site_raw = site_row.get("original_site_index_zero")
    original_site = aligned_site if original_site_raw in {None, ""} else int(float(original_site_raw))
    codons = {taxon: _site_codon(sequence, aligned_site) for taxon, sequence in records.items()}
    ids = {taxon: _codon_id(codon) for taxon, codon in codons.items()}
    values = np.array(list(ids.values()), dtype=float) if ids else np.array([0.0])
    foreground_id = ids.get(foreground, 0)
    branch_id_value = ids.get(branch_id, 0)
    background = [value for taxon, value in ids.items() if taxon != foreground]
    background_mean = float(np.mean(background)) if background else 0.0
    n_codons = int(validation.get("n_codons") or (len(next(iter(records.values()))) // 3 if records else 0))
    row = {
        "site_index_zero": original_site,
        "aligned_site_index_zero": aligned_site,
        "original_site_index_zero": original_site,
        "site_relative_position": 0.0 if n_codons <= 1 else original_site / max(1, n_codons - 1),
        "n_taxa": len(records),
        "n_codons": n_codons,
        "codon_id_mean": float(np.mean(values)),
        "codon_id_std": float(np.std(values)),
        "codon_id_min": float(np.min(values)),
        "codon_id_max": float(np.max(values)),
        "codon_id_range": float(np.max(values) - np.min(values)),
        "codon_id_unique_count": int(len(set(ids.values()))),
        "gap_fraction": sum(1 for codon in codons.values() if "-" in codon) / max(1, len(codons)),
        "non_gap_fraction": sum(1 for codon in codons.values() if "-" not in codon) / max(1, len(codons)),
        "taxon_codon_variability": float(np.std(values)),
        "foreground_codon_id": foreground_id,
        "foreground_gap": 1 if "-" in codons.get(foreground, "---") else 0,
        "branch_codon_id": branch_id_value,
        "branch_gap": 1 if "-" in codons.get(branch_id, "---") else 0,
        "background_mean_codon_id": background_mean,
        "foreground_background_codon_delta": foreground_id - background_mean,
        "branch_background_codon_delta": branch_id_value - background_mean,
    }
    return {feature: row.get(feature, "") for feature in expected_features}


def _aggregate_scores(rows: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(str(row.get(key, "")) for key in keys), []).append(row)
    result: List[Dict[str, Any]] = []
    for group_key, group_rows in grouped.items():
        probs = [float(row["prob_positive"]) for row in group_rows]
        out = {key: value for key, value in zip(keys, group_key)}
        out.update({
            "n_branch_site_rows": len(group_rows),
            "max_prob_positive": max(probs),
            "mean_prob_positive": sum(probs) / len(probs),
            "n_called_positive": sum(int(row["called_positive"]) for row in group_rows),
            "diagnostic_only": any(str(row.get("diagnostic_only")) == "True" for row in group_rows),
        })
        result.append(out)
    return result


def _read_fasta_with_duplicates(path: Path) -> Tuple[Dict[str, str], List[str]]:
    records: Dict[str, List[str]] = {}
    current: Optional[str] = None
    duplicates: List[str] = []
    if not path.exists():
        return {}, []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            current = line[1:].split()[0]
            if current in records and current not in duplicates:
                duplicates.append(current)
            records.setdefault(current, [])
        elif current is not None:
            records[current].append(line.upper().replace("U", "T"))
    return {key: "".join(value) for key, value in records.items()}, duplicates


def _parse_newick_tips(text: str) -> set[str]:
    tips = set()
    for token in re.findall(r"(?<=[(,])\s*([^():,;\s]+)\s*(?=[:),;])", text):
        if token:
            tips.add(token)
    return tips


def _codons(sequence: str) -> List[str]:
    return [sequence[index:index + 3].upper().replace("U", "T") for index in range(0, len(sequence), 3) if len(sequence[index:index + 3]) == 3]


def _site_codon(sequence: str, site: int) -> str:
    start = site * 3
    codon = sequence[start:start + 3].upper().replace("U", "T")
    return codon if len(codon) == 3 else "---"


def _codon_id(codon: str) -> int:
    codon = codon.upper().replace("U", "T")
    if "-" in codon or len(codon) != 3:
        return 0
    alphabet = "ACGT"
    value = 1
    for char in codon:
        if char not in alphabet:
            return 0
        value = value * 4 + alphabet.index(char)
    return value


def _ambiguous_fraction(records: Dict[str, str]) -> float:
    total = sum(len(seq) for seq in records.values())
    if total == 0:
        return 0.0
    ambiguous = sum(1 for seq in records.values() for char in seq.upper() if char not in {"A", "C", "G", "T", "U", "-"})
    return ambiguous / total


def _gap_fraction(records: Dict[str, str]) -> float:
    total = sum(len(seq) for seq in records.values())
    return 0.0 if total == 0 else sum(seq.count("-") for seq in records.values()) / total


def _mean_pairwise_p_distance(records: Dict[str, str]) -> float:
    values = list(records.values())
    if len(values) < 2:
        return 0.0
    distances: List[float] = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            comparable = 0
            diff = 0
            for a, b in zip(values[i].upper(), values[j].upper()):
                if a == "-" or b == "-":
                    continue
                comparable += 1
                diff += int(a != b)
            distances.append(0.0 if comparable == 0 else diff / comparable)
    return float(sum(distances) / len(distances)) if distances else 0.0


def _alignment_mean_pairwise_p_distance(empirical_feature_dir: Path) -> Optional[float]:
    manifest_path = empirical_feature_dir / "empirical_feature_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        feature_manifest = _read_json(manifest_path)
        alignment_dir = Path(str(feature_manifest.get("alignment_dir", "")))
        align_manifest = _read_json(alignment_dir / "empirical_alignment_manifest.json")
    except (OSError, ValueError, TypeError):
        return None
    distances: List[float] = []
    for files in (align_manifest.get("created_files") or {}).values():
        codon_fasta = files.get("codon_fasta") if isinstance(files, dict) else None
        if not codon_fasta:
            continue
        try:
            records = read_fasta(Path(str(codon_fasta)))
        except OSError:
            continue
        if records:
            distances.append(_mean_pairwise_p_distance(records))
    return float(sum(distances) / len(distances)) if distances else None


def _saturation_proxy(p_distance: float) -> str:
    return _recommended_tier(p_distance)


def _recommended_tier(p_distance: Optional[float]) -> str:
    if p_distance is None:
        return "low"
    if p_distance < 0.05:
        return "low"
    if p_distance < 0.12:
        return "moderate"
    if p_distance < 0.25:
        return "high"
    return "extreme"


def _tree_shape_summary(tree: str, tips: set[str]) -> str:
    return f"n_tips={len(tips)};n_internal_commas={tree.count(',')}"


def _site_map_fields(rows: List[Dict[str, Any]]) -> List[str]:
    return list(rows[0]) if rows else [
        "family_id",
        "method",
        "aligned_site_index_zero",
        "aligned_site_index_one",
        "original_site_index_zero",
        "original_site_index_one",
        "mapping_status",
        "n_taxa_mapped",
        "n_taxa_gap",
        "n_taxa_conflict",
        "mapping_confidence",
    ]


def _forbidden_columns(columns: Iterable[str]) -> List[str]:
    found = []
    for column in columns:
        lowered = str(column).lower()
        for token in FORBIDDEN_EMPIRICAL_INPUT_COLUMNS:
            if token in lowered:
                found.append(str(column))
                break
    return sorted(set(found))


def _flat_validation_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": payload.get("status"),
        "foreground": payload.get("foreground"),
        "n_taxa": payload.get("n_taxa"),
        "n_codons": payload.get("n_codons"),
        "ambiguous_base_fraction": payload.get("ambiguous_base_fraction"),
        "gap_fraction": payload.get("gap_fraction"),
        "mean_pairwise_p_distance": payload.get("mean_pairwise_p_distance"),
        "saturation_proxy": payload.get("saturation_proxy"),
        "failures": ";".join(payload.get("failures") or []),
        "warnings": ";".join(payload.get("warnings") or []),
    }


def _flat_applicability_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": payload.get("status"),
        "recommended_tier": payload.get("recommended_tier"),
        "reasons": ";".join(payload.get("reasons") or []),
        "feature_rows": payload.get("feature_rows"),
    }


def _flat_report_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": payload.get("status"),
        "no_simulator_truth_used": payload.get("no_simulator_truth_used"),
        "applicability_status": payload.get("applicability", {}).get("applicability_status"),
        "scoring_status": payload.get("scoring", {}).get("status"),
        "not_final_empirical_inference": payload.get("not_final_empirical_inference"),
    }


def _read_header(path: Path) -> List[str]:
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return []
    return first.split("\t")


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _max_level(current: str, candidate: str) -> str:
    order = {"in_domain": 0, "borderline": 1, "out_of_domain": 2}
    return candidate if order[candidate] > order[current] else current


def _torch_load(torch: Any, path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return data


def _read_optional_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_deployable_package_path(package: str | Path) -> Path:
    package_path = Path(package)
    if package_path.exists():
        return package_path
    if str(package) == "deployable_model_conservative_branch_site_100k_mps":
        try:
            bundled = resources.files("babappa") / "model_packages" / "deployable_model_conservative_branch_site_100k_mps"
        except (AttributeError, ModuleNotFoundError):
            bundled = None
        if bundled is not None and bundled.is_dir():
            return Path(str(bundled))
    raise FileNotFoundError(f"deployable model package not found: {package}")


def _parse_csv(value: Sequence[str] | str) -> List[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    parsed: List[str] = []
    for item in value:
        parsed.extend(part.strip() for part in str(item).split(",") if part.strip())
    return parsed


def _render_empirical_input_md(payload: Dict[str, Any]) -> str:
    lines = ["# Empirical input validation", "", f"- status: `{payload['status']}`", f"- foreground: `{payload['foreground']}`", f"- n_taxa: `{payload['n_taxa']}`", f"- n_codons: `{payload['n_codons']}`", f"- p-distance: `{payload['mean_pairwise_p_distance']}`", f"- saturation proxy: `{payload['saturation_proxy']}`", ""]
    if payload["failures"]:
        lines.extend(["## Failures", *[f"- {item}" for item in payload["failures"]], ""])
    if payload["warnings"]:
        lines.extend(["## Warnings", *[f"- {item}" for item in payload["warnings"]], ""])
    return "\n".join(lines)


def _render_alignment_report(manifest: Dict[str, Any]) -> str:
    lines = ["# Empirical alignment ensemble", "", f"- status: `{manifest['status']}`", f"- methods run: `{','.join(manifest['methods_run'])}`", ""]
    for row in manifest["method_rows"]:
        lines.append(f"- {row['method']}: {row['status']} ({row['reason']})")
    lines.append("")
    return "\n".join(lines)


def _render_policy_md(payload: Dict[str, Any]) -> str:
    lines = ["# Empirical method policy", "", "## Usable methods", ""]
    lines.extend(f"- {method}" for method in payload["usable_methods"] or ["none"])
    lines.extend(["", "## Quarantined methods", ""])
    lines.extend(f"- {method}" for method in payload["quarantined_methods"] or ["none"])
    lines.append("")
    return "\n".join(lines)


def _render_feature_report(manifest: Dict[str, Any]) -> str:
    return "\n".join([
        "# Empirical branch-site features",
        "",
        f"- status: `{manifest['status']}`",
        f"- rows: `{manifest['n_rows']}`",
        f"- feature policy: `{manifest['feature_policy']}`",
        "- truth-derived inputs excluded: `True`",
        "",
    ])


def _render_feature_audit_md(payload: Dict[str, Any]) -> str:
    return "\n".join([
        "# Empirical feature audit",
        "",
        f"- status: `{payload['status']}`",
        f"- rows: `{payload['n_rows']}`",
        f"- forbidden columns: `{','.join(payload['forbidden_columns'])}`",
        "",
    ])


def _render_applicability_md(payload: Dict[str, Any]) -> str:
    lines = ["# Empirical applicability/OOD", "", f"- status: `{payload['status']}`", f"- recommended tier: `{payload['recommended_tier']}`", "", "## Reasons", ""]
    lines.extend(f"- {reason}" for reason in payload["reasons"])
    lines.append("")
    return "\n".join(lines)


def _render_scoring_failure_md(payload: Dict[str, Any]) -> str:
    return "\n".join(["# Empirical scoring failed", "", f"- reason: `{payload['reason']}`", "- PyTorch is required for scoring; metadata-only mode is not allowed.", ""])


def _render_scoring_report(payload: Dict[str, Any]) -> str:
    return "\n".join([
        "# Empirical branch-site scoring",
        "",
        f"- status: `{payload['status']}`",
        f"- device: `{payload['device']}`",
        f"- tier model: `{payload['tier_model']}`",
        f"- rows: `{payload['n_rows']}`",
        f"- diagnostic only: `{payload['diagnostic_only']}`",
        "",
    ])


def _render_empirical_report_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# BABAPPA empirical branch-site smoke report",
        "",
        "No simulator truth used: `True`.",
        "",
        "The model is simulation-trained. Scores are not final empirical claims unless calibration and OOD checks pass.",
        "",
        f"- applicability: `{payload['applicability'].get('applicability_status')}`",
        f"- scoring status: `{payload.get('scoring', {}).get('status', 'missing')}`",
        f"- methods run: `{','.join(payload['alignment'].get('methods_run') or [])}`",
        "",
        "External validation with codeml/HyPhy-style workflows is recommended.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["limitations"])
    lines.append("")
    return "\n".join(lines)


def _validate_direct_msa(records: Dict[str, str], tree_tips: set[str], min_taxa: int, min_codons: int) -> None:
    if len(records) < min_taxa:
        raise ValueError(f"too few taxa in MSA: {len(records)} < {min_taxa}")
    lengths = {record_id: len(sequence) for record_id, sequence in records.items()}
    unique_lengths = sorted(set(lengths.values()))
    if len(unique_lengths) != 1:
        detail = ",".join(f"{record_id}:{length}" for record_id, length in sorted(lengths.items()))
        raise ValueError("MSA sequence lengths are not equal; BABAPPA direct prediction requires an aligned codon MSA: " + detail)
    length = unique_lengths[0]
    if length % 3 != 0:
        raise ValueError(f"MSA length is not divisible by 3: {length}")
    if (length // 3) < min_codons:
        raise ValueError(f"too few codons in MSA: {length // 3} < {min_codons}")
    fasta_ids = set(records)
    missing_in_tree = sorted(fasta_ids - tree_tips)
    missing_in_msa = sorted(tree_tips - fasta_ids)
    if missing_in_tree or missing_in_msa:
        pieces = []
        if missing_in_tree:
            pieces.append("msa_ids_missing_from_tree:" + ",".join(missing_in_tree))
        if missing_in_msa:
            pieces.append("tree_tips_missing_from_msa:" + ",".join(missing_in_msa))
        raise ValueError("MSA/tree labels do not match: " + ";".join(pieces))


def _resolve_direct_foregrounds(foreground: str, records: Dict[str, str], tree_tips: set[str]) -> List[str]:
    requested = str(foreground or "all").strip()
    if requested.lower() == "all":
        return [record_id for record_id in records if record_id in tree_tips]
    foregrounds = [item.strip() for item in requested.split(",") if item.strip()]
    if not foregrounds:
        raise ValueError("foreground must be 'all' or a comma-separated list of tree tips")
    missing = [item for item in foregrounds if item not in records or item not in tree_tips]
    if missing:
        raise ValueError("foreground tip(s) missing from MSA/tree: " + ",".join(missing))
    return foregrounds


def _write_user_msa_alignment_artifacts(
    msa_path: Path,
    tree_path: Path,
    records: Dict[str, str],
    foreground_requested: str,
    foregrounds: List[str],
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    method_dir = outdir / "methods" / "user_msa"
    site_map_dir = outdir / "site_map"
    policy_dir = outdir / "method_policy"
    method_dir.mkdir(parents=True, exist_ok=True)
    site_map_dir.mkdir(parents=True, exist_ok=True)
    policy_dir.mkdir(parents=True, exist_ok=True)
    aligned_fasta = method_dir / "empirical.user_msa.codon.fasta"
    write_fasta(records, aligned_fasta)
    site_rows = _build_user_msa_identity_site_map(records, family_id="empirical", method="user_msa")
    site_map_path = site_map_dir / "user_msa.site_map.tsv"
    write_tsv(site_map_path, site_rows, _site_map_fields(site_rows))
    qc_path = method_dir / "empirical.user_msa.qc.json"
    qc = {
        "method": "user_msa",
        "status": "ok",
        "reason": "user_supplied_authoritative_codon_msa",
        "validation": _validate_aligned_fasta(aligned_fasta, set(records)),
        "realignment_performed": False,
    }
    _write_json(qc_path, qc)
    policy_rows = [{
        "method": "user_msa",
        "attempted_families": 1,
        "successful_families": 1,
        "failed_families": 0,
        "failure_fraction": "0",
        "site_map_unique_fraction": "1",
        "site_map_conflict_fraction": "0",
        "site_map_frame_error_fraction": "0",
        "recommendation": "usable",
        "reason": "user_msa_is_authoritative_input",
    }]
    policy_payload = {"usable_methods": ["user_msa"], "quarantined_methods": [], "methods": policy_rows}
    _write_json(policy_dir / "method_policy.json", policy_payload)
    write_tsv(policy_dir / "method_policy.tsv", policy_rows, list(policy_rows[0]))
    (policy_dir / "method_policy.md").write_text(_render_policy_md(policy_payload), encoding="utf-8")
    method_rows = [{
        "method": "user_msa",
        "status": "ok",
        "reason": "user_supplied_authoritative_codon_msa",
        "runtime_seconds": "0",
        "aligned_fasta": str(aligned_fasta),
        "site_map": str(site_map_path),
    }]
    manifest = {
        "empirical_alignment_version": __version__,
        "status": "ok",
        "cds_fasta": str(msa_path),
        "tree": str(tree_path),
        "foreground": foreground_requested,
        "foregrounds_resolved": foregrounds,
        "methods_requested": ["user_msa"],
        "methods_run": ["user_msa"],
        "method_rows": method_rows,
        "created_files": {
            "user_msa": {
                "codon_fasta": str(aligned_fasta),
                "qc": str(qc_path),
                "site_map": str(site_map_path),
            }
        },
        "site_map_dir": str(site_map_dir),
        "method_policy_dir": str(policy_dir),
        "method_policy": policy_payload,
        "realignment_performed": False,
        "user_msa_is_authoritative": True,
        "failures": [],
        "warnings": [],
    }
    _write_json(outdir / "empirical_alignment_manifest.json", manifest)
    write_tsv(outdir / "empirical_alignment_summary.tsv", method_rows, ["method", "status", "reason", "runtime_seconds", "aligned_fasta", "site_map"])
    (outdir / "empirical_alignment_report.md").write_text(
        "\n".join([
            "# User-supplied MSA",
            "",
            "- status: `ok`",
            "- method: `user_msa`",
            "- realignment performed: `False`",
            "- BABAPPA treated the supplied codon MSA as the authoritative alignment.",
            "",
        ]),
        encoding="utf-8",
    )


def _extract_direct_branch_site_features(
    validation_dir: Path,
    alignment_dir: Path,
    deployable_model_package: Path,
    outdir: Path,
    foregrounds: List[str],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    feature_schema = _read_json(deployable_model_package / "feature_schema.json")
    expected_features = [str(item) for item in feature_schema.get("expected_feature_columns", [])]
    if not expected_features:
        raise ValueError("deployable package feature_schema.json has no expected_feature_columns")
    validation = _read_json(validation_dir / "empirical_input_validation.json")
    align_manifest = _read_json(alignment_dir / "empirical_alignment_manifest.json")
    files = align_manifest.get("created_files", {}).get("user_msa", {})
    aligned_fasta = files.get("codon_fasta")
    site_map_file = files.get("site_map")
    if not aligned_fasta or not site_map_file:
        raise ValueError("direct user MSA artifacts are missing codon_fasta or site_map")
    records = read_fasta(Path(aligned_fasta))
    site_rows = read_tsv(Path(site_map_file))
    rows: List[Dict[str, Any]] = []
    missing_features: List[str] = []
    for site_row in site_rows:
        if site_row.get("mapping_status") not in {"unique", "conflict"}:
            continue
        for branch_id in foregrounds:
            feature_values = _feature_row_from_site(
                records,
                site_row,
                branch_id,
                branch_id,
                validation,
                expected_features,
            )
            row = {
                "family_id": "empirical",
                "method": "user_msa",
                "branch_id": branch_id,
                "foreground_taxon": branch_id,
                "aligned_site_index_zero": site_row.get("aligned_site_index_zero", ""),
                "original_site_index_zero": site_row.get("original_site_index_zero", ""),
                "mapping_status": site_row.get("mapping_status", ""),
            }
            row.update(feature_values)
            missing_features.extend([name for name in expected_features if row.get(name) in {None, ""}])
            rows.append(row)
    forbidden = _forbidden_columns(FEATURE_OUTPUT_FIELDS + expected_features)
    schema_match = "pass" if rows and not missing_features and not forbidden else "fail"
    fields = FEATURE_OUTPUT_FIELDS + expected_features
    write_tsv(outdir / "empirical_branch_site_features.tsv", rows, fields)
    schema_check = {
        "status": schema_match,
        "feature_schema_match": schema_match,
        "expected_feature_columns": expected_features,
        "n_rows": len(rows),
        "foregrounds_scored": foregrounds,
        "missing_features": sorted(set(missing_features)),
        "forbidden_columns": forbidden,
    }
    _write_json(outdir / "empirical_feature_schema_check.json", schema_check)
    manifest = {
        "empirical_feature_version": __version__,
        "status": schema_match,
        "feature_policy": feature_schema.get("feature_policy"),
        "n_rows": len(rows),
        "foreground": "all" if len(foregrounds) == len(records) else ",".join(foregrounds),
        "foregrounds_scored": foregrounds,
        "alignment_dir": str(alignment_dir),
        "deployable_model_package": str(deployable_model_package),
        "feature_schema_check": schema_check,
        "truth_derived_inputs_excluded": True,
        "user_msa_is_authoritative": True,
        "realignment_performed": False,
        "generated_files": {
            "features": str(outdir / "empirical_branch_site_features.tsv"),
            "schema_check": str(outdir / "empirical_feature_schema_check.json"),
            "report": str(outdir / "empirical_feature_report.md"),
        },
    }
    _write_json(outdir / "empirical_feature_manifest.json", manifest)
    (outdir / "empirical_feature_report.md").write_text(_render_feature_report(manifest), encoding="utf-8")
    if schema_match != "pass":
        raise ValueError("direct MSA feature schema check failed: " + ",".join(schema_check["missing_features"] + forbidden))


def _build_user_msa_identity_site_map(records: Dict[str, str], family_id: str, method: str) -> List[Dict[str, Any]]:
    n_codons = len(next(iter(records.values()))) // 3 if records else 0
    rows: List[Dict[str, Any]] = []
    for site_index in range(n_codons):
        codons = [_site_codon(sequence, site_index) for sequence in records.values()]
        n_gap = sum(1 for codon in codons if codon == "---")
        rows.append({
            "family_id": family_id,
            "method": method,
            "aligned_site_index_zero": site_index,
            "aligned_site_index_one": site_index + 1,
            "original_site_index_zero": site_index,
            "original_site_index_one": site_index + 1,
            "mapping_status": "unique",
            "n_taxa_mapped": len(codons) - n_gap,
            "n_taxa_gap": n_gap,
            "n_taxa_conflict": 0,
            "mapping_confidence": 1.0,
        })
    return rows


def _write_direct_prediction_outputs(outdir: Path, scores_dir: Path, applicability_dir: Path) -> None:
    scores = read_tsv(scores_dir / "empirical_branch_site_scores.tsv")
    branch_scores = read_tsv(scores_dir / "empirical_branch_scores.tsv")
    gene_support = read_tsv(scores_dir / "empirical_gene_support.tsv")
    applicability = _read_json(applicability_dir / "empirical_applicability.json")
    user_msa_path = outdir / "user_msa" / "methods" / "user_msa" / "empirical.user_msa.codon.fasta"
    user_msa_records = read_fasta(user_msa_path) if user_msa_path.exists() else {}
    prediction_rows: List[Dict[str, Any]] = []
    for row in scores:
        prob = _as_float(row.get("prob_positive"), 0.0) or 0.0
        threshold = _as_float(row.get("calibrated_threshold"), 0.5) or 0.5
        called = int(_as_float(row.get("called_positive"), 0.0) or 0.0)
        diagnostic_only = str(row.get("diagnostic_only", "")).lower() == "true"
        aligned_zero = int(float(row.get("aligned_site_index_zero") or 0))
        original_zero = int(float(row.get("original_site_index_zero") or aligned_zero))
        branch_id = row.get("branch_id", "")
        branch_codon = _site_codon(user_msa_records.get(branch_id, ""), aligned_zero) if user_msa_records else ""
        prediction_rows.append({
            "branch_id": branch_id,
            "codon_site": original_zero + 1,
            "msa_codon_site": aligned_zero + 1,
            "aligned_codon_site": aligned_zero + 1,
            "branch_degapped_codon_site": _degapped_codon_site(user_msa_records.get(branch_id, ""), aligned_zero),
            "branch_codon": branch_codon,
            "prob_positive": prob,
            "called_positive": called,
            "confidence": _prediction_confidence(prob, threshold, called, diagnostic_only),
            "tier_model": row.get("tier_model", ""),
            "calibrated_threshold": threshold,
            "diagnostic_only": diagnostic_only,
            "method": row.get("method", "user_msa"),
        })
    write_tsv(
        outdir / "branch_site_predictions.tsv",
        prediction_rows,
        [
            "branch_id",
            "codon_site",
            "msa_codon_site",
            "aligned_codon_site",
            "branch_degapped_codon_site",
            "branch_codon",
            "prob_positive",
            "called_positive",
            "confidence",
            "tier_model",
            "calibrated_threshold",
            "diagnostic_only",
            "method",
        ],
    )
    write_tsv(outdir / "branch_predictions.tsv", branch_scores, list(branch_scores[0]) if branch_scores else ["family_id", "branch_id"])
    max_gene_support = max((_as_float(row.get("max_prob_positive"), 0.0) or 0.0 for row in gene_support), default=0.0)
    n_called = sum(int(_as_float(row.get("called_positive"), 0.0) or 0.0) for row in scores)
    summary_rows = [{
        "family_id": "empirical",
        "n_branches_scored": len({row.get("branch_id", "") for row in scores}),
        "n_branch_site_rows": len(scores),
        "n_called_positive": n_called,
        "max_gene_support": max_gene_support,
        "applicability_status": applicability.get("applicability_status"),
        "tier_model": scores[0].get("tier_model", "") if scores else "",
        "diagnostic_only": any(str(row.get("diagnostic_only", "")).lower() == "true" for row in scores),
        "result_class": "diagnostic_positive" if n_called > 0 else "diagnostic_negative",
    }]
    write_tsv(outdir / "gene_summary.tsv", summary_rows, list(summary_rows[0]))


def _degapped_codon_site(sequence: str, aligned_site_zero: int) -> str:
    if not sequence:
        return ""
    count = 0
    for site_index in range(0, aligned_site_zero + 1):
        codon = _site_codon(sequence, site_index)
        if "-" not in codon:
            count += 1
    target = _site_codon(sequence, aligned_site_zero)
    return "" if "-" in target else str(count)


def _prediction_confidence(prob: float, threshold: float, called: int, diagnostic_only: bool) -> str:
    if diagnostic_only:
        return "diagnostic_only"
    if not called:
        return "below_threshold"
    if prob >= max(0.9, threshold):
        return "high"
    return "moderate"


def _direct_prediction_manifest(
    config: DirectBranchSitePredictionConfig,
    outdir: Path,
    foregrounds: List[str],
    records: Dict[str, str],
    validation_summary: Dict[str, Any],
    applicability_summary: Dict[str, Any],
    scoring_summary: Dict[str, Any],
    status: str,
) -> Dict[str, Any]:
    gene_summary_path = outdir / "gene_summary.tsv"
    branch_site_path = outdir / "branch_site_predictions.tsv"
    gene_summary_rows = read_tsv(gene_summary_path) if gene_summary_path.exists() else []
    prediction_rows = read_tsv(branch_site_path) if branch_site_path.exists() else []
    return {
        "direct_prediction_version": __version__,
        "status": status,
        "msa": config.msa,
        "tree": config.tree,
        "model_package": config.model_package,
        "foreground_requested": config.foreground,
        "foregrounds_scored": foregrounds,
        "n_taxa": len(records),
        "n_codons": len(next(iter(records.values()))) // 3,
        "user_msa_is_authoritative": True,
        "realignment_performed": False,
        "no_simulator_truth_used": True,
        "truth_derived_inputs_excluded": True,
        "validation_status": validation_summary.get("status"),
        "applicability_status": applicability_summary.get("status"),
        "applicability_reasons": applicability_summary.get("reasons"),
        "scoring": scoring_summary,
        "summary": gene_summary_rows[0] if gene_summary_rows else {},
        "n_prediction_rows": len(prediction_rows),
        "outputs": {
            "branch_site_predictions": str(branch_site_path) if branch_site_path.exists() else "",
            "branch_predictions": str(outdir / "branch_predictions.tsv") if (outdir / "branch_predictions.tsv").exists() else "",
            "gene_summary": str(gene_summary_path) if gene_summary_path.exists() else "",
            "prediction_report": str(outdir / "prediction_report.md"),
            "qc_report": str(outdir / "qc_report.md"),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _render_direct_qc_report(manifest: Dict[str, Any]) -> str:
    return "\n".join([
        "# BABAPPA Direct MSA QC",
        "",
        f"- status: `{manifest['status']}`",
        f"- n_taxa: `{manifest['n_taxa']}`",
        f"- n_codons: `{manifest['n_codons']}`",
        f"- foreground requested: `{manifest['foreground_requested']}`",
        f"- foregrounds scored: `{','.join(manifest['foregrounds_scored'])}`",
        f"- input validation: `{manifest['validation_status']}`",
        f"- applicability: `{manifest['applicability_status']}`",
        "- user MSA is authoritative: `True`",
        "- realignment performed: `False`",
        "- simulator truth used during empirical inference: `False`",
        "",
    ])


def _render_direct_prediction_report(manifest: Dict[str, Any], outdir: Path) -> str:
    summary = manifest.get("summary") or {}
    lines = [
        "# BABAPPA Branch-Site Prediction",
        "",
        "BABAPPA used the supplied codon MSA as the authoritative alignment. No realignment or aligner-disagreement analysis was performed.",
        "",
        f"- status: `{manifest['status']}`",
        f"- applicability: `{manifest['applicability_status']}`",
        f"- foreground requested: `{manifest['foreground_requested']}`",
        f"- branches scored: `{summary.get('n_branches_scored', len(manifest['foregrounds_scored']))}`",
        f"- branch-site rows: `{summary.get('n_branch_site_rows', manifest['n_prediction_rows'])}`",
        f"- called positive branch-site rows: `{summary.get('n_called_positive', 'not_scored')}`",
        f"- max gene support: `{summary.get('max_gene_support', 'not_scored')}`",
        f"- result class: `{summary.get('result_class', manifest['status'])}`",
        "",
        "## Main Outputs",
        "",
        f"- branch-site predictions: `{outdir / 'branch_site_predictions.tsv'}`",
        f"- branch summaries: `{outdir / 'branch_predictions.tsv'}`",
        f"- gene summary: `{outdir / 'gene_summary.tsv'}`",
        "",
        "## Interpretation Boundary",
        "",
        "A BABAPPA diagnostic-positive result is not, by itself, a publishable empirical positive-selection claim. Interpret scores with dataset-specific QC, matched-null calibration, biological controls, and reference-tool comparison when making manuscript claims.",
        "",
    ]
    return "\n".join(lines)


def _default_panel_rows() -> List[Dict[str, str]]:
    categories = [
        "known_positive",
        "likely_negative",
        "alignment_sensitive",
        "saturated",
        "short_low_information",
        "paralogy_risk",
    ]
    return [
        {
            "family_id": category,
            "category": category,
            "cds_fasta": f"panel/{category}.cds.fasta",
            "tree": f"panel/{category}.treefile",
            "foreground": "foreground_taxon",
            "notes": "placeholder",
        }
        for category in categories
    ]


def _comparison_schema() -> Dict[str, Any]:
    return {
        "fields": [
            "family_id",
            "category",
            "babappa_applicability",
            "babappa_gene_support",
            "codeml_lrt_pvalue",
            "hyphy_pvalue",
            "concordance_class",
            "notes",
        ],
        "claim_boundary": "benchmark comparison only; not automatic empirical truth",
    }


def _render_benchmark_plan_md(payload: Dict[str, Any], rows: List[Dict[str, str]]) -> str:
    return "\n".join([
        "# External benchmark panel plan",
        "",
        f"- status: `{payload['status']}`",
        f"- panel entries: `{len(rows)}`",
        f"- classical tools: `{','.join(payload['classical_tools'])}`",
        "- USER-RUN ONLY command templates are generated but not executed.",
        "",
    ])


def _render_benchmark_babappa_commands(rows: List[Dict[str, str]], config: ExternalBenchmarkPanelPlanConfig, methods: List[str]) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "echo 'USER-RUN ONLY - DO NOT EXECUTE IN CODEX'", ""]
    for row in rows:
        lines.append(
            "# babappa plan-empirical-scoring "
            f"--cds-fasta {row['cds_fasta']} --tree {row['tree']} --foreground {row['foreground']} "
            f"--deployable-model-package {config.deployable_model_package} "
            f"--outdir empirical_scoring_plan_{row['family_id']} --methods {','.join(methods)} --device auto"
        )
    lines.append("")
    return "\n".join(lines)


def _render_classical_commands(rows: List[Dict[str, str]], tool: str) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "echo 'USER-RUN ONLY - DO NOT EXECUTE IN CODEX'", ""]
    for row in rows:
        if tool == "codeml":
            lines.append(f"# codeml {row['family_id']}.branch_site.ctl")
        else:
            lines.append(f"# hyphy absrel --alignment {row['cds_fasta']} --tree {row['tree']}")
    lines.append("")
    return "\n".join(lines)
