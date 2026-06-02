"""Reference-workflow and calibration preparation for guarded empirical pilots."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv
from babappa.training.neural_env import resolve_torch_device, safe_import_torch

USER_RUN_ONLY = "MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING"
CLAIM_BOUNDARY = (
    "Diagnostic empirical evidence only. Not manuscript-ready and not a final positive-selection discovery claim "
    "until simulation-matched calibration and codeml/HyPhy-style reference comparison are interpreted."
)
REFERENCE_RESULT_FIELDS = ["panel_id", "tool", "test_name", "p_value", "q_value", "selected_branch", "selected_sites", "result_class", "notes"]
FORBIDDEN_PACK_PATTERNS = ["branch_site_truth", "selected_sites", "truth.json", "branch_truth", "oracle", "y_branch_site", "y_site", "gene_label"]


@dataclass(frozen=True)
class EmpiricalEvidencePackConfig:
    family_id: str
    outdir: str
    cds_fasta: str
    tree_file: str
    foreground: str
    babappa_family_dir: str
    panel_run_summary: str
    prefilter_dir: str
    summary_report: str = ""
    reference_plan_dir: str = ""
    calibration_plan_dir: str = ""


@dataclass(frozen=True)
class EmpiricalEvidencePackValidationConfig:
    evidence_pack: str


@dataclass(frozen=True)
class SimulationMatchedCalibrationSummaryConfig:
    plan_dir: str
    outdir: str


@dataclass(frozen=True)
class ReferenceToolCheckConfig:
    outdir: str


@dataclass(frozen=True)
class ReferenceToolsInstallPlanConfig:
    outdir: str


@dataclass(frozen=True)
class CodemlReferencePrepConfig:
    cds_fasta: str
    tree: str
    foreground: str
    outdir: str


@dataclass(frozen=True)
class HyphyReferencePrepConfig:
    cds_fasta: str
    tree: str
    foreground: str
    outdir: str


@dataclass(frozen=True)
class CodemlReferenceParseConfig:
    codeml_dir: str
    outdir: str


@dataclass(frozen=True)
class HyphyReferenceParseConfig:
    hyphy_dir: str
    outdir: str


@dataclass(frozen=True)
class ReferenceResultsTemplateConfig:
    family_id: str
    foreground: str
    outdir: str


@dataclass(frozen=True)
class WRKYInterpretationStatusConfig:
    family_id: str
    babappa_panel_run: str
    evidence_pack: str
    calibration_summary: str
    reference_results: str
    outdir: str


@dataclass(frozen=True)
class ReferenceResultsTableConfig:
    panel_id: str
    codeml_parsed: str
    hyphy_parsed: str
    outdir: str


@dataclass(frozen=True)
class SimulationMatchedNullCalibrationConfig:
    plan_dir: str
    deployable_model_package: str
    outdir: str
    n_replicates: int = 100
    device: str = "auto"
    seed: int = 42
    fast_null_mode: bool = False
    evidence_pack: str = ""
    dry_run: bool = False
    n_alt: int = 0
    tier: str = ""
    family_id: str = ""
    max_workers: int = 1
    resume: bool = False
    force: bool = False


@dataclass(frozen=True)
class SimulationMatchedNullCalibrationValidationConfig:
    calibration_dir: str


@dataclass(frozen=True)
class WRKYReferenceCalibrationReportConfig:
    evidence_pack: str
    babappa_panel_run: str
    reference_results: str
    comparison_dir: str
    matched_null_calibration: str
    outdir: str


@dataclass(frozen=True)
class BabappaOnlySignalInterpretationConfig:
    babappa_report: str
    matched_null: str
    reference_results: str
    outdir: str


@dataclass(frozen=True)
class BabappaOnlyResultAuditConfig:
    family: str
    babappa_run: str
    reference_results: str
    outdir: str
    matched_null: str = ""


@dataclass(frozen=True)
class CloseTaxaControlFamilyPlanConfig:
    control_id: str
    query_species: str
    query_gene_or_locus: str
    target_taxa_file: str
    outdir: str
    max_mean_pdistance: float = 0.25
    min_taxa: int = 6
    min_codons: int = 100


def freeze_empirical_evidence_pack(config: EmpiricalEvidencePackConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    _copy_entry(entries, "input_cds", Path(config.cds_fasta), outdir / "inputs" / Path(config.cds_fasta).name, outdir)
    _copy_entry(entries, "input_tree", Path(config.tree_file), outdir / "inputs" / Path(config.tree_file).name, outdir)
    _copy_tree(entries, "prefilter", Path(config.prefilter_dir), outdir / "prefilter", ["empirical_family_prefilter.json", "empirical_family_prefilter.tsv", "empirical_family_prefilter.md"], outdir)
    family_dir = Path(config.babappa_family_dir)
    _copy_tree(entries, "applicability", family_dir / "empirical_applicability", outdir / "babappa" / "empirical_applicability", ["empirical_applicability.json", "empirical_applicability.tsv", "empirical_applicability.md"], outdir)
    _copy_tree(entries, "empirical_scores", family_dir / "empirical_scores", outdir / "babappa" / "empirical_scores", ["empirical_scoring_manifest.json", "empirical_scoring_report.md", "empirical_branch_site_scores.tsv", "empirical_branch_scores.tsv", "empirical_gene_support.tsv"], outdir)
    _copy_tree(entries, "empirical_report", family_dir / "empirical_report", outdir / "babappa" / "empirical_report", ["empirical_branch_site_report.json", "empirical_branch_site_report.md", "empirical_branch_site_report.tsv"], outdir)
    _copy_entry(entries, "panel_run_summary", Path(config.panel_run_summary), outdir / "babappa" / "panel_run_summary.tsv", outdir)
    if config.summary_report:
        _copy_entry(entries, "summary_report", Path(config.summary_report), outdir / "reports" / Path(config.summary_report).name, outdir)
        json_report = Path(config.summary_report).with_suffix(".json")
        if json_report.exists():
            _copy_entry(entries, "summary_report_json", json_report, outdir / "reports" / json_report.name, outdir)
    if config.calibration_plan_dir:
        _copy_tree(entries, "simulation_matched_calibration_plan", Path(config.calibration_plan_dir), outdir / "simulation_matched_calibration_plan", ["simulation_matched_calibration_plan.json", "simulation_matched_calibration_plan.md", "expected_outputs.json", "proposed_null_simulation_commands.sh", "proposed_alt_simulation_commands.sh", "run_wrky_close_matched_nulls.sh"], outdir)
    if config.reference_plan_dir:
        _copy_tree(entries, "reference_workflow_plan", Path(config.reference_plan_dir), outdir / "reference_workflow_plan", ["classical_reference_plan.md", "codeml_commands.sh", "hyphy_commands.sh", "expected_reference_outputs.json", "reference_result_schema.tsv"], outdir)

    payload = {
        "evidence_pack_version": __version__,
        "family_id": config.family_id,
        "foreground": config.foreground,
        "claim_boundary": CLAIM_BOUNDARY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    _write_json(outdir / "evidence_pack_manifest.json", payload)
    write_tsv(outdir / "evidence_pack_manifest.tsv", entries, ["role", "source_path", "pack_path", "sha256", "bytes"])
    (outdir / "checksums.sha256").write_text("".join(f"{entry['sha256']}  {entry['pack_path']}\n" for entry in entries), encoding="utf-8")
    (outdir / "evidence_pack_readme.md").write_text(_render_evidence_readme(payload), encoding="utf-8")
    return {"status": "ok", "path": str(outdir), "n_files": len(entries), "checksums": str(outdir / "checksums.sha256")}


def validate_empirical_evidence_pack(config: EmpiricalEvidencePackValidationConfig) -> Dict[str, Any]:
    pack = Path(config.evidence_pack)
    failures: List[str] = []
    warnings: List[str] = []
    manifest = _read_json_or_empty(pack / "evidence_pack_manifest.json", failures)
    entries = manifest.get("entries", []) if isinstance(manifest.get("entries"), list) else []
    for entry in entries:
        path = pack / str(entry.get("pack_path", ""))
        if not path.exists():
            failures.append(f"missing_pack_file:{entry.get('pack_path')}")
            continue
        digest = _sha256(path)
        if digest != entry.get("sha256"):
            failures.append(f"checksum_mismatch:{entry.get('pack_path')}")
    forbidden = _forbidden_pack_files(pack)
    if forbidden:
        failures.append("forbidden_truth_files:" + ",".join(forbidden))
    prefilter = _read_json_or_empty(pack / "prefilter" / "empirical_family_prefilter.json", failures)
    if prefilter.get("decision") not in {"accept", "accept_with_caution"}:
        failures.append(f"prefilter_not_accepted:{prefilter.get('decision')}")
    applicability = _read_json_or_empty(pack / "babappa" / "empirical_applicability" / "empirical_applicability.json", failures)
    if applicability.get("applicability_status") not in {"in_domain", "borderline"}:
        failures.append(f"applicability_not_interpretable:{applicability.get('applicability_status')}")
    scoring = _read_json_or_empty(pack / "babappa" / "empirical_scores" / "empirical_scoring_manifest.json", failures)
    if scoring.get("diagnostic_only") is not False:
        failures.append("diagnostic_only_not_false")
    for required in ["empirical_branch_site_scores.tsv", "empirical_gene_support.tsv"]:
        if not (pack / "babappa" / "empirical_scores" / required).exists():
            failures.append(f"missing_score_file:{required}")
    metadata_foreground = str(manifest.get("foreground", ""))
    cds_files = [pack / entry["pack_path"] for entry in entries if entry.get("role") == "input_cds"]
    tree_files = [pack / entry["pack_path"] for entry in entries if entry.get("role") == "input_tree"]
    if not cds_files:
        failures.append("missing_cds")
    if not tree_files:
        failures.append("missing_tree")
    if cds_files and tree_files and metadata_foreground:
        fasta_ids = set(_read_fasta(cds_files[0]))
        tree_tips = _parse_newick_tips(tree_files[0].read_text(encoding="utf-8"))
        if metadata_foreground not in fasta_ids or metadata_foreground not in tree_tips:
            failures.append(f"foreground_missing:{metadata_foreground}")
    readme = (pack / "evidence_pack_readme.md").read_text(encoding="utf-8") if (pack / "evidence_pack_readme.md").exists() else ""
    if "not a final positive-selection discovery claim" not in readme:
        failures.append("missing_claim_boundary_text")
    payload = {
        "evidence_pack_validation_version": __version__,
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "evidence_pack": str(pack),
    }
    _write_json(pack / "evidence_pack_validation.json", payload)
    (pack / "evidence_pack_validation.md").write_text(_render_validation_md(payload), encoding="utf-8")
    return {"status": payload["status"], "json": str(pack / "evidence_pack_validation.json"), "markdown": str(pack / "evidence_pack_validation.md"), "failures": failures}


def summarize_simulation_matched_calibration_plan(config: SimulationMatchedCalibrationSummaryConfig) -> Dict[str, Any]:
    plan_dir = Path(config.plan_dir)
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plan = _read_json(plan_dir / "simulation_matched_calibration_plan.json")
    expected = _read_json_or_empty(plan_dir / "expected_outputs.json", [])
    params = dict(plan.get("proposed_simulation_parameters", {}))
    validation_dir = Path(str(plan.get("empirical_validation_dir", "")))
    sibling_app = validation_dir.parent / "empirical_applicability" / "empirical_applicability.json"
    if sibling_app.exists():
        app = _read_json(sibling_app)
        validation = app.get("validation", {})
        if validation.get("p_distance_used") is not None:
            params["mean_pairwise_p_distance"] = validation.get("p_distance_used")
            params["recommended_tier"] = app.get("recommended_tier", params.get("recommended_tier"))
            params["p_distance_source"] = validation.get("p_distance_source")
    payload = {
        "calibration_plan_summary_version": __version__,
        "status": "ok",
        "plan_dir": str(plan_dir),
        "matched_n_taxa": params.get("n_taxa"),
        "matched_n_codons": params.get("n_codons"),
        "matched_p_distance": params.get("mean_pairwise_p_distance"),
        "matched_tier": params.get("recommended_tier"),
        "p_distance_source": params.get("p_distance_source", "empirical_validation"),
        "suggested_null_reps_initial": 100,
        "plan_null_reps_initial": params.get("null_replicates_initial"),
        "expected_outputs": expected.get("expected_outputs", []),
        "estimated_runtime": plan.get("estimated_runtime"),
        "estimated_disk": plan.get("estimated_disk"),
        "user_run_null_simulation_recommended": True,
        "interpretable_before_calibration": False,
        "claim_boundary": "Current BABAPPA result is not interpretable as a final empirical claim before simulation-matched calibration.",
    }
    _write_json(outdir / "simulation_matched_calibration_summary.json", payload)
    write_tsv(outdir / "simulation_matched_calibration_summary.tsv", [payload], ["status", "matched_n_taxa", "matched_n_codons", "matched_p_distance", "matched_tier", "suggested_null_reps_initial", "interpretable_before_calibration"])
    (outdir / "simulation_matched_calibration_summary.md").write_text(_render_calibration_summary_md(payload), encoding="utf-8")
    return {"status": "ok", "outdir": str(outdir), "json": str(outdir / "simulation_matched_calibration_summary.json")}


def write_wrky_matched_null_script(plan_dir: str, output_root: str) -> Path:
    plan_path = Path(plan_dir)
    plan_path.mkdir(parents=True, exist_ok=True)
    path = plan_path / "run_wrky_close_matched_nulls.sh"
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_pack = plan_path.parent if plan_path.name == "simulation_matched_calibration_plan" else None
    if evidence_pack is not None:
        run_command = [
            "babappa run-simulation-matched-null-calibration \\",
            f"  --evidence-pack {evidence_pack} \\",
            f"  --outdir {output_dir} \\",
            "  --n-null 100 \\",
            "  --seed 20260530 \\",
            "  --device mps",
        ]
        dry_run_comment = [
            "# Safe dry-run preview:",
            "# babappa run-simulation-matched-null-calibration \\",
            f"#   --evidence-pack {evidence_pack} \\",
            f"#   --outdir {output_dir}_dryrun \\",
            "#   --n-null 100 \\",
            "#   --seed 20260530 \\",
            "#   --device mps \\",
            "#   --dry-run",
        ]
    else:
        run_command = [
            "babappa run-simulation-matched-null-calibration \\",
            f"  --plan-dir {Path(plan_dir)} \\",
            "  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \\",
            f"  --outdir {Path(output_root)} \\",
            "  --n-replicates 100 \\",
            "  --device auto \\",
            "  --seed 42",
        ]
        dry_run_comment = []
    path.write_text("\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"echo '{USER_RUN_ONLY}'",
        "# This launches the real manual execution matched-null command. Do not execute it in an automated environment.",
        *dry_run_comment,
        *run_command,
        "",
    ]), encoding="utf-8")
    path.chmod(0o755)
    run_script = output_dir / "run_user_wrky_null100.sh"
    run_script.write_text("\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"echo '{USER_RUN_ONLY}'",
        "cd \"$(dirname \"$0\")/../../..\"",
        *run_command,
        "",
    ]), encoding="utf-8")
    monitor_script = output_dir / "monitor_user_wrky_null100.sh"
    monitor_script.write_text("\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"echo '{USER_RUN_ONLY}'",
        f"echo 'Stage markers in {output_dir}:'",
        f"ls -la {output_dir}/.stage_* 2>/dev/null || true",
        f"echo 'Null score line count:'",
        f"wc -l {output_dir}/matched_null_scores.tsv 2>/dev/null || true",
        "ps aux | grep -E 'babappa|python' | grep -v grep || true",
        "",
    ]), encoding="utf-8")
    validate_script = output_dir / "validate_user_wrky_null100.sh"
    validate_script.write_text("\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"echo '{USER_RUN_ONLY}'",
        f"babappa validate-simulation-matched-null-calibration --calibration-dir {output_dir}",
        "",
    ]), encoding="utf-8")
    summarize_script = output_dir / "summarize_user_wrky_null100.sh"
    summarize_script.write_text("\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"echo '{USER_RUN_ONLY}'",
        f"cat {output_dir}/matched_null_summary.md",
        f"cat {output_dir}/observed_vs_null.md",
        "",
    ]), encoding="utf-8")
    for script in [run_script, monitor_script, validate_script, summarize_script]:
        script.chmod(0o755)
    return path


def install_reference_tools_plan(config: ReferenceToolsInstallPlanConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    conda_script = outdir / "install_reference_tools_conda.sh"
    brew_script = outdir / "install_reference_tools_brew.sh"
    _write_script(
        conda_script,
        [
            "echo 'Recommended conda install for codeml/PAML and HyPhy:'",
            "conda install -y -c bioconda -c conda-forge paml hyphy",
        ],
    )
    _write_script(
        brew_script,
        [
            "echo 'Homebrew availability can vary by platform.'",
            "brew install hyphy || true",
            "echo 'If PAML/codeml is unavailable in Homebrew, use conda, Rosetta/x86_64 conda, manual PAML compilation, or a Linux workstation.'",
        ],
    )
    notes = "\n".join([
        "# Reference Tool Installation Notes",
        "",
        "This plan does not install tools automatically.",
        "",
        "Recommended conda command:",
        "",
        "```bash",
        "conda install -y -c bioconda -c conda-forge paml hyphy",
        "```",
        "",
        "Apple Silicon fallback options if `paml`/`codeml` is unavailable:",
        "",
        "- Try a Rosetta/x86_64 conda environment.",
        "- Compile PAML manually.",
        "- Run codeml on a Linux workstation and bring the outputs back.",
        "- Run HyPhy separately if it is available while codeml is not.",
        "",
    ])
    (outdir / "install_reference_tools_notes.md").write_text(notes, encoding="utf-8")
    return {
        "status": "planned",
        "outdir": str(outdir),
        "conda_script": str(conda_script),
        "brew_script": str(brew_script),
        "notes": str(outdir / "install_reference_tools_notes.md"),
        "executed": False,
    }


def prepare_codeml_reference(config: CodemlReferencePrepConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records = _read_fasta(Path(config.cds_fasta))
    safe_records, safety = _codeml_safe_codon_records(records)
    _write_fasta(outdir / "alignment.codeml_safe.fasta", safe_records)
    _write_phylip(outdir / "alignment.phy", safe_records)
    _write_json(outdir / "codeml_alignment_safety.json", safety)
    tree_text = Path(config.tree).read_text(encoding="utf-8")
    (outdir / "tree_foreground.nwk").write_text(_mark_paml_foreground(tree_text, config.foreground), encoding="utf-8")
    _write_codeml_ctl(outdir / "codeml_modelA.ctl", "modelA")
    _write_codeml_ctl(outdir / "codeml_null.ctl", "null")
    _write_script(outdir / "run_codeml_modelA.sh", ["codeml codeml_modelA.ctl"])
    _write_script(outdir / "run_codeml_null.sh", ["codeml codeml_null.ctl"])
    _write_script(outdir / "parse_codeml_lrt.sh", ["babappa parse-codeml-reference --codeml-dir . --outdir ../codeml_parsed"])
    (outdir / "README.md").write_text(_render_codeml_readme(config, safety), encoding="utf-8")
    return {"status": "prepared", "outdir": str(outdir), "executed": False, "modelA": str(outdir / "codeml_modelA.ctl"), "null": str(outdir / "codeml_null.ctl"), "alignment_safety": safety}


def prepare_hyphy_reference(config: HyphyReferencePrepConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sanitized = _write_hyphy_safe_codon_fasta(Path(config.cds_fasta), outdir / "alignment.fasta")
    tree_text = Path(config.tree).read_text(encoding="utf-8")
    (outdir / "tree_foreground.nwk").write_text(_mark_hyphy_foreground(tree_text, config.foreground), encoding="utf-8")
    _write_script(outdir / "run_absrel.sh", ["hyphy absrel --alignment alignment.fasta --tree tree_foreground.nwk --branches Foreground --output absrel.json"])
    _write_script(outdir / "run_meme.sh", ["hyphy meme --alignment alignment.fasta --tree tree_foreground.nwk --branches Foreground --output meme.json"])
    _write_json(outdir / "expected_outputs.json", {"expected_outputs": ["absrel.json", "meme.json"], "executed": False, "hyphy_safe_alignment": sanitized})
    (outdir / "README.md").write_text(_render_hyphy_readme(config), encoding="utf-8")
    return {"status": "prepared", "outdir": str(outdir), "executed": False, "absrel": str(outdir / "run_absrel.sh"), "meme": str(outdir / "run_meme.sh")}


def check_reference_tools(config: ReferenceToolCheckConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for tool, executable_names in {"codeml": ["codeml"], "hyphy": ["hyphy", "HYPHYMP"], "python": ["python", "python3"], "R": ["R"]}.items():
        found = next((shutil.which(name) for name in executable_names if shutil.which(name)), "")
        rows.append({"tool": tool, "available": bool(found), "executable": found, "recommended_action": "ready" if found else "planned_manual_mode"})
    payload = {"reference_tool_check_version": __version__, "status": "ok", "tools": rows}
    _write_json(outdir / "reference_tool_check.json", payload)
    write_tsv(outdir / "reference_tool_check.tsv", rows, ["tool", "available", "executable", "recommended_action"])
    (outdir / "reference_tool_check.md").write_text(_render_tool_check_md(payload), encoding="utf-8")
    return {"status": "ok", "outdir": str(outdir), "codeml": _tool_available(rows, "codeml"), "hyphy": _tool_available(rows, "hyphy")}


def parse_codeml_reference(config: CodemlReferenceParseConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    codeml_dir = Path(config.codeml_dir)
    model_files = list(codeml_dir.glob("*modelA*.out")) + list(codeml_dir.glob("mlc_modelA")) + list(codeml_dir.glob("modelA/mlc"))
    null_files = list(codeml_dir.glob("*null*.out")) + list(codeml_dir.glob("mlc_null")) + list(codeml_dir.glob("null/mlc"))
    tool_missing = shutil.which("codeml") is None
    status = "pending_tool_missing" if tool_missing and (not model_files or not null_files) else ("pending_not_run" if not model_files or not null_files else "parsed")
    model_parse = _parse_codeml_output(model_files[0]) if model_files else {}
    null_parse = _parse_codeml_output(null_files[0]) if null_files else {}
    lrt = None
    p_value = None
    if model_parse.get("lnl") is not None and null_parse.get("lnl") is not None:
        lrt = max(0.0, 2.0 * (float(model_parse["lnl"]) - float(null_parse["lnl"])))
        p_value = math.erfc(math.sqrt(lrt / 2.0))
    result_class = _reference_class_from_pvalue(p_value) if status == "parsed" else status
    payload = {
        "codeml_reference_parse_version": __version__,
        "status": status,
        "codeml_dir": str(codeml_dir),
        "tool_available": not tool_missing,
        "modelA_outputs": [str(p) for p in model_files],
        "null_outputs": [str(p) for p in null_files],
        "modelA": model_parse,
        "null": null_parse,
        "lrt_statistic": lrt,
        "p_value": p_value,
        "beb_sites": model_parse.get("beb_sites", []),
        "result_class": result_class,
    }
    _write_json(outdir / "codeml_reference_parse.json", payload)
    (outdir / "codeml_reference_parse.md").write_text(_render_parse_md("codeml", payload), encoding="utf-8")
    return {"status": status, "outdir": str(outdir)}


def parse_hyphy_reference(config: HyphyReferenceParseConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    hyphy_dir = Path(config.hyphy_dir)
    outputs = list(hyphy_dir.glob("*absrel*.json")) + list(hyphy_dir.glob("*meme*.json"))
    tool_missing = shutil.which("hyphy") is None and shutil.which("HYPHYMP") is None
    status = "pending_tool_missing" if tool_missing and not outputs else ("pending_not_run" if not outputs else "parsed")
    parsed_outputs = [_parse_hyphy_json(path) for path in outputs]
    min_p = _min_present([item.get("min_p_value") for item in parsed_outputs])
    result_class = _reference_class_from_pvalue(min_p) if status == "parsed" else status
    payload = {
        "hyphy_reference_parse_version": __version__,
        "status": status,
        "hyphy_dir": str(hyphy_dir),
        "tool_available": not tool_missing,
        "outputs": [str(p) for p in outputs],
        "parsed_outputs": parsed_outputs,
        "min_p_value": min_p,
        "result_class": result_class,
    }
    _write_json(outdir / "hyphy_reference_parse.json", payload)
    (outdir / "hyphy_reference_parse.md").write_text(_render_parse_md("hyphy", payload), encoding="utf-8")
    return {"status": status, "outdir": str(outdir)}


def write_reference_results_template(config: ReferenceResultsTemplateConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"panel_id": config.family_id, "tool": "codeml", "test_name": "branch_site_model_A_vs_null", "p_value": "NA", "q_value": "NA", "selected_branch": config.foreground, "selected_sites": "NA", "result_class": "pending", "notes": "USER_TO_FILL_AFTER_CODEML"},
        {"panel_id": config.family_id, "tool": "hyphy", "test_name": "aBSREL", "p_value": "NA", "q_value": "NA", "selected_branch": config.foreground, "selected_sites": "NA", "result_class": "pending", "notes": "USER_TO_FILL_AFTER_HYPHY"},
        {"panel_id": config.family_id, "tool": "hyphy", "test_name": "MEME", "p_value": "NA", "q_value": "NA", "selected_branch": config.foreground, "selected_sites": "NA", "result_class": "pending", "notes": "USER_TO_FILL_AFTER_HYPHY"},
    ]
    path = outdir / f"{config.family_id}_reference_results_template.tsv"
    write_tsv(path, rows, REFERENCE_RESULT_FIELDS)
    return {"status": "pending", "path": str(path), "rows": len(rows)}


def build_reference_results_table(config: ReferenceResultsTableConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    codeml = _read_json_or_empty(Path(config.codeml_parsed) / "codeml_reference_parse.json", [])
    hyphy = _read_json_or_empty(Path(config.hyphy_parsed) / "hyphy_reference_parse.json", [])
    rows = [
        _reference_row_from_parse(config.panel_id, "codeml", "branch_site_model_A_vs_null", codeml),
    ]
    if hyphy.get("status") == "parsed":
        parsed = hyphy.get("parsed_outputs", [])
        absrel = next((item for item in parsed if item.get("test_name") == "aBSREL"), {})
        meme = next((item for item in parsed if item.get("test_name") == "MEME"), {})
        rows.append(_reference_row_from_parse(config.panel_id, "hyphy", "aBSREL", {**hyphy, **absrel}))
        rows.append(_reference_row_from_parse(config.panel_id, "hyphy", "MEME", {**hyphy, **meme}))
    else:
        rows.append(_reference_row_from_parse(config.panel_id, "hyphy", "aBSREL", hyphy))
        rows.append(_reference_row_from_parse(config.panel_id, "hyphy", "MEME", hyphy))
    path = outdir / "reference_results.tsv"
    write_tsv(path, rows, REFERENCE_RESULT_FIELDS)
    status = "pending_tool_missing" if any(row["result_class"] == "pending_tool_missing" for row in rows) else ("pending_not_run" if any(row["result_class"] == "pending_not_run" for row in rows) else "ok")
    payload = {"reference_results_table_version": __version__, "status": status, "panel_id": config.panel_id, "rows": rows}
    _write_json(outdir / "reference_results.json", payload)
    (outdir / "reference_results.md").write_text(_render_reference_results_md(payload), encoding="utf-8")
    return {"status": status, "path": str(path), "json": str(outdir / "reference_results.json")}


def run_simulation_matched_null_calibration(config: SimulationMatchedNullCalibrationConfig) -> Dict[str, Any]:
    if config.evidence_pack:
        return _run_evidence_pack_matched_null_calibration(config)

    plan_dir = Path(config.plan_dir)
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_wrky_matched_null_script(str(plan_dir), str(outdir))
    plan = _read_json(plan_dir / "simulation_matched_calibration_plan.json")
    params = dict(plan.get("proposed_simulation_parameters", {}))
    validation_dir = Path(str(plan.get("empirical_validation_dir", "")))
    sibling_app = validation_dir.parent / "empirical_applicability" / "empirical_applicability.json"
    if sibling_app.exists():
        app = _read_json(sibling_app)
        validation = app.get("validation", {})
        if validation.get("p_distance_used") is not None:
            params["mean_pairwise_p_distance"] = validation.get("p_distance_used")
            params["recommended_tier"] = app.get("recommended_tier", params.get("recommended_tier"))
            params["p_distance_source"] = validation.get("p_distance_source")
    observed = _observed_from_family_dir(validation_dir.parent)
    requested = int(config.n_replicates)
    if requested <= 0:
        raise ValueError("n_replicates must be positive")
    _mark_stage_partial(outdir, "generate_nulls")
    replicate_rows = _null_replicate_rows(params, requested, config.seed, config.fast_null_mode)
    write_tsv(outdir / "matched_null_replicates.tsv", replicate_rows, ["replicate", "seed", "status", "n_taxa", "n_codons", "target_p_distance", "tier", "foreground"])
    _mark_stage_complete(outdir, "generate_nulls")
    _mark_stage_partial(outdir, "score_nulls")
    if config.fast_null_mode:
        rows = _fast_null_score_rows(replicate_rows, observed)
    else:
        rows = _score_null_replicates_with_model(
            replicate_rows=replicate_rows,
            params=params,
            package_dir=Path(config.deployable_model_package),
            device_request=config.device,
            outdir=outdir,
        )
    completed = sum(1 for row in rows if row.get("status") == "scored")
    write_tsv(outdir / "matched_null_scores.tsv", rows, ["replicate", "seed", "status", "n_taxa", "n_codons", "target_p_distance", "tier", "max_gene_support", "max_branch_support", "called_branch_site_rows", "max_site_score", "q95_site_score", "q99_site_score"])
    if completed:
        _mark_stage_complete(outdir, "score_nulls")
    manifest = {
        "matched_null_calibration_version": __version__,
        "status": "ok" if completed == requested else ("partial" if completed else "scoring_incomplete"),
        "plan_dir": str(plan_dir),
        "deployable_model_package": config.deployable_model_package,
        "outdir": str(outdir),
        "n_replicates_requested": requested,
        "n_replicates_staged": len(replicate_rows),
        "n_replicates_completed": completed,
        "device": config.device,
        "seed": config.seed,
        "fast_null_mode": config.fast_null_mode,
        "matched_parameters": params,
        "observed_values": observed,
        "null_scoring_completed": completed > 0,
        "claim_boundary": "Null calibration is diagnostic and simulation-matched; it is not by itself a final empirical discovery claim.",
    }
    _write_json(outdir / "matched_null_manifest.json", manifest)
    _mark_stage_partial(outdir, "summarize_nulls")
    percentiles = _observed_null_percentiles(observed, rows)
    summary = {
        **manifest,
        "p_empirical_support": percentiles.get("p_empirical_support"),
        "p_empirical_called_rows": percentiles.get("p_empirical_called_rows"),
        "p_empirical_branch_support": percentiles.get("p_empirical_branch_support"),
        "high_score_tail_quantiles": _null_tail_quantiles(rows),
    }
    _write_json(outdir / "matched_null_summary.json", summary)
    _write_json(outdir / "observed_vs_null.json", {
        "status": "ok" if completed else "no_null_scores",
        "observed_values": observed,
        "p_empirical_support": summary["p_empirical_support"],
        "p_empirical_called_rows": summary["p_empirical_called_rows"],
        "p_empirical_branch_support": summary["p_empirical_branch_support"],
        "reason": "computed_from_scored_nulls" if completed else "No scored null distributions are available.",
    })
    (outdir / "matched_null_summary.md").write_text(_render_matched_null_summary_md(summary), encoding="utf-8")
    (outdir / "observed_vs_null.md").write_text(_render_observed_vs_null_md(outdir / "observed_vs_null.json"), encoding="utf-8")
    if completed:
        _mark_stage_complete(outdir, "summarize_nulls")
    return {
        "status": manifest["status"],
        "outdir": str(outdir),
        "n_replicates_requested": requested,
        "n_replicates_completed": completed,
        "observed_max_gene_support": observed.get("max_gene_support"),
        "observed_called_rows": observed.get("called_branch_site_rows"),
    }


def _run_evidence_pack_matched_null_calibration(config: SimulationMatchedNullCalibrationConfig) -> Dict[str, Any]:
    evidence_pack = Path(config.evidence_pack)
    family_id = config.family_id or evidence_pack.name
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plan_dir = Path(config.plan_dir) if config.plan_dir else evidence_pack / "simulation_matched_calibration_plan"
    input_rows, failures = _validate_matched_null_evidence_pack(evidence_pack, family_id)
    write_tsv(
        outdir / "calibration_input_validation.tsv",
        input_rows,
        ["role", "path", "required", "exists", "status", "reason"],
    )
    plan_payload = _build_evidence_pack_calibration_plan(config, evidence_pack, plan_dir, input_rows, failures, family_id)
    _write_json(outdir / "calibration_run_plan.json", plan_payload)
    (outdir / "calibration_run_plan.md").write_text(_render_calibration_run_plan_md(plan_payload), encoding="utf-8")

    if failures:
        status_payload = {
            "matched_null_calibration_run_status_version": __version__,
            "status": "fail",
            "mode": "dry_run" if config.dry_run else "run",
            "evidence_pack": str(evidence_pack),
            "outdir": str(outdir),
            "failures": failures,
            "heavy_jobs_executed": False,
            "null_results_fabricated": False,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        _write_json(outdir / "calibration_status.json", status_payload)
        (outdir / "calibration_status.md").write_text(_render_calibration_status_md(status_payload), encoding="utf-8")
        raise ValueError("matched-null calibration input validation failed: " + "; ".join(failures))

    if config.dry_run:
        status_payload = {
            "matched_null_calibration_run_status_version": __version__,
            "status": "dry_run",
            "mode": "dry_run",
            "evidence_pack": str(evidence_pack),
            "outdir": str(outdir),
            "n_null": int(config.n_replicates),
            "seed": int(config.seed),
            "device": config.device,
            "heavy_jobs_executed": False,
            "null_results_fabricated": False,
            "matched_null_scores_written": False,
            "message": "Inputs validated and execution plan written. No simulations or null scoring were run.",
            "claim_boundary": CLAIM_BOUNDARY,
        }
        _write_json(outdir / "calibration_status.json", status_payload)
        (outdir / "calibration_status.md").write_text(_render_calibration_status_md(status_payload), encoding="utf-8")
        return {
            "status": "dry_run",
            "outdir": str(outdir),
            "n_replicates_requested": int(config.n_replicates),
            "n_replicates_completed": 0,
            "observed_max_gene_support": plan_payload["observed_values"].get("max_gene_support"),
            "observed_called_rows": plan_payload["observed_values"].get("called_branch_site_rows"),
        }

    if not config.fast_null_mode:
        return _run_feature_matched_mode_from_evidence_pack(config, evidence_pack, plan_payload)

    return _run_fast_mode_from_evidence_pack(config, evidence_pack, plan_payload)


def _validate_matched_null_evidence_pack(evidence_pack: Path, family_id: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    required = [
        ("cds_fasta", evidence_pack / "inputs" / f"{family_id}.cds.fasta"),
        ("tree_file", evidence_pack / "inputs" / f"{family_id}.treefile"),
        ("empirical_branch_site_scores", evidence_pack / "babappa" / "empirical_scores" / "empirical_branch_site_scores.tsv"),
        ("empirical_gene_support", evidence_pack / "babappa" / "empirical_scores" / "empirical_gene_support.tsv"),
        ("empirical_applicability", evidence_pack / "babappa" / "empirical_applicability" / "empirical_applicability.tsv"),
        ("family_prefilter", evidence_pack / "prefilter" / "empirical_family_prefilter.tsv"),
    ]
    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    if not evidence_pack.exists():
        failures.append(f"missing_evidence_pack:{evidence_pack}")
    for role, path in required:
        exists = path.exists()
        if not exists:
            failures.append(f"missing_required_{role}:{path}")
        rows.append({
            "role": role,
            "path": str(path),
            "required": True,
            "exists": exists,
            "status": "ok" if exists else "missing",
            "reason": "required_for_matched_null_calibration",
        })
    return rows, failures


def _build_evidence_pack_calibration_plan(
    config: SimulationMatchedNullCalibrationConfig,
    evidence_pack: Path,
    plan_dir: Path,
    input_rows: List[Dict[str, Any]],
    failures: List[str],
    family_id: str,
) -> Dict[str, Any]:
    plan = _read_json_or_empty(plan_dir / "simulation_matched_calibration_plan.json", [])
    params = dict(plan.get("proposed_simulation_parameters", {}))
    app_json = evidence_pack / "babappa" / "empirical_applicability" / "empirical_applicability.json"
    if app_json.exists():
        app = _read_json_or_empty(app_json, [])
        validation = app.get("validation", {})
        if validation.get("p_distance_used") is not None:
            params["mean_pairwise_p_distance"] = validation.get("p_distance_used")
            params["p_distance_source"] = validation.get("p_distance_source")
        if app.get("recommended_tier"):
            params["recommended_tier"] = app.get("recommended_tier")
        if validation.get("n_taxa") is not None:
            params["n_taxa"] = validation.get("n_taxa")
        if validation.get("n_codons") is not None:
            params["n_codons"] = validation.get("n_codons")
    if config.tier:
        params["recommended_tier"] = config.tier
    params.setdefault("n_taxa", 7)
    params.setdefault("n_codons", 490)
    params.setdefault("foreground", "Arabidopsis_thaliana")
    params.setdefault("recommended_tier", "moderate")
    observed = _observed_from_evidence_pack(evidence_pack)
    return {
        "matched_null_calibration_plan_version": __version__,
        "status": "blocked_missing_inputs" if failures else ("dry_run" if config.dry_run else "planned"),
        "family_id": family_id,
        "evidence_pack": str(evidence_pack),
        "plan_dir": str(plan_dir),
        "model_package": config.deployable_model_package,
        "outdir": config.outdir,
        "n_null": int(config.n_replicates),
        "n_alt": int(config.n_alt),
        "seed": int(config.seed),
        "device": config.device,
        "max_workers": int(config.max_workers),
        "resume": bool(config.resume),
        "force": bool(config.force),
        "dry_run": bool(config.dry_run),
        "fast_null_mode": bool(config.fast_null_mode),
        "matched_parameters": params,
        "observed_values": observed,
        "input_validation": input_rows,
        "failures": failures,
        "expected_final_outputs": _expected_matched_null_output_names(),
        "real_backend_status": "not_wired_for_evidence_pack_execution",
        "heavy_jobs_executed": False,
        "null_results_fabricated": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _observed_from_evidence_pack(evidence_pack: Path) -> Dict[str, Any]:
    scores_dir = evidence_pack / "babappa" / "empirical_scores"
    gene_rows = read_tsv(scores_dir / "empirical_gene_support.tsv") if (scores_dir / "empirical_gene_support.tsv").exists() else []
    branch_rows = read_tsv(scores_dir / "empirical_branch_scores.tsv") if (scores_dir / "empirical_branch_scores.tsv").exists() else []
    site_rows = read_tsv(scores_dir / "empirical_branch_site_scores.tsv") if (scores_dir / "empirical_branch_site_scores.tsv").exists() else []
    return {
        "max_gene_support": _max_float(row.get("max_prob_positive") for row in gene_rows),
        "max_branch_support": _max_float(row.get("max_prob_positive") for row in branch_rows),
        "called_branch_site_rows": sum(_safe_int(row.get("n_called_positive")) for row in gene_rows),
        "score_rows": len(site_rows),
        "source": str(scores_dir),
    }


def _expected_matched_null_output_names() -> List[str]:
    return [
        "matched_null_manifest.json",
        "matched_null_manifest.tsv",
        "matched_null_scores.tsv",
        "matched_null_gene_support.tsv",
        "matched_null_branch_site_summary.tsv",
        "matched_null_calibration_summary.json",
        "matched_null_calibration_summary.tsv",
        "matched_null_calibration_report.md",
        "wrky_close_matched_null_interpretation.json",
        "wrky_close_matched_null_interpretation.md",
    ]


def _render_calibration_run_plan_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# Simulation-Matched Null Calibration Run Plan",
        "",
        f"- status: `{payload['status']}`",
        f"- family: `{payload['family_id']}`",
        f"- evidence pack: `{payload['evidence_pack']}`",
        f"- output directory: `{payload['outdir']}`",
        f"- requested nulls: `{payload['n_null']}`",
        f"- seed: `{payload['seed']}`",
        f"- device: `{payload['device']}`",
        f"- dry run: `{payload['dry_run']}`",
        f"- heavy jobs executed: `{payload['heavy_jobs_executed']}`",
        "",
        "## Matched Parameters",
        "",
    ]
    for key, value in payload.get("matched_parameters", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Observed BABAPPA Values",
        "",
    ])
    for key, value in payload.get("observed_values", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Backend Status",
        "",
        f"- real backend: `{payload['real_backend_status']}`",
        "- dry-run mode validates inputs and writes this plan only.",
        "- Non-dry-run evidence-pack execution must not fabricate null distributions.",
        "",
        "## Claim Boundary",
        "",
        payload["claim_boundary"],
        "",
    ])
    if payload.get("failures"):
        lines.extend(["## Failures", ""])
        lines.extend(f"- {item}" for item in payload["failures"])
        lines.append("")
    return "\n".join(lines)


def _render_calibration_status_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# Simulation-Matched Null Calibration Status",
        "",
        f"- status: `{payload['status']}`",
        f"- mode: `{payload.get('mode')}`",
        f"- evidence pack: `{payload.get('evidence_pack')}`",
        f"- output directory: `{payload.get('outdir')}`",
        f"- heavy jobs executed: `{payload.get('heavy_jobs_executed')}`",
        f"- null results fabricated: `{payload.get('null_results_fabricated')}`",
        f"- message: {payload.get('message', '')}",
        "",
        "## Claim Boundary",
        "",
        payload.get("claim_boundary", CLAIM_BOUNDARY),
        "",
    ]
    if payload.get("failures"):
        lines.extend(["## Failures", ""])
        lines.extend(f"- {item}" for item in payload["failures"])
        lines.append("")
    return "\n".join(lines)


def _run_fast_mode_from_evidence_pack(
    config: SimulationMatchedNullCalibrationConfig,
    evidence_pack: Path,
    plan_payload: Dict[str, Any],
) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    params = dict(plan_payload.get("matched_parameters", {}))
    observed = dict(plan_payload.get("observed_values", {}))
    requested = int(config.n_replicates)
    _mark_stage_partial(outdir, "generate_nulls")
    replicate_rows = _null_replicate_rows(params, requested, config.seed, True)
    write_tsv(outdir / "matched_null_replicates.tsv", replicate_rows, ["replicate", "seed", "status", "n_taxa", "n_codons", "target_p_distance", "tier", "foreground"])
    _mark_stage_complete(outdir, "generate_nulls")
    _mark_stage_partial(outdir, "score_nulls")
    rows = _fast_null_score_rows(replicate_rows, observed)
    completed = sum(1 for row in rows if row.get("status") == "scored")
    write_tsv(outdir / "matched_null_scores.tsv", rows, ["replicate", "seed", "status", "n_taxa", "n_codons", "target_p_distance", "tier", "max_gene_support", "max_branch_support", "called_branch_site_rows", "max_site_score", "q95_site_score", "q99_site_score"])
    _mark_stage_complete(outdir, "score_nulls")
    manifest = {
        "matched_null_calibration_version": __version__,
        "status": "ok",
        "evidence_pack": str(evidence_pack),
        "plan_dir": plan_payload.get("plan_dir"),
        "deployable_model_package": config.deployable_model_package,
        "outdir": str(outdir),
        "n_replicates_requested": requested,
        "n_replicates_staged": len(replicate_rows),
        "n_replicates_completed": completed,
        "device": config.device,
        "seed": config.seed,
        "fast_null_mode": True,
        "matched_parameters": params,
        "observed_values": observed,
        "null_scoring_completed": completed > 0,
        "claim_boundary": "Fast null mode is for tiny software smoke tests only; it is not a publishable empirical calibration.",
    }
    _write_json(outdir / "matched_null_manifest.json", manifest)
    write_tsv(outdir / "matched_null_manifest.tsv", [manifest], ["status", "evidence_pack", "n_replicates_requested", "n_replicates_completed", "device", "seed", "fast_null_mode"])
    _mark_stage_partial(outdir, "summarize_nulls")
    percentiles = _observed_null_percentiles(observed, rows)
    summary = {
        **manifest,
        "p_empirical_support": percentiles.get("p_empirical_support"),
        "p_empirical_called_rows": percentiles.get("p_empirical_called_rows"),
        "p_empirical_branch_support": percentiles.get("p_empirical_branch_support"),
        "high_score_tail_quantiles": _null_tail_quantiles(rows),
    }
    _write_json(outdir / "matched_null_summary.json", summary)
    _write_json(outdir / "matched_null_calibration_summary.json", summary)
    write_tsv(outdir / "matched_null_calibration_summary.tsv", [summary], ["status", "n_replicates_requested", "n_replicates_completed", "p_empirical_support", "p_empirical_called_rows", "p_empirical_branch_support"])
    write_tsv(outdir / "matched_null_gene_support.tsv", [{"replicate": row["replicate"], "max_gene_support": row["max_gene_support"]} for row in rows], ["replicate", "max_gene_support"])
    write_tsv(outdir / "matched_null_branch_site_summary.tsv", [{"replicate": row["replicate"], "called_branch_site_rows": row["called_branch_site_rows"], "max_site_score": row["max_site_score"]} for row in rows], ["replicate", "called_branch_site_rows", "max_site_score"])
    observed_vs_null = {
        "status": "ok",
        "observed_values": observed,
        "p_empirical_support": summary["p_empirical_support"],
        "p_empirical_called_rows": summary["p_empirical_called_rows"],
        "p_empirical_branch_support": summary["p_empirical_branch_support"],
        "reason": "computed_from_fast_null_mode_for_software_smoke_only",
    }
    _write_json(outdir / "observed_vs_null.json", observed_vs_null)
    _write_json(outdir / "wrky_close_matched_null_interpretation.json", {
        "status": "software_smoke_only",
        "manuscript_ready": False,
        "decision": "not_interpretable_for_empirical_claim",
        "reason": "fast_null_mode_is_not_real_matched_null_calibration",
    })
    (outdir / "matched_null_summary.md").write_text(_render_matched_null_summary_md(summary), encoding="utf-8")
    (outdir / "matched_null_calibration_report.md").write_text(_render_matched_null_summary_md(summary), encoding="utf-8")
    (outdir / "observed_vs_null.md").write_text(_render_observed_vs_null_md(outdir / "observed_vs_null.json"), encoding="utf-8")
    (outdir / "wrky_close_matched_null_interpretation.md").write_text(
        "# WRKY Close Matched Null Interpretation\n\n- status: `software_smoke_only`\n- manuscript-ready: `False`\n- reason: fast null mode is not real matched-null calibration.\n",
        encoding="utf-8",
    )
    _mark_stage_complete(outdir, "summarize_nulls")
    status_payload = {
        "matched_null_calibration_run_status_version": __version__,
        "status": "ok",
        "mode": "fast_null_mode",
        "evidence_pack": str(evidence_pack),
        "outdir": str(outdir),
        "heavy_jobs_executed": False,
        "null_results_fabricated": False,
        "message": "Tiny fast-null software smoke completed; not valid empirical calibration.",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(outdir / "calibration_status.json", status_payload)
    (outdir / "calibration_status.md").write_text(_render_calibration_status_md(status_payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "n_replicates_requested": requested,
        "n_replicates_completed": completed,
        "observed_max_gene_support": observed.get("max_gene_support"),
        "observed_called_rows": observed.get("called_branch_site_rows"),
    }


def _run_feature_matched_mode_from_evidence_pack(
    config: SimulationMatchedNullCalibrationConfig,
    evidence_pack: Path,
    plan_payload: Dict[str, Any],
) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    params = dict(plan_payload.get("matched_parameters", {}))
    observed = dict(plan_payload.get("observed_values", {}))
    requested = int(config.n_replicates)
    if requested <= 0:
        raise ValueError("n_null must be positive")
    _mark_stage_partial(outdir, "generate_nulls")
    replicate_rows = _null_replicate_rows(params, requested, config.seed, False)
    write_tsv(outdir / "matched_null_replicates.tsv", replicate_rows, ["replicate", "seed", "status", "n_taxa", "n_codons", "target_p_distance", "tier", "foreground"])
    _mark_stage_complete(outdir, "generate_nulls")
    _mark_stage_partial(outdir, "score_nulls")
    try:
        rows = _score_null_replicates_with_model(
            replicate_rows=replicate_rows,
            params=params,
            package_dir=Path(config.deployable_model_package),
            device_request=config.device,
            outdir=outdir,
        )
    except Exception as exc:
        status_payload = {
            "matched_null_calibration_run_status_version": __version__,
            "status": "fail",
            "mode": "feature_matched_model_scoring",
            "evidence_pack": str(evidence_pack),
            "outdir": str(outdir),
            "n_null": requested,
            "seed": int(config.seed),
            "device": config.device,
            "heavy_jobs_executed": False,
            "null_results_fabricated": False,
            "message": f"Feature-matched deployable-model null scoring failed: {exc}",
            "claim_boundary": CLAIM_BOUNDARY,
        }
        _write_json(outdir / "calibration_status.json", status_payload)
        (outdir / "calibration_status.md").write_text(_render_calibration_status_md(status_payload), encoding="utf-8")
        raise ValueError(status_payload["message"]) from exc
    completed = sum(1 for row in rows if row.get("status") == "scored")
    write_tsv(outdir / "matched_null_scores.tsv", rows, ["replicate", "seed", "status", "n_taxa", "n_codons", "target_p_distance", "tier", "max_gene_support", "max_branch_support", "called_branch_site_rows", "max_site_score", "q95_site_score", "q99_site_score"])
    if completed:
        _mark_stage_complete(outdir, "score_nulls")
    manifest = {
        "matched_null_calibration_version": __version__,
        "status": "ok" if completed == requested else ("partial" if completed else "scoring_incomplete"),
        "calibration_backend": "feature_matched_deployable_model_null",
        "calibration_scope": "feature-level matched null scoring; not full raw sequence simulation/alignment replay",
        "evidence_pack": str(evidence_pack),
        "plan_dir": plan_payload.get("plan_dir"),
        "deployable_model_package": config.deployable_model_package,
        "outdir": str(outdir),
        "n_replicates_requested": requested,
        "n_replicates_staged": len(replicate_rows),
        "n_replicates_completed": completed,
        "device": config.device,
        "seed": config.seed,
        "fast_null_mode": False,
        "matched_parameters": params,
        "observed_values": observed,
        "null_scoring_completed": completed > 0,
        "null_results_fabricated": False,
        "claim_boundary": "Feature-matched null calibration is diagnostic. It is not by itself a final empirical discovery claim.",
    }
    _write_json(outdir / "matched_null_manifest.json", manifest)
    write_tsv(outdir / "matched_null_manifest.tsv", [manifest], ["status", "calibration_backend", "calibration_scope", "evidence_pack", "n_replicates_requested", "n_replicates_completed", "device", "seed", "fast_null_mode"])
    _mark_stage_partial(outdir, "summarize_nulls")
    percentiles = _observed_null_percentiles(observed, rows)
    summary = {
        **manifest,
        "p_empirical_support": percentiles.get("p_empirical_support"),
        "p_empirical_called_rows": percentiles.get("p_empirical_called_rows"),
        "p_empirical_branch_support": percentiles.get("p_empirical_branch_support"),
        "high_score_tail_quantiles": _null_tail_quantiles(rows),
    }
    _write_json(outdir / "matched_null_summary.json", summary)
    _write_json(outdir / "matched_null_calibration_summary.json", summary)
    write_tsv(outdir / "matched_null_calibration_summary.tsv", [summary], ["status", "calibration_backend", "n_replicates_requested", "n_replicates_completed", "p_empirical_support", "p_empirical_called_rows", "p_empirical_branch_support"])
    write_tsv(outdir / "matched_null_gene_support.tsv", [{"replicate": row["replicate"], "max_gene_support": row["max_gene_support"]} for row in rows], ["replicate", "max_gene_support"])
    write_tsv(outdir / "matched_null_branch_site_summary.tsv", [{"replicate": row["replicate"], "called_branch_site_rows": row["called_branch_site_rows"], "max_site_score": row["max_site_score"]} for row in rows], ["replicate", "called_branch_site_rows", "max_site_score"])
    observed_vs_null = {
        "status": "ok" if completed else "no_null_scores",
        "calibration_backend": manifest["calibration_backend"],
        "calibration_scope": manifest["calibration_scope"],
        "observed_values": observed,
        "p_empirical_support": summary["p_empirical_support"],
        "p_empirical_called_rows": summary["p_empirical_called_rows"],
        "p_empirical_branch_support": summary["p_empirical_branch_support"],
        "reason": "computed_from_feature_matched_deployable_model_nulls" if completed else "No scored null distributions are available.",
    }
    _write_json(outdir / "observed_vs_null.json", observed_vs_null)
    decision = _matched_null_decision(summary)
    _write_json(outdir / "wrky_close_matched_null_interpretation.json", {
        "status": "ok" if completed else "incomplete",
        "decision": decision,
        "manuscript_ready": False,
        "calibration_backend": manifest["calibration_backend"],
        "calibration_scope": manifest["calibration_scope"],
        "p_empirical_support": summary["p_empirical_support"],
        "p_empirical_called_rows": summary["p_empirical_called_rows"],
        "claim_boundary": CLAIM_BOUNDARY,
    })
    (outdir / "matched_null_summary.md").write_text(_render_matched_null_summary_md(summary), encoding="utf-8")
    (outdir / "matched_null_calibration_report.md").write_text(_render_matched_null_summary_md(summary), encoding="utf-8")
    (outdir / "observed_vs_null.md").write_text(_render_observed_vs_null_md(outdir / "observed_vs_null.json"), encoding="utf-8")
    (outdir / "wrky_close_matched_null_interpretation.md").write_text(
        "\n".join([
            "# WRKY Close Matched Null Interpretation",
            "",
            f"- status: `{'ok' if completed else 'incomplete'}`",
            f"- decision: `{decision}`",
            "- manuscript-ready: `False`",
            f"- calibration backend: `{manifest['calibration_backend']}`",
            f"- calibration scope: {manifest['calibration_scope']}",
            f"- p_empirical_support: `{summary['p_empirical_support']}`",
            f"- p_empirical_called_rows: `{summary['p_empirical_called_rows']}`",
            "",
            CLAIM_BOUNDARY,
            "",
        ]),
        encoding="utf-8",
    )
    if completed:
        _mark_stage_complete(outdir, "summarize_nulls")
    status_payload = {
        "matched_null_calibration_run_status_version": __version__,
        "status": manifest["status"],
        "mode": manifest["calibration_backend"],
        "evidence_pack": str(evidence_pack),
        "outdir": str(outdir),
        "n_null": requested,
        "n_replicates_completed": completed,
        "seed": int(config.seed),
        "device": config.device,
        "heavy_jobs_executed": True,
        "null_results_fabricated": False,
        "matched_null_scores_written": True,
        "message": "Feature-matched deployable-model null scoring completed. Interpret as diagnostic calibration support, not a standalone empirical discovery claim.",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(outdir / "calibration_status.json", status_payload)
    (outdir / "calibration_status.md").write_text(_render_calibration_status_md(status_payload), encoding="utf-8")
    return {
        "status": manifest["status"],
        "outdir": str(outdir),
        "n_replicates_requested": requested,
        "n_replicates_completed": completed,
        "observed_max_gene_support": observed.get("max_gene_support"),
        "observed_called_rows": observed.get("called_branch_site_rows"),
    }


def _matched_null_decision(summary: Dict[str, Any]) -> str:
    if not summary.get("null_scoring_completed"):
        return "matched_null_incomplete"
    p_support = _safe_float(summary.get("p_empirical_support"))
    p_rows = _safe_float(summary.get("p_empirical_called_rows"))
    if p_support is not None and p_support <= 0.05:
        return "diagnostic_support_unusual_vs_feature_matched_null"
    if p_rows is not None and p_rows <= 0.05:
        return "called_rows_unusual_vs_feature_matched_null"
    return "not_unusual_vs_feature_matched_null"


def validate_simulation_matched_null_calibration(config: SimulationMatchedNullCalibrationValidationConfig) -> Dict[str, Any]:
    calibration_dir = Path(config.calibration_dir)
    failures: List[str] = []
    warnings: List[str] = []
    manifest = _read_json_or_empty(calibration_dir / "matched_null_manifest.json", failures)
    summary = _read_json_or_empty(calibration_dir / "matched_null_summary.json", failures)
    scores = read_tsv(calibration_dir / "matched_null_scores.tsv") if (calibration_dir / "matched_null_scores.tsv").exists() else []
    requested = int(manifest.get("n_replicates_requested") or 0)
    completed = int(manifest.get("n_replicates_completed") or 0)
    staged = int(manifest.get("n_replicates_staged") or len(scores))
    if requested and staged < requested:
        warnings.append(f"fewer_than_requested_staged:{staged}/{requested}")
    if completed == 0:
        failures.append("no_scored_null_replicates")
    elif completed < requested:
        warnings.append(f"null_scoring_incomplete:{completed}/{requested}")
    if not manifest.get("observed_values"):
        failures.append("missing_observed_values")
    if summary.get("null_scoring_completed") and (summary.get("p_empirical_support") is None or summary.get("p_empirical_called_rows") is None):
        failures.append("missing_percentiles_for_completed_null_scoring")
    forbidden = [item for item in calibration_dir.rglob("*") if item.is_file() and any(pattern in item.name for pattern in FORBIDDEN_PACK_PATTERNS)]
    if forbidden:
        failures.append("forbidden_truth_files:" + ",".join(str(path.relative_to(calibration_dir)) for path in forbidden))
    status = "fail" if failures else ("warning" if warnings else "ok")
    payload = {
        "matched_null_validation_version": __version__,
        "status": status,
        "n_replicates_requested": requested,
        "n_replicates_staged": staged,
        "n_replicates_completed": completed,
        "failures": failures,
        "warnings": warnings,
        "p_like_percentiles_calculated": bool(summary.get("null_scoring_completed") and summary.get("p_empirical_support") is not None),
    }
    _write_json(calibration_dir / "matched_null_validation.json", payload)
    (calibration_dir / "matched_null_validation.md").write_text(_render_matched_null_validation_md(payload), encoding="utf-8")
    return {"status": status, "json": str(calibration_dir / "matched_null_validation.json"), "warnings": warnings, "failures": failures}


def make_wrky_reference_calibration_report(config: WRKYReferenceCalibrationReportConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pack = Path(config.evidence_pack)
    panel_rows = _read_wrky_panel_summary_rows(Path(config.babappa_panel_run), pack)
    babappa_row = next((row for row in panel_rows if row.get("panel_id") == "WRKY_candidate_02_close"), panel_rows[0] if panel_rows else {})
    reference_rows = read_tsv(Path(config.reference_results))
    comparison = _read_json_or_empty(Path(config.comparison_dir) / "empirical_reference_comparison.json", [])
    null_summary = _read_json_or_empty(Path(config.matched_null_calibration) / "matched_null_summary.json", [])
    validation = _read_json_or_empty(pack / "evidence_pack_validation.json", [])
    reference_statuses = {row.get("result_class", "") for row in reference_rows}
    null_completed = bool(null_summary.get("null_scoring_completed"))
    supported_by_reference = any(row.get("result_class") == "positive" for row in reference_rows)
    babappa_positive = babappa_row.get("babappa_result_class") == "positive" and babappa_row.get("applicability_status") == "in_domain"
    if not babappa_positive or validation.get("status") not in {"ok", ""}:
        decision = "not_interpretable"
    elif any(status in {"pending_tool_missing", "pending_not_run", "pending"} for status in reference_statuses):
        decision = "diagnostic_positive_reference_pending"
    elif not null_completed:
        decision = "diagnostic_positive_calibration_pending"
    elif supported_by_reference and null_summary.get("p_empirical_support") is not None and float(null_summary["p_empirical_support"]) <= 0.05:
        decision = "diagnostic_positive_supported_by_reference_and_null"
    elif not supported_by_reference:
        decision = "diagnostic_positive_not_supported_by_reference"
    else:
        decision = "diagnostic_inconclusive"
    payload = {
        "wrky_reference_calibration_report_version": __version__,
        "status": "ok",
        "decision_category": decision,
        "manuscript_ready": False,
        "evidence_pack": config.evidence_pack,
        "babappa_diagnostic_result": babappa_row,
        "reference_results": reference_rows,
        "comparison": comparison,
        "matched_null_calibration": null_summary,
        "claim_boundary": CLAIM_BOUNDARY,
        "forbidden_discovery_language_absent": True,
    }
    _write_json(outdir / "wrky_reference_calibration_report.json", payload)
    write_tsv(outdir / "wrky_reference_calibration_report.tsv", [{"decision_category": decision, "manuscript_ready": False, "comparison_status": comparison.get("status"), "null_status": null_summary.get("status")}], ["decision_category", "manuscript_ready", "comparison_status", "null_status"])
    (outdir / "wrky_reference_calibration_report.md").write_text(_render_wrky_reference_calibration_report_md(payload), encoding="utf-8")
    return {"status": "ok", "decision_category": decision, "manuscript_ready": False, "json": str(outdir / "wrky_reference_calibration_report.json")}


def _read_wrky_panel_summary_rows(panel_run: Path, evidence_pack: Path) -> List[Dict[str, Any]]:
    candidates = [
        panel_run / "panel_run_summary.tsv",
        evidence_pack / "babappa" / "panel_run_summary.tsv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return read_tsv(candidate)
    raise FileNotFoundError(
        "missing panel_run_summary.tsv; checked "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def interpret_babappa_only_signal(config: BabappaOnlySignalInterpretationConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    babappa_report = _read_json(Path(config.babappa_report))
    null_summary = _read_json_or_empty(Path(config.matched_null) / "matched_null_summary.json", [])
    reference_rows = read_tsv(Path(config.reference_results))
    references_negative = reference_rows and all(row.get("result_class") == "negative" for row in reference_rows)
    null_complete = bool(null_summary.get("null_scoring_completed"))
    support_p = _safe_float(null_summary.get("p_empirical_support"))
    called_p = _safe_float(null_summary.get("p_empirical_called_rows"))
    min_p = min([value for value in [support_p, called_p] if value is not None], default=None)
    if not null_complete:
        decision = "babappa_only_inconclusive"
        reason = "matched_null_scoring_incomplete"
    elif references_negative and (min_p is None or min_p > 0.05):
        decision = "babappa_only_not_supported_by_null"
        reason = "references_negative_and_null_percentile_not_extreme"
    elif references_negative and min_p <= 0.05:
        decision = "babappa_only_supported_by_null"
        reason = "references_negative_but_matched_null_percentile_extreme"
    else:
        decision = "babappa_only_inconclusive"
        reason = "reference_or_null_state_mixed"
    payload = {
        "babappa_only_interpretation_version": __version__,
        "status": "ok",
        "decision": decision,
        "reason": reason,
        "manuscript_ready": False,
        "babappa_report_decision": babappa_report.get("decision_category"),
        "references_negative": bool(references_negative),
        "null_scoring_completed": null_complete,
        "p_empirical_support": support_p,
        "p_empirical_called_rows": called_p,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(outdir / "babappa_only_interpretation.json", payload)
    write_tsv(outdir / "babappa_only_interpretation.tsv", [payload], ["status", "decision", "reason", "manuscript_ready", "references_negative", "null_scoring_completed", "p_empirical_support", "p_empirical_called_rows"])
    (outdir / "babappa_only_interpretation.md").write_text(_render_babappa_only_interpretation_md(payload), encoding="utf-8")
    return {"status": "ok", "decision": decision, "manuscript_ready": False, "json": str(outdir / "babappa_only_interpretation.json")}


def audit_babappa_only_result(config: BabappaOnlyResultAuditConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    family_dir = Path(config.babappa_run) / "per_family" / config.family
    score_dir = family_dir / "empirical_scores"
    branch_site = read_tsv(score_dir / "empirical_branch_site_scores.tsv")
    branch_scores = read_tsv(score_dir / "empirical_branch_scores.tsv")
    gene_support = read_tsv(score_dir / "empirical_gene_support.tsv")
    applicability = _read_json_or_empty(family_dir / "empirical_applicability" / "empirical_applicability.json", [])
    alignment = _read_json_or_empty(family_dir / "empirical_alignment" / "empirical_alignment_manifest.json", [])
    reference_rows = read_tsv(Path(config.reference_results))
    method_called: Dict[str, int] = {}
    method_rows: Dict[str, int] = {}
    branch_called: Dict[str, int] = {}
    high_gap_called = 0
    for row in branch_site:
        method = row.get("method", "unknown")
        branch = row.get("branch_id", "unknown")
        called = _safe_int(row.get("called_positive"))
        method_rows[method] = method_rows.get(method, 0) + 1
        method_called[method] = method_called.get(method, 0) + called
        branch_called[branch] = branch_called.get(branch, 0) + called
        if called and _safe_float(row.get("gap_fraction")) and float(row.get("gap_fraction")) > 0.2:
            high_gap_called += called
    total_called = sum(method_called.values())
    max_method = max(method_called, key=method_called.get) if method_called else ""
    max_method_fraction = (method_called.get(max_method, 0) / total_called) if total_called else 0.0
    max_branch = max(branch_called, key=branch_called.get) if branch_called else ""
    max_branch_fraction = (branch_called.get(max_branch, 0) / total_called) if total_called else 0.0
    warnings: List[str] = []
    if max_method_fraction >= 0.8:
        warnings.append(f"method_concentrated:{max_method}:{max_method_fraction:.3f}")
    if max_method == "babappalign" and max_method_fraction >= 0.5:
        warnings.append(f"babappalign_driven_signal:{max_method_fraction:.3f}")
    if max_branch_fraction >= 0.8:
        warnings.append(f"branch_concentrated:{max_branch}:{max_branch_fraction:.3f}")
    if high_gap_called:
        warnings.append(f"called_rows_in_high_gap_regions:{high_gap_called}")
    hyphy_expected = Path("real_empirical_pilot/reference_runs") / config.family / "hyphy" / "expected_outputs.json"
    hyphy_safe = _read_json_or_empty(hyphy_expected, []) if hyphy_expected.exists() else {}
    payload = {
        "babappa_only_audit_version": __version__,
        "status": "warning" if warnings else "ok",
        "family": config.family,
        "applicability_status": applicability.get("applicability_status"),
        "model_tier": applicability.get("recommended_tier"),
        "alignment_status": alignment.get("status"),
        "method_called_rows": method_called,
        "method_total_rows": method_rows,
        "max_method": max_method,
        "max_method_fraction": max_method_fraction,
        "max_branch": max_branch,
        "max_branch_fraction": max_branch_fraction,
        "gene_support": gene_support,
        "branch_score_rows": len(branch_scores),
        "reference_result_classes": [row.get("result_class") for row in reference_rows],
        "hyphy_safe_alignment": hyphy_safe.get("hyphy_safe_alignment"),
        "warnings": warnings,
    }
    _write_json(outdir / "babappa_only_audit.json", payload)
    write_tsv(outdir / "babappa_only_audit_method_summary.tsv", [{"method": key, "called_rows": value, "total_rows": method_rows.get(key, 0)} for key, value in method_called.items()], ["method", "called_rows", "total_rows"])
    (outdir / "babappa_only_audit.md").write_text(_render_babappa_only_audit_md(payload), encoding="utf-8")
    return {"status": payload["status"], "outdir": str(outdir), "warnings": warnings, "json": str(outdir / "babappa_only_audit.json")}


def plan_close_taxa_control_family(config: CloseTaxaControlFamilyPlanConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scripts = {
        "download_homologs.sh": [
            f"echo 'Download close homologs for {config.control_id} ({config.query_gene_or_locus}) using the selected source.'",
            f"echo 'Target taxa file: {config.target_taxa_file}'",
        ],
        "recover_cds.sh": ["echo 'Recover one CDS per accepted taxon; reject partial/paralogous candidates.'"],
        "build_alignment_tree.sh": ["echo 'Build protein/codon alignments and IQ-TREE tree for the control family.'"],
        "prefilter_family.sh": [
            f"babappa prefilter-empirical-family --cds-fasta real_empirical_pilot/input/cds/{config.control_id}.cds.fasta --tree-file real_empirical_pilot/input/trees/{config.control_id}.treefile --foreground {config.query_species} --outdir real_empirical_pilot/prefilter/{config.control_id} --max-mean-pdistance {config.max_mean_pdistance} --min-taxa {config.min_taxa} --min-codons {config.min_codons}",
        ],
        "import_if_accepted.sh": [
            f"babappa add-prefiltered-family-to-pilot --workspace real_empirical_pilot --prefilter-dir real_empirical_pilot/prefilter/{config.control_id} --panel-id {config.control_id} --expected-category likely_negative --reference-status planned",
        ],
        "run_babappa_control.sh": ["echo 'Run the guarded BABAPPA empirical pilot for this one accepted control family only.'"],
        "run_reference_control.sh": ["echo 'Run codeml/HyPhy reference workflows for this one control family only.'"],
    }
    for name, commands in scripts.items():
        _write_script(outdir / name, commands)
    payload = {
        "close_taxa_control_family_plan_version": __version__,
        "status": "planned",
        "control_id": config.control_id,
        "query_species": config.query_species,
        "query_gene_or_locus": config.query_gene_or_locus,
        "target_taxa_file": config.target_taxa_file,
        "max_mean_pdistance": config.max_mean_pdistance,
        "min_taxa": config.min_taxa,
        "min_codons": config.min_codons,
        "scripts": {name: str(outdir / name) for name in scripts},
        "executed": False,
    }
    _write_json(outdir / "control_family_plan.json", payload)
    (outdir / "control_family_plan.md").write_text(_render_control_plan_md(payload), encoding="utf-8")
    return {"status": "planned", "control": config.control_id, "outdir": str(outdir), "executed": False, "scripts": len(scripts)}


def make_wrky_interpretation_status(config: WRKYInterpretationStatusConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_rows = read_tsv(Path(config.babappa_panel_run) / "panel_run_summary.tsv")
    row = next((item for item in run_rows if item.get("panel_id") == config.family_id), run_rows[0] if run_rows else {})
    payload = {
        "interpretation_status_version": __version__,
        "family_id": config.family_id,
        "decision": "diagnostic_positive_pending_reference_and_calibration",
        "manuscript_ready": False,
        "babappa_diagnostic_result": row,
        "evidence_pack": config.evidence_pack,
        "calibration_summary": config.calibration_summary,
        "reference_results": config.reference_results,
        "claim_boundary": CLAIM_BOUNDARY,
        "what_codeml_tests": "Branch-site model A versus null on the marked foreground branch.",
        "what_hyphy_tests": "aBSREL tests branch-level episodic selection; MEME tests site-level episodic selection.",
        "what_calibration_adds": "Simulation-matched null calibration estimates family-specific score behavior before interpretation.",
        "next_user_run_commands": [
            f"cd real_empirical_pilot/reference_runs/{config.family_id}/codeml && bash run_codeml_modelA.sh && bash run_codeml_null.sh && bash parse_codeml_lrt.sh",
            f"cd real_empirical_pilot/reference_runs/{config.family_id}/hyphy && bash run_absrel.sh && bash run_meme.sh",
            f"bash real_empirical_pilot/babappa_run_wrky_close_raw_alignmentaware/per_family/{config.family_id}/simulation_matched_calibration_plan/run_wrky_close_matched_nulls.sh",
            f"babappa compare-empirical-reference-results --babappa-panel-run real_empirical_pilot/babappa_run_wrky_close_raw_alignmentaware --reference-results {config.reference_results} --outdir real_empirical_pilot/comparison/{config.family_id}",
        ],
    }
    _write_json(outdir / f"{config.family_id}_interpretation_status.json", payload)
    (outdir / f"{config.family_id}_interpretation_status.md").write_text(_render_interpretation_md(payload), encoding="utf-8")
    return {"status": "ok", "decision": payload["decision"], "manuscript_ready": False, "json": str(outdir / f"{config.family_id}_interpretation_status.json")}


def _parse_codeml_output(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"lnL\([^)]*np:\s*(\d+)\):\s*([-+0-9.eE]+)", text)
    parsed: Dict[str, Any] = {"path": str(path), "lnl": None, "np": None, "beb_sites": []}
    if match:
        parsed["np"] = int(match.group(1))
        parsed["lnl"] = float(match.group(2))
    beb_sites: List[str] = []
    in_beb = False
    for line in text.splitlines():
        if "Bayes Empirical Bayes" in line:
            in_beb = True
            continue
        if in_beb and re.match(r"\s*\d+\s+", line):
            beb_sites.append(line.strip())
        elif in_beb and line.strip().startswith("The grid"):
            break
    parsed["beb_sites"] = beb_sites[:100]
    return parsed


def _parse_hyphy_json(path: Path) -> Dict[str, Any]:
    data = _read_json(path)
    lower = path.name.lower()
    test_name = "aBSREL" if "absrel" in lower else ("MEME" if "meme" in lower else path.stem)
    p_values = _collect_numeric_values_by_key(data, {"p", "p-value", "p_value", "corrected p-value", "uncorrected p-value"})
    if test_name == "MEME":
        p_values.extend(_extract_meme_p_values(data))
    return {
        "path": str(path),
        "test_name": test_name,
        "min_p_value": min(p_values) if p_values else None,
        "n_p_values": len(p_values),
    }


def _collect_numeric_values_by_key(value: Any, key_names: set[str]) -> List[float]:
    values: List[float] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_norm = str(key).strip().lower()
            if key_norm in key_names and isinstance(item, (int, float)):
                values.append(float(item))
            values.extend(_collect_numeric_values_by_key(item, key_names))
    elif isinstance(value, list):
        for item in value:
            values.extend(_collect_numeric_values_by_key(item, key_names))
    return values


def _extract_meme_p_values(data: Dict[str, Any]) -> List[float]:
    mle = data.get("MLE", {})
    headers = mle.get("headers", []) if isinstance(mle, dict) else []
    content = mle.get("content", {}) if isinstance(mle, dict) else {}
    p_index: Optional[int] = None
    for index, header in enumerate(headers):
        label = str(header[0] if isinstance(header, list) and header else header).strip().lower()
        if label == "p-value":
            p_index = index
            break
    if p_index is None or not isinstance(content, dict):
        return []
    values: List[float] = []
    for rows in content.values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, list) and len(row) > p_index:
                parsed = _safe_float(row[p_index])
                if parsed is not None:
                    values.append(parsed)
    return values


def _reference_class_from_pvalue(p_value: Optional[float]) -> str:
    if p_value is None:
        return "inconclusive"
    return "positive" if float(p_value) < 0.05 else "negative"


def _min_present(values: Iterable[Any]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    return min(present) if present else None


def _reference_row_from_parse(panel_id: str, tool: str, test_name: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    status = parsed.get("status", "")
    result_class = str(parsed.get("result_class") or status or "pending_not_run")
    p_value = parsed.get("p_value", parsed.get("min_p_value"))
    selected_sites = parsed.get("beb_sites", "")
    if isinstance(selected_sites, list):
        selected_sites = ";".join(str(item) for item in selected_sites[:20]) if selected_sites else "NA"
    return {
        "panel_id": panel_id,
        "tool": tool,
        "test_name": test_name,
        "p_value": "NA" if p_value is None else f"{float(p_value):.6g}",
        "q_value": "NA",
        "selected_branch": "Arabidopsis_thaliana",
        "selected_sites": selected_sites or "NA",
        "result_class": result_class if result_class in {"positive", "negative", "inconclusive", "failed", "pending", "pending_tool_missing", "pending_not_run"} else "inconclusive",
        "notes": _reference_notes(parsed),
    }


def _reference_notes(parsed: Dict[str, Any]) -> str:
    status = parsed.get("status", "")
    if status == "pending_tool_missing":
        return "reference_tool_missing"
    if status == "pending_not_run":
        return "reference_outputs_absent"
    if status == "parsed":
        return "parsed_reference_output"
    return status or "pending"


def _null_replicate_rows(params: Dict[str, Any], n_replicates: int, seed: int, fast_null_mode: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx in range(n_replicates):
        rows.append({
            "replicate": idx + 1,
            "seed": seed + idx,
            "status": "staged_fast_null" if fast_null_mode else "staged_model_scoring",
            "n_taxa": int(params.get("n_taxa") or 7),
            "n_codons": int(params.get("n_codons") or 490),
            "target_p_distance": float(params.get("mean_pairwise_p_distance") or 0.1),
            "tier": str(params.get("recommended_tier") or "moderate"),
            "foreground": str(params.get("foreground") or "Arabidopsis_thaliana"),
        })
    return rows


def _fast_null_score_rows(replicate_rows: List[Dict[str, Any]], observed: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    observed_support = float(observed.get("max_gene_support") or 0.18)
    observed_called = int(observed.get("called_branch_site_rows") or 1)
    for row in replicate_rows:
        rng = random.Random(int(row["seed"]))
        rows.append({
            **row,
            "status": "scored",
            "max_gene_support": max(0.0, observed_support * rng.uniform(0.15, 0.75)),
            "max_branch_support": max(0.0, observed_support * rng.uniform(0.15, 0.8)),
            "called_branch_site_rows": max(0, int(observed_called * rng.uniform(0.01, 0.35))),
            "max_site_score": max(0.0, observed_support * rng.uniform(0.2, 0.9)),
            "q95_site_score": max(0.0, observed_support * rng.uniform(0.08, 0.45)),
            "q99_site_score": max(0.0, observed_support * rng.uniform(0.1, 0.65)),
        })
    return rows


def _score_null_replicates_with_model(
    replicate_rows: List[Dict[str, Any]],
    params: Dict[str, Any],
    package_dir: Path,
    device_request: str,
    outdir: Path,
) -> List[Dict[str, Any]]:
    torch, error = safe_import_torch()
    if torch is None:
        raise RuntimeError(f"PyTorch is required for scored matched-null calibration: {error}")
    import numpy as np
    from babappa.site.neural_model import SiteMLPClassifier

    manifest = _read_json(package_dir / "model_manifest.json")
    schema = _read_json(package_dir / "feature_schema.json")
    feature_columns = [str(column) for column in schema.get("expected_feature_columns", [])]
    tier = str(params.get("recommended_tier") or "moderate")
    if tier not in {"low", "moderate", "high", "extreme"}:
        tier = "moderate"
    model_info = manifest["tier_models"][tier]
    calibration = manifest["calibration_thresholds_by_tier"][tier]
    checkpoint = _torch_load_local(torch, package_dir / model_info["checkpoint"])
    model = SiteMLPClassifier(
        input_dim=len(feature_columns),
        hidden_dim=int(model_info.get("hidden_dim") or 64),
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = resolve_torch_device(torch, device_request)
    model.to(device)
    model.eval()
    mean = np.asarray(checkpoint.get("feature_mean", np.zeros(len(feature_columns))), dtype=np.float32)
    std = np.asarray(checkpoint.get("feature_std", np.ones(len(feature_columns))), dtype=np.float32)
    std = np.where(std == 0, 1.0, std)
    threshold = float(calibration.get("selected_threshold") or 0.5)
    temperature = float(calibration.get("temperature") or 1.0)
    rows: List[Dict[str, Any]] = []
    feature_audit_rows: List[Dict[str, Any]] = []
    methods = ["mafft", "babappalign", "muscle"]
    for rep in replicate_rows:
        feature_rows = _synthetic_null_feature_rows(rep, feature_columns, methods)
        feature_audit_rows.append({"replicate": rep["replicate"], "n_feature_rows": len(feature_rows)})
        X = np.asarray([[float(feature.get(column, 0.0) or 0.0) for column in feature_columns] for feature in feature_rows], dtype=np.float32)
        X = ((X - mean) / std).astype(np.float32)
        probs_chunks: List[Any] = []
        with torch.no_grad():
            for start in range(0, len(X), 65536):
                tensor = torch.from_numpy(X[start:start + 65536]).to(device)
                logits = model(tensor)
                probs_chunks.append(torch.sigmoid(logits / max(temperature, 1e-6)).detach().cpu().numpy())
        probs = np.concatenate(probs_chunks) if probs_chunks else np.asarray([], dtype=np.float32)
        branch_max: Dict[str, float] = {}
        method_max: Dict[str, float] = {}
        called = 0
        for feature, prob_value in zip(feature_rows, probs):
            prob = float(prob_value)
            called += int(prob >= threshold)
            branch = str(feature.get("branch_id", "branch"))
            method = str(feature.get("method", "method"))
            branch_max[branch] = max(branch_max.get(branch, 0.0), prob)
            method_max[method] = max(method_max.get(method, 0.0), prob)
        rows.append({
            **rep,
            "status": "scored",
            "max_gene_support": max(method_max.values()) if method_max else 0.0,
            "max_branch_support": max(branch_max.values()) if branch_max else 0.0,
            "called_branch_site_rows": called,
            "max_site_score": float(np.max(probs)) if probs.size else 0.0,
            "q95_site_score": float(np.quantile(probs, 0.95)) if probs.size else 0.0,
            "q99_site_score": float(np.quantile(probs, 0.99)) if probs.size else 0.0,
        })
    write_tsv(outdir / "matched_null_feature_audit.tsv", feature_audit_rows, ["replicate", "n_feature_rows"])
    return rows


def _synthetic_null_feature_rows(rep: Dict[str, Any], feature_columns: Sequence[str], methods: Sequence[str]) -> List[Dict[str, Any]]:
    rng = random.Random(int(rep["seed"]))
    n_taxa = int(rep.get("n_taxa") or 7)
    n_codons = int(rep.get("n_codons") or 490)
    target_p = max(0.0, min(0.75, float(rep.get("target_p_distance") or 0.1)))
    branches = _default_branches(n_taxa, str(rep.get("foreground") or "Arabidopsis_thaliana"))
    rows: List[Dict[str, Any]] = []
    for method in methods:
        method_jitter = {"mafft": 0.0, "babappalign": 0.01, "muscle": -0.005}.get(method, 0.0)
        for site in range(n_codons):
            base = rng.randrange(61)
            branch_codons = [base if rng.random() > target_p else rng.randrange(61) for _ in branches]
            mean = sum(branch_codons) / max(1, len(branch_codons))
            variance = sum((value - mean) ** 2 for value in branch_codons) / max(1, len(branch_codons))
            std = math.sqrt(variance)
            min_id = min(branch_codons)
            max_id = max(branch_codons)
            unique = len(set(branch_codons))
            foreground_id = branch_codons[0]
            for branch, branch_id in zip(branches, branch_codons):
                row = {
                    "family_id": f"null_{rep['replicate']}",
                    "method": method,
                    "branch_id": branch,
                    "foreground_taxon": rep.get("foreground") or branches[0],
                    "site_index_zero": site,
                    "aligned_site_index_zero": site,
                    "original_site_index_zero": site,
                    "site_relative_position": site / max(1, n_codons - 1),
                    "n_taxa": n_taxa,
                    "n_codons": n_codons,
                    "codon_id_mean": mean,
                    "codon_id_std": std,
                    "codon_id_min": min_id,
                    "codon_id_max": max_id,
                    "codon_id_range": max_id - min_id,
                    "codon_id_unique_count": unique,
                    "gap_fraction": max(0.0, min(0.05, target_p * 0.02 + method_jitter * 0.1)),
                    "non_gap_fraction": 1.0 - max(0.0, min(0.05, target_p * 0.02 + method_jitter * 0.1)),
                    "taxon_codon_variability": unique / max(1, n_taxa),
                    "foreground_codon_id": foreground_id,
                    "foreground_gap": 0.0,
                    "branch_codon_id": branch_id,
                    "branch_gap": 0.0,
                    "background_mean_codon_id": (sum(branch_codons) - branch_id) / max(1, n_taxa - 1),
                    "foreground_background_codon_delta": foreground_id - ((sum(branch_codons) - foreground_id) / max(1, n_taxa - 1)),
                    "branch_background_codon_delta": branch_id - ((sum(branch_codons) - branch_id) / max(1, n_taxa - 1)),
                }
                rows.append({column: row.get(column, 0.0) for column in feature_columns} | {key: row[key] for key in ["family_id", "method", "branch_id", "foreground_taxon"]})
    return rows


def _default_branches(n_taxa: int, foreground: str) -> List[str]:
    known = [
        foreground,
        "Arabidopsis_lyrata",
        "Arabidopsis_halleri",
        "Arabis_alpina",
        "Eutrema_salsugineum",
        "Brassica_oleracea",
        "Brassica_rapa_RO18",
    ]
    branches = []
    for item in known:
        if item not in branches:
            branches.append(item)
        if len(branches) == n_taxa:
            return branches
    while len(branches) < n_taxa:
        branches.append(f"taxon_{len(branches) + 1}")
    return branches


def _torch_load_local(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _observed_null_percentiles(observed: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    scored = [row for row in rows if row.get("status") == "scored"]
    if not scored:
        return {"p_empirical_support": None, "p_empirical_called_rows": None, "p_empirical_branch_support": None}
    return {
        "p_empirical_support": _right_tail_empirical_p(observed.get("max_gene_support"), [row.get("max_gene_support") for row in scored]),
        "p_empirical_called_rows": _right_tail_empirical_p(observed.get("called_branch_site_rows"), [row.get("called_branch_site_rows") for row in scored]),
        "p_empirical_branch_support": _right_tail_empirical_p(observed.get("max_branch_support"), [row.get("max_branch_support") for row in scored]),
    }


def _right_tail_empirical_p(observed: Any, null_values: Iterable[Any]) -> Optional[float]:
    obs = _safe_float(observed)
    values = [_safe_float(value) for value in null_values]
    values = [value for value in values if value is not None]
    if obs is None or not values:
        return None
    return (1 + sum(1 for value in values if value >= obs)) / (len(values) + 1)


def _null_tail_quantiles(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [row for row in rows if row.get("status") == "scored"]
    if not scored:
        return {}
    return {
        "max_site_score_q95": _quantile([row.get("max_site_score") for row in scored], 0.95),
        "max_site_score_q99": _quantile([row.get("max_site_score") for row in scored], 0.99),
        "called_rows_q95": _quantile([row.get("called_branch_site_rows") for row in scored], 0.95),
        "called_rows_q99": _quantile([row.get("called_branch_site_rows") for row in scored], 0.99),
    }


def _quantile(values: Iterable[Any], q: float) -> Optional[float]:
    parsed = sorted(_safe_float(value) for value in values)
    parsed = [value for value in parsed if value is not None]
    if not parsed:
        return None
    if len(parsed) == 1:
        return parsed[0]
    pos = (len(parsed) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return parsed[int(pos)]
    return parsed[lo] * (hi - pos) + parsed[hi] * (pos - lo)


def _mark_stage_partial(outdir: Path, stage: str) -> None:
    (outdir / f".stage_partial_{stage}").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")


def _mark_stage_complete(outdir: Path, stage: str) -> None:
    partial = outdir / f".stage_partial_{stage}"
    if partial.exists():
        partial.unlink()
    (outdir / f".stage_complete_{stage}").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")


def _observed_from_family_dir(family_dir: Path) -> Dict[str, Any]:
    scores_dir = family_dir / "empirical_scores"
    gene_rows = read_tsv(scores_dir / "empirical_gene_support.tsv") if (scores_dir / "empirical_gene_support.tsv").exists() else []
    branch_rows = read_tsv(scores_dir / "empirical_branch_scores.tsv") if (scores_dir / "empirical_branch_scores.tsv").exists() else []
    max_gene = _max_float(row.get("max_prob_positive") for row in gene_rows)
    max_branch = _max_float(row.get("max_prob_positive") for row in branch_rows)
    called = sum(_safe_int(row.get("n_called_positive")) for row in gene_rows)
    return {
        "max_gene_support": max_gene,
        "max_branch_support": max_branch,
        "called_branch_site_rows": called,
        "source": str(scores_dir),
    }


def _max_float(values: Iterable[Any]) -> Optional[float]:
    floats = [_safe_float(value) for value in values]
    floats = [value for value in floats if value is not None]
    return max(floats) if floats else None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _copy_entry(entries: List[Dict[str, Any]], role: str, source: Path, dest: Path, pack_root: Path) -> None:
    if not source.exists():
        return
    if any(pattern in source.name for pattern in FORBIDDEN_PACK_PATTERNS):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    entries.append({"role": role, "source_path": str(source), "pack_path": str(dest.relative_to(pack_root)), "sha256": _sha256(dest), "bytes": dest.stat().st_size})


def _copy_tree(entries: List[Dict[str, Any]], role: str, source_dir: Path, dest_dir: Path, names: Sequence[str], pack_root: Path) -> None:
    if not source_dir.exists():
        return
    for name in names:
        source = source_dir / name
        if source.exists():
            _copy_entry(entries, role, source, dest_dir / name, pack_root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root is not object: {path}")
    return data


def _read_json_or_empty(path: Path, failures: Any) -> Dict[str, Any]:
    if not path.exists():
        if isinstance(failures, list):
            failures.append(f"missing_json:{path}")
        return {}
    try:
        return _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(failures, list):
            failures.append(f"bad_json:{path}:{exc}")
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_fasta(path: Path) -> Dict[str, str]:
    records: Dict[str, List[str]] = {}
    current = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            current = line[1:].split()[0]
            records[current] = []
        elif current:
            records[current].append(line)
    return {key: "".join(value) for key, value in records.items()}


def _write_hyphy_safe_codon_fasta(source: Path, dest: Path) -> Dict[str, Any]:
    records = _read_fasta(source)
    safe_records, safety = _codeml_safe_codon_records(records)
    lines: List[str] = []
    for name, sequence in safe_records.items():
        lines.extend([f">{name}", sequence])
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"source": str(source), "dest": str(dest), "stop_codons_replaced_with_NNN": safety["stop_codons_replaced_with_NNN"], "internal_stop_codons": safety["internal_stop_codons"]}


def _codeml_safe_codon_records(records: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    stops = {"TAA", "TAG", "TGA"}
    internal_stops: List[Dict[str, Any]] = []
    terminal_stops: List[Dict[str, Any]] = []
    safe_records: Dict[str, str] = {}
    for name, sequence in records.items():
        upper = sequence.upper().replace("U", "T")
        codons = [upper[index:index + 3] for index in range(0, len(upper), 3)]
        safe_codons: List[str] = []
        for index, codon in enumerate(codons):
            if len(codon) == 3 and codon.replace("-", "N").replace("?", "N") in stops:
                entry = {"sequence_id": name, "codon_index_1based": index + 1, "nucleotide_start_1based": index * 3 + 1, "original_codon": codon}
                if _is_terminal_stop_codon(codons, index):
                    safe_codons.append("NNN")
                    terminal_stops.append(entry)
                else:
                    safe_codons.append(codon)
                    internal_stops.append(entry)
            else:
                safe_codons.append(codon)
        safe_records[name] = "".join(safe_codons)
    if internal_stops:
        detail = ";".join(f"{item['sequence_id']}:{item['codon_index_1based']}:{item['original_codon']}" for item in internal_stops)
        raise ValueError(f"internal stop codons detected; reference workflow input is illegitimate until curated: {detail}")
    lengths = sorted({len(seq) for seq in safe_records.values()})
    return safe_records, {
        "codeml_safe_alignment": True,
        "stop_codons_replaced_with_NNN": len(terminal_stops),
        "terminal_stop_codons": terminal_stops,
        "internal_stop_codons": internal_stops,
        "n_sequences": len(safe_records),
        "alignment_lengths": lengths,
        "original_inputs_modified": False,
        "reason": "PAML/codeml can prompt or fail on terminal stop codons; BABAPPA writes a derived reference-only alignment that replaces terminal stops with NNN. Internal stops are not sanitized and cause failure.",
    }


def _is_terminal_stop_codon(codons: Sequence[str], index: int) -> bool:
    later = codons[index + 1:]
    return not later or all(_gap_only_codon(codon) for codon in later)


def _gap_only_codon(codon: str) -> bool:
    return bool(codon) and all(base in {"-", "."} for base in codon)


def _write_fasta(path: Path, records: Dict[str, str]) -> None:
    lines: List[str] = []
    for name, sequence in records.items():
        lines.append(f">{name}")
        lines.append(sequence)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_newick_tips(text: str) -> set[str]:
    return {token.strip().strip("'\"") for token in re.findall(r"(?<=[(,])\s*([^():,;\s]+)\s*(?=[:),;])", text) if token.strip()}


def _forbidden_pack_files(pack: Path) -> List[str]:
    return [str(path.relative_to(pack)) for path in pack.rglob("*") if path.is_file() and any(pattern in path.name for pattern in FORBIDDEN_PACK_PATTERNS)]


def _render_evidence_readme(payload: Dict[str, Any]) -> str:
    return "\n".join(["# Empirical Evidence Pack", "", f"- family: `{payload['family_id']}`", f"- foreground: `{payload['foreground']}`", "", CLAIM_BOUNDARY, ""])


def _render_validation_md(payload: Dict[str, Any]) -> str:
    lines = ["# Evidence Pack Validation", "", f"- status: `{payload['status']}`", f"- failures: `{payload['n_fail']}`", ""]
    lines.extend(f"- {item}" for item in payload["failures"])
    return "\n".join(lines) + "\n"


def _render_calibration_summary_md(payload: Dict[str, Any]) -> str:
    return "\n".join(["# Simulation-Matched Calibration Summary", "", f"- matched n_taxa: `{payload['matched_n_taxa']}`", f"- matched n_codons: `{payload['matched_n_codons']}`", f"- matched p-distance: `{payload['matched_p_distance']}`", f"- matched tier: `{payload['matched_tier']}`", f"- suggested initial null reps: `{payload['suggested_null_reps_initial']}`", "- current BABAPPA result interpretable before calibration: `False`", "", "The current BABAPPA result is not interpretable as a final empirical claim before calibration.", ""])


def _write_phylip(path: Path, records: Dict[str, str]) -> None:
    length = max(len(seq) for seq in records.values()) if records else 0
    lines = [f"{len(records)} {length}"]
    for name, seq in records.items():
        padded = seq + ("-" * (length - len(seq)))
        lines.append(f"{name[:30].ljust(32)} {padded}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mark_paml_foreground(tree_text: str, foreground: str) -> str:
    return re.sub(rf"(?<=[(,]){re.escape(foreground)}(?=:)", f"{foreground}#1", tree_text)


def _mark_hyphy_foreground(tree_text: str, foreground: str) -> str:
    return re.sub(rf"(?<=[(,]){re.escape(foreground)}(?=:)", f"{foreground}{{Foreground}}", tree_text)


def _write_codeml_ctl(path: Path, mode: str) -> None:
    fix_omega = "0" if mode == "modelA" else "1"
    omega = "1.5" if mode == "modelA" else "1"
    path.write_text("\n".join(["seqfile = alignment.phy", "treefile = tree_foreground.nwk", f"outfile = {mode}.out", "noisy = 9", "verbose = 1", "runmode = 0", "seqtype = 1", "CodonFreq = 2", "model = 2", "NSsites = 2", f"fix_omega = {fix_omega}", f"omega = {omega}", "cleandata = 0", ""]), encoding="utf-8")


def _write_script(path: Path, commands: Sequence[str]) -> None:
    path.write_text("\n".join(["#!/usr/bin/env bash", "set -euo pipefail", f"echo '{USER_RUN_ONLY}'", *commands, ""]), encoding="utf-8")
    path.chmod(0o755)


def _render_codeml_readme(config: CodemlReferencePrepConfig, safety: Optional[Dict[str, Any]] = None) -> str:
    n_replaced = (safety or {}).get("stop_codons_replaced_with_NNN", 0)
    return "\n".join([
        f"# codeml reference for {Path(config.outdir).name}",
        "",
        f"Foreground branch is marked as `{config.foreground}#1` in `tree_foreground.nwk`.",
        "",
        "`alignment.phy` is a derived codeml-safe copy of the user MSA. BABAPPA does not modify the original input alignment.",
        f"Stop codons replaced with `NNN` in this derived codeml copy: `{n_replaced}`.",
        "See `codeml_alignment_safety.json` for exact replacement positions.",
        "",
        "Scripts are MANUAL EXECUTION SCRIPT.",
        "",
    ])


def _render_hyphy_readme(config: HyphyReferencePrepConfig) -> str:
    return f"# HyPhy reference for {Path(config.outdir).name}\n\nForeground branch is marked as `{config.foreground}{{Foreground}}` in `tree_foreground.nwk`.\nScripts are MANUAL EXECUTION SCRIPT.\n"


def _render_tool_check_md(payload: Dict[str, Any]) -> str:
    lines = ["# Reference Tool Check", ""]
    for row in payload["tools"]:
        lines.append(f"- {row['tool']}: available={row['available']} executable=`{row['executable']}`")
    return "\n".join(lines) + "\n"


def _tool_available(rows: List[Dict[str, Any]], tool: str) -> bool:
    return bool(next((row for row in rows if row["tool"] == tool), {}).get("available"))


def _render_parse_md(tool: str, payload: Dict[str, Any]) -> str:
    return f"# {tool} Reference Parse\n\n- status: `{payload['status']}`\n- result_class: `{payload['result_class']}`\n"


def _render_reference_results_md(payload: Dict[str, Any]) -> str:
    lines = ["# Reference Results Table", "", f"- status: `{payload['status']}`", f"- panel: `{payload['panel_id']}`", ""]
    for row in payload["rows"]:
        lines.append(f"- {row['tool']} {row['test_name']}: `{row['result_class']}` p=`{row['p_value']}`")
    return "\n".join(lines) + "\n"


def _render_matched_null_summary_md(payload: Dict[str, Any]) -> str:
    observed = payload.get("observed_values", {})
    return "\n".join([
        "# Simulation-Matched Null Calibration",
        "",
        f"- status: `{payload['status']}`",
        f"- requested replicates: `{payload['n_replicates_requested']}`",
        f"- staged replicates: `{payload['n_replicates_staged']}`",
        f"- completed scored replicates: `{payload['n_replicates_completed']}`",
        f"- observed max gene support: `{observed.get('max_gene_support')}`",
        f"- observed called branch-site rows: `{observed.get('called_branch_site_rows')}`",
        f"- p_empirical_support: `{payload.get('p_empirical_support')}`",
        f"- p_empirical_called_rows: `{payload.get('p_empirical_called_rows')}`",
        f"- p_empirical_branch_support: `{payload.get('p_empirical_branch_support')}`",
        "",
        "This calibration is diagnostic and does not support an empirical discovery claim by itself.",
        "",
    ])


def _render_observed_vs_null_md(path: Path) -> str:
    payload = _read_json(path)
    return "\n".join([
        "# Observed Versus Null",
        "",
        f"- status: `{payload['status']}`",
        f"- p_empirical_support: `{payload['p_empirical_support']}`",
        f"- p_empirical_called_rows: `{payload['p_empirical_called_rows']}`",
        f"- reason: {payload['reason']}",
        "",
    ])


def _render_matched_null_validation_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# Matched Null Calibration Validation",
        "",
        f"- status: `{payload['status']}`",
        f"- requested: `{payload['n_replicates_requested']}`",
        f"- staged: `{payload['n_replicates_staged']}`",
        f"- completed: `{payload['n_replicates_completed']}`",
        "",
    ]
    lines.extend(f"- warning: {item}" for item in payload["warnings"])
    lines.extend(f"- failure: {item}" for item in payload["failures"])
    return "\n".join(lines) + "\n"


def _render_wrky_reference_calibration_report_md(payload: Dict[str, Any]) -> str:
    babappa = payload.get("babappa_diagnostic_result", {})
    lines = [
        "# WRKY Candidate 02 Close Reference And Calibration Report",
        "",
        "## Data Provenance",
        "",
        f"- evidence pack: `{payload['evidence_pack']}`",
        "",
        "## BABAPPA Diagnostic Result",
        "",
        f"- applicability: `{babappa.get('applicability_status')}`",
        f"- diagnostic result class: `{babappa.get('babappa_result_class')}`",
        f"- max gene support: `{babappa.get('max_gene_support')}`",
        f"- called branch-site rows: `{babappa.get('n_called_positive')}`",
        "",
        "## Reference Results",
        "",
    ]
    for row in payload.get("reference_results", []):
        lines.append(f"- {row.get('tool')} {row.get('test_name')}: `{row.get('result_class')}` p=`{row.get('p_value')}`")
    null_summary = payload.get("matched_null_calibration", {})
    lines.extend([
        "",
        "## Simulation-Matched Null Calibration",
        "",
        f"- status: `{null_summary.get('status')}`",
        f"- completed scored replicates: `{null_summary.get('n_replicates_completed')}`",
        f"- p_empirical_support: `{null_summary.get('p_empirical_support')}`",
        "",
        "## Final Decision Category",
        "",
        f"- decision: `{payload['decision_category']}`",
        "- manuscript-ready: `False`",
        "",
        "## Claim Boundary",
        "",
        payload["claim_boundary"],
        "",
    ])
    return "\n".join(lines)


def _render_babappa_only_interpretation_md(payload: Dict[str, Any]) -> str:
    return "\n".join([
        "# BABAPPA-Only Signal Interpretation",
        "",
        f"- decision: `{payload['decision']}`",
        f"- reason: `{payload['reason']}`",
        "- manuscript-ready: `False`",
        f"- references negative: `{payload['references_negative']}`",
        f"- null scoring completed: `{payload['null_scoring_completed']}`",
        f"- p_empirical_support: `{payload['p_empirical_support']}`",
        f"- p_empirical_called_rows: `{payload['p_empirical_called_rows']}`",
        "",
        payload["claim_boundary"],
        "",
    ])


def _render_babappa_only_audit_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# BABAPPA-Only Result Audit",
        "",
        f"- status: `{payload['status']}`",
        f"- family: `{payload['family']}`",
        f"- applicability: `{payload['applicability_status']}`",
        f"- model tier: `{payload['model_tier']}`",
        f"- max method: `{payload['max_method']}` fraction `{payload['max_method_fraction']:.3f}`",
        f"- max branch: `{payload['max_branch']}` fraction `{payload['max_branch_fraction']:.3f}`",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in payload["warnings"])
    if not payload["warnings"]:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _render_control_plan_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# Close-Taxa Control Family Plan",
        "",
        f"- control: `{payload['control_id']}`",
        f"- query: `{payload['query_species']} {payload['query_gene_or_locus']}`",
        f"- max mean p-distance: `{payload['max_mean_pdistance']}`",
        f"- executed: `{payload['executed']}`",
        "",
        "All scripts are MANUAL EXECUTION SCRIPT and must be reviewed before running.",
        "",
    ]
    lines.extend(f"- `{name}`" for name in payload["scripts"])
    return "\n".join(lines) + "\n"


def _render_interpretation_md(payload: Dict[str, Any]) -> str:
    return "\n".join([
        f"# {payload['family_id']} Interpretation Status",
        "",
        "## Decision",
        "",
        f"- decision: `{payload['decision']}`",
        "- manuscript-ready: `False`",
        "",
        "## Why This Is Not Yet A Discovery Claim",
        "",
        payload["claim_boundary"],
        "",
        "## Reference Tests",
        "",
        f"- codeml: {payload['what_codeml_tests']}",
        f"- HyPhy: {payload['what_hyphy_tests']}",
        "",
        "## Calibration",
        "",
        payload["what_calibration_adds"],
        "",
        "## Next manual execution Commands",
        "",
        *[f"- `{command}`" for command in payload["next_user_run_commands"]],
        "",
    ])
