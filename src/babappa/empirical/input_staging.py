"""Real empirical pilot input staging and readiness helpers."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv
from babappa.empirical.bridge import EmpiricalInputValidationConfig, validate_empirical_input

PILOT_MANIFEST_FIELDS = [
    "panel_id",
    "gene_family",
    "species_group",
    "cds_fasta",
    "tree_file",
    "foreground",
    "expected_category",
    "reference_status",
    "notes",
]
VALID_EXPECTED_CATEGORIES = {
    "known_positive",
    "likely_positive",
    "likely_negative",
    "alignment_sensitive",
    "saturated",
    "short_low_information",
    "paralogy_risk",
    "unknown",
}
VALID_REFERENCE_STATUS = {"codeml_available", "hyphy_available", "both_available", "unavailable", "planned"}
STOP_CODONS = {"TAA", "TAG", "TGA"}
ALLOWED_DNA = set("ACGTURYSWKMBDHVN-.?")
USER_RUN_ONLY = "USER-RUN ONLY - DO NOT EXECUTE IN CODEX"


@dataclass(frozen=True)
class RealPilotInputStagingConfig:
    workspace: str
    manifest: str
    outdir: str


@dataclass(frozen=True)
class RealPilotFamilyImportConfig:
    workspace: str
    panel_id: str
    gene_family: str
    species_group: str
    cds_fasta: str
    tree_file: str
    foreground: str
    expected_category: str
    reference_status: str = "planned"
    notes: str = ""


@dataclass(frozen=True)
class RealPilotBatchImportConfig:
    workspace: str
    batch_manifest: str


@dataclass(frozen=True)
class RealPilotTreeBuildingPlanConfig:
    workspace: str
    manifest: str
    outdir: str
    method: str = "iqtree"


@dataclass(frozen=True)
class CdsFastaSanitizeConfig:
    input: str
    output: str
    report: str
    mode: str = "strict"
    min_taxa: int = 3
    min_codons: int = 1


@dataclass(frozen=True)
class ForegroundCandidateConfig:
    cds_fasta: str
    tree_file: str
    outdir: str
    foreground: Optional[str] = None


@dataclass(frozen=True)
class RealPilotReadinessConfig:
    workspace: str
    manifest: str
    outdir: str


@dataclass(frozen=True)
class LocalPilotFileDiscoveryConfig:
    search_dir: str
    outdir: str


def prepare_real_pilot_inputs(config: RealPilotInputStagingConfig) -> Dict[str, Any]:
    """Create real-pilot input folders and inventory missing files."""

    workspace = Path(config.workspace)
    outdir = Path(config.outdir)
    _ensure_input_layout(workspace)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = _resolve_manifest(workspace, config.manifest)
    rows = _read_manifest_or_empty(manifest)
    inventory: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    suggestions: List[Dict[str, Any]] = []
    for row in rows:
        panel_id = row.get("panel_id", "").strip()
        cds_manifest = _resolve_panel_path(manifest, row.get("cds_fasta", ""))
        tree_manifest = _resolve_panel_path(manifest, row.get("tree_file", ""))
        suggested_cds = workspace / "input" / "cds" / f"{panel_id}.cds.fasta"
        suggested_tree = workspace / "input" / "trees" / f"{panel_id}.treefile"
        suggested_meta = workspace / "input" / "metadata" / f"{panel_id}.metadata.tsv"
        row_status = "ready" if cds_manifest.exists() and tree_manifest.exists() else "missing_inputs"
        if not cds_manifest.exists():
            missing.append(_missing_row(row, "cds_fasta", cds_manifest, suggested_cds))
        if not tree_manifest.exists():
            missing.append(_missing_row(row, "tree_file", tree_manifest, suggested_tree))
        inventory.append({
            "panel_id": panel_id,
            "gene_family": row.get("gene_family", ""),
            "foreground": row.get("foreground", ""),
            "expected_category": row.get("expected_category", ""),
            "manifest_cds_fasta": str(cds_manifest),
            "manifest_tree_file": str(tree_manifest),
            "manifest_cds_exists": cds_manifest.exists(),
            "manifest_tree_exists": tree_manifest.exists(),
            "suggested_cds_fasta": str(suggested_cds),
            "suggested_tree_file": str(suggested_tree),
            "suggested_cds_exists": suggested_cds.exists(),
            "suggested_tree_exists": suggested_tree.exists(),
            "status": row_status,
        })
        suggestions.append({
            "panel_id": panel_id,
            "cds_fasta": str(suggested_cds),
            "tree_file": str(suggested_tree),
            "metadata": str(suggested_meta),
        })
    readiness = {
        "real_pilot_input_staging_version": __version__,
        "status": "ready" if rows and not missing else "missing_inputs",
        "workspace": str(workspace),
        "manifest": str(manifest),
        "n_families": len(rows),
        "n_missing_inputs": len(missing),
        "ready_to_run": bool(rows and not missing),
        "missing_inputs": missing,
        "suggested_paths": suggestions,
        "claim_boundary": "Input staging only; no empirical discovery claim.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_tsv(outdir / "real_pilot_input_inventory.tsv", inventory, _inventory_fields())
    write_tsv(outdir / "missing_inputs.tsv", missing, _missing_fields())
    write_tsv(outdir / "suggested_file_names.tsv", suggestions, ["panel_id", "cds_fasta", "tree_file", "metadata"])
    _write_json(outdir / "real_pilot_input_readiness.json", readiness)
    (outdir / "real_pilot_input_readiness.md").write_text(_render_input_staging_md(readiness), encoding="utf-8")
    return {
        "status": readiness["status"],
        "workspace": str(workspace),
        "outdir": str(outdir),
        "families": len(rows),
        "missing_inputs": len(missing),
        "ready_to_run": readiness["ready_to_run"],
        "inventory": str(outdir / "real_pilot_input_inventory.tsv"),
        "suggested_paths": str(outdir / "suggested_file_names.tsv"),
        "readiness": str(outdir / "real_pilot_input_readiness.json"),
    }


def import_real_pilot_family(config: RealPilotFamilyImportConfig) -> Dict[str, Any]:
    """Copy one user-supplied family into the canonical real-pilot layout and update the manifest."""

    workspace = Path(config.workspace)
    _ensure_input_layout(workspace)
    outdir = workspace / "input_staging" / "import_reports"
    outdir.mkdir(parents=True, exist_ok=True)
    panel_id = _clean_panel_id(config.panel_id)
    cds_dest = workspace / "input" / "cds" / f"{panel_id}.cds.fasta"
    tree_dest = workspace / "input" / "trees" / f"{panel_id}.treefile"
    shutil.copyfile(Path(config.cds_fasta).expanduser(), cds_dest)
    shutil.copyfile(Path(config.tree_file).expanduser(), tree_dest)
    manifest = workspace / "manifest" / "real_empirical_pilot_panel.tsv"
    row = {
        "panel_id": panel_id,
        "gene_family": config.gene_family,
        "species_group": config.species_group,
        "cds_fasta": f"../input/cds/{panel_id}.cds.fasta",
        "tree_file": f"../input/trees/{panel_id}.treefile",
        "foreground": config.foreground,
        "expected_category": config.expected_category,
        "reference_status": config.reference_status,
        "notes": config.notes,
    }
    _upsert_manifest_row(manifest, row)
    validation = _validate_one_family(cds_dest, tree_dest, config.foreground, outdir / panel_id / "validation")
    report = {
        "real_pilot_family_import_version": __version__,
        "status": "ok" if validation.get("status") in {"pass", "warning"} else "fail",
        "panel_id": panel_id,
        "cds_fasta": str(cds_dest),
        "tree_file": str(tree_dest),
        "manifest": str(manifest),
        "validation": validation,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(outdir / f"{panel_id}_import_report.json", report)
    (outdir / f"{panel_id}_import_report.md").write_text(_render_import_report_md(report), encoding="utf-8")
    return {
        "status": report["status"],
        "panel_id": panel_id,
        "manifest": str(manifest),
        "cds_fasta": str(cds_dest),
        "tree_file": str(tree_dest),
        "report": str(outdir / f"{panel_id}_import_report.md"),
    }


def import_real_pilot_batch(config: RealPilotBatchImportConfig) -> Dict[str, Any]:
    """Import a batch of user-supplied real pilot families."""

    workspace = Path(config.workspace)
    batch_path = Path(config.batch_manifest)
    rows = read_tsv(batch_path)
    reports: List[Dict[str, Any]] = []
    for row in rows:
        result = import_real_pilot_family(
            RealPilotFamilyImportConfig(
                workspace=str(workspace),
                panel_id=row.get("panel_id", ""),
                gene_family=row.get("gene_family", ""),
                species_group=row.get("species_group", ""),
                cds_fasta=row.get("cds_fasta", ""),
                tree_file=row.get("tree_file", ""),
                foreground=row.get("foreground", ""),
                expected_category=row.get("expected_category", "unknown"),
                reference_status=row.get("reference_status", "planned"),
                notes=row.get("notes", ""),
            )
        )
        reports.append(result)
    outdir = workspace / "input_staging"
    outdir.mkdir(parents=True, exist_ok=True)
    write_tsv(outdir / "batch_import_report.tsv", reports, ["panel_id", "status", "cds_fasta", "tree_file", "report", "manifest"])
    payload = {
        "batch_import_version": __version__,
        "status": "ok" if all(row["status"] == "ok" for row in reports) else "warning",
        "batch_manifest": str(batch_path),
        "n_imported": len(reports),
        "status_counts": dict(Counter(row["status"] for row in reports)),
        "manifest": str(workspace / "manifest" / "real_empirical_pilot_panel.tsv"),
    }
    _write_json(outdir / "batch_import_report.json", payload)
    (outdir / "batch_import_report.md").write_text(_render_batch_import_md(payload, reports), encoding="utf-8")
    return {
        "status": payload["status"],
        "workspace": str(workspace),
        "n_imported": len(reports),
        "manifest": payload["manifest"],
        "report": str(outdir / "batch_import_report.md"),
    }


def plan_real_pilot_tree_building(config: RealPilotTreeBuildingPlanConfig) -> Dict[str, Any]:
    """Plan user-run tree building for families with CDS present and tree missing."""

    workspace = Path(config.workspace)
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = _resolve_manifest(workspace, config.manifest)
    rows = _read_manifest_or_empty(manifest)
    planned: List[Dict[str, Any]] = []
    for row in rows:
        panel_id = row.get("panel_id", "")
        cds = _resolve_panel_path(manifest, row.get("cds_fasta", ""))
        tree = _resolve_panel_path(manifest, row.get("tree_file", ""))
        if cds.exists() and not tree.exists():
            out_prefix = workspace / "trees" / panel_id / panel_id
            planned.append({
                "panel_id": panel_id,
                "cds_fasta": str(cds),
                "expected_tree": str(tree),
                "out_prefix": str(out_prefix),
                "method": config.method,
            })
    script = _render_tree_building_script(planned, config.method)
    (outdir / "build_missing_trees.sh").write_text(script, encoding="utf-8")
    (outdir / "build_missing_trees.sh").chmod(0o755)
    write_tsv(outdir / "expected_tree_outputs.tsv", planned, ["panel_id", "cds_fasta", "expected_tree", "out_prefix", "method"])
    payload = {
        "tree_building_plan_version": __version__,
        "status": "planned",
        "workspace": str(workspace),
        "manifest": str(manifest),
        "n_trees_to_build": len(planned),
        "executed": False,
        "script": str(outdir / "build_missing_trees.sh"),
    }
    _write_json(outdir / "tree_building_plan.json", payload)
    (outdir / "tree_building_plan.md").write_text(_render_tree_plan_md(payload, planned), encoding="utf-8")
    return {
        "status": "planned",
        "outdir": str(outdir),
        "n_trees_to_build": len(planned),
        "script": str(outdir / "build_missing_trees.sh"),
        "executed": False,
    }


def sanitize_cds_fasta(config: CdsFastaSanitizeConfig) -> Dict[str, Any]:
    """Sanitize and QC a CDS FASTA."""

    mode = config.mode.lower()
    if mode not in {"strict", "permissive"}:
        raise ValueError("mode must be strict or permissive")
    records, duplicate_headers = _read_fasta_ordered(Path(config.input))
    failures: List[str] = []
    warnings: List[str] = []
    sanitized: List[Tuple[str, str]] = []
    seen: set[str] = set()
    lengths: List[int] = []
    for header, sequence in records:
        clean_header = re.sub(r"\s+", "_", header.strip())
        if clean_header != header.strip():
            warnings.append(f"header_whitespace_sanitized:{header}")
        if clean_header in seen:
            failures.append(f"duplicate_id:{clean_header}")
        seen.add(clean_header)
        seq = re.sub(r"\s+", "", sequence).upper().replace("U", "T")
        illegal = sorted(set(seq) - ALLOWED_DNA)
        if illegal:
            failures.append(f"illegal_characters:{clean_header}:{''.join(illegal)}")
        if len(seq) % 3 != 0:
            failures.append(f"length_not_divisible_by_3:{clean_header}:{len(seq)}")
        codons = [seq[i:i + 3] for i in range(0, len(seq), 3)]
        internal_stops = [str(i) for i, codon in enumerate(codons[:-1]) if codon in STOP_CODONS]
        if internal_stops:
            failures.append(f"internal_stop_codon:{clean_header}:{','.join(internal_stops)}")
        if codons and codons[-1] in STOP_CODONS:
            warnings.append(f"terminal_stop_codon:{clean_header}:{codons[-1]}")
        ambiguous = sum(1 for base in seq if base not in {"A", "C", "G", "T", "-"})
        gaps = seq.count("-")
        if ambiguous:
            warnings.append(f"ambiguous_bases:{clean_header}:{ambiguous}")
        if gaps:
            warnings.append(f"gaps_present:{clean_header}:{gaps}")
        lengths.append(len(seq))
        sanitized.append((clean_header, seq))
    if duplicate_headers:
        failures.extend(f"duplicate_raw_header:{item}" for item in duplicate_headers)
    if len(sanitized) < config.min_taxa:
        failures.append(f"too_few_taxa:{len(sanitized)}<{config.min_taxa}")
    min_codons = min((len(seq) // 3 for _header, seq in sanitized), default=0)
    if min_codons < config.min_codons:
        failures.append(f"too_few_codons:{min_codons}<{config.min_codons}")
    if len(set(lengths)) > 1:
        warnings.append("unequal_sequence_lengths")
    status = "fail" if failures and mode == "strict" else ("warning" if failures or warnings else "ok")
    if status != "fail":
        _write_fasta(Path(config.output), sanitized)
    report = {
        "sanitize_cds_fasta_version": __version__,
        "status": status,
        "mode": mode,
        "input": config.input,
        "output": config.output,
        "n_sequences": len(sanitized),
        "min_codons": min_codons,
        "failures": failures,
        "warnings": warnings,
        "output_written": status != "fail",
    }
    _write_json(Path(config.report), report)
    Path(str(config.report) + ".md").write_text(_render_sanitize_md(report), encoding="utf-8")
    return {
        "status": status,
        "output": config.output,
        "report": config.report,
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "output_written": status != "fail",
        "failures": failures,
        "warnings": warnings,
    }


def list_foreground_candidates(config: ForegroundCandidateConfig) -> Dict[str, Any]:
    """List FASTA/tree taxa and possible foreground labels."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fasta_records, _duplicates = _read_fasta_ordered(Path(config.cds_fasta))
    fasta_taxa = [header for header, _seq in fasta_records]
    tree_text = Path(config.tree_file).read_text(encoding="utf-8") if Path(config.tree_file).exists() else ""
    tree_tips = sorted(_parse_newick_tips(tree_text))
    matching = sorted(set(fasta_taxa) & set(tree_tips))
    missing_in_tree = sorted(set(fasta_taxa) - set(tree_tips))
    missing_in_fasta = sorted(set(tree_tips) - set(fasta_taxa))
    foreground_valid = None if config.foreground is None else config.foreground in matching
    payload = {
        "foreground_candidates_version": __version__,
        "cds_fasta": config.cds_fasta,
        "tree_file": config.tree_file,
        "fasta_taxa": fasta_taxa,
        "tree_tips": tree_tips,
        "matching_tips": matching,
        "missing_in_tree": missing_in_tree,
        "missing_in_fasta": missing_in_fasta,
        "suggested_foreground_labels": matching,
        "foreground": config.foreground,
        "foreground_valid": foreground_valid,
        "warnings": _foreground_warnings(missing_in_tree, missing_in_fasta, foreground_valid),
    }
    _write_json(outdir / "foreground_candidates.json", payload)
    write_tsv(outdir / "foreground_candidates.tsv", [{"taxon": item, "in_fasta": item in fasta_taxa, "in_tree": item in tree_tips} for item in sorted(set(fasta_taxa) | set(tree_tips))], ["taxon", "in_fasta", "in_tree"])
    (outdir / "foreground_candidates.md").write_text(_render_foreground_md(payload), encoding="utf-8")
    return {
        "status": "ok" if not payload["warnings"] else "warning",
        "outdir": str(outdir),
        "matching_tips": len(matching),
        "foreground_valid": foreground_valid,
        "json": str(outdir / "foreground_candidates.json"),
        "warnings": payload["warnings"],
    }


def validate_real_pilot_readiness(config: RealPilotReadinessConfig) -> Dict[str, Any]:
    """Gate the real empirical pilot before BABAPPA scoring is allowed."""

    workspace = Path(config.workspace)
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = _resolve_manifest(workspace, config.manifest)
    rows = _read_manifest_or_empty(manifest)
    result_rows: List[Dict[str, Any]] = []
    for row in rows:
        panel_id = row.get("panel_id", "")
        cds = _resolve_panel_path(manifest, row.get("cds_fasta", ""))
        tree = _resolve_panel_path(manifest, row.get("tree_file", ""))
        status = "not_run"
        validation_failures: List[str] = []
        validation_warnings: List[str] = []
        foreground_valid = False
        tree_compatible = False
        if not cds.exists() or not tree.exists():
            status = "missing_files"
            if not cds.exists():
                validation_failures.append(f"missing_cds_fasta:{cds}")
            if not tree.exists():
                validation_failures.append(f"missing_tree_file:{tree}")
        else:
            validation = _validate_one_family(cds, tree, row.get("foreground", ""), outdir / "per_family" / panel_id)
            status = str(validation.get("status"))
            validation_failures = list(validation.get("failures") or [])
            validation_warnings = list(validation.get("warnings") or [])
            candidates = _foreground_payload(cds, tree, row.get("foreground", ""))
            foreground_valid = bool(candidates.get("foreground_valid"))
            tree_compatible = not candidates.get("missing_in_tree") and not candidates.get("missing_in_fasta")
            if not foreground_valid:
                validation_failures.append(f"foreground_invalid:{row.get('foreground', '')}")
            if not tree_compatible:
                validation_failures.append("tree_fasta_tip_mismatch")
        result_rows.append({
            "panel_id": panel_id,
            "gene_family": row.get("gene_family", ""),
            "expected_category": row.get("expected_category", ""),
            "cds_fasta": str(cds),
            "tree_file": str(tree),
            "files_present": cds.exists() and tree.exists(),
            "input_qc_status": status,
            "foreground": row.get("foreground", ""),
            "foreground_valid": foreground_valid,
            "tree_compatible": tree_compatible,
            "ready": status in {"pass", "warning"} and foreground_valid and tree_compatible,
            "failures": ";".join(validation_failures),
            "warnings": ";".join(validation_warnings),
        })
    ready_to_run = bool(result_rows and all(row["ready"] for row in result_rows))
    payload = {
        "real_pilot_readiness_version": __version__,
        "status": "ready" if ready_to_run else "not_ready",
        "ready_to_run": ready_to_run,
        "workspace": str(workspace),
        "manifest": str(manifest),
        "total_families": len(result_rows),
        "files_present": sum(1 for row in result_rows if row["files_present"]),
        "files_missing": sum(1 for row in result_rows if not row["files_present"]),
        "input_qc_counts": dict(Counter(row["input_qc_status"] for row in result_rows)),
        "foreground_invalid": sum(1 for row in result_rows if row["files_present"] and not row["foreground_valid"]),
        "tree_incompatible": sum(1 for row in result_rows if row["files_present"] and not row["tree_compatible"]),
        "recommended_next_action": _readiness_next_action(result_rows, ready_to_run),
        "claim_boundary": "Readiness gate only; do not run BABAPPA pilot until ready_to_run is true.",
        "rows": result_rows,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(outdir / "real_pilot_readiness.json", payload)
    write_tsv(outdir / "real_pilot_readiness.tsv", result_rows, _readiness_fields())
    (outdir / "real_pilot_readiness.md").write_text(_render_readiness_md(payload), encoding="utf-8")
    return {
        "status": payload["status"],
        "ready_to_run": ready_to_run,
        "outdir": str(outdir),
        "total_families": len(result_rows),
        "files_missing": payload["files_missing"],
        "foreground_invalid": payload["foreground_invalid"],
        "tree_incompatible": payload["tree_incompatible"],
        "json": str(outdir / "real_pilot_readiness.json"),
    }


def discover_local_pilot_files(config: LocalPilotFileDiscoveryConfig) -> Dict[str, Any]:
    """Discover local FASTA/tree files and suggest likely pairs without editing manifests."""

    search_dir = Path(config.search_dir).expanduser()
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fasta_exts = {".fasta", ".fa", ".fas"}
    tree_exts = {".treefile", ".nwk", ".newick"}
    fasta_files: List[Path] = []
    tree_files: List[Path] = []
    if search_dir.exists():
        for path in search_dir.rglob("*"):
            if not path.is_file():
                continue
            lower = path.name.lower()
            if lower.endswith((".cds.fasta", ".fasta", ".fa", ".fas")):
                fasta_files.append(path)
            if any(lower.endswith(ext) for ext in tree_exts):
                tree_files.append(path)
    inventory = [{"path": str(path), "kind": "fasta" if path in fasta_files else "tree", "stem_key": _stem_key(path)} for path in fasta_files + tree_files]
    pairs: List[Dict[str, Any]] = []
    for fasta in fasta_files:
        best_tree, score = _best_tree_match(fasta, tree_files)
        pairs.append({
            "cds_fasta": str(fasta),
            "suggested_tree_file": str(best_tree) if best_tree else "",
            "similarity_score": f"{score:.3f}",
            "panel_id_suggestion": _stem_key(fasta),
        })
    write_tsv(outdir / "local_file_inventory.tsv", inventory, ["path", "kind", "stem_key"])
    write_tsv(outdir / "local_pair_suggestions.tsv", pairs, ["panel_id_suggestion", "cds_fasta", "suggested_tree_file", "similarity_score"])
    payload = {
        "local_pilot_file_discovery_version": __version__,
        "status": "ok",
        "search_dir": str(search_dir),
        "n_fasta": len(fasta_files),
        "n_tree": len(tree_files),
        "n_pair_suggestions": len(pairs),
        "manifest_modified": False,
    }
    _write_json(outdir / "local_discovery.json", payload)
    (outdir / "local_discovery_report.md").write_text(_render_discovery_md(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "n_fasta": len(fasta_files),
        "n_tree": len(tree_files),
        "n_pair_suggestions": len(pairs),
        "manifest_modified": False,
    }


def _ensure_input_layout(workspace: Path) -> None:
    for path in [
        workspace / "input",
        workspace / "input" / "cds",
        workspace / "input" / "trees",
        workspace / "input" / "metadata",
        workspace / "manifest",
        workspace / "input_staging",
        workspace / "logs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _resolve_manifest(workspace: Path, manifest: str) -> Path:
    raw = Path(manifest).expanduser()
    if raw.is_absolute() or raw.exists():
        return raw
    candidate = workspace / "manifest" / raw
    if candidate.exists():
        return candidate
    return candidate


def _resolve_panel_path(manifest: Path, raw: str | None) -> Path:
    path = Path(str(raw or "")).expanduser()
    if path.is_absolute():
        return path
    return (manifest.parent / path).resolve()


def _read_manifest_or_empty(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    return read_tsv(path)


def _missing_row(row: Dict[str, str], kind: str, manifest_path: Path, suggested_path: Path) -> Dict[str, Any]:
    return {
        "panel_id": row.get("panel_id", ""),
        "gene_family": row.get("gene_family", ""),
        "missing_kind": kind,
        "manifest_path": str(manifest_path),
        "suggested_path": str(suggested_path),
    }


def _clean_panel_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not cleaned:
        raise ValueError("panel_id is required")
    return cleaned


def _upsert_manifest_row(manifest: Path, row: Dict[str, str]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = read_tsv(manifest) if manifest.exists() else []
    updated = False
    out_rows: List[Dict[str, str]] = []
    for existing in rows:
        if existing.get("panel_id") == row["panel_id"]:
            out_rows.append(row)
            updated = True
        else:
            out_rows.append({field: existing.get(field, "") for field in PILOT_MANIFEST_FIELDS})
    if not updated:
        out_rows.append(row)
    write_tsv(manifest, out_rows, PILOT_MANIFEST_FIELDS)


def _validate_one_family(cds: Path, tree: Path, foreground: str, outdir: Path) -> Dict[str, Any]:
    try:
        return validate_empirical_input(
            EmpiricalInputValidationConfig(
                cds_fasta=str(cds),
                tree=str(tree),
                foreground=foreground,
                outdir=str(outdir),
                min_taxa=6,
                min_codons=60,
            )
        )
    except Exception as exc:
        return {"status": "fail", "failures": [str(exc)], "warnings": []}


def _read_fasta_ordered(path: Path) -> Tuple[List[Tuple[str, str]], List[str]]:
    records: List[Tuple[str, str]] = []
    duplicates: List[str] = []
    seen: set[str] = set()
    header: Optional[str] = None
    seq_parts: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if header is not None:
                if header in seen:
                    duplicates.append(header)
                seen.add(header)
                records.append((header, "".join(seq_parts)))
            header = line[1:].strip()
            seq_parts = []
        elif header is not None:
            seq_parts.append(line.strip())
    if header is not None:
        if header in seen:
            duplicates.append(header)
        records.append((header, "".join(seq_parts)))
    return records, duplicates


def _write_fasta(path: Path, records: Sequence[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for header, seq in records:
        lines.append(f">{header}")
        for start in range(0, len(seq), 80):
            lines.append(seq[start:start + 80])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_newick_tips(tree_text: str) -> set[str]:
    # Tip labels are introduced after "(" or ",". Internal support labels from
    # IQ-TREE, such as ")100/100:0.1", must not be counted as taxa.
    return {
        token.strip().strip("'\"")
        for token in re.findall(r"(?<=[(,])\s*([^():,;\s]+)\s*(?=[:),;])", tree_text)
        if token.strip()
    }


def _foreground_payload(cds: Path, tree: Path, foreground: str) -> Dict[str, Any]:
    records, _duplicates = _read_fasta_ordered(cds)
    fasta_taxa = [header for header, _seq in records]
    tree_tips = sorted(_parse_newick_tips(tree.read_text(encoding="utf-8")))
    matching = sorted(set(fasta_taxa) & set(tree_tips))
    return {
        "fasta_taxa": fasta_taxa,
        "tree_tips": tree_tips,
        "matching_tips": matching,
        "missing_in_tree": sorted(set(fasta_taxa) - set(tree_tips)),
        "missing_in_fasta": sorted(set(tree_tips) - set(fasta_taxa)),
        "foreground_valid": foreground in matching,
    }


def _foreground_warnings(missing_in_tree: List[str], missing_in_fasta: List[str], foreground_valid: Optional[bool]) -> List[str]:
    warnings: List[str] = []
    if missing_in_tree:
        warnings.append("fasta_taxa_missing_from_tree:" + ",".join(missing_in_tree))
    if missing_in_fasta:
        warnings.append("tree_tips_missing_from_fasta:" + ",".join(missing_in_fasta))
    if foreground_valid is False:
        warnings.append("foreground_not_in_matching_tips")
    return warnings


def _readiness_next_action(rows: List[Dict[str, Any]], ready: bool) -> str:
    if ready:
        return "Run the real BABAPPA empirical pilot with run-empirical-pilot-panel."
    missing = [row["panel_id"] for row in rows if not row["files_present"]]
    if missing:
        return "Provide CDS FASTA and tree files for: " + ",".join(missing[:12])
    invalid_fg = [row["panel_id"] for row in rows if row["files_present"] and not row["foreground_valid"]]
    if invalid_fg:
        return "Repair foreground labels for: " + ",".join(invalid_fg[:12])
    return "Repair input QC failures shown in real_pilot_readiness.tsv."


def _best_tree_match(fasta: Path, trees: List[Path]) -> Tuple[Optional[Path], float]:
    if not trees:
        return None, 0.0
    fasta_key = set(_stem_key(fasta).split("_"))
    best: Optional[Path] = None
    best_score = -1.0
    for tree in trees:
        tree_key = set(_stem_key(tree).split("_"))
        union = fasta_key | tree_key
        score = len(fasta_key & tree_key) / len(union) if union else 0.0
        if score > best_score:
            best = tree
            best_score = score
    return best, max(best_score, 0.0)


def _stem_key(path: Path) -> str:
    name = path.name.lower()
    for suffix in [".cds.fasta", ".treefile", ".newick", ".fasta", ".fas", ".fa", ".nwk"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _inventory_fields() -> List[str]:
    return [
        "panel_id",
        "gene_family",
        "foreground",
        "expected_category",
        "manifest_cds_fasta",
        "manifest_tree_file",
        "manifest_cds_exists",
        "manifest_tree_exists",
        "suggested_cds_fasta",
        "suggested_tree_file",
        "suggested_cds_exists",
        "suggested_tree_exists",
        "status",
    ]


def _missing_fields() -> List[str]:
    return ["panel_id", "gene_family", "missing_kind", "manifest_path", "suggested_path"]


def _readiness_fields() -> List[str]:
    return [
        "panel_id",
        "gene_family",
        "expected_category",
        "cds_fasta",
        "tree_file",
        "files_present",
        "input_qc_status",
        "foreground",
        "foreground_valid",
        "tree_compatible",
        "ready",
        "failures",
        "warnings",
    ]


def _render_input_staging_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# Real Pilot Input Readiness",
        "",
        f"- status: `{payload['status']}`",
        f"- families: `{payload['n_families']}`",
        f"- missing inputs: `{payload['n_missing_inputs']}`",
        f"- ready to run: `{payload['ready_to_run']}`",
        "",
        "No biological data were fabricated.",
        "",
    ]
    if payload["missing_inputs"]:
        lines.extend(["## Missing Inputs", ""])
        lines.extend(f"- {row['panel_id']} {row['missing_kind']}: `{row['suggested_path']}`" for row in payload["missing_inputs"])
        lines.append("")
    return "\n".join(lines)


def _render_import_report_md(report: Dict[str, Any]) -> str:
    return "\n".join([
        "# Real Pilot Family Import",
        "",
        f"- panel_id: `{report['panel_id']}`",
        f"- status: `{report['status']}`",
        f"- CDS FASTA: `{report['cds_fasta']}`",
        f"- tree: `{report['tree_file']}`",
        f"- validation: `{report['validation'].get('status')}`",
        "",
    ])


def _render_batch_import_md(payload: Dict[str, Any], reports: List[Dict[str, Any]]) -> str:
    lines = ["# Real Pilot Batch Import", "", f"- status: `{payload['status']}`", f"- imported: `{payload['n_imported']}`", ""]
    for row in reports:
        lines.append(f"- {row['panel_id']}: {row['status']}")
    lines.append("")
    return "\n".join(lines)


def _render_tree_building_script(rows: List[Dict[str, Any]], method: str) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"echo '{USER_RUN_ONLY}'", ""]
    for row in rows:
        prefix = row["out_prefix"]
        lines.append(f"mkdir -p {shlex_quote(str(Path(prefix).parent))}")
        lines.append(f"# IQ-TREE2/iqtree command for {row['panel_id']}")
        lines.append(f"iqtree2 -s {shlex_quote(row['cds_fasta'])} -m MFP -B 1000 -alrt 1000 -T AUTO --seed 42 -pre {shlex_quote(prefix)}")
        lines.append(f"# FastTree fallback template: FastTree -nt {shlex_quote(row['cds_fasta'])} > {shlex_quote(row['expected_tree'])}")
        lines.append("")
    if not rows:
        lines.append("echo 'No missing-tree families with CDS FASTA present.'")
    return "\n".join(lines)


def _render_tree_plan_md(payload: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    lines = ["# Real Pilot Tree Building Plan", "", USER_RUN_ONLY, "", f"- trees to build: `{len(rows)}`", "- executed: `False`", ""]
    for row in rows:
        lines.append(f"- {row['panel_id']}: `{row['cds_fasta']}` -> `{row['expected_tree']}`")
    lines.append("")
    return "\n".join(lines)


def _render_sanitize_md(report: Dict[str, Any]) -> str:
    lines = ["# CDS FASTA Sanitation", "", f"- status: `{report['status']}`", f"- mode: `{report['mode']}`", f"- output written: `{report['output_written']}`", ""]
    if report["failures"]:
        lines.extend(["## Failures", *[f"- {item}" for item in report["failures"]], ""])
    if report["warnings"]:
        lines.extend(["## Warnings", *[f"- {item}" for item in report["warnings"]], ""])
    return "\n".join(lines)


def _render_foreground_md(payload: Dict[str, Any]) -> str:
    lines = ["# Foreground Candidates", "", f"- matching tips: `{len(payload['matching_tips'])}`", f"- foreground valid: `{payload['foreground_valid']}`", "", "## Suggested Labels", ""]
    lines.extend(f"- {item}" for item in payload["suggested_foreground_labels"])
    lines.append("")
    return "\n".join(lines)


def _render_readiness_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# Real Pilot Readiness Gate",
        "",
        f"- status: `{payload['status']}`",
        f"- ready_to_run: `{payload['ready_to_run']}`",
        f"- total families: `{payload['total_families']}`",
        f"- files missing: `{payload['files_missing']}`",
        f"- foreground invalid: `{payload['foreground_invalid']}`",
        f"- tree incompatible: `{payload['tree_incompatible']}`",
        "",
        "## Recommended Next Action",
        "",
        payload["recommended_next_action"],
        "",
    ]
    return "\n".join(lines)


def _render_discovery_md(payload: Dict[str, Any]) -> str:
    return "\n".join([
        "# Local Pilot File Discovery",
        "",
        f"- search dir: `{payload['search_dir']}`",
        f"- FASTA files: `{payload['n_fasta']}`",
        f"- tree files: `{payload['n_tree']}`",
        f"- pair suggestions: `{payload['n_pair_suggestions']}`",
        "- manifest modified: `False`",
        "",
    ])


def shlex_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"
