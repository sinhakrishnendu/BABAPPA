"""OOD-aware empirical family prefiltering and acquisition planning."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv

USER_RUN_ONLY = "USER-RUN ONLY - DO NOT EXECUTE IN CODEX"
STOP_CODONS = {"TAA", "TAG", "TGA"}
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


@dataclass(frozen=True)
class EmpiricalFamilyPrefilterConfig:
    cds_fasta: str
    tree_file: str
    foreground: str
    outdir: str
    max_mean_pdistance: float = 0.35
    min_taxa: int = 6
    min_codons: int = 100


@dataclass(frozen=True)
class EmpiricalFamilyAcquisitionPlanConfig:
    family_id: str
    query_species: str
    query_gene_or_locus: str
    target_taxa_file: str
    outdir: str
    source: str = "ensembl_plants"
    strategy: str = "blastp_best_hit"


@dataclass(frozen=True)
class TargetTaxaRecommendationConfig:
    pilot_type: str
    outdir: str


@dataclass(frozen=True)
class OODAwareFamilyBuildPlanConfig:
    family_id: str
    query_species: str
    query_gene_or_locus: str
    target_taxa_file: str
    outdir: str
    max_mean_pdistance: float = 0.35
    min_taxa: int = 6
    min_codons: int = 100


@dataclass(frozen=True)
class AddPrefilteredFamilyConfig:
    workspace: str
    prefilter_dir: str
    panel_id: str
    expected_category: str
    reference_status: str = "planned"
    allow_diagnostic_only: bool = False


@dataclass(frozen=True)
class EmpiricalOODSummaryConfig:
    workspace: str
    outdir: str


def prefilter_empirical_family(config: EmpiricalFamilyPrefilterConfig) -> Dict[str, Any]:
    """Screen a real empirical family before BABAPPA scoring."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cds_path = Path(config.cds_fasta)
    tree_path = Path(config.tree_file)
    records, duplicate_ids = _read_fasta(cds_path)
    tree_text = tree_path.read_text(encoding="utf-8") if tree_path.exists() else ""
    tree_tips = _parse_newick_tips(tree_text)
    fasta_ids = set(records)
    distances = _pairwise_distances(records)
    mean_p = sum(distances) / len(distances) if distances else 0.0
    max_p = max(distances) if distances else 0.0
    lengths = [len(seq.replace("-", "")) for seq in records.values()]
    codons = [length // 3 for length in lengths if length >= 3]
    min_codons = min(codons) if codons else 0
    length_ratio = (max(lengths) / min(lengths)) if lengths and min(lengths) else 0.0
    branch_lengths = _branch_lengths(tree_text)
    long_branch_outliers = _long_branch_outliers(branch_lengths)
    internal_stops, terminal_stops = _stop_codon_status(records)
    gap_fraction = _gap_fraction(records)
    ambiguous_fraction = _ambiguous_fraction(records)
    duplicate_species = _duplicate_species_labels(records)
    missing_in_tree = sorted(fasta_ids - tree_tips)
    missing_in_fasta = sorted(tree_tips - fasta_ids)
    foreground_present = config.foreground in fasta_ids and config.foreground in tree_tips
    paralogy_flags = []
    if duplicate_ids:
        paralogy_flags.append("duplicate_ids")
    if duplicate_species:
        paralogy_flags.append("duplicate_species:" + ",".join(duplicate_species))
    if length_ratio > 1.75:
        paralogy_flags.append(f"extreme_length_ratio:{length_ratio:.3f}")
    if long_branch_outliers:
        paralogy_flags.append("long_branch_outliers:" + ",".join(long_branch_outliers))
    if max_p > 0.65:
        paralogy_flags.append(f"unusually_divergent_pair:{max_p:.6g}")
    decision, recommended_action = _prefilter_decision(
        n_taxa=len(records),
        min_codons=min_codons,
        mean_p=mean_p,
        max_p=max_p,
        tree_ok=not missing_in_tree and not missing_in_fasta and foreground_present,
        paralogy_flags=paralogy_flags,
        max_mean_pdistance=config.max_mean_pdistance,
        min_taxa=config.min_taxa,
        required_min_codons=config.min_codons,
    )
    payload = {
        "empirical_family_prefilter_version": __version__,
        "status": "ok",
        "decision": decision,
        "recommended_action": recommended_action,
        "cds_fasta": str(cds_path),
        "tree_file": str(tree_path),
        "foreground": config.foreground,
        "n_taxa": len(records),
        "n_codons": min_codons,
        "mean_pairwise_p_distance": mean_p,
        "max_pairwise_p_distance": max_p,
        "saturation_proxy": _saturation_proxy(mean_p),
        "gap_fraction": gap_fraction,
        "ambiguous_fraction": ambiguous_fraction,
        "terminal_stop_codons": terminal_stops,
        "internal_stop_codons": internal_stops,
        "foreground_present": foreground_present,
        "tree_tip_compatible": not missing_in_tree and not missing_in_fasta,
        "missing_in_tree": missing_in_tree,
        "missing_in_fasta": missing_in_fasta,
        "length_ratio": length_ratio,
        "long_branch_outliers": long_branch_outliers,
        "paralogy_risk_flags": paralogy_flags,
        "max_mean_pdistance": config.max_mean_pdistance,
        "min_taxa_required": config.min_taxa,
        "min_codons_required": config.min_codons,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(outdir / "empirical_family_prefilter.json", payload)
    write_tsv(outdir / "empirical_family_prefilter.tsv", [_prefilter_tsv_row(payload)], _prefilter_fields())
    (outdir / "empirical_family_prefilter.md").write_text(_render_prefilter_md(payload), encoding="utf-8")
    return {
        "status": "ok",
        "decision": decision,
        "outdir": str(outdir),
        "n_taxa": len(records),
        "n_codons": min_codons,
        "mean_pdistance": mean_p,
        "recommended_action": recommended_action,
        "json": str(outdir / "empirical_family_prefilter.json"),
    }


def plan_empirical_family_acquisition(config: EmpiricalFamilyAcquisitionPlanConfig) -> Dict[str, Any]:
    """Create USER-RUN scripts for family acquisition; do not execute downloads."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    taxa_rows = _read_tsv_optional(Path(config.target_taxa_file))
    payload = {
        "empirical_family_acquisition_plan_version": __version__,
        "status": "planned",
        "family_id": config.family_id,
        "query_species": config.query_species,
        "query_gene_or_locus": config.query_gene_or_locus,
        "target_taxa_file": config.target_taxa_file,
        "source": config.source,
        "strategy": config.strategy,
        "n_target_taxa": len(taxa_rows),
        "executed": False,
    }
    scripts = {
        "download_ensembl_proteome_cds.sh": _script_download(config),
        "run_blastp_best_hit.sh": _script_blast(config),
        "recover_cds_from_best_hits.sh": _script_recover(config),
        "build_tree.sh": _script_tree(config),
        "prefilter_family.sh": _script_prefilter(config),
    }
    for name, text in scripts.items():
        path = outdir / name
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)
    _write_json(outdir / "expected_outputs.json", _expected_acquisition_outputs(config))
    _write_json(outdir / "acquisition_plan.json", payload)
    (outdir / "acquisition_plan.md").write_text(_render_acquisition_plan_md(payload), encoding="utf-8")
    return {"status": "planned", "outdir": str(outdir), "scripts": sorted(scripts), "executed": False}


def recommend_target_taxa(config: TargetTaxaRecommendationConfig) -> Dict[str, Any]:
    """Write a recommended target-taxa template for a pilot type."""

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows, note = _target_taxa_rows(config.pilot_type)
    write_tsv(outdir / "recommended_target_taxa.tsv", rows, ["taxon_label", "ensembl_dir_or_source_hint", "category", "notes"])
    payload = {
        "target_taxa_recommendation_version": __version__,
        "status": "ok",
        "pilot_type": config.pilot_type,
        "n_taxa": len(rows),
        "recommendation": note,
    }
    _write_json(outdir / "recommendation.json", payload)
    (outdir / "recommendation.md").write_text(_render_recommendation_md(payload, rows), encoding="utf-8")
    return {"status": "ok", "pilot_type": config.pilot_type, "outdir": str(outdir), "n_taxa": len(rows), "recommendation": note}


def plan_ood_aware_family_build(config: OODAwareFamilyBuildPlanConfig) -> Dict[str, Any]:
    """Plan an OOD-gated family build workflow without executing it."""

    base = EmpiricalFamilyAcquisitionPlanConfig(
        family_id=config.family_id,
        query_species=config.query_species,
        query_gene_or_locus=config.query_gene_or_locus,
        target_taxa_file=config.target_taxa_file,
        outdir=config.outdir,
        source="ensembl_plants",
        strategy="blastp_best_hit",
    )
    plan_empirical_family_acquisition(base)
    outdir = Path(config.outdir)
    payload = {
        "ood_aware_family_build_plan_version": __version__,
        "status": "planned",
        "family_id": config.family_id,
        "max_mean_pdistance": config.max_mean_pdistance,
        "min_taxa": config.min_taxa,
        "min_codons": config.min_codons,
        "import_gate": "accept_or_accept_with_caution_only",
        "executed": False,
    }
    _write_json(outdir / "ood_aware_family_build_plan.json", payload)
    (outdir / "run_ood_aware_family_build.sh").write_text(_render_ood_build_script(config), encoding="utf-8")
    (outdir / "run_ood_aware_family_build.sh").chmod(0o755)
    (outdir / "ood_aware_family_build_plan.md").write_text(_render_ood_build_md(payload), encoding="utf-8")
    return {
        "status": "planned",
        "outdir": str(outdir),
        "family_id": config.family_id,
        "max_mean_pdistance": config.max_mean_pdistance,
        "script": str(outdir / "run_ood_aware_family_build.sh"),
        "executed": False,
    }


def add_prefiltered_family_to_pilot(config: AddPrefilteredFamilyConfig) -> Dict[str, Any]:
    """Add a prefiltered family to the pilot manifest only if it passes the OOD gate."""

    workspace = Path(config.workspace)
    prefilter = _read_json(Path(config.prefilter_dir) / "empirical_family_prefilter.json")
    decision = str(prefilter.get("decision", ""))
    allowed = decision in {"accept", "accept_with_caution"} or (config.allow_diagnostic_only and decision == "diagnostic_only")
    outdir = workspace / "input_staging" / "prefilter_import_reports"
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "add_prefiltered_family_version": __version__,
        "status": "ok" if allowed else "blocked",
        "panel_id": config.panel_id,
        "decision": decision,
        "allow_diagnostic_only": config.allow_diagnostic_only,
        "reason": "" if allowed else f"prefilter_decision_not_importable:{decision}",
    }
    if allowed:
        manifest = workspace / "manifest" / "real_empirical_pilot_panel.tsv"
        row = {
            "panel_id": config.panel_id,
            "gene_family": config.panel_id,
            "species_group": "empirical_prefiltered",
            "cds_fasta": _relative_to_manifest(manifest, Path(str(prefilter.get("cds_fasta", "")))),
            "tree_file": _relative_to_manifest(manifest, Path(str(prefilter.get("tree_file", "")))),
            "foreground": str(prefilter.get("foreground", "")),
            "expected_category": config.expected_category,
            "reference_status": config.reference_status,
            "notes": f"prefilter_decision={decision}; diagnostic_only={decision == 'diagnostic_only'}",
        }
        _upsert_manifest_row(manifest, row)
        report["manifest"] = str(manifest)
    _write_json(outdir / f"{config.panel_id}_prefilter_import_report.json", report)
    (outdir / f"{config.panel_id}_prefilter_import_report.md").write_text(_render_add_prefilter_md(report), encoding="utf-8")
    return {"status": report["status"], "decision": decision, "panel_id": config.panel_id, "report": str(outdir / f"{config.panel_id}_prefilter_import_report.md")}


def summarize_empirical_ood(config: EmpiricalOODSummaryConfig) -> Dict[str, Any]:
    """Summarize prefilter and empirical applicability OOD status."""

    workspace = Path(config.workspace)
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: Dict[str, Dict[str, Any]] = {}
    for path in workspace.glob("prefilter/*/empirical_family_prefilter.json"):
        data = _read_json(path)
        family = path.parent.name
        rows.setdefault(family, {"family": family})
        rows[family].update({
            "n_taxa": data.get("n_taxa", ""),
            "n_codons": data.get("n_codons", ""),
            "mean_pdistance": data.get("mean_pairwise_p_distance", ""),
            "saturation_proxy": data.get("saturation_proxy", ""),
            "prefilter_decision": data.get("decision", ""),
            "recommended_action": data.get("recommended_action", ""),
        })
    for path in workspace.glob("babappa_run*/per_family/*/empirical_applicability/empirical_applicability.json"):
        family = path.parent.parent.name
        data = _read_json(path)
        rows.setdefault(family, {"family": family})
        rows[family].update({
            "applicability": data.get("applicability_status", ""),
            "diagnostic_only": data.get("diagnostic_only_if_scored", ""),
        })
    for path in workspace.glob("babappa_run*/per_family/*/empirical_scores/empirical_scoring_manifest.json"):
        family = path.parent.parent.name
        data = _read_json(path)
        rows.setdefault(family, {"family": family})
        rows[family].update({
            "diagnostic_only": data.get("diagnostic_only", rows[family].get("diagnostic_only", "")),
            "tier_model": data.get("tier_model", ""),
            "score_rows": data.get("n_rows", ""),
        })
    table = [dict(row) for row in rows.values()]
    for row in table:
        row.setdefault("recommended_action", _ood_recommended_action(row))
    fields = ["family", "n_taxa", "n_codons", "mean_pdistance", "saturation_proxy", "applicability", "diagnostic_only", "prefilter_decision", "tier_model", "score_rows", "recommended_action"]
    write_tsv(outdir / "empirical_ood_summary.tsv", table, fields)
    payload = {
        "empirical_ood_summary_version": __version__,
        "status": "ok",
        "workspace": str(workspace),
        "n_families": len(table),
        "decision_counts": dict(Counter(str(row.get("prefilter_decision", "")) for row in table)),
        "rows": table,
    }
    _write_json(outdir / "empirical_ood_summary.json", payload)
    (outdir / "empirical_ood_summary.md").write_text(_render_ood_summary_md(payload), encoding="utf-8")
    return {"status": "ok", "outdir": str(outdir), "n_families": len(table), "json": str(outdir / "empirical_ood_summary.json")}


def _read_fasta(path: Path) -> Tuple[Dict[str, str], List[str]]:
    records: Dict[str, List[str]] = {}
    duplicates: List[str] = []
    current: Optional[str] = None
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


def _pairwise_distances(records: Dict[str, str]) -> List[float]:
    values = list(records.values())
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
    return distances


def _parse_newick_tips(text: str) -> set[str]:
    return {token for token in re.findall(r"(?<=[(,])\s*([^():,;\s]+)\s*(?=[:),;])", text) if token}


def _branch_lengths(text: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for idx, match in enumerate(re.finditer(r"([^(),:;\s]+)?\s*:\s*([0-9.eE+-]+)", text)):
        label = (match.group(1) or f"internal_{idx}").strip()
        try:
            result[label] = float(match.group(2))
        except ValueError:
            continue
    return result


def _long_branch_outliers(lengths: Dict[str, float]) -> List[str]:
    values = sorted(value for value in lengths.values() if value >= 0)
    if len(values) < 3:
        return []
    median = values[len(values) // 2]
    threshold = max(0.5, median * 3.0)
    return sorted(label for label, value in lengths.items() if value > threshold)


def _stop_codon_status(records: Dict[str, str]) -> Tuple[List[str], List[str]]:
    internal: List[str] = []
    terminal: List[str] = []
    for label, seq in records.items():
        codons = [seq[i:i + 3].upper().replace("U", "T") for i in range(0, len(seq), 3) if len(seq[i:i + 3]) == 3]
        for idx, codon in enumerate(codons[:-1]):
            if codon in STOP_CODONS:
                internal.append(f"{label}:{idx}:{codon}")
        if codons and codons[-1] in STOP_CODONS:
            terminal.append(f"{label}:{codons[-1]}")
    return internal, terminal


def _gap_fraction(records: Dict[str, str]) -> float:
    total = sum(len(seq) for seq in records.values())
    return 0.0 if total == 0 else sum(seq.count("-") for seq in records.values()) / total


def _ambiguous_fraction(records: Dict[str, str]) -> float:
    total = sum(len(seq) for seq in records.values())
    if total == 0:
        return 0.0
    return sum(1 for seq in records.values() for char in seq.upper() if char not in {"A", "C", "G", "T", "U", "-"}) / total


def _duplicate_species_labels(records: Dict[str, str]) -> List[str]:
    species = [label.split("|")[0].split(".")[0] for label in records]
    return sorted(name for name, count in Counter(species).items() if count > 1)


def _saturation_proxy(mean_p: float) -> str:
    if mean_p < 0.05:
        return "low"
    if mean_p < 0.12:
        return "moderate"
    if mean_p < 0.25:
        return "high"
    return "extreme"


def _prefilter_decision(
    *,
    n_taxa: int,
    min_codons: int,
    mean_p: float,
    max_p: float,
    tree_ok: bool,
    paralogy_flags: List[str],
    max_mean_pdistance: float,
    min_taxa: int,
    required_min_codons: int,
) -> Tuple[str, str]:
    if not tree_ok:
        return "reject_tree_mismatch", "repair FASTA/tree tips and foreground before scoring"
    if n_taxa < min_taxa:
        return "reject_too_few_taxa", "add closer orthologs until the minimum taxon count is met"
    if min_codons < required_min_codons:
        return "reject_too_short", "choose a longer coding region or another family"
    severe_paralogy = [flag for flag in paralogy_flags if flag.startswith(("duplicate", "extreme_length_ratio"))]
    if severe_paralogy:
        return "reject_possible_paralogy", "curate one ortholog per species and remove obvious paralogs"
    if mean_p > 0.50 or max_p > 0.75:
        return "diagnostic_only", "use this only as an OOD stress test; build a closer-taxa panel"
    if mean_p > max_mean_pdistance:
        return "reject_too_divergent", "reduce taxonomic breadth until mean p-distance is below the configured gate"
    if mean_p > 0.25 or paralogy_flags:
        return "accept_with_caution", "run as a guarded pilot and compare reference workflows"
    return "accept", "eligible for guarded empirical pilot after reference planning"


def _prefilter_fields() -> List[str]:
    return [
        "decision", "n_taxa", "n_codons", "mean_pairwise_p_distance", "max_pairwise_p_distance",
        "saturation_proxy", "gap_fraction", "ambiguous_fraction", "foreground_present",
        "tree_tip_compatible", "paralogy_risk_flags", "recommended_action",
    ]


def _prefilter_tsv_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "decision": payload["decision"],
        "n_taxa": payload["n_taxa"],
        "n_codons": payload["n_codons"],
        "mean_pairwise_p_distance": f"{payload['mean_pairwise_p_distance']:.6g}",
        "max_pairwise_p_distance": f"{payload['max_pairwise_p_distance']:.6g}",
        "saturation_proxy": payload["saturation_proxy"],
        "gap_fraction": f"{payload['gap_fraction']:.6g}",
        "ambiguous_fraction": f"{payload['ambiguous_fraction']:.6g}",
        "foreground_present": payload["foreground_present"],
        "tree_tip_compatible": payload["tree_tip_compatible"],
        "paralogy_risk_flags": ";".join(payload["paralogy_risk_flags"]),
        "recommended_action": payload["recommended_action"],
    }


def _render_prefilter_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# Empirical Family Prefilter",
        "",
        f"- decision: `{payload['decision']}`",
        f"- n_taxa: `{payload['n_taxa']}`",
        f"- n_codons: `{payload['n_codons']}`",
        f"- mean p-distance: `{payload['mean_pairwise_p_distance']:.6g}`",
        f"- max p-distance: `{payload['max_pairwise_p_distance']:.6g}`",
        f"- saturation proxy: `{payload['saturation_proxy']}`",
        f"- recommended action: {payload['recommended_action']}",
        "",
        "No empirical positive-selection claim is made by this prefilter.",
        "",
    ]
    return "\n".join(lines)


def _read_tsv_optional(path: Path) -> List[Dict[str, str]]:
    return read_tsv(path) if path.exists() else []


def _target_taxa_rows(pilot_type: str) -> Tuple[List[Dict[str, str]], str]:
    templates = {
        "plant_close": ([
            ("Arabidopsis_thaliana", "Ensembl Plants Arabidopsis_thaliana", "query_or_close", "query species / close Brassicaceae anchor"),
            ("Arabidopsis_lyrata", "Ensembl Plants Arabidopsis_lyrata", "close", "close Arabidopsis relative"),
            ("Capsella_rubella", "Ensembl Plants Capsella_rubella", "close", "Brassicaceae close panel"),
            ("Brassica_rapa", "Ensembl Plants Brassica_rapa", "close_moderate", "Brassicaceae crop relative"),
            ("Brassica_oleracea", "Ensembl Plants Brassica_oleracea", "close_moderate", "Brassicaceae crop relative"),
            ("Eutrema_salsugineum", "Ensembl Plants Eutrema_salsugineum", "close_moderate", "Brassicaceae stress-tolerant relative"),
        ], "Use a Brassicaceae-heavy panel; avoid mixing monocots and legumes in the first WRKY pilot."),
        "plant_moderate": ([
            ("Arabidopsis_thaliana", "Ensembl Plants", "query", "anchor"),
            ("Brassica_rapa", "Ensembl Plants", "moderate", "Brassicaceae"),
            ("Glycine_max", "Ensembl Plants", "moderate", "legume"),
            ("Medicago_truncatula", "Ensembl Plants", "moderate", "legume"),
            ("Vitis_vinifera", "Ensembl Plants", "moderate", "rosid"),
            ("Solanum_lycopersicum", "Ensembl Plants", "moderate", "asterid"),
        ], "Moderate plant panel; prefilter before scoring because saturation can rise quickly."),
        "monocot_close": ([
            ("Oryza_sativa", "Ensembl Plants", "query_or_close", "rice anchor"),
            ("Oryza_barthii", "Ensembl Plants", "close", "close Oryza"),
            ("Oryza_glaberrima", "Ensembl Plants", "close", "close Oryza"),
            ("Brachypodium_distachyon", "Ensembl Plants", "moderate", "grass"),
            ("Sorghum_bicolor", "Ensembl Plants", "moderate", "grass"),
            ("Zea_mays", "Ensembl Plants", "moderate", "grass"),
        ], "Use monocots together rather than mixing deep dicot/monocot panels."),
        "metazoan_close": ([
            ("Drosophila_melanogaster", "Ensembl Metazoa", "query_or_close", "future animal pilot"),
            ("Drosophila_simulans", "Ensembl Metazoa", "close", "close Drosophila"),
            ("Drosophila_sechellia", "Ensembl Metazoa", "close", "close Drosophila"),
            ("Drosophila_yakuba", "Ensembl Metazoa", "moderate", "moderate Drosophila"),
            ("Drosophila_erecta", "Ensembl Metazoa", "moderate", "moderate Drosophila"),
            ("Drosophila_ananassae", "Ensembl Metazoa", "caution", "prefilter required"),
        ], "For animal immune/detox pilots, start with close species and avoid deep mixtures."),
        "custom": ([], "Custom target taxa file requested; populate manually."),
    }
    tuples, note = templates.get(pilot_type, templates["custom"])
    return [
        {"taxon_label": taxon, "ensembl_dir_or_source_hint": source, "category": category, "notes": notes}
        for taxon, source, category, notes in tuples
    ], note


def _script_header() -> List[str]:
    return ["#!/usr/bin/env bash", "set -euo pipefail", f"echo '{USER_RUN_ONLY}'", ""]


def _script_download(config: EmpiricalFamilyAcquisitionPlanConfig) -> str:
    lines = _script_header()
    lines.extend([
        f"# Download proteome/CDS resources for {config.family_id} from {config.source}.",
        f"# Target taxa file: {config.target_taxa_file}",
        "# Fill exact Ensembl/NCBI source URLs before running.",
        "",
    ])
    return "\n".join(lines)


def _script_blast(config: EmpiricalFamilyAcquisitionPlanConfig) -> str:
    lines = _script_header()
    lines.extend([
        f"# Run BLASTP best-hit search for {config.query_species} {config.query_gene_or_locus}.",
        "# makeblastdb -in proteome.fasta -dbtype prot",
        "# blastp -query query.fasta -db proteome.fasta -outfmt 6 -max_target_seqs 5 > best_hits.tsv",
        "",
    ])
    return "\n".join(lines)


def _script_recover(config: EmpiricalFamilyAcquisitionPlanConfig) -> str:
    lines = _script_header()
    lines.extend([
        "# Recover matching CDS records from best protein hits.",
        "# Write curated candidate CDS to candidate.cds.fasta after manual review.",
        "",
    ])
    return "\n".join(lines)


def _script_tree(config: EmpiricalFamilyAcquisitionPlanConfig) -> str:
    lines = _script_header()
    lines.extend([
        "# Align proteins/CDS and infer an ML tree.",
        "# mafft --auto candidate.protein.fasta > candidate.protein.aln.fasta",
        "# iqtree -s candidate.protein.aln.fasta -m MFP -bb 1000 -nt AUTO -pre tree",
        "",
    ])
    return "\n".join(lines)


def _script_prefilter(config: EmpiricalFamilyAcquisitionPlanConfig) -> str:
    lines = _script_header()
    lines.extend([
        "babappa prefilter-empirical-family \\",
        "  --cds-fasta candidate.cds.fasta \\",
        "  --tree-file tree.treefile \\",
        f"  --foreground {config.query_species} \\",
        f"  --outdir prefilter/{config.family_id}",
        "",
    ])
    return "\n".join(lines)


def _expected_acquisition_outputs(config: EmpiricalFamilyAcquisitionPlanConfig) -> Dict[str, Any]:
    return {
        "raw_downloads": "proteome/CDS FASTA files",
        "best_hits": "best_hits.tsv",
        "curated_cds": f"{config.family_id}.cds.fasta",
        "tree": f"{config.family_id}.treefile",
        "prefilter": "empirical_family_prefilter.json",
    }


def _render_acquisition_plan_md(payload: Dict[str, Any]) -> str:
    return "\n".join([
        "# Empirical Family Acquisition Plan",
        "",
        USER_RUN_ONLY,
        "",
        f"- family: `{payload['family_id']}`",
        f"- query: `{payload['query_species']} {payload['query_gene_or_locus']}`",
        f"- strategy: `{payload['strategy']}`",
        f"- target taxa: `{payload['n_target_taxa']}`",
        "- executed: `False`",
        "",
    ])


def _render_recommendation_md(payload: Dict[str, Any], rows: List[Dict[str, str]]) -> str:
    lines = ["# Target Taxa Recommendation", "", f"- pilot type: `{payload['pilot_type']}`", f"- recommendation: {payload['recommendation']}", ""]
    for row in rows:
        lines.append(f"- {row['taxon_label']}: {row['category']} ({row['notes']})")
    lines.append("")
    return "\n".join(lines)


def _render_ood_build_script(config: OODAwareFamilyBuildPlanConfig) -> str:
    lines = _script_header()
    lines.extend([
        f"# OOD-aware family build for {config.family_id}.",
        "# Steps: download -> BLASTP best hit -> recover CDS -> sanitize -> align -> tree -> prefilter -> gated import.",
        f"MAX_MEAN_PDISTANCE={config.max_mean_pdistance}",
        f"MIN_TAXA={config.min_taxa}",
        f"MIN_CODONS={config.min_codons}",
        f"FAMILY_ID={config.family_id}",
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        "WORKSPACE_DIR=\"${WORKSPACE_DIR:-$(cd \"${SCRIPT_DIR}/../..\" && pwd)}\"",
        "cd \"${SCRIPT_DIR}\"",
        "PREFILTER_DIR=\"${WORKSPACE_DIR}/prefilter/${FAMILY_ID}\"",
        "",
        "echo \"Fill the acquisition scripts with exact data sources before running this workflow.\"",
        "bash download_ensembl_proteome_cds.sh",
        "bash run_blastp_best_hit.sh",
        "bash recover_cds_from_best_hits.sh",
        "bash build_tree.sh",
        "babappa prefilter-empirical-family \\",
        "  --cds-fasta candidate.cds.fasta \\",
        "  --tree-file tree.treefile \\",
        f"  --foreground {config.query_species} \\",
        "  --outdir \"${PREFILTER_DIR}\" \\",
        "  --max-mean-pdistance \"${MAX_MEAN_PDISTANCE}\" \\",
        "  --min-taxa \"${MIN_TAXA}\" \\",
        "  --min-codons \"${MIN_CODONS}\"",
        "decision=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"decision\"])' \"${PREFILTER_DIR}/empirical_family_prefilter.json\")",
        "case \"${decision}\" in",
        "  accept|accept_with_caution)",
        "    babappa add-prefiltered-family-to-pilot \\",
        "      --workspace \"${WORKSPACE_DIR}\" \\",
        "      --prefilter-dir \"${PREFILTER_DIR}\" \\",
        "      --panel-id \"${FAMILY_ID}\" \\",
        "      --expected-category likely_positive \\",
        "      --reference-status planned",
        "    ;;",
        "  *)",
        "    echo \"Not importing ${FAMILY_ID}; prefilter decision was ${decision}.\"",
        "    exit 2",
        "    ;;",
        "esac",
        "",
    ])
    return "\n".join(lines)


def _render_ood_build_md(payload: Dict[str, Any]) -> str:
    return "\n".join([
        "# OOD-Aware Family Build Plan",
        "",
        USER_RUN_ONLY,
        "",
        f"- family: `{payload['family_id']}`",
        f"- max mean p-distance: `{payload['max_mean_pdistance']}`",
        f"- min taxa: `{payload['min_taxa']}`",
        f"- min codons: `{payload['min_codons']}`",
        f"- import gate: `{payload['import_gate']}`",
        "",
    ])


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root is not object: {path}")
    return data


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_to_manifest(manifest: Path, path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), start=manifest.parent.resolve())
    except OSError:
        return str(path)


def _upsert_manifest_row(manifest: Path, row: Dict[str, str]) -> None:
    rows = read_tsv(manifest) if manifest.exists() else []
    updated = False
    out: List[Dict[str, str]] = []
    for existing in rows:
        if existing.get("panel_id") == row["panel_id"]:
            out.append(row)
            updated = True
        else:
            out.append({field: existing.get(field, "") for field in PILOT_MANIFEST_FIELDS})
    if not updated:
        out.append(row)
    write_tsv(manifest, out, PILOT_MANIFEST_FIELDS)


def _render_add_prefilter_md(report: Dict[str, Any]) -> str:
    return "\n".join([
        "# Add Prefiltered Family",
        "",
        f"- status: `{report['status']}`",
        f"- panel_id: `{report['panel_id']}`",
        f"- decision: `{report['decision']}`",
        f"- reason: `{report.get('reason', '')}`",
        "",
    ])


def _ood_recommended_action(row: Dict[str, Any]) -> str:
    if str(row.get("diagnostic_only", "")).lower() == "true" or row.get("applicability") == "out_of_domain":
        return "use as OOD stress test only; build closer-taxa candidate"
    if row.get("prefilter_decision") in {"accept", "accept_with_caution"}:
        return "eligible for guarded pilot and reference comparison"
    return "run prefilter before empirical interpretation"


def _render_ood_summary_md(payload: Dict[str, Any]) -> str:
    lines = ["# Empirical OOD Summary", "", f"- families: `{payload['n_families']}`", "", "No empirical positive-selection discovery claim is made.", ""]
    for row in payload["rows"]:
        lines.append(f"- {row.get('family')}: prefilter={row.get('prefilter_decision', '')}, applicability={row.get('applicability', '')}, action={row.get('recommended_action', '')}")
    lines.append("")
    return "\n".join(lines)
