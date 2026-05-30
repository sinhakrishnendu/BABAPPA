"""Map aligned codon columns back to original simulated codon sites."""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from babappa import __version__
from babappa.datasets.index import write_tsv
from babappa.simulate.audit import read_fasta

SITE_MAP_VERSION = __version__
SITE_MAP_FIELDNAMES = [
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
SITE_MAP_SUMMARY_FIELDNAMES = [
    "family_id",
    "method",
    "n_aligned_sites",
    "unique_fraction",
    "conflict_fraction",
    "all_gap_fraction",
    "frame_error_fraction",
    "map_tsv",
]
SITE_MAP_METHOD_SUMMARY_FIELDNAMES = [
    "method",
    "n_family_method_maps",
    "total_aligned_sites",
    "unique_fraction",
    "conflict_fraction",
    "all_gap_fraction",
    "frame_error_fraction",
    "mean_mapping_confidence",
    "recommendation",
]


@dataclass(frozen=True)
class SiteMapConfig:
    """Configuration for building alignment-to-original site maps."""

    sim_dir: str
    align_dir: str
    outdir: Optional[str] = None
    methods: Optional[List[str]] = None
    require_complete: bool = False
    workers: int = 1

    def __post_init__(self) -> None:
        sim_path = Path(self.sim_dir)
        align_path = Path(self.align_dir)
        if not sim_path.exists():
            raise ValueError(f"sim_dir does not exist: {sim_path}")
        if not (sim_path / "manifest.json").exists():
            raise ValueError(f"sim_dir is missing manifest.json: {sim_path}")
        if not align_path.exists():
            raise ValueError(f"align_dir does not exist: {align_path}")
        if not (align_path / "alignment_manifest.json").exists():
            raise ValueError(f"align_dir is missing alignment_manifest.json: {align_path}")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        out_path = Path(self.outdir) if self.outdir else align_path / "site_maps"
        out_path.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "outdir", str(out_path))


def build_site_map_for_alignment(
    original_fasta: str | Path,
    aligned_fasta: str | Path,
    family_id: str = "",
    method: str = "",
) -> List[dict]:
    """Build a consensus aligned-site to original-site codon map."""
    original_records = read_fasta(Path(original_fasta))
    aligned_records = read_fasta(Path(aligned_fasta))
    if not aligned_records:
        return []
    lengths = [len(sequence) for sequence in aligned_records.values()]
    n_aligned_codons = max(lengths) // 3 if lengths else 0
    per_taxon_maps: Dict[str, List[Optional[int]]] = {}
    per_taxon_errors: Dict[str, set[int]] = {}

    for taxon, aligned_sequence in aligned_records.items():
        original_sequence = original_records.get(taxon, "")
        n_original_codons = len(original_sequence) // 3
        pointer = 0
        taxon_map: List[Optional[int]] = []
        error_sites: set[int] = set()
        if len(aligned_sequence) % 3 != 0:
            error_sites.update(range(n_aligned_codons))
        for site_index in range(n_aligned_codons):
            codon = aligned_sequence[site_index * 3: site_index * 3 + 3]
            if len(codon) != 3:
                taxon_map.append(None)
                error_sites.add(site_index)
                continue
            if codon == "---":
                taxon_map.append(None)
                continue
            if "-" in codon:
                taxon_map.append(pointer if pointer < n_original_codons else None)
                error_sites.add(site_index)
                if pointer < n_original_codons:
                    pointer += 1
                continue
            if pointer < n_original_codons:
                taxon_map.append(pointer)
                pointer += 1
            else:
                taxon_map.append(None)
                error_sites.add(site_index)
        per_taxon_maps[taxon] = taxon_map
        per_taxon_errors[taxon] = error_sites

    rows: List[dict] = []
    for site_index in range(n_aligned_codons):
        mapped_values = [
            taxon_map[site_index]
            for taxon_map in per_taxon_maps.values()
            if site_index < len(taxon_map) and taxon_map[site_index] is not None
        ]
        n_gap = sum(
            1
            for taxon_map in per_taxon_maps.values()
            if site_index >= len(taxon_map) or taxon_map[site_index] is None
        )
        has_frame_error = any(site_index in errors for errors in per_taxon_errors.values())
        original_site: Optional[int] = None
        n_conflict = 0
        confidence = 0.0
        if has_frame_error:
            status = "frame_error"
        elif not mapped_values:
            status = "all_gap"
        else:
            counts = Counter(mapped_values)
            original_site, support = counts.most_common(1)[0]
            confidence = support / len(mapped_values)
            if len(counts) == 1:
                status = "unique"
                confidence = 1.0
            else:
                status = "conflict"
                n_conflict = len(mapped_values) - support
        rows.append(
            {
                "family_id": family_id,
                "method": method,
                "aligned_site_index_zero": site_index,
                "aligned_site_index_one": site_index + 1,
                "original_site_index_zero": "" if original_site is None else original_site,
                "original_site_index_one": "" if original_site is None else original_site + 1,
                "mapping_status": status,
                "n_taxa_mapped": len(mapped_values),
                "n_taxa_gap": n_gap,
                "n_taxa_conflict": n_conflict,
                "mapping_confidence": confidence,
            }
        )
    return rows


def build_alignment_site_maps(config: SiteMapConfig) -> dict:
    """Build site maps for every family/method in an alignment directory."""
    sim_path = Path(config.sim_dir)
    align_path = Path(config.align_dir)
    outdir = Path(config.outdir or align_path / "site_maps")
    manifest = _read_json(align_path / "alignment_manifest.json")
    family_ids = manifest.get("family_ids")
    if not isinstance(family_ids, list):
        raise ValueError("alignment manifest does not contain a family_ids list")
    methods = list(config.methods or manifest.get("methods") or [])
    if not methods:
        raise ValueError("no alignment methods available for site-map construction")

    warnings: List[str] = []
    summary_rows: List[dict] = []
    map_files: Dict[str, Dict[str, str]] = {}
    aggregate_counts = Counter()
    method_stats: Dict[str, dict] = {}
    total_sites = 0

    tasks = [
        (str(sim_path), str(align_path), str(outdir), str(family_id), str(method))
        for family_id in family_ids
        for method in methods
    ]
    if config.workers <= 1 or len(tasks) <= 1:
        results = [_build_site_map_task(task) for task in tasks]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=min(config.workers, len(tasks))) as executor:
            futures = {executor.submit(_build_site_map_task, task): task for task in tasks}
            for future in as_completed(futures):
                results.append(future.result())

    for result in sorted(results, key=lambda item: (item["family_id"], item["method"])):
        family_id = result["family_id"]
        method = result["method"]
        map_files.setdefault(family_id, {})
        if result["status"] != "ok":
            warnings.extend(result["warnings"])
            continue
        map_files[family_id][method] = result["map_tsv"]
        status_counts = Counter(result["status_counts"])
        n_sites = int(result["n_sites"])
        total_sites += n_sites
        aggregate_counts.update(status_counts)
        _update_method_stats_from_summary(
            method_stats,
            method,
            status_counts,
            n_sites,
            float(result["confidence_sum"]),
            int(result["confidence_n"]),
        )
        summary_rows.append(result["summary_row"])

    summary_path = outdir / "site_map_summary.tsv"
    method_summary_path = outdir / "site_map_method_summary.tsv"
    manifest_path = outdir / "site_map_manifest.json"
    report_path = outdir / "site_map_report.md"
    write_tsv(summary_path, summary_rows, SITE_MAP_SUMMARY_FIELDNAMES)
    method_summary_rows = _method_summary_rows(method_stats)
    write_tsv(method_summary_path, method_summary_rows, SITE_MAP_METHOD_SUMMARY_FIELDNAMES)
    conflict_dominance = _conflict_dominance(method_summary_rows)
    payload = {
        "site_map_version": SITE_MAP_VERSION,
        "sim_dir": str(sim_path),
        "align_dir": str(align_path),
        "methods": methods,
        "require_complete": config.require_complete,
        "workers": config.workers,
        "n_family_method_maps": len(summary_rows),
        "total_aligned_sites": total_sites,
        "unique_fraction": _fraction(aggregate_counts["unique"], total_sites),
        "conflict_fraction": _fraction(aggregate_counts["conflict"], total_sites),
        "all_gap_fraction": _fraction(aggregate_counts["all_gap"], total_sites),
        "frame_error_fraction": _fraction(aggregate_counts["frame_error"], total_sites),
        "method_summary": method_summary_rows,
        "usable_methods": [
            row["method"]
            for row in method_summary_rows
            if str(row["recommendation"]).startswith("usable")
        ],
        "quarantined_methods": [
            row["method"]
            for row in method_summary_rows
            if "quarantine" in str(row["recommendation"])
        ],
        "dominant_conflict_method": conflict_dominance.get("method"),
        "dominant_conflict_fraction": conflict_dominance.get("fraction"),
        "map_files": map_files,
        "warnings": sorted(set(warnings)),
        "generated_files": {
            "manifest": str(manifest_path),
            "summary": str(summary_path),
            "method_summary": str(method_summary_path),
            "markdown": str(report_path),
        },
    }
    _write_json(manifest_path, payload)
    report_path.write_text(_render_report(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "manifest": str(manifest_path),
        "summary": str(summary_path),
        "method_summary": str(method_summary_path),
        "markdown": str(report_path),
        "n_family_method_maps": len(summary_rows),
        "unique_fraction": payload["unique_fraction"],
        "conflict_fraction": payload["conflict_fraction"],
        "frame_error_fraction": payload["frame_error_fraction"],
        "warnings": payload["warnings"],
    }


def _build_site_map_task(task: tuple[str, str, str, str, str]) -> dict:
    sim_path_s, align_path_s, outdir_s, family_id, method = task
    sim_path = Path(sim_path_s)
    align_path = Path(align_path_s)
    outdir = Path(outdir_s)
    original_fasta = sim_path / "families" / family_id / f"{family_id}.fasta"
    aligned_fasta = align_path / "families" / family_id / f"{family_id}.{method}.codon.fasta"
    if not aligned_fasta.exists():
        return {
            "status": "missing",
            "family_id": family_id,
            "method": method,
            "warnings": [f"missing_aligned_fasta:{family_id}:{method}"],
        }
    rows = build_site_map_for_alignment(
        original_fasta=original_fasta,
        aligned_fasta=aligned_fasta,
        family_id=family_id,
        method=method,
    )
    family_outdir = outdir / "families" / family_id
    family_outdir.mkdir(parents=True, exist_ok=True)
    map_path = family_outdir / f"{family_id}.{method}.site_map.tsv"
    write_tsv(map_path, rows, SITE_MAP_FIELDNAMES)
    status_counts = Counter(row["mapping_status"] for row in rows)
    n_sites = len(rows)
    confidence_sum = sum(float(row.get("mapping_confidence") or 0.0) for row in rows)
    confidence_n = len(rows)
    return {
        "status": "ok",
        "family_id": family_id,
        "method": method,
        "map_tsv": str(map_path.relative_to(outdir)),
        "status_counts": dict(status_counts),
        "n_sites": n_sites,
        "confidence_sum": confidence_sum,
        "confidence_n": confidence_n,
        "warnings": [],
        "summary_row": {
            "family_id": family_id,
            "method": method,
            "n_aligned_sites": n_sites,
            "unique_fraction": _format_fraction(_fraction(status_counts["unique"], n_sites)),
            "conflict_fraction": _format_fraction(_fraction(status_counts["conflict"], n_sites)),
            "all_gap_fraction": _format_fraction(_fraction(status_counts["all_gap"], n_sites)),
            "frame_error_fraction": _format_fraction(_fraction(status_counts["frame_error"], n_sites)),
            "map_tsv": str(map_path.relative_to(outdir)),
        },
    }


def _fraction(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


def _update_method_stats(
    method_stats: Dict[str, dict],
    method: str,
    rows: List[dict],
    status_counts: Counter,
    n_sites: int,
) -> None:
    stats = method_stats.setdefault(
        method,
        {
            "n_family_method_maps": 0,
            "total_aligned_sites": 0,
            "status_counts": Counter(),
            "confidence_sum": 0.0,
            "confidence_n": 0,
        },
    )
    stats["n_family_method_maps"] += 1
    stats["total_aligned_sites"] += n_sites
    stats["status_counts"].update(status_counts)
    for row in rows:
        stats["confidence_sum"] += float(row.get("mapping_confidence") or 0.0)
        stats["confidence_n"] += 1


def _update_method_stats_from_summary(
    method_stats: Dict[str, dict],
    method: str,
    status_counts: Counter,
    n_sites: int,
    confidence_sum: float,
    confidence_n: int,
) -> None:
    stats = method_stats.setdefault(
        method,
        {
            "n_family_method_maps": 0,
            "total_aligned_sites": 0,
            "status_counts": Counter(),
            "confidence_sum": 0.0,
            "confidence_n": 0,
        },
    )
    stats["n_family_method_maps"] += 1
    stats["total_aligned_sites"] += n_sites
    stats["status_counts"].update(status_counts)
    stats["confidence_sum"] += confidence_sum
    stats["confidence_n"] += confidence_n


def _method_summary_rows(method_stats: Dict[str, dict]) -> List[dict]:
    rows: List[dict] = []
    for method in sorted(method_stats):
        stats = method_stats[method]
        total = int(stats["total_aligned_sites"])
        counts = stats["status_counts"]
        conflict_fraction = _fraction(counts["conflict"], total)
        frame_error_fraction = _fraction(counts["frame_error"], total)
        row = {
            "method": method,
            "n_family_method_maps": int(stats["n_family_method_maps"]),
            "total_aligned_sites": total,
            "unique_fraction": _format_fraction(_fraction(counts["unique"], total)),
            "conflict_fraction": _format_fraction(conflict_fraction),
            "all_gap_fraction": _format_fraction(_fraction(counts["all_gap"], total)),
            "frame_error_fraction": _format_fraction(frame_error_fraction),
            "mean_mapping_confidence": _format_fraction(
                _fraction(stats["confidence_sum"], int(stats["confidence_n"]))
            ),
            "recommendation": _recommend_method(method, conflict_fraction, frame_error_fraction),
        }
        rows.append(row)
    return rows


def _recommend_method(method: str, conflict_fraction: float, frame_error_fraction: float) -> str:
    if method == "codon_dropout" and (conflict_fraction > 0.10 or frame_error_fraction > 0):
        return "quarantine_unmappable_noise_control"
    if frame_error_fraction > 0 or conflict_fraction > 0.10:
        return "quarantine"
    if conflict_fraction > 0.03:
        return "caution"
    return "usable"


def _conflict_dominance(method_summary_rows: List[dict]) -> dict:
    if not method_summary_rows:
        return {"method": None, "fraction": None}
    totals = []
    total_conflicts = 0.0
    for row in method_summary_rows:
        conflicts = float(row["conflict_fraction"]) * int(row["total_aligned_sites"])
        total_conflicts += conflicts
        totals.append((str(row["method"]), conflicts))
    if total_conflicts <= 0:
        return {"method": None, "fraction": 0.0}
    method, conflicts = max(totals, key=lambda item: item[1])
    return {"method": method, "fraction": conflicts / total_conflicts}


def _format_fraction(value: float) -> str:
    return f"{float(value):.12g}"


def _render_report(payload: dict) -> str:
    lines = [
        "# BABAPPA alignment site-map report",
        "",
        "## Inputs",
        "",
        f"- Simulation directory: `{payload.get('sim_dir')}`",
        f"- Alignment directory: `{payload.get('align_dir')}`",
        f"- Methods: {', '.join(payload.get('methods') or [])}",
        "",
        "## Mapping summary",
        "",
        f"- Family-method maps: {payload.get('n_family_method_maps')}",
        f"- Total aligned sites: {payload.get('total_aligned_sites')}",
        f"- Unique fraction: {payload.get('unique_fraction')}",
        f"- Conflict fraction: {payload.get('conflict_fraction')}",
        f"- All-gap fraction: {payload.get('all_gap_fraction')}",
        f"- Frame-error fraction: {payload.get('frame_error_fraction')}",
        f"- Usable methods: {', '.join(payload.get('usable_methods') or []) or 'none'}",
        f"- Quarantined methods: {', '.join(payload.get('quarantined_methods') or []) or 'none'}",
        f"- Dominant conflict method: {payload.get('dominant_conflict_method') or 'none'}",
        "",
        "## Per-method QC",
        "",
    ]
    method_summary = payload.get("method_summary") or []
    if method_summary:
        for row in method_summary:
            lines.append(
                "- {method}: unique={unique_fraction}, conflict={conflict_fraction}, "
                "frame_error={frame_error_fraction}, recommendation={recommendation}".format(
                    **row
                )
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Quarantine guidance",
            "",
            "Methods marked usable can be used for mapped oracle-label training.",
            "Methods marked quarantine should be excluded from mapped oracle-label training.",
            "codon_dropout is a taxon-specific dropout noise control; high consensus conflicts classify it as unmappable for mapped site labels.",
            "",
            "## Warnings",
            "",
        ]
    )
    warnings = payload.get("warnings") or []
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- none"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Oracle site labels must be assigned through this map when external aligners insert or move codon columns.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
