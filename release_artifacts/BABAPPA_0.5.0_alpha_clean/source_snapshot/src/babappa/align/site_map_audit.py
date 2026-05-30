"""Validation for BABAPPA alignment site-map outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

REQUIRED_MAP_COLUMNS = {
    "family_id",
    "method",
    "aligned_site_index_zero",
    "original_site_index_zero",
    "mapping_status",
    "mapping_confidence",
}


def validate_site_map_dir(
    site_map_dir: str | Path,
    methods: Optional[List[str]] = None,
    max_conflict_fraction: Optional[float] = None,
    quarantine_methods: Optional[List[str]] = None,
) -> dict:
    """Validate an alignment site-map directory."""
    path = Path(site_map_dir)
    failures: List[str] = []
    warnings: List[str] = []
    selected_methods = set(methods or [])
    quarantined = set(quarantine_methods or [])
    manifest_path = path / "site_map_manifest.json"
    summary_path = path / "site_map_summary.tsv"
    method_summary_path = path / "site_map_method_summary.tsv"
    markdown_path = path / "site_map_report.md"
    manifest = _load_json(manifest_path, failures)
    if not summary_path.exists():
        failures.append(f"missing_file:{summary_path}")
    elif not _has_header(summary_path):
        failures.append(f"missing_header:{summary_path}")
    summary_rows = _read_tsv(summary_path, failures) if summary_path.exists() else []
    method_summary_rows = (
        _read_tsv(method_summary_path, failures)
        if method_summary_path.exists()
        else _method_summary_from_family_rows(summary_rows)
    )
    if not method_summary_path.exists():
        warnings.append("method_summary_missing_recomputed_from_summary")
    if not markdown_path.exists():
        failures.append(f"missing_file:{markdown_path}")
    elif not markdown_path.read_text(encoding="utf-8").strip():
        failures.append(f"empty_markdown:{markdown_path}")

    map_files = manifest.get("map_files", {}) if isinstance(manifest, dict) else {}
    n_maps = 0
    non_unique_rows = 0
    if isinstance(map_files, dict):
        for family_maps in map_files.values():
            if not isinstance(family_maps, dict):
                continue
            for method, rel_path in family_maps.items():
                if selected_methods and method not in selected_methods and method not in quarantined:
                    continue
                n_maps += 1
                map_path = path / str(rel_path)
                rows = _read_tsv(map_path, failures)
                if rows:
                    missing = sorted(REQUIRED_MAP_COLUMNS - set(rows[0].keys()))
                    if missing:
                        failures.append(f"missing_required_columns:{map_path}:{','.join(missing)}")
                for row in rows:
                    confidence = _safe_float(row.get("mapping_confidence"), default=-1.0)
                    if confidence < 0.0 or confidence > 1.0:
                        failures.append(f"mapping_confidence_out_of_range:{map_path}:{confidence}")
                    original = row.get("original_site_index_zero", "")
                    if original not in ("", None) and _safe_int(original, -1) < 0:
                        failures.append(f"negative_original_site_index:{map_path}:{original}")
                    if row.get("mapping_status") != "unique":
                        non_unique_rows += 1
    else:
        failures.append("manifest_map_files_not_object")

    method_summary = _method_summary_lookup(method_summary_rows)
    selected_for_qc = selected_methods or set(method_summary)
    for method in sorted(selected_for_qc):
        if method in quarantined:
            warnings.append(f"method_quarantined:{method}")
            continue
        if method not in method_summary:
            failures.append(f"selected_method_missing:{method}")
            continue
        row = method_summary[method]
        conflict_fraction = _safe_float(row.get("conflict_fraction"), default=0.0)
        frame_error_fraction = _safe_float(row.get("frame_error_fraction"), default=0.0)
        if frame_error_fraction > 0:
            failures.append(f"method_frame_error_fraction_above_0:{method}:{frame_error_fraction}")
        if max_conflict_fraction is None:
            if conflict_fraction > 0.01:
                warnings.append(f"method_conflict_fraction_above_0_01:{method}:{conflict_fraction}")
        elif conflict_fraction > max_conflict_fraction:
            failures.append(
                f"method_conflict_fraction_above_max:{method}:{conflict_fraction}>{max_conflict_fraction}"
            )
    for method in sorted(quarantined):
        if method in method_summary:
            warnings.append(f"quarantined_method_exempted:{method}")
    conflict_fraction = _safe_float(manifest.get("conflict_fraction"), default=0.0)
    frame_error_fraction = _safe_float(manifest.get("frame_error_fraction"), default=0.0)
    if not selected_methods and max_conflict_fraction is None and conflict_fraction > 0.01:
        warnings.append("conflict_fraction_above_0_01")
    if not selected_methods and frame_error_fraction > 0:
        warnings.append("frame_error_fraction_above_0")
    if manifest.get("require_complete") and non_unique_rows:
        failures.append("require_complete_non_unique_mappings_present")
    if n_maps == 0:
        failures.append("no_map_tsv_files")

    return {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "n_maps": n_maps,
        "methods_checked": sorted(selected_for_qc),
        "quarantine_methods": sorted(quarantined),
        "failures": failures,
        "warnings": warnings,
    }


def _load_json(path: Path, failures: List[str]) -> dict:
    if not path.exists():
        failures.append(f"missing_file:{path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"could_not_parse_json:{path}:{exc}")
        return {}
    if not isinstance(payload, dict):
        failures.append(f"json_not_object:{path}")
        return {}
    return payload


def _read_tsv(path: Path, failures: List[str]) -> List[dict]:
    if not path.exists():
        failures.append(f"missing_file:{path}")
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    except OSError as exc:
        failures.append(f"could_not_read_tsv:{path}:{exc}")
        return []


def _has_header(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return bool(handle.readline().strip())
    except OSError:
        return False


def _method_summary_lookup(rows: List[dict]) -> Dict[str, dict]:
    return {str(row.get("method", "")): row for row in rows if row.get("method")}


def _method_summary_from_family_rows(rows: List[dict]) -> List[dict]:
    totals: Dict[str, dict] = {}
    for row in rows:
        method = row.get("method", "")
        if not method:
            continue
        stats = totals.setdefault(
            method,
            {
                "method": method,
                "n_family_method_maps": 0,
                "total_aligned_sites": 0,
                "unique_sites": 0.0,
                "conflict_sites": 0.0,
                "all_gap_sites": 0.0,
                "frame_error_sites": 0.0,
            },
        )
        n_sites = _safe_int(row.get("n_aligned_sites"), 0)
        stats["n_family_method_maps"] += 1
        stats["total_aligned_sites"] += n_sites
        stats["unique_sites"] += _safe_float(row.get("unique_fraction"), 0.0) * n_sites
        stats["conflict_sites"] += _safe_float(row.get("conflict_fraction"), 0.0) * n_sites
        stats["all_gap_sites"] += _safe_float(row.get("all_gap_fraction"), 0.0) * n_sites
        stats["frame_error_sites"] += _safe_float(row.get("frame_error_fraction"), 0.0) * n_sites
    summary = []
    for method, stats in sorted(totals.items()):
        total = int(stats["total_aligned_sites"])
        conflict_fraction = _fraction(stats["conflict_sites"], total)
        frame_error_fraction = _fraction(stats["frame_error_sites"], total)
        summary.append(
            {
                "method": method,
                "n_family_method_maps": stats["n_family_method_maps"],
                "total_aligned_sites": total,
                "unique_fraction": _fraction(stats["unique_sites"], total),
                "conflict_fraction": conflict_fraction,
                "all_gap_fraction": _fraction(stats["all_gap_sites"], total),
                "frame_error_fraction": frame_error_fraction,
                "recommendation": _recommend_method(method, conflict_fraction, frame_error_fraction),
            }
        )
    return summary


def _fraction(count: float, total: int) -> float:
    return 0.0 if total <= 0 else float(count) / total


def _recommend_method(method: str, conflict_fraction: float, frame_error_fraction: float) -> str:
    if method == "codon_dropout" and (conflict_fraction > 0.10 or frame_error_fraction > 0):
        return "quarantine_unmappable_noise_control"
    if frame_error_fraction > 0 or conflict_fraction > 0.10:
        return "quarantine"
    if conflict_fraction > 0.03:
        return "caution"
    return "usable"


def _safe_float(value: object, default: float) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default
