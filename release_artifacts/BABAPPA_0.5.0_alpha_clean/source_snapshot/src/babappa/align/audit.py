"""Validation utilities for BABAPPA internal alignment outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

from babappa.simulate.audit import read_fasta

MAP_HEADER = ["alignment_column_0based", "homology_id", "note"]


def validate_alignment_directory(align_dir: Union[str, Path]) -> dict:
    """Validate a BABAPPA alignment scaffold directory."""
    align_path = Path(align_dir)
    warnings: List[str] = []
    failures: List[str] = []
    manifest_path = align_path / "alignment_manifest.json"

    if not manifest_path.exists():
        return _summary(
            status="fail",
            n_families_expected=0,
            n_families_checked=0,
            n_fail=1,
            n_warning=0,
            methods=[],
            warnings=warnings,
            failures=[f"missing alignment_manifest.json: {manifest_path}"],
        )

    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _summary(
            status="fail",
            n_families_expected=0,
            n_families_checked=0,
            n_fail=1,
            n_warning=0,
            methods=[],
            warnings=warnings,
            failures=[f"unreadable alignment_manifest.json: {exc}"],
        )

    family_ids = manifest.get("family_ids", [])
    methods = manifest.get("methods", [])
    created_files = manifest.get("created_files", {})
    if not isinstance(family_ids, list):
        failures.append("alignment_manifest.json family_ids is not a list")
        family_ids = []
    if not isinstance(methods, list):
        failures.append("alignment_manifest.json methods is not a list")
        methods = []
    if created_files and not isinstance(created_files, dict):
        failures.append("alignment_manifest.json created_files is not an object")
        created_files = {}

    family_fail_count = 0
    family_warning_count = 0
    for family_id in family_ids:
        family_failures_before = len(failures)
        family_warnings_before = len(warnings)
        if not isinstance(family_id, str):
            failures.append("alignment_manifest.json contains a non-string family id")
            family_fail_count += 1
            continue
        family_dir = align_path / "families" / family_id
        if not family_dir.exists():
            failures.append(f"missing family directory: {family_dir}")
            family_fail_count += 1
            continue

        family_created = created_files.get(family_id, {}) if created_files else {}
        if created_files and not isinstance(family_created, dict):
            failures.append(f"alignment_manifest.json created_files entry is not an object: {family_id}")
            family_created = {}
        for method in methods:
            if not isinstance(method, str):
                failures.append("alignment_manifest.json contains a non-string method")
                continue
            if created_files and method not in family_created:
                warnings.append(f"family_method_not_manifested:{family_id}:{method}")
                continue
            _validate_method_outputs(family_dir, family_id, method, warnings, failures)

        if len(failures) > family_failures_before:
            family_fail_count += 1
        elif len(warnings) > family_warnings_before:
            family_warning_count += 1

    status = "fail" if failures else "ok"
    return _summary(
        status=status,
        n_families_expected=manifest.get("n_families", len(family_ids)),
        n_families_checked=len(family_ids),
        n_fail=family_fail_count + (1 if failures and not family_ids else 0),
        n_warning=family_warning_count,
        methods=methods,
        warnings=warnings,
        failures=failures,
    )


def _validate_method_outputs(
    family_dir: Path,
    family_id: str,
    method: str,
    warnings: List[str],
    failures: List[str],
) -> None:
    codon_fasta = family_dir / f"{family_id}.{method}.codon.fasta"
    map_tsv = family_dir / f"{family_id}.{method}.map.tsv"
    qc_json = family_dir / f"{family_id}.{method}.qc.json"

    if not codon_fasta.exists():
        failures.append(f"missing codon FASTA: {codon_fasta}")
    elif codon_fasta.stat().st_size == 0:
        failures.append(f"empty codon FASTA: {codon_fasta}")
    else:
        try:
            records = read_fasta(codon_fasta)
            _validate_alignment_records(codon_fasta, records, warnings, failures)
        except ValueError as exc:
            failures.append(f"unreadable codon FASTA {codon_fasta}: {exc}")

    if not map_tsv.exists():
        failures.append(f"missing map TSV: {map_tsv}")
    else:
        _validate_map_header(map_tsv, failures)

    if not qc_json.exists():
        failures.append(f"missing QC JSON: {qc_json}")
    else:
        try:
            _read_json(qc_json)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"unreadable QC JSON {qc_json}: {exc}")


def _validate_alignment_records(
    path: Path,
    records: dict,
    warnings: List[str],
    failures: List[str],
) -> None:
    lengths = [len(sequence) for sequence in records.values()]
    if any(length % 3 != 0 for length in lengths):
        failures.append(f"alignment length is not a multiple of 3: {path}")
    if len(set(lengths)) > 1:
        failures.append(f"alignment records have unequal lengths: {path}")
    if not records:
        warnings.append(f"alignment FASTA has no records: {path}")


def _validate_map_header(path: Path, failures: List[str]) -> None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n").split("\t")
    except OSError as exc:
        failures.append(f"unreadable map TSV {path}: {exc}")
        return

    if header != MAP_HEADER:
        failures.append(f"unexpected map TSV header: {path}")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def _summary(
    status: str,
    n_families_expected: int,
    n_families_checked: int,
    n_fail: int,
    n_warning: int,
    methods: list,
    warnings: List[str],
    failures: List[str],
) -> dict:
    return {
        "status": status,
        "n_families_expected": n_families_expected,
        "n_families_checked": n_families_checked,
        "n_fail": n_fail,
        "n_warning": n_warning,
        "methods": methods,
        "warnings": warnings,
        "failures": failures,
    }
