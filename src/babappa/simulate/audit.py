"""Simulation audit and dataset-level QC utilities for BABAPPA."""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Union

REQUIRED_FAMILY_SUFFIXES = {
    "fasta": ".fasta",
    "treefile": ".treefile",
    "truth": ".truth.json",
    "branch_truth": ".branch_truth.json",
    "homology": ".homology.tsv",
    "events": ".events.tsv",
    "meta": ".meta.json",
}
BRANCH_SITE_TRUTH_HEADER = [
    "family_id",
    "saturation_tier",
    "branch_id",
    "foreground_taxon",
    "branch_type",
    "site_index_zero",
    "site_index_one",
    "y_branch_site",
    "selection_event_id",
    "truth_source",
]
EVENTS_HEADER = [
    "family_id",
    "taxon",
    "codon_index_0based",
    "old_codon",
    "new_codon",
    "event_type",
    "is_selected_site",
    "is_foreground",
]
HOMOLOGY_HEADER = ["taxon", "codon_index_0based", "homology_id", "codon"]
AUDIT_FIELDNAMES = [
    "family_id",
    "status",
    "n_taxa",
    "min_length_nt",
    "max_length_nt",
    "same_length",
    "n_codons_min",
    "n_codons_max",
    "truth_gene_label",
    "truth_has_positive_selection",
    "foreground_taxon",
    "n_selected_sites",
    "saturation_tier",
    "n_events",
    "n_synonymous_events",
    "n_nonsynonymous_events",
    "n_selected_events",
    "mean_pairwise_nt_pdist",
    "codon_pos1_pdist",
    "codon_pos2_pdist",
    "codon_pos3_pdist",
    "transition_count",
    "transversion_count",
    "other_nt_change_count",
    "ti_tv_ratio",
    "branch_truth_present",
    "n_branch_site_truth_rows",
    "n_branch_positive_rows",
    "branch_truth_status",
    "warnings",
]
TRANSITIONS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
DNA_BASES = {"A", "C", "G", "T"}


def read_fasta(path: Union[str, Path]) -> Dict[str, str]:
    """Read a simple FASTA file into a record-id to sequence mapping."""
    fasta_path = Path(path)
    if not fasta_path.exists():
        raise ValueError(f"FASTA file does not exist: {fasta_path}")
    if not fasta_path.is_file():
        raise ValueError(f"FASTA path is not a file: {fasta_path}")

    records: Dict[str, str] = {}
    current_id: Optional[str] = None
    current_chunks: List[str] = []

    for line_number, raw_line in enumerate(
        fasta_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                if not current_chunks:
                    raise ValueError(f"empty FASTA record: {current_id}")
                records[current_id] = "".join(current_chunks).upper()
            current_id = line[1:].split()[0] if line[1:].strip() else ""
            if not current_id:
                raise ValueError(f"missing FASTA record id on line {line_number}")
            if current_id in records:
                raise ValueError(f"duplicate FASTA record id: {current_id}")
            current_chunks = []
            continue

        if current_id is None:
            raise ValueError(f"FASTA sequence encountered before header on line {line_number}")
        current_chunks.append(line.replace(" ", "").upper())

    if current_id is not None:
        if not current_chunks:
            raise ValueError(f"empty FASTA record: {current_id}")
        records[current_id] = "".join(current_chunks).upper()

    if not records:
        raise ValueError(f"FASTA file is empty or malformed: {fasta_path}")

    return records


def pairwise_p_distance(seq1: str, seq2: str) -> float:
    """Return nucleotide p-distance across comparable non-gap positions."""
    comparable = 0
    differences = 0
    for left, right in zip(seq1.upper(), seq2.upper()):
        if left == "-" or right == "-":
            continue
        comparable += 1
        if left != right:
            differences += 1

    if comparable == 0:
        return 0.0
    return differences / comparable


def classify_nt_change(old: str, new: str) -> str:
    """Classify a single-nucleotide change as transition or transversion."""
    old_base = old.upper()
    new_base = new.upper()
    if (
        len(old_base) != 1
        or len(new_base) != 1
        or old_base == new_base
        or old_base not in DNA_BASES
        or new_base not in DNA_BASES
    ):
        return "other"
    if (old_base, new_base) in TRANSITIONS:
        return "transition"
    return "transversion"


def compute_ti_tv_from_events(events_tsv: Path) -> dict:
    """Compute transition/transversion counts from a Cycle 2 events TSV."""
    transition_count = 0
    transversion_count = 0
    other_count = 0

    with Path(events_tsv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            old_codon = row.get("old_codon", "").upper()
            new_codon = row.get("new_codon", "").upper()
            changed_positions = [
                index
                for index, (old, new) in enumerate(zip(old_codon, new_codon))
                if old != new
            ]
            if len(old_codon) != 3 or len(new_codon) != 3 or len(changed_positions) != 1:
                other_count += 1
                continue

            change_type = classify_nt_change(
                old_codon[changed_positions[0]], new_codon[changed_positions[0]]
            )
            if change_type == "transition":
                transition_count += 1
            elif change_type == "transversion":
                transversion_count += 1
            else:
                other_count += 1

    ti_tv_ratio = (
        None if transversion_count == 0 else transition_count / transversion_count
    )
    return {
        "transition_count": transition_count,
        "transversion_count": transversion_count,
        "other_count": other_count,
        "ti_tv_ratio": ti_tv_ratio,
    }


def compute_codon_position_distances(fasta_records: Dict[str, str]) -> dict:
    """Compute mean pairwise p-distance for each codon position."""
    return {
        "codon_pos1_pdist": _mean_pairwise_distance_for_slices(fasta_records, 0),
        "codon_pos2_pdist": _mean_pairwise_distance_for_slices(fasta_records, 1),
        "codon_pos3_pdist": _mean_pairwise_distance_for_slices(fasta_records, 2),
    }


def compute_family_audit(family_dir: Union[str, Path]) -> dict:
    """Audit one simulated family directory."""
    family_path = Path(family_dir)
    family_id = family_path.name
    warnings: List[str] = []
    fail = False
    files = _detect_required_files(family_path, warnings)
    if any(path is None for key, path in files.items() if key != "branch_truth"):
        fail = True

    audit = _empty_family_audit(family_id)
    fasta_records: Dict[str, str] = {}
    truth: dict = {}
    event_rows: List[dict] = []

    if files["fasta"] is not None:
        try:
            fasta_records = read_fasta(files["fasta"])
            _populate_sequence_metrics(audit, fasta_records, warnings)
        except (OSError, ValueError) as exc:
            warnings.append(f"unreadable_required_file:fasta:{exc}")
            fail = True

    if files["truth"] is not None:
        try:
            truth = _read_json(files["truth"])
            _populate_truth_metrics(audit, truth, warnings)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"unreadable_required_file:truth:{exc}")
            fail = True

    if files["branch_truth"] is not None:
        try:
            branch_truth = _read_json(files["branch_truth"])
            _populate_branch_truth_metrics(audit, branch_truth, warnings)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"unreadable_optional_file:branch_truth:{exc}")
            audit["branch_truth_status"] = "invalid"
    else:
        warnings.append("branch_truth_missing")

    if files["events"] is not None:
        try:
            event_rows = _read_tsv(files["events"], EVENTS_HEADER)
            _populate_event_metrics(audit, event_rows)
            ti_tv = compute_ti_tv_from_events(files["events"])
            audit["transition_count"] = ti_tv["transition_count"]
            audit["transversion_count"] = ti_tv["transversion_count"]
            audit["other_nt_change_count"] = ti_tv["other_count"]
            audit["ti_tv_ratio"] = ti_tv["ti_tv_ratio"]
        except (OSError, ValueError) as exc:
            warnings.append(f"unreadable_required_file:events:{exc}")
            fail = True

    if files["homology"] is not None:
        try:
            _read_tsv(files["homology"], HOMOLOGY_HEADER)
        except (OSError, ValueError) as exc:
            warnings.append(f"unreadable_required_file:homology:{exc}")
            fail = True

    for key in ("treefile", "meta"):
        if files[key] is not None:
            try:
                if files[key].stat().st_size == 0:
                    raise ValueError("file is empty")
                if key == "meta":
                    _read_json(files[key])
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                warnings.append(f"unreadable_required_file:{key}:{exc}")
                fail = True

    if audit["n_events"] == 0:
        warnings.append("no_events_recorded")
    if (
        audit["truth_has_positive_selection"] is True
        and audit["n_selected_sites"] == 0
    ):
        warnings.append("positive_family_without_selected_sites")

    audit["warnings"] = sorted(set(warnings))
    if fail:
        audit["status"] = "fail"
    elif audit["warnings"]:
        audit["status"] = "warning"
    else:
        audit["status"] = "ok"
    return audit


def audit_simulation_directory(
    sim_dir: Union[str, Path], outdir: Optional[Union[str, Path]] = None
) -> dict:
    """Audit every family listed in a simulation manifest."""
    sim_path = Path(sim_dir)
    manifest_path = sim_path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"missing manifest.json: {manifest_path}")

    try:
        manifest = _read_json(manifest_path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed manifest.json: {exc}") from exc

    family_ids = manifest.get("family_ids")
    if not isinstance(family_ids, list):
        raise ValueError("manifest.json does not contain a family_ids list")

    audit_dir = Path(outdir) if outdir is not None else sim_path / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    family_audits = [
        compute_family_audit(sim_path / "families" / family_id)
        for family_id in family_ids
    ]
    dataset_branch_truth = _dataset_branch_truth_summary(sim_path)

    family_audit_path = audit_dir / "family_audit.tsv"
    dataset_summary_path = audit_dir / "dataset_summary.json"
    _write_family_audit_tsv(family_audit_path, family_audits)
    summary = _build_dataset_summary(
        manifest=manifest,
        family_audits=family_audits,
        family_audit_path=family_audit_path,
        dataset_summary_path=dataset_summary_path,
        dataset_branch_truth=dataset_branch_truth,
    )
    _write_json(dataset_summary_path, summary)
    return summary


def _detect_required_files(
    family_path: Path, warnings: List[str]
) -> Dict[str, Optional[Path]]:
    detected: Dict[str, Optional[Path]] = {}
    for key, suffix in REQUIRED_FAMILY_SUFFIXES.items():
        matches = sorted(family_path.glob(f"*{suffix}"))
        if not matches:
            if key == "branch_truth":
                warnings.append("missing_optional_branch_truth_file")
            else:
                warnings.append("missing_required_file")
            detected[key] = None
            continue
        detected[key] = matches[0]
    return detected


def _empty_family_audit(family_id: str) -> dict:
    return {
        "family_id": family_id,
        "status": "fail",
        "n_taxa": 0,
        "min_length_nt": 0,
        "max_length_nt": 0,
        "same_length": False,
        "n_codons_min": 0,
        "n_codons_max": 0,
        "truth_gene_label": None,
        "truth_has_positive_selection": None,
        "foreground_taxon": None,
        "n_selected_sites": 0,
        "saturation_tier": None,
        "n_events": 0,
        "n_synonymous_events": 0,
        "n_nonsynonymous_events": 0,
        "n_selected_events": 0,
        "mean_pairwise_nt_pdist": 0.0,
        "codon_pos1_pdist": 0.0,
        "codon_pos2_pdist": 0.0,
        "codon_pos3_pdist": 0.0,
        "transition_count": 0,
        "transversion_count": 0,
        "other_nt_change_count": 0,
        "ti_tv_ratio": None,
        "branch_truth_present": False,
        "n_branch_site_truth_rows": 0,
        "n_branch_positive_rows": 0,
        "branch_truth_status": "missing",
        "warnings": [],
    }


def _populate_sequence_metrics(
    audit: dict, fasta_records: Dict[str, str], warnings: List[str]
) -> None:
    lengths = [len(sequence) for sequence in fasta_records.values()]
    audit["n_taxa"] = len(fasta_records)
    audit["min_length_nt"] = min(lengths)
    audit["max_length_nt"] = max(lengths)
    audit["same_length"] = len(set(lengths)) == 1
    audit["n_codons_min"] = min(length // 3 for length in lengths)
    audit["n_codons_max"] = max(length // 3 for length in lengths)
    audit["mean_pairwise_nt_pdist"] = _mean_pairwise_distance(fasta_records.values())
    audit.update(compute_codon_position_distances(fasta_records))

    if any(length % 3 != 0 for length in lengths):
        warnings.append("length_not_multiple_of_three")
    if not audit["same_length"]:
        warnings.append("unequal_sequence_lengths")


def _populate_truth_metrics(audit: dict, truth: dict, warnings: List[str]) -> None:
    labels = truth.get("labels", {})
    selected_sites = truth.get("selected_sites_0based", [])
    audit["truth_gene_label"] = labels.get("gene_label")
    audit["truth_has_positive_selection"] = truth.get("has_positive_selection")
    audit["foreground_taxon"] = truth.get("foreground_taxon")
    audit["n_selected_sites"] = (
        len(selected_sites) if isinstance(selected_sites, list) else 0
    )
    audit["saturation_tier"] = truth.get("saturation_tier")

    if (
        audit["truth_has_positive_selection"] is True
        and audit["n_selected_sites"] == 0
    ):
        warnings.append("positive_family_without_selected_sites")


def _populate_event_metrics(audit: dict, event_rows: List[dict]) -> None:
    audit["n_events"] = len(event_rows)
    audit["n_synonymous_events"] = sum(
        1 for row in event_rows if row.get("event_type") == "synonymous"
    )
    audit["n_nonsynonymous_events"] = sum(
        1 for row in event_rows if row.get("event_type") == "nonsynonymous"
    )
    audit["n_selected_events"] = sum(
        1 for row in event_rows if _is_truthy(row.get("is_selected_site", "0"))
    )


def _populate_branch_truth_metrics(audit: dict, branch_truth: dict, warnings: List[str]) -> None:
    records = branch_truth.get("branch_site_records")
    if not isinstance(records, list):
        warnings.append("branch_truth_records_missing_or_invalid")
        audit["branch_truth_status"] = "invalid"
        return
    audit["branch_truth_present"] = True
    audit["n_branch_site_truth_rows"] = len(records)
    audit["n_branch_positive_rows"] = sum(
        1 for row in records if _is_truthy(row.get("y_branch_site", "0"))
    )
    sources = {str(row.get("truth_source", "")) for row in records if isinstance(row, dict)}
    if branch_truth.get("truth_source") == "explicit_simulator_branch_truth" or not sources:
        audit["branch_truth_status"] = "explicit_truth_ok"
    elif sources == {"explicit_simulator_branch_truth"}:
        audit["branch_truth_status"] = "explicit_truth_ok"
    else:
        audit["branch_truth_status"] = "invalid"
        warnings.append("branch_truth_source_not_explicit")


def _read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def _read_tsv(path: Path, expected_header: List[str]) -> List[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != expected_header:
            raise ValueError(f"unexpected TSV header in {path}")
        return list(reader)


def _mean_pairwise_distance(sequences: Iterable[str]) -> float:
    distances = [
        pairwise_p_distance(left, right)
        for left, right in combinations(list(sequences), 2)
    ]
    return mean(distances) if distances else 0.0


def _mean_pairwise_distance_for_slices(
    fasta_records: Dict[str, str], start_index: int
) -> float:
    sliced = [sequence[start_index::3] for sequence in fasta_records.values()]
    return _mean_pairwise_distance(sliced)


def _is_truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _write_family_audit_tsv(path: Path, family_audits: List[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDNAMES, delimiter="\t")
        writer.writeheader()
        for audit in family_audits:
            row = dict(audit)
            row["warnings"] = ";".join(row["warnings"])
            writer.writerow(row)


def _build_dataset_summary(
    manifest: dict,
    family_audits: List[dict],
    family_audit_path: Path,
    dataset_summary_path: Path,
    dataset_branch_truth: dict,
) -> dict:
    pdists = [
        audit["mean_pairwise_nt_pdist"]
        for audit in family_audits
        if audit["mean_pairwise_nt_pdist"] is not None
    ]
    saturation_tier_counts: Dict[str, int] = {}
    for audit in family_audits:
        tier = audit.get("saturation_tier") or "unknown"
        saturation_tier_counts[tier] = saturation_tier_counts.get(tier, 0) + 1

    n_branch_truth_files = sum(1 for audit in family_audits if audit.get("branch_truth_present"))
    n_branch_site_truth_rows = sum(int(audit.get("n_branch_site_truth_rows") or 0) for audit in family_audits)
    n_branch_positive_rows = sum(int(audit.get("n_branch_positive_rows") or 0) for audit in family_audits)
    branch_truth_status = _dataset_branch_truth_status(family_audits, dataset_branch_truth)
    return {
        "simulator_version": manifest.get("simulator_version"),
        "n_families_expected": manifest.get("n_families", len(family_audits)),
        "n_families_audited": len(family_audits),
        "n_ok": sum(1 for audit in family_audits if audit["status"] == "ok"),
        "n_warning": sum(
            1 for audit in family_audits if audit["status"] == "warning"
        ),
        "n_fail": sum(1 for audit in family_audits if audit["status"] == "fail"),
        "positive_family_count": sum(
            1 for audit in family_audits if audit["truth_has_positive_selection"] is True
        ),
        "mean_pairwise_nt_pdist_mean": mean(pdists) if pdists else 0.0,
        "mean_pairwise_nt_pdist_min": min(pdists) if pdists else 0.0,
        "mean_pairwise_nt_pdist_max": max(pdists) if pdists else 0.0,
        "saturation_tier_counts": saturation_tier_counts,
        "branch_truth_present": n_branch_truth_files == len(family_audits) and bool(family_audits),
        "n_branch_truth_files": n_branch_truth_files,
        "n_branch_site_truth_rows": dataset_branch_truth.get("n_branch_site_truth_rows", n_branch_site_truth_rows),
        "n_branch_positive_rows": dataset_branch_truth.get("n_branch_positive_rows", n_branch_positive_rows),
        "branch_truth_status": branch_truth_status,
        "audit_files": {
            "family_audit_tsv": str(family_audit_path),
            "dataset_summary_json": str(dataset_summary_path),
        },
    }


def _dataset_branch_truth_summary(sim_path: Path) -> dict:
    summary = {
        "branch_truth_manifest_present": False,
        "branch_site_truth_tsv_present": False,
        "n_branch_site_truth_rows": 0,
        "n_branch_positive_rows": 0,
    }
    manifest_path = sim_path / "branch_truth_manifest.json"
    if manifest_path.exists():
        try:
            payload = _read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            payload = {}
        summary["branch_truth_manifest_present"] = True
        for key in ["n_branch_site_truth_rows", "n_branch_positive_rows"]:
            if key in payload:
                summary[key] = int(payload.get(key) or 0)
    tsv_path = sim_path / "branch_site_truth.tsv"
    if tsv_path.exists():
        summary["branch_site_truth_tsv_present"] = True
        try:
            n_rows, n_positive = _count_branch_site_truth_tsv(tsv_path)
        except (OSError, ValueError):
            n_rows, n_positive = 0, 0
        if n_rows:
            summary["n_branch_site_truth_rows"] = n_rows
            summary["n_branch_positive_rows"] = n_positive
    return summary


def _count_branch_site_truth_tsv(path: Path) -> tuple[int, int]:
    n_rows = 0
    n_positive = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != BRANCH_SITE_TRUTH_HEADER:
            raise ValueError(f"unexpected TSV header in {path}")
        for row in reader:
            n_rows += 1
            if _is_truthy(row.get("y_branch_site", "0")):
                n_positive += 1
    return n_rows, n_positive


def _dataset_branch_truth_status(family_audits: List[dict], dataset_branch_truth: dict) -> str:
    if not family_audits:
        return "missing"
    statuses = {str(audit.get("branch_truth_status", "missing")) for audit in family_audits}
    if statuses == {"explicit_truth_ok"}:
        if dataset_branch_truth.get("branch_truth_manifest_present") and dataset_branch_truth.get("branch_site_truth_tsv_present"):
            return "explicit_truth_ok"
        return "explicit_family_truth_only"
    if "explicit_truth_ok" in statuses:
        return "partial"
    if "invalid" in statuses:
        return "invalid"
    return "missing"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
