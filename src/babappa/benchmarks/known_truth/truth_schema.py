"""Truth schemas and shared I/O for known-truth BABAPPA benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv

FAMILY_TRUTH_FIELDS = [
    "family_id",
    "regime",
    "truth_class",
    "n_taxa",
    "n_codons",
    "foreground_branches",
    "positive_branches",
    "selected_sites",
    "selected_branch_site_pairs",
    "omega_background",
    "omega_positive",
    "effect_size",
    "saturation_tier",
    "expected_applicability",
    "expected_decision",
]

BRANCH_SITE_TRUTH_FIELDS = [
    "family_id",
    "branch",
    "site",
    "label",
    "truth_class",
    "regime",
]

SELECTED_SITE_FIELDS = ["family_id", "site", "label", "regime"]
SELECTED_BRANCH_FIELDS = ["family_id", "branch", "label", "regime"]

TRUTH_MANIFEST_FIELDS = [
    "family_id",
    "regime",
    "truth_class",
    "n_taxa",
    "n_codons",
    "foreground_branches",
    "positive_branches",
    "n_selected_sites",
    "n_selected_branch_site_pairs",
    "saturation_tier",
    "expected_applicability",
    "expected_decision",
    "family_dir",
    "cds_fasta",
    "tree_file",
    "family_truth_json",
    "branch_site_truth_tsv",
]

TRUTH_BOUNDARY = (
    "Known-truth files are for benchmark evaluation only. They must never be used "
    "as empirical inference inputs or as features for BABAPPA scoring."
)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_csv(values: Iterable[Any]) -> str:
    return ",".join(str(value) for value in values)


def csv_list(value: str) -> List[str]:
    if value is None:
        return []
    return [part for part in str(value).split(",") if part]


def validate_family_truth_dir(family_dir: Path) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []
    required = [
        "family_truth.json",
        "branch_site_truth.tsv",
        "selected_sites.tsv",
        "selected_branches.tsv",
        "regime_metadata.json",
        "cds.fasta",
        "tree.nwk",
    ]
    for name in required:
        if not (family_dir / name).exists():
            failures.append(f"missing:{name}")
    family_truth: Dict[str, Any] = {}
    if (family_dir / "family_truth.json").exists():
        family_truth = read_json(family_dir / "family_truth.json")
        for field in FAMILY_TRUTH_FIELDS:
            if field not in family_truth:
                failures.append(f"family_truth_missing_field:{field}")
    if (family_dir / "branch_site_truth.tsv").exists():
        rows = read_tsv(family_dir / "branch_site_truth.tsv")
        if not rows:
            warnings.append("branch_site_truth_empty")
        for row in rows[:20]:
            for field in BRANCH_SITE_TRUTH_FIELDS:
                if field not in row:
                    failures.append(f"branch_site_truth_missing_field:{field}")
                    break
    if "empirical" in str(family_dir).lower():
        warnings.append("truth_files_inside_empirical_named_path")
    return {
        "status": "fail" if failures else "ok",
        "family_id": family_truth.get("family_id", family_dir.name),
        "failures": failures,
        "warnings": warnings,
    }


def validate_truth_manifest(manifest_path: Path) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []
    rows = read_tsv(manifest_path) if manifest_path.exists() else []
    if not rows:
        failures.append("missing_or_empty_truth_manifest")
    else:
        for field in TRUTH_MANIFEST_FIELDS:
            if field not in rows[0]:
                failures.append(f"manifest_missing_field:{field}")
        seen: set[str] = set()
        for row in rows:
            family_id = row.get("family_id", "")
            if family_id in seen:
                failures.append(f"duplicate_family_id:{family_id}")
            seen.add(family_id)
            family_dir = Path(row.get("family_dir", ""))
            if family_dir.exists():
                family_status = validate_family_truth_dir(family_dir)
                failures.extend(f"{family_id}:{item}" for item in family_status["failures"])
                warnings.extend(f"{family_id}:{item}" for item in family_status["warnings"])
            else:
                failures.append(f"missing_family_dir:{family_dir}")
    return {
        "known_truth_schema_version": __version__,
        "status": "fail" if failures else "ok",
        "n_families": len(rows),
        "failures": failures,
        "warnings": warnings,
        "truth_boundary": TRUTH_BOUNDARY,
    }


def write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    write_tsv(path, rows, TRUTH_MANIFEST_FIELDS)

