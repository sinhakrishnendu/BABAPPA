"""Truth-status audit for branch-conditioned labels."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from babappa import __version__
from babappa.branch.summary import _tier_prefix, _unsuffixed_prefix
from babappa.datasets.index import write_tsv

AUDIT_FILES = {
    "json": "branch_truth_status_audit.json",
    "tsv": "branch_truth_status_audit.tsv",
    "markdown": "branch_truth_status_audit.md",
}

AUDIT_FIELDS = [
    "tier",
    "label_dir",
    "audit_status",
    "label_status",
    "explicit_branch_site_truth_available",
    "proxy_from_foreground_taxon",
    "not_available",
    "n_branch_site_rows",
    "n_positive_branch_sites",
    "positive_branch_site_fraction",
    "n_branch_ids_represented",
    "branch_ids_represented",
    "n_foreground_taxa_represented",
    "foreground_taxa_represented",
    "y_branch_site_derivation",
    "status_counts",
    "branch_label_source_counts",
    "warnings",
]

REQUIRED_TIERS = {"low", "moderate", "high", "extreme"}


@dataclass(frozen=True)
class BranchTruthStatusAuditConfig:
    """Configuration for branch truth-status audit."""

    tiers: Union[str, Sequence[str]]
    outdir: str
    run_name: str = "fast_external_10k_streamed"
    output_suffix: Optional[str] = None
    allow_streamed: bool = True


def audit_branch_truth_status(config: BranchTruthStatusAuditConfig) -> Dict[str, Any]:
    """Audit whether branch-site labels are explicit simulator truth or proxies."""

    tiers = _parse_tiers(config.tiers)
    if not tiers:
        raise ValueError("at least one tier must be supplied")
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    failures: List[str] = []
    tier_records = [
        _audit_tier(tier, config.run_name, warnings, failures, config.output_suffix, config.allow_streamed)
        for tier in tiers
    ]
    if failures:
        raise ValueError("; ".join(failures))

    proxy_tiers = [row["tier"] for row in tier_records if row["proxy_from_foreground_taxon"]]
    explicit_tiers = [row["tier"] for row in tier_records if row["audit_status"] == "explicit_truth_ok"]
    payload = {
        "branch_truth_status_audit_version": __version__,
        "title": "BABAPPA branch truth-status audit",
        "run_name": config.run_name,
        "output_suffix": config.output_suffix,
        "allow_streamed": config.allow_streamed,
        "tiers_requested": tiers,
        "tiers_included": [row["tier"] for row in tier_records],
        "explicit_truth_available": len(explicit_tiers) == len(tier_records) and bool(tier_records),
        "explicit_truth_tiers": explicit_tiers,
        "proxy_label_tiers": proxy_tiers,
        "audit_status_counts": dict(sorted(Counter(row["audit_status"] for row in tier_records).items())),
        "tier_records": tier_records,
        "warnings": warnings,
        "interpretation": _interpretation(proxy_tiers, explicit_tiers, tier_records),
        "recommended_next_step": _recommended_next_step(tier_records),
        "generated_files": {
            name: str(outdir / filename) for name, filename in AUDIT_FILES.items()
        },
    }
    _write_json(outdir / AUDIT_FILES["json"], payload)
    write_tsv(outdir / AUDIT_FILES["tsv"], _tsv_rows(tier_records), AUDIT_FIELDS)
    (outdir / AUDIT_FILES["markdown"]).write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "json": str(outdir / AUDIT_FILES["json"]),
        "tsv": str(outdir / AUDIT_FILES["tsv"]),
        "markdown": str(outdir / AUDIT_FILES["markdown"]),
        "explicit_truth_available": payload["explicit_truth_available"],
        "proxy_label_tiers": proxy_tiers,
        "n_warning": len(warnings),
        "warnings": warnings,
    }


def validate_branch_truth_status_audit_dir(audit_dir: Union[str, Path]) -> Dict[str, Any]:
    """Validate branch truth-status audit artifacts."""

    path = Path(audit_dir)
    failures: List[str] = []
    warnings: List[str] = []
    payload = _load_json(path / AUDIT_FILES["json"], failures, "audit_json")

    for filename in [AUDIT_FILES["tsv"], AUDIT_FILES["markdown"]]:
        candidate = path / filename
        if not candidate.exists():
            failures.append(f"missing_file:{candidate}")
        elif candidate.stat().st_size == 0:
            failures.append(f"empty_file:{candidate}")

    if payload:
        tiers = payload.get("tiers_included")
        if not isinstance(tiers, list):
            failures.append("audit_json_missing_tiers_included")
        else:
            missing = sorted(REQUIRED_TIERS - {str(tier) for tier in tiers})
            if missing:
                failures.append("missing_required_tiers:" + ",".join(missing))
        records = payload.get("tier_records")
        if not isinstance(records, list) or not records:
            failures.append("audit_json_missing_tier_records")
        else:
            for record in records:
                if not isinstance(record, dict):
                    failures.append("audit_record_not_object")
                    continue
                if record.get("proxy_from_foreground_taxon"):
                    warnings.append(f"proxy_labels_used:{record.get('tier')}")
                if record.get("audit_status") == "explicit_truth_ok" and record.get("warnings"):
                    failures.append(f"explicit_truth_ok_has_warnings:{record.get('tier')}")
                if record.get("y_branch_site_derivation") in (None, "", "unknown"):
                    failures.append(f"missing_y_branch_site_derivation:{record.get('tier')}")
        warnings.extend(str(warning) for warning in payload.get("warnings", []))
        if payload.get("proxy_label_tiers") and "proxy validation" not in str(payload.get("interpretation", "")):
            failures.append("proxy_interpretation_missing")

    markdown_path = path / AUDIT_FILES["markdown"]
    if markdown_path.exists():
        text = markdown_path.read_text(encoding="utf-8")
        for required in [
            "# BABAPPA branch truth-status audit",
            "## Scientific boundary",
            "## Truth-status by tier",
            "## Recommendation",
        ]:
            if required not in text:
                failures.append(f"markdown_missing_section:{required}")

    return {
        "status": "fail" if failures else "ok",
        "n_fail": len(failures),
        "n_warning": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _audit_tier(
    tier: str,
    run_name: str,
    warnings: List[str],
    failures: List[str],
    output_suffix: Optional[str] = None,
    allow_streamed: bool = True,
) -> Dict[str, Any]:
    prefix = _tier_prefix(run_name, tier, output_suffix=output_suffix, allow_streamed=allow_streamed)
    label_dir = _label_dir_for_tier(tier, prefix)
    summary_path = label_dir / "branch_site_oracle_summary.json"
    summary = _load_json(summary_path, failures, f"{tier}:branch_site_oracle_summary")
    if not summary:
        return _missing_tier_record(tier, label_dir)

    labels_path = _generated_label_path(summary, label_dir)
    dataset_index = _dataset_index_for_tier(prefix)
    row_info = _row_info(labels_path, summary, dataset_index)
    status_counts = _counter_from_mapping(summary.get("status_counts"))
    source_counts = _counter_from_mapping(summary.get("branch_label_source_counts"))
    if not source_counts and labels_path.exists() and _safe_int(summary.get("n_branch_site_rows"), 0) <= 1_000_000:
        source_counts = row_info["source_counts"]

    status_values = [str(summary.get("branch_site_labels_status", ""))]
    status_values.extend(status_counts.keys())
    status_values.extend(source_counts.keys())
    explicit = bool(summary.get("explicit_branch_site_truth_available")) or any(_looks_explicit(value) for value in status_values)
    proxy = bool(summary.get("proxy_labels_used")) or any("proxy" in value.lower() for value in status_values)
    not_available = any("not_available" in value.lower() for value in status_values)
    derivation = _derivation(explicit, proxy, not_available)
    audit_status = _audit_status(explicit, proxy, not_available)

    tier_warnings = list(summary.get("warnings", []))
    if proxy:
        tier_warnings.append(
            "proxy_labels_used:y_branch_site derives from foreground branch/taxon labels crossed with selected-site labels"
        )
    if not explicit:
        tier_warnings.append("explicit_branch_site_truth_not_available")
    if audit_status == "explicit_truth_ok":
        tier_warnings = []
    warnings.extend(f"{tier}: {warning}" for warning in tier_warnings)

    return {
        "tier": tier,
        "label_dir": str(label_dir),
        "audit_status": audit_status,
        "summary_json": str(summary_path),
        "labels_tsv": str(labels_path),
        "label_status": summary.get("branch_site_labels_status", "unknown"),
        "explicit_branch_site_truth_available": explicit,
        "proxy_from_foreground_taxon": proxy,
        "not_available": not_available,
        "n_branch_site_rows": summary.get("n_branch_site_rows", row_info["n_rows"]),
        "n_positive_branch_sites": summary.get("n_positive_branch_sites", row_info["n_positive"]),
        "positive_branch_site_fraction": summary.get("positive_branch_site_fraction"),
        "branch_ids_represented": row_info["branch_ids"],
        "foreground_taxa_represented": row_info["foreground_taxa"],
        "branch_id_source": row_info["branch_id_source"],
        "y_branch_site_derivation": derivation,
        "status_counts": dict(status_counts),
        "branch_label_source_counts": dict(source_counts),
        "warnings": sorted(set(tier_warnings)),
    }


def _missing_tier_record(tier: str, label_dir: Path) -> Dict[str, Any]:
    return {
        "tier": tier,
        "label_dir": str(label_dir),
        "audit_status": "missing",
        "summary_json": str(label_dir / "branch_site_oracle_summary.json"),
        "labels_tsv": str(label_dir / "branch_site_oracle_labels.tsv"),
        "label_status": "not_available",
        "explicit_branch_site_truth_available": False,
        "proxy_from_foreground_taxon": False,
        "not_available": True,
        "n_branch_site_rows": 0,
        "n_positive_branch_sites": 0,
        "positive_branch_site_fraction": None,
        "branch_ids_represented": [],
        "foreground_taxa_represented": [],
        "branch_id_source": "missing",
        "y_branch_site_derivation": "not_available",
        "status_counts": {"not_available": 1},
        "branch_label_source_counts": {},
        "warnings": ["branch_site_oracle_summary_missing"],
    }


def _row_info(labels_path: Path, summary: Dict[str, Any], dataset_index: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if dataset_index and isinstance(dataset_index.get("branch_counts"), dict):
        return {
            "n_rows": _safe_int(summary.get("n_branch_site_rows"), 0),
            "n_positive": _safe_int(summary.get("n_positive_branch_sites"), 0),
            "branch_ids": sorted(str(key) for key in dataset_index["branch_counts"].keys()),
            "foreground_taxa": [],
            "branch_id_source": "branch_site_dataset_index.branch_counts",
            "source_counts": Counter(),
        }

    n_summary_rows = _safe_int(summary.get("n_branch_site_rows"), 0)
    if not labels_path.exists() or n_summary_rows > 1_000_000:
        return {
            "n_rows": n_summary_rows,
            "n_positive": _safe_int(summary.get("n_positive_branch_sites"), 0),
            "branch_ids": [],
            "foreground_taxa": [],
            "branch_id_source": "not_scanned",
            "source_counts": Counter(),
        }

    branch_ids = set()
    foreground_taxa = set()
    source_counts: Counter[str] = Counter()
    n_rows = 0
    n_positive = 0
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            n_rows += 1
            if row.get("branch_id"):
                branch_ids.add(str(row["branch_id"]))
            if row.get("foreground_taxon"):
                foreground_taxa.add(str(row["foreground_taxon"]))
            if row.get("branch_label_source"):
                source_counts[str(row["branch_label_source"])] += 1
            if row.get("y_branch_site") == "1":
                n_positive += 1
    return {
        "n_rows": n_rows,
        "n_positive": n_positive,
        "branch_ids": sorted(branch_ids),
        "foreground_taxa": sorted(foreground_taxa),
        "branch_id_source": "branch_site_oracle_labels.tsv",
        "source_counts": source_counts,
    }


def _label_dir_for_tier(tier: str, prefix: str) -> Path:
    run_summary_path = Path(f"branch_site_run_summary_{prefix}") / "branch_site_run_summary.json"
    summary = _load_json(run_summary_path, [], f"{tier}:run_summary")
    section = ((summary or {}).get("sections") or {}).get("branch_site_labels")
    if isinstance(section, dict) and section.get("directory"):
        candidate = Path(str(section["directory"]))
        if candidate.exists():
            return candidate
    unsuffixed = _unsuffixed_prefix(prefix)
    candidates = [
        Path(f"branch_site_oracle_{unsuffixed}"),
        Path(f"branch_site_oracle_{prefix}"),
        Path(f"branch_site_labels_{unsuffixed}"),
        Path(f"branch_site_labels_{prefix}"),
        Path(f"branch_site_oracle_fast_external_10k_{tier}"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _dataset_index_for_tier(prefix: str) -> Optional[Dict[str, Any]]:
    run_summary_path = Path(f"branch_site_run_summary_{prefix}") / "branch_site_run_summary.json"
    run_summary = _load_json(run_summary_path, [], "run_summary")
    section = ((run_summary or {}).get("sections") or {}).get("branch_site_dataset")
    candidates = []
    if isinstance(section, dict) and section.get("directory"):
        candidates.append(Path(str(section["directory"])))
    candidates.append(Path(f"branch_site_dataset_{prefix}"))
    for directory in candidates:
        payload = _load_json(directory / "branch_site_dataset_index.json", [], "dataset_index")
        if payload:
            return payload
    return None


def _generated_label_path(summary: Dict[str, Any], label_dir: Path) -> Path:
    generated = summary.get("generated_files")
    if isinstance(generated, dict) and generated.get("labels_tsv"):
        return Path(str(generated["labels_tsv"]))
    return label_dir / "branch_site_oracle_labels.tsv"


def _interpretation(proxy_tiers: List[str], explicit_tiers: List[str], tier_records: List[Dict[str, Any]]) -> str:
    explicit_ok = [row for row in tier_records if row.get("audit_status") == "explicit_truth_ok"]
    if proxy_tiers and len(proxy_tiers) >= max(1, len(tier_records) // 2):
        return (
            "Proxy labels dominate this audit. This is branch-conditioned proxy validation, "
            "not final branch-site truth validation. The simulator should be upgraded to emit "
            "explicit branch-site selected-event truth."
        )
    if proxy_tiers:
        return (
            "Some tiers use proxy labels. Treat results as mixed branch-conditioned validation "
            "until explicit branch-site truth is available for every tier."
        )
    if explicit_ok and len(explicit_ok) == len(tier_records):
        return "All audited tiers report explicit branch-site truth availability with status explicit_truth_ok."
    return "Branch-site truth status is incomplete or unavailable."


def _all_tiers_explicit_truth_ok(tier_records: List[Dict[str, Any]]) -> bool:
    if not tier_records:
        return False
    return all(
        bool(record.get("explicit_branch_site_truth_available"))
        and not bool(record.get("proxy_from_foreground_taxon"))
        and record.get("audit_status") == "explicit_truth_ok"
        for record in tier_records
    )


def _recommended_next_step(tier_records: List[Dict[str, Any]]) -> str:
    if _all_tiers_explicit_truth_ok(tier_records):
        return (
            "Explicit simulator branch-site truth is available for all audited tiers. "
            "Proceed to explicit branch-truth validation at larger scale only after "
            "checking foreground-context ablation and aggregation controls."
        )
    return "Upgrade the simulator to emit explicit branch-site selected-event truth."


def _render_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# BABAPPA branch truth-status audit",
        "",
        "## Scientific boundary",
        "",
        payload.get("interpretation", ""),
        "",
        "## Truth-status by tier",
        "",
        "| Tier | Audit status | Label status | Explicit truth | Proxy labels | Rows | Positives | Derivation |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for record in payload.get("tier_records", []):
        lines.append(
            f"| {record.get('tier')} | `{record.get('audit_status')}` | `{record.get('label_status')}` | "
            f"{record.get('explicit_branch_site_truth_available')} | "
            f"{record.get('proxy_from_foreground_taxon')} | "
            f"{record.get('n_branch_site_rows')} | "
            f"{record.get('n_positive_branch_sites')} | "
            f"`{record.get('y_branch_site_derivation')}` |"
        )
    lines.extend([
        "",
        "## Branch IDs and taxa",
        "",
    ])
    for record in payload.get("tier_records", []):
        branch_ids = ", ".join(record.get("branch_ids_represented") or []) or "unavailable"
        foreground_taxa = ", ".join(record.get("foreground_taxa_represented") or []) or "unavailable"
        lines.append(f"- {record.get('tier')}: branch IDs `{branch_ids}`; foreground taxa `{foreground_taxa}`")
    lines.extend([
        "",
        "## Recommendation",
        "",
        payload.get("recommended_next_step", ""),
        "Do not run final 100K until explicit branch-site truth validation passes.",
        "",
    ])
    return "\n".join(lines)


def _tsv_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for record in records:
        rows.append({
            "tier": record.get("tier"),
            "label_dir": record.get("label_dir"),
            "audit_status": record.get("audit_status"),
            "label_status": record.get("label_status"),
            "explicit_branch_site_truth_available": record.get("explicit_branch_site_truth_available"),
            "proxy_from_foreground_taxon": record.get("proxy_from_foreground_taxon"),
            "not_available": record.get("not_available"),
            "n_branch_site_rows": record.get("n_branch_site_rows"),
            "n_positive_branch_sites": record.get("n_positive_branch_sites"),
            "positive_branch_site_fraction": record.get("positive_branch_site_fraction"),
            "n_branch_ids_represented": len(record.get("branch_ids_represented") or []),
            "branch_ids_represented": ";".join(record.get("branch_ids_represented") or []),
            "n_foreground_taxa_represented": len(record.get("foreground_taxa_represented") or []),
            "foreground_taxa_represented": ";".join(record.get("foreground_taxa_represented") or []),
            "y_branch_site_derivation": record.get("y_branch_site_derivation"),
            "status_counts": json.dumps(record.get("status_counts", {}), sort_keys=True),
            "branch_label_source_counts": json.dumps(record.get("branch_label_source_counts", {}), sort_keys=True),
            "warnings": ";".join(record.get("warnings") or []),
        })
    return rows


def _looks_explicit(value: str) -> bool:
    text = value.lower()
    if "proxy" in text or "not_available" in text:
        return False
    return "explicit" in text or "branch_site_truth" in text or "y_branch_site_matrix" in text


def _derivation(explicit: bool, proxy: bool, not_available: bool) -> str:
    if explicit and proxy:
        return "mixed_explicit_and_proxy"
    if explicit:
        return "direct_simulator_branch_site_truth"
    if proxy:
        return "proxy_foreground_branch_x_selected_site"
    if not_available:
        return "not_available"
    return "unknown"


def _audit_status(explicit: bool, proxy: bool, not_available: bool) -> str:
    if explicit and not proxy:
        return "explicit_truth_ok"
    if explicit and proxy:
        return "mixed"
    if proxy:
        return "proxy_warning"
    if not_available:
        return "missing"
    return "missing"


def _counter_from_mapping(value: Any) -> Counter[str]:
    counter: Counter[str] = Counter()
    if isinstance(value, dict):
        for key, count in value.items():
            counter[str(key)] = _safe_int(count, 0)
    return counter


def _parse_tiers(value: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _load_json(path: Path, failures: List[str], label: str) -> Dict[str, Any]:
    if not path.exists():
        if failures is not None:
            failures.append(f"missing_file:{path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if failures is not None:
            failures.append(f"could_not_parse_json:{label}:{path}:{exc}")
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
