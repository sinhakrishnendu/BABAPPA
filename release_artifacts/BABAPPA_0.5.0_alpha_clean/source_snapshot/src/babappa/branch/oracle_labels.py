"""Branch-conditioned oracle label extraction for BABAPPA."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv
from babappa.site.oracle_labels import (
    OracleSiteLabelConfig,
    _feature_lookup,
    _first_nonempty,
    _load_optional_json,
    _load_tensor_shape,
    _mapped_site_iter,
    _merge_feature_context,
    _resolve_labels_path,
    normalize_site_indices,
)
from babappa.training.neural_data import resolve_tensor_file

BRANCH_SITE_ORACLE_VERSION = __version__
ALIGNED_SITE_MODES = {"original", "mapped"}
FOREGROUND_SOURCES = {"auto", "truth"}
TRUTH_MODES = {"auto", "explicit", "required", "proxy"}
SELECTED_SITE_KEYS = [
    "selected_sites_0based",
    "selected_sites",
    "positive_sites",
    "oracle_selected_sites",
    "selected_site_indices",
    "selected_sites_1based",
]
LABEL_FIELDNAMES = [
    "family_id",
    "method",
    "split",
    "saturation_tier",
    "branch_id",
    "foreground_taxon",
    "site_index_zero",
    "aligned_site_index_zero",
    "original_site_index_zero",
    "y_branch_site",
    "y_site",
    "gene_label",
    "foreground_branch_present",
    "branch_label_source",
    "mapping_status",
    "mapping_confidence",
    "mappable_site",
    "original_family_id",
    "source_dataset",
    "tensor_file",
    "labels_file",
    "n_taxa",
    "n_codons",
]


@dataclass(frozen=True)
class BranchSiteOracleLabelConfig:
    """Configuration for branch-conditioned oracle label extraction."""

    dataset_dir: str
    outdir: str
    site_map_dir: Optional[str] = None
    aligned_site_mode: str = "mapped"
    foreground_source: str = "auto"
    truth_mode: str = "auto"
    streaming_output: bool = True

    def __post_init__(self) -> None:
        dataset = Path(self.dataset_dir)
        if not dataset.exists():
            raise ValueError(f"dataset_dir does not exist: {dataset}")
        for filename in ("dataset_index.json", "splits.tsv"):
            if not (dataset / filename).exists():
                raise ValueError(f"dataset_dir is missing {filename}: {dataset}")
        if self.site_map_dir is not None and not Path(self.site_map_dir).exists():
            raise ValueError(f"site_map_dir does not exist: {self.site_map_dir}")
        if self.aligned_site_mode not in ALIGNED_SITE_MODES:
            raise ValueError("aligned_site_mode must be mapped or original")
        if self.foreground_source not in FOREGROUND_SOURCES:
            raise ValueError("foreground_source must be auto or truth")
        if self.truth_mode not in TRUTH_MODES:
            raise ValueError("truth_mode must be auto, explicit, required, or proxy")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def extract_branch_site_labels(config: BranchSiteOracleLabelConfig) -> dict:
    """Extract branch x site supervised labels from simulator truth where available."""
    dataset_dir = Path(config.dataset_dir)
    outdir = Path(config.outdir)
    labels_path = outdir / "branch_site_oracle_labels.tsv"
    summary_path = outdir / "branch_site_oracle_summary.json"
    markdown_path = outdir / "branch_site_oracle_labels.md"
    warnings: List[str] = []
    rows_out: List[dict] = []
    site_map_cache: Dict[Tuple[str, str], List[dict]] = {}
    feature_lookup = _feature_lookup(dataset_dir)
    split_rows = read_tsv(dataset_dir / "splits.tsv")
    status_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    row_count = 0
    positives = 0
    branch_positive_rows = 0
    explicit_branch_truth_available = False
    proxy_labels_used = False
    n_explicit_branch_truth_rows = 0
    n_proxy_rows = 0
    branch_truth_cache: Dict[str, Optional[dict]] = {}

    site_config = OracleSiteLabelConfig(
        dataset_dir=str(dataset_dir),
        outdir=str(outdir),
        site_map_dir=config.site_map_dir,
        aligned_site_mode=config.aligned_site_mode,
        selected_sites_key_candidates=SELECTED_SITE_KEYS,
    )

    label_handle = None
    writer = None
    if config.streaming_output:
        label_handle = labels_path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(label_handle, fieldnames=LABEL_FIELDNAMES, delimiter="\t", lineterminator="\n")
        writer.writeheader()
    try:
        for split_row in split_rows:
            row = _merge_feature_context(split_row, feature_lookup)
            try:
                tensor_path: Optional[Path] = None
                taxa_order: List[str] = []
                n_taxa = _safe_int(row.get("n_taxa"), -1)
                n_codons = _safe_int(row.get("n_codons"), -1)
                labels_file = _resolve_labels_path_from_row(row, dataset_dir)
                needs_tensor_metadata = (
                    n_taxa <= 0
                    or n_codons <= 0
                    or labels_file is None
                    or config.truth_mode not in {"explicit", "required"}
                )
                if needs_tensor_metadata:
                    tensor_path = resolve_tensor_file(row.get("tensor_file", ""), dataset_dir)
                    n_taxa, n_codons, _n_channels = _load_tensor_shape(tensor_path)
                    taxa_order = _tensor_taxa_order(tensor_path)
                    labels_file = _resolve_labels_path(row, dataset_dir, tensor_path)
            except (OSError, ValueError, FileNotFoundError) as exc:
                warnings.append(f"tensor_unresolved:{row.get('family_id', '')}:{row.get('method', '')}:{exc}")
                continue
            labels = _load_optional_json(labels_file, warnings) or {}
            gene_label = _first_nonempty(
                row.get("gene_label"),
                labels.get("gene_label"),
                (labels.get("labels") or {}).get("gene_label") if isinstance(labels.get("labels"), dict) else "",
            )
            explicit_truth = None
            if config.truth_mode != "proxy":
                explicit_truth = _load_explicit_branch_truth(
                    labels=labels,
                    labels_file=labels_file,
                    row=row,
                    cache=branch_truth_cache,
                    warnings=warnings,
                )
            if config.truth_mode in {"explicit", "required"} and explicit_truth is None:
                raise ValueError(
                    "explicit branch-site truth is required but unavailable for "
                    f"{row.get('family_id', '')}:{row.get('method', '')}"
                )
            if explicit_truth is not None:
                explicit_branch_truth_available = True
                selected_set = set(explicit_truth["selected_sites"])
                branch_status = "explicit_simulator_branch_truth"
                branch_source = "explicit_simulator_branch_truth"
                status_counts[branch_status] += 1
                source_counts[branch_source] += 1
                branch_ids = explicit_truth["branch_ids"]
                site_rows = _mapped_site_iter(
                    row=row,
                    n_codons=n_codons,
                    selected_set=selected_set,
                    config=site_config,
                    dataset_dir=dataset_dir,
                    site_map_cache=site_map_cache,
                    warnings=warnings,
                )
                foreground_taxon = _first_nonempty(explicit_truth.get("foreground_taxon"), row.get("foreground_taxon"), labels.get("foreground_taxon"))
                for branch_id in branch_ids:
                    selected_by_site = explicit_truth["selected_by_branch"].get(branch_id, {})
                    foreground_branch = int(bool(selected_by_site))
                    for site_payload in site_rows:
                        truth_site_index = _truth_site_index(site_payload)
                        y_branch_site = int(truth_site_index in selected_by_site)
                        y_site = int(truth_site_index in selected_set)
                        output_row = _output_row(
                            row=row,
                            branch_id=branch_id,
                            foreground_taxon=foreground_taxon,
                            site_payload=site_payload,
                            y_branch_site=y_branch_site,
                            y_site=y_site,
                            gene_label=gene_label,
                            foreground_branch=foreground_branch,
                            branch_source=branch_source,
                            labels_file=labels_file,
                            n_taxa=n_taxa,
                            n_codons=n_codons,
                        )
                        if writer is not None:
                            writer.writerow(output_row)
                        else:
                            rows_out.append(output_row)
                        row_count += 1
                        n_explicit_branch_truth_rows += 1
                        positives += y_branch_site
                        branch_positive_rows += foreground_branch
                        split_counts[output_row["split"]] += 1
                        tier_counts[output_row["saturation_tier"]] += 1
                        method_counts[output_row["method"]] += 1
                continue

            if config.truth_mode == "auto":
                warnings.append(
                    f"explicit_branch_truth_unavailable_falling_back_to_proxy:{row.get('family_id', '')}:{row.get('method', '')}"
                )
            selected_sites, selected_source, selected_warnings = _selected_sites(labels, n_codons)
            warnings.extend(
                f"{row.get('family_id', '')}:{row.get('method', '')}:{warning}"
                for warning in selected_warnings
            )
            selected_set = set(selected_sites)
            branch_labels, branch_status, branch_source = _branch_labels(labels, row, taxa_order)
            status_counts[branch_status] += 1
            source_counts[branch_source] += 1
            if branch_status == "not_available":
                warnings.append(f"branch_truth_not_available:{row.get('family_id', '')}:{row.get('method', '')}")
                continue
            proxy_labels_used = True
            site_rows = _mapped_site_iter(
                row=row,
                n_codons=n_codons,
                selected_set=selected_set,
                config=site_config,
                dataset_dir=dataset_dir,
                site_map_cache=site_map_cache,
                warnings=warnings,
            )
            foreground_taxon = _first_nonempty(row.get("foreground_taxon"), labels.get("foreground_taxon"))
            for branch_id, branch_label in sorted(branch_labels.items()):
                foreground_branch = int(branch_label == 1)
                for site_payload in site_rows:
                    y_site = int(site_payload.get("y_site") or 0)
                    y_branch_site = int(foreground_branch and y_site)
                    output_row = _output_row(
                        row=row,
                        branch_id=branch_id,
                        foreground_taxon=foreground_taxon,
                        site_payload=site_payload,
                        y_branch_site=y_branch_site,
                        y_site=y_site,
                        gene_label=gene_label,
                        foreground_branch=foreground_branch,
                        branch_source=branch_source if selected_source else f"{branch_source}:missing_selected_sites",
                        labels_file=labels_file,
                        n_taxa=n_taxa,
                        n_codons=n_codons,
                    )
                    if writer is not None:
                        writer.writerow(output_row)
                    else:
                        rows_out.append(output_row)
                    row_count += 1
                    n_proxy_rows += 1
                    positives += y_branch_site
                    branch_positive_rows += foreground_branch
                    split_counts[output_row["split"]] += 1
                    tier_counts[output_row["saturation_tier"]] += 1
                    method_counts[output_row["method"]] += 1
    finally:
        if label_handle is not None:
            label_handle.close()

    if not config.streaming_output:
        rows_out.sort(
            key=lambda r: (
                r.get("family_id", ""),
                r.get("method", ""),
                r.get("branch_id", ""),
                _safe_int(r.get("site_index_zero"), 0),
            )
        )
        write_tsv(labels_path, rows_out, LABEL_FIELDNAMES)
        row_count = len(rows_out)
        positives = sum(int(row["y_branch_site"]) for row in rows_out)
        branch_positive_rows = sum(int(row["foreground_branch_present"]) for row in rows_out)
        split_counts = Counter(row["split"] for row in rows_out)
        tier_counts = Counter(row["saturation_tier"] for row in rows_out)
        method_counts = Counter(row["method"] for row in rows_out)
        n_explicit_branch_truth_rows = sum(
            1 for row in rows_out if row.get("branch_label_source") == "explicit_simulator_branch_truth"
        )
        n_proxy_rows = len(rows_out) - n_explicit_branch_truth_rows
        explicit_branch_truth_available = n_explicit_branch_truth_rows > 0
        proxy_labels_used = n_proxy_rows > 0
    branch_site_labels_status = _overall_status(status_counts)
    payload = {
        "branch_site_oracle_version": BRANCH_SITE_ORACLE_VERSION,
        "dataset_dir": str(dataset_dir),
        "site_map_dir": str(config.site_map_dir or ""),
        "aligned_site_mode": config.aligned_site_mode,
        "foreground_source": config.foreground_source,
        "truth_mode": config.truth_mode,
        "streaming_output": config.streaming_output,
        "branch_site_labels_status": branch_site_labels_status,
        "explicit_branch_site_truth_available": explicit_branch_truth_available,
        "proxy_labels_used": proxy_labels_used,
        "n_explicit_branch_truth_rows": n_explicit_branch_truth_rows,
        "n_proxy_rows": n_proxy_rows,
        "n_input_family_method_rows": len(split_rows),
        "n_branch_site_rows": row_count,
        "n_positive_branch_sites": positives,
        "positive_branch_site_fraction": None if row_count == 0 else positives / row_count,
        "n_foreground_branch_rows": branch_positive_rows,
        "status_counts": dict(sorted(status_counts.items())),
        "branch_label_source_counts": dict(sorted(source_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "saturation_tier_counts": dict(sorted(tier_counts.items())),
        "method_counts": dict(sorted(method_counts.items())),
        "warnings": sorted(set(warnings)),
        "generated_files": {
            "labels_tsv": str(labels_path),
            "summary_json": str(summary_path),
            "markdown": str(markdown_path),
        },
        "interpretation": (
            "y_branch_site is a branch-specific supervised target. Explicit simulator "
            "branch-site truth is preferred when available; proxy foreground-conditioned "
            "labels are fallback only."
        ),
    }
    _write_json(summary_path, payload)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "labels_tsv": str(labels_path),
        "branch_site_labels_tsv": str(labels_path),
        "summary_json": str(summary_path),
        "markdown": str(markdown_path),
        "branch_site_labels_status": branch_site_labels_status,
        "explicit_branch_site_truth_available": explicit_branch_truth_available,
        "proxy_labels_used": proxy_labels_used,
        "n_branch_site_rows": row_count,
        "n_positive_branch_sites": positives,
        "warnings": payload["warnings"],
    }


def validate_branch_site_label_dir(label_dir: str | Path) -> dict:
    """Validate branch-site oracle label artifacts."""
    path = Path(label_dir)
    failures: List[str] = []
    warnings: List[str] = []
    required = {
        "summary": path / "branch_site_oracle_summary.json",
        "labels": path / "branch_site_oracle_labels.tsv",
        "markdown": path / "branch_site_oracle_labels.md",
    }
    for label, file_path in required.items():
        if not file_path.exists():
            failures.append(f"missing_{label}:{file_path}")
        elif file_path.stat().st_size == 0:
            failures.append(f"empty_{label}:{file_path}")
    summary = _load_json(required["summary"], failures)
    n_rows = 0
    n_positive = 0
    if required["labels"].exists():
        with required["labels"].open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            missing = sorted(set(LABEL_FIELDNAMES[:17]) - set(reader.fieldnames or []))
            if missing:
                failures.append("missing_required_columns:" + ",".join(missing))
            for row in reader:
                n_rows += 1
                if row.get("y_branch_site") not in {"0", "1"}:
                    failures.append(f"invalid_y_branch_site:{row.get('family_id')}:{row.get('y_branch_site')}")
                if row.get("y_site") not in {"0", "1"}:
                    failures.append(f"invalid_y_site:{row.get('family_id')}:{row.get('y_site')}")
                if row.get("y_branch_site") == "1":
                    n_positive += 1
                if not row.get("branch_label_source"):
                    warnings.append("missing_branch_label_source")
    if n_rows == 0:
        failures.append("no_branch_site_rows")
    if summary and int(summary.get("n_branch_site_rows") or -1) != n_rows:
        warnings.append("summary_row_count_mismatch")
    if summary and summary.get("branch_site_labels_status") == "not_available":
        warnings.append("branch_site_labels_not_available")
    if summary and summary.get("proxy_labels_used"):
        warnings.append("proxy_labels_used")
    return {
        "status": "fail" if failures else "ok",
        "n_rows": n_rows,
        "n_positive_branch_sites": n_positive,
        "n_fail": len(failures),
        "n_warning": len(set(warnings)),
        "failures": failures,
        "warnings": sorted(set(warnings)),
    }


def _selected_sites(labels: dict, n_codons: int) -> Tuple[List[int], str, List[str]]:
    warnings: List[str] = []
    for key in SELECTED_SITE_KEYS:
        if key not in labels:
            continue
        base = "auto"
        if key.endswith("_0based"):
            base = "zero"
        elif key.endswith("_1based"):
            base = "one"
        indices, index_warnings = normalize_site_indices(labels.get(key), n_codons, base)
        warnings.extend(f"{key}:{warning}" for warning in index_warnings)
        return indices, key, warnings
    return [], "", ["missing_selected_sites"]


def _branch_labels(labels: dict, row: dict, taxa_order: List[str]) -> Tuple[Dict[str, int], str, str]:
    nested = labels.get("labels") if isinstance(labels.get("labels"), dict) else {}
    raw = labels.get("branch_labels") or nested.get("branch_labels")
    if isinstance(raw, dict) and raw:
        parsed = {str(branch): _safe_binary(value) for branch, value in raw.items()}
        return parsed, "proxy_from_foreground_taxon", "proxy_from_foreground_taxon:branch_labels_x_selected_sites"
    foreground = _first_nonempty(row.get("foreground_taxon"), labels.get("foreground_taxon"))
    if foreground and taxa_order:
        return {taxon: int(taxon == foreground) for taxon in taxa_order}, "proxy_from_foreground_taxon", "proxy_from_foreground_taxon:foreground_taxon_x_selected_sites"
    if taxa_order and str(row.get("gene_label", labels.get("gene_label", "0"))) in {"0", "0.0", ""}:
        return {taxon: 0 for taxon in taxa_order}, "proxy_from_foreground_taxon", "proxy_from_foreground_taxon:null_all_branches"
    return {}, "not_available", "not_available"


def _load_explicit_branch_truth(
    labels: dict,
    labels_file: Optional[Path],
    row: dict,
    cache: Dict[str, Optional[dict]],
    warnings: List[str],
) -> Optional[dict]:
    branch_truth_path = _resolve_branch_truth_path(labels, labels_file)
    if branch_truth_path is None:
        return None
    cache_key = str(branch_truth_path)
    if cache_key not in cache:
        try:
            cache[cache_key] = _load_json(branch_truth_path, warnings)
        except Exception:
            cache[cache_key] = None
    branch_truth = cache[cache_key]
    if not isinstance(branch_truth, dict):
        return None
    if branch_truth.get("truth_source") != "explicit_simulator_branch_truth":
        warnings.append(f"branch_truth_not_explicit:{row.get('family_id', '')}:{branch_truth_path}")
        return None
    records = branch_truth.get("branch_site_records")
    if not isinstance(records, list):
        warnings.append(f"branch_truth_records_missing:{row.get('family_id', '')}:{branch_truth_path}")
        return None
    selected_by_branch: Dict[str, Dict[int, str]] = {}
    branch_ids = set()
    selected_sites = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        branch_id = str(record.get("branch_id", ""))
        if not branch_id:
            continue
        branch_ids.add(branch_id)
        if _safe_binary(record.get("y_branch_site", 0)):
            site_index = _safe_int(record.get("site_index_zero"), -1)
            if site_index >= 0:
                selected_by_branch.setdefault(branch_id, {})[site_index] = str(
                    record.get("selection_event_id") or record.get("event_id", "")
                )
                selected_sites.add(site_index)
    foreground_taxon = _first_nonempty(
        *(branch.get("foreground_taxon") for branch in branch_truth.get("foreground_branches", []) if isinstance(branch, dict))
    )
    return {
        "branch_truth_path": str(branch_truth_path),
        "branch_ids": sorted(branch_ids),
        "selected_by_branch": selected_by_branch,
        "selected_sites": sorted(selected_sites),
        "foreground_taxon": foreground_taxon,
        "n_source_rows": len(records),
    }


def _resolve_branch_truth_path(labels: dict, labels_file: Optional[Path]) -> Optional[Path]:
    value = labels.get("branch_truth_file") if isinstance(labels, dict) else None
    candidates = []
    if value not in ("", None):
        raw = Path(str(value))
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend([
                Path.cwd() / raw,
                labels_file.parent / raw if labels_file is not None else raw,
            ])
    if labels_file is not None:
        family_id = str(labels.get("family_id") or labels_file.name.replace(".labels.json", ""))
        candidates.append(labels_file.with_name(f"{family_id}.branch_truth.json"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_labels_path_from_row(row: dict, dataset_dir: Path) -> Optional[Path]:
    value = row.get("labels_file", "")
    if value in ("", None):
        return None
    raw = Path(str(value))
    candidates = [raw] if raw.is_absolute() else [
        Path.cwd() / raw,
        dataset_dir / raw,
        dataset_dir.parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _output_row(
    row: dict,
    branch_id: str,
    foreground_taxon: str,
    site_payload: dict,
    y_branch_site: int,
    y_site: int,
    gene_label: object,
    foreground_branch: int,
    branch_source: str,
    labels_file: Optional[Path],
    n_taxa: int,
    n_codons: int,
) -> dict:
    return {
        "family_id": row.get("family_id", ""),
        "method": row.get("method", ""),
        "split": row.get("split", ""),
        "saturation_tier": row.get("saturation_tier", "unknown") or "unknown",
        "branch_id": branch_id,
        "foreground_taxon": foreground_taxon,
        "site_index_zero": site_payload.get("site_index_zero", ""),
        "aligned_site_index_zero": site_payload.get("aligned_site_index_zero", ""),
        "original_site_index_zero": site_payload.get("original_site_index_zero", ""),
        "y_branch_site": y_branch_site,
        "y_site": y_site,
        "gene_label": gene_label,
        "foreground_branch_present": foreground_branch,
        "branch_label_source": branch_source,
        "mapping_status": site_payload.get("mapping_status", ""),
        "mapping_confidence": site_payload.get("mapping_confidence", ""),
        "mappable_site": site_payload.get("mappable_site", ""),
        "original_family_id": row.get("original_family_id", ""),
        "source_dataset": row.get("source_dataset", ""),
        "tensor_file": row.get("tensor_file", ""),
        "labels_file": str(labels_file) if labels_file is not None else "",
        "n_taxa": n_taxa,
        "n_codons": n_codons,
    }


def _truth_site_index(site_payload: dict) -> int:
    original = site_payload.get("original_site_index_zero", "")
    if original not in ("", None):
        return _safe_int(original, -1)
    return _safe_int(site_payload.get("site_index_zero"), -1)


def _tensor_taxa_order(tensor_path: Path) -> List[str]:
    try:
        with np.load(tensor_path, allow_pickle=False) as shard:
            if "taxa_order" in shard.files:
                return [str(value) for value in shard["taxa_order"].tolist()]
    except OSError:
        return []
    return []


def _overall_status(counts: Counter[str]) -> str:
    if not counts:
        return "not_available"
    explicit = counts.get("explicit_simulator_branch_truth", 0)
    proxy = counts.get("proxy_from_foreground_taxon", 0)
    if explicit and proxy:
        return "mixed_explicit_and_proxy"
    if explicit and explicit == sum(counts.values()):
        return "explicit_simulator_branch_truth"
    if counts.get("not_available") and counts.get("not_available") == sum(counts.values()):
        return "not_available"
    if counts.get("not_available"):
        return "mixed_proxy_and_missing"
    return "proxy_from_foreground_taxon"


def _safe_binary(value: object) -> int:
    return 1 if str(value).strip().lower() in {"1", "1.0", "true", "yes"} else 0


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _load_json(path: Path, failures: List[str]) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"could_not_parse_json:{path}:{exc}")
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_markdown(payload: dict) -> str:
    warnings = payload.get("warnings") or []
    lines = [
        "# Branch-site oracle labels",
        "",
        f"- Dataset: `{payload.get('dataset_dir')}`",
        f"- Site-map dir: `{payload.get('site_map_dir')}`",
        f"- Label status: `{payload.get('branch_site_labels_status')}`",
        f"- Branch-site rows: {payload.get('n_branch_site_rows')}",
        f"- Positive branch-sites: {payload.get('n_positive_branch_sites')}",
        "",
        "## Interpretation",
        "",
        payload.get("interpretation", ""),
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- none"])
    lines.append("")
    return "\n".join(lines)
