"""Extract oracle site-level supervised targets from BABAPPA tensor datasets."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv
from babappa.training.neural_data import resolve_tensor_file

ORACLE_SITE_LABEL_VERSION = __version__
SITE_INDEX_BASES = {"auto", "zero", "one"}
ALIGNED_SITE_MODES = {"original", "mapped"}
DEFAULT_SELECTED_SITE_KEYS = [
    "selected_sites",
    "positive_sites",
    "oracle_selected_sites",
    "site_labels",
    "selected_site_indices",
    "selected_sites_0based",
    "selected_sites_1based",
]
SITE_LABEL_FIELDNAMES = [
    "family_id",
    "original_family_id",
    "source_dataset",
    "method",
    "split",
    "saturation_tier",
    "tensor_file",
    "labels_file",
    "tensor_meta_file",
    "n_taxa",
    "n_codons",
    "site_index_zero",
    "site_index_one",
    "aligned_site_index_zero",
    "aligned_site_index_one",
    "original_site_index_zero",
    "original_site_index_one",
    "mapping_status",
    "mapping_confidence",
    "mappable_site",
    "y_site",
    "foreground_taxon",
    "oracle_label_source",
]


@dataclass(frozen=True)
class OracleSiteLabelConfig:
    """Configuration for extracting site-level oracle labels."""

    dataset_dir: str
    outdir: str
    selected_sites_key_candidates: Optional[List[str]] = None
    site_index_base: str = "auto"
    site_map_dir: Optional[str] = None
    aligned_site_mode: str = "mapped"

    def __post_init__(self) -> None:
        dataset_path = Path(self.dataset_dir)
        out_path = Path(self.outdir)
        if not dataset_path.exists():
            raise ValueError(f"dataset_dir does not exist: {dataset_path}")
        for filename in ("dataset_index.json", "splits.tsv"):
            if not (dataset_path / filename).exists():
                raise ValueError(f"dataset_dir is missing {filename}: {dataset_path}")
        if self.site_index_base not in SITE_INDEX_BASES:
            allowed = ", ".join(sorted(SITE_INDEX_BASES))
            raise ValueError(f"site_index_base must be one of: {allowed}")
        if self.aligned_site_mode not in ALIGNED_SITE_MODES:
            allowed = ", ".join(sorted(ALIGNED_SITE_MODES))
            raise ValueError(f"aligned_site_mode must be one of: {allowed}")
        if self.site_map_dir is not None and not Path(self.site_map_dir).exists():
            raise ValueError(f"site_map_dir does not exist: {self.site_map_dir}")
        candidates = self.selected_sites_key_candidates or DEFAULT_SELECTED_SITE_KEYS
        if not candidates:
            raise ValueError("selected_sites_key_candidates must be non-empty")
        object.__setattr__(self, "selected_sites_key_candidates", list(candidates))
        out_path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    """Load a JSON object."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def normalize_site_indices(
    values: object, n_codons: int, site_index_base: str
) -> Tuple[List[int], List[str]]:
    """Normalize selected-site values into sorted zero-based codon indices."""
    if site_index_base not in SITE_INDEX_BASES:
        raise ValueError("site_index_base must be auto, zero, or one")
    warnings: List[str] = []
    raw_values = _coerce_site_values(values)
    if not raw_values:
        return [], warnings

    if len(raw_values) == n_codons and _looks_like_indicator(raw_values):
        indices = [index for index, value in enumerate(raw_values) if _truthy_indicator(value)]
        return sorted(set(indices)), warnings

    parsed: List[int] = []
    for value in raw_values:
        try:
            parsed.append(int(float(str(value).strip())))
        except (TypeError, ValueError):
            warnings.append(f"non_integer_site_index_ignored:{value}")

    if not parsed:
        return [], warnings

    resolved_base = site_index_base
    if resolved_base == "auto":
        if any(index == 0 for index in parsed):
            resolved_base = "zero"
        elif max(parsed) == n_codons:
            resolved_base = "one"
        else:
            resolved_base = "zero"
            warnings.append("ambiguous_site_index_base_assumed_zero")

    if resolved_base == "one":
        parsed = [index - 1 for index in parsed]

    normalized = []
    for index in parsed:
        if 0 <= index < n_codons:
            normalized.append(index)
        else:
            warnings.append(f"out_of_range_site_index_ignored:{index}")
    return sorted(set(normalized)), warnings


def extract_oracle_site_labels(config: OracleSiteLabelConfig) -> dict:
    """Extract per-site oracle labels for every family/method split row."""
    dataset_dir = Path(config.dataset_dir)
    outdir = Path(config.outdir)
    site_labels_path = outdir / "site_oracle_labels.tsv"
    summary_path = outdir / "site_oracle_summary.json"
    markdown_path = outdir / "site_oracle_labels.md"
    warnings: List[str] = []

    split_rows = read_tsv(dataset_dir / "splits.tsv")
    feature_lookup = _feature_lookup(dataset_dir)
    output_rows: List[dict] = []
    family_method_records = 0
    site_map_cache: Dict[Tuple[str, str], List[dict]] = {}

    for split_row in split_rows:
        row = _merge_feature_context(split_row, feature_lookup)
        try:
            tensor_path = resolve_tensor_file(row.get("tensor_file", ""), dataset_dir)
        except (OSError, FileNotFoundError) as exc:
            warnings.append(f"tensor_file_unresolved:{row.get('tensor_file', '')}:{exc}")
            continue
        tensor_shape = _load_tensor_shape(tensor_path)
        n_taxa, n_codons = int(tensor_shape[0]), int(tensor_shape[1])
        labels_path = _resolve_labels_path(row, dataset_dir, tensor_path)
        meta_path = _resolve_meta_path(row, dataset_dir, tensor_path)
        labels = _load_optional_json(labels_path, warnings)
        meta = _load_optional_json(meta_path, warnings)
        selected_sites, selected_source, selected_warnings = _extract_selected_sites(
            labels=labels,
            n_codons=n_codons,
            config=config,
        )
        warnings.extend(
            f"{row.get('family_id', '')}:{row.get('method', '')}:{warning}"
            for warning in selected_warnings
        )
        if labels_path is None or labels is None:
            warnings.append(
                f"missing_labels_file:{row.get('family_id', '')}:{row.get('method', '')}"
            )
            selected_source = "empty_missing_labels"
        elif not selected_source:
            warnings.append(
                f"missing_selected_sites:{row.get('family_id', '')}:{row.get('method', '')}"
            )
            selected_source = "empty_missing_selected_sites"
        selected_set = set(selected_sites)
        foreground_taxon = _first_nonempty(
            row.get("foreground_taxon"),
            (labels or {}).get("foreground_taxon"),
            (meta or {}).get("foreground_taxon"),
        )
        family_method_records += 1
        site_iter = _mapped_site_iter(
            row=row,
            n_codons=n_codons,
            selected_set=selected_set,
            config=config,
            dataset_dir=dataset_dir,
            site_map_cache=site_map_cache,
            warnings=warnings,
        )
        for site_payload in site_iter:
            site_index = site_payload["site_index_zero"]
            output_rows.append(
                {
                    "family_id": row.get("family_id", ""),
                    "original_family_id": row.get("original_family_id", ""),
                    "source_dataset": row.get("source_dataset", ""),
                    "method": row.get("method", ""),
                    "split": row.get("split", ""),
                    "saturation_tier": row.get("saturation_tier", "unknown") or "unknown",
                    "tensor_file": row.get("tensor_file", ""),
                    "labels_file": _display_path(labels_path, row.get("labels_file", "")),
                    "tensor_meta_file": _display_path(
                        meta_path, row.get("tensor_meta_file", "")
                    ),
                    "n_taxa": n_taxa,
                    "n_codons": n_codons,
                    "site_index_zero": site_index,
                    "site_index_one": site_index + 1,
                    "aligned_site_index_zero": site_payload.get("aligned_site_index_zero", ""),
                    "aligned_site_index_one": site_payload.get("aligned_site_index_one", ""),
                    "original_site_index_zero": site_payload.get("original_site_index_zero", ""),
                    "original_site_index_one": site_payload.get("original_site_index_one", ""),
                    "mapping_status": site_payload.get("mapping_status", ""),
                    "mapping_confidence": site_payload.get("mapping_confidence", ""),
                    "mappable_site": site_payload.get("mappable_site", ""),
                    "y_site": site_payload["y_site"],
                    "foreground_taxon": foreground_taxon,
                    "oracle_label_source": selected_source,
                }
            )

    write_tsv(site_labels_path, output_rows, SITE_LABEL_FIELDNAMES)
    n_positive = sum(int(row["y_site"]) for row in output_rows)
    split_counts = Counter(row["split"] for row in output_rows)
    tier_counts = Counter(row["saturation_tier"] for row in output_rows)
    positive_by_split = Counter(
        row["split"] for row in output_rows if int(row["y_site"]) == 1
    )
    payload = {
        "oracle_site_label_version": ORACLE_SITE_LABEL_VERSION,
        "dataset_dir": str(dataset_dir),
        "n_rows": len(split_rows),
        "n_family_method_records": family_method_records,
        "n_site_records": len(output_rows),
        "n_positive_sites": n_positive,
        "positive_site_fraction": (
            None if not output_rows else n_positive / len(output_rows)
        ),
        "split_counts": dict(sorted(split_counts.items())),
        "saturation_tier_counts": dict(sorted(tier_counts.items())),
        "positive_counts_by_split": dict(sorted(positive_by_split.items())),
        "warnings": sorted(set(warnings)),
        "generated_files": {
            "site_labels_tsv": str(site_labels_path),
            "summary_json": str(summary_path),
            "markdown": str(markdown_path),
        },
        "note": "Oracle site labels are supervised targets only and must not be used as input features.",
    }
    _write_json(summary_path, payload)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "site_labels_tsv": str(site_labels_path),
        "summary_json": str(summary_path),
        "markdown": str(markdown_path),
        "warnings": payload["warnings"],
    }


def _mapped_site_iter(
    row: dict,
    n_codons: int,
    selected_set: set[int],
    config: OracleSiteLabelConfig,
    dataset_dir: Path,
    site_map_cache: Dict[Tuple[str, str], List[dict]],
    warnings: List[str],
) -> List[dict]:
    if not config.site_map_dir or config.aligned_site_mode == "original":
        return [
            {
                "site_index_zero": site_index,
                "y_site": 1 if site_index in selected_set else 0,
            }
            for site_index in range(n_codons)
        ]
    map_rows = _load_site_map_rows(row, dataset_dir, Path(config.site_map_dir), site_map_cache, warnings)
    if not map_rows:
        warnings.append(f"missing_site_map_rows:{row.get('family_id', '')}:{row.get('method', '')}")
        return []
    mapped: List[dict] = []
    for map_row in map_rows:
        aligned_index = _safe_int(map_row.get("aligned_site_index_zero"), default=-1)
        if aligned_index < 0:
            continue
        original_raw = map_row.get("original_site_index_zero", "")
        original_index = _safe_int(original_raw, default=-1) if original_raw not in ("", None) else -1
        status = map_row.get("mapping_status", "")
        confidence = _safe_float(map_row.get("mapping_confidence"), default=0.0)
        mappable = 1 if status == "unique" and original_index >= 0 else 0
        mapped.append(
            {
                "site_index_zero": aligned_index,
                "aligned_site_index_zero": aligned_index,
                "aligned_site_index_one": aligned_index + 1,
                "original_site_index_zero": "" if original_index < 0 else original_index,
                "original_site_index_one": "" if original_index < 0 else original_index + 1,
                "mapping_status": status,
                "mapping_confidence": confidence,
                "mappable_site": mappable,
                "y_site": 1 if mappable and original_index in selected_set else 0,
            }
        )
    return mapped


def _load_site_map_rows(
    row: dict,
    dataset_dir: Path,
    site_map_dir: Path,
    cache: Dict[Tuple[str, str], List[dict]],
    warnings: List[str],
) -> List[dict]:
    method = row.get("method", "")
    family_candidates = [
        row.get("family_id", ""),
        row.get("original_family_id", ""),
    ]
    for family_id in [candidate for candidate in family_candidates if candidate]:
        key = (family_id, method)
        if key in cache:
            return cache[key]
        map_path = _site_map_path(site_map_dir, family_id, method)
        if map_path is not None:
            rows = read_tsv(map_path)
            cache[key] = rows
            return rows
    warnings.append(
        f"site_map_not_found:{row.get('family_id', '')}:{row.get('original_family_id', '')}:{method}"
    )
    return []


def _site_map_path(site_map_dir: Path, family_id: str, method: str) -> Optional[Path]:
    candidates = [
        site_map_dir / "families" / family_id / f"{family_id}.{method}.site_map.tsv",
    ]
    manifest_path = site_map_dir / "site_map_manifest.json"
    if manifest_path.exists():
        try:
            manifest = load_json(manifest_path)
            rel = manifest.get("map_files", {}).get(family_id, {}).get(method)
            if rel:
                candidates.insert(0, site_map_dir / rel)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _feature_lookup(dataset_dir: Path) -> Dict[Tuple[str, str, str], dict]:
    features_path = dataset_dir / "features.tsv"
    if not features_path.exists():
        return {}
    rows = read_tsv(features_path)
    lookup: Dict[Tuple[str, str, str], dict] = {}
    for row in rows:
        keys = [
            (row.get("family_id", ""), row.get("method", ""), row.get("tensor_file", "")),
            (row.get("family_id", ""), row.get("method", ""), ""),
        ]
        for key in keys:
            lookup[key] = row
    return lookup


def _merge_feature_context(split_row: dict, lookup: Dict[Tuple[str, str, str], dict]) -> dict:
    key = (
        split_row.get("family_id", ""),
        split_row.get("method", ""),
        split_row.get("tensor_file", ""),
    )
    feature_row = lookup.get(key) or lookup.get((key[0], key[1], "")) or {}
    merged = dict(feature_row)
    merged.update(split_row)
    for optional in ("original_family_id", "source_dataset", "labels_file", "tensor_meta_file"):
        if optional not in merged:
            merged[optional] = feature_row.get(optional, "")
    if not merged.get("foreground_taxon"):
        merged["foreground_taxon"] = feature_row.get("foreground_taxon", "")
    return merged


def _load_tensor_shape(tensor_path: Path) -> Tuple[int, int, int]:
    with np.load(tensor_path, allow_pickle=False) as shard:
        if "X" not in shard.files:
            raise ValueError(f"tensor shard missing X array: {tensor_path}")
        tensor = shard["X"]
        if tensor.ndim != 3:
            raise ValueError(f"X array is not 3-dimensional: {tensor_path}")
        return tuple(int(value) for value in tensor.shape)


def _resolve_labels_path(row: dict, dataset_dir: Path, tensor_path: Path) -> Optional[Path]:
    value = row.get("labels_file", "")
    resolved = _resolve_optional_path(value, dataset_dir, tensor_path)
    if resolved is not None:
        return resolved
    family_id = row.get("original_family_id") or row.get("family_id", "")
    candidate = tensor_path.parent / f"{family_id}.labels.json"
    if candidate.exists():
        return candidate
    return None


def _resolve_meta_path(row: dict, dataset_dir: Path, tensor_path: Path) -> Optional[Path]:
    value = row.get("tensor_meta_file", "")
    resolved = _resolve_optional_path(value, dataset_dir, tensor_path)
    if resolved is not None:
        return resolved
    candidate = tensor_path.with_name(tensor_path.name.replace(".tensor.npz", ".tensor_meta.json"))
    if candidate.exists():
        return candidate
    return None


def _resolve_optional_path(value: object, dataset_dir: Path, tensor_path: Path) -> Optional[Path]:
    if value in ("", None):
        return None
    raw = Path(str(value))
    candidates = [raw] if raw.is_absolute() else [
        Path.cwd() / raw,
        dataset_dir / raw,
        dataset_dir.parent / raw,
        tensor_path.parent / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_optional_json(path: Optional[Path], warnings: List[str]) -> Optional[dict]:
    if path is None:
        return None
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        warnings.append(f"could_not_load_json:{path}:{exc}")
        return None


def _display_path(path: Optional[Path], original_value: object) -> str:
    if original_value not in ("", None):
        return str(original_value)
    if path is None:
        return ""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _extract_selected_sites(
    labels: Optional[dict], n_codons: int, config: OracleSiteLabelConfig
) -> Tuple[List[int], str, List[str]]:
    if not labels:
        return [], "", []
    warnings: List[str] = []
    for key in config.selected_sites_key_candidates or []:
        if key not in labels:
            continue
        base = config.site_index_base
        if key.endswith("_0based"):
            base = "zero"
        elif key.endswith("_1based"):
            base = "one"
        indices, index_warnings = normalize_site_indices(labels.get(key), n_codons, base)
        warnings.extend(f"{key}:{warning}" for warning in index_warnings)
        return indices, key, warnings
    return [], "", warnings


def _coerce_site_values(values: object) -> List[object]:
    if values in ("", None):
        return []
    if isinstance(values, str):
        stripped = values.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return list(parsed)
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in stripped.split(",") if part.strip()]
    if isinstance(values, dict):
        return [key for key, value in values.items() if _truthy_indicator(value)]
    if isinstance(values, Iterable):
        return list(values)
    return [values]


def _looks_like_indicator(values: List[object]) -> bool:
    allowed = {"0", "1", "0.0", "1.0", "false", "true", "False", "True"}
    return all(str(value).strip() in allowed for value in values)


def _truthy_indicator(value: object) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _first_nonempty(*values: object) -> str:
    for value in values:
        if value not in ("", None):
            return str(value)
    return ""


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Oracle site-label extraction",
        "",
        "## Dataset",
        "",
        f"- Dataset directory: `{payload.get('dataset_dir')}`",
        f"- Family-method records: {payload.get('n_family_method_records')}",
        "",
        "## Site-label summary",
        "",
        f"- Site records: {payload.get('n_site_records')}",
        f"- Positive sites: {payload.get('n_positive_sites')}",
        f"- Positive-site fraction: {payload.get('positive_site_fraction')}",
        "",
        "## Positive-site distribution",
        "",
        f"- Positive counts by split: {payload.get('positive_counts_by_split')}",
        f"- Saturation tier counts: {payload.get('saturation_tier_counts')}",
        "",
        "## Warnings",
        "",
    ]
    warnings = payload.get("warnings") or []
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- none"])
    lines.extend(
        [
            "",
            "## Leakage note",
            "",
            "Oracle selected-site labels are supervised targets only and must not be used as input features.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
