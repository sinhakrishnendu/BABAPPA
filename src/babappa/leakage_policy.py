"""Shared leakage-column policy for BABAPPA feature-based models."""

from __future__ import annotations

STRICT_LEAKAGE_COLUMNS = {
    "gene_label",
    "selected_sites",
    "n_selected_sites",
    "positive_sites",
    "selected_site_count",
    "foreground_taxon",
    "true_label",
    "truth_label",
    "positive_family",
    "is_positive",
    "label",
    "y",
}

METADATA_NOT_FEATURE_COLUMNS = {
    "family_id",
    "original_family_id",
    "merged_family_id",
    "source_dataset",
    "method",
    "tensor_file",
    "tensor_meta_file",
    "labels_file",
    "split",
    "saturation_tier",
}

SUSPICIOUS_NAME_FRAGMENTS = (
    "selected",
    "truth",
    "label",
    "positive",
    "foreground",
)


def normalized_column_name(column: str) -> str:
    """Normalize a column name for leakage policy matching."""
    return str(column).strip().lower()


def is_metadata_column(column: str) -> bool:
    """Return whether a column is metadata rather than model feature input."""
    return normalized_column_name(column) in METADATA_NOT_FEATURE_COLUMNS


def is_strict_leakage_column(column: str) -> bool:
    """Return whether a column is explicitly truth-derived or label-like."""
    return normalized_column_name(column) in STRICT_LEAKAGE_COLUMNS


def is_suspicious_feature_name(column: str) -> bool:
    """Return whether a column name suggests possible leakage."""
    normalized = normalized_column_name(column)
    return any(fragment in normalized for fragment in SUSPICIOUS_NAME_FRAGMENTS)


def should_exclude_from_feature_model(column: str) -> bool:
    """Return whether feature-based models should exclude a column."""
    return (
        is_metadata_column(column)
        or is_strict_leakage_column(column)
        or is_suspicious_feature_name(column)
    )
