"""Branch-site feature policy definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from babappa.branch.context_ablation import (
    CONTEXT_ONLY_COLUMNS,
    FOREGROUND_ALL_COLUMNS,
    FOREGROUND_CODON_CONTEXT_COLUMNS,
    FOREGROUND_IDENTITY_COLUMNS,
)


@dataclass(frozen=True)
class BranchFeaturePolicy:
    """Named branch-site feature policy metadata."""

    name: str
    label: str
    recommended_role: str
    production_default: bool
    excluded_columns: List[str]
    included_columns: List[str]
    warning: str = ""


POLICIES: Dict[str, BranchFeaturePolicy] = {
    "full_context": BranchFeaturePolicy(
        name="full_context",
        label="context-aware upper-bound",
        recommended_role="diagnostic upper-bound; do not use as main conservative branch-site claim",
        production_default=False,
        excluded_columns=[],
        included_columns=["all_available_branch_site_features"],
        warning="Foreground/context features are highly predictive; treat as an upper-bound.",
    ),
    "no_foreground_identity": BranchFeaturePolicy(
        name="no_foreground_identity",
        label="identity-context restricted",
        recommended_role="conservative validation candidate",
        production_default=False,
        excluded_columns=list(FOREGROUND_IDENTITY_COLUMNS),
        included_columns=["all_available_features_except_foreground_identity"],
    ),
    "no_foreground_codon_context": BranchFeaturePolicy(
        name="no_foreground_codon_context",
        label="codon-context restricted",
        recommended_role="diagnostic; codon-context removal alone was insufficient in explicit 1K ablation",
        production_default=False,
        excluded_columns=list(FOREGROUND_CODON_CONTEXT_COLUMNS),
        included_columns=["all_available_features_except_foreground_codon_context"],
    ),
    "no_foreground_all": BranchFeaturePolicy(
        name="no_foreground_all",
        label="all-foreground-context restricted",
        recommended_role="strict diagnostic conservative profile",
        production_default=False,
        excluded_columns=list(FOREGROUND_ALL_COLUMNS),
        included_columns=["all_available_features_except_foreground_context"],
    ),
    "context_only": BranchFeaturePolicy(
        name="context_only",
        label="shortcut-detection diagnostic",
        recommended_role="diagnostic shortcut-detection profile; never production default",
        production_default=False,
        excluded_columns=["all_non_context_features"],
        included_columns=list(CONTEXT_ONLY_COLUMNS),
        warning="Diagnostic only. High context_only performance indicates shortcut risk.",
    ),
    "conservative_branch_site": BranchFeaturePolicy(
        name="conservative_branch_site",
        label="conservative branch-site default",
        recommended_role="recommended default for scientific validation after explicit 1K ablation",
        production_default=True,
        excluded_columns=list(FOREGROUND_IDENTITY_COLUMNS),
        included_columns=["all_available_features_except_foreground_identity"],
        warning=(
            "Uses no_foreground_identity at minimum. Treat full_context as optional upper-bound only."
        ),
    ),
}


def list_branch_feature_policies() -> List[Dict[str, object]]:
    """Return branch feature policies as serializable rows."""

    rows = []
    for name in sorted(POLICIES):
        policy = POLICIES[name]
        rows.append(
            {
                "policy": policy.name,
                "label": policy.label,
                "included_columns": ",".join(policy.included_columns),
                "excluded_columns": ",".join(policy.excluded_columns),
                "recommended_role": policy.recommended_role,
                "production_default": policy.production_default,
                "warning": policy.warning,
            }
        )
    return rows


def get_branch_feature_policy(name: str) -> BranchFeaturePolicy:
    """Return one named branch feature policy."""

    try:
        return POLICIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown branch feature policy: {name}") from exc


def columns_for_policy(feature_columns: Sequence[str], policy_name: str) -> List[str]:
    """Apply a named branch feature policy to available feature columns."""

    if policy_name == "full_model":
        policy_name = "full_context"
    policy = get_branch_feature_policy(policy_name)
    columns = list(feature_columns)
    if policy.name == "full_context":
        return columns
    if policy.name == "context_only":
        return [column for column in columns if column in CONTEXT_ONLY_COLUMNS]
    excluded = set(policy.excluded_columns)
    return [column for column in columns if column not in excluded]
