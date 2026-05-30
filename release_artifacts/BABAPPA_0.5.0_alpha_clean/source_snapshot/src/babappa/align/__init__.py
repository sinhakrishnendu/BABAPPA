"""Alignment ensemble scaffold for BABAPPA."""

from babappa.align.audit import validate_alignment_directory
from babappa.align.backends import (
    AlignerBackend,
    babappalign_model_path,
    babappalign_model_status,
    detect_aligner_backends,
    supported_alignment_methods,
    validate_alignment_methods,
)
from babappa.align.ensemble import AlignmentConfig, align_simulation_directory
from babappa.align.external import ExternalAlignmentConfig, run_alignment_ensemble, smoke_aligner
from babappa.align.method_policy import (
    MethodPolicyConfig,
    build_method_policy,
    validate_method_policy_dir,
)
from babappa.align.site_map import (
    SiteMapConfig,
    build_alignment_site_maps,
    build_site_map_for_alignment,
)
from babappa.align.site_map_audit import validate_site_map_dir

__all__ = [
    "AlignerBackend",
    "AlignmentConfig",
    "ExternalAlignmentConfig",
    "MethodPolicyConfig",
    "SiteMapConfig",
    "align_simulation_directory",
    "babappalign_model_path",
    "babappalign_model_status",
    "build_alignment_site_maps",
    "build_site_map_for_alignment",
    "build_method_policy",
    "detect_aligner_backends",
    "run_alignment_ensemble",
    "smoke_aligner",
    "supported_alignment_methods",
    "validate_alignment_directory",
    "validate_alignment_methods",
    "validate_method_policy_dir",
    "validate_site_map_dir",
]
