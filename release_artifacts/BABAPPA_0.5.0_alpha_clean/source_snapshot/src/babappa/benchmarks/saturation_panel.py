"""Multi-saturation benchmark panel construction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

from babappa import __version__
from babappa.align import (
    AlignmentConfig,
    align_simulation_directory,
    validate_alignment_directory,
)
from babappa.datasets import DatasetIndexConfig, build_dataset_index, validate_dataset_index
from babappa.simulate import SimulationConfig, audit_simulation_directory, simulate_families
from babappa.tensors import TensorBuildConfig, build_tensor_dataset, validate_tensor_directory

SATURATION_PANEL_VERSION = __version__
ALLOWED_TIERS = {"low", "moderate", "high", "extreme"}
ALLOWED_METHODS = {"identity", "codon_dropout"}


@dataclass(frozen=True)
class SaturationPanelConfig:
    """Configuration for building a tiered saturation benchmark panel."""

    outdir: str
    n_families_per_tier: int = 10
    tiers: List[str] = field(
        default_factory=lambda: ["low", "moderate", "high", "extreme"]
    )
    n_taxa: int = 8
    n_codons: int = 120
    seed: int = 42
    positive_rate: float = 0.5
    selected_site_fraction: float = 0.05
    mutation_rate: float = 0.03
    indel_rate: float = 0.0
    methods: List[str] = field(
        default_factory=lambda: ["identity", "codon_dropout"]
    )
    dropout_rate: float = 0.02
    build_tensors: bool = True
    index_datasets: bool = True

    def __post_init__(self) -> None:
        if self.n_families_per_tier < 1:
            raise ValueError("n_families_per_tier must be >= 1")
        if self.n_taxa < 3:
            raise ValueError("n_taxa must be >= 3")
        if self.n_codons < 30:
            raise ValueError("n_codons must be >= 30")
        if not self.tiers:
            raise ValueError("tiers must be non-empty")
        unknown_tiers = sorted(set(self.tiers) - ALLOWED_TIERS)
        if unknown_tiers:
            allowed = ", ".join(sorted(ALLOWED_TIERS))
            unknown = ", ".join(unknown_tiers)
            raise ValueError(f"unknown saturation tier(s): {unknown}; allowed: {allowed}")
        if not 0 <= self.positive_rate <= 1:
            raise ValueError("positive_rate must be between 0 and 1")
        if not 0 <= self.selected_site_fraction <= 1:
            raise ValueError("selected_site_fraction must be between 0 and 1")
        if self.mutation_rate < 0:
            raise ValueError("mutation_rate must be >= 0")
        if self.indel_rate < 0:
            raise ValueError("indel_rate must be >= 0")
        if not self.methods:
            raise ValueError("methods must be non-empty")
        unknown_methods = sorted(set(self.methods) - ALLOWED_METHODS)
        if unknown_methods:
            allowed = ", ".join(sorted(ALLOWED_METHODS))
            unknown = ", ".join(unknown_methods)
            raise ValueError(f"unknown alignment method(s): {unknown}; allowed: {allowed}")
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        if self.index_datasets and not self.build_tensors:
            raise ValueError("index_datasets requires build_tensors")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def build_saturation_panel(config: SaturationPanelConfig) -> dict:
    """Build tier-specific simulation, alignment, tensor, and dataset outputs."""
    outdir = Path(config.outdir)
    tier_outputs: Dict[str, dict] = {}
    warnings: List[str] = []

    for tier_index, tier in enumerate(config.tiers):
        tier_seed = config.seed + tier_index * 100_000
        tier_root = outdir / "tiers" / tier
        sim_dir = tier_root / "sim"
        audit_dir = sim_dir / "audit"
        align_dir = tier_root / "align"
        tensor_dir = tier_root / "tensors"
        dataset_dir = tier_root / "dataset"

        sim_summary = simulate_families(
            SimulationConfig(
                outdir=str(sim_dir),
                n_families=config.n_families_per_tier,
                n_taxa=config.n_taxa,
                n_codons=config.n_codons,
                seed=tier_seed,
                positive_rate=config.positive_rate,
                selected_site_fraction=config.selected_site_fraction,
                mutation_rate=config.mutation_rate,
                indel_rate=config.indel_rate,
                saturation_tier=tier,
            )
        )
        audit_summary = audit_simulation_directory(sim_dir, audit_dir)
        align_summary = align_simulation_directory(
            AlignmentConfig(
                sim_dir=str(sim_dir),
                outdir=str(align_dir),
                methods=list(config.methods),
                seed=tier_seed,
                dropout_rate=config.dropout_rate,
            )
        )
        align_validation = validate_alignment_directory(align_dir)
        if align_validation["status"] != "ok":
            warnings.append(f"{tier}:alignment_validation_{align_validation['status']}")

        tensor_summary = None
        tensor_validation = None
        dataset_summary = None
        dataset_validation = None
        if config.build_tensors:
            tensor_summary = build_tensor_dataset(
                TensorBuildConfig(
                    sim_dir=str(sim_dir),
                    align_dir=str(align_dir),
                    outdir=str(tensor_dir),
                    methods=list(config.methods),
                )
            )
            tensor_validation = validate_tensor_directory(tensor_dir)
            if tensor_validation["status"] != "ok":
                warnings.append(f"{tier}:tensor_validation_{tensor_validation['status']}")

        if config.index_datasets:
            dataset_summary = build_dataset_index(
                DatasetIndexConfig(
                    tensor_dir=str(tensor_dir),
                    outdir=str(dataset_dir),
                    methods=list(config.methods),
                    seed=tier_seed,
                )
            )
            dataset_validation = validate_dataset_index(dataset_dir)
            if dataset_validation["status"] != "ok":
                warnings.append(
                    f"{tier}:dataset_validation_{dataset_validation['status']}"
                )

        tier_outputs[tier] = {
            "tier_seed": tier_seed,
            "tier_dir": str(tier_root),
            "sim_dir": str(sim_dir),
            "sim_summary": sim_summary,
            "sim_audit_dir": str(audit_dir),
            "sim_audit_summary": audit_summary,
            "align_dir": str(align_dir),
            "align_summary": align_summary,
            "align_validation": align_validation,
            "tensor_dir": str(tensor_dir) if config.build_tensors else None,
            "tensor_summary": tensor_summary,
            "tensor_validation": tensor_validation,
            "dataset_dir": str(dataset_dir) if config.index_datasets else None,
            "dataset_summary": dataset_summary,
            "dataset_validation": dataset_validation,
        }

    panel_json = outdir / "saturation_panel.json"
    panel_markdown = outdir / "saturation_panel.md"
    payload = {
        "saturation_panel_version": SATURATION_PANEL_VERSION,
        "config": asdict(config),
        "tiers": list(config.tiers),
        "n_families_per_tier": config.n_families_per_tier,
        "total_families_expected": config.n_families_per_tier * len(config.tiers),
        "tier_outputs": tier_outputs,
        "warnings": sorted(set(warnings)),
        "generated_files": {
            "panel_json": str(panel_json),
            "panel_markdown": str(panel_markdown),
        },
    }
    _write_json(panel_json, payload)
    panel_markdown.write_text(_render_markdown(payload), encoding="utf-8")

    return {
        "status": "ok",
        "outdir": str(outdir),
        "panel_json": str(panel_json),
        "panel_markdown": str(panel_markdown),
        "tier_outputs": tier_outputs,
        "warnings": payload["warnings"],
    }


def _render_markdown(payload: dict) -> str:
    lines = [
        "# BABAPPA saturation panel",
        "",
        "## Configuration",
        "",
        f"- Tiers: {', '.join(payload['tiers'])}",
        f"- Families per tier: {payload['n_families_per_tier']}",
        f"- Total expected families: {payload['total_families_expected']}",
        f"- Methods: {', '.join(payload['config'].get('methods', []))}",
        "",
        "## Tier outputs",
        "",
        "| Tier | Seed | Simulation | Alignment | Tensors | Dataset |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for tier, outputs in payload["tier_outputs"].items():
        lines.append(
            "| {tier} | {seed} | `{sim}` | `{align}` | `{tensor}` | `{dataset}` |".format(
                tier=tier,
                seed=outputs["tier_seed"],
                sim=outputs["sim_dir"],
                align=outputs["align_dir"],
                tensor=outputs.get("tensor_dir") or "not built",
                dataset=outputs.get("dataset_dir") or "not indexed",
            )
        )
    lines.extend(["", "## Generated files", ""])
    for key, value in payload["generated_files"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Warnings", ""])
    warnings = payload.get("warnings") or []
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This is a gene-level saturation benchmark substrate, not the final branch-site model.",
            "- Current alignments are internal identity/codon_dropout scaffolds.",
            "- Current simulator remains lightweight and is not yet the final codon-likelihood simulator.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
