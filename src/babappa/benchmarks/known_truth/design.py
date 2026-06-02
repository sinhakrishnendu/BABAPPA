"""Design BABAPPA known-truth simulation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv

from .truth_schema import write_json


@dataclass(frozen=True)
class KnownTruthBenchmarkDesignConfig:
    outdir: str
    benchmark_name: str = "BABAPPA-BENCH-SIM-v1"
    seed: int = 42


REGIME_FIELDS = [
    "regime",
    "regime_group",
    "n_families_default",
    "n_taxa",
    "n_codons",
    "tree_shape",
    "foreground_branch_count",
    "foreground_branch_length",
    "omega_background",
    "omega_foreground_positive",
    "selected_site_fraction",
    "positive_effect_size",
    "saturation_tier",
    "gap_rate",
    "indel_rate",
    "compositional_bias",
    "alignment_difficulty",
    "expected_applicability",
    "expected_truth_class",
    "expected_decision_behavior",
]


PROFILE_SIZES = {
    "smoke": 12,
    "pilot": 300,
    "paper": 5000,
    "extended": 20000,
}


def benchmark_regimes() -> List[Dict[str, Any]]:
    nulls = [
        ("null_low_divergence", 10, 420, "low", "in_domain", "null", "negative"),
        ("null_moderate_divergence", 10, 420, "moderate", "in_domain", "null", "negative"),
        ("null_high_divergence", 10, 420, "high", "borderline", "null", "cautious_negative"),
        ("null_extreme_divergence", 10, 420, "extreme", "out_of_domain", "ood_null", "abstain"),
        ("null_alignment_difficult", 10, 420, "moderate", "borderline", "null", "cautious_negative"),
        ("null_short_gene", 10, 60, "moderate", "borderline", "null", "cautious_negative"),
        ("null_few_taxa", 4, 420, "moderate", "out_of_domain", "ood_null", "abstain"),
        ("null_long_branch", 10, 420, "high", "borderline", "null", "cautious_negative"),
    ]
    positives = [
        ("positive_weak_branch_site", 10, 420, "moderate", "in_domain", "positive", "detect_if_calibrated", 2.0, 0.02),
        ("positive_moderate_branch_site", 10, 420, "moderate", "in_domain", "positive", "detect", 4.0, 0.04),
        ("positive_strong_branch_site", 10, 420, "moderate", "in_domain", "positive", "detect", 8.0, 0.06),
        ("positive_short_foreground", 10, 420, "moderate", "in_domain", "positive", "detect_if_power", 4.0, 0.04),
        ("positive_long_foreground", 10, 420, "moderate", "borderline", "positive", "detect_with_caution", 4.0, 0.04),
        ("positive_sparse_sites", 10, 420, "moderate", "in_domain", "positive", "detect_if_power", 5.0, 0.01),
        ("positive_clustered_sites", 10, 420, "moderate", "in_domain", "positive", "detect", 5.0, 0.05),
        ("positive_multi_branch", 10, 420, "moderate", "in_domain", "positive", "detect", 4.0, 0.04),
        ("positive_alignment_difficult", 10, 420, "high", "borderline", "positive", "detect_with_alignment_audit", 4.0, 0.04),
    ]
    oods = [
        ("ood_extreme_saturation", 10, 420, "extreme", "out_of_domain", "ood_null", "abstain"),
        ("ood_too_few_taxa", 3, 420, "moderate", "out_of_domain", "ood_null", "abstain"),
        ("ood_too_short", 10, 45, "moderate", "out_of_domain", "ood_null", "abstain"),
        ("ood_high_gap", 10, 420, "moderate", "out_of_domain", "ood_null", "abstain"),
        ("ood_compositional_bias", 10, 420, "moderate", "out_of_domain", "ood_null", "abstain"),
        ("ood_paralogy_like", 10, 420, "high", "out_of_domain", "ood_null", "abstain"),
        ("ood_tree_mismatch_like", 10, 420, "moderate", "out_of_domain", "ambiguous", "fail_or_abstain"),
    ]
    rows: List[Dict[str, Any]] = []
    for regime, n_taxa, n_codons, tier, applicability, truth, decision in nulls:
        rows.append(_regime_row(regime, "null", n_taxa, n_codons, tier, applicability, truth, decision))
    for regime, n_taxa, n_codons, tier, applicability, truth, decision, effect, frac in positives:
        row = _regime_row(regime, "positive", n_taxa, n_codons, tier, applicability, truth, decision)
        row["omega_foreground_positive"] = effect
        row["positive_effect_size"] = effect
        row["selected_site_fraction"] = frac
        row["foreground_branch_count"] = 2 if "multi_branch" in regime else 1
        if "short_foreground" in regime:
            row["foreground_branch_length"] = 0.02
        if "long_foreground" in regime:
            row["foreground_branch_length"] = 0.35
        rows.append(row)
    for regime, n_taxa, n_codons, tier, applicability, truth, decision in oods:
        row = _regime_row(regime, "ood", n_taxa, n_codons, tier, applicability, truth, decision)
        if "high_gap" in regime:
            row["gap_rate"] = 0.25
            row["alignment_difficulty"] = "high_gap"
        if "bias" in regime:
            row["compositional_bias"] = "at_rich"
        if "paralogy" in regime:
            row["alignment_difficulty"] = "paralogy_like"
        rows.append(row)
    return rows


def _regime_row(
    regime: str,
    group: str,
    n_taxa: int,
    n_codons: int,
    tier: str,
    applicability: str,
    truth: str,
    decision: str,
) -> Dict[str, Any]:
    return {
        "regime": regime,
        "regime_group": group,
        "n_families_default": 1,
        "n_taxa": n_taxa,
        "n_codons": n_codons,
        "tree_shape": "balanced",
        "foreground_branch_count": 1,
        "foreground_branch_length": 0.12,
        "omega_background": 0.3,
        "omega_foreground_positive": 1.0,
        "selected_site_fraction": 0.0,
        "positive_effect_size": 0.0,
        "saturation_tier": tier,
        "gap_rate": 0.0,
        "indel_rate": 0.0,
        "compositional_bias": "none",
        "alignment_difficulty": "moderate" if tier in {"moderate", "high"} else tier,
        "expected_applicability": applicability,
        "expected_truth_class": truth,
        "expected_decision_behavior": decision,
    }


def design_known_truth_benchmark(config: KnownTruthBenchmarkDesignConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = benchmark_regimes()
    payload = {
        "known_truth_benchmark_design_version": __version__,
        "benchmark_name": config.benchmark_name,
        "seed": config.seed,
        "profiles": PROFILE_SIZES,
        "n_regimes": len(rows),
        "regimes": rows,
        "scientific_boundary": (
            "This benchmark validates BABAPPA against explicit simulated truth. "
            "codeml/HyPhy comparisons are comparator analyses, not truth labels."
        ),
    }
    write_json(outdir / "benchmark_design.json", payload)
    write_tsv(outdir / "benchmark_design.tsv", rows, REGIME_FIELDS)
    write_tsv(outdir / "regime_manifest.tsv", rows, REGIME_FIELDS)
    write_json(
        outdir / "expected_outputs.json",
        {
            "simulated_families": "family FASTA/tree/truth files",
            "alignments": "per-method alignments and method policy",
            "babappa_scores": "gene/branch/site scores plus applicability",
            "evaluation": "AUROC/AUPRC/FDR/OOD metrics against truth",
            "calibration_evaluation": "BH q-values and FDR/power tables",
            "report": "manuscript-ready known-truth benchmark summary",
        },
    )
    (outdir / "benchmark_design.md").write_text(_render_design_md(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "benchmark_name": config.benchmark_name,
        "n_regimes": len(rows),
        "profiles": ",".join(PROFILE_SIZES),
    }


def load_design(design_dir: Path) -> Dict[str, Any]:
    design_json = design_dir / "benchmark_design.json"
    if design_json.exists():
        import json

        return json.loads(design_json.read_text(encoding="utf-8"))
    rows = read_tsv(design_dir / "regime_manifest.tsv")
    return {"benchmark_name": "BABAPPA-BENCH-SIM-v1", "regimes": rows, "profiles": PROFILE_SIZES}


def _render_design_md(payload: Dict[str, Any]) -> str:
    lines = [
        f"# {payload['benchmark_name']}",
        "",
        "This design defines a known-truth simulation benchmark for BABAPPA.",
        "",
        "Truth labels are generated by the simulator and are used only during evaluation.",
        "",
        "## Profiles",
        "",
    ]
    for name, size in payload["profiles"].items():
        lines.append(f"- `{name}`: {size} families")
    lines.extend(["", "## Regimes", ""])
    for row in payload["regimes"]:
        lines.append(
            f"- `{row['regime']}`: {row['expected_truth_class']}, "
            f"{row['saturation_tier']}, expected {row['expected_applicability']}"
        )
    lines.extend(["", "## Claim Boundary", "", payload["scientific_boundary"], ""])
    return "\n".join(lines)

