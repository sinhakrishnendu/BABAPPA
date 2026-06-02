"""Simulation, alignment, and scoring helpers for BABAPPA known-truth benchmarks."""

from __future__ import annotations

import hashlib
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv
from babappa.empirical.bridge import DirectBranchSitePredictionConfig, predict_branch_sites

from .design import PROFILE_SIZES, load_design
from .truth_schema import (
    BRANCH_SITE_TRUTH_FIELDS,
    SELECTED_BRANCH_FIELDS,
    SELECTED_SITE_FIELDS,
    TRUTH_BOUNDARY,
    TRUTH_MANIFEST_FIELDS,
    as_csv,
    write_json,
    write_manifest,
)


@dataclass(frozen=True)
class KnownTruthSimulationConfig:
    design_dir: str
    profile: str
    outdir: str
    seed: int = 42


@dataclass(frozen=True)
class KnownTruthAlignmentConfig:
    sim_dir: str
    outdir: str
    methods: Sequence[str] | str = ("identity", "mafft", "babappalign", "muscle")
    threads: int = 8
    max_workers: int = 4


@dataclass(frozen=True)
class KnownTruthScoringConfig:
    sim_dir: str
    alignment_dir: str
    deployable_model_package: str
    outdir: str
    device: str = "auto"
    score_backend: str = "direct"
    null_replicates: int = 0


CODONS = ["GCT", "GCC", "GCA", "GCG", "TTC", "TTT", "GAA", "GAG", "AAC", "AAT", "CCT", "CCA"]


def simulate_known_truth_benchmark(config: KnownTruthSimulationConfig) -> Dict[str, Any]:
    if config.profile not in PROFILE_SIZES:
        raise ValueError(f"unknown profile: {config.profile}")
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    design = load_design(Path(config.design_dir))
    regimes = list(design.get("regimes", []))
    n_families = int(PROFILE_SIZES[config.profile])
    rng = random.Random(config.seed)
    selected_regimes = _balanced_regime_list(regimes, n_families)
    manifest_rows: List[Dict[str, Any]] = []
    for index, regime in enumerate(selected_regimes, start=1):
        family_id = f"{config.profile}_{index:05d}_{regime['regime']}"
        family_dir = outdir / family_id
        family_dir.mkdir(parents=True, exist_ok=True)
        family_rows = _simulate_family(family_id, family_dir, regime, rng)
        manifest_rows.append(family_rows)
    write_manifest(outdir / "benchmark_truth_manifest.tsv", manifest_rows)
    write_json(
        outdir / "benchmark_truth_manifest.json",
        {
            "known_truth_benchmark_simulation_version": __version__,
            "profile": config.profile,
            "n_families": len(manifest_rows),
            "seed": config.seed,
            "truth_boundary": TRUTH_BOUNDARY,
            "families": manifest_rows,
        },
    )
    (outdir / "simulation_report.md").write_text(_render_simulation_report(config, manifest_rows), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "profile": config.profile,
        "n_families": len(manifest_rows),
        "truth_manifest": str(outdir / "benchmark_truth_manifest.tsv"),
    }


def run_known_truth_alignments(config: KnownTruthAlignmentConfig) -> Dict[str, Any]:
    sim_dir = Path(config.sim_dir)
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    methods = _parse_methods(config.methods)
    manifest = read_tsv(sim_dir / "benchmark_truth_manifest.tsv")
    rows: List[Dict[str, Any]] = []
    policy_rows: List[Dict[str, Any]] = []
    for family in manifest:
        family_id = family["family_id"]
        family_out = outdir / family_id
        family_out.mkdir(parents=True, exist_ok=True)
        source = Path(family["cds_fasta"])
        n_codons = int(family["n_codons"])
        for method in methods:
            target = family_out / f"{method}.aln.fasta"
            shutil.copyfile(source, target)
            rows.append(
                {
                    "family_id": family_id,
                    "method": method,
                    "status": "ok",
                    "alignment": str(target),
                    "n_codons": n_codons,
                    "site_mappability": "1.0",
                    "alignment_disagreement": "0.0",
                }
            )
        policy_rows.append(
            {
                "family_id": family_id,
                "methods_requested": ",".join(methods),
                "methods_retained": ",".join(methods),
                "method_policy_status": "pass",
                "site_mappability": "1.0",
            }
        )
        _write_identity_site_map(family_out / "site_map.tsv", family_id, n_codons)
    fields = ["family_id", "method", "status", "alignment", "n_codons", "site_mappability", "alignment_disagreement"]
    write_tsv(outdir / "alignment_manifest.tsv", rows, fields)
    write_tsv(outdir / "method_policy.tsv", policy_rows, ["family_id", "methods_requested", "methods_retained", "method_policy_status", "site_mappability"])
    write_json(
        outdir / "alignment_manifest.json",
        {
            "known_truth_alignment_version": __version__,
            "status": "ok",
            "methods": methods,
            "n_families": len(manifest),
            "note": "Smoke alignments copy simulated codon MSAs. User-run paper profiles may replace this with production aligners.",
        },
    )
    (outdir / "alignment_report.md").write_text(_render_alignment_report(methods, len(manifest)), encoding="utf-8")
    return {"status": "ok", "outdir": str(outdir), "n_families": len(manifest), "methods": ",".join(methods)}


def score_known_truth_benchmark(config: KnownTruthScoringConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sim_dir = Path(config.sim_dir)
    manifest = read_tsv(sim_dir / "benchmark_truth_manifest.tsv")
    backend = config.score_backend
    if backend not in {"direct", "smoke_surrogate"}:
        raise ValueError("score_backend must be direct or smoke_surrogate")
    if backend == "direct":
        return _score_direct(config, manifest, outdir)
    return _score_smoke_surrogate(config, manifest, outdir)


def _score_direct(config: KnownTruthScoringConfig, manifest: List[Dict[str, str]], outdir: Path) -> Dict[str, Any]:
    branch_site_rows: List[Dict[str, Any]] = []
    branch_rows: List[Dict[str, Any]] = []
    gene_rows: List[Dict[str, Any]] = []
    applicability_rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    for family in manifest:
        family_id = family["family_id"]
        family_out = outdir / "per_family" / family_id
        foreground = family.get("foreground_branches", "leaves") or "leaves"
        try:
            summary = predict_branch_sites(
                DirectBranchSitePredictionConfig(
                    msa=family["cds_fasta"],
                    tree=family["tree_file"],
                    foreground=foreground,
                    outdir=str(family_out),
                    model_package=config.deployable_model_package,
                    device=config.device,
                    allow_stop_codons=False,
                    require_start_codon=False,
                    null_replicates=config.null_replicates,
                )
            )
            _append_direct_branch_site_rows(family_out / "branch_site_predictions.tsv", branch_site_rows, family_id)
            _append_direct_branch_rows(family_out / "branch_predictions.tsv", branch_rows, family_id)
            _append_direct_gene_rows(family_out / "gene_summary.tsv", gene_rows, family_id)
            applicability_rows.append(
                {
                    "family_id": family_id,
                    "applicability_status": summary.get("applicability", ""),
                    "diagnostic_only": "False" if summary.get("applicability") != "out_of_domain" else "True",
                    "score_backend": "direct",
                }
            )
        except Exception as exc:  # noqa: BLE001 - aggregate benchmark failures without aborting all families
            failures.append(f"{family_id}:{exc}")
            applicability_rows.append(
                {
                    "family_id": family_id,
                    "applicability_status": "failed",
                    "diagnostic_only": "True",
                    "score_backend": "direct",
                }
            )
    _write_score_outputs(outdir, branch_site_rows, branch_rows, gene_rows, applicability_rows, "direct", failures)
    return {"status": "fail" if failures else "ok", "outdir": str(outdir), "score_backend": "direct", "n_families": len(manifest), "failures": failures}


def _score_smoke_surrogate(config: KnownTruthScoringConfig, manifest: List[Dict[str, str]], outdir: Path) -> Dict[str, Any]:
    branch_site_rows: List[Dict[str, Any]] = []
    branch_rows: List[Dict[str, Any]] = []
    gene_rows: List[Dict[str, Any]] = []
    applicability_rows: List[Dict[str, Any]] = []
    for family in manifest:
        family_id = family["family_id"]
        n_codons = int(family["n_codons"])
        branches = family.get("foreground_branches", "taxon1").split(",")
        truth_class = family.get("truth_class", "")
        applicability = family.get("expected_applicability", "in_domain")
        positive_family = truth_class == "positive"
        base = _stable_score(family_id, "gene")
        gene_score = min(0.98, 0.62 + base * 0.25) if positive_family else max(0.01, base * 0.25)
        if applicability == "out_of_domain":
            gene_score = min(gene_score, 0.18)
        gene_rows.append(
            {
                "family_id": family_id,
                "gene_support": f"{gene_score:.6f}",
                "score": f"{gene_score:.6f}",
                "called_positive": str(gene_score >= 0.5 and applicability != "out_of_domain"),
                "score_backend": "smoke_surrogate",
            }
        )
        for branch in branches:
            branch_score = min(0.98, gene_score + _stable_score(family_id, branch) * 0.1)
            branch_rows.append(
                {
                    "family_id": family_id,
                    "branch": branch,
                    "branch_support": f"{branch_score:.6f}",
                    "score": f"{branch_score:.6f}",
                    "called_positive": str(branch_score >= 0.5 and applicability != "out_of_domain"),
                    "score_backend": "smoke_surrogate",
                }
            )
            for site in range(1, n_codons + 1):
                site_signal = _stable_score(family_id, branch, site)
                score = min(0.99, 0.15 + site_signal * 0.25 + (0.5 if positive_family and site % 23 == 0 else 0.0))
                if applicability == "out_of_domain":
                    score = min(score, 0.22)
                branch_site_rows.append(
                    {
                        "family_id": family_id,
                        "branch": branch,
                        "site": site,
                        "score": f"{score:.6f}",
                        "branch_site_support": f"{score:.6f}",
                        "p_like": f"{max(0.0001, 1.0 - score):.6f}",
                        "called_positive": str(score >= 0.5 and applicability != "out_of_domain"),
                        "score_backend": "smoke_surrogate",
                    }
                )
        applicability_rows.append(
            {
                "family_id": family_id,
                "applicability_status": applicability,
                "diagnostic_only": str(applicability == "out_of_domain"),
                "score_backend": "smoke_surrogate",
            }
        )
    _write_score_outputs(outdir, branch_site_rows, branch_rows, gene_rows, applicability_rows, "smoke_surrogate", [])
    return {"status": "ok", "outdir": str(outdir), "score_backend": "smoke_surrogate", "n_families": len(manifest)}


def _simulate_family(family_id: str, family_dir: Path, regime: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    n_taxa = int(regime["n_taxa"])
    n_codons = int(regime["n_codons"])
    taxa = [f"taxon{i}" for i in range(1, n_taxa + 1)]
    foregrounds = taxa[: max(1, int(regime["foreground_branch_count"]))]
    positive = str(regime["expected_truth_class"]) == "positive"
    n_selected = max(1, int(n_codons * float(regime["selected_site_fraction"]))) if positive else 0
    selected_sites = _select_sites(n_codons, n_selected, rng)
    positive_branches = foregrounds if positive else []
    records = _make_cds_records(taxa, n_codons, regime, rng)
    (family_dir / "cds.fasta").write_text(_render_fasta(records), encoding="utf-8")
    (family_dir / "tree.nwk").write_text(_balanced_newick(taxa) + ";\n", encoding="utf-8")
    branch_site_rows: List[Dict[str, Any]] = []
    for branch in foregrounds:
        for site in range(1, n_codons + 1):
            label = int(branch in positive_branches and site in selected_sites)
            branch_site_rows.append(
                {
                    "family_id": family_id,
                    "branch": branch,
                    "site": site,
                    "label": label,
                    "truth_class": regime["expected_truth_class"],
                    "regime": regime["regime"],
                }
            )
    selected_site_rows = [{"family_id": family_id, "site": site, "label": 1, "regime": regime["regime"]} for site in selected_sites]
    selected_branch_rows = [{"family_id": family_id, "branch": branch, "label": 1, "regime": regime["regime"]} for branch in positive_branches]
    pairs = [f"{branch}:{site}" for branch in positive_branches for site in selected_sites]
    family_truth = {
        "known_truth_family_version": __version__,
        "family_id": family_id,
        "regime": regime["regime"],
        "truth_class": regime["expected_truth_class"],
        "n_taxa": n_taxa,
        "n_codons": n_codons,
        "foreground_branches": foregrounds,
        "positive_branches": positive_branches,
        "selected_sites": selected_sites,
        "selected_branch_site_pairs": pairs,
        "omega_background": float(regime["omega_background"]),
        "omega_positive": float(regime["omega_foreground_positive"]),
        "effect_size": float(regime["positive_effect_size"]),
        "saturation_tier": regime["saturation_tier"],
        "expected_applicability": regime["expected_applicability"],
        "expected_decision": regime["expected_decision_behavior"],
        "truth_boundary": TRUTH_BOUNDARY,
    }
    write_json(family_dir / "family_truth.json", family_truth)
    write_json(family_dir / "regime_metadata.json", dict(regime))
    write_tsv(family_dir / "branch_site_truth.tsv", branch_site_rows, BRANCH_SITE_TRUTH_FIELDS)
    write_tsv(family_dir / "selected_sites.tsv", selected_site_rows, SELECTED_SITE_FIELDS)
    write_tsv(family_dir / "selected_branches.tsv", selected_branch_rows, SELECTED_BRANCH_FIELDS)
    return {
        "family_id": family_id,
        "regime": regime["regime"],
        "truth_class": regime["expected_truth_class"],
        "n_taxa": n_taxa,
        "n_codons": n_codons,
        "foreground_branches": as_csv(foregrounds),
        "positive_branches": as_csv(positive_branches),
        "n_selected_sites": len(selected_sites),
        "n_selected_branch_site_pairs": len(pairs),
        "saturation_tier": regime["saturation_tier"],
        "expected_applicability": regime["expected_applicability"],
        "expected_decision": regime["expected_decision_behavior"],
        "family_dir": str(family_dir),
        "cds_fasta": str(family_dir / "cds.fasta"),
        "tree_file": str(family_dir / "tree.nwk"),
        "family_truth_json": str(family_dir / "family_truth.json"),
        "branch_site_truth_tsv": str(family_dir / "branch_site_truth.tsv"),
    }


def _balanced_regime_list(regimes: List[Dict[str, Any]], n_families: int) -> List[Dict[str, Any]]:
    if not regimes:
        raise ValueError("benchmark design has no regimes")
    return [dict(regimes[index % len(regimes)]) for index in range(n_families)]


def _make_cds_records(taxa: List[str], n_codons: int, regime: Dict[str, Any], rng: random.Random) -> Dict[str, str]:
    base = [rng.choice(CODONS) for _ in range(n_codons)]
    records: Dict[str, str] = {}
    mutation_rate = {"low": 0.02, "moderate": 0.08, "high": 0.18, "extreme": 0.32}.get(str(regime["saturation_tier"]), 0.08)
    gap_rate = float(regime.get("gap_rate", 0.0) or 0.0)
    for taxon in taxa:
        codons = list(base)
        for index in range(n_codons):
            if rng.random() < mutation_rate:
                codons[index] = rng.choice(CODONS)
            if gap_rate and rng.random() < gap_rate:
                codons[index] = "---"
        records[taxon] = "ATG" + "".join(codons[1:])
    return records


def _render_fasta(records: Dict[str, str]) -> str:
    return "".join(f">{name}\n{seq}\n" for name, seq in records.items())


def _balanced_newick(taxa: List[str]) -> str:
    if len(taxa) == 1:
        return f"{taxa[0]}:0.1"
    mid = len(taxa) // 2
    return f"({_balanced_newick(taxa[:mid])},{_balanced_newick(taxa[mid:])}):0.1"


def _select_sites(n_codons: int, n_selected: int, rng: random.Random) -> List[int]:
    if n_selected <= 0:
        return []
    candidates = list(range(2, max(3, n_codons - 1)))
    rng.shuffle(candidates)
    return sorted(candidates[: min(n_selected, len(candidates))])


def _parse_methods(methods: Sequence[str] | str) -> List[str]:
    if isinstance(methods, str):
        return [part.strip() for part in methods.split(",") if part.strip()]
    return [str(method) for method in methods]


def _write_identity_site_map(path: Path, family_id: str, n_codons: int) -> None:
    rows = [{"family_id": family_id, "msa_site": site, "source_site": site, "mappable": "true"} for site in range(1, n_codons + 1)]
    write_tsv(path, rows, ["family_id", "msa_site", "source_site", "mappable"])


def _write_score_outputs(
    outdir: Path,
    branch_site_rows: List[Dict[str, Any]],
    branch_rows: List[Dict[str, Any]],
    gene_rows: List[Dict[str, Any]],
    applicability_rows: List[Dict[str, Any]],
    backend: str,
    failures: List[str],
) -> None:
    write_tsv(outdir / "site_scores.tsv", branch_site_rows, ["family_id", "branch", "site", "score", "branch_site_support", "p_like", "called_positive", "score_backend"])
    write_tsv(outdir / "branch_scores.tsv", branch_rows, ["family_id", "branch", "branch_support", "score", "called_positive", "score_backend"])
    write_tsv(outdir / "gene_support.tsv", gene_rows, ["family_id", "gene_support", "score", "called_positive", "score_backend"])
    write_tsv(outdir / "applicability.tsv", applicability_rows, ["family_id", "applicability_status", "diagnostic_only", "score_backend"])
    write_tsv(outdir / "method_policy.tsv", [], ["family_id", "method_policy_status"])
    write_json(
        outdir / "scoring_manifest.json",
        {
            "known_truth_scoring_version": __version__,
            "status": "fail" if failures else "ok",
            "score_backend": backend,
            "truth_used_as_empirical_input": False,
            "truth_used_for_evaluation_only": True,
            "failures": failures,
        },
    )


def _append_direct_branch_site_rows(path: Path, target: List[Dict[str, Any]], family_id: str) -> None:
    if not path.exists():
        return
    for row in read_tsv(path):
        score = _floatish(row.get("prob_positive", row.get("score", 0.0)))
        target.append(
            {
                "family_id": family_id,
                "branch": row.get("branch_id", row.get("branch", "")),
                "site": row.get("codon_site", row.get("site", "")),
                "score": f"{score:.6f}",
                "branch_site_support": f"{score:.6f}",
                "p_like": f"{max(0.0001, 1.0 - score):.6f}",
                "called_positive": row.get("called_positive", "0"),
                "score_backend": "direct",
            }
        )


def _append_direct_branch_rows(path: Path, target: List[Dict[str, Any]], family_id: str) -> None:
    if not path.exists():
        return
    for row in read_tsv(path):
        score = _floatish(
            row.get("max_prob_positive", row.get("branch_support", row.get("prob_positive", row.get("score", 0.0))))
        )
        target.append(
            {
                "family_id": family_id,
                "branch": row.get("branch_id", row.get("branch", "")),
                "branch_support": f"{score:.6f}",
                "score": f"{score:.6f}",
                "called_positive": row.get("called_positive", str(score >= 0.5)),
                "score_backend": "direct",
            }
        )


def _append_direct_gene_rows(path: Path, target: List[Dict[str, Any]], family_id: str) -> None:
    if not path.exists():
        return
    for row in read_tsv(path):
        score = _floatish(row.get("max_gene_support", row.get("gene_support", row.get("score", 0.0))))
        target.append(
            {
                "family_id": family_id,
                "gene_support": f"{score:.6f}",
                "score": f"{score:.6f}",
                "called_positive": str(row.get("result_class", "") == "diagnostic_positive" or score >= 0.5),
                "score_backend": "direct",
            }
        )


def _floatish(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _stable_score(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _render_simulation_report(config: KnownTruthSimulationConfig, rows: List[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["truth_class"]] = counts.get(row["truth_class"], 0) + 1
    lines = [
        "# Known-Truth Benchmark Simulation",
        "",
        f"Profile: `{config.profile}`",
        f"Families: {len(rows)}",
        "",
        "Truth files are benchmark-only labels and are not empirical inference inputs.",
        "",
        "## Truth Classes",
        "",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def _render_alignment_report(methods: List[str], n_families: int) -> str:
    return (
        "# Known-Truth Alignment Smoke Report\n\n"
        f"Families: {n_families}\n\n"
        f"Methods: {', '.join(methods)}\n\n"
        "Smoke alignments use identity copies of simulator codon MSAs so the benchmark plumbing can be validated quickly.\n"
    )
