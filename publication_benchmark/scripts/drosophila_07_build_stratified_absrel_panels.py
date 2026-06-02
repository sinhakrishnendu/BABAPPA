#!/usr/bin/env python3
"""Build stratified Drosophila BABAPPA vs HyPhy aBSREL benchmark panels.

The goal is publication-quality benchmarking, not just cherry-picking clean
families. The resulting panel intentionally covers strict in-domain families,
relaxed in-domain families, borderline cases, and OOD stress tests.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


STRATA = [
    {
        "name": "strict_in_domain",
        "description": "Low-divergence, low-gap families expected to be BABAPPA in-domain.",
        "max_gap_fraction": 0.15,
        "min_gap_fraction": None,
        "max_mean_pdistance": 0.20,
        "min_mean_pdistance": None,
    },
    {
        "name": "relaxed_in_domain",
        "description": "Moderate but still clean families near the upper in-domain range.",
        "max_gap_fraction": 0.20,
        "min_gap_fraction": None,
        "max_mean_pdistance": 0.25,
        "min_mean_pdistance": 0.20,
    },
    {
        "name": "borderline_distance",
        "description": "Low-gap families whose divergence is expected to push BABAPPA toward borderline.",
        "max_gap_fraction": 0.20,
        "min_gap_fraction": None,
        "max_mean_pdistance": 0.35,
        "min_mean_pdistance": 0.25,
    },
    {
        "name": "borderline_gap",
        "description": "Moderate-divergence families with elevated gap burden.",
        "max_gap_fraction": 0.35,
        "min_gap_fraction": 0.20,
        "max_mean_pdistance": 0.25,
        "min_mean_pdistance": None,
    },
    {
        "name": "ood_high_distance",
        "description": "Low-gap but high-divergence OOD stress-test families.",
        "max_gap_fraction": 0.25,
        "min_gap_fraction": None,
        "max_mean_pdistance": 0.50,
        "min_mean_pdistance": 0.35,
    },
    {
        "name": "ood_high_gap",
        "description": "High-gap OOD stress-test families with non-extreme divergence.",
        "max_gap_fraction": 0.50,
        "min_gap_fraction": 0.35,
        "max_mean_pdistance": 0.35,
        "min_mean_pdistance": None,
    },
    {
        "name": "extreme_distance_stress",
        "description": "Very high-divergence families used to test abstention and failure modes.",
        "max_gap_fraction": 0.50,
        "min_gap_fraction": None,
        "max_mean_pdistance": None,
        "min_mean_pdistance": 0.50,
    },
]


def load_builder_module():
    path = Path(__file__).with_name("drosophila_03_build_babappa_absrel_inputs.py")
    spec = importlib.util.spec_from_file_location("drosophila_absrel_input_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load builder module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def status_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def build(args: argparse.Namespace) -> int:
    builder = load_builder_module()
    outdir = Path(args.outdir)
    panel_root = outdir / "panels"
    combined_rows: list[dict[str, Any]] = []
    stratum_summaries: list[dict[str, Any]] = []

    for stratum in STRATA:
        name = stratum["name"]
        stratum_out = panel_root / name
        ns = SimpleNamespace(
            orthofinder_root=args.orthofinder_root,
            prepared_dir=args.prepared_dir,
            outdir=str(stratum_out),
            max_families=args.families_per_stratum,
            min_taxa=args.min_taxa,
            min_codons=args.min_codons,
            max_codons=args.max_codons,
            min_gap_fraction=stratum["min_gap_fraction"],
            max_gap_fraction=stratum["max_gap_fraction"],
            min_mean_pdistance=stratum["min_mean_pdistance"],
            max_mean_pdistance=stratum["max_mean_pdistance"],
            foreground=args.foreground,
            allow_missing_start_codon=args.allow_missing_start_codon,
        )
        builder.build(ns)
        panel = read_tsv(stratum_out / "drosophila_babappa_absrel_panel.tsv")
        summary = read_tsv(stratum_out / "input_build_summary.tsv")
        for row in panel:
            row["benchmark_stratum"] = name
            row["notes"] = f"{row.get('notes', '')} Stratum={name}: {stratum['description']}"
            combined_rows.append(row)
        stratum_summaries.append({
            "benchmark_stratum": name,
            "description": stratum["description"],
            "families_selected": len(panel),
            "status_counts": json.dumps(status_counts(summary), sort_keys=True),
            "min_gap_fraction": stratum["min_gap_fraction"],
            "max_gap_fraction": stratum["max_gap_fraction"],
            "min_mean_pdistance": stratum["min_mean_pdistance"],
            "max_mean_pdistance": stratum["max_mean_pdistance"],
        })

    panel_path = outdir / "stratified_drosophila_babappa_absrel_panel.tsv"
    write_tsv(
        panel_path,
        combined_rows,
        ["panel_id", "gene_family", "cds_msa", "tree_file", "foreground", "expected_category", "notes", "benchmark_stratum"],
    )
    write_tsv(
        outdir / "stratified_panel_summary.tsv",
        stratum_summaries,
        [
            "benchmark_stratum", "description", "families_selected", "status_counts",
            "min_gap_fraction", "max_gap_fraction", "min_mean_pdistance", "max_mean_pdistance",
        ],
    )
    payload = {
        "status": "ok",
        "panel": str(panel_path),
        "n_families": len(combined_rows),
        "families_per_stratum_requested": args.families_per_stratum,
        "strata": stratum_summaries,
        "publication_rationale": (
            "This design separates clean in-domain performance from borderline and OOD "
            "stress-test behavior, avoiding a single mixed average that hides applicability."
        ),
    }
    (outdir / "stratified_panel_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Stratified Drosophila BABAPPA/aBSREL Benchmark Panel",
        "",
        f"- combined panel: `{panel_path}`",
        f"- total families selected: `{len(combined_rows)}`",
        f"- families requested per stratum: `{args.families_per_stratum}`",
        "",
        "## Rationale",
        "",
        "The panel is stratified by applicability-related properties so the publication benchmark can report:",
        "",
        "- clean in-domain behavior",
        "- near-boundary behavior",
        "- OOD abstention and stress-test behavior",
        "- concordance/discordance with HyPhy aBSREL within each stratum",
        "",
        "This is more defensible than reporting a single average across heterogeneous orthologs.",
        "",
        "## Strata",
        "",
    ]
    for item in stratum_summaries:
        md.append(f"- `{item['benchmark_stratum']}`: {item['description']} selected `{item['families_selected']}` families")
    (outdir / "stratified_panel_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote stratified panel with {len(combined_rows)} families")
    print(panel_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orthofinder-root", default="publication_benchmark/drosophila_orthofinder")
    parser.add_argument("--prepared-dir", default="publication_benchmark/drosophila_orthofinder")
    parser.add_argument("--outdir", default="publication_benchmark/drosophila_absrel_benchmark_stratified")
    parser.add_argument("--families-per-stratum", type=int, default=20)
    parser.add_argument("--min-taxa", type=int, default=6)
    parser.add_argument("--min-codons", type=int, default=100)
    parser.add_argument("--max-codons", type=int, default=1500)
    parser.add_argument("--foreground", default="leaves")
    parser.add_argument("--allow-missing-start-codon", action="store_true")
    args = parser.parse_args()
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
