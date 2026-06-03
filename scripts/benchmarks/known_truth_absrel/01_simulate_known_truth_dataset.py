#!/usr/bin/env python
"""Create a lightweight known-truth codon benchmark dataset."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import config_int, read_config, resolve_outdir, write_fasta, write_json, write_tsv


CODONS = ["GCT", "GCC", "GCA", "GCG", "GGT", "GGC", "GGA", "GTT", "GTC", "AAG", "AAC", "CCT", "TTC", "TAC", "CAA"]
MUT_CODONS = ["CGT", "AGA", "TGG", "CAT", "GAA", "ACT", "ATT", "CTG"]
REGIMES = [
    ("null_low_divergence", "null", "in_domain"),
    ("null_moderate_divergence", "null", "in_domain"),
    ("null_alignment_difficult", "null", "borderline"),
    ("positive_weak_branch_site", "positive", "in_domain"),
    ("positive_moderate_branch_site", "positive", "in_domain"),
    ("positive_strong_branch_site", "positive", "in_domain"),
    ("positive_sparse_sites", "positive", "borderline"),
    ("ood_extreme_saturation", "ood_null", "out_of_domain"),
    ("ood_too_short", "ood_null", "out_of_domain"),
    ("ood_high_gap", "ood_null", "out_of_domain"),
    ("ood_positive_extreme", "ood_positive", "out_of_domain"),
    ("null_long_branch", "null", "borderline"),
]


def _build_tree(taxa: List[str], branch_length: float) -> str:
    tips = ",".join(f"{taxon}:{branch_length:.3f}" for taxon in taxa)
    return f"({tips});\n"


def _mutate_codon(codon: str, rng: random.Random, high: bool = False) -> str:
    choices = MUT_CODONS if high else CODONS
    replacement = rng.choice(choices)
    return replacement if replacement != codon else rng.choice(choices)


def _simulate_family(family_id: str, regime: str, truth_class: str, applicability: str, seed: int, n_taxa: int, n_codons: int) -> Dict[str, object]:
    rng = random.Random(f"{seed}:{family_id}")
    if "short" in regime:
        n_codons = min(n_codons, 45)
    taxa = [f"taxon{i+1}" for i in range(n_taxa)]
    foreground = taxa[0]
    base = [rng.choice(CODONS) for _ in range(n_codons)]
    selected_sites: List[int] = []
    if truth_class in {"positive", "ood_positive"}:
        fraction = 0.03 if "sparse" in regime else 0.08
        selected_sites = sorted(rng.sample(range(n_codons), max(1, int(n_codons * fraction))))
    records: Dict[str, str] = {}
    for taxon_index, taxon in enumerate(taxa):
        seq = list(base)
        drift_rate = 0.02
        if "moderate" in regime:
            drift_rate = 0.06
        if "alignment_difficult" in regime:
            drift_rate = 0.10
        if applicability == "out_of_domain":
            drift_rate = 0.25
        for site in range(n_codons):
            if rng.random() < drift_rate * (taxon_index + 1) / max(n_taxa, 1):
                seq[site] = _mutate_codon(seq[site], rng, high=applicability == "out_of_domain")
        if taxon == foreground:
            for site in selected_sites:
                seq[site] = _mutate_codon(seq[site], rng, high=True)
        records[taxon] = "".join(seq)
    return {
        "records": records,
        "tree": _build_tree(taxa, 0.65 if applicability == "out_of_domain" else 0.1),
        "foreground": foreground,
        "selected_sites": selected_sites,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    config = read_config(args.config)
    outdir = resolve_outdir(config, args.outdir)
    n_families = config_int(config, "n_families", 12)
    seed = config_int(config, "seed", 42)
    n_taxa = config_int(config, "n_taxa", 6)
    n_codons = config_int(config, "n_codons", 90)

    families_dir = outdir / "families"
    truth_dir = outdir / "truth"
    rows = []
    branch_site_rows = []
    selected_site_rows = []
    selected_branch_rows = []
    for i in range(n_families):
        regime, truth_class, applicability = REGIMES[i % len(REGIMES)]
        family_id = f"SIM{i+1:05d}_{regime}"
        simulated = _simulate_family(family_id, regime, truth_class, applicability, seed + i, n_taxa, n_codons)
        family_dir = families_dir / family_id
        codon_fasta = family_dir / "codon.fasta"
        tree_path = family_dir / "tree.nwk"
        write_fasta(codon_fasta, simulated["records"])  # type: ignore[arg-type]
        family_dir.mkdir(parents=True, exist_ok=True)
        tree_path.write_text(str(simulated["tree"]), encoding="utf-8")
        foreground = str(simulated["foreground"])
        selected_sites = list(simulated["selected_sites"])  # type: ignore[arg-type]
        row = {
            "family_id": family_id,
            "regime": regime,
            "truth_class": truth_class,
            "expected_applicability": applicability,
            "codon_fasta": str(codon_fasta.relative_to(outdir)),
            "tree": str(tree_path.relative_to(outdir)),
            "foreground": foreground,
            "positive_branch": foreground if truth_class in {"positive", "ood_positive"} else "",
            "n_taxa": n_taxa,
            "n_codons": min(n_codons, 45) if "short" in regime else n_codons,
            "selected_site_count": len(selected_sites),
        }
        rows.append(row)
        if row["positive_branch"]:
            selected_branch_rows.append({"family_id": family_id, "branch": foreground, "truth_class": truth_class})
        for site in selected_sites:
            selected_site_rows.append({"family_id": family_id, "site_index_one_based": site + 1, "foreground": foreground})
            branch_site_rows.append({"family_id": family_id, "branch": foreground, "site_index_one_based": site + 1, "selected": 1})
        write_json(family_dir / "family_truth.json", row)

    fields = ["family_id", "regime", "truth_class", "expected_applicability", "codon_fasta", "tree", "foreground", "positive_branch", "n_taxa", "n_codons", "selected_site_count"]
    write_tsv(outdir / "manifest.tsv", rows, fields)
    write_tsv(truth_dir / "family_truth.tsv", rows, fields)
    write_tsv(truth_dir / "branch_site_truth.tsv", branch_site_rows, ["family_id", "branch", "site_index_one_based", "selected"])
    write_tsv(truth_dir / "selected_sites.tsv", selected_site_rows, ["family_id", "site_index_one_based", "foreground"])
    write_tsv(truth_dir / "selected_branches.tsv", selected_branch_rows, ["family_id", "branch", "truth_class"])
    write_json(outdir / "simulation_manifest.json", {"status": "ok", "n_families": n_families, "seed": seed, "outdir": str(outdir)})
    print(f"Simulated {n_families} known-truth families under {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
