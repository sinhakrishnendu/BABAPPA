from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .hash_utils import sha256_file
from .io import Alignment
from .neutral import GY94NeutralSimulator, NeutralSpec


@dataclass
class FetchResult:
    dataset_json: Path
    metadata_tsv: Path
    fetch_manifest_json: Path
    synthetic_fallback: bool
    n_genes: int


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _star_tree(names: list[str], branch_length: float = 0.1) -> str:
    leaves = [f"{name}:{branch_length}" for name in names]
    return f"({','.join(leaves)});"


def _write_fasta(aln: Alignment, path: Path) -> None:
    lines: list[str] = []
    for name, seq in zip(aln.names, aln.sequences):
        lines.append(f">{name}")
        lines.append(seq)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_neutral_model(path: Path, tree_path: Path, kappa: float) -> Path:
    payload = {
        "schema_version": 1,
        "model_family": "GY94",
        "genetic_code_table": "standard",
        "tree_file": str(tree_path.resolve()),
        "kappa": float(kappa),
        "omega": 1.0,
        "codon_frequencies_method": "F3x4",
        "frozen_values": True,
    }
    _write_json(path, payload)
    return path


def _simulate_dataset(
    *,
    track: str,
    source_label: str,
    release: str,
    outdir: Path,
    taxa_names: list[str],
    max_genes: int,
    min_len_codons: int,
    max_len_codons: int,
    seed: int,
) -> FetchResult:
    outdir = outdir.resolve()
    genes_dir = outdir / "genes"
    genes_dir.mkdir(parents=True, exist_ok=True)

    tree_newick = _star_tree(taxa_names)
    tree_path = outdir / "tree.nwk"
    tree_path.write_text(tree_newick + "\n", encoding="utf-8")
    neutral_model_path = _write_neutral_model(outdir / "neutral_model.json", tree_path, kappa=2.0)

    spec = NeutralSpec(tree_newick=tree_newick, kappa=2.0, omega=1.0)
    sim = GY94NeutralSimulator(spec)
    rng = np.random.default_rng(seed)

    genes: list[dict[str, str]] = []
    metadata_rows: list[dict[str, Any]] = []
    for i in range(max_genes):
        length_codons = int(rng.integers(min_len_codons, max_len_codons + 1))
        length_nt = int(length_codons * 3)
        gene_id = f"{source_label}_{i + 1:04d}"
        gpath = genes_dir / f"{gene_id}.fasta"
        aln = sim.simulate_alignment(length_nt=length_nt, seed=int(rng.integers(0, 2**32 - 1)))
        _write_fasta(aln, gpath)
        digest = sha256_file(gpath)
        genes.append({"gene_id": gene_id, "alignment_path": str(gpath.resolve())})
        metadata_rows.append(
            {
                "gene_id": gene_id,
                "alignment_path": str(gpath.resolve()),
                "length_nt": length_nt,
                "n_taxa": len(taxa_names),
                "source": source_label,
                "release": release,
                "sha256": digest,
            }
        )

    metadata_tsv = outdir / "metadata.tsv"
    pd.DataFrame(metadata_rows).to_csv(metadata_tsv, sep="\t", index=False)

    dataset_payload = {
        "schema_version": 1,
        "tree_path": str(tree_path.resolve()),
        "genes": genes,
        "neutral_model_json": str(neutral_model_path.resolve()),
        "foreground_taxon": taxa_names[0],
        "metadata": {
            "source": source_label,
            "track": track,
            "release": release,
            "synthetic_fallback": True,
        },
    }
    if track == "ortholog":
        positives = [g["gene_id"] for g in genes[:: max(1, len(genes) // 10)]]
        dataset_payload["selectome_positive_genes"] = positives

    dataset_json = outdir / "dataset.json"
    _write_json(dataset_json, dataset_payload)

    fetch_manifest = {
        "schema_version": 1,
        "source": source_label,
        "track": track,
        "release": release,
        "seed": int(seed),
        "synthetic_fallback": True,
        "tree_path": str(tree_path.resolve()),
        "neutral_model_json": str(neutral_model_path.resolve()),
        "dataset_json": str(dataset_json.resolve()),
        "metadata_tsv": str(metadata_tsv.resolve()),
        "n_genes": len(genes),
        "n_taxa": len(taxa_names),
        "source_urls": {
            "orthomam": "https://orthomam.mbb.cnrs.fr/",
            "hiv_lanl": "https://www.hiv.lanl.gov/",
            "ncbi_virus": "https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/",
        },
    }
    fetch_manifest_json = outdir / "fetch_manifest.json"
    _write_json(fetch_manifest_json, fetch_manifest)
    return FetchResult(
        dataset_json=dataset_json,
        metadata_tsv=metadata_tsv,
        fetch_manifest_json=fetch_manifest_json,
        synthetic_fallback=True,
        n_genes=len(genes),
    )


def fetch_orthomam(
    *,
    release: str,
    species_set: str,
    min_length_codons: int,
    max_genes: int,
    outdir: str | Path,
    seed: int,
) -> FetchResult:
    names_small8 = ["Hsap", "Mmus", "Rnor", "Cfam", "Btau", "Sscr", "Mmul", "Ggal"]
    names_medium16 = names_small8 + [
        "Ptro",
        "Cjac",
        "Oane",
        "Tbel",
        "Lcha",
        "Ecab",
        "Fcat",
        "Ocun",
    ]
    key = species_set.lower().strip()
    if key == "small8":
        taxa = names_small8
    elif key == "medium16":
        taxa = names_medium16
    else:
        raise ValueError(f"Unsupported OrthoMaM species set: {species_set}")
    return _simulate_dataset(
        track="ortholog",
        source_label="orthomam",
        release=release,
        outdir=Path(outdir),
        taxa_names=taxa,
        max_genes=int(max_genes),
        min_len_codons=int(min_length_codons),
        max_len_codons=max(int(min_length_codons), int(min_length_codons) + 300),
        seed=int(seed),
    )


def fetch_hiv_env_b(
    *,
    release: str,
    min_length_codons: int,
    max_genes: int,
    outdir: str | Path,
    seed: int,
) -> FetchResult:
    taxa = [f"HIV_B_{i + 1:02d}" for i in range(24)]
    return _simulate_dataset(
        track="viral",
        source_label="hiv_env_b",
        release=release,
        outdir=Path(outdir),
        taxa_names=taxa,
        max_genes=int(max_genes),
        min_len_codons=int(min_length_codons),
        max_len_codons=max(int(min_length_codons), int(min_length_codons) + 240),
        seed=int(seed),
    )


def fetch_sars_2020(
    *,
    release: str,
    min_length_codons: int,
    max_genes: int,
    outdir: str | Path,
    seed: int,
) -> FetchResult:
    taxa = [f"SARS2_{i + 1:02d}" for i in range(32)]
    return _simulate_dataset(
        track="viral",
        source_label="sars_2020",
        release=release,
        outdir=Path(outdir),
        taxa_names=taxa,
        max_genes=int(max_genes),
        min_len_codons=int(min_length_codons),
        max_len_codons=max(int(min_length_codons), int(min_length_codons) + 180),
        seed=int(seed),
    )
