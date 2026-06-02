#!/usr/bin/env python3
"""Prepare OrthoFinder protein inputs from NCBI Drosophila CDS FASTA files.

This script is intentionally standalone so the publication benchmark can be
run without changing BABAPPA's package code. It translates CDS records to
proteins, keeps the longest isoform per gene per species, and writes matched
protein/CDS files plus metadata for later single-copy ortholog extraction.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


@dataclass
class CdsRecord:
    header: str
    seq: str


@dataclass
class SelectedRecord:
    species: str
    assembly: str
    source: str
    sequence_id: str
    gene: str
    protein_id: str
    cds: str
    protein: str
    original_header: str


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def parse_fasta(path: Path):
    header = None
    chunks: list[str] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield CdsRecord(header, "".join(chunks))
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        yield CdsRecord(header, "".join(chunks))


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for header, seq in records:
            handle.write(f">{header}\n")
            for idx in range(0, len(seq), 80):
                handle.write(seq[idx : idx + 80] + "\n")


def parse_bracket_fields(header: str) -> dict[str, str]:
    return {k: v for k, v in re.findall(r"\[([^=\]]+)=([^\]]+)\]", header)}


def translate_cds(seq: str) -> tuple[str | None, str]:
    cleaned = re.sub(r"\s+", "", seq.upper().replace("U", "T"))
    cleaned = cleaned.replace("-", "")
    if len(cleaned) % 3 != 0:
        return None, "length_not_divisible_by_3"
    protein = []
    for idx in range(0, len(cleaned), 3):
        codon = cleaned[idx : idx + 3]
        if codon in CODON_TABLE:
            protein.append(CODON_TABLE[codon])
        elif set(codon) <= set("ACGTNRYMKSWHBVD"):
            protein.append("X")
        else:
            return None, f"invalid_codon:{codon}"
    aa = "".join(protein)
    if "*" in aa[:-1]:
        return None, "internal_stop"
    if aa.endswith("*"):
        aa = aa[:-1]
    if not aa:
        return None, "empty_protein"
    return aa, "ok"


def load_summary(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            acc = row.get("Assembly Accession", "").strip()
            if acc:
                rows[acc] = row
    return rows


def choose_assemblies(data_dir: Path, summary: dict[str, dict[str, str]], prefer_refseq: bool) -> list[str]:
    fasta_assemblies = {
        path.parent.name
        for path in data_dir.glob("*/cds_from_genomic.fasta")
    }
    if not prefer_refseq:
        return sorted(fasta_assemblies)
    by_species: dict[str, list[str]] = {}
    for acc in fasta_assemblies:
        species = summary.get(acc, {}).get("Organism Scientific Name", acc)
        by_species.setdefault(species, []).append(acc)
    selected = []
    for species, accessions in sorted(by_species.items()):
        gcf = sorted(acc for acc in accessions if acc.startswith("GCF_"))
        selected.append(gcf[0] if gcf else sorted(accessions)[0])
    return selected


def prepare(args: argparse.Namespace) -> int:
    data_dir = Path(args.ncbi_data_dir)
    outdir = Path(args.outdir)
    summary = load_summary(data_dir / "data_summary.tsv")
    selected_assemblies = choose_assemblies(data_dir, summary, args.prefer_refseq)

    protein_dir = outdir / "orthofinder_input"
    cds_dir = outdir / "curated_cds"
    meta_dir = outdir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    all_metadata: list[dict[str, str | int]] = []
    summary_rows: list[dict[str, str | int]] = []

    for assembly in selected_assemblies:
        row = summary.get(assembly, {})
        species = safe_name(row.get("Organism Scientific Name", assembly))
        source = row.get("Source", "unknown")
        fasta = data_dir / assembly / "cds_from_genomic.fasta"
        if not fasta.exists():
            summary_rows.append({
                "species": species,
                "assembly": assembly,
                "status": "missing_cds",
                "selected_records": 0,
                "rejected_records": 0,
            })
            continue

        longest_by_gene: dict[str, SelectedRecord] = {}
        rejection_counts: dict[str, int] = {}
        n_records = 0

        for record in parse_fasta(fasta):
            n_records += 1
            fields = parse_bracket_fields(record.header)
            protein_id = fields.get("protein_id") or record.header.split()[0]
            gene = fields.get("gene") or fields.get("locus_tag") or protein_id
            gene = safe_name(gene)
            protein_id_safe = safe_name(protein_id)
            protein, reason = translate_cds(record.seq)
            if protein is None:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            if len(protein) < args.min_protein_aa:
                rejection_counts["too_short_protein"] = rejection_counts.get("too_short_protein", 0) + 1
                continue

            seq_id = f"{species}__{gene}__{protein_id_safe}"
            selected = SelectedRecord(
                species=species,
                assembly=assembly,
                source=source,
                sequence_id=seq_id,
                gene=gene,
                protein_id=protein_id,
                cds=re.sub(r"\s+", "", record.seq.upper().replace("U", "T")),
                protein=protein,
                original_header=record.header,
            )
            old = longest_by_gene.get(gene)
            if old is None or len(selected.protein) > len(old.protein):
                longest_by_gene[gene] = selected

        chosen = sorted(longest_by_gene.values(), key=lambda item: item.sequence_id)
        write_fasta([(item.sequence_id, item.protein) for item in chosen], protein_dir / f"{species}.faa")
        write_fasta([(item.sequence_id, item.cds) for item in chosen], cds_dir / f"{species}.cds.fasta")

        for item in chosen:
            all_metadata.append({
                "species": item.species,
                "assembly": item.assembly,
                "source": item.source,
                "sequence_id": item.sequence_id,
                "gene": item.gene,
                "protein_id": item.protein_id,
                "cds_length": len(item.cds),
                "protein_length": len(item.protein),
                "original_header": item.original_header,
            })

        summary_rows.append({
            "species": species,
            "assembly": assembly,
            "source": source,
            "status": "ok",
            "raw_cds_records": n_records,
            "selected_longest_gene_records": len(chosen),
            "rejected_records": sum(rejection_counts.values()),
            "rejection_counts": json.dumps(rejection_counts, sort_keys=True),
        })

    with (meta_dir / "cds_protein_map.tsv").open("w", newline="") as handle:
        fieldnames = [
            "species", "assembly", "source", "sequence_id", "gene",
            "protein_id", "cds_length", "protein_length", "original_header",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_metadata)

    with (outdir / "orthofinder_input_summary.tsv").open("w", newline="") as handle:
        fieldnames = [
            "species", "assembly", "source", "status", "raw_cds_records",
            "selected_longest_gene_records", "rejected_records", "rejection_counts",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    with (outdir / "orthofinder_input_summary.md").open("w") as handle:
        handle.write("# Drosophila OrthoFinder Input Preparation\n\n")
        handle.write(f"- NCBI data directory: `{data_dir}`\n")
        handle.write(f"- Species/assemblies selected: {len(selected_assemblies)}\n")
        handle.write(f"- Protein FASTA directory: `{protein_dir}`\n")
        handle.write(f"- Matched CDS directory: `{cds_dir}`\n")
        handle.write(f"- Metadata: `{meta_dir / 'cds_protein_map.tsv'}`\n\n")
        handle.write("RefSeq assemblies are preferred when duplicate GenBank/RefSeq assemblies exist.\n")

    print(f"Prepared {len(selected_assemblies)} species for OrthoFinder")
    print(f"Protein input: {protein_dir}")
    print(f"Summary: {outdir / 'orthofinder_input_summary.tsv'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncbi-data-dir", default="drosophilas/ncbi_dataset/data")
    parser.add_argument("--outdir", default="publication_benchmark/drosophila_orthofinder")
    parser.add_argument("--prefer-refseq", action="store_true", help="Prefer GCF RefSeq assemblies over duplicate GCA assemblies.")
    parser.add_argument("--min-protein-aa", type=int, default=30)
    args = parser.parse_args()
    return prepare(args)


if __name__ == "__main__":
    raise SystemExit(main())
