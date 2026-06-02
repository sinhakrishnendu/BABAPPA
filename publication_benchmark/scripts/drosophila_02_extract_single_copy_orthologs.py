#!/usr/bin/env python3
"""Extract single-copy ortholog CDS families from an OrthoFinder run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    header = None
    chunks: list[str] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(chunks)
                header = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        records[header] = "".join(chunks)
    return records


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for header, seq in records:
            handle.write(f">{header}\n")
            for idx in range(0, len(seq), 80):
                handle.write(seq[idx : idx + 80] + "\n")


def latest_results_dir(root: Path) -> Path:
    if (root / "Orthogroups").exists():
        return root
    candidates = sorted(
        root.glob("**/Results_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No OrthoFinder Results_* directory found under {root}")
    return candidates[0]


def load_single_copy_ids(results: Path) -> set[str]:
    path = results / "Orthogroups" / "Orthogroups_SingleCopyOrthologues.txt"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def cell_genes(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_cds_records(prepared_dir: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for fasta in sorted((prepared_dir / "curated_cds").glob("*.cds.fasta")):
        records.update(parse_fasta(fasta))
    return records


def extract(args: argparse.Namespace) -> int:
    prepared_dir = Path(args.prepared_dir)
    results = latest_results_dir(Path(args.orthofinder_root))
    outdir = Path(args.outdir)
    cds_out = outdir / "single_copy_cds"
    outdir.mkdir(parents=True, exist_ok=True)

    orthogroups_tsv = results / "Orthogroups" / "Orthogroups.tsv"
    if not orthogroups_tsv.exists():
        raise FileNotFoundError(f"Missing OrthoFinder table: {orthogroups_tsv}")

    cds_records = load_cds_records(prepared_dir)
    single_copy_ids = load_single_copy_ids(results)
    rows: list[dict[str, str | int]] = []
    panel_rows: list[dict[str, str]] = []
    n_written = 0

    with orthogroups_tsv.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        species_cols = [field for field in reader.fieldnames or [] if field != "Orthogroup"]
        required_species = args.min_species or len(species_cols)
        for row in reader:
            orthogroup = row["Orthogroup"]
            if single_copy_ids and orthogroup not in single_copy_ids:
                continue
            chosen: list[tuple[str, str]] = []
            missing = []
            multi = []
            for species in species_cols:
                genes = cell_genes(row.get(species, ""))
                if len(genes) == 1:
                    seq = cds_records.get(genes[0])
                    if seq:
                        chosen.append((genes[0], seq))
                    else:
                        missing.append(species)
                elif len(genes) == 0:
                    missing.append(species)
                else:
                    multi.append(species)
            if multi:
                continue
            if len(chosen) < required_species:
                continue

            if args.max_orthogroups and n_written >= args.max_orthogroups:
                break

            cds_path = cds_out / f"{orthogroup}.cds.fasta"
            write_fasta(chosen, cds_path)
            n_written += 1

            rows.append({
                "orthogroup": orthogroup,
                "n_species": len(chosen),
                "cds_fasta": str(cds_path),
                "missing_species": ",".join(missing),
            })
            panel_rows.append({
                "panel_id": orthogroup,
                "gene_family": orthogroup,
                "cds_unaligned": str(cds_path),
                "tree_file": "",
                "foreground": "leaves",
                "expected_category": "unknown",
                "notes": "Single-copy Drosophila ortholog set from OrthoFinder; align and build a tree before BABAPPA benchmarking.",
            })

    with (outdir / "single_copy_orthologs.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["orthogroup", "n_species", "cds_fasta", "missing_species"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    with (outdir / "babappa_unaligned_panel.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["panel_id", "gene_family", "cds_unaligned", "tree_file", "foreground", "expected_category", "notes"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(panel_rows)

    with (outdir / "single_copy_orthologs_summary.md").open("w") as handle:
        handle.write("# Drosophila Single-Copy Ortholog Extraction\n\n")
        handle.write(f"- OrthoFinder results: `{results}`\n")
        handle.write(f"- Prepared CDS metadata: `{prepared_dir}`\n")
        handle.write(f"- Single-copy CDS FASTA files written: {n_written}\n")
        handle.write(f"- Table: `{outdir / 'single_copy_orthologs.tsv'}`\n")
        handle.write(f"- BABAPPA pre-panel: `{outdir / 'babappa_unaligned_panel.tsv'}`\n\n")
        handle.write("These CDS files are unaligned. Build codon MSAs and trees before running BABAPPA direct prediction.\n")

    print(f"Extracted {n_written} single-copy ortholog CDS families")
    print(f"Output: {outdir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orthofinder-root", default="publication_benchmark/drosophila_orthofinder/orthofinder_run")
    parser.add_argument("--prepared-dir", default="publication_benchmark/drosophila_orthofinder")
    parser.add_argument("--outdir", default="publication_benchmark/drosophila_orthofinder/single_copy_orthologs")
    parser.add_argument("--min-species", type=int, default=0, help="Require at least this many species; default requires all OrthoFinder species.")
    parser.add_argument("--max-orthogroups", type=int, default=500, help="Maximum CDS families to write; use 0 for all.")
    args = parser.parse_args()
    return extract(args)


if __name__ == "__main__":
    raise SystemExit(main())
