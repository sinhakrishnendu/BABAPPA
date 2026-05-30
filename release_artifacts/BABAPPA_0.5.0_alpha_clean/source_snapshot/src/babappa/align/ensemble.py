"""Internal alignment-ensemble scaffold for BABAPPA."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

from babappa import __version__
from babappa.simulate.audit import read_fasta

ALIGNER_SCAFFOLD_VERSION = __version__
ALLOWED_ALIGNMENT_METHODS = {"identity", "codon_dropout"}
METHOD_NOTES = {
    "identity": "identity_schema",
    "codon_dropout": "codon_dropout_schema",
}


@dataclass(frozen=True)
class AlignmentConfig:
    """Configuration for BABAPPA internal alignment scaffold generation."""

    sim_dir: str
    outdir: str
    methods: List[str] = field(
        default_factory=lambda: ["identity", "codon_dropout"]
    )
    seed: int = 42
    dropout_rate: float = 0.02

    def __post_init__(self) -> None:
        sim_path = Path(self.sim_dir)
        if not sim_path.exists():
            raise ValueError(f"sim_dir does not exist: {sim_path}")
        if not (sim_path / "manifest.json").exists():
            raise ValueError(f"sim_dir is missing manifest.json: {sim_path}")
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        if not self.methods:
            raise ValueError("methods must be non-empty")
        unknown_methods = sorted(set(self.methods) - ALLOWED_ALIGNMENT_METHODS)
        if unknown_methods:
            allowed = ", ".join(sorted(ALLOWED_ALIGNMENT_METHODS))
            unknown = ", ".join(unknown_methods)
            raise ValueError(f"unknown alignment method(s): {unknown}; allowed: {allowed}")


def write_fasta(records: Dict[str, str], path: Path) -> None:
    """Write FASTA records using stable insertion order."""
    with path.open("w", encoding="utf-8") as handle:
        for record_id, sequence in records.items():
            handle.write(f">{record_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(f"{sequence[start:start + 80]}\n")


def align_simulation_directory(config: AlignmentConfig) -> dict:
    """Create internal alignment channels for every simulated family."""
    sim_path = Path(config.sim_dir)
    outdir = Path(config.outdir)
    families_outdir = outdir / "families"
    families_outdir.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(sim_path / "manifest.json")
    family_ids = manifest.get("family_ids")
    if not isinstance(family_ids, list):
        raise ValueError("simulation manifest does not contain a family_ids list")

    created_files: Dict[str, Dict[str, Dict[str, str]]] = {}
    for family_index, family_id in enumerate(family_ids):
        if not isinstance(family_id, str):
            raise ValueError("simulation manifest contains a non-string family id")
        source_family_dir = sim_path / "families" / family_id
        source_fasta = source_family_dir / f"{family_id}.fasta"
        records = read_fasta(source_fasta)
        family_outdir = families_outdir / family_id
        family_outdir.mkdir(parents=True, exist_ok=True)
        created_files[family_id] = {}

        for method in config.methods:
            method_records = _make_method_records(
                records=records,
                method=method,
                config=config,
                family_index=family_index,
            )
            output_paths = _write_method_outputs(
                family_id=family_id,
                method=method,
                records=method_records,
                family_outdir=family_outdir,
                source_family_dir=source_family_dir,
                source_fasta=source_fasta,
            )
            created_files[family_id][method] = {
                key: str(path.relative_to(outdir))
                for key, path in output_paths.items()
            }

    alignment_manifest_path = outdir / "alignment_manifest.json"
    alignment_manifest = {
        "aligner_scaffold_version": ALIGNER_SCAFFOLD_VERSION,
        "sim_dir": str(sim_path),
        "n_families": len(family_ids),
        "family_ids": family_ids,
        "methods": list(config.methods),
        "seed": config.seed,
        "dropout_rate": config.dropout_rate,
        "created_files": created_files,
    }
    _write_json(alignment_manifest_path, alignment_manifest)

    return {
        "status": "ok",
        "outdir": str(outdir),
        "n_families": len(family_ids),
        "methods": list(config.methods),
        "manifest": str(alignment_manifest_path),
    }


def _make_method_records(
    records: Dict[str, str],
    method: str,
    config: AlignmentConfig,
    family_index: int,
) -> Dict[str, str]:
    if method == "identity":
        return dict(records)
    if method == "codon_dropout":
        rng = random.Random(_method_seed(config.seed, family_index, method))
        return {
            record_id: _apply_codon_dropout(sequence, rng, config.dropout_rate)
            for record_id, sequence in records.items()
        }
    raise ValueError(f"unsupported alignment method: {method}")


def _apply_codon_dropout(sequence: str, rng: random.Random, dropout_rate: float) -> str:
    if len(sequence) % 3 != 0:
        raise ValueError("source sequence length is not a multiple of 3")
    codons = []
    for start in range(0, len(sequence), 3):
        codon = sequence[start:start + 3]
        if rng.random() < dropout_rate:
            codons.append("---")
        else:
            codons.append(codon)
    return "".join(codons)


def _method_seed(seed: int, family_index: int, method: str) -> int:
    method_offset = sum((index + 1) * ord(char) for index, char in enumerate(method))
    return seed + (family_index + 1) * 100_003 + method_offset


def _write_method_outputs(
    family_id: str,
    method: str,
    records: Dict[str, str],
    family_outdir: Path,
    source_family_dir: Path,
    source_fasta: Path,
) -> Dict[str, Path]:
    codon_fasta_path = family_outdir / f"{family_id}.{method}.codon.fasta"
    map_path = family_outdir / f"{family_id}.{method}.map.tsv"
    qc_path = family_outdir / f"{family_id}.{method}.qc.json"

    write_fasta(records, codon_fasta_path)
    _write_alignment_map(records, map_path, method)
    _write_json(
        qc_path,
        _build_qc_payload(
            family_id=family_id,
            method=method,
            records=records,
            source_family_dir=source_family_dir,
            source_fasta=source_fasta,
        ),
    )

    return {
        "codon_fasta": codon_fasta_path,
        "map": map_path,
        "qc": qc_path,
    }


def _write_alignment_map(records: Dict[str, str], path: Path, method: str) -> None:
    alignment_length_codons = _alignment_length_codons(records)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("alignment_column_0based\thomology_id\tnote\n")
        for codon_index in range(alignment_length_codons):
            handle.write(
                f"{codon_index}\tH{codon_index + 1:06d}\t{METHOD_NOTES[method]}\n"
            )


def _build_qc_payload(
    family_id: str,
    method: str,
    records: Dict[str, str],
    source_family_dir: Path,
    source_fasta: Path,
) -> dict:
    lengths = [len(sequence) for sequence in records.values()]
    sequence_length_equal = len(set(lengths)) == 1
    alignment_length_nt = lengths[0] if lengths else 0
    alignment_length_codons = alignment_length_nt // 3 if alignment_length_nt else 0
    gap_codon_count = sum(
        1
        for sequence in records.values()
        for start in range(0, len(sequence), 3)
        if sequence[start:start + 3] == "---"
    )
    total_codons = len(records) * alignment_length_codons
    gap_codon_fraction = 0.0 if total_codons == 0 else gap_codon_count / total_codons

    return {
        "family_id": family_id,
        "method": method,
        "n_taxa": len(records),
        "alignment_length_codons": alignment_length_codons,
        "alignment_length_nt": alignment_length_nt,
        "gap_codon_count": gap_codon_count,
        "gap_codon_fraction": gap_codon_fraction,
        "sequence_length_equal": sequence_length_equal,
        "source_family_dir": str(source_family_dir),
        "source_fasta": str(source_fasta),
        "note": METHOD_NOTES[method],
    }


def _alignment_length_codons(records: Dict[str, str]) -> int:
    if not records:
        return 0
    first_sequence = next(iter(records.values()))
    if len(first_sequence) % 3 != 0:
        raise ValueError("alignment length is not a multiple of 3")
    return len(first_sequence) // 3


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
