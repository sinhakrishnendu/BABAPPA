"""Tensor dataset builder for BABAPPA alignment scaffold outputs."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from babappa import __version__
from babappa.simulate.audit import read_fasta
from babappa.simulate.simulator import STANDARD_GENETIC_CODE

TENSORIZER_VERSION = __version__
PAD_OR_UNKNOWN_ID = 0
GAP_CODON_ID = 1
TENSOR_AUDIT_FIELDNAMES = [
    "family_id",
    "method",
    "tensor_file",
    "n_taxa",
    "n_codons",
    "n_channels",
    "gap_codon_count",
    "gap_codon_fraction",
    "gene_label",
    "saturation_tier",
    "status",
    "warning",
]


@dataclass(frozen=True)
class TensorBuildConfig:
    """Configuration for converting BABAPPA alignments into tensor shards."""

    sim_dir: str
    align_dir: str
    outdir: str
    methods: Optional[List[str]] = None
    include_gap_channel: bool = True
    workers: int = 1

    def __post_init__(self) -> None:
        sim_path = Path(self.sim_dir)
        align_path = Path(self.align_dir)
        out_path = Path(self.outdir)
        if not sim_path.exists():
            raise ValueError(f"sim_dir does not exist: {sim_path}")
        if not (sim_path / "manifest.json").exists():
            raise ValueError(f"sim_dir is missing manifest.json: {sim_path}")
        if not align_path.exists():
            raise ValueError(f"align_dir does not exist: {align_path}")
        alignment_manifest_path = align_path / "alignment_manifest.json"
        if not alignment_manifest_path.exists():
            raise ValueError(f"align_dir is missing alignment_manifest.json: {align_path}")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")

        alignment_manifest = _read_json(alignment_manifest_path)
        available_methods = alignment_manifest.get("methods")
        if not isinstance(available_methods, list) or not available_methods:
            raise ValueError("alignment manifest does not contain non-empty methods")

        if self.methods is None:
            resolved_methods = list(available_methods)
        else:
            if not self.methods:
                resolved_methods = list(available_methods)
            else:
                unknown_methods = sorted(set(self.methods) - set(available_methods))
                if unknown_methods:
                    unknown = ", ".join(unknown_methods)
                    allowed = ", ".join(str(method) for method in available_methods)
                    raise ValueError(
                        f"unknown tensor method(s): {unknown}; available: {allowed}"
                    )
                resolved_methods = list(self.methods)

        object.__setattr__(self, "methods", resolved_methods)
        out_path.mkdir(parents=True, exist_ok=True)


def build_codon_vocab() -> Dict[str, int]:
    """Build deterministic codon-token vocabulary."""
    sense_codons = sorted(
        codon for codon, amino_acid in STANDARD_GENETIC_CODE.items() if amino_acid != "*"
    )
    vocab = {"PAD_OR_UNKNOWN": PAD_OR_UNKNOWN_ID, "---": GAP_CODON_ID}
    for offset, codon in enumerate(sense_codons, start=2):
        vocab[codon] = offset
    return vocab


def codon_to_id(codon: str, vocab: Dict[str, int]) -> int:
    """Convert a codon string into a deterministic integer token."""
    normalized = codon.upper().replace("U", "T")
    if normalized == "---":
        return vocab["---"]
    return vocab.get(normalized, vocab["PAD_OR_UNKNOWN"])


def read_codon_alignment_as_codons(path: Path) -> Dict[str, List[str]]:
    """Read codon-alignment FASTA and split records into codons."""
    fasta_records = read_fasta(path)
    codon_records: Dict[str, List[str]] = {}
    lengths = []
    for taxon, sequence in fasta_records.items():
        if len(sequence) % 3 != 0:
            raise ValueError(f"alignment sequence length is not a multiple of 3: {taxon}")
        codons = [
            sequence[start:start + 3].upper()
            for start in range(0, len(sequence), 3)
        ]
        codon_records[taxon] = codons
        lengths.append(len(codons))

    if len(set(lengths)) != 1:
        raise ValueError(f"alignment records have unequal codon lengths: {path}")
    return codon_records


def alignment_to_tensor(
    records: Dict[str, List[str]],
    vocab: Dict[str, int],
    include_gap_channel: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Convert codon alignment records into an integer NumPy tensor."""
    if not records:
        raise ValueError("alignment records are empty")
    taxa_order = list(records.keys())
    n_taxa = len(taxa_order)
    n_codons = len(records[taxa_order[0]])
    if any(len(codons) != n_codons for codons in records.values()):
        raise ValueError("alignment records have unequal codon counts")

    n_channels = 2 if include_gap_channel else 1
    tensor = np.zeros((n_taxa, n_codons, n_channels), dtype=np.int32)
    gap_codon_count = 0
    for taxon_index, taxon in enumerate(taxa_order):
        for codon_index, codon in enumerate(records[taxon]):
            tensor[taxon_index, codon_index, 0] = codon_to_id(codon, vocab)
            if codon == "---":
                gap_codon_count += 1
                if include_gap_channel:
                    tensor[taxon_index, codon_index, 1] = 1

    total_codons = n_taxa * n_codons
    metadata = {
        "n_taxa": n_taxa,
        "n_codons": n_codons,
        "n_channels": n_channels,
        "taxa_order": taxa_order,
        "gap_codon_count": gap_codon_count,
        "gap_codon_fraction": 0.0 if total_codons == 0 else gap_codon_count / total_codons,
    }
    return tensor, metadata


def load_truth_labels(truth_json: Path) -> dict:
    """Load simulator truth labels in the tensorizer label schema."""
    truth = _read_json(truth_json)
    labels = truth.get("labels", {})
    selected_sites_0based = truth.get("selected_sites_0based", [])
    selected_sites_1based = truth.get("selected_sites_1based", [])
    branch_truth_path = truth_json.with_name(truth_json.name.replace(".truth.json", ".branch_truth.json"))
    payload = {
        "family_id": truth.get("family_id"),
        "gene_label": labels.get("gene_label"),
        "has_positive_selection": truth.get("has_positive_selection"),
        "foreground_taxon": truth.get("foreground_taxon"),
        "branch_labels": labels.get("branch_labels", {}),
        "selected_sites_0based": selected_sites_0based,
        "selected_sites_1based": selected_sites_1based,
        "n_selected_sites": len(selected_sites_0based)
        if isinstance(selected_sites_0based, list)
        else 0,
        "saturation_tier": truth.get("saturation_tier"),
    }
    if branch_truth_path.exists():
        payload["branch_truth_file"] = str(branch_truth_path)
        payload["explicit_branch_site_truth_available"] = True
        payload["branch_truth_source"] = "explicit_simulator_branch_truth"
    else:
        payload["explicit_branch_site_truth_available"] = False
    return payload


def build_tensor_dataset(config: TensorBuildConfig) -> dict:
    """Build tensor shards and manifests from simulation and alignment outputs."""
    sim_path = Path(config.sim_dir)
    align_path = Path(config.align_dir)
    outdir = Path(config.outdir)
    families_outdir = outdir / "families"
    families_outdir.mkdir(parents=True, exist_ok=True)

    simulation_manifest = _read_json(sim_path / "manifest.json")
    alignment_manifest = _read_json(align_path / "alignment_manifest.json")
    family_ids = alignment_manifest.get("family_ids")
    if not isinstance(family_ids, list):
        raise ValueError("alignment manifest does not contain a family_ids list")
    methods = config.methods or alignment_manifest["methods"]
    alignment_created_files = alignment_manifest.get("created_files", {})
    use_manifest_created_files = isinstance(alignment_created_files, dict) and bool(
        alignment_created_files
    )
    vocab = build_codon_vocab()
    created_files: Dict[str, Dict[str, object]] = {}
    audit_rows: List[dict] = []

    task_payloads = [
        (
            str(sim_path),
            str(align_path),
            str(outdir),
            str(family_id),
            list(methods),
            alignment_created_files.get(str(family_id), {}) if use_manifest_created_files else {},
            bool(use_manifest_created_files),
            dict(vocab),
            bool(config.include_gap_channel),
        )
        for family_id in family_ids
    ]
    if any(not isinstance(family_id, str) for family_id in family_ids):
        raise ValueError("alignment manifest contains a non-string family id")
    if config.workers <= 1 or len(task_payloads) <= 1:
        results = [_build_tensor_family_task(payload) for payload in task_payloads]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=min(config.workers, len(task_payloads))) as executor:
            futures = {executor.submit(_build_tensor_family_task, payload): payload[3] for payload in task_payloads}
            for future in as_completed(futures):
                results.append(future.result())

    for result in sorted(results, key=lambda item: item["family_id"]):
        created_files[result["family_id"]] = result["created_files"]
        audit_rows.extend(result["audit_rows"])

    tensor_manifest_path = outdir / "tensor_manifest.json"
    tensor_audit_path = outdir / "tensor_audit.tsv"
    _write_tensor_audit(tensor_audit_path, audit_rows)
    tensor_manifest = {
        "tensorizer_version": TENSORIZER_VERSION,
        "sim_dir": str(sim_path),
        "align_dir": str(align_path),
        "n_families": len(family_ids),
        "family_ids": family_ids,
        "methods": list(methods),
        "include_gap_channel": config.include_gap_channel,
        "workers": config.workers,
        "codon_vocab": vocab,
        "created_files": created_files,
        "source_simulator_version": simulation_manifest.get("simulator_version"),
    }
    _write_json(tensor_manifest_path, tensor_manifest)

    return {
        "status": "ok",
        "outdir": str(outdir),
        "n_families": len(family_ids),
        "methods": list(methods),
        "manifest": str(tensor_manifest_path),
        "audit": str(tensor_audit_path),
    }


def _build_tensor_family_task(payload: tuple) -> dict:
    (
        sim_path_s,
        align_path_s,
        outdir_s,
        family_id,
        methods,
        family_alignment_files,
        use_manifest_created_files,
        vocab,
        include_gap_channel,
    ) = payload
    sim_path = Path(sim_path_s)
    align_path = Path(align_path_s)
    outdir = Path(outdir_s)
    family_outdir = outdir / "families" / family_id
    family_outdir.mkdir(parents=True, exist_ok=True)
    truth_path = sim_path / "families" / family_id / f"{family_id}.truth.json"
    labels = load_truth_labels(truth_path)
    labels_path = family_outdir / f"{family_id}.labels.json"
    _write_json(labels_path, labels)
    created = {"labels": str(labels_path.relative_to(outdir))}
    audit_rows: List[dict] = []

    if use_manifest_created_files and not isinstance(family_alignment_files, dict):
        family_alignment_files = {}
    for method in methods:
        if use_manifest_created_files and method not in family_alignment_files:
            continue
        audit_row = _build_family_method_tensor(
            family_id=family_id,
            method=method,
            align_path=align_path,
            outdir=outdir,
            family_outdir=family_outdir,
            truth_path=truth_path,
            labels=labels,
            vocab=vocab,
            include_gap_channel=include_gap_channel,
        )
        audit_rows.append(audit_row)
        created[method] = {
            "tensor": audit_row.get("tensor_file", ""),
            "meta": audit_row.get("meta_file", ""),
        }
    return {"family_id": family_id, "created_files": created, "audit_rows": audit_rows}


def _build_family_method_tensor(
    family_id: str,
    method: str,
    align_path: Path,
    outdir: Path,
    family_outdir: Path,
    truth_path: Path,
    labels: dict,
    vocab: Dict[str, int],
    include_gap_channel: bool,
) -> dict:
    tensor_path = family_outdir / f"{family_id}.{method}.tensor.npz"
    meta_path = family_outdir / f"{family_id}.{method}.tensor_meta.json"
    source_alignment = (
        align_path / "families" / family_id / f"{family_id}.{method}.codon.fasta"
    )
    try:
        records = read_codon_alignment_as_codons(source_alignment)
        tensor, metadata = alignment_to_tensor(
            records=records,
            vocab=vocab,
            include_gap_channel=include_gap_channel,
        )
        np.savez_compressed(
            tensor_path,
            X=tensor,
            taxa_order=np.array(metadata["taxa_order"], dtype=str),
            codon_vocab_json=json.dumps(vocab, sort_keys=True),
            method=np.array(method),
            family_id=np.array(family_id),
        )
        tensor_meta = {
            "family_id": family_id,
            "method": method,
            "tensor_file": str(tensor_path.relative_to(outdir)),
            "shape": list(tensor.shape),
            "n_taxa": metadata["n_taxa"],
            "n_codons": metadata["n_codons"],
            "n_channels": metadata["n_channels"],
            "taxa_order": metadata["taxa_order"],
            "gap_codon_count": metadata["gap_codon_count"],
            "gap_codon_fraction": metadata["gap_codon_fraction"],
            "source_alignment": str(source_alignment),
            "source_truth": str(truth_path),
            "include_gap_channel": include_gap_channel,
        }
        _write_json(meta_path, tensor_meta)
        return {
            "family_id": family_id,
            "method": method,
            "tensor_file": str(tensor_path.relative_to(outdir)),
            "meta_file": str(meta_path.relative_to(outdir)),
            "n_taxa": metadata["n_taxa"],
            "n_codons": metadata["n_codons"],
            "n_channels": metadata["n_channels"],
            "gap_codon_count": metadata["gap_codon_count"],
            "gap_codon_fraction": metadata["gap_codon_fraction"],
            "gene_label": labels.get("gene_label"),
            "saturation_tier": labels.get("saturation_tier"),
            "status": "ok",
            "warning": "",
        }
    except (OSError, ValueError) as exc:
        return {
            "family_id": family_id,
            "method": method,
            "tensor_file": str(tensor_path.relative_to(outdir)),
            "meta_file": str(meta_path.relative_to(outdir)),
            "n_taxa": 0,
            "n_codons": 0,
            "n_channels": 0,
            "gap_codon_count": 0,
            "gap_codon_fraction": 0.0,
            "gene_label": labels.get("gene_label"),
            "saturation_tier": labels.get("saturation_tier"),
            "status": "fail",
            "warning": str(exc),
        }


def _write_tensor_audit(path: Path, audit_rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=TENSOR_AUDIT_FIELDNAMES, delimiter="\t"
        )
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(
                {fieldname: row.get(fieldname, "") for fieldname in TENSOR_AUDIT_FIELDNAMES}
            )


def _read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
