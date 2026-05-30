"""Lightweight deterministic simulator for BABAPPA Cycle 2.

This module intentionally implements a simple biologically inspired generator,
not the final saturation-aware codon-likelihood simulator.
"""

from __future__ import annotations

import csv
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from babappa import __version__

SIMULATOR_VERSION = __version__
BRANCH_TRUTH_VERSION = "0.1"
ALLOWED_SATURATION_TIERS = {"low", "moderate", "high", "extreme"}
SATURATION_MULTIPLIERS = {
    "low": 1.0,
    "moderate": 2.0,
    "high": 4.0,
    "extreme": 8.0,
}
BASES = ("A", "C", "G", "T")

STANDARD_GENETIC_CODE: Dict[str, str] = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}
STOP_CODONS = {codon for codon, aa in STANDARD_GENETIC_CODE.items() if aa == "*"}
SENSE_CODONS = sorted(
    codon for codon, aa in STANDARD_GENETIC_CODE.items() if aa != "*"
)


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a lightweight BABAPPA simulation run."""

    outdir: str
    n_families: int = 5
    n_taxa: int = 8
    n_codons: int = 120
    seed: int = 42
    positive_rate: float = 0.3
    selected_site_fraction: float = 0.05
    mutation_rate: float = 0.03
    indel_rate: float = 0.0
    saturation_tier: str = "low"
    workers: int = 1

    def __post_init__(self) -> None:
        if self.n_families < 1:
            raise ValueError("n_families must be >= 1")
        if self.n_taxa < 3:
            raise ValueError("n_taxa must be >= 3")
        if self.n_codons < 30:
            raise ValueError("n_codons must be >= 30")
        if not 0 <= self.positive_rate <= 1:
            raise ValueError("positive_rate must be between 0 and 1")
        if not 0 <= self.selected_site_fraction <= 1:
            raise ValueError("selected_site_fraction must be between 0 and 1")
        if self.mutation_rate < 0:
            raise ValueError("mutation_rate must be >= 0")
        if self.indel_rate < 0:
            raise ValueError("indel_rate must be >= 0")
        if self.saturation_tier not in ALLOWED_SATURATION_TIERS:
            allowed = ", ".join(sorted(ALLOWED_SATURATION_TIERS))
            raise ValueError(f"saturation_tier must be one of: {allowed}")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")


def translate_codon(codon: str) -> str:
    """Translate a DNA codon using the standard genetic code."""
    normalized = codon.upper().replace("U", "T")
    if len(normalized) != 3 or any(base not in BASES for base in normalized):
        raise ValueError(f"invalid codon: {codon}")
    return STANDARD_GENETIC_CODE[normalized]


def is_synonymous(codon1: str, codon2: str) -> bool:
    """Return whether two codons encode the same amino acid."""
    return translate_codon(codon1) == translate_codon(codon2)


def random_codon(rng: random.Random) -> str:
    """Draw a non-stop codon."""
    return rng.choice(SENSE_CODONS)


def mutate_codon(
    codon: str, rng: random.Random, prefer_nonsynonymous: bool = False
) -> Tuple[str, str]:
    """Mutate one nucleotide in a codon while avoiding stop codons if possible."""
    codon = codon.upper().replace("U", "T")
    translate_codon(codon)

    random_attempts = 50 if prefer_nonsynonymous else 20
    first_viable: Optional[str] = None

    for _ in range(random_attempts):
        candidate = _random_single_base_mutation(codon, rng)
        if candidate in STOP_CODONS:
            continue
        if first_viable is None:
            first_viable = candidate
        if prefer_nonsynonymous and not is_synonymous(codon, candidate):
            return candidate, "nonsynonymous"
        if not prefer_nonsynonymous:
            return candidate, _classify_event(codon, candidate)

    viable_candidates = _single_base_mutation_candidates(codon)
    if prefer_nonsynonymous:
        nonsynonymous = [
            candidate
            for candidate in viable_candidates
            if not is_synonymous(codon, candidate)
        ]
        if nonsynonymous:
            candidate = rng.choice(nonsynonymous)
            return candidate, "nonsynonymous"

    if first_viable is not None:
        return first_viable, _classify_event(codon, first_viable)

    if viable_candidates:
        candidate = rng.choice(viable_candidates)
        return candidate, _classify_event(codon, candidate)

    return codon, "silent_no_change"


def generate_balanced_newick(taxa: List[str], rng: random.Random) -> str:
    """Generate a simple random balanced Newick tree for the provided taxa."""
    if not taxa:
        raise ValueError("at least one taxon is required")

    shuffled_taxa = list(taxa)
    rng.shuffle(shuffled_taxa)

    def build(names: Sequence[str]) -> str:
        if len(names) == 1:
            return names[0]

        split = len(names) // 2
        left = build(names[:split])
        right = build(names[split:])
        left_length = rng.uniform(0.05, 0.5)
        right_length = rng.uniform(0.05, 0.5)
        return f"({left}:{left_length:.4f},{right}:{right_length:.4f})"

    return f"{build(shuffled_taxa)};"


def simulate_one_family(
    family_id: str, config: SimulationConfig, rng: random.Random
) -> dict:
    """Simulate one lightweight synthetic coding-sequence family."""
    taxa = [f"taxon_{index:03d}" for index in range(1, config.n_taxa + 1)]
    root_codons = [random_codon(rng) for _ in range(config.n_codons)]
    sequences: Dict[str, List[str]] = {taxon: list(root_codons) for taxon in taxa}
    has_positive_selection = rng.random() < config.positive_rate
    foreground_taxon = rng.choice(taxa) if has_positive_selection else None
    selected_sites = _choose_selected_sites(config, rng) if has_positive_selection else []
    mutation_attempts = _mutation_attempts(config)
    events: List[dict] = []

    for taxon in taxa:
        for _ in range(mutation_attempts):
            codon_index = rng.randrange(config.n_codons)
            _apply_mutation(
                events=events,
                family_id=family_id,
                taxon=taxon,
                codon_index=codon_index,
                sequence=sequences[taxon],
                rng=rng,
                is_selected_site=codon_index in selected_sites,
                is_foreground=taxon == foreground_taxon,
                prefer_nonsynonymous=False,
            )

        if has_positive_selection and taxon == foreground_taxon:
            for codon_index in selected_sites:
                _apply_mutation(
                    events=events,
                    family_id=family_id,
                    taxon=taxon,
                    codon_index=codon_index,
                    sequence=sequences[taxon],
                    rng=rng,
                    is_selected_site=True,
                    is_foreground=True,
                    prefer_nonsynonymous=True,
                )

    tree = generate_balanced_newick(taxa, rng)
    branch_labels = {
        taxon: int(has_positive_selection and taxon == foreground_taxon)
        for taxon in taxa
    }
    branch_truth = _build_branch_truth(
        family_id=family_id,
        config=config,
        taxa=taxa,
        tree=tree,
        has_positive_selection=has_positive_selection,
        foreground_taxon=foreground_taxon,
        selected_sites=selected_sites,
    )
    truth = {
        "family_id": family_id,
        "has_positive_selection": has_positive_selection,
        "foreground_taxon": foreground_taxon,
        "selected_sites_0based": selected_sites,
        "selected_sites_1based": [site + 1 for site in selected_sites],
        "saturation_tier": config.saturation_tier,
        "n_taxa": config.n_taxa,
        "n_codons": config.n_codons,
        "labels": {
            "gene_label": int(has_positive_selection),
            "branch_labels": branch_labels,
        },
        "explicit_branch_site_truth_available": True,
    }

    return {
        "family_id": family_id,
        "taxa": taxa,
        "sequences": {
            taxon: "".join(codons) for taxon, codons in sequences.items()
        },
        "codons": sequences,
        "tree": tree,
        "truth": truth,
        "branch_truth": branch_truth,
        "events": events,
    }


def simulate_families(config: SimulationConfig) -> dict:
    """Simulate families and write BABAPPA Cycle 2 output files."""
    outdir = Path(config.outdir)
    families_dir = outdir / "families"
    families_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    family_ids = [
        f"family_{index:06d}" for index in range(1, config.n_families + 1)
    ]
    created_files: Dict[str, Dict[str, str]] = {}
    branch_site_truth_path = outdir / "branch_site_truth.tsv"
    branch_truth_rows = 0
    branch_positive_rows = 0

    with branch_site_truth_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_branch_site_truth_fieldnames(), delimiter="\t")
        writer.writeheader()
        if config.workers <= 1:
            for family_id in family_ids:
                family = simulate_one_family(family_id, config, rng)
                family_dir = families_dir / family_id
                family_dir.mkdir(parents=True, exist_ok=True)
                output_files = _write_family_outputs(family, family_dir, config)
                created_files[family_id] = {
                    key: str(path.relative_to(outdir)) for key, path in output_files.items()
                }
                for row in _branch_site_truth_rows(family["branch_truth"]):
                    writer.writerow(row)
                    branch_truth_rows += 1
                    branch_positive_rows += int(row["y_branch_site"])
        else:
            max_workers = min(config.workers, config.n_families)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_simulate_family_worker, index, family_id, str(outdir), config)
                    for index, family_id in enumerate(family_ids, start=1)
                ]
                for future in as_completed(futures):
                    family_id, family_files, branch_rows, positive_rows = future.result()
                    created_files[family_id] = family_files
                    for row in branch_rows:
                        writer.writerow(row)
                    branch_truth_rows += len(branch_rows)
                    branch_positive_rows += positive_rows

    branch_truth_manifest_path = outdir / "branch_truth_manifest.json"
    branch_truth_manifest = {
        "branch_truth_manifest_version": BRANCH_TRUTH_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "truth_source": "explicit_simulator_branch_truth",
        "branch_truth_status": "explicit_truth_ok",
        "n_families": config.n_families,
        "n_branch_truth_files": config.n_families,
        "n_branch_site_truth_rows": branch_truth_rows,
        "n_branch_positive_rows": branch_positive_rows,
        "family_ids": family_ids,
        "branch_truth_files": {
            family_id: created_files[family_id].get("branch_truth", "")
            for family_id in family_ids
        },
        "branch_site_truth_tsv": str(branch_site_truth_path.relative_to(outdir)),
    }
    _write_json(branch_truth_manifest_path, branch_truth_manifest)

    manifest_path = outdir / "manifest.json"
    manifest = {
        "simulator_version": SIMULATOR_VERSION,
        "n_families": config.n_families,
        "config": asdict(config),
        "family_ids": family_ids,
        "created_files": created_files,
        "branch_truth_present": True,
        "branch_truth_manifest": str(branch_truth_manifest_path.relative_to(outdir)),
        "branch_site_truth_tsv": str(branch_site_truth_path.relative_to(outdir)),
        "n_branch_truth_files": config.n_families,
        "n_branch_site_truth_rows": branch_truth_rows,
        "n_branch_positive_rows": branch_positive_rows,
        "branch_truth_status": "explicit_truth_ok",
    }
    _write_json(manifest_path, manifest)

    return {
        "status": "ok",
        "outdir": str(outdir),
        "n_families": config.n_families,
        "workers": config.workers,
        "family_ids": family_ids,
        "manifest": str(manifest_path),
        "branch_truth_manifest": str(branch_truth_manifest_path),
        "branch_site_truth_tsv": str(branch_site_truth_path),
        "n_branch_site_truth_rows": branch_truth_rows,
        "n_branch_positive_rows": branch_positive_rows,
    }


def _simulate_family_worker(index: int, family_id: str, outdir: str, config: SimulationConfig):
    out_path = Path(outdir)
    rng = random.Random(_family_seed(config.seed, index))
    family = simulate_one_family(family_id, config, rng)
    family_dir = out_path / "families" / family_id
    family_dir.mkdir(parents=True, exist_ok=True)
    output_files = _write_family_outputs(family, family_dir, config)
    branch_rows = _branch_site_truth_rows(family["branch_truth"])
    positive_rows = sum(int(row["y_branch_site"]) for row in branch_rows)
    created = {key: str(path.relative_to(out_path)) for key, path in output_files.items()}
    return family_id, created, branch_rows, positive_rows


def _family_seed(seed: int, index: int) -> int:
    return int((seed + index * 1_000_003) % (2**32))


def _random_single_base_mutation(codon: str, rng: random.Random) -> str:
    position = rng.randrange(3)
    replacement_options = [base for base in BASES if base != codon[position]]
    replacement = rng.choice(replacement_options)
    return f"{codon[:position]}{replacement}{codon[position + 1:]}"


def _single_base_mutation_candidates(codon: str) -> List[str]:
    candidates = []
    for position in range(3):
        for replacement in BASES:
            if replacement == codon[position]:
                continue
            candidate = f"{codon[:position]}{replacement}{codon[position + 1:]}"
            if candidate not in STOP_CODONS:
                candidates.append(candidate)
    return candidates


def _classify_event(old_codon: str, new_codon: str) -> str:
    if old_codon == new_codon:
        return "silent_no_change"
    if is_synonymous(old_codon, new_codon):
        return "synonymous"
    return "nonsynonymous"


def _choose_selected_sites(config: SimulationConfig, rng: random.Random) -> List[int]:
    if config.selected_site_fraction <= 0:
        return []
    selected_count = max(1, round(config.n_codons * config.selected_site_fraction))
    selected_count = min(config.n_codons, selected_count)
    return sorted(rng.sample(range(config.n_codons), selected_count))


def _mutation_attempts(config: SimulationConfig) -> int:
    multiplier = SATURATION_MULTIPLIERS[config.saturation_tier]
    return max(0, round(config.n_codons * config.mutation_rate * multiplier))


def _apply_mutation(
    events: List[dict],
    family_id: str,
    taxon: str,
    codon_index: int,
    sequence: List[str],
    rng: random.Random,
    is_selected_site: bool,
    is_foreground: bool,
    prefer_nonsynonymous: bool,
) -> None:
    old_codon = sequence[codon_index]
    new_codon, event_type = mutate_codon(
        old_codon, rng, prefer_nonsynonymous=prefer_nonsynonymous
    )
    sequence[codon_index] = new_codon
    events.append(
        {
            "family_id": family_id,
            "taxon": taxon,
            "codon_index_0based": codon_index,
            "old_codon": old_codon,
            "new_codon": new_codon,
            "event_type": event_type,
            "is_selected_site": int(is_selected_site),
            "is_foreground": int(is_foreground),
        }
    )


def _build_branch_truth(
    family_id: str,
    config: SimulationConfig,
    taxa: List[str],
    tree: str,
    has_positive_selection: bool,
    foreground_taxon: Optional[str],
    selected_sites: List[int],
) -> dict:
    selection_event_id = (
        f"{family_id}:{foreground_taxon}:selection_0001"
        if has_positive_selection and foreground_taxon and selected_sites
        else ""
    )
    selected_by_branch = {
        taxon: list(selected_sites) if taxon == foreground_taxon else []
        for taxon in taxa
    }
    foreground_branches = []
    if has_positive_selection and foreground_taxon:
        foreground_branches.append(
            {
                "branch_id": foreground_taxon,
                "foreground_taxon": foreground_taxon,
                "branch_type": "leaf",
                "branch_length": None,
                "selected_sites_zero": list(selected_sites),
                "selected_sites_one": [site + 1 for site in selected_sites],
                "n_selected_sites": len(selected_sites),
                "selection_event_id": selection_event_id,
            }
        )
    records = []
    selected_set = set(selected_sites)
    for taxon in taxa:
        branch_selected = taxon == foreground_taxon
        for site_index in range(config.n_codons):
            y_branch_site = int(branch_selected and site_index in selected_set)
            event_id = selection_event_id if y_branch_site else ""
            records.append(
                {
                    "branch_id": taxon,
                    "site_index_zero": site_index,
                    "site_index_one": site_index + 1,
                    "y_branch_site": y_branch_site,
                    "event_id": event_id,
                    "selection_event_id": event_id,
                    "branch_type": "leaf",
                    "foreground_taxon": foreground_taxon or "",
                }
            )
    return {
        "branch_truth_version": BRANCH_TRUTH_VERSION,
        "family_id": family_id,
        "saturation_tier": config.saturation_tier,
        "n_taxa": config.n_taxa,
        "n_codons": config.n_codons,
        "tree_newick": tree,
        "selected_sites_zero": list(selected_sites),
        "selected_sites_one": [site + 1 for site in selected_sites],
        "selected_site_by_branch": selected_by_branch,
        "foreground_branches": foreground_branches,
        "branch_site_records": records,
        "truth_source": "explicit_simulator_branch_truth",
    }


def _write_family_outputs(
    family: dict, family_dir: Path, config: SimulationConfig
) -> Dict[str, Path]:
    family_id = family["family_id"]
    output_files = {
        "fasta": family_dir / f"{family_id}.fasta",
        "treefile": family_dir / f"{family_id}.treefile",
        "truth": family_dir / f"{family_id}.truth.json",
        "branch_truth": family_dir / f"{family_id}.branch_truth.json",
        "homology": family_dir / f"{family_id}.homology.tsv",
        "events": family_dir / f"{family_id}.events.tsv",
        "meta": family_dir / f"{family_id}.meta.json",
    }

    _write_fasta(output_files["fasta"], family["sequences"])
    output_files["treefile"].write_text(f"{family['tree']}\n", encoding="utf-8")
    _write_json(output_files["truth"], family["truth"])
    _write_json(output_files["branch_truth"], family["branch_truth"])
    _write_homology(output_files["homology"], family)
    _write_events(output_files["events"], family["events"])

    meta = {
        "family_id": family_id,
        "seed": config.seed,
        "simulator_version": SIMULATOR_VERSION,
        "config": asdict(config),
        "output_files": {
            key: str(path.name) for key, path in output_files.items() if key != "meta"
        },
        "note": (
            "This is a lightweight initial simulator and not the final "
            "codon-likelihood simulator."
        ),
    }
    _write_json(output_files["meta"], meta)

    return output_files


def _branch_site_truth_fieldnames() -> List[str]:
    return [
        "family_id",
        "saturation_tier",
        "branch_id",
        "foreground_taxon",
        "branch_type",
        "site_index_zero",
        "site_index_one",
        "y_branch_site",
        "selection_event_id",
        "truth_source",
    ]


def _branch_site_truth_rows(branch_truth: dict) -> List[dict]:
    rows = []
    family_id = branch_truth.get("family_id", "")
    tier = branch_truth.get("saturation_tier", "")
    for record in branch_truth.get("branch_site_records", []):
        rows.append(
            {
                "family_id": family_id,
                "saturation_tier": tier,
                "branch_id": record.get("branch_id", ""),
                "foreground_taxon": record.get("foreground_taxon", ""),
                "branch_type": record.get("branch_type", "leaf"),
                "site_index_zero": record.get("site_index_zero", ""),
                "site_index_one": record.get("site_index_one", ""),
                "y_branch_site": record.get("y_branch_site", 0),
                "selection_event_id": record.get("selection_event_id") or record.get("event_id", ""),
                "truth_source": "explicit_simulator_branch_truth",
            }
        )
    return rows


def _write_fasta(path: Path, sequences: Dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for taxon, sequence in sequences.items():
            handle.write(f">{taxon}\n")
            for start in range(0, len(sequence), 80):
                handle.write(f"{sequence[start:start + 80]}\n")


def _write_homology(path: Path, family: dict) -> None:
    fieldnames = ["taxon", "codon_index_0based", "homology_id", "codon"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for taxon in family["taxa"]:
            for index, codon in enumerate(family["codons"][taxon]):
                writer.writerow(
                    {
                        "taxon": taxon,
                        "codon_index_0based": index,
                        "homology_id": f"H{index + 1:06d}",
                        "codon": codon,
                    }
                )


def _write_events(path: Path, events: List[dict]) -> None:
    fieldnames = [
        "family_id",
        "taxon",
        "codon_index_0based",
        "old_codon",
        "new_codon",
        "event_type",
        "is_selected_site",
        "is_foreground",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(events)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
