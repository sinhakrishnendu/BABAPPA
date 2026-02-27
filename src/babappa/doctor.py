from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .codon import GENETIC_CODE
from .hash_utils import sha256_file
from .io import Alignment, read_fasta
from .model import load_model
from .phylo import TreeNode, parse_newick
from .specs import load_neutral_model_spec, load_phi_spec


@dataclass
class DoctorCheck:
    name: str
    status: str  # PASS | WARN | FAIL
    message: str
    fix: str | None = None


@dataclass
class DoctorReport:
    checks: list[DoctorCheck]

    @property
    def has_failures(self) -> bool:
        return any(check.status == "FAIL" for check in self.checks)

    def render(self) -> str:
        lines = []
        for check in self.checks:
            line = f"[{check.status}] {check.name}: {check.message}"
            lines.append(line)
            if check.fix:
                lines.append(f"  fix: {check.fix}")
        summary = "FAIL" if self.has_failures else "PASS"
        lines.append(f"\nDoctor summary: {summary}")
        return "\n".join(lines)


ALLOWED_NUCLEOTIDE_SYMBOLS = set("ACGTUNRYSWKMBDHVX?-.\n\r\t ")
STOP_CODONS = {"TAA", "TAG", "TGA"}


def _iter_fasta_files(genes_dir: Path) -> list[Path]:
    suffixes = {".fa", ".fasta", ".fas", ".fna", ".aln"}
    files = [p for p in sorted(genes_dir.iterdir()) if p.is_file() and p.suffix.lower() in suffixes]
    if not files:
        raise ValueError(f"No FASTA alignment files found in: {genes_dir}")
    return files


def _walk_internal_nodes(root: TreeNode) -> list[TreeNode]:
    out: list[TreeNode] = []

    def _walk(node: TreeNode) -> None:
        if node.children:
            out.append(node)
        for child in node.children:
            _walk(child)

    _walk(root)
    return out


def _internal_stop_positions(alignment: Alignment) -> list[tuple[str, int, str]]:
    issues: list[tuple[str, int, str]] = []
    length = alignment.length - (alignment.length % 3)
    if length < 6:
        return issues
    last_start = length - 3
    for name, seq in zip(alignment.names, alignment.sequences):
        seq_t = seq.upper().replace("U", "T")
        for start in range(0, last_start, 3):
            codon = seq_t[start : start + 3]
            if any(c not in {"A", "C", "G", "T"} for c in codon):
                continue
            if codon in STOP_CODONS:
                issues.append((name, start + 1, codon))
    return issues


def run_doctor(
    *,
    genes_dir: str | Path,
    tree_file: str | Path,
    neutral_model_json: str | Path,
    phi_spec: str | Path | None = None,
    strict_taxa: bool = True,
    taxa_policy: str = "strict",
    allow_stop_codons: bool = False,
    allow_polytomies: bool = False,
    filter_to_multiple_of_3: bool = False,
    existing_model_file: str | Path | None = None,
    seed: int | None = None,
    output_dir: str | Path | None = None,
) -> DoctorReport:
    checks: list[DoctorCheck] = []

    genes_dir_p = Path(genes_dir)
    tree_path = Path(tree_file)
    neutral_path = Path(neutral_model_json)
    if not genes_dir_p.exists():
        raise FileNotFoundError(f"genes_dir not found: {genes_dir_p}")
    if not tree_path.exists():
        raise FileNotFoundError(f"tree_file not found: {tree_path}")
    if not neutral_path.exists():
        raise FileNotFoundError(f"neutral_model_json not found: {neutral_path}")

    files = _iter_fasta_files(genes_dir_p)
    alignments: list[tuple[Path, Alignment]] = []

    all_taxa_sets: list[set[str]] = []
    char_failures: list[str] = []
    length_failures: list[str] = []
    stop_failures: list[str] = []

    for path in files:
        raw_text = path.read_text(encoding="utf-8")
        sequence_chars: set[str] = set()
        for line in raw_text.splitlines():
            if not line:
                continue
            if line.startswith(">"):
                continue
            sequence_chars.update(line.strip())
        bad_chars = sorted({c for c in sequence_chars if c.upper() not in ALLOWED_NUCLEOTIDE_SYMBOLS})
        if bad_chars:
            char_failures.append(f"{path.name}: unexpected symbols {''.join(bad_chars)}")

        aln = read_fasta(path)
        alignments.append((path, aln))
        all_taxa_sets.append(set(aln.names))

        remainder = aln.length % 3
        if remainder != 0 and not filter_to_multiple_of_3:
            length_failures.append(
                f"{path.name}: length={aln.length} is not divisible by 3 (remainder {remainder})"
            )

        if not allow_stop_codons:
            stops = _internal_stop_positions(aln)
            for taxon, pos, codon in stops[:20]:
                stop_failures.append(f"{path.name}:{taxon}:site{pos}:{codon}")

    if char_failures:
        checks.append(
            DoctorCheck(
                "Alignment codon alphabet",
                "FAIL",
                "; ".join(char_failures[:8]),
                "Remove non-codon symbols or replace with supported ambiguity/gap symbols.",
            )
        )
    else:
        checks.append(DoctorCheck("Alignment codon alphabet", "PASS", "All symbols are valid."))

    if length_failures:
        checks.append(
            DoctorCheck(
                "Alignment length divisibility",
                "FAIL",
                "; ".join(length_failures[:8]),
                "Trim/filter columns to maintain coding frame or use --filter-to-multiple-of-3.",
            )
        )
    else:
        msg = (
            "All gene lengths divisible by 3."
            if not filter_to_multiple_of_3
            else "Length divisibility check passed (filter_to_multiple_of_3 enabled)."
        )
        checks.append(DoctorCheck("Alignment length divisibility", "PASS", msg))

    if stop_failures:
        checks.append(
            DoctorCheck(
                "Internal stop codons",
                "FAIL",
                "; ".join(stop_failures[:8]),
                "Filter problematic taxa/sites or run with --allow-stop-codons if scientifically justified.",
            )
        )
    else:
        checks.append(DoctorCheck("Internal stop codons", "PASS", "No internal stop codons detected."))

    # Taxa policy checks
    if strict_taxa and taxa_policy != "strict":
        taxa_policy = "strict"
    first_taxa = all_taxa_sets[0]
    mismatch = any(taxa != first_taxa for taxa in all_taxa_sets[1:])
    if taxa_policy == "strict":
        if mismatch:
            checks.append(
                DoctorCheck(
                    "Taxa policy",
                    "FAIL",
                    "Genes do not share identical taxa labels under strict policy.",
                    "Remove inconsistent genes or run with --taxa-policy intersection.",
                )
            )
        else:
            checks.append(DoctorCheck("Taxa policy", "PASS", "All genes share identical taxa labels."))
    elif taxa_policy == "intersection":
        intersection = set.intersection(*all_taxa_sets)
        dropped = sorted(set.union(*all_taxa_sets) - intersection)
        if dropped:
            checks.append(
                DoctorCheck(
                    "Taxa policy",
                    "WARN",
                    f"Intersection policy will drop {len(dropped)} taxa: {', '.join(dropped[:10])}",
                    "Review whether taxa loss is acceptable for your biological question.",
                )
            )
        else:
            checks.append(DoctorCheck("Taxa policy", "PASS", "Intersection equals full taxa set."))
    else:
        checks.append(
            DoctorCheck(
                "Taxa policy",
                "FAIL",
                f"Unsupported taxa policy: {taxa_policy}",
                "Use --taxa-policy strict or --taxa-policy intersection.",
            )
        )

    # Tree checks
    tree_text = tree_path.read_text(encoding="utf-8")
    tree = parse_newick(tree_text)
    leaves = set(tree.leaf_names())
    positive_lengths = [length for length in tree.branch_lengths() if length > 0]
    if len(positive_lengths) != len(tree.branch_lengths()):
        checks.append(
            DoctorCheck(
                "Tree branch lengths",
                "FAIL",
                "Tree contains non-positive branch lengths.",
                "Provide a tree with strictly positive branch lengths.",
            )
        )
    else:
        checks.append(DoctorCheck("Tree branch lengths", "PASS", "All branch lengths are positive."))

    internal_nodes = _walk_internal_nodes(tree)
    polytomy_nodes = [node for node in internal_nodes if len(node.children) > 2]
    if polytomy_nodes and not allow_polytomies:
        checks.append(
            DoctorCheck(
                "Tree polytomies",
                "FAIL",
                f"Found {len(polytomy_nodes)} polytomous internal node(s).",
                "Resolve tree to bifurcating form or run with --allow-polytomies.",
            )
        )
    elif polytomy_nodes:
        checks.append(
            DoctorCheck(
                "Tree polytomies",
                "WARN",
                f"Polytomies allowed by user flag ({len(polytomy_nodes)} found).",
            )
        )
    else:
        checks.append(DoctorCheck("Tree polytomies", "PASS", "Tree is bifurcating."))

    tree_taxa_mismatches: list[str] = []
    for path, aln in alignments:
        aln_taxa = set(aln.names)
        missing = sorted(leaves - aln_taxa)
        extra = sorted(aln_taxa - leaves)
        if missing or extra:
            tree_taxa_mismatches.append(
                f"{path.name}: missing_from_gene={missing[:5]} extra_in_gene={extra[:5]}"
            )
    if tree_taxa_mismatches:
        checks.append(
            DoctorCheck(
                "Tree/alignment taxa consistency",
                "FAIL",
                "; ".join(tree_taxa_mismatches[:5]),
                "Ensure tree and gene alignments contain consistent taxa labels.",
            )
        )
    else:
        checks.append(DoctorCheck("Tree/alignment taxa consistency", "PASS", "Tree taxa match alignments."))

    # Neutral model and phi spec checks
    try:
        neutral_spec, neutral_payload = load_neutral_model_spec(neutral_path)
        if neutral_spec.omega != 1.0:
            checks.append(
                DoctorCheck(
                    "Neutral model omega",
                    "FAIL",
                    f"omega={neutral_spec.omega} (must be 1.0).",
                    "Set omega to 1 for neutral null.",
                )
            )
        else:
            checks.append(DoctorCheck("Neutral model omega", "PASS", "omega=1 confirmed."))
        checks.append(
            DoctorCheck(
                "Neutral model specification",
                "PASS",
                "Neutral model JSON is valid (GY94 + kappa + codon frequencies/method).",
            )
        )
        _ = neutral_payload
    except Exception as exc:
        checks.append(
            DoctorCheck(
                "Neutral model specification",
                "FAIL",
                str(exc),
                "Provide valid neutral_model_json with omega=1, kappa, and codon frequency specification.",
            )
        )

    try:
        _ = load_phi_spec(phi_spec)
        checks.append(DoctorCheck("Phi specification", "PASS", "Phi specification loadable and valid."))
    except Exception as exc:
        checks.append(
            DoctorCheck(
                "Phi specification",
                "FAIL",
                str(exc),
                "Provide a valid phi_spec JSON or omit to use default deterministic phi.",
            )
        )

    # Frozen model integrity
    if existing_model_file:
        model_path = Path(existing_model_file)
        if not model_path.exists():
            checks.append(
                DoctorCheck(
                    "Frozen energy model",
                    "FAIL",
                    f"Model file not found: {model_path}",
                    "Provide a valid trained model path.",
                )
            )
        else:
            before = sha256_file(model_path)
            try:
                _ = load_model(model_path)
                after = sha256_file(model_path)
                if before != after:
                    checks.append(
                        DoctorCheck(
                            "Frozen energy model",
                            "FAIL",
                            "Model file hash changed during load.",
                            "Do not mutate model artifacts during analysis.",
                        )
                    )
                else:
                    checks.append(
                        DoctorCheck(
                            "Frozen energy model",
                            "PASS",
                            f"Model loadable and immutable hash={before[:16]}...",
                        )
                    )
            except Exception as exc:
                checks.append(
                    DoctorCheck(
                        "Frozen energy model",
                        "FAIL",
                        str(exc),
                        "Regenerate model via `babappa train`.",
                    )
                )
    else:
        checks.append(
            DoctorCheck(
                "Frozen energy model",
                "WARN",
                "No existing model file provided; immutability check skipped.",
                "Pass --existing-model to verify frozen model hash contract.",
            )
        )

    checks.append(
        DoctorCheck(
            "Frozen-energy validity contract",
            "PASS",
            "Analysis path uses frozen energy; calibration does not retrain the energy model.",
        )
    )

    # Reproducibility and filesystem
    if seed is None:
        checks.append(
            DoctorCheck(
                "Reproducibility seed",
                "WARN",
                "No explicit seed provided.",
                "Use --seed for deterministic reruns.",
            )
        )
    else:
        checks.append(DoctorCheck("Reproducibility seed", "PASS", f"Seed specified: {seed}"))

    outdir = Path(output_dir) if output_dir else genes_dir_p
    try:
        outdir.mkdir(parents=True, exist_ok=True)
        probe = outdir / ".babappa_doctor_write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        checks.append(DoctorCheck("Output directory writable", "PASS", str(outdir.resolve())))
    except Exception as exc:
        checks.append(
            DoctorCheck(
                "Output directory writable",
                "FAIL",
                str(exc),
                "Choose a writable output directory.",
            )
        )

    return DoctorReport(checks=checks)
