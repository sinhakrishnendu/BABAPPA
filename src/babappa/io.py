from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Alignment:
    names: tuple[str, ...]
    sequences: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("Alignment has no sequences.")
        if len(self.names) != len(self.sequences):
            raise ValueError("Alignment names and sequences are misaligned.")
        lengths = {len(seq) for seq in self.sequences}
        if len(lengths) != 1:
            raise ValueError("All sequences in an alignment must have equal length.")

    @property
    def length(self) -> int:
        return len(self.sequences[0])

    @property
    def n_sequences(self) -> int:
        return len(self.sequences)

    def column(self, index: int) -> str:
        return "".join(seq[index] for seq in self.sequences)

    def iter_columns(self) -> Iterable[str]:
        for i in range(self.length):
            yield self.column(i)


def _normalize_sequence(text: str) -> str:
    return "".join(text.split()).upper()


def read_fasta(path: str | Path) -> Alignment:
    """Read a FASTA file as a strict rectangular alignment."""
    path = Path(path)
    names: list[str] = []
    sequences: list[str] = []
    chunks: list[str] = []

    if not path.exists():
        raise FileNotFoundError(f"Alignment file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if chunks:
                    sequences.append(_normalize_sequence("".join(chunks)))
                    chunks = []
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Missing FASTA header name at line {line_no} in {path}")
                names.append(header)
                continue
            if not names:
                raise ValueError(f"FASTA sequence without header at line {line_no} in {path}")
            chunks.append(line)

    if chunks:
        sequences.append(_normalize_sequence("".join(chunks)))

    if not names:
        raise ValueError(f"No FASTA records found in {path}")
    if len(names) != len(sequences):
        raise ValueError(f"Malformed FASTA in {path}: header/sequence count mismatch.")

    return Alignment(names=tuple(names), sequences=tuple(sequences))


def read_alignments(paths: Iterable[str | Path]) -> list[Alignment]:
    return [read_fasta(path) for path in paths]
