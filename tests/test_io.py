from pathlib import Path

import pytest

from babappa.io import read_fasta


def test_read_fasta_rectangular_alignment(tmp_path: Path) -> None:
    fasta = tmp_path / "a.fasta"
    fasta.write_text(">s1\nACGT\n>s2\nA-GT\n", encoding="utf-8")
    alignment = read_fasta(fasta)
    assert alignment.n_sequences == 2
    assert alignment.length == 4
    assert alignment.column(1) == "C-"


def test_read_fasta_raises_on_non_rectangular_input(tmp_path: Path) -> None:
    fasta = tmp_path / "bad.fasta"
    fasta.write_text(">s1\nACGT\n>s2\nACG\n", encoding="utf-8")
    with pytest.raises(ValueError, match="equal length"):
        read_fasta(fasta)
