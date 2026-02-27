from pathlib import Path

import pytest

from babappa.engine import analyze_alignment, analyze_batch, train_energy_model
from babappa.io import read_fasta


def _write_alignment(path: Path, records: dict[str, str]) -> Path:
    lines: list[str] = []
    for name, seq in records.items():
        lines.append(f">{name}")
        lines.append(seq)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_train_and_analyze_single_gene(tmp_path: Path) -> None:
    neutral_seq = "ACGT" * 15  # 60 nt
    neutral = _write_alignment(
        tmp_path / "neutral.fasta",
        {
            "a": neutral_seq,
            "b": neutral_seq,
            "c": neutral_seq,
            "d": neutral_seq,
        },
    )
    observed = _write_alignment(
        tmp_path / "observed.fasta",
        {
            "a": "ACGTACGTAAAA",
            "b": "ACGTACGTAAAA",
            "c": "ACGTACGTTTTT",
            "d": "ACGTACGTTTTT",
        },
    )

    model = train_energy_model([read_fasta(neutral)], seed=7)
    result = analyze_alignment(
        read_fasta(observed),
        model=model,
        calibration_size=50,
        seed=12,
        name="observed",
    )
    assert result.alignment_name == "observed"
    assert 0.0 <= result.p_value <= 1.0
    assert result.calibration_size == 50
    assert result.tail == "right"
    assert isinstance(result.warnings, tuple)


def test_batch_adds_q_values(tmp_path: Path) -> None:
    neutral_seq = "ACGT" * 15  # 60 nt
    neutral = _write_alignment(
        tmp_path / "neutral.fasta",
        {
            "a": neutral_seq,
            "b": neutral_seq,
            "c": neutral_seq,
        },
    )
    g1 = _write_alignment(
        tmp_path / "g1.fasta",
        {"a": "ACGTACGTACGT", "b": "ACGTACGTACGT", "c": "ACGTACGTACGT"},
    )
    g2 = _write_alignment(
        tmp_path / "g2.fasta",
        {"a": "AAAAAAAACCCC", "b": "TTTTTTTTGGGG", "c": "AAAAAAAAGGGG"},
    )
    model = train_energy_model([read_fasta(neutral)], seed=5)
    results = analyze_batch(
        [("g1", read_fasta(g1)), ("g2", read_fasta(g2))],
        model=model,
        calibration_size=30,
        seed=99,
    )
    assert len(results) == 2
    assert all(result.q_value is not None for result in results)
    assert all(0.0 <= float(result.q_value) <= 1.0 for result in results)


def test_guardrail_raises_without_override(tmp_path: Path) -> None:
    neutral = _write_alignment(
        tmp_path / "neutral_short.fasta",
        {
            "a": "ACGTACGTACGT",
            "b": "ACGTACGTACGT",
            "c": "ACGTACGTACGT",
        },
    )
    observed = _write_alignment(
        tmp_path / "observed_short.fasta",
        {
            "a": "ACGTACGTAAAA",
            "b": "ACGTACGTAAAA",
            "c": "ACGTACGTTTTT",
        },
    )
    model = train_energy_model([read_fasta(neutral)], seed=11)
    with pytest.raises(ValueError):
        analyze_alignment(
            read_fasta(observed),
            model=model,
            calibration_size=30,
            seed=12,
            name="obs_short",
        )
