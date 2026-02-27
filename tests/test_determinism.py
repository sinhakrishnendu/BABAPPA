from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from babappa.cli import main


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_alignment(path: Path, records: dict[str, str]) -> Path:
    lines: list[str] = []
    for name, seq in records.items():
        lines.append(f">{name}")
        lines.append(seq)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_once(workdir: Path, neutral: Path, observed: Path) -> tuple[Path, Path]:
    model = workdir / "model.json"
    results = workdir / "results.tsv"
    manifest = workdir / "manifest.json"

    assert (
        main(
            [
                "train",
                "--neutral",
                str(neutral),
                "--model",
                str(model),
                "--seed",
                "17",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "batch",
                "--model",
                str(model),
                "--alignments",
                str(observed),
                "--calibration-size",
                "40",
                "--seed",
                "33",
                "--output",
                str(results),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    return results, manifest


def test_train_analyze_deterministic_hashes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BABAPPA_FIXED_TIMESTAMP_UTC", "2026-02-24T00:00:00+00:00")

    neutral = _write_alignment(
        tmp_path / "neutral.fasta",
        {
            "a": "ACGT" * 15,  # 60 nt
            "b": "ACGT" * 15,
            "c": "ACGT" * 15,
            "d": "ACGT" * 15,
        },
    )
    observed = _write_alignment(
        tmp_path / "obs.fasta",
        {
            "a": "ACGTACGTAAAA",
            "b": "ACGTACGTAAAA",
            "c": "ACGTACGTTTTT",
            "d": "ACGTACGTTTTT",
        },
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    results_1, manifest_1 = _run_once(run_dir, neutral, observed)
    snapshot_results = tmp_path / "results_first.tsv"
    snapshot_manifest = tmp_path / "manifest_first.json"
    shutil.copy2(results_1, snapshot_results)
    shutil.copy2(manifest_1, snapshot_manifest)

    results_2, manifest_2 = _run_once(run_dir, neutral, observed)
    assert _sha256_file(snapshot_results) == _sha256_file(results_2)
    assert _sha256_file(snapshot_manifest) == _sha256_file(manifest_2)

