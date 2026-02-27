from __future__ import annotations

import json
from pathlib import Path

import pytest

import babappa.benchmark.orchestrator as orchestrator
from babappa.cli import main
from babappa.neutral import GY94NeutralSimulator, NeutralSpec
from babappa.benchmark.orchestrator import BenchmarkRunConfig, run_benchmark_track


def _write_alignment(path: Path, records: dict[str, str]) -> None:
    lines: list[str] = []
    for name, seq in records.items():
        lines.append(f">{name}")
        lines.append(seq)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_benchmark_run_and_rebuild_ortholog_track(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BABAPPA_BENCHMARK_FAST", "1")

    tree = "(T1:0.1,T2:0.1,T3:0.1,T4:0.1,T5:0.1,T6:0.1,T7:0.1,T8:0.1);"
    tree_path = tmp_path / "tree.nwk"
    tree_path.write_text(tree + "\n", encoding="utf-8")

    neutral_model = {
        "schema_version": 1,
        "model_family": "GY94",
        "genetic_code_table": "standard",
        "tree_file": str(tree_path),
        "kappa": 2.0,
        "omega": 1.0,
        "codon_frequencies_method": "F3x4",
        "frozen_values": True,
    }
    neutral_path = tmp_path / "neutral_model.json"
    neutral_path.write_text(json.dumps(neutral_model, indent=2) + "\n", encoding="utf-8")

    spec = NeutralSpec(tree_newick=tree, kappa=2.0, omega=1.0)
    sim = GY94NeutralSimulator(spec)
    genes: list[dict[str, str]] = []
    for i in range(2):
        aln = sim.simulate_alignment(length_nt=900, seed=10 + i)
        gpath = tmp_path / f"gene_{i+1}.fasta"
        _write_alignment(gpath, dict(zip(aln.names, aln.sequences)))
        genes.append({"gene_id": gpath.stem, "alignment_path": str(gpath)})

    dataset = {
        "schema_version": 1,
        "tree_path": str(tree_path),
        "genes": genes,
        "neutral_model_json": str(neutral_path),
    }
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")

    outdir = tmp_path / "pack"
    assert (
        main(
            [
                "benchmark",
                "run",
                "--track",
                "ortholog",
                "--preset",
                "ORTHOMAM_SMALL8",
                "--outdir",
                str(outdir),
                "--seed",
                "7",
                "--dataset-json",
                str(dataset_path),
                "--allow-qc-fail",
            ]
        )
        == 0
    )

    assert (outdir / "raw" / "babappa_results.tsv").exists()
    assert (outdir / "raw" / "babappa_shards").is_dir()
    assert len(list((outdir / "raw" / "babappa_shards").glob("*.json"))) >= 1
    assert (outdir / "raw" / "progress_babappa.tsv").exists()
    assert (outdir / "logs" / "heartbeat.txt").exists()
    assert (outdir / "tables" / "T3_ortholog_discovery.tsv").exists()
    assert (outdir / "figures" / "F3_ortholog_overlap_upset.pdf").exists()
    assert (outdir / "manifests" / "benchmark_track_manifest.json").exists()
    assert (outdir / "scripts" / "rebuild_all.sh").exists()
    assert (outdir / "checksums.txt").exists()

    shard_paths = sorted((outdir / "raw" / "babappa_shards").glob("*.json"))
    assert shard_paths
    missing_shard = shard_paths[0]
    missing_shard.unlink()
    assert not missing_shard.exists()
    assert main(["benchmark", "resume", "--outdir", str(outdir), "--no-plots"]) == 0
    assert missing_shard.exists()

    assert main(["benchmark", "rebuild", "--outdir", str(outdir)]) == 0


def test_benchmark_refuses_baseline_comparison_when_doctor_fails(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("BABAPPA_BENCHMARK_FAST", "1")
    monkeypatch.setenv("BABAPPA_CODEML_BACKEND", "local")
    monkeypatch.setenv("BABAPPA_CODEML_BIN", "/definitely/missing/codeml")
    monkeypatch.setenv("BABAPPA_HYPHY_BACKEND", "local")
    monkeypatch.setenv("BABAPPA_HYPHY_BIN", "/definitely/missing/hyphy")

    cfg = BenchmarkRunConfig(
        track="simulation",
        preset="power_full",
        outdir=tmp_path / "pack_fail",
        seed=13,
        include_baselines=True,
        include_relax=True,
        allow_baseline_fail=False,
    )
    with pytest.raises(ValueError, match="Baseline doctor failed"):
        run_benchmark_track(cfg)


def test_babappa_gene_runner_writes_shards_progress_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def _fake_worker(task):
        idx, gene, *_ = task
        calls.append(int(idx))
        row = {
            "gene_id": str(gene["gene_id"]),
            "status": "OK",
            "p": 0.25 + 0.1 * int(idx),
            "q_value": None,
            "n_used": 99,
            "D_obs": 0.1,
            "mu0_hat": 0.05,
            "sigma0_hat": 0.02,
            "L": 900,
        }
        runtime = {"method": "babappa", "gene_id": str(gene["gene_id"]), "runtime_sec": 0.12, "L": 900}
        return int(idx), row, runtime

    monkeypatch.setattr(orchestrator, "_run_babappa_gene_worker", _fake_worker)

    genes = [
        {"gene_id": "g1", "alignment_path": str(tmp_path / "g1.fasta")},
        {"gene_id": "g2", "alignment_path": str(tmp_path / "g2.fasta")},
    ]

    outdir = tmp_path / "pack"
    df, runtime_rows = orchestrator._run_babappa_for_genes(  # type: ignore[attr-defined]
        genes=genes,
        model={},
        calibration_size=99,
        tail="right",
        rng=orchestrator.np.random.default_rng(7),
        jobs=1,
        outdir=outdir,
        resume=False,
        heartbeat_interval_sec=1,
    )

    assert len(df) == 2
    assert len(runtime_rows) == 2
    assert (outdir / "raw" / "babappa_shards" / "0001.json").exists()
    assert (outdir / "raw" / "babappa_shards" / "0002.json").exists()
    progress_lines = (outdir / "raw" / "progress_babappa.tsv").read_text(encoding="utf-8").strip().splitlines()
    assert len(progress_lines) >= 3
    heartbeat = (outdir / "logs" / "heartbeat.txt").read_text(encoding="utf-8")
    assert "completed=2/total=2" in heartbeat
    assert "active_workers=0" in heartbeat

    # Simulate interruption: remove one shard, then resume should run only missing idx.
    (outdir / "raw" / "babappa_shards" / "0002.json").unlink()
    calls.clear()
    df_resume, runtime_rows_resume = orchestrator._run_babappa_for_genes(  # type: ignore[attr-defined]
        genes=genes,
        model={},
        calibration_size=99,
        tail="right",
        rng=orchestrator.np.random.default_rng(8),
        jobs=1,
        outdir=outdir,
        resume=True,
        heartbeat_interval_sec=1,
    )
    assert calls == [2]
    assert len(df_resume) == 2
    assert len(runtime_rows_resume) == 2
    assert (outdir / "raw" / "babappa_shards" / "0002.json").exists()
