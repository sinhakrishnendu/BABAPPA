import json
from pathlib import Path

import pytest

from babappa.cli import main


def _write_alignment(path: Path, records: dict[str, str]) -> Path:
    lines: list[str] = []
    for name, seq in records.items():
        lines.append(f">{name}")
        lines.append(seq)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_cli_train_analyze_batch(tmp_path: Path) -> None:
    neutral_seq = "ACGT" * 15  # 60 nt -> n is safely larger than toy L for guardrail
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
        tmp_path / "obs.fasta",
        {
            "a": "ACGTACGTAAAA",
            "b": "ACGTACGTAAAA",
            "c": "ACGTACGTTTTT",
            "d": "ACGTACGTTTTT",
        },
    )
    model_path = tmp_path / "model.json"
    result_path = tmp_path / "result.json"
    manifest_path = tmp_path / "manifest.json"
    batch_tsv = tmp_path / "batch.tsv"
    batch_json = tmp_path / "batch.json"
    pack_dir = tmp_path / "results_pack"

    assert (
        main(
            [
                "train",
                "--neutral",
                str(neutral),
                "--model",
                str(model_path),
                "--seed",
                "1",
            ]
        )
        == 0
    )
    assert model_path.exists()

    assert (
        main(
            [
                "analyze",
                "--model",
                str(model_path),
                "--alignment",
                str(observed),
                "--calibration-size",
                "40",
                "--seed",
                "2",
                "--output",
                str(result_path),
                "--manifest",
                str(manifest_path),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert 0.0 <= payload["p_value"] <= 1.0
    assert payload["tail"] == "right"
    assert manifest_path.exists()

    assert (
        main(
            [
                "batch",
                "--model",
                str(model_path),
                "--alignments",
                str(observed),
                str(observed),
                "--calibration-size",
                "20",
                "--output",
                str(batch_tsv),
                "--json-output",
                str(batch_json),
            ]
        )
        == 0
    )
    assert batch_tsv.exists()
    assert batch_json.exists()

    assert (
        main(
            [
                "benchmark",
                "--results-pack",
                str(pack_dir),
                "--L-grid",
                "30",
                "--taxa-grid",
                "4",
                "--N-grid",
                "9",
                "--n-null-genes",
                "2",
                "--n-alt-genes",
                "1",
                "--training-replicates",
                "4",
                "--training-length",
                "60",
                "--seed",
                "9",
            ]
        )
        == 0
    )
    assert (pack_dir / "raw" / "null_calibration.tsv").exists()
    assert (pack_dir / "raw" / "power.tsv").exists()
    assert (pack_dir / "raw" / "runtime.tsv").exists()
    assert (pack_dir / "figures" / "qq_null.pdf").exists()
    assert (pack_dir / "tables" / "empirical_size.tsv").exists()
    assert (pack_dir / "scripts" / "rebuild_all.sh").exists()


def test_cli_guardrail_fails_without_override(tmp_path: Path) -> None:
    # n=12 and L=12 => L/n=1.0 > 0.25 should fail by default.
    neutral = _write_alignment(
        tmp_path / "neutral_short.fasta",
        {
            "a": "ACGTACGTACGT",
            "b": "ACGTACGTACGT",
            "c": "ACGTACGTACGT",
        },
    )
    observed = _write_alignment(
        tmp_path / "obs_short.fasta",
        {
            "a": "ACGTACGTAAAA",
            "b": "ACGTACGTAAAA",
            "c": "ACGTACGTTTTT",
        },
    )
    model_path = tmp_path / "guardrail_model.json"
    assert main(["train", "--neutral", str(neutral), "--model", str(model_path), "--seed", "3"]) == 0

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "analyze",
                "--model",
                str(model_path),
                "--alignment",
                str(observed),
                "--calibration-size",
                "12",
                "--seed",
                "4",
            ]
        )
    assert int(exc.value.code) != 0


def test_cli_benchmark_no_plots_skips_figures(tmp_path: Path) -> None:
    pack_dir = tmp_path / "results_pack_no_plots"
    assert (
        main(
            [
                "benchmark",
                "--results-pack",
                str(pack_dir),
                "--L-grid",
                "30",
                "--taxa-grid",
                "4",
                "--N-grid",
                "9",
                "--n-null-genes",
                "2",
                "--n-alt-genes",
                "0",
                "--training-replicates",
                "4",
                "--training-length",
                "60",
                "--seed",
                "9",
                "--no-plots",
            ]
        )
        == 0
    )
    assert (pack_dir / "raw" / "null_calibration.tsv").exists()
    assert (pack_dir / "raw" / "runtime.tsv").exists()
    assert not any((pack_dir / "figures").glob("*.pdf"))


def test_cli_guardrail_override_records_manifest_and_warning(tmp_path: Path) -> None:
    neutral = _write_alignment(
        tmp_path / "neutral_short.fasta",
        {
            "a": "ACGTACGTACGT",
            "b": "ACGTACGTACGT",
            "c": "ACGTACGTACGT",
        },
    )
    observed = _write_alignment(
        tmp_path / "obs_short.fasta",
        {
            "a": "ACGTACGTAAAA",
            "b": "ACGTACGTAAAA",
            "c": "ACGTACGTTTTT",
        },
    )
    model_path = tmp_path / "guardrail_model.json"
    result_path = tmp_path / "guardrail_result.json"
    manifest_path = tmp_path / "guardrail_manifest.json"
    assert main(["train", "--neutral", str(neutral), "--model", str(model_path), "--seed", "5"]) == 0

    assert (
        main(
            [
                "analyze",
                "--model",
                str(model_path),
                "--alignment",
                str(observed),
                "--calibration-size",
                "12",
                "--seed",
                "6",
                "--override-scaling",
                "--output",
                str(result_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert any("Scaling warning" in str(x) for x in payload.get("warnings", []))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert bool(manifest.get("override_scaling")) is True


def test_cli_explain_generates_pdf_and_json(tmp_path: Path) -> None:
    neutral = _write_alignment(
        tmp_path / "neutral_explain.fasta",
        {
            "a": "ACGT" * 20,
            "b": "ACGT" * 20,
            "c": "ACGT" * 20,
        },
    )
    observed = _write_alignment(
        tmp_path / "obs_explain.fasta",
        {
            "a": "ACGTACGTAAAAACGTACGTAAAA",
            "b": "ACGTACGTAAAAACGTACGTAAAA",
            "c": "ACGTACGTTTTTACGTACGTTTTT",
        },
    )
    model_path = tmp_path / "explain_model.json"
    out_pdf = tmp_path / "explain.pdf"
    out_json = tmp_path / "explain.json"
    assert main(["train", "--neutral", str(neutral), "--model", str(model_path), "--seed", "11"]) == 0
    assert (
        main(
            [
                "explain",
                "--model",
                str(model_path),
                "--alignment",
                str(observed),
                "--calibration-size",
                "20",
                "--seed",
                "12",
                "--out",
                str(out_pdf),
                "--json-out",
                str(out_json),
            ]
        )
        == 0
    )
    assert out_pdf.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert 0.0 <= float(payload["p_value"]) <= 1.0


def test_cli_baseline_doctor_returns_nonzero_when_backends_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("BABAPPA_CODEML_BACKEND", "local")
    monkeypatch.setenv("BABAPPA_CODEML_BIN", "/definitely/missing/codeml")
    rc = main(
        [
            "baseline",
            "doctor",
            "--methods",
            "codeml",
            "--timeout-sec",
            "1",
            "--work-dir",
            str(tmp_path / "doctor"),
            "--no-pull-images",
        ]
    )
    assert rc == 1


def test_cli_baseline_doctor_hyphy_shortcut_reports_busted_and_relax(
    capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BABAPPA_HYPHY_BACKEND", "local")
    monkeypatch.setenv("BABAPPA_HYPHY_BIN", "/definitely/missing/hyphy")
    rc = main(
        [
            "baseline",
            "doctor",
            "--hyphy",
            "--container",
            "local",
            "--timeout",
            "1",
            "--json",
        ]
    )
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    methods = {str(m["method"]) for m in payload["methods"]}
    assert methods == {"busted", "relax"}


def test_cli_publication_mode_rejects_allow_baseline_fail(
    capsys, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "benchmark",
                "run",
                "--track",
                "simulation",
                "--preset",
                "simulation_null_full",
                "--outdir",
                str(tmp_path / "pack_pub"),
                "--seed",
                "7",
                "--mode",
                "publication",
                "--allow-baseline-fail",
            ]
        )
    assert "Publication mode forbids --allow-baseline-fail" in capsys.readouterr().err


def test_cli_reproduce_publication_rejects_fast(
    capsys, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "reproduce",
                "--mode",
                "publication",
                "--outdir",
                str(tmp_path / "paper_runs"),
                "--fast",
            ]
        )
    assert "Publication mode forbids --fast" in capsys.readouterr().err


def test_cli_dataset_import_orthomam_bundle(tmp_path: Path) -> None:
    source_dir = tmp_path / "orth_src"
    source_dir.mkdir(parents=True, exist_ok=True)
    _write_alignment(
        source_dir / "geneA.fasta",
        {
            "Hsap": "ACGT" * 240,
            "Mmus": "ACGT" * 240,
            "Rnor": "ACGT" * 240,
            "Cfam": "ACGT" * 240,
            "Btau": "ACGT" * 240,
            "Sscr": "ACGT" * 240,
            "Mmul": "ACGT" * 240,
            "Ggal": "ACGT" * 240,
        },
    )
    _write_alignment(
        source_dir / "geneB.fna",
        {
            "Hsap": "ACGT" * 250,
            "Mmus": "ACGT" * 250,
            "Rnor": "ACGT" * 250,
            "Cfam": "ACGT" * 250,
            "Btau": "ACGT" * 250,
            "Sscr": "ACGT" * 250,
            "Mmul": "ACGT" * 250,
            "Ggal": "ACGT" * 250,
        },
    )
    outdir = tmp_path / "orthomam_small8"
    assert (
        main(
            [
                "dataset",
                "import",
                "orthomam",
                "--source-dir",
                str(source_dir),
                "--release",
                "v13",
                "--species-set",
                "small8",
                "--min-length",
                "300",
                "--max-genes",
                "2",
                "--outdir",
                str(outdir),
                "--seed",
                "17",
            ]
        )
        == 0
    )
    dataset_path = outdir / "dataset.json"
    metadata_path = outdir / "metadata.tsv"
    manifest_path = outdir / "fetch_manifest.json"
    assert dataset_path.exists()
    assert metadata_path.exists()
    assert manifest_path.exists()
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["source"] == "orthomam_import"
    assert bool(payload["metadata"]["synthetic_fallback"]) is False
    assert len(payload["genes"]) == 2
    for gene in payload["genes"]:
        assert Path(str(gene["tree_path"])).exists()


def test_cli_dataset_import_sarscov2_orf_bundle(tmp_path: Path) -> None:
    fasta_path = tmp_path / "genomes.fasta"
    seq = "ACG" * 10000  # 30000 nt
    _write_alignment(
        fasta_path,
        {
            "sample_a": seq,
            "sample_b": seq,
            "sample_c": seq,
            "sample_d": seq,
        },
    )
    meta_path = tmp_path / "metadata.tsv"
    meta_path.write_text(
        "strain\tcollection_date\tcountry\n"
        "sample_a\t2020-03-01\tUS\n"
        "sample_b\t2020-03-02\tUS\n"
        "sample_c\t2020-03-03\tIN\n"
        "sample_d\t2020-03-04\tGB\n",
        encoding="utf-8",
    )
    outdir = tmp_path / "sars_bundle"
    assert (
        main(
            [
                "dataset",
                "import",
                "sarscov2",
                "--fasta",
                str(fasta_path),
                "--metadata",
                str(meta_path),
                "--outdir",
                str(outdir),
            ]
        )
        == 0
    )
    payload = json.loads((outdir / "dataset.json").read_text(encoding="utf-8"))
    gene_ids = {str(g["gene_id"]) for g in payload["genes"]}
    assert "sarscov2_spike" in gene_ids
    assert any("orf1ab" in gid for gid in gene_ids)
    meta_df = (outdir / "metadata.tsv").read_text(encoding="utf-8")
    assert "orf_start_1based" in meta_df
    assert "n_retained_sequences" in meta_df


def test_cli_dataset_import_hiv_records_alignment_id(tmp_path: Path) -> None:
    aln_path = _write_alignment(
        tmp_path / "env.fasta",
        {
            "s1": "ACG" * 300,
            "s2": "ACG" * 300,
            "s3": "ACG" * 300,
        },
    )
    outdir = tmp_path / "hiv_bundle"
    assert (
        main(
            [
                "dataset",
                "import",
                "hiv",
                "--alignment-fasta",
                str(aln_path),
                "--alignment-id",
                "LANL-ENV-B-TEST-001",
                "--outdir",
                str(outdir),
            ]
        )
        == 0
    )
    payload = json.loads((outdir / "dataset.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["alignment_id"] == "LANL-ENV-B-TEST-001"
    manifest = json.loads((outdir / "fetch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["fetch_parameters"]["alignment_id"] == "LANL-ENV-B-TEST-001"


def test_cli_dataset_audit_outputs_drop_audit(tmp_path: Path) -> None:
    source_dir = tmp_path / "orth_src_audit"
    source_dir.mkdir(parents=True, exist_ok=True)
    _write_alignment(
        source_dir / "geneA.fasta",
        {
            "Hsap": "ACG" * 400,
            "Mmus": "ACG" * 400,
            "Rnor": "ACG" * 400,
            "Cfam": "ACG" * 400,
            "Btau": "ACG" * 400,
            "Sscr": "ACG" * 400,
            "Mmul": "ACG" * 400,
            "Ggal": "ACG" * 400,
        },
    )
    outdir = tmp_path / "orth_bundle_audit"
    assert (
        main(
            [
                "dataset",
                "import",
                "orthomam",
                "--source-dir",
                str(source_dir),
                "--release",
                "v13",
                "--species-set",
                "small8",
                "--min-length",
                "300",
                "--max-genes",
                "1",
                "--outdir",
                str(outdir),
                "--seed",
                "17",
            ]
        )
        == 0
    )
    audit_out = tmp_path / "audit_out"
    assert (
        main(
            [
                "dataset",
                "audit",
                "--dataset",
                str(outdir),
                "--out",
                str(audit_out),
                "--min-length-codons",
                "300",
                "--min-taxa",
                "6",
            ]
        )
        == 0
    )
    assert (audit_out / "drop_audit.tsv").exists()
    summary = json.loads((audit_out / "summary.json").read_text(encoding="utf-8"))
    assert int(summary["kept"]) >= 1


def test_cli_report_compare_builds_summary(tmp_path: Path) -> None:
    def _make_pack(root: Path, name: str) -> Path:
        pack = root / name
        (pack / "raw").mkdir(parents=True, exist_ok=True)
        (pack / "manifests").mkdir(parents=True, exist_ok=True)
        (pack / "raw" / "babappa_results.tsv").write_text(
            "gene_id\tstatus\tp\tq_value\n"
            "g1\tOK\t0.01\t0.02\n"
            "g2\tOK\t0.2\t0.3\n",
            encoding="utf-8",
        )
        (pack / "raw" / "baseline_all.tsv").write_text(
            "gene_id\tmethod\tstatus\treason\tp_value\truntime_sec\tunit_kind\n"
            "g1\tbusted\tOK\tok\t0.03\t1.2\tfull_gene\n"
            "g2\tbusted\tOK\tok\t0.7\t1.5\tfull_gene\n",
            encoding="utf-8",
        )
        (pack / "raw" / "runtime.tsv").write_text(
            "method\tgene_id\truntime_sec\tunit_kind\n"
            "babappa\tg1\t0.4\tfull_gene\n"
            "babappa\tg2\t0.5\tfull_gene\n"
            "busted\tg1\t1.2\tfull_gene\n"
            "busted\tg2\t1.5\tfull_gene\n",
            encoding="utf-8",
        )
        (pack / "manifests" / "benchmark_track_manifest.json").write_text(
            json.dumps({"track": "ortholog", "preset": f"{name}"}, indent=2) + "\n",
            encoding="utf-8",
        )
        return pack

    p1 = _make_pack(tmp_path, "pack1")
    p2 = _make_pack(tmp_path, "pack2")
    out = tmp_path / "compare_report"
    assert (
        main(
            [
                "report",
                "compare",
                "--inputs",
                str(p1),
                str(p2),
                "--outdir",
                str(out),
            ]
        )
        == 0
    )
    assert (out / "tables" / "summary.tsv").exists()
    assert (out / "report" / "report.pdf").exists()


def test_cli_report_compare_pack_and_audit_mode(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    audit = tmp_path / "audit"
    (pack / "tables").mkdir(parents=True, exist_ok=True)
    (pack / "raw").mkdir(parents=True, exist_ok=True)
    (pack / "report").mkdir(parents=True, exist_ok=True)
    (pack / "report" / "report.pdf").write_text("x", encoding="utf-8")
    (pack / "benchmark_track_manifest.json").write_text("{}\n", encoding="utf-8")
    (pack / "checksums.txt").write_text("x\n", encoding="utf-8")
    (pack / "raw" / "babappa_results.tsv").write_text("gene_id\tstatus\tp\tq_value\n", encoding="utf-8")
    (pack / "raw" / "baseline_all.tsv").write_text("gene_id\tmethod\tstatus\tp_value\n", encoding="utf-8")
    (pack / "tables" / "ortholog_results.tsv").write_text(
        "gene_id\tL_codons\tn_taxa\tbabappa_p\tbabappa_q\tbusted_p\tbusted_q\tbabappa_status\tbusted_status\tbabappa_runtime_sec\tbusted_runtime_sec\tnotes\n"
        "g1\t300\t8\t0.01\t0.02\t0.03\t0.03\tOK\tOK\t1.0\t1.5\t\n"
        "g2\t300\t8\t0.2\t0.3\t0.8\t0.8\tOK\tOK\t1.1\t1.6\t\n",
        encoding="utf-8",
    )
    (audit / "tables").mkdir(parents=True, exist_ok=True)
    (audit / "manifests").mkdir(parents=True, exist_ok=True)
    (audit / "tables" / "pi0.tsv").write_text("method\tpi0_hat\nbabappa\t0.9\nbusted\t0.95\n", encoding="utf-8")
    (audit / "tables" / "null_size.tsv").write_text("alpha\tsize_hat\n0.05\t0.05\n", encoding="utf-8")
    (audit / "tables" / "fdr_inputs.tsv").write_text("method\tn_total_units\nbabappa\t2\n", encoding="utf-8")
    (audit / "tables" / "fdr_q_diff.tsv").write_text("method\tmax_abs_diff\nbabappa\t0.0\nbusted\t0.0\n", encoding="utf-8")
    (audit / "tables" / "covariate_correlations.tsv").write_text("method\tcovariate\tspearman_rho\n", encoding="utf-8")
    (audit / "manifests" / "audit_manifest.json").write_text(
        json.dumps(
            {
                "calibration_fail": False,
                "frozen_energy_invariant_fail": False,
                "testable_set_mismatch": False,
                "severe_inflation": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "compare_pack_audit"
    assert (
        main(
            [
                "report",
                "compare",
                "--pack",
                str(pack),
                "--audit",
                str(audit),
                "--outdir",
                str(out),
            ]
        )
        == 0
    )
    summary = (out / "tables" / "summary.tsv").read_text(encoding="utf-8")
    assert "frozen_energy_invariant_fail" in summary
    assert "testable_set_mismatch" in summary
    assert "severe_inflation" in summary


def test_cli_dataset_fetch_orthomam_cache_default_outdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_root = tmp_path / "cache"
    src = cache_root / "orthomam" / "v12" / "small8"
    src.mkdir(parents=True, exist_ok=True)
    _write_alignment(
        src / "geneA.fasta",
        {
            "A": "ACG" * 300,
            "B": "ACG" * 300,
            "C": "ACG" * 300,
        },
    )
    monkeypatch.setenv("BABAPPA_DATASET_CACHE", str(cache_root))
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "dataset",
                "fetch",
                "orthomam",
                "--release",
                "v12",
                "--species-set",
                "small8",
                "--mode",
                "cache",
                "--max-genes",
                "1",
                "--min-length",
                "300",
            ]
        )
        == 0
    )
    assert (tmp_path / "data" / "orthomam_v12_small8" / "dataset.json").exists()


def test_cli_benchmark_realdata_outdir_defaults(capsys, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "benchmark",
                "realdata",
                "--preset",
                "ortholog_small8_full",
                "--data",
                str(tmp_path / "missing_dataset_dir"),
                "--results-only",
            ]
        )
    captured = capsys.readouterr()
    assert "the following arguments are required: --outdir" not in captured.err


def test_cli_benchmark_realdata_alias_preset(capsys, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "benchmark",
                "realdata",
                "--preset",
                "ortholog_real_v12",
                "--data",
                str(tmp_path / "missing_dataset_dir"),
                "--results-only",
            ]
        )
    captured = capsys.readouterr()
    assert "Unsupported realdata preset" not in captured.err


def test_cli_benchmark_resume_dispatches_handler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, str] = {}

    def _fake_resume(args):
        called["outdir"] = str(args.outdir)
        return 0

    monkeypatch.setattr("babappa.cli._cmd_benchmark_resume", _fake_resume)
    outdir = tmp_path / "pack_resume"
    rc = main(["benchmark", "resume", "--outdir", str(outdir), "--no-plots"])
    assert rc == 0
    assert called["outdir"] == str(outdir)


def test_cli_dataset_cache_show_and_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_root = tmp_path / "cache_root"
    toy_cache = cache_root / "orthomam"
    toy_cache.mkdir(parents=True, exist_ok=True)
    (toy_cache / "x.txt").write_text("x\n", encoding="utf-8")
    monkeypatch.setenv("BABAPPA_DATASET_CACHE", str(cache_root))

    assert main(["dataset", "cache", "--show"]) == 0
    assert main(["dataset", "clear-cache", "--name", "orthomam"]) == 0
    assert not toy_cache.exists()


def test_cli_baseline_container_timeout_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BABAPPA_CODEML_BACKEND", "local")
    monkeypatch.setenv("BABAPPA_CODEML_BIN", "/definitely/missing/codeml")
    monkeypatch.setenv("BABAPPA_HYPHY_BACKEND", "local")
    monkeypatch.setenv("BABAPPA_HYPHY_BIN", "/definitely/missing/hyphy")

    rc = main(
        [
            "baseline",
            "doctor",
            "--methods",
            "codeml,busted",
            "--container",
            "local",
            "--timeout",
            "1",
            "--json",
        ]
    )
    assert rc == 1

    aln = _write_alignment(
        tmp_path / "g.fasta",
        {
            "T1": "ACGTACGTACGT",
            "T2": "ACGTACGTACGT",
            "T3": "ACGTACGTACGT",
        },
    )
    tree = tmp_path / "t.nwk"
    tree.write_text("(T1:0.1,T2:0.1,T3:0.1);\n", encoding="utf-8")
    ds = {
        "schema_version": 1,
        "tree_path": str(tree),
        "genes": [{"gene_id": "g1", "alignment_path": str(aln)}],
    }
    dataset_json = tmp_path / "dataset.json"
    dataset_json.write_text(json.dumps(ds, indent=2) + "\n", encoding="utf-8")
    out_tsv = tmp_path / "baseline.tsv"
    assert (
        main(
            [
                "baseline",
                "run",
                "--method",
                "codeml",
                "--container",
                "local",
                "--timeout",
                "1",
                "--input",
                str(dataset_json),
                "--out",
                str(out_tsv),
            ]
        )
        == 0
    )
    assert "FAIL" in out_tsv.read_text(encoding="utf-8")
