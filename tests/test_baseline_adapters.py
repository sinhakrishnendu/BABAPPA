from __future__ import annotations

import json
from pathlib import Path

from babappa.baseline import run_baseline_doctor
from babappa.baseline_adapters import (
    _extract_lnl_from_mlc,
    _parse_hyphy_output,
    run_method_for_gene,
)


def _write_alignment(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                ">T1",
                "ACGTACGTACGT",
                ">T2",
                "ACGTACGTACGT",
                ">T3",
                "ACGTACGTACGT",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_extract_lnl_from_mlc_reads_last_value(tmp_path: Path) -> None:
    mlc = tmp_path / "mlc.txt"
    mlc.write_text(
        "\n".join(
            [
                "lnL(ntime: 5  np: 10): -1234.5000 +0.0000",
                "lnL(ntime: 5  np: 12): -1200.2500 +0.0000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert _extract_lnl_from_mlc(mlc) == -1200.25


def test_parse_hyphy_output_from_json(tmp_path: Path) -> None:
    out = tmp_path / "busted.json"
    payload = {"test results": {"p-value": 0.0123, "LRT": 9.8}}
    out.write_text(json.dumps(payload), encoding="utf-8")
    p, lrt = _parse_hyphy_output(out, "", "")
    assert p == 0.0123
    assert lrt == 9.8


def test_codeml_adapter_returns_fail_records_when_binary_missing(
    tmp_path: Path, monkeypatch
) -> None:
    alignment = _write_alignment(tmp_path / "g.fasta")
    tree = tmp_path / "t.nwk"
    tree.write_text("(T1:0.1,T2:0.1,T3:0.1);\n", encoding="utf-8")

    monkeypatch.setenv("BABAPPA_CODEML_BACKEND", "local")
    monkeypatch.setenv("BABAPPA_CODEML_BIN", "/definitely/missing/codeml")

    recs = run_method_for_gene(
        method="codeml",
        alignment_path=alignment,
        tree_path=tree,
        workdir=tmp_path / "work",
        foreground_taxon="T1",
        run_site_model=True,
    )
    assert len(recs) == 2
    assert all(r.status == "FAIL" for r in recs)
    assert all(r.method in {"codeml_branchsite", "codeml_site_m7m8"} for r in recs)


def test_baseline_doctor_reports_failure_when_backends_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("BABAPPA_CODEML_BACKEND", "local")
    monkeypatch.setenv("BABAPPA_CODEML_BIN", "/definitely/missing/codeml")
    monkeypatch.setenv("BABAPPA_HYPHY_BACKEND", "local")
    monkeypatch.setenv("BABAPPA_HYPHY_BIN", "/definitely/missing/hyphy")

    report = run_baseline_doctor(
        methods=("codeml", "busted", "relax"),
        timeout_sec=1,
        work_dir=tmp_path / "doctor",
        pull_images=False,
    )
    assert report.has_failures
    assert {m.method for m in report.methods} == {"codeml", "busted", "relax"}
    assert all(m.status == "FAIL" for m in report.methods)
