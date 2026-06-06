import importlib.util
import json
import subprocess
from pathlib import Path

from babappa.datasets.index import read_tsv


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "benchmarks" / "known_truth_absrel"
BENCH = ROOT / "benchmarks" / "known_truth_absrel"


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_config(tmp_path: Path) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "profile: smoke",
                "n_families: 12",
                "seed: 7",
                "n_taxa: 5",
                "n_codons: 30",
                f"outdir: {tmp_path / 'run'}",
                "device: auto",
                "babappa_null_replicates: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def test_smoke_benchmark_creates_truth_files(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    result = subprocess.run(["python", str(SCRIPTS / "01_simulate_known_truth_dataset.py"), "--config", str(config)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    run = tmp_path / "run"
    assert (run / "truth" / "family_truth.tsv").exists()
    assert (run / "truth" / "branch_site_truth.tsv").exists()
    assert len(read_tsv(run / "truth" / "family_truth.tsv")) == 12


def test_babappa_result_table_can_be_generated_from_smoke_with_test_mock(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    subprocess.run(["python", str(SCRIPTS / "01_simulate_known_truth_dataset.py"), "--config", str(config)], check=True)
    result = subprocess.run(["python", str(SCRIPTS / "02_run_babappa_on_dataset.py"), "--config", str(config), "--smoke-surrogate"], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    rows = read_tsv(tmp_path / "run" / "babappa_results.tsv")
    assert len(rows) == 12
    assert {row["status"] for row in rows} == {"ok"}


def test_babappa_result_writer_is_rectangular_with_fixed_schema(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    subprocess.run(["python", str(SCRIPTS / "01_simulate_known_truth_dataset.py"), "--config", str(config)], check=True)
    subprocess.run(["python", str(SCRIPTS / "02_run_babappa_on_dataset.py"), "--config", str(config), "--smoke-surrogate"], check=True)
    raw = (tmp_path / "run" / "babappa_results.tsv").read_text(encoding="utf-8").splitlines()
    assert {len(line.split("\t")) for line in raw} == {13}
    rows = read_tsv(tmp_path / "run" / "babappa_results.tsv")
    assert list(rows[0]) == [
        "family_id",
        "method",
        "truth_class",
        "truth_positive",
        "expected_applicability",
        "status",
        "status_class",
        "score",
        "call",
        "result_class",
        "diagnostic_only",
        "applicability",
        "failure_reason",
    ]


def test_babappa_runner_supports_jobs_help() -> None:
    result = subprocess.run(["python", str(SCRIPTS / "02_run_babappa_on_dataset.py"), "--help"], text=True, capture_output=True, check=False)
    assert result.returncode == 0
    assert "--jobs" in result.stdout


def test_absrel_runner_supports_jobs_help() -> None:
    result = subprocess.run(["python", str(SCRIPTS / "04_run_absrel.py"), "--help"], text=True, capture_output=True, check=False)
    assert result.returncode == 0
    assert "--jobs" in result.stdout


def test_prepare_runner_supports_jobs_help() -> None:
    result = subprocess.run(["python", str(SCRIPTS / "03_prepare_absrel_inputs.py"), "--help"], text=True, capture_output=True, check=False)
    assert result.returncode == 0
    assert "--jobs" in result.stdout


def test_parallel_babappa_result_collection_is_deterministic_and_unique(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    subprocess.run(["python", str(SCRIPTS / "01_simulate_known_truth_dataset.py"), "--config", str(config)], check=True)
    result = subprocess.run(
        ["python", str(SCRIPTS / "02_run_babappa_on_dataset.py"), "--config", str(config), "--smoke-surrogate", "--jobs", "2"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    rows = read_tsv(tmp_path / "run" / "babappa_results.tsv")
    family_ids = [row["family_id"] for row in rows]
    assert len(family_ids) == len(set(family_ids)) == 12
    assert family_ids == sorted(family_ids)


def test_babappa_parallel_failure_is_recorded_without_corrupting_tsv(tmp_path: Path) -> None:
    module = _load_script_module("parallel_babappa_failure", SCRIPTS / "02_run_babappa_on_dataset.py")

    class FakeCompleted:
        returncode = 2
        stdout = "fake stdout"
        stderr = "fake stderr"

    module.run_command = lambda *args, **kwargs: FakeCompleted()
    row = {
        "family_id": "F1",
        "truth_class": "positive",
        "expected_applicability": "in_domain",
        "codon_fasta": "missing.fasta",
        "tree": "missing.nwk",
        "foreground": "taxon1",
    }
    result, failure = module._run_family(
        row,
        benchmark_dir=tmp_path,
        results_dir=tmp_path / "scores",
        model_package="model",
        device="cpu",
        null_replicates="0",
        seed=1,
        smoke_surrogate=False,
        force=True,
    )
    assert result["status"] == "failed"
    assert failure and failure["family_id"] == "F1"
    module.write_tsv(tmp_path / "babappa_results.tsv", [result], module.RESULT_FIELDS)
    raw = (tmp_path / "babappa_results.tsv").read_text(encoding="utf-8").splitlines()
    assert {len(line.split("\t")) for line in raw} == {13}


def test_babappa_resume_skips_completed_family(tmp_path: Path) -> None:
    module = _load_script_module("parallel_babappa_resume", SCRIPTS / "02_run_babappa_on_dataset.py")
    row = {
        "family_id": "F1",
        "truth_class": "positive",
        "expected_applicability": "in_domain",
        "codon_fasta": "missing.fasta",
        "tree": "missing.nwk",
        "foreground": "taxon1",
    }
    family_outdir = tmp_path / "scores" / "F1"
    (family_outdir / "scores").mkdir(parents=True)
    (family_outdir / "gene_summary.tsv").write_text(
        "max_gene_support\tdiagnostic_only\tresult_class\tn_called_positive\tapplicability_status\n"
        "0.7\tFalse\tdiagnostic_positive\t1\tin_domain\n",
        encoding="utf-8",
    )
    (family_outdir / "scores" / "empirical_branch_site_scores.tsv").write_text("prob_positive\n0.7\n", encoding="utf-8")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("resume should skip command execution")

    module.run_command = fail_if_called
    result, failure = module._run_family(
        row,
        benchmark_dir=tmp_path,
        results_dir=tmp_path / "scores",
        model_package="model",
        device="cpu",
        null_replicates="0",
        seed=1,
        smoke_surrogate=False,
        force=False,
    )
    assert failure is None
    assert result["status"] == "ok"
    status = json.loads((family_outdir / "benchmark_family_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "skipped_completed"


def test_absrel_parser_handles_missing_pending_outputs(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    subprocess.run(["python", str(SCRIPTS / "01_simulate_known_truth_dataset.py"), "--config", str(config)], check=True)
    subprocess.run(["python", str(SCRIPTS / "03_prepare_absrel_inputs.py"), "--config", str(config)], check=True)
    result = subprocess.run(["python", str(SCRIPTS / "04_run_absrel.py"), "--config", str(config), "--parse-only", "--continue-on-failure"], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    rows = read_tsv(tmp_path / "run" / "absrel_results.tsv")
    assert rows
    assert {row["status"] for row in rows} == {"pending_not_run"}


def test_absrel_thread_limiting_environment_is_set() -> None:
    module = _load_script_module("absrel_thread_env", SCRIPTS / "04_run_absrel.py")
    env = module._absrel_thread_env()
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["VECLIB_MAXIMUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"


def test_absrel_resume_skips_completed_json(tmp_path: Path) -> None:
    module = _load_script_module("absrel_resume", SCRIPTS / "04_run_absrel.py")
    output_json = tmp_path / "F1.absrel.json"
    output_json.write_text(json.dumps({"test results": {"positive test results": 1}}), encoding="utf-8")
    row = {
        "family_id": "F1",
        "alignment": str(tmp_path / "alignment.fasta"),
        "tree": str(tmp_path / "tree.nwk"),
        "branches": "Foreground",
        "output_json": str(output_json),
    }

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("resume should skip hyphy execution")

    module.run_command = fail_if_called
    failure = module._run_one_absrel(row, benchmark_dir=tmp_path, force=False)
    assert failure is None
    status = json.loads((tmp_path / "absrel_logs" / "F1" / "absrel_family_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "skipped_completed"


def test_shell_scripts_pass_benchmark_jobs() -> None:
    for name, default in [
        ("run_smoke.sh", "4"),
        ("run_pilot.sh", "12"),
        ("run_paper.sh", "14"),
        ("run_validation.sh", "14"),
        ("run_absrel_pilot.sh", "12"),
        ("run_absrel_paper.sh", "14"),
        ("run_absrel_validation.sh", "14"),
    ]:
        text = (BENCH / name).read_text(encoding="utf-8")
        assert "BABAPPA_BENCH_JOBS" in text
        assert f":-{default}" in text
        assert "--jobs" in text


def test_run_absrel_smoke_script_exists_and_executes_real_path() -> None:
    script = BENCH / "run_absrel_smoke.sh"
    assert script.exists()
    assert script.stat().st_mode & 0o111
    text = script.read_text(encoding="utf-8")
    assert "USER-RUN ONLY" in text
    assert "hyphy" in text.lower()
    assert "04_run_absrel.py" in text


def test_diagnose_smoke_script_exists() -> None:
    script = BENCH / "diagnose_smoke.sh"
    assert script.exists()
    assert script.stat().st_mode & 0o111
    assert "07_diagnose_run.py" in script.read_text(encoding="utf-8")


def test_diagnose_reports_score_unique_and_malformed_counts(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    subprocess.run(["python", str(SCRIPTS / "01_simulate_known_truth_dataset.py"), "--config", str(config)], check=True)
    subprocess.run(["python", str(SCRIPTS / "02_run_babappa_on_dataset.py"), "--config", str(config), "--smoke-surrogate"], check=True)
    subprocess.run(["python", str(SCRIPTS / "03_prepare_absrel_inputs.py"), "--config", str(config)], check=True)
    subprocess.run(["python", str(SCRIPTS / "04_run_absrel.py"), "--config", str(config), "--parse-only", "--continue-on-failure"], check=True)
    subprocess.run(["python", str(SCRIPTS / "05_compare_against_truth.py"), "--config", str(config)], check=True)
    result = subprocess.run(["python", str(SCRIPTS / "07_diagnose_run.py"), "--config", str(config)], text=True, capture_output=True, check=False)
    assert result.returncode == 0
    assert "babappa_score_unique_count" in result.stdout
    assert "babappa_malformed_rows" in result.stdout


def test_comparison_against_truth_computes_metrics(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    subprocess.run(["python", str(SCRIPTS / "01_simulate_known_truth_dataset.py"), "--config", str(config)], check=True)
    subprocess.run(["python", str(SCRIPTS / "02_run_babappa_on_dataset.py"), "--config", str(config), "--smoke-surrogate"], check=True)
    subprocess.run(["python", str(SCRIPTS / "03_prepare_absrel_inputs.py"), "--config", str(config)], check=True)
    subprocess.run(["python", str(SCRIPTS / "04_run_absrel.py"), "--config", str(config), "--parse-only", "--continue-on-failure"], check=True)
    result = subprocess.run(["python", str(SCRIPTS / "05_compare_against_truth.py"), "--config", str(config)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    subprocess.run(["python", str(SCRIPTS / "06_make_benchmark_report.py"), "--config", str(config)], check=True)
    table = read_tsv(tmp_path / "run" / "manuscript_table_babappa_vs_absrel.tsv")
    assert {row["method"] for row in table} == {"BABAPPA", "aBSREL"}
    assert "precision" in table[0]
    assert "simulator labels are the ground truth" in (tmp_path / "run" / "benchmark_summary.md").read_text().lower()
    raw = (tmp_path / "run" / "method_comparison.tsv").read_text(encoding="utf-8").splitlines()
    assert {len(line.split("\t")) for line in raw} == {10}


def test_validator_catches_shifted_malformed_rows(tmp_path: Path) -> None:
    bad = tmp_path / "method_comparison.tsv"
    bad.write_text(
        "family_id\tmethod\ttruth_class\ttruth_positive\texpected_applicability\tstatus\tstatus_class\tscore\tcall\tresult_class\n"
        "F1\tBABAPPA\tpositive\t1\tin_domain\tok\tmethod_positive\t0.9\t1\tdiagnostic_positive\n"
        "F2\tBABAPPA\tpositive\t1\tin_domain\tok\tmethod_negative\t0.0\tdiagnostic_negative\n",
        encoding="utf-8",
    )
    result = subprocess.run(["python", str(SCRIPTS / "validate_result_tables.py"), "--table", str(bad), "--outdir", str(tmp_path)], text=True, capture_output=True, check=False)
    assert result.returncode == 1
    assert "columns" in (tmp_path / "result_table_validation.json").read_text(encoding="utf-8")


def test_validator_catches_exact_malformed_babappa_method_row(tmp_path: Path) -> None:
    bad = tmp_path / "method_comparison.tsv"
    bad.write_text(
        "family_id\tmethod\ttruth_class\ttruth_positive\texpected_applicability\tstatus\tstatus_class\tscore\tcall\tresult_class\n"
        "SIM00002_null_moderate_divergence\tBABAPPA\tnull\t0\tin_domain\tok\tmethod_positive\t0.99998831749\tdiagnostic_positive\n",
        encoding="utf-8",
    )
    result = subprocess.run(["python", str(SCRIPTS / "validate_result_tables.py"), "--table", str(bad), "--outdir", str(tmp_path)], text=True, capture_output=True, check=False)
    assert result.returncode == 1
    payload = (tmp_path / "result_table_validation.json").read_text(encoding="utf-8")
    assert "line 2 has 9 columns" in payload
    assert "result_class appears shifted into call column" in payload


def test_validator_catches_exact_malformed_absrel_method_row(tmp_path: Path) -> None:
    bad = tmp_path / "method_comparison.tsv"
    bad.write_text(
        "family_id\tmethod\ttruth_class\ttruth_positive\texpected_applicability\tstatus\tstatus_class\tscore\tcall\tresult_class\n"
        "SIM00004_positive_weak_branch_site\taBSREL\tpositive\t1\tin_domain\tok\tmethod_positive\t1.0\tpositive\n",
        encoding="utf-8",
    )
    result = subprocess.run(["python", str(SCRIPTS / "validate_result_tables.py"), "--table", str(bad), "--outdir", str(tmp_path)], text=True, capture_output=True, check=False)
    assert result.returncode == 1
    payload = (tmp_path / "result_table_validation.json").read_text(encoding="utf-8")
    assert "line 2 has 9 columns" in payload
    assert "result_class appears shifted into call column" in payload


def test_validator_detects_result_class_shifted_into_call_column(tmp_path: Path) -> None:
    bad = tmp_path / "method_comparison.tsv"
    bad.write_text(
        "family_id\tmethod\ttruth_class\ttruth_positive\texpected_applicability\tstatus\tstatus_class\tscore\tcall\tresult_class\n"
        "F1\tBABAPPA\tpositive\t1\tin_domain\tok\tmethod_positive\t0.9\tdiagnostic_positive\t\n",
        encoding="utf-8",
    )
    result = subprocess.run(["python", str(SCRIPTS / "validate_result_tables.py"), "--table", str(bad), "--outdir", str(tmp_path)], text=True, capture_output=True, check=False)
    assert result.returncode == 1
    payload = (tmp_path / "result_table_validation.json").read_text(encoding="utf-8")
    assert "result_class appears shifted into call column" in payload
    assert "empty result_class" in payload


def test_diagnose_reports_malformed_method_comparison_rows(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    run = tmp_path / "run"
    (run / "truth").mkdir(parents=True)
    (run / "truth" / "family_truth.tsv").write_text(
        "family_id\tregime\ttruth_class\texpected_applicability\tcodon_fasta\ttree\tforeground\tpositive_branch\tn_taxa\tn_codons\tselected_site_count\n"
        "F1\tr\tpositive\tin_domain\ta\tb\ttaxon1\ttaxon1\t3\t10\t1\n",
        encoding="utf-8",
    )
    (run / "babappa_results.tsv").write_text(
        "family_id\tmethod\ttruth_class\ttruth_positive\texpected_applicability\tstatus\tstatus_class\tscore\tcall\tresult_class\tdiagnostic_only\tapplicability\tfailure_reason\n"
        "F1\tBABAPPA\tpositive\t1\tin_domain\tok\tmethod_positive\t0.9\t1\tdiagnostic_positive\tFalse\tin_domain\t\n",
        encoding="utf-8",
    )
    (run / "absrel_results.tsv").write_text(
        "family_id\tstatus\tpositive_count\tcall\tp_value\tnotes\n"
        "F1\tok\t1\t1\tNA\tofficial\n",
        encoding="utf-8",
    )
    (run / "method_comparison.tsv").write_text(
        "family_id\tmethod\ttruth_class\ttruth_positive\texpected_applicability\tstatus\tstatus_class\tscore\tcall\tresult_class\n"
        "F1\tBABAPPA\tpositive\t1\tin_domain\tok\tmethod_positive\t0.9\tdiagnostic_positive\n",
        encoding="utf-8",
    )
    result = subprocess.run(["python", str(SCRIPTS / "07_diagnose_run.py"), "--config", str(config)], text=True, capture_output=True, check=False)
    assert result.returncode == 0
    assert "method_comparison_malformed_rows\t" in result.stdout
    assert "method_comparison_malformed_rows\t0" not in result.stdout


def test_audit_pass_message_does_not_block_pilot(tmp_path: Path) -> None:
    module = _load_script_module("run_babappa_script_audit_message", SCRIPTS / "02_run_babappa_on_dataset.py")
    rows = [
        {
            "family_id": "F1",
            "method": "BABAPPA",
            "truth_class": "positive",
            "truth_positive": 1,
            "expected_applicability": "in_domain",
            "status": "ok",
            "status_class": "method_positive",
            "score": "0.1",
            "call": 1,
            "result_class": "diagnostic_positive",
            "diagnostic_only": "False",
            "applicability": "in_domain",
            "failure_reason": "",
        },
        {
            "family_id": "F2",
            "method": "BABAPPA",
            "truth_class": "positive",
            "truth_positive": 1,
            "expected_applicability": "in_domain",
            "status": "ok",
            "status_class": "method_positive",
            "score": "0.9",
            "call": 1,
            "result_class": "diagnostic_positive",
            "diagnostic_only": "False",
            "applicability": "in_domain",
            "failure_reason": "",
        },
    ]
    audit = module._audit_results(rows, tmp_path, allow_constant_scores=False)
    text = (tmp_path / "babappa_score_audit.md").read_text(encoding="utf-8")
    assert audit["status"] == "pass"
    assert "BABAPPA score audit passed" in text
    assert "must not be scaled" not in text


def test_smoke_gate_fails_if_method_comparison_malformed(tmp_path: Path) -> None:
    module = _load_script_module("compare_script_smoke_gate", SCRIPTS / "05_compare_against_truth.py")
    (tmp_path / "babappa_score_audit.json").write_text('{"status":"pass"}\n', encoding="utf-8")
    validation = {"status": "fail", "n_errors": 1, "errors": [{"file": "method_comparison.tsv", "row": 2, "reason": "call is not one of 0,1,NA"}]}
    status = module._smoke_status(tmp_path, [], [{"family_id": "F1", "status": "ok"}], "", validation)
    assert status["status"] == "smoke_fail_result_table_schema"
    assert status["ready_for_pilot"] is False


def test_all_zero_babappa_scores_fail_audit_unless_allowed(tmp_path: Path) -> None:
    module = _load_script_module("run_babappa_script", SCRIPTS / "02_run_babappa_on_dataset.py")
    rows = [
        {
            "family_id": "F1",
            "method": "BABAPPA",
            "truth_class": "positive",
            "truth_positive": 1,
            "expected_applicability": "in_domain",
            "status": "ok",
            "status_class": "method_negative",
            "score": "0.0",
            "call": 0,
            "result_class": "diagnostic_negative",
            "diagnostic_only": "False",
            "applicability": "in_domain",
            "failure_reason": "",
        }
    ]
    audit = module._audit_results(rows, tmp_path, allow_constant_scores=False)
    assert audit["status"] == "fail"
    assert "scores_all_zero" in audit["reasons"]
    allowed = module._audit_results(rows, tmp_path / "allowed", allow_constant_scores=True)
    assert "scores_all_zero" not in allowed["reasons"]


def test_constant_scores_warn_and_disable_score_metrics(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    run = tmp_path / "run"
    (run / "truth").mkdir(parents=True)
    (run / "truth" / "family_truth.tsv").write_text(
        "family_id\tregime\ttruth_class\texpected_applicability\tcodon_fasta\ttree\tforeground\tpositive_branch\tn_taxa\tn_codons\tselected_site_count\n"
        "F1\tr\tpositive\tin_domain\ta\tb\ttaxon1\ttaxon1\t3\t10\t1\n"
        "F2\tr\tnull\tin_domain\ta\tb\ttaxon1\t\t3\t10\t0\n",
        encoding="utf-8",
    )
    (run / "babappa_results.tsv").write_text(
        "family_id\tmethod\ttruth_class\ttruth_positive\texpected_applicability\tstatus\tstatus_class\tscore\tcall\tresult_class\tdiagnostic_only\tapplicability\tfailure_reason\n"
        "F1\tBABAPPA\tpositive\t1\tin_domain\tok\tmethod_negative\t0.5\t0\tdiagnostic_negative\tFalse\tin_domain\t\n"
        "F2\tBABAPPA\tnull\t0\tin_domain\tok\tmethod_negative\t0.5\t0\tdiagnostic_negative\tFalse\tin_domain\t\n",
        encoding="utf-8",
    )
    result = subprocess.run(["python", str(SCRIPTS / "05_compare_against_truth.py"), "--config", str(config)], text=True, capture_output=True, check=False)
    assert result.returncode == 0
    table = read_tsv(run / "manuscript_table_babappa_vs_absrel.tsv")
    babappa = next(row for row in table if row["method"] == "BABAPPA")
    assert babappa["auroc"] == "unavailable"
    assert babappa["warnings"] == "constant_or_missing_scores"


def test_ood_false_call_rate_uses_ood_null_denominator(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    run = tmp_path / "run"
    (run / "truth").mkdir(parents=True)
    (run / "truth" / "family_truth.tsv").write_text(
        "family_id\tregime\ttruth_class\texpected_applicability\tcodon_fasta\ttree\tforeground\tpositive_branch\tn_taxa\tn_codons\tselected_site_count\n"
        "N1\tr\tood_null\tout_of_domain\ta\tb\ttaxon1\t\t3\t10\t0\n"
        "P1\tr\tood_positive\tout_of_domain\ta\tb\ttaxon1\ttaxon1\t3\t10\t1\n",
        encoding="utf-8",
    )
    (run / "babappa_results.tsv").write_text(
        "family_id\tmethod\ttruth_class\ttruth_positive\texpected_applicability\tstatus\tstatus_class\tscore\tcall\tresult_class\tdiagnostic_only\tapplicability\tfailure_reason\n"
        "N1\tBABAPPA\tood_null\t0\tout_of_domain\tok\tmethod_positive\t0.9\t1\tdiagnostic_positive\tFalse\tout_of_domain\t\n"
        "P1\tBABAPPA\tood_positive\t1\tout_of_domain\tok\tmethod_positive\t0.8\t1\tdiagnostic_positive\tFalse\tout_of_domain\t\n",
        encoding="utf-8",
    )
    result = subprocess.run(["python", str(SCRIPTS / "05_compare_against_truth.py"), "--config", str(config)], text=True, capture_output=True, check=False)
    assert result.returncode == 0
    babappa = next(row for row in read_tsv(run / "manuscript_table_babappa_vs_absrel.tsv") if row["method"] == "BABAPPA")
    assert babappa["ood_null_denominator"] == "1"
    assert babappa["ood_null_false_calls"] == "1"
    assert babappa["ood_positive_denominator"] == "1"
    assert babappa["ood_positive_calls"] == "1"


def test_compare_warns_if_absrel_outputs_are_absent(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    subprocess.run(["python", str(SCRIPTS / "01_simulate_known_truth_dataset.py"), "--config", str(config)], check=True)
    subprocess.run(["python", str(SCRIPTS / "02_run_babappa_on_dataset.py"), "--config", str(config), "--smoke-surrogate"], check=True)
    result = subprocess.run(["python", str(SCRIPTS / "05_compare_against_truth.py"), "--config", str(config)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "run_absrel_smoke.sh" in result.stdout
    table = read_tsv(tmp_path / "run" / "manuscript_table_babappa_vs_absrel.tsv")
    absrel = next(row for row in table if row["method"] == "aBSREL")
    assert absrel["pending_not_run"] == "12"


def test_pilot_scripts_are_user_run_but_execute_real_steps() -> None:
    for name in ["run_pilot.sh", "run_absrel_pilot.sh", "run_paper.sh", "run_absrel_smoke.sh"]:
        text = (BENCH / name).read_text(encoding="utf-8")
        assert "USER-RUN ONLY" in text
        assert "python" in text
        assert "01_simulate_known_truth_dataset.py" in text or "04_run_absrel.py" in text


def test_no_extra_reference_workflows_appear_in_simplified_benchmark() -> None:
    haystack = "\n".join(path.read_text(encoding="utf-8") for path in list(SCRIPTS.glob("*.py")) + list(BENCH.glob("*")) if path.is_file())
    assert "codeml" not in haystack.lower()
    assert "meme" not in haystack.lower()


def test_benchmark_readme_states_not_replacement() -> None:
    text = (BENCH / "README.md").read_text(encoding="utf-8")
    assert "not presented here as a replacement for aBSREL" in text


def _write_method_comparison(path: Path, rows: list[str]) -> None:
    path.write_text(
        "family_id\tmethod\ttruth_class\ttruth_positive\texpected_applicability\tstatus\tstatus_class\tscore\tcall\tresult_class\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def test_threshold_sweep_handles_zero_positive_default_calls(tmp_path: Path) -> None:
    module = _load_script_module("threshold_sweep", SCRIPTS / "08_threshold_sweep.py")
    run = tmp_path / "run"
    run.mkdir()
    _write_method_comparison(
        run / "method_comparison.tsv",
        [
            "P1\tBABAPPA\tpositive\t1\tin_domain\tok\tmethod_negative\t0.9\t0\tdiagnostic_negative",
            "N1\tBABAPPA\tnull\t0\tin_domain\tok\tmethod_negative\t0.2\t0\tdiagnostic_negative",
        ],
    )
    rows = module._load_babappa_rows(run)
    sweep = module.compute_threshold_sweep(rows)
    rec = module.recommend_policies(rows, sweep)
    current = next(row for row in rec["policies"] if row["policy"] == "ultra_conservative_current")
    assert current["positive_calls"] == 0
    assert any(int(row["positive_calls"]) > 0 for row in sweep)


def test_threshold_sweep_computes_fdr_recall_specificity(tmp_path: Path) -> None:
    module = _load_script_module("threshold_sweep_metrics", SCRIPTS / "08_threshold_sweep.py")
    rows = [
        {"family_id": "P1", "truth_class": "positive", "truth_positive": 1, "expected_applicability": "in_domain", "score": 0.9, "current_call": 0},
        {"family_id": "N1", "truth_class": "null", "truth_positive": 0, "expected_applicability": "in_domain", "score": 0.8, "current_call": 0},
        {"family_id": "N2", "truth_class": "null", "truth_positive": 0, "expected_applicability": "in_domain", "score": 0.1, "current_call": 0},
    ]
    metrics = module._metrics_for_calls(rows, [1, 1, 0], 0.8)
    assert metrics["empirical_fdr"] == 0.5
    assert metrics["recall_power"] == 1.0
    assert metrics["specificity"] == 0.5


def test_fdr_005_policy_returns_no_call_if_no_valid_threshold(tmp_path: Path) -> None:
    module = _load_script_module("threshold_sweep_no_fdr", SCRIPTS / "08_threshold_sweep.py")
    rows = [
        {"family_id": "N1", "truth_class": "null", "truth_positive": 0, "expected_applicability": "in_domain", "score": 0.9, "current_call": 0},
        {"family_id": "P1", "truth_class": "positive", "truth_positive": 1, "expected_applicability": "in_domain", "score": 0.1, "current_call": 0},
    ]
    rec = module.recommend_policies(rows, module.compute_threshold_sweep(rows))
    fdr = next(row for row in rec["policies"] if row["policy"] == "FDR_0.05_policy")
    assert fdr["status"] == "no_valid_threshold"
    assert fdr["threshold"] == "NA"


def test_balanced_mcc_policy_selects_expected_threshold(tmp_path: Path) -> None:
    module = _load_script_module("threshold_sweep_mcc", SCRIPTS / "08_threshold_sweep.py")
    rows = [
        {"family_id": "P1", "truth_class": "positive", "truth_positive": 1, "expected_applicability": "in_domain", "score": 0.9, "current_call": 0},
        {"family_id": "N1", "truth_class": "null", "truth_positive": 0, "expected_applicability": "in_domain", "score": 0.1, "current_call": 0},
    ]
    rec = module.recommend_policies(rows, module.compute_threshold_sweep(rows))
    balanced = next(row for row in rec["policies"] if row["policy"] == "balanced_MCC_policy")
    assert float(balanced["threshold"]) == 0.9
    assert float(balanced["mcc"]) == 1.0


def test_ood_safe_policy_enforces_zero_ood_false_calls(tmp_path: Path) -> None:
    module = _load_script_module("threshold_sweep_ood", SCRIPTS / "08_threshold_sweep.py")
    rows = [
        {"family_id": "P1", "truth_class": "positive", "truth_positive": 1, "expected_applicability": "in_domain", "score": 0.8, "current_call": 0},
        {"family_id": "O1", "truth_class": "ood_null", "truth_positive": 0, "expected_applicability": "out_of_domain", "score": 1.0, "current_call": 0},
    ]
    rec = module.recommend_policies(rows, module.compute_threshold_sweep(rows))
    ood = next(row for row in rec["policies"] if row["policy"] == "OOD_safe_policy")
    assert float(ood["ood_null_false_call_rate"]) == 0.0
    assert int(ood["positive_calls"]) == 1


def test_operating_point_table_includes_absrel_default(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    _write_method_comparison(
        run / "method_comparison.tsv",
        [
            "P1\tBABAPPA\tpositive\t1\tin_domain\tok\tmethod_negative\t0.9\t0\tdiagnostic_negative",
            "N1\tBABAPPA\tnull\t0\tin_domain\tok\tmethod_negative\t0.1\t0\tdiagnostic_negative",
            "P1\taBSREL\tpositive\t1\tin_domain\tok\tmethod_positive\t1.0\t1\tpositive",
            "N1\taBSREL\tnull\t0\tin_domain\tok\tmethod_negative\t0.0\t0\tnegative",
        ],
    )
    subprocess.run(["python", str(SCRIPTS / "08_threshold_sweep.py"), "--config", str(config)], check=True)
    subprocess.run(["python", str(SCRIPTS / "09_compare_operating_points.py"), "--config", str(config)], check=True)
    table = read_tsv(run / "operating_point_comparison.tsv")
    assert any(row["method"] == "aBSREL" and row["operating_point"] == "default" for row in table)


def test_report_includes_zero_positive_default_call_warning(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "benchmark_summary.json").write_text(json.dumps({"methods": []}) + "\n", encoding="utf-8")
    (run / "manuscript_table_babappa_vs_absrel.tsv").write_text(
        "method\tfamilies_total\tfamilies_evaluable\tpending_not_run\tfailed\tpositive\tnegative\tdiagnostic_only\tno_call\tinconclusive\tfailure_rate\tscore_metrics_available\tauroc\tauprc\twarnings\tprecision\trecall_power\tspecificity\tf1\tmcc\tfpr\tfnr\tempirical_fdr\tood_null_denominator\tood_null_false_calls\tood_false_call_rate\tood_positive_denominator\tood_positive_calls\tno_positive_call_note\n"
        "BABAPPA\t2\t2\t0\t0\t0\t2\t0\t0\t0\t0.0\tTrue\t0.8\t0.8\t\t0.0\t0.0\t1.0\t0.0\t0.0\t0.0\t1.0\t0.0\t0\t0\t0.0\t0\t0\tno positive calls made\n",
        encoding="utf-8",
    )
    subprocess.run(["python", str(SCRIPTS / "06_make_benchmark_report.py"), "--config", str(config)], check=True)
    text = (run / "benchmark_summary.md").read_text(encoding="utf-8")
    assert "default threshold produced zero positive calls" in text


def test_threshold_policy_yaml_contains_fixed_pilot_thresholds() -> None:
    text = (BENCH / "threshold_policy.yaml").read_text(encoding="utf-8")
    assert "primary_calibrated_caller:" in text
    assert "balanced_MCC_policy_from_pilot" in text
    assert "7.25302066731e-31" in text
    assert "FDR_0.05_policy_from_pilot" in text
    assert "2.84844072913e-23" in text
    assert "frozen_for_paper: true" in text


def test_validation_profile_uses_independent_seed_and_output_dir() -> None:
    paper = (BENCH / "config_paper.yaml").read_text(encoding="utf-8")
    validation = (BENCH / "config_validation.yaml").read_text(encoding="utf-8")
    assert "profile: validation" in validation
    assert "n_families: 5000" in validation
    assert "outdir: benchmark_runs/known_truth_absrel_validation" in validation
    assert "seed: 42" in paper
    assert "seed: 20260604" in validation


def test_validation_candidate_policy_freezes_posthoc_paper_thresholds() -> None:
    text = (BENCH / "threshold_policy_validation_candidate.yaml").read_text(encoding="utf-8")
    assert "frozen_for_independent_validation: true" in text
    assert "posthoc_best_mcc_candidate:" in text
    assert "posthoc_best_MCC_from_paper" in text
    assert "1.31794959637e-15" in text
    assert "6.67112455467e-16" in text
    assert "4.17948058384e-07" in text
    assert "validation hypothesis only" in text


def test_frozen_policy_application_computes_expected_metrics_on_synthetic_data(tmp_path: Path) -> None:
    module = _load_script_module("frozen_policy", SCRIPTS / "10_apply_frozen_threshold_policy.py")
    run = tmp_path / "run"
    run.mkdir()
    _write_method_comparison(
        run / "method_comparison.tsv",
        [
            "P1\tBABAPPA\tpositive\t1\tin_domain\tok\tmethod_negative\t0.9\t0\tdiagnostic_negative",
            "N1\tBABAPPA\tnull\t0\tin_domain\tok\tmethod_negative\t0.1\t0\tdiagnostic_negative",
            "O1\tBABAPPA\tood_null\t0\tout_of_domain\tok\tdiagnostic_only\t1.0\t0\tdiagnostic_only",
        ],
    )
    policy = tmp_path / "threshold_policy.yaml"
    policy.write_text(
        "policy_version: 1\n"
        "policies:\n"
        "  primary_score_ranking:\n"
        "    policy_name: primary_score_ranking\n"
        "    policy_type: score_ranking\n"
        "    threshold: NA\n"
        "    threshold_source: none\n"
        "  primary_calibrated_caller:\n"
        "    policy_name: balanced_MCC_policy_from_pilot\n"
        "    policy_type: binary_threshold\n"
        "    threshold: 0.5\n"
        "    threshold_source: pilot threshold sweep\n",
        encoding="utf-8",
    )
    rows = module.apply_frozen_policy(run, policy)
    balanced = next(row for row in rows if row["policy_id"] == "primary_calibrated_caller")
    assert balanced["positive_calls"] == 1
    assert balanced["precision"] == 1.0
    assert balanced["recall_power"] == 1.0
    assert balanced["ood_null_false_call_rate"] == 0.0


def test_paper_comparison_refuses_without_threshold_policy_copy(tmp_path: Path) -> None:
    bench = tmp_path / "benchmarks" / "known_truth_absrel"
    scripts = tmp_path / "scripts" / "benchmarks" / "known_truth_absrel"
    bench.mkdir(parents=True)
    scripts.mkdir(parents=True)
    compare = bench / "compare_paper.sh"
    compare.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "ROOT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/../..\" && pwd)\"\n"
        "POLICY=\"${ROOT_DIR}/benchmarks/known_truth_absrel/threshold_policy.yaml\"\n"
        "if [[ ! -s \"${POLICY}\" ]]; then\n"
        "  echo \"Freeze threshold policy from pilot before paper benchmark.\"\n"
        "  exit 1\n"
        "fi\n",
        encoding="utf-8",
    )
    compare.chmod(0o755)
    result = subprocess.run(["bash", str(compare)], text=True, capture_output=True, check=False)
    assert result.returncode == 1
    assert "Freeze threshold policy from pilot before paper benchmark." in result.stdout


def test_paper_comparison_script_uses_frozen_policy_without_recalibration() -> None:
    text = (BENCH / "compare_paper.sh").read_text(encoding="utf-8")
    assert "threshold_policy.yaml" in text
    assert "10_apply_frozen_threshold_policy.py" in text
    assert "08_threshold_sweep.py" not in text
    assert "09_compare_operating_points.py" not in text


def test_validation_comparison_script_uses_validation_candidate_policy_without_recalibration() -> None:
    text = (BENCH / "compare_validation.sh").read_text(encoding="utf-8")
    assert "config_validation.yaml" in text
    assert "threshold_policy_validation_candidate.yaml" in text
    assert "10_apply_frozen_threshold_policy.py" in text
    assert "08_threshold_sweep.py" not in text
    assert "09_compare_operating_points.py" not in text
    assert "threshold_policy.yaml" not in text


def test_report_states_pilot_selected_threshold_is_frozen_for_paper(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "benchmark_summary.json").write_text(json.dumps({"methods": []}) + "\n", encoding="utf-8")
    (run / "manuscript_table_babappa_vs_absrel.tsv").write_text(
        "method\tfamilies_total\tfamilies_evaluable\tpending_not_run\tfailed\tpositive\tnegative\tdiagnostic_only\tno_call\tinconclusive\tfailure_rate\tscore_metrics_available\tauroc\tauprc\twarnings\tprecision\trecall_power\tspecificity\tf1\tmcc\tfpr\tfnr\tempirical_fdr\tood_null_denominator\tood_null_false_calls\tood_false_call_rate\tood_positive_denominator\tood_positive_calls\tno_positive_call_note\n"
        "BABAPPA\t2\t2\t0\t0\t0\t2\t0\t0\t0\t0.0\tTrue\t0.8\t0.8\t\t0.0\t0.0\t1.0\t0.0\t0.0\t0.0\t1.0\t0.0\t0\t0\t0.0\t0\t0\tno positive calls made\n",
        encoding="utf-8",
    )
    (run / "frozen_policy_results.tsv").write_text(
        "method\tpolicy_id\tpolicy_name\tpolicy_type\tthreshold_source\tthreshold\tpositive_calls\tprecision\trecall_power\tspecificity\tf1\tmcc\tempirical_fdr\tfpr\tfnr\tood_null_denominator\tood_null_false_calls\tood_null_false_call_rate\tood_positive_denominator\tood_positive_calls\tood_positive_call_rate\tscore_auroc\tscore_auprc\tnotes\n"
        "BABAPPA\tprimary_calibrated_caller\tbalanced_MCC_policy_from_pilot\tbinary_threshold\tpilot threshold sweep\t0.5\t1\t1.0\t1.0\t1.0\t1.0\t1.0\t0.0\t0.0\t0.0\t0\t0\t0.0\t0\t0\t0.0\t0.8\t0.8\tapply unchanged\n",
        encoding="utf-8",
    )
    subprocess.run(["python", str(SCRIPTS / "06_make_benchmark_report.py"), "--config", str(config)], check=True)
    text = (run / "benchmark_summary.md").read_text(encoding="utf-8")
    assert "calibrated policy was selected on the pilot profile" in text
    assert "paper profile must use this policy unchanged" in text
    assert "evaluation set, not a threshold-tuning set" in text


def test_report_states_paper_derived_policy_requires_independent_validation(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "benchmark_summary.json").write_text(json.dumps({"methods": []}) + "\n", encoding="utf-8")
    (run / "manuscript_table_babappa_vs_absrel.tsv").write_text(
        "method\tfamilies_total\tfamilies_evaluable\tpending_not_run\tfailed\tpositive\tnegative\tdiagnostic_only\tno_call\tinconclusive\tfailure_rate\tscore_metrics_available\tauroc\tauprc\twarnings\tprecision\trecall_power\tspecificity\tf1\tmcc\tfpr\tfnr\tempirical_fdr\tood_null_denominator\tood_null_false_calls\tood_false_call_rate\tood_positive_denominator\tood_positive_calls\tno_positive_call_note\n"
        "BABAPPA\t2\t2\t0\t0\t0\t2\t0\t0\t0\t0.0\tTrue\t0.8\t0.8\t\t0.0\t0.0\t1.0\t0.0\t0.0\t0.0\t1.0\t0.0\t0\t0\t0.0\t0\t0\tno positive calls made\n",
        encoding="utf-8",
    )
    (run / "frozen_policy_results.tsv").write_text(
        "method\tpolicy_id\tpolicy_name\tpolicy_type\tthreshold_source\tthreshold\tpositive_calls\tprecision\trecall_power\tspecificity\tf1\tmcc\tempirical_fdr\tfpr\tfnr\tood_null_denominator\tood_null_false_calls\tood_null_false_call_rate\tood_positive_denominator\tood_positive_calls\tood_positive_call_rate\tscore_auroc\tscore_auprc\tnotes\n"
        "BABAPPA\tposthoc_best_mcc_candidate\tposthoc_best_MCC_from_paper\tbinary_threshold\tposthoc paper threshold sweep\t1.31794959637e-15\t1\t1.0\t1.0\t1.0\t1.0\t1.0\t0.0\t0.0\t0.0\t0\t0\t0.0\t0\t0\t0.0\t0.8\t0.8\tapply unchanged\n",
        encoding="utf-8",
    )
    subprocess.run(["python", str(SCRIPTS / "06_make_benchmark_report.py"), "--config", str(config)], check=True)
    text = (run / "benchmark_summary.md").read_text(encoding="utf-8")
    assert "Frozen Paper-Derived Validation-Candidate Threshold Policy" in text
    assert "applied unchanged to an independent validation profile" in text
    assert "evaluation set, not a threshold-tuning set" in text
