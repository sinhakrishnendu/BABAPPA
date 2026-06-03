import subprocess
from pathlib import Path

from babappa.datasets.index import read_tsv


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "benchmarks" / "known_truth_absrel"
BENCH = ROOT / "benchmarks" / "known_truth_absrel"


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
    result = subprocess.run(["python", str(SCRIPTS / "02_run_babappa_on_dataset.py"), "--config", str(config), "--mock-for-tests"], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    rows = read_tsv(tmp_path / "run" / "babappa_results.tsv")
    assert len(rows) == 12
    assert {row["status"] for row in rows} == {"ok"}


def test_absrel_parser_handles_missing_pending_outputs(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    subprocess.run(["python", str(SCRIPTS / "01_simulate_known_truth_dataset.py"), "--config", str(config)], check=True)
    subprocess.run(["python", str(SCRIPTS / "03_prepare_absrel_inputs.py"), "--config", str(config)], check=True)
    result = subprocess.run(["python", str(SCRIPTS / "04_run_absrel.py"), "--config", str(config), "--parse-only", "--continue-on-failure"], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    rows = read_tsv(tmp_path / "run" / "absrel_results.tsv")
    assert rows
    assert {row["status"] for row in rows} == {"pending_not_run"}


def test_comparison_against_truth_computes_metrics(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    subprocess.run(["python", str(SCRIPTS / "01_simulate_known_truth_dataset.py"), "--config", str(config)], check=True)
    subprocess.run(["python", str(SCRIPTS / "02_run_babappa_on_dataset.py"), "--config", str(config), "--mock-for-tests"], check=True)
    subprocess.run(["python", str(SCRIPTS / "03_prepare_absrel_inputs.py"), "--config", str(config)], check=True)
    subprocess.run(["python", str(SCRIPTS / "04_run_absrel.py"), "--config", str(config), "--parse-only", "--continue-on-failure"], check=True)
    result = subprocess.run(["python", str(SCRIPTS / "05_compare_against_truth.py"), "--config", str(config)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    subprocess.run(["python", str(SCRIPTS / "06_make_benchmark_report.py"), "--config", str(config)], check=True)
    table = read_tsv(tmp_path / "run" / "manuscript_table_babappa_vs_absrel.tsv")
    assert {row["method"] for row in table} == {"BABAPPA", "aBSREL"}
    assert "precision" in table[0]
    assert "simulator labels are the ground truth" in (tmp_path / "run" / "benchmark_summary.md").read_text().lower()


def test_pilot_scripts_are_user_run_but_execute_real_steps() -> None:
    for name in ["run_pilot.sh", "run_absrel_pilot.sh", "run_paper.sh"]:
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
