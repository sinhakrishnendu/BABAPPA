import json

from typer.testing import CliRunner

from babappa.branch import (
    ExplicitBranchTruth10kMacPlanConfig,
    ExplicitBranchTruth100kMacPlanConfig,
    plan_explicit_branch_truth_10k_mac,
    plan_explicit_branch_truth_100k_mac,
)
from babappa.cli import app
from babappa.training.mps import MPSTrainingSmokeConfig, run_mps_training_smoke, validate_mps_smoke_dir
from babappa.training.neural_env import resolve_torch_device


runner = CliRunner()


class _FakeCuda:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available

    def device_count(self):
        return 1 if self._available else 0

    def get_device_name(self, index):
        return f"fake_cuda_{index}"


class _FakeMPS:
    def __init__(self, available):
        self._available = available

    def is_built(self):
        return True

    def is_available(self):
        return self._available


class _FakeBackends:
    def __init__(self, mps_available):
        self.mps = _FakeMPS(mps_available)


class _FakeTorch:
    def __init__(self, cuda_available=False, mps_available=False):
        self.cuda = _FakeCuda(cuda_available)
        self.backends = _FakeBackends(mps_available)


def test_device_resolver_detects_mps_with_monkeypatch() -> None:
    assert resolve_torch_device(_FakeTorch(cuda_available=False, mps_available=True), "auto") == "mps"
    assert resolve_torch_device(_FakeTorch(cuda_available=True, mps_available=True), "auto") == "cuda"
    assert resolve_torch_device(_FakeTorch(cuda_available=True, mps_available=True), "auto", prefer_mps=True) == "mps"


def test_branch_and_site_neural_clis_accept_mps_device(tmp_path) -> None:
    branch = runner.invoke(
        app,
        [
            "train-branch-site-neural",
            "--branch-site-dataset-dir",
            str(tmp_path / "missing_branch_dataset"),
            "--outdir",
            str(tmp_path / "branch_model"),
            "--device",
            "mps",
        ],
    )
    site = runner.invoke(
        app,
        [
            "train-site-neural",
            "--site-dataset-dir",
            str(tmp_path / "missing_site_dataset"),
            "--outdir",
            str(tmp_path / "site_model"),
            "--device",
            "mps",
        ],
    )

    assert branch.exit_code != 0
    assert site.exit_code != 0
    assert "device must be one of" not in branch.output
    assert "device must be one of" not in site.output
    assert "not exist" in branch.output
    assert "not exist" in site.output


def test_mps_smoke_skips_gracefully_without_torch(tmp_path, monkeypatch) -> None:
    import babappa.training.mps as mps_module

    monkeypatch.setattr(mps_module, "safe_import_torch", lambda: (None, "missing"))
    outdir = tmp_path / "mps_smoke"

    summary = run_mps_training_smoke(MPSTrainingSmokeConfig(outdir=str(outdir), device="mps"))
    validation = validate_mps_smoke_dir(outdir)

    assert summary["status"] == "skipped"
    assert (outdir / "mps_smoke_report.json").exists()
    assert (outdir / "mps_smoke_report.md").exists()
    assert validation["status"] == "skipped"


def test_mac_10k_planner_generates_mps_scripts_without_cuda(tmp_path) -> None:
    outdir = tmp_path / "explicit_branch_truth_10k_mps_plan"

    summary = plan_explicit_branch_truth_10k_mac(
        ExplicitBranchTruth10kMacPlanConfig(outdir=str(outdir), tiers="low")
    )
    expected = json.loads((outdir / "expected_outputs.json").read_text("utf-8"))
    run_text = (outdir / "run_explicit_branch_truth_10k_mps.sh").read_text("utf-8")
    monitor_text = (outdir / "monitor_explicit_branch_truth_10k_mps.sh").read_text("utf-8")

    assert summary["does_not_run_jobs"] is True
    assert "USER-RUN ONLY" in run_text
    assert "PYTORCH_ENABLE_MPS_FALLBACK=1" in run_text
    assert "babappalign_model_missing" in run_text
    assert "$HOME/.cache/babappalign/models/babappascore.pt" in run_text
    assert "curl -L \"https://zenodo.org/record/18053201/files/babappascore.pt\"" in run_text
    assert "CUDA_VISIBLE_DEVICES" not in run_text
    assert "nvidia-smi" not in run_text
    assert "nvidia-smi" not in monitor_text
    assert "vm_stat" in monitor_text
    assert "top -l 1 -o mem -n 20" in monitor_text
    assert expected["feature_policy"] == "conservative_branch_site"
    assert expected["truth_mode"] == "explicit"
    assert "--truth-mode explicit" in run_text
    assert "--feature-policy conservative_branch_site" in run_text
    assert "--saturation-tier low --workers \"$BABAPPA_PERF_WORKERS\"" in run_text
    assert "--threads \"$BABAPPA_TORCH_THREADS\"" in run_text
    assert "--n-permutations 100 --seed 42 --workers \"$BABAPPA_PERF_WORKERS\"" in run_text
    assert "BABAPPA_PERF_WORKERS" in run_text
    assert "BABAPPA_MPS_BATCH_SIZE" in run_text
    assert "BABAPPA_TORCH_THREADS" in run_text
    assert "branch_site_dataset_explicit_branch_truth_10k_mps_low_streamed" in run_text


def test_mac_100k_planner_is_gated_until_10k_passes(tmp_path) -> None:
    outdir = tmp_path / "explicit_branch_truth_100k_mps_plan"

    plan_explicit_branch_truth_100k_mac(
        ExplicitBranchTruth100kMacPlanConfig(outdir=str(outdir), tiers="low")
    )
    run_text = (outdir / "run_explicit_branch_truth_100k_mps.sh").read_text("utf-8")
    markdown = (outdir / "explicit_branch_truth_100k_mps_plan.md").read_text("utf-8")
    monitor_text = (outdir / "monitor_explicit_branch_truth_100k_mps.sh").read_text("utf-8")

    assert "DO NOT RUN 100K until the conservative 10K MPS plan completes and validates" in run_text
    assert "DO NOT RUN 100K until the 10K MPS plan completes and validates" in markdown
    assert "BABAPPA_ALLOW_100K_AFTER_10K=1" in run_text
    assert "nvidia-smi" not in monitor_text
    assert "memory_pressure" in monitor_text
