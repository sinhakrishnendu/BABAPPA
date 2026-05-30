import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from babappa.branch import (
    ExplicitBranchTruth10kMacPlanConfig,
    ExplicitBranchTruth100kMacPlanConfig,
    plan_explicit_branch_truth_10k_mac,
    plan_explicit_branch_truth_100k_mac,
)
from babappa.branch import mps_preflight
from babappa.branch.mps_preflight import (
    MPSPlanPreflightConfig,
    MPSPlanScriptValidationConfig,
    preflight_explicit_branch_truth_mps_plan,
    validate_mps_plan_script,
)
from babappa.cli import app

runner = CliRunner()


def test_preflight_command_exists_and_writes_reports(tmp_path) -> None:
    plan_dir = _synthetic_plan(tmp_path)

    result = runner.invoke(
        app,
        [
            "preflight-explicit-branch-truth-mps-plan",
            "--plan-dir",
            str(plan_dir),
            "--scale",
            "10k",
            "--require-babappalign",
            "false",
            "--require-mps",
            "false",
            "--conda-env",
            "molevo",
        ],
    )

    assert (plan_dir / "preflight_report.json").exists()
    assert (plan_dir / "preflight_report.tsv").exists()
    assert (plan_dir / "preflight_report.md").exists()
    assert "Preflight" in result.output


def test_generated_10k_mps_script_calls_preflight_before_simulation(tmp_path) -> None:
    outdir = tmp_path / "explicit_branch_truth_10k_mps_plan"
    plan_explicit_branch_truth_10k_mac(
        ExplicitBranchTruth10kMacPlanConfig(outdir=str(outdir), tiers="low")
    )
    text = (outdir / "run_explicit_branch_truth_10k_mps.sh").read_text("utf-8")
    assert text.index("preflight-explicit-branch-truth-mps-plan") < text.index("babappa simulate")


def test_generated_100k_mps_script_calls_preflight_before_simulation(tmp_path) -> None:
    outdir = tmp_path / "explicit_branch_truth_100k_mps_plan"
    plan_explicit_branch_truth_100k_mac(
        ExplicitBranchTruth100kMacPlanConfig(outdir=str(outdir), tiers="low")
    )
    text = (outdir / "run_explicit_branch_truth_100k_mps.sh").read_text("utf-8")
    assert text.index("preflight-explicit-branch-truth-mps-plan") < text.index("babappa simulate")


def test_preflight_detects_missing_babappalign_model(tmp_path, monkeypatch) -> None:
    plan_dir = _synthetic_plan(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "empty_home"))

    summary = preflight_explicit_branch_truth_mps_plan(
        MPSPlanPreflightConfig(
            plan_dir=str(plan_dir),
            scale="10k",
            require_babappalign=True,
            require_mps=False,
        )
    )

    model_check = _check(summary, "babappalign_model_cache")
    assert summary["status"] == "fail"
    assert model_check["status"] == "fail"
    assert model_check["message"] == "babappalign_model_missing"


def test_preflight_detects_mps_unavailable_when_required(tmp_path, monkeypatch) -> None:
    plan_dir = _synthetic_plan(tmp_path)

    def fake_mps(rows, required):
        mps_preflight._add_check(rows, "torch_mps_tiny_tensor", "fail", required, "MPS unavailable")

    monkeypatch.setattr(mps_preflight, "_check_mps", fake_mps)
    summary = preflight_explicit_branch_truth_mps_plan(
        MPSPlanPreflightConfig(
            plan_dir=str(plan_dir),
            scale="10k",
            require_babappalign=False,
            require_mps=True,
        )
    )

    assert _check(summary, "torch_mps_tiny_tensor")["status"] == "fail"
    assert summary["status"] == "fail"


def test_preflight_detects_bare_t_sim_in_bad_script(tmp_path) -> None:
    plan_dir = _synthetic_plan(tmp_path, extra_run_line="t_sim")

    summary = validate_mps_plan_script(MPSPlanScriptValidationConfig(plan_dir=str(plan_dir)))

    token_check = _check(summary, "suspicious_bare_tokens")
    assert summary["status"] == "fail"
    assert token_check["status"] == "fail"
    assert token_check["details"]["findings"][0]["token"] == "t_sim"


def test_preflight_detects_nvidia_smi_in_mac_script(tmp_path) -> None:
    plan_dir = _synthetic_plan(tmp_path, extra_run_line="nvidia-smi")

    summary = validate_mps_plan_script(MPSPlanScriptValidationConfig(plan_dir=str(plan_dir)))

    assert _check(summary, "no_nvidia_smi")["status"] == "fail"


def test_preflight_detects_cuda_visible_devices_in_mac_script(tmp_path) -> None:
    plan_dir = _synthetic_plan(tmp_path, extra_run_line="export CUDA_VISIBLE_DEVICES=0")

    summary = validate_mps_plan_script(MPSPlanScriptValidationConfig(plan_dir=str(plan_dir)))

    assert _check(summary, "no_cuda_visible_devices")["status"] == "fail"


def test_preflight_detects_absent_mps_fallback_export(tmp_path) -> None:
    plan_dir = _synthetic_plan(tmp_path, include_mps_fallback=False)

    summary = validate_mps_plan_script(MPSPlanScriptValidationConfig(plan_dir=str(plan_dir)))

    assert _check(summary, "mps_fallback_export")["status"] == "fail"


def test_preflight_babappalign_smoke_success(monkeypatch, tmp_path) -> None:
    def fake_smoke(method, outdir, device="cpu", timeout_seconds=60):
        path = Path(outdir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tiny.babappalign.codon.fasta").write_text(
            ">taxon_a\nATGAAA\n>taxon_b\nATGAAA\n",
            encoding="utf-8",
        )
        return {"status": "ok", "reason": "ok"}

    rows = []
    monkeypatch.setattr(mps_preflight, "smoke_aligner", fake_smoke)
    mps_preflight._check_babappalign_smoke(rows, required=True, plan_dir=tmp_path)
    assert rows[0]["name"] == "babappalign_tiny_smoke"
    assert rows[0]["status"] == "pass"


def test_validate_mps_plan_script_catches_script_syntax_errors(tmp_path) -> None:
    plan_dir = _synthetic_plan(tmp_path, extra_run_line="if broken")

    summary = validate_mps_plan_script(MPSPlanScriptValidationConfig(plan_dir=str(plan_dir)))

    assert summary["status"] == "fail"
    assert _check(summary, "bash_n:run")["status"] == "fail"


def test_generated_scripts_pass_bash_n(tmp_path) -> None:
    outdir = tmp_path / "explicit_branch_truth_10k_mps_plan"
    plan_explicit_branch_truth_10k_mac(
        ExplicitBranchTruth10kMacPlanConfig(outdir=str(outdir), tiers="low")
    )
    for script in outdir.glob("*.sh"):
        proc = subprocess.run(["bash", "-n", str(script)], check=False)
        assert proc.returncode == 0, script


def test_generated_scripts_contain_no_bare_t_stage_tokens(tmp_path) -> None:
    outdir = tmp_path / "explicit_branch_truth_10k_mps_plan"
    plan_explicit_branch_truth_10k_mac(
        ExplicitBranchTruth10kMacPlanConfig(outdir=str(outdir), tiers="low")
    )
    text = (outdir / "run_explicit_branch_truth_10k_mps.sh").read_text("utf-8")
    findings = mps_preflight._find_suspicious_bare_tokens(text)
    assert findings == []


def _check(summary: dict, name: str) -> dict:
    for row in summary["checks"]:
        if row["name"] == name:
            return row
    raise AssertionError(f"missing check: {name}")


def _synthetic_plan(
    tmp_path,
    extra_run_line: str = "",
    include_mps_fallback: bool = True,
) -> Path:
    plan_dir = tmp_path / "explicit_branch_truth_10k_mps_plan"
    plan_dir.mkdir()
    run_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "set +u",
        "if [ -f \"$HOME/miniforge3/etc/profile.d/conda.sh\" ]; then source \"$HOME/miniforge3/etc/profile.d/conda.sh\"; fi",
        "conda activate molevo",
        "set -u",
    ]
    if include_mps_fallback:
        run_lines.append("export PYTORCH_ENABLE_MPS_FALLBACK=1")
    run_lines.extend(
        [
            "lock_dir=/tmp/babappa_explicit_branch_truth_10k_mps.lock",
            "cleanup_lock() { rm -rf \"$lock_dir\"; }",
            "if mkdir \"$lock_dir\" 2>/dev/null; then trap cleanup_lock EXIT; else echo \"rm -rf \\\"$lock_dir\\\"\"; exit 1; fi",
            "babappa preflight-explicit-branch-truth-mps-plan --plan-dir explicit_branch_truth_10k_mps_plan --scale 10k --require-babappalign true --require-mps true --conda-env molevo",
            "run_stage_dir .stage_complete_low_branch_dataset branch_site_dataset_explicit_branch_truth_10k_mps_low_streamed babappa build-branch-site-dataset --streaming --max-output-rows 1000",
            "babappa train-branch-site-neural --device mps --batch-size 128 --feature-policy conservative_branch_site",
            "babappa extract-branch-site-labels --truth-mode explicit",
            "babappa align-external --methods identity,mafft,babappalign,muscle",
        ]
    )
    if extra_run_line:
        run_lines.append(extra_run_line)
    run_lines.append("babappa simulate --outdir sim_explicit_branch_truth_10k_mps_low")
    for name in [
        "run_explicit_branch_truth_10k_mps.sh",
        "monitor_explicit_branch_truth_10k_mps.sh",
        "summarize_explicit_branch_truth_10k_mps.sh",
    ]:
        (plan_dir / name).write_text("\n".join(run_lines) + "\n", encoding="utf-8")
    (plan_dir / "validate_explicit_branch_truth_10k_mps.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "babappa validate-align --align-dir align_explicit_branch_truth_10k_mps_low\n"
        "babappa validate-site-map --site-map-dir site_map_explicit_branch_truth_10k_mps_low\n",
        encoding="utf-8",
    )
    (plan_dir / "expected_outputs.json").write_text(json.dumps({"scale": "10k"}), encoding="utf-8")
    return plan_dir
