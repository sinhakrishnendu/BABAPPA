"""Storage-safe variable-length branch-site retraining planner."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union

from babappa.branch.feature_policy import get_branch_feature_policy


TIERS = ["low", "moderate", "high", "extreme"]
DEFAULT_REGIMES = [
    ("short_6taxa", 6, 120, 0.040, 0.03),
    ("compact_8taxa", 8, 240, 0.045, 0.04),
    ("standard_12taxa", 12, 420, 0.050, 0.05),
    ("long_20taxa", 20, 720, 0.055, 0.04),
    ("very_long_32taxa", 32, 960, 0.060, 0.03),
]


@dataclass(frozen=True)
class VariableLength100KRetrainingPlanConfig:
    """Configuration for storage-safe 100K variable-length retraining planning."""

    outdir: str = "variable_length_retraining_plan"
    workspace: str = "branch_site_v2_100k_workspace"
    package_outdir: str = "deployable_model_conservative_branch_site_v2_100k_mps"
    n_families_per_tier: int = 25000
    tiers: Union[str, Sequence[str]] = "low,moderate,high,extreme"
    methods: Union[str, Sequence[str]] = "identity,mafft,babappalign,muscle"
    feature_policy: str = "conservative_branch_site_normalized_v2"
    device: str = "mps"
    threads: int = 18
    batch_size: int = 64
    min_free_gb: int = 250
    negative_downsample_ratio: float = 5.0
    max_output_rows_per_chunk: int = 700000
    max_train_items: int = 300000
    max_eval_items: int = 75000
    n_control_permutations: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_families_per_tier < len(DEFAULT_REGIMES):
            raise ValueError("n_families_per_tier must be at least the number of variable-length regimes")
        if self.threads < 1:
            raise ValueError("threads must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.min_free_gb < 1:
            raise ValueError("min_free_gb must be >= 1")
        if self.max_output_rows_per_chunk < 1:
            raise ValueError("max_output_rows_per_chunk must be >= 1")
        if self.negative_downsample_ratio <= 0:
            raise ValueError("negative_downsample_ratio must be > 0")
        tiers = _parse_csv(self.tiers)
        methods = _parse_csv(self.methods)
        if not tiers:
            raise ValueError("tiers must not be empty")
        unknown_tiers = sorted(set(tiers) - set(TIERS))
        if unknown_tiers:
            raise ValueError("unknown tiers: " + ",".join(unknown_tiers))
        if methods != ["identity", "mafft", "babappalign", "muscle"]:
            raise ValueError("methods must be identity,mafft,babappalign,muscle for deployable retraining")
        policy = get_branch_feature_policy(self.feature_policy)
        if policy.name != "conservative_branch_site_normalized_v2":
            raise ValueError("storage-safe variable-length retraining requires conservative_branch_site_normalized_v2")
        object.__setattr__(self, "tiers", tiers)
        object.__setattr__(self, "methods", methods)
        object.__setattr__(self, "feature_policy", policy.name)
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def plan_variable_length_100k_retraining(config: VariableLength100KRetrainingPlanConfig) -> Dict[str, object]:
    """Write storage-safe retraining scripts without running heavy jobs."""

    outdir = Path(config.outdir)
    regime_manifest = outdir / "variable_length_regime_manifest.tsv"
    cleanup_policy = outdir / "cleanup_policy.tsv"
    expected_outputs = outdir / "expected_outputs.json"
    run_script = outdir / "run_variable_length_100k_retraining.sh"
    monitor_script = outdir / "monitor_variable_length_100k_retraining.sh"
    validate_script = outdir / "validate_variable_length_100k_retraining.sh"
    package_script = outdir / "package_variable_length_deployable.sh"
    markdown = outdir / "variable_length_100k_retraining_plan.md"

    regimes = _regime_rows(config)
    _write_tsv(regime_manifest, regimes, ["tier", "regime_id", "n_families", "n_taxa", "n_codons", "mutation_rate", "selected_site_fraction", "seed"])
    _write_tsv(cleanup_policy, _cleanup_rows(), ["path_class", "default_action", "reason"])
    expected = _expected_outputs(config, regimes)
    expected_outputs.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_script.write_text(_run_script(config, regimes), encoding="utf-8")
    monitor_script.write_text(_monitor_script(config), encoding="utf-8")
    validate_script.write_text(_validate_script(config), encoding="utf-8")
    package_script.write_text(_package_script(config), encoding="utf-8")
    markdown.write_text(_markdown(config, regimes), encoding="utf-8")
    for script in [run_script, monitor_script, validate_script, package_script]:
        os.chmod(script, 0o755)

    return {
        "status": "ok",
        "outdir": str(outdir),
        "run": str(run_script),
        "monitor": str(monitor_script),
        "validate": str(validate_script),
        "package": str(package_script),
        "regime_manifest": str(regime_manifest),
        "cleanup_policy": str(cleanup_policy),
        "expected_outputs": str(expected_outputs),
        "markdown": str(markdown),
        "feature_policy": config.feature_policy,
        "workspace": config.workspace,
        "package_outdir": config.package_outdir,
        "does_not_run_jobs": True,
    }


def _regime_rows(config: VariableLength100KRetrainingPlanConfig) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    base = config.n_families_per_tier // len(DEFAULT_REGIMES)
    remainder = config.n_families_per_tier % len(DEFAULT_REGIMES)
    for tier_index, tier in enumerate(config.tiers):
        for regime_index, (regime_id, n_taxa, n_codons, mutation_rate, site_fraction) in enumerate(DEFAULT_REGIMES):
            n_families = base + (1 if regime_index < remainder else 0)
            rows.append({
                "tier": tier,
                "regime_id": regime_id,
                "n_families": n_families,
                "n_taxa": n_taxa,
                "n_codons": n_codons,
                "mutation_rate": mutation_rate,
                "selected_site_fraction": site_fraction,
                "seed": config.seed + (tier_index * 1000) + regime_index,
            })
    return rows


def _run_script(config: VariableLength100KRetrainingPlanConfig, regimes: List[Dict[str, object]]) -> str:
    methods = ",".join(config.methods)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Long-running local retraining script. Review settings before running.",
        "# Set BABAPPA_RETRAIN_CLEANUP_MODE=delete and BABAPPA_RETRAIN_DELETE_INTERMEDIATES=YES",
        "# to free disk after each completed chunk. Use quarantine or none for safer dry operation.",
        "",
        f"WORKSPACE=\"${{BABAPPA_RETRAIN_WORKSPACE:-{config.workspace}}}\"",
        f"PLAN_DIR=\"{config.outdir}\"",
        f"METHODS=\"{methods}\"",
        f"FEATURE_POLICY=\"{config.feature_policy}\"",
        f"DEVICE=\"${{BABAPPA_DEVICE:-{config.device}}}\"",
        f"THREADS=\"${{BABAPPA_THREADS:-{config.threads}}}\"",
        f"BATCH_SIZE=\"${{BABAPPA_BATCH_SIZE:-{config.batch_size}}}\"",
        f"MIN_FREE_GB=\"${{BABAPPA_MIN_FREE_GB:-{config.min_free_gb}}}\"",
        f"CLEANUP_MODE=\"${{BABAPPA_RETRAIN_CLEANUP_MODE:-quarantine}}\"",
        "QUARANTINE_ROOT=\"${BABAPPA_RETRAIN_QUARANTINE_ROOT:-${WORKSPACE}_quarantine_$(date +%Y%m%d_%H%M%S)}\"",
        "mkdir -p \"$WORKSPACE\" \"$PLAN_DIR/stage_markers\" \"$PLAN_DIR/logs\"",
        "exec > >(tee -a \"$PLAN_DIR/logs/variable_length_retraining_$(date +%Y%m%d_%H%M%S).log\") 2>&1",
        "echo \"BABAPPA variable-length normalized-v2 retraining started\"",
        "echo \"cleanup_mode=$CLEANUP_MODE workspace=$WORKSPACE device=$DEVICE threads=$THREADS\"",
        "",
        "available_gb() { df -Pk . | awk 'NR==2 {printf \"%d\", $4/1024/1024}'; }",
        "require_free_space() { local free; free=$(available_gb); if [ \"$free\" -lt \"$MIN_FREE_GB\" ]; then echo \"free disk ${free}GB is below required ${MIN_FREE_GB}GB\" >&2; exit 3; fi; }",
        "cleanup_path() {",
        "  local path=\"$1\"",
        "  [ -e \"$path\" ] || return 0",
        "  case \"$CLEANUP_MODE\" in",
        "    delete)",
        "      if [ \"${BABAPPA_RETRAIN_DELETE_INTERMEDIATES:-NO}\" != \"YES\" ]; then echo \"delete cleanup requires BABAPPA_RETRAIN_DELETE_INTERMEDIATES=YES\" >&2; exit 4; fi",
        "      rm -rf \"$path\"",
        "      ;;",
        "    quarantine)",
        "      mkdir -p \"$QUARANTINE_ROOT\"",
        "      mkdir -p \"$QUARANTINE_ROOT/$(dirname \"$path\")\"",
        "      mv \"$path\" \"$QUARANTINE_ROOT/$path\"",
        "      ;;",
        "    none)",
        "      echo \"keeping $path\"",
        "      ;;",
        "    *) echo \"unknown cleanup mode: $CLEANUP_MODE\" >&2; exit 5 ;;",
        "  esac",
        "}",
        "run_once() { local marker=\"$1\"; shift; if [ -f \"$marker\" ]; then echo \"skip completed $marker\"; else require_free_space; \"$@\"; touch \"$marker\"; fi; }",
        "require_alignment_methods() {",
        "  python - \"$1\" \"$2\" <<'PY'",
        "import json, sys",
        "from pathlib import Path",
        "align_dir = Path(sys.argv[1])",
        "expected = [m.strip() for m in sys.argv[2].split(',') if m.strip()]",
        "manifest = align_dir / 'alignment_manifest.json'",
        "if not manifest.exists():",
        "    raise SystemExit(f'missing alignment manifest: {manifest}')",
        "data = json.loads(manifest.read_text())",
        "available = set(data.get('methods') or [])",
        "missing = [m for m in expected if m not in available]",
        "if missing:",
        "    warnings = data.get('warnings') or []",
        "    raise SystemExit('alignment methods missing: ' + ','.join(missing) + '; warnings=' + ','.join(map(str, warnings)))",
        "print('alignment methods available: ' + ','.join(expected))",
        "PY",
        "}",
        "",
    ]
    for tier in config.tiers:
        tier_regimes = [row for row in regimes if row["tier"] == tier]
        tier_dataset_dirs = " ".join(f"\"$WORKSPACE/branch_dataset_{tier}_{row['regime_id']}\"" for row in tier_regimes)
        lines.extend([
            f"echo \"=== tier {tier} ===\"",
            f"mkdir -p \"$WORKSPACE/tier_{tier}_datasets\"",
        ])
        for row in tier_regimes:
            prefix = f"{tier}_{row['regime_id']}"
            sim = f"$WORKSPACE/sim_{prefix}"
            align = f"$WORKSPACE/align_{prefix}"
            site_map = f"$WORKSPACE/site_map_{prefix}"
            policy = f"$WORKSPACE/method_policy_{prefix}"
            tensors = f"$WORKSPACE/tensors_{prefix}"
            dataset = f"$WORKSPACE/dataset_{prefix}"
            labels = f"$WORKSPACE/labels_{prefix}"
            branch_dataset = f"$WORKSPACE/branch_dataset_{prefix}"
            marker_prefix = f"$PLAN_DIR/stage_markers/.stage_complete_{prefix}"
            lines.extend([
                f"echo \"--- chunk {prefix}: families={row['n_families']} taxa={row['n_taxa']} codons={row['n_codons']} ---\"",
                f"run_once \"{marker_prefix}_simulate\" babappa simulate --outdir \"{sim}\" --n-families {row['n_families']} --n-taxa {row['n_taxa']} --n-codons {row['n_codons']} --seed {row['seed']} --positive-rate 0.5 --selected-site-fraction {row['selected_site_fraction']} --mutation-rate {row['mutation_rate']} --saturation-tier {tier} --workers \"$THREADS\"",
                f"run_once \"{marker_prefix}_validate_sim\" babappa validate-sim --sim-dir \"{sim}\" --require-branch-truth",
                f"run_once \"{marker_prefix}_align\" babappa align-external --sim-dir \"{sim}\" --outdir \"{align}\" --methods \"$METHODS\" --threads \"$THREADS\"",
                f"run_once \"{marker_prefix}_align_methods\" require_alignment_methods \"{align}\" \"$METHODS\"",
                f"run_once \"{marker_prefix}_site_map\" babappa build-site-map --sim-dir \"{sim}\" --align-dir \"{align}\" --outdir \"{site_map}\" --methods \"$METHODS\" --workers \"$THREADS\"",
                f"run_once \"{marker_prefix}_method_policy\" babappa aligner-method-policy --align-dir \"{align}\" --site-map-dir \"{site_map}\" --outdir \"{policy}\" --max-conflict-fraction 0.03 --max-frame-error-fraction 0.0 --max-method-failure-fraction 0.01",
                f"run_once \"{marker_prefix}_tensors\" babappa build-tensors --sim-dir \"{sim}\" --align-dir \"{align}\" --outdir \"{tensors}\" --methods \"$METHODS\" --workers \"$THREADS\"",
                f"run_once \"{marker_prefix}_index\" babappa index-dataset --tensor-dir \"{tensors}\" --outdir \"{dataset}\" --methods \"$METHODS\" --workers \"$THREADS\"",
                f"run_once \"{marker_prefix}_labels\" babappa extract-branch-site-labels --dataset-dir \"{dataset}\" --site-map-dir \"{site_map}\" --outdir \"{labels}\" --truth-mode explicit --aligned-site-mode mapped --foreground-source truth --streaming-output",
                f"run_once \"{marker_prefix}_branch_dataset\" babappa build-branch-site-dataset --dataset-dir \"{dataset}\" --branch-site-labels \"{labels}/branch_site_oracle_labels.tsv\" --outdir \"{branch_dataset}\" --negative-downsample-ratio {config.negative_downsample_ratio:g} --seed {config.seed} --require-mappable-sites --streaming --max-output-rows {config.max_output_rows_per_chunk}",
                f"run_once \"{marker_prefix}_validate_branch_dataset\" babappa validate-branch-site-dataset --branch-site-dataset-dir \"{branch_dataset}\"",
                f"cleanup_path \"{sim}\"; cleanup_path \"{align}\"; cleanup_path \"{site_map}\"; cleanup_path \"{policy}\"; cleanup_path \"{tensors}\"; cleanup_path \"{dataset}\"; cleanup_path \"{labels}\"",
                "",
            ])
        model = f"$WORKSPACE/branch_site_neural_{tier}"
        calibration = f"$WORKSPACE/branch_site_calibration_{tier}"
        aggregation = f"$WORKSPACE/branch_aggregation_{tier}"
        controls = f"$WORKSPACE/branch_aggregation_controls_{tier}"
        threshold = f"$WORKSPACE/branch_site_threshold_policy_{tier}"
        aggregation_policy = f"$WORKSPACE/branch_aggregation_threshold_policy_{tier}"
        summary = f"$WORKSPACE/branch_site_run_summary_{tier}"
        merged = f"$WORKSPACE/branch_site_dataset_{tier}_merged"
        ds_csv = ",".join([f"$WORKSPACE/branch_dataset_{tier}_{row['regime_id']}" for row in tier_regimes])
        lines.extend([
            f"run_once \"$PLAN_DIR/stage_markers/.stage_complete_{tier}_merge\" babappa merge-branch-site-datasets --dataset-dirs \"{ds_csv}\" --outdir \"{merged}\"",
            f"run_once \"$PLAN_DIR/stage_markers/.stage_complete_{tier}_train\" babappa train-branch-site-neural --branch-site-dataset-dir \"{merged}\" --outdir \"{model}\" --device \"$DEVICE\" --batch-size \"$BATCH_SIZE\" --threads \"$THREADS\" --feature-policy \"$FEATURE_POLICY\" --epochs 10 --learning-rate 0.001 --weight-decay 0.0001 --hidden-dim 64 --dropout 0.1 --positive-class-weight auto --monitor-metric val_auroc --max-train-items {config.max_train_items} --max-val-items {config.max_eval_items} --max-calib-items {config.max_eval_items} --max-test-items {config.max_eval_items}",
            f"run_once \"$PLAN_DIR/stage_markers/.stage_complete_{tier}_validate_train\" babappa validate-branch-site-neural --model-dir \"{model}\"",
            f"run_once \"$PLAN_DIR/stage_markers/.stage_complete_{tier}_calibrate\" babappa calibrate-branch-site-neural --model-dir \"{model}\" --outdir \"{calibration}\"",
            f"run_once \"$PLAN_DIR/stage_markers/.stage_complete_{tier}_validate_calibration\" babappa validate-branch-site-calibration --calibration-dir \"{calibration}\"",
            f"run_once \"$PLAN_DIR/stage_markers/.stage_complete_{tier}_aggregate\" babappa aggregate-branch-sites --predictions \"{model}/branch_site_neural_predictions.tsv\" --outdir \"{aggregation}\"",
            f"run_once \"$PLAN_DIR/stage_markers/.stage_complete_{tier}_controls\" babappa branch-aggregation-controls --predictions \"{model}/branch_site_neural_predictions.tsv\" --outdir \"{controls}\" --n-permutations {config.n_control_permutations} --seed {config.seed} --workers \"$THREADS\"",
            f"run_once \"$PLAN_DIR/stage_markers/.stage_complete_{tier}_threshold\" babappa branch-site-threshold-policy --predictions \"{calibration}/branch_site_calibrated_predictions.tsv\" --outdir \"{threshold}\" --probability-column prob_positive_raw --calibrated-probability-column prob_positive_calibrated",
            f"run_once \"$PLAN_DIR/stage_markers/.stage_complete_{tier}_aggregation_policy\" babappa branch-aggregation-threshold-policy --aggregation-dir \"{aggregation}\" --outdir \"{aggregation_policy}\"",
            f"run_once \"$PLAN_DIR/stage_markers/.stage_complete_{tier}_summary\" babappa summarize-branch-site-run --outdir \"{summary}\" --title \"BABAPPA variable-length normalized-v2 {tier} summary\" --branch-site-dataset-dir \"{merged}\" --branch-site-neural-dir \"{model}\" --branch-site-calibration-dir \"{calibration}\" --branch-aggregation-dir \"{aggregation}\" --branch-aggregation-controls-dir \"{controls}\" --branch-site-threshold-policy-dir \"{threshold}\" --branch-aggregation-threshold-policy-dir \"{aggregation_policy}\"",
            f"cleanup_path \"{merged}\"; for d in {tier_dataset_dirs}; do cleanup_path \"$d\"; done",
            "",
        ])
    lines.extend([
        f"mkdir -p \"$WORKSPACE/truth_audit\"",
        "cat > \"$WORKSPACE/truth_audit/branch_truth_status_audit.tsv\" <<'EOF'",
        "tier\texplicit_branch_site_truth_available\tproxy_from_foreground_taxon",
    ])
    for tier in config.tiers:
        lines.append(f"{tier}\tTrue\tFalse")
    lines.extend([
        "EOF",
        "cat > \"$WORKSPACE/truth_audit/branch_truth_status_audit.json\" <<'EOF'",
        json.dumps({"status": "ok", "explicit_truth_available": True, "proxy_label_tiers": []}, indent=2),
        "EOF",
        "cat > \"$WORKSPACE/variable_length_100k_validation_report.json\" <<'EOF'",
        json.dumps(
            {
                "run_identity": {
                    "run_name": "variable_length_normalized_v2_100k_mps",
                    "feature_policy": config.feature_policy,
                    "truth_mode": "explicit",
                    "methods": list(config.methods),
                },
                "decision": {
                    "status": "PENDING_REVIEW",
                    "reason": "Long-run script generated retained model/calibration artifacts; inspect tier summaries before release.",
                },
                "neural_rows": [],
                "branch_aggregation_rows": [],
                "gene_aggregation_rows": [],
                "controls_rows": [],
            },
            indent=2,
        ),
        "EOF",
        "echo \"Variable-length retraining stages completed. Run package_variable_length_deployable.sh after reviewing validation summaries.\"",
    ])
    return "\n".join(lines) + "\n"


def _monitor_script(config: VariableLength100KRetrainingPlanConfig) -> str:
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"WORKSPACE=\"${{BABAPPA_RETRAIN_WORKSPACE:-{config.workspace}}}\"",
        "echo \"Disk usage:\"",
        "df -h .",
        "du -sh \"$WORKSPACE\" 2>/dev/null || true",
        "echo \"Latest retraining log:\"",
        f"ls -t {config.outdir}/logs/variable_length_retraining_*.log 2>/dev/null | head -1 | xargs tail -80 2>/dev/null || true",
        "echo \"Stage markers:\"",
        f"find {config.outdir}/stage_markers -type f 2>/dev/null | wc -l",
    ]) + "\n"


def _validate_script(config: VariableLength100KRetrainingPlanConfig) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"WORKSPACE=\"${{BABAPPA_RETRAIN_WORKSPACE:-{config.workspace}}}\"",
    ]
    for tier in config.tiers:
        lines.extend([
            f"echo \"=== validate tier {tier} retained artifacts ===\"",
            f"babappa validate-branch-site-neural --model-dir \"$WORKSPACE/branch_site_neural_{tier}\"",
            f"babappa validate-branch-site-calibration --calibration-dir \"$WORKSPACE/branch_site_calibration_{tier}\"",
            f"babappa validate-branch-aggregation --aggregation-dir \"$WORKSPACE/branch_aggregation_{tier}\"",
            f"babappa validate-branch-aggregation-controls --controls-dir \"$WORKSPACE/branch_aggregation_controls_{tier}\"",
            f"babappa validate-branch-site-threshold-policy --policy-dir \"$WORKSPACE/branch_site_threshold_policy_{tier}\"",
            f"babappa validate-branch-aggregation-threshold-policy --policy-dir \"$WORKSPACE/branch_aggregation_threshold_policy_{tier}\"",
            "",
        ])
    return "\n".join(lines) + "\n"


def _package_script(config: VariableLength100KRetrainingPlanConfig) -> str:
    model_dirs = ",".join(f"{config.workspace}/branch_site_neural_{tier}" for tier in config.tiers)
    calibration_dirs = ",".join(f"{config.workspace}/branch_site_calibration_{tier}" for tier in config.tiers)
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Package retained normalized-v2 artifacts after validation.",
        f"babappa package-deployable-model --run-name variable_length_normalized_v2_100k_mps --model-dirs {model_dirs} --calibration-dirs {calibration_dirs} --truth-audit-dir {config.workspace}/truth_audit --validation-report {config.workspace}/variable_length_100k_validation_report.json --feature-policy {config.feature_policy} --truth-mode explicit --methods {','.join(config.methods)} --outdir {config.package_outdir}",
        f"babappa validate-deployable-model-package --package-dir {config.package_outdir}",
    ]) + "\n"


def _markdown(config: VariableLength100KRetrainingPlanConfig, regimes: List[Dict[str, object]]) -> str:
    total = sum(int(row["n_families"]) for row in regimes)
    return "\n".join([
        "# BABAPPA variable-length normalized-v2 100K retraining plan",
        "",
        f"- Target families: `{total}`",
        f"- Feature policy: `{config.feature_policy}`",
        f"- Workspace: `{config.workspace}`",
        f"- Package output: `{config.package_outdir}`",
        f"- Cleanup default: `quarantine`; set `BABAPPA_RETRAIN_CLEANUP_MODE=delete` and `BABAPPA_RETRAIN_DELETE_INTERMEDIATES=YES` to actually free disk after completed chunks.",
        "",
        "## Why this plan exists",
        "",
        "The previous deployable model used raw length/site features that can become far outside its training envelope on real user MSAs. This plan retrains with normalized/log length features and broader simulated gene lengths and taxon counts.",
        "",
        "## Storage policy",
        "",
        "Each chunk is simulated, aligned, mapped, tensorized, indexed, labelled, and reduced to a branch-site feature table. After that feature table validates, raw simulation, alignment, tensor, index, and label intermediates are cleaned before the next chunk.",
        "",
        "## First command",
        "",
        "```bash",
        f"bash {config.outdir}/run_variable_length_100k_retraining.sh",
        "```",
    ]) + "\n"


def _cleanup_rows() -> List[Dict[str, str]]:
    return [
        {"path_class": "simulated_families", "default_action": "cleanup_after_branch_dataset_validates", "reason": "raw simulator output is reproducible from regime manifest and seed"},
        {"path_class": "alignments_site_maps_tensors_indexes", "default_action": "cleanup_after_branch_dataset_validates", "reason": "large intermediates are reproducible and not needed after feature extraction"},
        {"path_class": "per_chunk_branch_site_datasets", "default_action": "cleanup_after_tier_training_validates", "reason": "merged tier dataset and trained model supersede per-chunk feature tables"},
        {"path_class": "merged_tier_branch_site_dataset", "default_action": "cleanup_after_tier_summary_validates", "reason": "retained model/calibration/summary are sufficient for packaging"},
        {"path_class": "model_calibration_threshold_summary", "default_action": "keep", "reason": "required for deployable package and validation"},
    ]


def _expected_outputs(config: VariableLength100KRetrainingPlanConfig, regimes: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "workspace": config.workspace,
        "package_outdir": config.package_outdir,
        "feature_policy": config.feature_policy,
        "n_families_total": sum(int(row["n_families"]) for row in regimes),
        "tiers": list(config.tiers),
        "retained_per_tier": {
            tier: [
                f"{config.workspace}/branch_site_neural_{tier}",
                f"{config.workspace}/branch_site_calibration_{tier}",
                f"{config.workspace}/branch_aggregation_{tier}",
                f"{config.workspace}/branch_aggregation_controls_{tier}",
                f"{config.workspace}/branch_site_threshold_policy_{tier}",
                f"{config.workspace}/branch_aggregation_threshold_policy_{tier}",
                f"{config.workspace}/branch_site_run_summary_{tier}",
            ]
            for tier in config.tiers
        },
    }


def _write_tsv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _parse_csv(value: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = []
        for item in value:
            parts.extend(str(item).split(","))
    return [part.strip() for part in parts if part.strip()]
