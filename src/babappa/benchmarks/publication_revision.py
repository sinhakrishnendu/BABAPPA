"""Publication-revision benchmark planning utilities.

These helpers generate lightweight, reviewable benchmark plans for the
empirical and reproducibility gaps that remain after the current BABAPPA
known-truth validation runs.  They deliberately do not execute analyses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from babappa import __version__


@dataclass(frozen=True)
class PublicationRevisionPlanConfig:
    outdir: str
    retained_validation_families: int = 10_000
    null_replicates: int = 1_000
    threads: int = 8
    device: str = "auto"


def plan_publication_revision_benchmarks(config: PublicationRevisionPlanConfig) -> dict[str, Any]:
    """Write benchmark plans addressing the current manuscript-revision gaps."""
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scripts_dir = outdir / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    _write_tsv(outdir / "known_positive_control_panel_template.tsv", _known_positive_rows())
    _write_tsv(outdir / "empirical_transfer_panel_template.tsv", _transfer_rows())
    _write_tsv(outdir / "sensitivity_analysis_grid.tsv", _sensitivity_rows())
    _write_tsv(outdir / "revision_response_matrix.tsv", _revision_matrix_rows())

    retained_plan = _retained_validation_plan(config)
    _write_json(outdir / "retained_validation_plan.json", retained_plan)
    (outdir / "retained_validation_plan.md").write_text(
        _render_retained_validation_plan(retained_plan),
        encoding="utf-8",
    )
    (outdir / "publication_revision_plan.md").write_text(
        _render_publication_revision_plan(config),
        encoding="utf-8",
    )
    _write_json(outdir / "publication_revision_plan.json", _plan_payload(config))

    scripts = {
        "run_known_positive_controls.sh": _known_positive_script(config),
        "run_empirical_transfer_panel.sh": _transfer_script(config),
        "run_sensitivity_analysis.sh": _sensitivity_script(config),
        "run_retained_validation_profile.sh": _retained_validation_script(config),
        "summarize_revision_benchmarks.sh": _summary_script(),
    }
    for name, text in scripts.items():
        path = scripts_dir / name
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)

    return {
        "status": "planned",
        "outdir": str(outdir),
        "known_positive_template": str(outdir / "known_positive_control_panel_template.tsv"),
        "transfer_template": str(outdir / "empirical_transfer_panel_template.tsv"),
        "sensitivity_grid": str(outdir / "sensitivity_analysis_grid.tsv"),
        "retained_validation_families": config.retained_validation_families,
        "scripts": len(scripts),
    }


def _known_positive_rows() -> list[dict[str, str]]:
    return [
        {
            "panel_id": "known_positive_hiv_env",
            "control_type": "known_positive",
            "evidence_level": "literature_supported_experimental",
            "gene_family": "HIV_env_or_equivalent_viral_envelope",
            "organism_group": "virus",
            "msa_path": "TODO",
            "tree_path": "TODO",
            "foreground": "TODO",
            "expected_branch_or_clade": "TODO",
            "expected_sites": "TODO",
            "literature_reference": "TODO",
            "notes": "Use only if curated orthologous/homologous codon alignment and branch hypothesis are defensible.",
        },
        {
            "panel_id": "known_positive_influenza_ha",
            "control_type": "known_positive",
            "evidence_level": "literature_supported_experimental",
            "gene_family": "influenza_HA_or_equivalent_antigenic_gene",
            "organism_group": "virus",
            "msa_path": "TODO",
            "tree_path": "TODO",
            "foreground": "TODO",
            "expected_branch_or_clade": "TODO",
            "expected_sites": "TODO",
            "literature_reference": "TODO",
            "notes": "Prefer a compact, curated subtype/clade panel with published selected-site support.",
        },
        {
            "panel_id": "known_positive_plant_r_gene",
            "control_type": "known_positive",
            "evidence_level": "literature_supported_functional_or_comparative",
            "gene_family": "plant_NLR_or_R_gene",
            "organism_group": "plants",
            "msa_path": "TODO",
            "tree_path": "TODO",
            "foreground": "TODO",
            "expected_branch_or_clade": "TODO",
            "expected_sites": "TODO",
            "literature_reference": "TODO",
            "notes": "Use as a positive control only with careful paralogy filtering.",
        },
        {
            "panel_id": "known_negative_housekeeping_control",
            "control_type": "known_negative",
            "evidence_level": "conserved_control",
            "gene_family": "housekeeping_control",
            "organism_group": "matched_to_positive_panel",
            "msa_path": "TODO",
            "tree_path": "TODO",
            "foreground": "leaves",
            "expected_branch_or_clade": "none",
            "expected_sites": "none",
            "literature_reference": "TODO",
            "notes": "Matched negative control for empirical false-positive probing.",
        },
    ]


def _transfer_rows() -> list[dict[str, str]]:
    return [
        {
            "panel_id": "transfer_known_positive_01",
            "family_type": "known_positive",
            "source": "literature_curated",
            "msa_path": "TODO",
            "tree_path": "TODO",
            "foreground": "TODO",
            "expected_behavior": "recover branch/site support with BABAPPA-native calibration",
            "comparison": "codeml_hyphy_literature",
            "notes": "Primary simulator-to-real transfer probe.",
        },
        {
            "panel_id": "transfer_known_negative_01",
            "family_type": "known_negative",
            "source": "housekeeping_or_conserved_control",
            "msa_path": "TODO",
            "tree_path": "TODO",
            "foreground": "leaves",
            "expected_behavior": "no calibrated BABAPPA support",
            "comparison": "codeml_hyphy_optional",
            "notes": "Empirical false-positive control.",
        },
        {
            "panel_id": "transfer_ood_stress_01",
            "family_type": "ood_stress",
            "source": "high_divergence_or_alignment_difficult",
            "msa_path": "TODO",
            "tree_path": "TODO",
            "foreground": "leaves",
            "expected_behavior": "abstain_or_diagnostic_only",
            "comparison": "applicability_gate",
            "notes": "Tests whether OOD behavior transfers to empirical data.",
        },
    ]


def _sensitivity_rows() -> list[dict[str, str]]:
    return [
        {
            "axis": "score_threshold",
            "baseline": "frozen_best_mcc",
            "values": "high_precision;fdr_0.20_candidate;default_abstention",
            "primary_metrics": "precision,recall,FDR,MCC,OOD_FPR",
            "purpose": "show threshold-dependent operating points",
        },
        {
            "axis": "tier_boundary",
            "baseline": "training_envelope_pdistance",
            "values": "minus_10_percent;baseline;plus_10_percent",
            "primary_metrics": "tier_stability,AUROC,AUPRC,OOD_FPR",
            "purpose": "justify p-distance tier selection robustness",
        },
        {
            "axis": "temperature",
            "baseline": "packaged_calibration",
            "values": "0.75x;1.0x;1.25x",
            "primary_metrics": "calibration_error,FDR,power",
            "purpose": "test calibration sensitivity",
        },
        {
            "axis": "architecture_width",
            "baseline": "two_hidden_layers_width_64",
            "values": "32;64;128",
            "primary_metrics": "AUROC,AUPRC,FDR,OOD_FPR",
            "purpose": "justify neural capacity choice",
        },
        {
            "axis": "training_seed",
            "baseline": "packaged_model_seed",
            "values": "3_to_5_independent_seeds",
            "primary_metrics": "mean_sd_AUROC,mean_sd_FDR,threshold_stability",
            "purpose": "quantify stochastic training robustness",
        },
    ]


def _revision_matrix_rows() -> list[dict[str, str]]:
    return [
        {
            "review_concern": "empirical_demonstration_weak",
            "response_artifact": "known_positive_control_panel_template.tsv",
            "required_result": "at_least_one_literature_supported_positive_control_with_native_null_support",
            "current_status": "planned_not_claimed",
        },
        {
            "review_concern": "simulator_to_real_transfer_unvalidated",
            "response_artifact": "empirical_transfer_panel_template.tsv",
            "required_result": "stratified transfer panel with positives, negatives, and OOD stress families",
            "current_status": "planned_not_claimed",
        },
        {
            "review_concern": "conditional_100k_pass_due_to_pruned_intermediates",
            "response_artifact": "retained_validation_plan.json",
            "required_result": "smaller fully retained validation profile with archive manifest and checksums",
            "current_status": "planned_not_claimed",
        },
        {
            "review_concern": "hyperparameter_justification",
            "response_artifact": "sensitivity_analysis_grid.tsv",
            "required_result": "sensitivity table over thresholds, tier boundaries, temperature, width, and seeds",
            "current_status": "planned_not_claimed",
        },
        {
            "review_concern": "defensive_repetition",
            "response_artifact": "manuscript_revision_text",
            "required_result": "concise limitations plus positive control and transfer-test roadmap",
            "current_status": "implemented_in_text",
        },
    ]


def _retained_validation_plan(config: PublicationRevisionPlanConfig) -> dict[str, Any]:
    return {
        "plan_version": __version__,
        "benchmark_name": "BABAPPA-RETAINED-VALIDATION-v1",
        "n_families": config.retained_validation_families,
        "purpose": "address the conditional-pass reproducibility concern with a smaller fully retained known-truth validation profile",
        "retain": [
            "simulation_manifest",
            "family_truth_files",
            "input_cds",
            "tree_files",
            "feature_tables",
            "scores",
            "applicability_reports",
            "threshold_policy_outputs",
            "evaluation_tables",
            "checksums",
        ],
        "exclude_from_git": [
            "raw aligner scratch",
            "large logs",
            "temporary tensor caches",
        ],
        "archive_expected": True,
        "primary_metrics": [
            "AUROC",
            "AUPRC",
            "FDR",
            "MCC",
            "OOD false-call rate",
            "non-OOD positive recall",
        ],
    }


def _plan_payload(config: PublicationRevisionPlanConfig) -> dict[str, Any]:
    return {
        "planner_version": __version__,
        "retained_validation_families": config.retained_validation_families,
        "null_replicates": config.null_replicates,
        "threads": config.threads,
        "device": config.device,
        "heavy_execution": "not_performed_by_planner",
        "claim_boundary": "planned artifacts do not constitute completed empirical validation until filled with real results",
    }


def _render_publication_revision_plan(config: PublicationRevisionPlanConfig) -> str:
    return f"""# Publication Revision Benchmark Plan

This plan addresses four reviewer-facing gaps without inventing results:

1. Known biological positive controls.
2. Simulator-to-real transfer testing.
3. Sensitivity analysis for thresholds, tier boundaries, temperature, architecture width, and seeds.
4. A smaller fully retained validation profile to repair the reproducibility concern from pruned 100K intermediates.

The generated templates are planning artifacts. Manuscript claims should be updated only after real data are added and the corresponding analyses complete.

Recommended retained validation size: {config.retained_validation_families} families.
Recommended empirical native-null replicates: {config.null_replicates}.
Device: `{config.device}`.
Threads: {config.threads}.
"""


def _render_retained_validation_plan(payload: dict[str, Any]) -> str:
    retained = "\n".join(f"- `{item}`" for item in payload["retain"])
    metrics = "\n".join(f"- {item}" for item in payload["primary_metrics"])
    return f"""# Fully Retained Validation Plan

Benchmark: `{payload['benchmark_name']}`

Families: {payload['n_families']}

Purpose: {payload['purpose']}.

Retained artifacts:
{retained}

Primary metrics:
{metrics}

This profile is designed to be small enough to archive completely while large enough to audit the main known-truth operating claims.
"""


def _known_positive_script(config: PublicationRevisionPlanConfig) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
echo "Long-run benchmark script: review inputs before running."
PANEL="${{1:-known_positive_control_panel_template.tsv}}"
OUTDIR="${{2:-known_positive_control_results}}"
INCLUDE_OPTIONAL="${{INCLUDE_OPTIONAL:-0}}"
DEVICE="${{BABAPPA_DEVICE:-{config.device}}}"
NULL_REPS="${{BABAPPA_NULL_REPLICATES:-{config.null_replicates}}}"
mkdir -p "$OUTDIR"
python - <<'PY' "$PANEL" "$OUTDIR" "$DEVICE" "$NULL_REPS" "$INCLUDE_OPTIONAL"
import csv, subprocess, sys
panel, outdir, device, null_reps, include_optional = sys.argv[1:]
with open(panel, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle, delimiter="\\t"):
        control_type = row.get("control_type", "")
        if row.get("msa_path", "TODO") == "TODO" or row.get("tree_path", "TODO") == "TODO":
            print(f"skip {{row.get('panel_id')}}: missing MSA/tree")
            continue
        if "not_runnable" in control_type or "pending" in control_type:
            print(f"skip {{row.get('panel_id')}}: {{control_type}}")
            continue
        if "optional" in control_type and include_optional not in {{"1", "true", "yes", "y"}}:
            print(f"skip {{row.get('panel_id')}}: optional; set INCLUDE_OPTIONAL=1 to run")
            continue
        cmd = [
            "babappa", "predict-branch-sites",
            "--msa", row["msa_path"],
            "--tree", row["tree_path"],
            "--foreground", row.get("foreground", "leaves"),
            "--outdir", f"{{outdir}}/{{row['panel_id']}}",
            "--device", device,
            "--null-replicates", str(null_reps),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
PY
"""


def _transfer_script(config: PublicationRevisionPlanConfig) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
echo "Long-run transfer-test script: review inputs before running."
PANEL="${{1:-empirical_transfer_panel_template.tsv}}"
OUTDIR="${{2:-empirical_transfer_results}}"
mkdir -p "$OUTDIR"
python - <<'PY' "$PANEL" "$OUTDIR" "{config.device}" "{config.null_replicates}"
import csv, subprocess, sys
panel, outdir, device, null_reps = sys.argv[1:]
with open(panel, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle, delimiter="\\t"):
        if row.get("msa_path", "TODO") == "TODO" or row.get("tree_path", "TODO") == "TODO":
            print(f"skip {{row.get('panel_id')}}: missing MSA/tree")
            continue
        cmd = [
            "babappa", "predict-branch-sites",
            "--msa", row["msa_path"],
            "--tree", row["tree_path"],
            "--foreground", row.get("foreground", "leaves"),
            "--outdir", f"{{outdir}}/{{row['panel_id']}}",
            "--device", device,
            "--null-replicates", str(null_reps),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
PY
"""


def _sensitivity_script(config: PublicationRevisionPlanConfig) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
echo "Sensitivity-analysis plan script: this writes a checklist and does not retrain models."
GRID="${{1:-sensitivity_analysis_grid.tsv}}"
OUTDIR="${{2:-sensitivity_analysis_results}}"
mkdir -p "$OUTDIR"
cp "$GRID" "$OUTDIR/sensitivity_analysis_grid.tsv"
cat > "$OUTDIR/sensitivity_analysis_readme.md" <<'EOF'
# Sensitivity Analysis Checklist

Run the axes in `sensitivity_analysis_grid.tsv` only when the required model-training or calibration resources are available. Record AUROC, AUPRC, FDR, MCC, OOD false-call rate, and threshold stability for each axis.
EOF
"""


def _retained_validation_script(config: PublicationRevisionPlanConfig) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
echo "Retained validation profile: review disk budget before running."
OUTDIR="${{1:-benchmark_runs/retained_validation_profile}}"
ROOT_DIR="${{BABAPPA_REPO_ROOT:-$(pwd)}}"
mkdir -p "$OUTDIR"
CONFIG="$OUTDIR/config_retained_validation.yaml"
cat > "$CONFIG" <<EOF
profile: retained_validation
n_families: {config.retained_validation_families}
seed: 20260605
n_taxa: 10
n_codons: 240
outdir: $OUTDIR
model_package: deployable_model_conservative_branch_site_100k_mps
device: {config.device}
babappa_null_replicates: 100
jobs:
  babappa: {config.threads}
  absrel: {config.threads}
  prepare: {config.threads}
EOF
cat > "$OUTDIR/retained_validation_run_plan.md" <<'EOF'
# Retained Validation Profile

This profile should retain all compact inputs, truth files, features, scores, summaries, manifests, and checksums. It is intended to repair the reproducibility limitation from the pruned 100K intermediates.
EOF
python "$ROOT_DIR/scripts/benchmarks/known_truth_absrel/01_simulate_known_truth_dataset.py" --config "$CONFIG"
python "$ROOT_DIR/scripts/benchmarks/known_truth_absrel/02_run_babappa_on_dataset.py" --config "$CONFIG" --continue-on-failure --jobs "{config.threads}"
python "$ROOT_DIR/scripts/benchmarks/known_truth_absrel/03_prepare_absrel_inputs.py" --config "$CONFIG" --jobs "{config.threads}"
python "$ROOT_DIR/scripts/benchmarks/known_truth_absrel/05_compare_against_truth.py" --config "$CONFIG"
python "$ROOT_DIR/scripts/benchmarks/known_truth_absrel/06_make_benchmark_report.py" --config "$CONFIG"
"""


def _summary_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
echo "Revision benchmark summary collection"
find . -maxdepth 3 -type f \\( -name '*summary*.md' -o -name '*summary*.tsv' -o -name '*results*.tsv' \\) | sort
"""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    header = list(rows[0].keys())
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row.get(column, "")) for column in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
