"""Null and decoy controls for branch aggregation."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np

from babappa import __version__
from babappa.branch.summary import _normalize_output_suffix, _parse_tiers, _tier_prefix
from babappa.datasets.index import read_tsv, write_tsv
from babappa.site.baseline import _compute_binary_metrics

BRANCH_AGGREGATION_CONTROLS_VERSION = __version__
CONTROL_NAMES = [
    "shuffle_branch_labels_within_split",
    "shuffle_site_probabilities_within_branch",
    "shuffle_branch_assignment_within_family",
    "random_uniform_probabilities",
    "family_permutation",
    "within_family_branch_label_shuffle",
    "within_family_site_label_shuffle",
    "branch_score_permutation_within_family",
    "family_label_preserving_random_scores",
    "degree_prevalence_matched_null",
]
CONTROL_METADATA = {
    "shuffle_branch_labels_within_split": {
        "control_interpretation": "Global split-level branch-label shuffle.",
        "expected_behavior": "Should approach random AUROC if split leakage is absent.",
        "whether_control_is_destructive_enough": "yes",
    },
    "shuffle_site_probabilities_within_branch": {
        "control_interpretation": "Permutes site probabilities only inside each branch.",
        "expected_behavior": "May remain high when branch-level max scores are unchanged.",
        "whether_control_is_destructive_enough": "no",
    },
    "shuffle_branch_assignment_within_family": {
        "control_interpretation": "Reassigns site records to branches within family and method.",
        "expected_behavior": "Should degrade if branch identity carries the signal.",
        "whether_control_is_destructive_enough": "partial",
    },
    "random_uniform_probabilities": {
        "control_interpretation": "Replaces site probabilities with independent uniform random scores.",
        "expected_behavior": "Should degrade, but family/prevalence geometry can still aggregate above random.",
        "whether_control_is_destructive_enough": "partial",
    },
    "family_permutation": {
        "control_interpretation": "Permutes family identifiers within split before branch aggregation.",
        "expected_behavior": "Tests whether family-level prevalence alone explains aggregation.",
        "whether_control_is_destructive_enough": "partial",
    },
    "within_family_branch_label_shuffle": {
        "control_interpretation": "Shuffles branch-site labels across branches within each family/method/split.",
        "expected_behavior": "Should degrade when branch-specific selected-site truth drives scores.",
        "whether_control_is_destructive_enough": "yes",
    },
    "within_family_site_label_shuffle": {
        "control_interpretation": "Shuffles site labels within each family/method/branch.",
        "expected_behavior": "Preserves branch prevalence and tests site-position specificity, not branch prevalence.",
        "whether_control_is_destructive_enough": "partial",
    },
    "branch_score_permutation_within_family": {
        "control_interpretation": "Permutes aggregated branch scores across branches within each family/method/split.",
        "expected_behavior": "Should degrade if score-to-branch assignment matters.",
        "whether_control_is_destructive_enough": "yes",
    },
    "family_label_preserving_random_scores": {
        "control_interpretation": "Generates branch scores from family-positive-prevalence-matched random distributions.",
        "expected_behavior": "Estimates how much family prevalence alone can explain aggregation.",
        "whether_control_is_destructive_enough": "partial",
    },
    "degree_prevalence_matched_null": {
        "control_interpretation": "Randomizes positive branch labels within family/method/split while preserving positive counts.",
        "expected_behavior": "Should degrade while preserving per-family branch prevalence and degree.",
        "whether_control_is_destructive_enough": "yes",
    },
}
TSV_FIELDNAMES = [
    "control",
    "n_permutations",
    "observed_auroc",
    "mean_auroc",
    "std_auroc",
    "min_auroc",
    "max_auroc",
    "q05_auroc",
    "q95_auroc",
    "empirical_p_value",
    "control_interpretation",
    "expected_behavior",
    "whether_control_is_destructive_enough",
]
_WORKER_ROWS: List[dict] | None = None
_WORKER_OBSERVED: List[dict] | None = None
_WORKER_DATA: "_ControlData | None" = None


@dataclass(frozen=True)
class _ControlData:
    prob: np.ndarray
    label: np.ndarray
    site_family_id: np.ndarray
    site_method_id: np.ndarray
    site_branch_name_id: np.ndarray
    site_split_id: np.ndarray
    site_branch_id: np.ndarray
    branch_score: np.ndarray
    branch_label: np.ndarray
    branch_split_id: np.ndarray
    branch_order: np.ndarray
    branch_starts: np.ndarray
    site_fm_order: np.ndarray
    site_fm_starts: np.ndarray
    site_fms_order: np.ndarray
    site_fms_starts: np.ndarray
    site_split_order: np.ndarray
    site_split_starts: np.ndarray
    branch_split_order: np.ndarray
    branch_split_starts: np.ndarray
    branch_fms_order: np.ndarray
    branch_fms_starts: np.ndarray


@dataclass(frozen=True)
class BranchAggregationControlConfig:
    predictions_tsv: str
    outdir: str
    probability_column: str = "prob_positive"
    label_column: str = "y_branch_site"
    n_permutations: int = 50
    seed: int = 42
    workers: int = 1

    def __post_init__(self) -> None:
        if not Path(self.predictions_tsv).exists():
            raise ValueError(f"predictions_tsv does not exist: {self.predictions_tsv}")
        if self.n_permutations < 1:
            raise ValueError("n_permutations must be >= 1")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BranchAggregationControlsRerunPlanConfig:
    """Configuration for a plan-only rerun of branch aggregation controls."""

    run_name: str
    tiers: Union[str, Sequence[str]]
    outdir: str
    output_suffix: str = "_streamed"
    n_permutations: int = 100
    seed: int = 42
    workers: int = 1

    def __post_init__(self) -> None:
        if self.n_permutations < 1:
            raise ValueError("n_permutations must be >= 1")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        tiers = _parse_tiers(self.tiers)
        if not tiers:
            raise ValueError("tiers must not be empty")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def run_branch_aggregation_controls(config: BranchAggregationControlConfig) -> dict:
    """Run branch null/decoy controls."""
    rng = np.random.default_rng(config.seed)
    rows = _records(read_tsv(Path(config.predictions_tsv)), config)
    if not rows:
        raise ValueError("predictions_tsv contains no rows")
    control_data = _prepare_control_data(rows)
    observed_auroc = _auroc_arrays(control_data.branch_label, control_data.branch_score)
    seeds = [int(seed) for seed in rng.integers(0, np.iinfo(np.int64).max, size=config.n_permutations)]
    values, effective_workers = _run_control_permutations(control_data, seeds, config.workers)
    summaries = {}
    tsv_rows = []
    for name, control_values in values.items():
        summary = _summarize_control(control_values, observed_auroc, config.n_permutations)
        summary.update(CONTROL_METADATA.get(name, {}))
        summaries[name] = summary
        tsv_rows.append({"control": name, **summary})
    outdir = Path(config.outdir)
    json_path = outdir / "branch_aggregation_controls.json"
    tsv_path = outdir / "branch_aggregation_controls.tsv"
    markdown_path = outdir / "branch_aggregation_controls.md"
    payload = {
        "branch_aggregation_controls_version": BRANCH_AGGREGATION_CONTROLS_VERSION,
        "predictions_tsv": str(Path(config.predictions_tsv)),
        "n_permutations": config.n_permutations,
        "workers": effective_workers,
        "requested_workers": config.workers,
        "effective_workers": effective_workers,
        "runtime_engine": "precomputed_numpy_groups",
        "observed": {"n_branch_rows": int(control_data.branch_score.size), "branch_auroc": observed_auroc},
        "controls": summaries,
        "generated_files": {"json": str(json_path), "tsv": str(tsv_path), "markdown": str(markdown_path)},
        "interpretation": "Branch aggregation must exceed branch/site decoy controls before stronger claims.",
    }
    _write_json(json_path, payload)
    write_tsv(tsv_path, tsv_rows, TSV_FIELDNAMES)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {"status": "ok", "outdir": str(outdir), "json": str(json_path), "tsv": str(tsv_path), "markdown": str(markdown_path), "observed_auroc": observed_auroc, "workers": effective_workers, "requested_workers": config.workers}


def plan_rerun_branch_aggregation_controls(config: BranchAggregationControlsRerunPlanConfig) -> dict:
    """Write a script to rerun only branch aggregation controls."""

    outdir = Path(config.outdir)
    tiers = _parse_tiers(config.tiers)
    suffix = _normalize_output_suffix(config.output_suffix)
    commands = []
    expected_outputs = {}
    for tier in tiers:
        prefix = _tier_prefix(config.run_name, tier, output_suffix=suffix)
        predictions = f"branch_site_neural_{prefix}/branch_site_neural_predictions.tsv"
        controls_outdir = f"branch_aggregation_controls_rerun_{prefix}"
        expected_outputs[tier] = {
            "predictions": predictions,
            "controls_outdir": controls_outdir,
        }
        commands.extend(
            [
                f"test -f {predictions}",
                " ".join(
                    [
                        "babappa branch-aggregation-controls",
                        f"--predictions {predictions}",
                        f"--outdir {controls_outdir}",
                        f"--n-permutations {config.n_permutations}",
                        f"--seed {config.seed}",
                        f"--workers {config.workers}",
                    ]
                ),
                f"babappa validate-branch-aggregation-controls --controls-dir {controls_outdir}",
                "",
            ]
        )

    run_path = outdir / "run_branch_aggregation_controls_rerun.sh"
    expected_path = outdir / "expected_outputs.json"
    markdown_path = outdir / "branch_aggregation_controls_rerun_plan.md"
    run_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                "echo 'BABAPPA branch aggregation controls rerun started: '\"$(date)\"",
                "echo 'This script reruns only branch aggregation controls on existing predictions.'",
                "",
                *commands,
                "echo 'BABAPPA branch aggregation controls rerun completed: '\"$(date)\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    run_path.chmod(0o755)
    payload = {
        "branch_aggregation_controls_version": BRANCH_AGGREGATION_CONTROLS_VERSION,
        "plan_only": True,
        "does_not_execute_jobs": True,
        "run_name": config.run_name,
        "tiers": tiers,
        "output_suffix": suffix,
        "n_permutations": config.n_permutations,
        "workers": config.workers,
        "seed": config.seed,
        "controls_included": CONTROL_NAMES,
        "expected_outputs": expected_outputs,
    }
    _write_json(expected_path, payload)
    markdown_path.write_text(_render_rerun_plan_markdown(payload, run_path), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "run": str(run_path),
        "expected_outputs": str(expected_path),
        "markdown": str(markdown_path),
        "controls_included": CONTROL_NAMES,
        "does_not_run_jobs": True,
    }


def validate_branch_aggregation_controls_dir(controls_dir: str | Path) -> dict:
    path = Path(controls_dir)
    failures: List[str] = []
    warnings: List[str] = []
    rows = _read_tsv(path / "branch_aggregation_controls.tsv", failures)
    _load_json(path / "branch_aggregation_controls.json", failures)
    markdown = path / "branch_aggregation_controls.md"
    if not markdown.exists():
        failures.append(f"missing_file:{markdown}")
    elif not markdown.read_text(encoding="utf-8").strip():
        failures.append("empty_markdown")
    if not rows:
        failures.append("no_control_rows")
    return {"status": "fail" if failures else "ok", "n_rows": len(rows), "n_fail": len(failures), "n_warning": len(warnings), "failures": failures, "warnings": warnings}


def _run_control_permutations(
    data: _ControlData,
    seeds: List[int],
    workers: int,
) -> tuple[Dict[str, List[float | None]], int]:
    values: Dict[str, List[float | None]] = {name: [] for name in CONTROL_NAMES}
    if workers <= 1 or len(seeds) <= 1:
        for seed in seeds:
            _append_control_values(values, _run_control_iteration_fast(data, seed))
        return values, 1

    n_workers = min(workers, len(seeds), os.cpu_count() or workers)
    try:
        chunks = _chunk_list(seeds, n_workers)
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_control_worker,
            initargs=(data,),
        ) as executor:
            futures = [executor.submit(_run_control_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                chunk_values = future.result()
                for name in CONTROL_NAMES:
                    values[name].extend(chunk_values[name])
        return values, n_workers
    except (OSError, PermissionError, NotImplementedError):
        values = {name: [] for name in CONTROL_NAMES}
        for seed in seeds:
            _append_control_values(values, _run_control_iteration_fast(data, seed))
        return values, 1


def _init_control_worker(data: _ControlData) -> None:
    global _WORKER_DATA
    _WORKER_DATA = data


def _run_control_chunk(seeds: List[int]) -> Dict[str, List[float | None]]:
    if _WORKER_DATA is None:
        raise RuntimeError("branch aggregation control worker was not initialized")
    values: Dict[str, List[float | None]] = {name: [] for name in CONTROL_NAMES}
    for seed in seeds:
        _append_control_values(
            values,
            _run_control_iteration_fast(_WORKER_DATA, seed),
        )
    return values


def _run_control_iteration_fast(
    data: _ControlData,
    seed: int,
) -> Dict[str, float | None]:
    rng = np.random.default_rng(seed)
    observed = _auroc_arrays(data.branch_label, data.branch_score)
    return {
        "shuffle_branch_labels_within_split": _control_shuffle_branch_labels_within_split(data, rng),
        "shuffle_site_probabilities_within_branch": observed,
        "shuffle_branch_assignment_within_family": _control_shuffle_branch_assignment_within_family(data, rng),
        "random_uniform_probabilities": _control_random_uniform_probabilities(data, rng),
        "family_permutation": _control_family_permutation(data, rng),
        "within_family_branch_label_shuffle": _control_within_family_branch_label_shuffle(data, rng),
        "within_family_site_label_shuffle": observed,
        "branch_score_permutation_within_family": _control_branch_score_permutation_within_family(data, rng),
        "family_label_preserving_random_scores": _control_family_label_preserving_random_scores(data, rng),
        "degree_prevalence_matched_null": _control_degree_prevalence_matched_null(data, rng),
    }


def _run_control_iteration(
    rows: List[dict],
    observed: List[dict],
    seed: int,
) -> Dict[str, float | None]:
    rng = np.random.default_rng(seed)
    return {
        "shuffle_branch_labels_within_split": _auroc(_shuffle_branch_labels(observed, rng)),
        "shuffle_site_probabilities_within_branch": _auroc(_aggregate_branch_records(_shuffle_probs_within_branch(rows, rng))),
        "shuffle_branch_assignment_within_family": _auroc(_aggregate_branch_records(_shuffle_branch_assignment(rows, rng))),
        "random_uniform_probabilities": _auroc(_aggregate_branch_records(_random_uniform(rows, rng))),
        "family_permutation": _auroc(_aggregate_branch_records(_permute_family(rows, rng))),
        "within_family_branch_label_shuffle": _auroc(_aggregate_branch_records(_shuffle_site_labels_across_branches_within_family(rows, rng))),
        "within_family_site_label_shuffle": _auroc(_aggregate_branch_records(_shuffle_site_labels_within_branch(rows, rng))),
        "branch_score_permutation_within_family": _auroc(_permute_branch_scores_within_family(observed, rng)),
        "family_label_preserving_random_scores": _auroc(_family_label_preserving_random_scores(observed, rng)),
        "degree_prevalence_matched_null": _auroc(_degree_prevalence_matched_null(observed, rng)),
    }


def _append_control_values(
    values: Dict[str, List[float | None]],
    iteration: Dict[str, float | None],
) -> None:
    for name in CONTROL_NAMES:
        values[name].append(iteration[name])


def _chunk_list(values: List[int], n_chunks: int) -> List[List[int]]:
    return [values[index::n_chunks] for index in range(n_chunks) if values[index::n_chunks]]


def _prepare_control_data(rows: List[dict]) -> _ControlData:
    family_values = [row["family_id"] for row in rows]
    method_values = [row["method"] for row in rows]
    branch_values = [row["branch_id"] for row in rows]
    split_values = [row["split"] for row in rows]
    site_family_id = _factorize_to_array(family_values)
    site_method_id = _factorize_to_array(method_values)
    site_branch_name_id = _factorize_to_array(branch_values)
    site_split_id = _factorize_to_array(split_values)
    site_branch_id = _factorize_to_array(zip(family_values, method_values, branch_values))
    site_fm_id = _factorize_to_array(zip(family_values, method_values))
    site_fms_id = _factorize_to_array(zip(family_values, method_values, split_values))
    prob = np.array([row["prob"] for row in rows], dtype=np.float64)
    label = np.array([row["label"] for row in rows], dtype=np.int8)

    n_branch = int(site_branch_id.max()) + 1
    branch_order, branch_starts = _group_sort(site_branch_id)
    branch_ids_sorted = site_branch_id[branch_order][branch_starts]
    branch_score = np.empty(n_branch, dtype=np.float64)
    branch_label = np.zeros(n_branch, dtype=np.int8)
    branch_score[branch_ids_sorted] = np.maximum.reduceat(prob[branch_order], branch_starts)
    branch_label[branch_ids_sorted] = np.maximum.reduceat(label[branch_order], branch_starts).astype(np.int8)

    branch_split_id = np.zeros(n_branch, dtype=np.int32)
    branch_fms_id = np.zeros(n_branch, dtype=np.int32)
    seen = np.zeros(n_branch, dtype=bool)
    for row_index, branch_id in enumerate(site_branch_id):
        if seen[branch_id]:
            continue
        seen[branch_id] = True
        branch_split_id[branch_id] = site_split_id[row_index]
        branch_fms_id[branch_id] = site_fms_id[row_index]

    site_fm_order, site_fm_starts = _group_sort(site_fm_id)
    site_fms_order, site_fms_starts = _group_sort(site_fms_id)
    site_split_order, site_split_starts = _group_sort(site_split_id)
    branch_split_order, branch_split_starts = _group_sort(branch_split_id)
    branch_fms_order, branch_fms_starts = _group_sort(branch_fms_id)

    return _ControlData(
        prob=prob,
        label=label,
        site_family_id=site_family_id,
        site_method_id=site_method_id,
        site_branch_name_id=site_branch_name_id,
        site_split_id=site_split_id,
        site_branch_id=site_branch_id,
        branch_score=branch_score,
        branch_label=branch_label,
        branch_split_id=branch_split_id,
        branch_order=branch_order,
        branch_starts=branch_starts,
        site_fm_order=site_fm_order,
        site_fm_starts=site_fm_starts,
        site_fms_order=site_fms_order,
        site_fms_starts=site_fms_starts,
        site_split_order=site_split_order,
        site_split_starts=site_split_starts,
        branch_split_order=branch_split_order,
        branch_split_starts=branch_split_starts,
        branch_fms_order=branch_fms_order,
        branch_fms_starts=branch_fms_starts,
    )


def _factorize_to_array(values) -> np.ndarray:
    mapping = {}
    ids = []
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
        ids.append(mapping[value])
    return np.array(ids, dtype=np.int32)


def _group_sort(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if ids.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    order = np.argsort(ids, kind="mergesort")
    sorted_ids = ids[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1].astype(np.int64)
    return order, starts


def _iter_groups(order: np.ndarray, starts: np.ndarray):
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(order)
        yield order[start:end]


def _reduce_original_branch_max(data: _ControlData, values: np.ndarray) -> np.ndarray:
    reduced = np.empty(data.branch_score.shape[0], dtype=values.dtype)
    branch_ids_sorted = data.site_branch_id[data.branch_order][data.branch_starts]
    reduced[branch_ids_sorted] = np.maximum.reduceat(values[data.branch_order], data.branch_starts)
    return reduced


def _aggregate_by_dynamic_branch_ids(
    data: _ControlData,
    branch_ids: np.ndarray,
    prob: np.ndarray,
    label: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order, starts = _group_sort(branch_ids)
    ids_sorted = branch_ids[order][starts]
    scores = np.empty(data.branch_score.shape[0], dtype=np.float64)
    labels = np.zeros(data.branch_label.shape[0], dtype=np.int8)
    scores[ids_sorted] = np.maximum.reduceat(prob[order], starts)
    labels[ids_sorted] = np.maximum.reduceat(label[order], starts).astype(np.int8)
    return labels, scores


def _control_shuffle_branch_labels_within_split(data: _ControlData, rng: np.random.Generator) -> float | None:
    labels = data.branch_label.copy()
    for indices in _iter_groups(data.branch_split_order, data.branch_split_starts):
        shuffled = labels[indices].copy()
        rng.shuffle(shuffled)
        labels[indices] = shuffled
    return _auroc_arrays(labels, data.branch_score)


def _control_shuffle_branch_assignment_within_family(data: _ControlData, rng: np.random.Generator) -> float | None:
    branch_ids = data.site_branch_id.copy()
    for indices in _iter_groups(data.site_fm_order, data.site_fm_starts):
        shuffled = branch_ids[indices].copy()
        rng.shuffle(shuffled)
        branch_ids[indices] = shuffled
    labels, scores = _aggregate_by_dynamic_branch_ids(data, branch_ids, data.prob, data.label)
    return _auroc_arrays(labels, scores)


def _control_random_uniform_probabilities(data: _ControlData, rng: np.random.Generator) -> float | None:
    scores = _reduce_original_branch_max(data, rng.random(data.prob.shape[0]))
    return _auroc_arrays(data.branch_label, scores)


def _control_family_permutation(data: _ControlData, rng: np.random.Generator) -> float | None:
    family_ids = data.site_family_id.copy()
    for indices in _iter_groups(data.site_split_order, data.site_split_starts):
        shuffled = family_ids[indices].copy()
        rng.shuffle(shuffled)
        family_ids[indices] = shuffled
    return _auroc_grouped_site_records(
        data.prob,
        data.label,
        family_ids,
        data.site_method_id,
        data.site_branch_name_id,
    )


def _control_within_family_branch_label_shuffle(data: _ControlData, rng: np.random.Generator) -> float | None:
    labels = data.label.copy()
    for indices in _iter_groups(data.site_fms_order, data.site_fms_starts):
        shuffled = labels[indices].copy()
        rng.shuffle(shuffled)
        labels[indices] = shuffled
    branch_labels = _reduce_original_branch_max(data, labels).astype(np.int8)
    return _auroc_arrays(branch_labels, data.branch_score)


def _control_branch_score_permutation_within_family(data: _ControlData, rng: np.random.Generator) -> float | None:
    scores = data.branch_score.copy()
    for indices in _iter_groups(data.branch_fms_order, data.branch_fms_starts):
        shuffled = scores[indices].copy()
        rng.shuffle(shuffled)
        scores[indices] = shuffled
    return _auroc_arrays(data.branch_label, scores)


def _control_family_label_preserving_random_scores(data: _ControlData, rng: np.random.Generator) -> float | None:
    scores = np.empty_like(data.branch_score)
    for indices in _iter_groups(data.branch_fms_order, data.branch_fms_starts):
        labels = data.branch_label[indices].astype(np.float64)
        prevalence = float(labels.mean()) if labels.size else 0.0
        concentration = 8.0
        alpha = max(0.5, prevalence * concentration + 0.5)
        beta = max(0.5, (1.0 - prevalence) * concentration + 0.5)
        scores[indices] = rng.beta(alpha, beta, size=len(indices))
    return _auroc_arrays(data.branch_label, scores)


def _control_degree_prevalence_matched_null(data: _ControlData, rng: np.random.Generator) -> float | None:
    labels = data.branch_label.copy()
    for indices in _iter_groups(data.branch_fms_order, data.branch_fms_starts):
        shuffled = labels[indices].copy()
        rng.shuffle(shuffled)
        labels[indices] = shuffled
    return _auroc_arrays(labels, data.branch_score)


def _auroc_grouped_site_records(
    prob: np.ndarray,
    label: np.ndarray,
    family_id: np.ndarray,
    method_id: np.ndarray,
    branch_name_id: np.ndarray,
) -> float | None:
    order = np.lexsort((branch_name_id, method_id, family_id))
    if order.size == 0:
        return None
    sorted_family = family_id[order]
    sorted_method = method_id[order]
    sorted_branch = branch_name_id[order]
    changes = (
        (np.diff(sorted_family) != 0)
        | (np.diff(sorted_method) != 0)
        | (np.diff(sorted_branch) != 0)
    )
    starts = np.r_[0, np.flatnonzero(changes) + 1].astype(np.int64)
    scores = np.maximum.reduceat(prob[order], starts)
    labels = np.maximum.reduceat(label[order], starts).astype(np.int8)
    return _auroc_arrays(labels, scores)


def _auroc_arrays(y: np.ndarray, score: np.ndarray) -> float | None:
    if y.size == 0:
        return None
    y = y.astype(np.int8, copy=False)
    score = score.astype(np.float64, copy=False)
    positive = y == 1
    n_pos = int(positive.sum())
    n_total = int(y.size)
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_score)) + 1]
    ranks_sorted = np.empty(n_total, dtype=np.float64)
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else n_total
        ranks_sorted[start:end] = (start + 1 + end) / 2.0
    ranks = np.empty(n_total, dtype=np.float64)
    ranks[order] = ranks_sorted
    rank_sum_pos = float(ranks[positive].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _records(rows: List[dict], config: BranchAggregationControlConfig) -> List[dict]:
    output = []
    for row in rows:
        output.append(
            {
                "family_id": row.get("family_id", ""),
                "method": row.get("method", ""),
                "split": row.get("split", ""),
                "branch_id": row.get("branch_id", ""),
                "prob": float(row.get(config.probability_column, 0.0)),
                "label": int(float(row.get(config.label_column, 0))),
            }
        )
    return output


def _aggregate_branch_records(records: List[dict]) -> List[dict]:
    grouped: Dict[tuple, List[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["family_id"], record["method"], record["branch_id"])].append(record)
    output = []
    for key, group in grouped.items():
        probs = np.array([row["prob"] for row in group], dtype=np.float64)
        output.append({"family_id": key[0], "method": key[1], "branch_id": key[2], "split": group[0]["split"], "branch_label": int(any(row["label"] == 1 for row in group)), "max_branch_site_probability": float(probs.max()) if probs.size else 0.0})
    return output


def _shuffle_branch_labels(rows: List[dict], rng: np.random.Generator) -> List[dict]:
    copied = [dict(row) for row in rows]
    for split in sorted({row["split"] for row in copied}):
        indices = [i for i, row in enumerate(copied) if row["split"] == split]
        labels = np.array([copied[i]["branch_label"] for i in indices], dtype=np.int32)
        rng.shuffle(labels)
        for idx, label in zip(indices, labels):
            copied[idx]["branch_label"] = int(label)
    return copied


def _shuffle_probs_within_branch(records: List[dict], rng: np.random.Generator) -> List[dict]:
    copied = [dict(record) for record in records]
    groups = sorted({(row["family_id"], row["method"], row["branch_id"]) for row in copied})
    for key in groups:
        indices = [i for i, row in enumerate(copied) if (row["family_id"], row["method"], row["branch_id"]) == key]
        probs = np.array([copied[i]["prob"] for i in indices], dtype=np.float64)
        rng.shuffle(probs)
        for idx, prob in zip(indices, probs):
            copied[idx]["prob"] = float(prob)
    return copied


def _shuffle_branch_assignment(records: List[dict], rng: np.random.Generator) -> List[dict]:
    copied = [dict(record) for record in records]
    groups = sorted({(row["family_id"], row["method"]) for row in copied})
    for key in groups:
        indices = [i for i, row in enumerate(copied) if (row["family_id"], row["method"]) == key]
        branches = np.array([copied[i]["branch_id"] for i in indices], dtype=object)
        rng.shuffle(branches)
        for idx, branch in zip(indices, branches):
            copied[idx]["branch_id"] = str(branch)
    return copied


def _random_uniform(records: List[dict], rng: np.random.Generator) -> List[dict]:
    copied = [dict(record) for record in records]
    probs = rng.random(len(copied))
    for record, prob in zip(copied, probs):
        record["prob"] = float(prob)
    return copied


def _permute_family(records: List[dict], rng: np.random.Generator) -> List[dict]:
    copied = [dict(record) for record in records]
    for split in sorted({row["split"] for row in copied}):
        indices = [i for i, row in enumerate(copied) if row["split"] == split]
        families = np.array([copied[i]["family_id"] for i in indices], dtype=object)
        rng.shuffle(families)
        for idx, family in zip(indices, families):
            copied[idx]["family_id"] = str(family)
    return copied


def _shuffle_site_labels_across_branches_within_family(records: List[dict], rng: np.random.Generator) -> List[dict]:
    copied = [dict(record) for record in records]
    groups = sorted({(row["family_id"], row["method"], row["split"]) for row in copied})
    for key in groups:
        indices = [i for i, row in enumerate(copied) if (row["family_id"], row["method"], row["split"]) == key]
        labels = np.array([copied[i]["label"] for i in indices], dtype=np.int32)
        rng.shuffle(labels)
        for idx, label in zip(indices, labels):
            copied[idx]["label"] = int(label)
    return copied


def _shuffle_site_labels_within_branch(records: List[dict], rng: np.random.Generator) -> List[dict]:
    copied = [dict(record) for record in records]
    groups = sorted({(row["family_id"], row["method"], row["split"], row["branch_id"]) for row in copied})
    for key in groups:
        indices = [i for i, row in enumerate(copied) if (row["family_id"], row["method"], row["split"], row["branch_id"]) == key]
        labels = np.array([copied[i]["label"] for i in indices], dtype=np.int32)
        rng.shuffle(labels)
        for idx, label in zip(indices, labels):
            copied[idx]["label"] = int(label)
    return copied


def _permute_branch_scores_within_family(rows: List[dict], rng: np.random.Generator) -> List[dict]:
    copied = [dict(row) for row in rows]
    groups = sorted({(row["family_id"], row["method"], row["split"]) for row in copied})
    for key in groups:
        indices = [i for i, row in enumerate(copied) if (row["family_id"], row["method"], row["split"]) == key]
        scores = np.array([copied[i]["max_branch_site_probability"] for i in indices], dtype=np.float64)
        rng.shuffle(scores)
        for idx, score in zip(indices, scores):
            copied[idx]["max_branch_site_probability"] = float(score)
    return copied


def _family_label_preserving_random_scores(rows: List[dict], rng: np.random.Generator) -> List[dict]:
    copied = [dict(row) for row in rows]
    groups = sorted({(row["family_id"], row["method"], row["split"]) for row in copied})
    for key in groups:
        indices = [i for i, row in enumerate(copied) if (row["family_id"], row["method"], row["split"]) == key]
        labels = np.array([copied[i]["branch_label"] for i in indices], dtype=np.float64)
        prevalence = float(labels.mean()) if labels.size else 0.0
        concentration = 8.0
        alpha = max(0.5, prevalence * concentration + 0.5)
        beta = max(0.5, (1.0 - prevalence) * concentration + 0.5)
        scores = rng.beta(alpha, beta, size=len(indices))
        for idx, score in zip(indices, scores):
            copied[idx]["max_branch_site_probability"] = float(score)
    return copied


def _degree_prevalence_matched_null(rows: List[dict], rng: np.random.Generator) -> List[dict]:
    copied = [dict(row) for row in rows]
    groups = sorted({(row["family_id"], row["method"], row["split"]) for row in copied})
    for key in groups:
        indices = [i for i, row in enumerate(copied) if (row["family_id"], row["method"], row["split"]) == key]
        labels = np.array([copied[i]["branch_label"] for i in indices], dtype=np.int32)
        rng.shuffle(labels)
        for idx, label in zip(indices, labels):
            copied[idx]["branch_label"] = int(label)
    return copied


def _auroc(rows: List[dict]) -> float | None:
    if not rows:
        return None
    y = np.array([row["branch_label"] for row in rows], dtype=np.int32)
    score = np.array([row["max_branch_site_probability"] for row in rows], dtype=np.float64)
    return _compute_binary_metrics(y, score, threshold=0.5).get("auroc")


def _summarize_control(values: List[float | None], observed: float | None, n: int) -> dict:
    numeric = np.array([value for value in values if value is not None], dtype=np.float64)
    if numeric.size == 0:
        return {"n_permutations": n, "observed_auroc": observed, "mean_auroc": None, "std_auroc": None, "min_auroc": None, "max_auroc": None, "q05_auroc": None, "q95_auroc": None, "empirical_p_value": None}
    empirical = None if observed is None else float((1 + int((numeric >= observed).sum())) / (numeric.size + 1))
    return {"n_permutations": n, "observed_auroc": observed, "mean_auroc": float(numeric.mean()), "std_auroc": float(numeric.std(ddof=0)), "min_auroc": float(numeric.min()), "max_auroc": float(numeric.max()), "q05_auroc": float(np.quantile(numeric, 0.05)), "q95_auroc": float(np.quantile(numeric, 0.95)), "empirical_p_value": empirical}


def _read_tsv(path: Path, failures: List[str]) -> List[dict]:
    if not path.exists():
        failures.append(f"missing_file:{path}")
        return []
    return read_tsv(path)


def _load_json(path: Path, failures: List[str]) -> dict:
    if not path.exists():
        failures.append(f"missing_file:{path}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"could_not_parse_json:{path}:{exc}")
        return {}


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Branch aggregation controls",
        "",
        f"- Observed branch AUROC: {payload.get('observed', {}).get('branch_auroc')}",
        f"- Permutations: {payload.get('n_permutations')}",
        "",
        "Controls are decoys for branch-conditioned aggregation.",
        "",
        "| Control | Mean AUROC | Destructive enough | Expected behavior |",
        "| --- | ---: | --- | --- |",
    ]
    for name, summary in sorted((payload.get("controls") or {}).items()):
        lines.append(
            f"| {name} | {summary.get('mean_auroc')} | "
            f"{summary.get('whether_control_is_destructive_enough', '')} | "
            f"{summary.get('expected_behavior', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_rerun_plan_markdown(payload: dict, run_path: Path) -> str:
    lines = [
        "# Branch aggregation controls rerun plan",
        "",
        "This is a plan-only artifact. It does not execute controls automatically.",
        "",
        f"- Run script: `{run_path}`",
        f"- Permutations: `{payload.get('n_permutations')}`",
        f"- Seed: `{payload.get('seed')}`",
        "",
        "## Controls included",
        "",
    ]
    for control in payload.get("controls_included", []):
        lines.append(f"- `{control}`")
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
