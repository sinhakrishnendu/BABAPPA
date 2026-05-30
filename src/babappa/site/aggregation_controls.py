"""Null and decoy controls for site-to-gene aggregation."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv
from babappa.site.baseline import _compute_binary_metrics

SITE_AGGREGATION_CONTROLS_VERSION = __version__
CONTROL_NAMES = [
    "shuffled_site_probabilities_within_split",
    "shuffled_gene_labels_within_split",
    "family_permutation",
    "random_uniform_probabilities",
]
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
]


@dataclass(frozen=True)
class SiteAggregationControlConfig:
    """Configuration for aggregation null/decoy controls."""

    predictions_tsv: str
    gene_dataset_dir: str
    outdir: str
    probability_column: str = "prob_positive"
    label_column: str = "y_site"
    n_permutations: int = 50
    seed: int = 42

    def __post_init__(self) -> None:
        if not Path(self.predictions_tsv).exists():
            raise ValueError(f"predictions_tsv does not exist: {self.predictions_tsv}")
        dataset = Path(self.gene_dataset_dir)
        if not dataset.exists():
            raise ValueError(f"gene_dataset_dir does not exist: {dataset}")
        if not (dataset / "splits.tsv").exists():
            raise ValueError(f"gene_dataset_dir is missing splits.tsv: {dataset}")
        if self.n_permutations < 1:
            raise ValueError("n_permutations must be >= 1")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def run_site_aggregation_controls(config: SiteAggregationControlConfig) -> dict:
    """Run null controls for site-to-gene aggregation."""
    rng = np.random.default_rng(config.seed)
    site_rows = read_tsv(Path(config.predictions_tsv))
    if not site_rows:
        raise ValueError("predictions_tsv contains no rows")
    gene_lookup = _gene_lookup(Path(config.gene_dataset_dir))
    base_records = _records(site_rows, config)
    observed = _aggregate_records(base_records, gene_lookup)
    observed_auroc = _auroc(observed)
    control_values: Dict[str, List[float | None]] = {name: [] for name in CONTROL_NAMES}

    for _ in range(config.n_permutations):
        control_values["shuffled_site_probabilities_within_split"].append(
            _auroc(_aggregate_records(_shuffle_probs_within_split(base_records, rng), gene_lookup))
        )
        control_values["shuffled_gene_labels_within_split"].append(
            _auroc(_shuffle_gene_labels_within_split(observed, rng))
        )
        control_values["family_permutation"].append(
            _auroc(_aggregate_records(_permute_family_within_split_method(base_records, rng), gene_lookup))
        )
        control_values["random_uniform_probabilities"].append(
            _auroc(_aggregate_records(_random_uniform_probs(base_records, rng), gene_lookup))
        )

    rows = []
    summaries = {}
    for name, values in control_values.items():
        summary = _summarize_control(values, observed_auroc, config.n_permutations)
        summaries[name] = summary
        rows.append({"control": name, **summary})

    outdir = Path(config.outdir)
    json_path = outdir / "site_aggregation_controls.json"
    tsv_path = outdir / "site_aggregation_controls.tsv"
    markdown_path = outdir / "site_aggregation_controls.md"
    payload = {
        "site_aggregation_controls_version": SITE_AGGREGATION_CONTROLS_VERSION,
        "predictions_tsv": str(Path(config.predictions_tsv)),
        "gene_dataset_dir": str(Path(config.gene_dataset_dir)),
        "n_permutations": config.n_permutations,
        "observed": {
            "n_family_method_rows": len(observed),
            "max_site_probability_auroc": observed_auroc,
        },
        "controls": summaries,
        "generated_files": {
            "json": str(json_path),
            "tsv": str(tsv_path),
            "markdown": str(markdown_path),
        },
        "interpretation": (
            "Observed aggregation must exceed decoy controls before claiming robust "
            "site-to-gene recovery. Perfect observed AUROC is not sufficient without null controls."
        ),
    }
    _write_json(json_path, payload)
    write_tsv(tsv_path, rows, TSV_FIELDNAMES)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "json": str(json_path),
        "tsv": str(tsv_path),
        "markdown": str(markdown_path),
        "observed_auroc": observed_auroc,
    }


def _records(rows: List[dict], config: SiteAggregationControlConfig) -> List[dict]:
    records = []
    for row in rows:
        records.append(
            {
                "family_id": row.get("family_id", ""),
                "method": row.get("method", ""),
                "split": row.get("split", ""),
                "prob": float(row.get(config.probability_column, 0.0)),
            }
        )
    return records


def _gene_lookup(dataset_dir: Path) -> Dict[Tuple[str, str], dict]:
    return {
        (row.get("family_id", ""), row.get("method", "")): row
        for row in read_tsv(dataset_dir / "splits.tsv")
    }


def _aggregate_records(records: List[dict], gene_lookup: Dict[Tuple[str, str], dict]) -> List[dict]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    split_lookup: Dict[Tuple[str, str], str] = {}
    for record in records:
        key = (record["family_id"], record["method"])
        grouped[key].append(float(record["prob"]))
        split_lookup[key] = record["split"]
    output = []
    for key, probs in grouped.items():
        gene = gene_lookup.get(key, {})
        label = gene.get("gene_label", "")
        if str(label) not in {"0", "1", "0.0", "1.0"}:
            continue
        output.append(
            {
                "family_id": key[0],
                "method": key[1],
                "split": gene.get("split", split_lookup.get(key, "")),
                "gene_label": int(float(label)),
                "max_site_probability": float(np.max(probs)),
            }
        )
    return output


def _shuffle_probs_within_split(records: List[dict], rng: np.random.Generator) -> List[dict]:
    copied = [dict(record) for record in records]
    for split in sorted({record["split"] for record in copied}):
        indices = [i for i, record in enumerate(copied) if record["split"] == split]
        probs = np.array([copied[i]["prob"] for i in indices], dtype=np.float64)
        rng.shuffle(probs)
        for idx, prob in zip(indices, probs):
            copied[idx]["prob"] = float(prob)
    return copied


def _shuffle_gene_labels_within_split(rows: List[dict], rng: np.random.Generator) -> List[dict]:
    copied = [dict(row) for row in rows]
    for split in sorted({row["split"] for row in copied}):
        indices = [i for i, row in enumerate(copied) if row["split"] == split]
        labels = np.array([copied[i]["gene_label"] for i in indices], dtype=np.int32)
        rng.shuffle(labels)
        for idx, label in zip(indices, labels):
            copied[idx]["gene_label"] = int(label)
    return copied


def _permute_family_within_split_method(records: List[dict], rng: np.random.Generator) -> List[dict]:
    copied = [dict(record) for record in records]
    groups = sorted({(record["split"], record["method"]) for record in copied})
    for split, method in groups:
        indices = [
            i for i, record in enumerate(copied)
            if record["split"] == split and record["method"] == method
        ]
        families = np.array([copied[i]["family_id"] for i in indices], dtype=object)
        rng.shuffle(families)
        for idx, family_id in zip(indices, families):
            copied[idx]["family_id"] = str(family_id)
    return copied


def _random_uniform_probs(records: List[dict], rng: np.random.Generator) -> List[dict]:
    copied = [dict(record) for record in records]
    probs = rng.random(len(copied))
    for record, prob in zip(copied, probs):
        record["prob"] = float(prob)
    return copied


def _auroc(rows: List[dict]) -> float | None:
    if not rows:
        return None
    y = np.array([row["gene_label"] for row in rows], dtype=np.int32)
    score = np.array([row["max_site_probability"] for row in rows], dtype=np.float64)
    return _compute_binary_metrics(y, score, threshold=0.5).get("auroc")


def _summarize_control(values: List[float | None], observed: float | None, n: int) -> dict:
    numeric = np.array([value for value in values if value is not None], dtype=np.float64)
    if numeric.size == 0:
        return {
            "n_permutations": n,
            "observed_auroc": observed,
            "mean_auroc": None,
            "std_auroc": None,
            "min_auroc": None,
            "max_auroc": None,
            "q05_auroc": None,
            "q95_auroc": None,
            "empirical_p_value": None,
        }
    empirical = None
    if observed is not None:
        empirical = float((1 + int((numeric >= observed).sum())) / (numeric.size + 1))
    return {
        "n_permutations": n,
        "observed_auroc": observed,
        "mean_auroc": float(numeric.mean()),
        "std_auroc": float(numeric.std(ddof=0)),
        "min_auroc": float(numeric.min()),
        "max_auroc": float(numeric.max()),
        "q05_auroc": float(np.quantile(numeric, 0.05)),
        "q95_auroc": float(np.quantile(numeric, 0.95)),
        "empirical_p_value": empirical,
    }


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Site aggregation controls",
        "",
        "## Observed aggregation",
        "",
        f"- Observed max-site AUROC: {payload.get('observed', {}).get('max_site_probability_auroc')}",
        "",
        "## Null/decoy controls",
        "",
    ]
    for name, summary in payload.get("controls", {}).items():
        lines.append(f"- {name}: mean AUROC {summary.get('mean_auroc')}, p={summary.get('empirical_p_value')}")
    lines.extend(
        [
            "",
            "## Empirical p-values",
            "",
            "Empirical p-values are computed as `(1 + null >= observed) / (n + 1)`.",
            "",
            "## Interpretation",
            "",
            payload.get("interpretation", ""),
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
