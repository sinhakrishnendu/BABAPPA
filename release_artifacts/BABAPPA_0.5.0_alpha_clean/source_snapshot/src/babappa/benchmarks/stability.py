"""Repeated-split and repeated-seed stability benchmark for BABAPPA."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from babappa import __version__
from babappa.datasets import ResplitDatasetConfig, resplit_dataset, write_tsv
from babappa.reports import NeuralDiagnosticsConfig, diagnose_neural_run
from babappa.training import NeuralFullTrainConfig, safe_import_torch, train_neural_model

STABILITY_BENCHMARK_VERSION = __version__
STABILITY_TSV_FIELDNAMES = [
    "seed",
    "preset",
    "split",
    "auroc",
    "accuracy",
    "f1",
    "mcc",
    "precision",
    "recall",
    "specificity",
    "probability_std",
    "separation",
    "warnings",
]
SPLITS = ["train", "val", "calib", "test", "all"]


@dataclass(frozen=True)
class StabilityBenchmarkConfig:
    """Configuration for repeated-seed stability benchmarking."""

    dataset_dir: str
    outdir: str
    seeds: List[int] = None  # type: ignore[assignment]
    presets: List[str] = None  # type: ignore[assignment]
    methods: List[str] = None  # type: ignore[assignment]
    device: str = "cpu"
    epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 0.001
    max_train_items: Optional[int] = 64
    max_val_items: Optional[int] = 32
    max_calib_items: Optional[int] = 16
    max_test_items: Optional[int] = 16
    run_training: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", list(self.seeds or [42, 43, 44]))
        object.__setattr__(
            self,
            "presets",
            list(
                self.presets
                or ["contrastive_v2", "saturation_embed_only", "site_attention_ranked"]
            ),
        )
        object.__setattr__(
            self,
            "methods",
            list(self.methods or ["identity", "codon_dropout"]),
        )
        dataset_path = Path(self.dataset_dir)
        if not dataset_path.exists():
            raise ValueError(f"dataset_dir does not exist: {dataset_path}")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if not self.presets:
            raise ValueError("presets must be non-empty")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def run_stability_benchmark(config: StabilityBenchmarkConfig) -> dict:
    """Run a repeated-split/repeated-seed stability benchmark."""
    outdir = Path(config.outdir)
    resplits_dir = outdir / "resplits"
    models_dir = outdir / "models"
    diagnostics_dir = outdir / "diagnostics"
    for directory in [resplits_dir, models_dir, diagnostics_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    torch, _error = safe_import_torch()
    warnings: List[str] = []
    if config.run_training and torch is None:
        warnings.append("torch_unavailable_training_skipped")

    result_rows: List[dict] = []
    model_runs = []
    for seed in config.seeds:
        resplit_dir = resplits_dir / f"seed_{seed}"
        resplit_summary = resplit_dataset(
            ResplitDatasetConfig(
                dataset_dir=config.dataset_dir,
                outdir=str(resplit_dir),
                seed=seed,
            )
        )
        for preset in config.presets:
            model_dir = models_dir / preset / f"seed_{seed}"
            run_payload = {
                "seed": seed,
                "preset": preset,
                "resplit_dataset": str(resplit_dir),
                "model_dir": str(model_dir),
                "trained": False,
                "metrics_by_split": {},
                "diagnostics": {},
                "warnings": [],
            }
            if config.run_training and torch is not None:
                try:
                    train_neural_model(
                        NeuralFullTrainConfig(
                            dataset_dir=str(resplit_dir),
                            outdir=str(model_dir),
                            seed=seed,
                            device=config.device,
                            methods=config.methods,
                            epochs=config.epochs,
                            batch_size=config.batch_size,
                            learning_rate=config.learning_rate,
                            training_preset=preset,
                            max_train_items=config.max_train_items,
                            max_val_items=config.max_val_items,
                            max_calib_items=config.max_calib_items,
                            max_test_items=config.max_test_items,
                            early_stopping_patience=max(1, min(2, config.epochs)),
                        )
                    )
                    metrics = _load_json(model_dir / "neural_metrics.json")
                    diag_out = diagnostics_dir / preset / f"seed_{seed}"
                    diagnostics = diagnose_neural_run(
                        NeuralDiagnosticsConfig(
                            model_dir=str(model_dir),
                            outdir=str(diag_out),
                            model_name=f"{preset}_seed_{seed}",
                        )
                    )
                    diag_payload = _load_json(Path(diagnostics["json"]))
                    run_payload.update(
                        {
                            "trained": True,
                            "metrics_by_split": metrics.get("metrics_by_split", {}),
                            "diagnostics": diag_payload,
                            "warnings": diagnostics.get("warnings", []),
                        }
                    )
                except Exception as exc:  # pragma: no cover - defensive benchmark logging
                    warning = f"training_failed:{preset}:seed_{seed}:{exc}"
                    warnings.append(warning)
                    run_payload["warnings"].append(warning)
            model_runs.append(run_payload)
            result_rows.extend(_result_rows(seed, preset, run_payload))
        warnings.extend(resplit_summary.get("warnings", []))

    aggregate = _aggregate_summary(result_rows)
    payload = {
        "stability_benchmark_version": STABILITY_BENCHMARK_VERSION,
        "config": {
            "dataset_dir": config.dataset_dir,
            "seeds": config.seeds,
            "presets": config.presets,
            "methods": config.methods,
            "device": config.device,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "max_train_items": config.max_train_items,
            "max_val_items": config.max_val_items,
            "max_calib_items": config.max_calib_items,
            "max_test_items": config.max_test_items,
            "run_training": config.run_training,
        },
        "model_runs": model_runs,
        "aggregate_summary": aggregate,
        "warnings": sorted(set(warnings)),
        "generated_files": {
            "json": str(outdir / "stability_benchmark.json"),
            "tsv": str(outdir / "stability_results.tsv"),
            "markdown": str(outdir / "stability_benchmark.md"),
        },
    }
    _write_json(outdir / "stability_benchmark.json", payload)
    write_tsv(outdir / "stability_results.tsv", result_rows, STABILITY_TSV_FIELDNAMES)
    (outdir / "stability_benchmark.md").write_text(
        _render_markdown(payload),
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "outdir": str(outdir),
        "json": str(outdir / "stability_benchmark.json"),
        "tsv": str(outdir / "stability_results.tsv"),
        "markdown": str(outdir / "stability_benchmark.md"),
        "warnings": payload["warnings"],
        "aggregate_summary": aggregate,
    }


def _result_rows(seed: int, preset: str, run_payload: dict) -> List[dict]:
    rows = []
    diagnostics_by_split = (
        (run_payload.get("diagnostics") or {}).get("probability_summary_by_split") or {}
    )
    warnings = ",".join(str(warning) for warning in run_payload.get("warnings", []))
    for split in SPLITS:
        metrics = (run_payload.get("metrics_by_split") or {}).get(split) or {}
        diag = diagnostics_by_split.get(split) or {}
        rows.append(
            {
                "seed": seed,
                "preset": preset,
                "split": split,
                "auroc": metrics.get("auroc"),
                "accuracy": metrics.get("accuracy"),
                "f1": metrics.get("f1"),
                "mcc": metrics.get("mcc"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "specificity": metrics.get("specificity"),
                "probability_std": diag.get("prob_std"),
                "separation": diag.get("separation"),
                "warnings": warnings,
            }
        )
    return rows


def _aggregate_summary(rows: List[dict]) -> dict:
    aggregate: Dict[str, dict] = {}
    collapse_counts: Dict[str, int] = {}
    for preset in sorted({row["preset"] for row in rows}):
        aggregate[preset] = {}
        collapse_counts[preset] = sum(
            1
            for row in rows
            if row["preset"] == preset and "probability_collapse" in str(row.get("warnings", ""))
        )
        for split in SPLITS:
            values = [
                _safe_float(row.get("auroc"))
                for row in rows
                if row["preset"] == preset and row["split"] == split
            ]
            values = [value for value in values if value is not None]
            if values:
                aggregate[preset][split] = {
                    "mean_auroc": float(statistics.mean(values)),
                    "std_auroc": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
                }
            else:
                aggregate[preset][split] = {"mean_auroc": None, "std_auroc": None}
    return {
        "auroc_by_preset_split": aggregate,
        "probability_collapse_count_by_preset": collapse_counts,
        "best_preset_by_mean_val_auroc": _best_preset(aggregate, "val"),
        "best_preset_by_mean_test_auroc": _best_preset(aggregate, "test"),
        "instability_warnings": _instability_warnings(aggregate),
    }


def _best_preset(aggregate: Dict[str, dict], split: str) -> Optional[str]:
    candidates = [
        (preset, splits[split].get("mean_auroc"))
        for preset, splits in aggregate.items()
        if split in splits and splits[split].get("mean_auroc") is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])[0]


def _instability_warnings(aggregate: Dict[str, dict]) -> List[str]:
    warnings = []
    for preset, splits in aggregate.items():
        for split in ["val", "test"]:
            std = (splits.get(split) or {}).get("std_auroc")
            if std is not None and std > 0.05:
                warnings.append(f"{preset}:{split}_auroc_std_gt_0.05")
    return warnings


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Stability benchmark",
        "",
        "## Configuration",
        "",
        f"- Dataset directory: {payload['config']['dataset_dir']}",
        f"- Seeds: {payload['config']['seeds']}",
        f"- Presets: {payload['config']['presets']}",
        f"- Run training: {payload['config']['run_training']}",
        "",
        "## Results by seed and preset",
        "",
        f"- Model runs: {len(payload['model_runs'])}",
        "",
        "## Aggregate summary",
        "",
        json.dumps(payload["aggregate_summary"], indent=2, sort_keys=True),
        "",
        "## Warnings",
        "",
    ]
    warnings = payload.get("warnings") or []
    lines.extend([f"- {warning}" for warning in warnings] or ["- none"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Repeated-seed stability is required before 10k scaling or branch-site modeling.",
            "- High validation/test AUROC standard deviation indicates split instability.",
            "- Probability-collapse counts indicate models whose scores remain compressed across seeds.",
            "",
        ]
    )
    return "\n".join(lines)


def _safe_float(value: object) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
