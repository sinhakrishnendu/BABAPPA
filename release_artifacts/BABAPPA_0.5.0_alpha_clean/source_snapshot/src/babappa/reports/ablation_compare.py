"""Compare BABAPPA neural ablation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from babappa import __version__
from babappa.datasets.index import write_tsv

ABLATION_COMPARISON_VERSION = __version__
SPLITS = ["train", "val", "calib", "test", "all"]
METRICS = ["accuracy", "auroc", "f1", "mcc", "precision", "recall", "specificity"]
TSV_FIELDNAMES = [
    "model_name",
    "architecture",
    "training_preset",
    "group_weighting",
    "sampler",
    "best_epoch",
    "probability_std_all",
    "separation_all",
    "diagnostic_warnings",
    "split",
    *METRICS,
]


@dataclass(frozen=True)
class AblationCompareConfig:
    """Configuration for neural ablation comparison."""

    outdir: str
    model_dirs: List[str]
    names: Optional[List[str]] = None
    stratified_eval_dirs: Optional[List[str]] = None
    threshold_policy_dirs: Optional[List[str]] = None
    neural_diagnostics_dirs: Optional[List[str]] = None
    title: str = "BABAPPA neural ablation comparison"

    def __post_init__(self) -> None:
        if len(self.model_dirs) < 2:
            raise ValueError("at least two model_dirs must be supplied")
        if self.names is not None:
            if len(self.names) != len(self.model_dirs):
                raise ValueError("names must match model_dirs length")
            if len(set(self.names)) != len(self.names):
                raise ValueError("names must be unique")
        for field_name in [
            "stratified_eval_dirs",
            "threshold_policy_dirs",
            "neural_diagnostics_dirs",
        ]:
            values = getattr(self, field_name)
            if values is not None and len(values) != len(self.model_dirs):
                raise ValueError(f"{field_name} must match model_dirs length")
        for model_dir in self.model_dirs:
            model_path = Path(model_dir)
            if not model_path.exists():
                raise ValueError(f"model_dir does not exist: {model_path}")
            if not (model_path / "neural_metrics.json").exists():
                raise ValueError(f"model_dir missing neural_metrics.json: {model_path}")
            if not (model_path / "neural_model_meta.json").exists():
                raise ValueError(f"model_dir missing neural_model_meta.json: {model_path}")
        _validate_optional_dirs(self.stratified_eval_dirs, "stratified_eval_dirs")
        _validate_optional_dirs(self.threshold_policy_dirs, "threshold_policy_dirs")
        _validate_optional_dirs(self.neural_diagnostics_dirs, "neural_diagnostics_dirs")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def compare_neural_ablations(config: AblationCompareConfig) -> dict:
    """Compare neural ablation models using existing metrics and metadata."""
    outdir = Path(config.outdir)
    model_names = config.names or [
        f"model_{index + 1}" for index in range(len(config.model_dirs))
    ]
    warnings: List[str] = []
    models = []
    rows = []
    for index, (name, model_dir) in enumerate(zip(model_names, config.model_dirs)):
        model_path = Path(model_dir)
        meta = _load_json(model_path / "neural_model_meta.json")
        metrics = _load_json(model_path / "neural_metrics.json")
        stratified_eval = _optional_json(
            config.stratified_eval_dirs, index, "stratified_eval.json", warnings
        )
        threshold_policy = _optional_json(
            config.threshold_policy_dirs, index, "threshold_profiles.json", warnings
        )
        neural_diagnostics = _optional_json(
            config.neural_diagnostics_dirs, index, "neural_diagnostics.json", warnings
        )
        model_payload = {
            "model_name": name,
            "model_dir": str(model_path),
            "architecture": meta.get("architecture"),
            "training_preset": meta.get("training_preset"),
            "group_weighting": meta.get("group_weighting"),
            "sampler": meta.get("sampler"),
            "positive_class_weight": meta.get("positive_class_weight"),
            "best_epoch": meta.get("best_epoch"),
            "epochs_completed": meta.get("epochs_completed"),
            "stopped_early": meta.get("stopped_early"),
            "metrics_by_split": metrics.get("metrics_by_split", {}),
            "stratified_key_findings": (stratified_eval or {}).get("key_findings"),
            "threshold_profiles": _threshold_profile_summary(threshold_policy),
            "diagnostics": _diagnostic_summary(neural_diagnostics),
        }
        models.append(model_payload)
        rows.extend(_metric_rows(model_payload))

    recommendation = _recommendation(models)
    payload = {
        "ablation_comparison_version": ABLATION_COMPARISON_VERSION,
        "title": config.title,
        "inputs": {
            "model_dirs": list(config.model_dirs),
            "names": model_names,
            "stratified_eval_dirs": list(config.stratified_eval_dirs or []),
            "threshold_policy_dirs": list(config.threshold_policy_dirs or []),
            "neural_diagnostics_dirs": list(config.neural_diagnostics_dirs or []),
        },
        "models": models,
        "recommendation": recommendation,
        "warnings": sorted(set(warnings)),
        "generated_files": {
            "json": str(outdir / "ablation_comparison.json"),
            "tsv": str(outdir / "ablation_comparison.tsv"),
            "markdown": str(outdir / "ablation_comparison.md"),
        },
    }
    _write_json(outdir / "ablation_comparison.json", payload)
    write_tsv(outdir / "ablation_comparison.tsv", rows, TSV_FIELDNAMES)
    (outdir / "ablation_comparison.md").write_text(
        _render_markdown(payload),
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "outdir": str(outdir),
        "json": str(outdir / "ablation_comparison.json"),
        "tsv": str(outdir / "ablation_comparison.tsv"),
        "markdown": str(outdir / "ablation_comparison.md"),
        "recommendation": recommendation,
        "warnings": payload["warnings"],
    }


def _metric_rows(model: dict) -> List[dict]:
    rows = []
    metrics_by_split = model.get("metrics_by_split") or {}
    for split in SPLITS:
        metrics = metrics_by_split.get(split, {})
        rows.append(
            {
                "model_name": model["model_name"],
                "architecture": model.get("architecture"),
                "training_preset": model.get("training_preset"),
                "group_weighting": model.get("group_weighting"),
                "sampler": model.get("sampler"),
                "best_epoch": model.get("best_epoch"),
                "probability_std_all": (model.get("diagnostics") or {}).get(
                    "probability_std_all"
                ),
                "separation_all": (model.get("diagnostics") or {}).get(
                    "separation_all"
                ),
                "diagnostic_warnings": ",".join(
                    (model.get("diagnostics") or {}).get("warnings", [])
                ),
                "split": split,
                **{metric: metrics.get(metric) for metric in METRICS},
            }
        )
    return rows


def _recommendation(models: List[dict]) -> dict:
    selected = _best_model(models, "val", "auroc")
    basis = "validation AUROC"
    if selected is None:
        selected = _best_model(models, "test", "auroc")
        basis = "test AUROC"
    if selected is None:
        selected = _best_model(models, "all", "auroc")
        basis = "all-split AUROC"
    text = "No AUROC-bearing split was available; inspect metrics manually."
    if selected is not None:
        text = f"Prefer {selected['model_name']} by {basis}."
    contrastive = [
        model for model in models if model.get("architecture") == "contrastive"
    ]
    saturation_aware = [
        model for model in models if model.get("architecture") == "saturation_aware"
    ]
    contrastive_best = _best_model(contrastive, "val", "auroc") or _best_model(
        contrastive, "all", "auroc"
    )
    saturation_best = _best_model(saturation_aware, "val", "auroc") or _best_model(
        saturation_aware, "all", "auroc"
    )
    if contrastive_best and saturation_best:
        c_value = _metric_value(contrastive_best, "val", "auroc")
        s_value = _metric_value(saturation_best, "val", "auroc")
        if c_value is None or s_value is None:
            c_value = _metric_value(contrastive_best, "all", "auroc")
            s_value = _metric_value(saturation_best, "all", "auroc")
        if c_value is not None and s_value is not None and c_value > s_value:
            text += (
                " Saturation-aware architecture underperformed the best contrastive "
                "variant; revert to contrastive and treat saturation as post-hoc risk "
                "until a better architecture is available."
            )
    if selected and selected.get("sampler") == "saturation_balanced":
        text += " Confirm that saturation-balanced sampling helped before keeping it."
    if selected and selected.get("group_weighting") == "saturation_inverse_frequency":
        text += " Confirm that group weighting helped before keeping it."
    return {
        "best_model": selected.get("model_name") if selected else None,
        "basis": basis if selected else None,
        "text": text,
    }


def _best_model(models: List[dict], split: str, metric: str) -> Optional[dict]:
    candidates = [
        model
        for model in models
        if _metric_value(model, split, metric) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda model: _metric_value(model, split, metric))


def _metric_value(model: dict, split: str, metric: str) -> Optional[float]:
    value = ((model.get("metrics_by_split") or {}).get(split) or {}).get(metric)
    if value is None:
        return None
    return float(value)


def _threshold_profile_summary(payload: Optional[dict]) -> dict:
    if not isinstance(payload, dict):
        return {}
    profiles = payload.get("profiles") or {}
    return {
        name: {
            "selected_threshold": profile.get("selected_threshold"),
            "warning": profile.get("warning"),
        }
        for name, profile in profiles.items()
        if isinstance(profile, dict)
    }


def _diagnostic_summary(payload: Optional[dict]) -> dict:
    if not isinstance(payload, dict):
        return {}
    all_summary = (payload.get("probability_summary_by_split") or {}).get("all") or {}
    warnings = [str(warning) for warning in payload.get("warnings", [])]
    return {
        "probability_std_all": all_summary.get("prob_std"),
        "separation_all": all_summary.get("separation"),
        "fraction_ge_0_5_all": all_summary.get("fraction_ge_0_5"),
        "warnings": warnings,
        "probability_collapse": any(
            "probability_collapse" in warning for warning in warnings
        ),
    }


def _render_markdown(payload: dict) -> str:
    lines = [f"# {payload['title']}", "", "## Models compared", ""]
    for model in payload["models"]:
        lines.append(
            "- {name}: architecture={architecture}, preset={preset}, "
            "group_weighting={group_weighting}, sampler={sampler}".format(
                name=model["model_name"],
                architecture=model.get("architecture"),
                preset=model.get("training_preset"),
                group_weighting=model.get("group_weighting"),
                sampler=model.get("sampler"),
            )
        )
    lines.extend(
        [
            "",
            "## Split-level metrics",
            "",
            "| Model | Split | AUROC | F1 | MCC | Precision | Recall |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in payload["models"]:
        for split in SPLITS:
            metrics = (model.get("metrics_by_split") or {}).get(split, {})
            lines.append(
                "| {model} | {split} | {auroc} | {f1} | {mcc} | {precision} | {recall} |".format(
                    model=model["model_name"],
                    split=split,
                    auroc=_format_float(metrics.get("auroc")),
                    f1=_format_float(metrics.get("f1")),
                    mcc=_format_float(metrics.get("mcc")),
                    precision=_format_float(metrics.get("precision")),
                    recall=_format_float(metrics.get("recall")),
                )
            )
    lines.extend(["", "## Stratified findings", ""])
    for model in payload["models"]:
        lines.append(f"- {model['model_name']}: {model.get('stratified_key_findings') or 'not supplied'}")
    lines.extend(["", "## Threshold-policy findings", ""])
    for model in payload["models"]:
        lines.append(f"- {model['model_name']}: {model.get('threshold_profiles') or 'not supplied'}")
    lines.extend(["", "## Neural diagnostics", ""])
    for model in payload["models"]:
        diagnostics = model.get("diagnostics") or {}
        if not diagnostics:
            lines.append(f"- {model['model_name']}: not supplied")
        else:
            lines.append(
                "- {name}: all-split probability std={std}, separation={sep}, warnings={warnings}".format(
                    name=model["model_name"],
                    std=_format_float(diagnostics.get("probability_std_all")),
                    sep=_format_float(diagnostics.get("separation_all")),
                    warnings=diagnostics.get("warnings") or [],
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Prefer validation AUROC when available; otherwise treat test/all comparisons as stability checks.",
            "- If saturation-aware variants underperform contrastive v2, treat saturation as post-hoc risk reporting until architecture improves.",
            "- If group weighting or balanced sampling hurts, disable it in the next repaired preset.",
            "",
            "## Recommendation",
            "",
            f"- {payload['recommendation']['text']}",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _validate_optional_dirs(values: Optional[List[str]], label: str) -> None:
    if values is None:
        return
    for value in values:
        if value and not Path(value).exists():
            raise ValueError(f"{label} entry does not exist: {value}")


def _optional_json(
    directories: Optional[List[str]], index: int, filename: str, warnings: List[str]
) -> Optional[dict]:
    if directories is None:
        return None
    directory = directories[index]
    if not directory:
        return None
    path = Path(directory) / filename
    if not path.exists():
        warnings.append(f"missing_optional_file:{path}")
        return None
    return _load_json(path)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def _format_float(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
