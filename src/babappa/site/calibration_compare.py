"""Compare site-level calibration runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from babappa import __version__
from babappa.datasets.index import write_tsv

SITE_CALIBRATION_COMPARISON_VERSION = __version__
FIELDNAMES = [
    "name",
    "method",
    "selected_threshold",
    "target_fdr",
    "calibration_brier",
    "calibration_ece",
    "all_precision",
    "all_recall",
    "all_f1",
    "all_mcc",
    "warnings",
]


@dataclass(frozen=True)
class SiteCalibrationCompareConfig:
    """Configuration for comparing calibration directories."""

    calibration_dirs: List[str]
    outdir: str
    names: Optional[List[str]] = None
    title: str = "BABAPPA site calibration comparison"

    def __post_init__(self) -> None:
        if len(self.calibration_dirs) < 2:
            raise ValueError("at least two calibration_dirs must be supplied")
        for directory in self.calibration_dirs:
            path = Path(directory)
            if not (path / "site_calibration.json").exists():
                raise ValueError(f"missing site_calibration.json: {path}")
            if not (path / "site_calibrated_metrics.json").exists():
                raise ValueError(f"missing site_calibrated_metrics.json: {path}")
        if self.names is not None:
            if len(self.names) != len(self.calibration_dirs):
                raise ValueError("names must match calibration_dirs length")
            if len(set(self.names)) != len(self.names):
                raise ValueError("names must be unique")
        Path(self.outdir).mkdir(parents=True, exist_ok=True)


def compare_site_calibrations(config: SiteCalibrationCompareConfig) -> dict:
    """Compare site calibration outputs."""
    outdir = Path(config.outdir)
    names = config.names or [Path(path).name for path in config.calibration_dirs]
    rows = []
    calibrations = {}
    for name, directory in zip(names, config.calibration_dirs):
        path = Path(directory)
        calibration = _load_json(path / "site_calibration.json")
        metrics = _load_json(path / "site_calibrated_metrics.json")
        all_metrics = metrics.get("metrics_by_split_calibrated", {}).get("all", {})
        calib_metrics = calibration.get("calibrated_calibration_metrics", {})
        row = {
            "name": name,
            "method": calibration.get("calibration_method"),
            "selected_threshold": calibration.get("selected_threshold"),
            "target_fdr": calibration.get("target_fdr"),
            "calibration_brier": calib_metrics.get("brier"),
            "calibration_ece": calib_metrics.get("ece"),
            "all_precision": all_metrics.get("precision"),
            "all_recall": all_metrics.get("recall"),
            "all_f1": all_metrics.get("f1"),
            "all_mcc": all_metrics.get("mcc"),
            "warnings": ",".join(calibration.get("warnings", [])),
        }
        rows.append(row)
        calibrations[name] = row
    recommendation = _recommend(rows)
    payload = {
        "site_calibration_comparison_version": SITE_CALIBRATION_COMPARISON_VERSION,
        "title": config.title,
        "inputs": dict(zip(names, config.calibration_dirs)),
        "calibrations": calibrations,
        "recommendation": recommendation,
        "generated_files": {
            "json": str(outdir / "site_calibration_comparison.json"),
            "tsv": str(outdir / "site_calibration_comparison.tsv"),
            "markdown": str(outdir / "site_calibration_comparison.md"),
        },
    }
    _write_json(outdir / "site_calibration_comparison.json", payload)
    write_tsv(outdir / "site_calibration_comparison.tsv", rows, FIELDNAMES)
    (outdir / "site_calibration_comparison.md").write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "outdir": str(outdir),
        "json": str(outdir / "site_calibration_comparison.json"),
        "tsv": str(outdir / "site_calibration_comparison.tsv"),
        "markdown": str(outdir / "site_calibration_comparison.md"),
        "recommendation": recommendation,
    }


def _recommend(rows: List[dict]) -> str:
    candidates = [row for row in rows if row.get("calibration_ece") is not None]
    if candidates:
        best = min(candidates, key=lambda row: float(row["calibration_ece"]))
        return f"Prefer {best['name']} by calibration-split ECE, subject to threshold warnings."
    return "Inspect warnings and threshold-dependent precision/recall before choosing a calibration method."


def _render_markdown(payload: dict) -> str:
    lines = [
        f"# {payload.get('title')}",
        "",
        "## Calibration methods",
        "",
    ]
    for name, row in payload.get("calibrations", {}).items():
        lines.append(
            f"- {name}: method={row.get('method')}, ECE={row.get('calibration_ece')}, "
            f"threshold={row.get('selected_threshold')}, warnings={row.get('warnings') or 'none'}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Temperature scaling preserves score ordering; quantile calibration can repair probability scale but should be checked against FDR and precision targets.",
            "",
            "## Recommendation",
            "",
            payload.get("recommendation", ""),
            "",
        ]
    )
    return "\n".join(lines)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
