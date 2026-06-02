#!/usr/bin/env python3
"""Summarize Drosophila BABAPPA vs HyPhy aBSREL benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float | None:
    try:
        if value in ("", "NA", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int:
    try:
        if value in ("", "NA", None):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def collect_numeric_values_by_key(value: Any, key_names: set[str]) -> list[float]:
    values: list[float] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_norm = str(key).strip().lower()
            if key_norm in key_names and isinstance(item, (int, float)):
                values.append(float(item))
            values.extend(collect_numeric_values_by_key(item, key_names))
    elif isinstance(value, list):
        for item in value:
            values.extend(collect_numeric_values_by_key(item, key_names))
    return values


def parse_absrel(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "min_p_value": None, "n_p_values": 0, "result_class": "pending"}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {"status": "parse_error", "min_p_value": None, "n_p_values": 0, "result_class": "failed", "error": str(exc)}
    p_values = collect_numeric_values_by_key(
        data,
        {"p", "p-value", "p_value", "corrected p-value", "uncorrected p-value"},
    )
    min_p = min(p_values) if p_values else None
    return {
        "status": "parsed",
        "min_p_value": min_p,
        "n_p_values": len(p_values),
        "result_class": "positive" if min_p is not None and min_p < 0.05 else ("negative" if min_p is not None else "inconclusive"),
    }


def load_babappa(panel_id: str, results_root: Path) -> dict[str, Any]:
    outdir = results_root / "babappa" / panel_id
    branches = read_tsv(outdir / "branch_predictions.tsv")
    gene = read_tsv(outdir / "gene_summary.tsv")
    manifest_path = outdir / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    called_rows = sum(safe_int(row.get("n_called_positive")) for row in branches)
    max_branch_score = max([safe_float(row.get("max_prob_positive")) or 0.0 for row in branches], default=0.0)
    max_gene_support = max([safe_float(row.get("max_gene_support")) or safe_float(row.get("gene_support")) or 0.0 for row in gene], default=0.0)
    if not branches and not gene:
        status = "missing"
        result_class = "pending"
    else:
        status = "ok"
        result_class = "positive" if called_rows > 0 else "negative"
    return {
        "status": status,
        "result_class": result_class,
        "called_branch_site_rows": called_rows,
        "max_branch_score": max_branch_score,
        "max_gene_support": max_gene_support,
        "applicability": manifest.get("applicability", manifest.get("applicability_status", "")),
        "diagnostic_only": manifest.get("diagnostic_only", ""),
    }


def concordance(babappa_class: str, hyphy_class: str) -> str:
    if babappa_class in {"pending", "missing"} or hyphy_class in {"pending", "missing", "failed"}:
        return "incomplete"
    if babappa_class == "positive" and hyphy_class == "positive":
        return "concordant_positive"
    if babappa_class == "negative" and hyphy_class == "negative":
        return "concordant_negative"
    if babappa_class == "positive" and hyphy_class == "negative":
        return "BABAPPA_only"
    if babappa_class == "negative" and hyphy_class == "positive":
        return "HyPhy_only"
    return "inconclusive"


def summarize(args: argparse.Namespace) -> int:
    panel = Path(args.panel)
    results_root = Path(args.results_root)
    outdir = Path(args.outdir)
    panel_rows = read_tsv(panel)
    rows: list[dict[str, Any]] = []
    for item in panel_rows:
        panel_id = item["panel_id"]
        babappa = load_babappa(panel_id, results_root)
        hyphy = parse_absrel(results_root / "hyphy_absrel" / panel_id / "absrel.json")
        rows.append({
            "panel_id": panel_id,
            "gene_family": item.get("gene_family", panel_id),
            "benchmark_stratum": item.get("benchmark_stratum", ""),
            "n_called_branch_site_rows": babappa["called_branch_site_rows"],
            "babappa_max_branch_score": f"{babappa['max_branch_score']:.6g}",
            "babappa_max_gene_support": f"{babappa['max_gene_support']:.6g}",
            "babappa_applicability": babappa["applicability"],
            "babappa_diagnostic_only": babappa["diagnostic_only"],
            "babappa_result_class": babappa["result_class"],
            "hyphy_absrel_min_p": "NA" if hyphy["min_p_value"] is None else f"{hyphy['min_p_value']:.6g}",
            "hyphy_absrel_n_p_values": hyphy["n_p_values"],
            "hyphy_result_class": hyphy["result_class"],
            "concordance": concordance(str(babappa["result_class"]), str(hyphy["result_class"])),
        })

    fieldnames = [
        "panel_id", "gene_family", "benchmark_stratum", "n_called_branch_site_rows",
        "babappa_max_branch_score", "babappa_max_gene_support",
        "babappa_applicability", "babappa_diagnostic_only",
        "babappa_result_class", "hyphy_absrel_min_p", "hyphy_absrel_n_p_values",
        "hyphy_result_class", "concordance",
    ]
    write_tsv(outdir / "babappa_vs_hyphy_absrel.tsv", rows, fieldnames)
    counts: dict[str, int] = {}
    stratum_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        counts[row["concordance"]] = counts.get(row["concordance"], 0) + 1
        stratum = str(row.get("benchmark_stratum") or "unstratified")
        stratum_counts.setdefault(stratum, {})
        stratum_counts[stratum][row["concordance"]] = stratum_counts[stratum].get(row["concordance"], 0) + 1
    payload = {"status": "ok", "n_families": len(rows), "concordance_counts": counts, "concordance_by_stratum": stratum_counts}
    (outdir / "babappa_vs_hyphy_absrel_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md = ["# Drosophila BABAPPA vs HyPhy aBSREL Benchmark", ""]
    md.append(f"- families summarized: `{len(rows)}`")
    for key, value in sorted(counts.items()):
        md.append(f"- {key}: `{value}`")
    if stratum_counts:
        md.append("")
        md.append("## Concordance By Stratum")
        md.append("")
        for stratum, values in sorted(stratum_counts.items()):
            details = ", ".join(f"{key}={value}" for key, value in sorted(values.items()))
            md.append(f"- `{stratum}`: {details}")
    md.append("")
    md.append("Interpretation: this is a publication benchmark comparator. HyPhy aBSREL is an external branch-level likelihood reference, not BABAPPA's training target or ground truth.")
    (outdir / "babappa_vs_hyphy_absrel_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {outdir / 'babappa_vs_hyphy_absrel.tsv'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="publication_benchmark/drosophila_absrel_benchmark/drosophila_babappa_absrel_panel.tsv")
    parser.add_argument("--results-root", default="publication_benchmark/drosophila_absrel_benchmark/results")
    parser.add_argument("--outdir", default="publication_benchmark/drosophila_absrel_benchmark/results/summary")
    args = parser.parse_args()
    return summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
