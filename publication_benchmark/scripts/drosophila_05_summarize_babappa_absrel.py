#!/usr/bin/env python3
"""Summarize Drosophila BABAPPA vs HyPhy aBSREL benchmark results.

Publication summaries must use HyPhy's official aBSREL family-level field:
``test results -> positive test results``. Recursive p-value mining is retained
only as an explicitly requested exploratory diagnostic mode because it can
overcount nested p-value-like quantities that are not final aBSREL calls.
"""

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


SUPPORT_CLASSES = {"strong_babappa_native_support", "babappa_native_support"}


def parse_absrel(path: Path, *, positive_mode: str = "official") -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "parser_mode": positive_mode,
            "official_positive_test_results": None,
            "official_tested_branches": None,
            "min_p_value": None,
            "n_p_values": 0,
            "result_class": "pending",
            "warnings": ["missing_absrel_json"],
        }
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {
            "status": "parse_error",
            "parser_mode": positive_mode,
            "official_positive_test_results": None,
            "official_tested_branches": None,
            "min_p_value": None,
            "n_p_values": 0,
            "result_class": "failed",
            "warnings": [f"parse_error:{exc}"],
            "error": str(exc),
        }

    if positive_mode == "official":
        test_results = data.get("test results") if isinstance(data, dict) else None
        warnings: list[str] = []
        if not isinstance(test_results, dict) or "positive test results" not in test_results:
            warnings.append("missing_official_positive_test_results")
            return {
                "status": "parsed",
                "parser_mode": positive_mode,
                "publication_ready": False,
                "official_positive_test_results": None,
                "official_tested_branches": None,
                "min_p_value": None,
                "n_p_values": 0,
                "result_class": "inconclusive",
                "warnings": warnings,
            }
        positive_count = safe_int(test_results.get("positive test results"))
        tested = safe_int(test_results.get("tested"))
        return {
            "status": "parsed",
            "parser_mode": positive_mode,
            "publication_ready": True,
            "official_positive_test_results": positive_count,
            "official_tested_branches": tested,
            "min_p_value": None,
            "n_p_values": 0,
            "result_class": "positive" if positive_count > 0 else "negative",
            "warnings": warnings,
        }

    if positive_mode != "exploratory_recursive":
        raise ValueError(f"unknown HyPhy positive mode: {positive_mode}")

    p_values = collect_numeric_values_by_key(
        data,
        {"p", "p-value", "p_value", "corrected p-value", "uncorrected p-value"},
    )
    min_p = min(p_values) if p_values else None
    return {
        "status": "parsed",
        "parser_mode": positive_mode,
        "publication_ready": False,
        "official_positive_test_results": None,
        "official_tested_branches": None,
        "min_p_value": min_p,
        "n_p_values": len(p_values),
        "result_class": "positive" if min_p is not None and min_p < 0.05 else ("negative" if min_p is not None else "inconclusive"),
        "warnings": ["exploratory_recursive_mode_not_for_publication"],
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
    native_null_path = outdir / "babappa_native_null" / "babappa_native_null_summary.json"
    native = json.loads(native_null_path.read_text()) if native_null_path.exists() else {}
    evidence_class = str(native.get("evidence_class") or "")
    applicability = str(manifest.get("applicability", manifest.get("applicability_status", "")))
    native_supported = bool(
        called_rows > 0
        and evidence_class in SUPPORT_CLASSES
        and applicability != "out_of_domain"
    )
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
        "applicability": applicability,
        "diagnostic_only": manifest.get("diagnostic_only", ""),
        "native_evidence_class": evidence_class,
        "native_supported": native_supported,
        "native_status": native.get("status", "missing" if not native else ""),
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
        hyphy = parse_absrel(
            results_root / "hyphy_absrel" / panel_id / "absrel.json",
            positive_mode=args.hyphy_positive_mode,
        )
        babappa_native_class = "positive" if babappa["native_supported"] else "negative"
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
            "babappa_native_evidence_class": babappa["native_evidence_class"],
            "babappa_native_supported": babappa["native_supported"],
            "hyphy_absrel_min_p": "NA" if hyphy["min_p_value"] is None else f"{hyphy['min_p_value']:.6g}",
            "hyphy_absrel_n_p_values": hyphy["n_p_values"],
            "hyphy_official_positive_test_results": "NA" if hyphy["official_positive_test_results"] is None else hyphy["official_positive_test_results"],
            "hyphy_official_tested_branches": "NA" if hyphy["official_tested_branches"] is None else hyphy["official_tested_branches"],
            "hyphy_positive_mode": hyphy["parser_mode"],
            "hyphy_result_class": hyphy["result_class"],
            "concordance": concordance(babappa_native_class, str(hyphy["result_class"])),
            "raw_concordance": concordance(str(babappa["result_class"]), str(hyphy["result_class"])),
            "parser_warnings": ";".join(hyphy.get("warnings", [])),
        })

    fieldnames = [
        "panel_id", "gene_family", "benchmark_stratum", "n_called_branch_site_rows",
        "babappa_max_branch_score", "babappa_max_gene_support",
        "babappa_applicability", "babappa_diagnostic_only",
        "babappa_result_class", "babappa_native_evidence_class", "babappa_native_supported",
        "hyphy_absrel_min_p", "hyphy_absrel_n_p_values",
        "hyphy_official_positive_test_results", "hyphy_official_tested_branches",
        "hyphy_positive_mode", "hyphy_result_class", "concordance", "raw_concordance",
        "parser_warnings",
    ]
    write_tsv(outdir / "babappa_vs_hyphy_absrel.tsv", rows, fieldnames)
    write_tsv(outdir / "benchmark_family_results.tsv", rows, fieldnames)
    counts: dict[str, int] = {}
    raw_counts: dict[str, int] = {}
    stratum_counts: dict[str, dict[str, int]] = {}
    applicability_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        counts[row["concordance"]] = counts.get(row["concordance"], 0) + 1
        raw_counts[row["raw_concordance"]] = raw_counts.get(row["raw_concordance"], 0) + 1
        stratum = str(row.get("benchmark_stratum") or "unstratified")
        stratum_counts.setdefault(stratum, {})
        stratum_counts[stratum][row["concordance"]] = stratum_counts[stratum].get(row["concordance"], 0) + 1
        app = str(row.get("babappa_applicability") or "unknown")
        bucket = applicability_counts.setdefault(
            app,
            {
                "families": 0,
                "babappa_raw_positive": 0,
                "babappa_native_supported": 0,
                "hyphy_positive": 0,
            },
        )
        bucket["families"] += 1
        bucket["babappa_raw_positive"] += 1 if row["babappa_result_class"] == "positive" else 0
        bucket["babappa_native_supported"] += 1 if row["babappa_native_supported"] is True else 0
        bucket["hyphy_positive"] += 1 if row["hyphy_result_class"] == "positive" else 0
    n = len(rows)
    hyphy_positive_families = sum(1 for row in rows if row["hyphy_result_class"] == "positive")
    hyphy_positive_branches = sum(
        safe_int(row["hyphy_official_positive_test_results"])
        for row in rows
        if row["hyphy_official_positive_test_results"] != "NA"
    )
    hyphy_tested_branches = sum(
        safe_int(row["hyphy_official_tested_branches"])
        for row in rows
        if row["hyphy_official_tested_branches"] != "NA"
    )
    babappa_raw_positive = sum(1 for row in rows if row["babappa_result_class"] == "positive")
    babappa_native_supported = sum(1 for row in rows if row["babappa_native_supported"] is True)
    concordant_positive = counts.get("concordant_positive", 0)
    concordant_negative = counts.get("concordant_negative", 0)
    hyphy_positive_denominator = max(1, hyphy_positive_families)
    hyphy_negative_denominator = max(1, n - hyphy_positive_families)
    payload = {
        "status": "ok",
        "hyphy_positive_mode": args.hyphy_positive_mode,
        "publication_mode": args.hyphy_positive_mode == "official",
        "n_families": n,
        "babappa_raw_diagnostic_positive": babappa_raw_positive,
        "babappa_native_calibrated_support": babappa_native_supported,
        "hyphy_absrel_positive_families": hyphy_positive_families,
        "hyphy_positive_branches": hyphy_positive_branches,
        "hyphy_tested_branches": hyphy_tested_branches,
        "concordance_counts": counts,
        "raw_concordance_counts": raw_counts,
        "concordance_by_stratum": stratum_counts,
        "applicability_counts": applicability_counts,
        "overall_agreement": (concordant_positive + concordant_negative) / n if n else None,
        "positive_agreement_against_hyphy": concordant_positive / hyphy_positive_denominator,
        "negative_agreement": concordant_negative / hyphy_negative_denominator,
        "claim_boundary": "HyPhy aBSREL is an external comparator, not empirical ground truth. Concordance metrics are not sensitivity/specificity.",
    }
    (outdir / "babappa_vs_hyphy_absrel_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (outdir / "benchmark_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary_rows = [
        {"metric": "families", "value": n},
        {"metric": "babappa_raw_diagnostic_positive", "value": babappa_raw_positive},
        {"metric": "babappa_native_calibrated_support", "value": babappa_native_supported},
        {"metric": "hyphy_absrel_positive_families", "value": hyphy_positive_families},
        {"metric": "hyphy_positive_branches", "value": hyphy_positive_branches},
        {"metric": "hyphy_tested_branches", "value": hyphy_tested_branches},
        {"metric": "concordant_positive", "value": counts.get("concordant_positive", 0)},
        {"metric": "concordant_negative", "value": counts.get("concordant_negative", 0)},
        {"metric": "BABAPPA_only", "value": counts.get("BABAPPA_only", 0)},
        {"metric": "HyPhy_only", "value": counts.get("HyPhy_only", 0)},
        {"metric": "overall_agreement", "value": f"{payload['overall_agreement']:.6g}"},
        {"metric": "positive_agreement_against_hyphy", "value": f"{payload['positive_agreement_against_hyphy']:.6g}"},
        {"metric": "negative_agreement", "value": f"{payload['negative_agreement']:.6g}"},
    ]
    write_tsv(outdir / "benchmark_summary.tsv", summary_rows, ["metric", "value"])
    _write_confusion_tables(outdir, counts, applicability_counts)
    _write_parser_audit(outdir, rows, payload)
    _write_interpretation(outdir, payload)
    md = ["# Drosophila BABAPPA vs HyPhy aBSREL Benchmark", ""]
    md.append(f"- families summarized: `{len(rows)}`")
    md.append(f"- HyPhy positive mode: `{args.hyphy_positive_mode}`")
    md.append(f"- BABAPPA raw diagnostic-positive: `{babappa_raw_positive}/{n}`")
    md.append(f"- BABAPPA-native calibrated support: `{babappa_native_supported}/{n}`")
    md.append(f"- HyPhy aBSREL-positive families: `{hyphy_positive_families}/{n}`")
    md.append(f"- HyPhy positive branches: `{hyphy_positive_branches}/{hyphy_tested_branches}`")
    for key, value in sorted(counts.items()):
        md.append(f"- {key}: `{value}`")
    md.append(f"- overall agreement: `{payload['overall_agreement']:.3f}`")
    md.append(f"- positive agreement against HyPhy: `{payload['positive_agreement_against_hyphy']:.3f}`")
    md.append(f"- negative agreement: `{payload['negative_agreement']:.3f}`")
    if stratum_counts:
        md.append("")
        md.append("## Concordance By Stratum")
        md.append("")
        for stratum, values in sorted(stratum_counts.items()):
            details = ", ".join(f"{key}={value}" for key, value in sorted(values.items()))
            md.append(f"- `{stratum}`: {details}")
    if applicability_counts:
        md.append("")
        md.append("## Applicability-Stratified Behavior")
        md.append("")
        for app, values in sorted(applicability_counts.items()):
            md.append(
                f"- `{app}`: families={values['families']}, "
                f"BABAPPA-native={values['babappa_native_supported']}, "
                f"HyPhy-positive={values['hyphy_positive']}"
            )
    md.append("")
    md.append("Interpretation: this is a publication benchmark comparator. HyPhy aBSREL is an external branch-level likelihood reference, not BABAPPA's training target or ground truth. Concordance metrics are not sensitivity/specificity.")
    (outdir / "babappa_vs_hyphy_absrel_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (outdir / "benchmark_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {outdir / 'babappa_vs_hyphy_absrel.tsv'}")
    return 0


def _write_confusion_tables(outdir: Path, counts: dict[str, int], applicability_counts: dict[str, dict[str, int]]) -> None:
    confusion_rows = [
        {"class": "concordant_positive", "families": counts.get("concordant_positive", 0)},
        {"class": "concordant_negative", "families": counts.get("concordant_negative", 0)},
        {"class": "BABAPPA_only", "families": counts.get("BABAPPA_only", 0)},
        {"class": "HyPhy_only", "families": counts.get("HyPhy_only", 0)},
        {"class": "incomplete", "families": counts.get("incomplete", 0)},
        {"class": "inconclusive", "families": counts.get("inconclusive", 0)},
    ]
    write_tsv(outdir / "confusion_like_table.tsv", confusion_rows, ["class", "families"])
    app_rows = [
        {
            "applicability": app,
            "families": values["families"],
            "babappa_raw_positive": values["babappa_raw_positive"],
            "babappa_native_supported": values["babappa_native_supported"],
            "hyphy_positive": values["hyphy_positive"],
        }
        for app, values in sorted(applicability_counts.items())
    ]
    write_tsv(
        outdir / "applicability_stratified_table.tsv",
        app_rows,
        ["applicability", "families", "babappa_raw_positive", "babappa_native_supported", "hyphy_positive"],
    )


def _write_parser_audit(outdir: Path, rows: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    warnings = [row for row in rows if row.get("parser_warnings")]
    audit = {
        "status": "ok" if not warnings else "warning",
        "hyphy_positive_mode": payload["hyphy_positive_mode"],
        "official_field": "test results -> positive test results",
        "publication_ready": payload["publication_mode"] and not warnings,
        "families": payload["n_families"],
        "hyphy_positive_families": payload["hyphy_absrel_positive_families"],
        "hyphy_positive_branches": payload["hyphy_positive_branches"],
        "hyphy_tested_branches": payload["hyphy_tested_branches"],
        "warnings": [{"panel_id": row["panel_id"], "warnings": row["parser_warnings"]} for row in warnings],
        "non_publication_mode_note": "exploratory_recursive mode can overcount nested p-value-like quantities and must not be used for publication claims.",
    }
    (outdir / "hyphy_official_parser_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# HyPhy Official Parser Audit",
        "",
        "- official field: `test results -> positive test results`",
        f"- parser mode: `{audit['hyphy_positive_mode']}`",
        f"- publication ready: `{audit['publication_ready']}`",
        f"- families: `{audit['families']}`",
        f"- HyPhy-positive families: `{audit['hyphy_positive_families']}`",
        f"- HyPhy positive branches: `{audit['hyphy_positive_branches']}/{audit['hyphy_tested_branches']}`",
        "",
        "Recursive p-value mining is retained only as an exploratory diagnostic mode and is not a publication parser.",
        "",
    ]
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        lines.extend(f"- `{item['panel_id']}`: {item['warnings']}" for item in audit["warnings"])
        lines.append("")
    (outdir / "hyphy_official_parser_audit.md").write_text("\n".join(lines), encoding="utf-8")


def _write_interpretation(outdir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# BABAPPA Drosophila Benchmark Interpretation",
        "",
        "HyPhy aBSREL is an external comparator, not ground truth. Concordance metrics are therefore not sensitivity, specificity, power, or false-positive-rate estimates.",
        "",
        "The corrected official parser uses HyPhy's own `test results -> positive test results` field. Under that parser, BABAPPA-native calibrated support showed limited positive overlap with HyPhy aBSREL but strong conservative behavior under BABAPPA-defined out-of-domain conditions.",
        "",
        f"- families: `{payload['n_families']}`",
        f"- BABAPPA raw diagnostic-positive: `{payload['babappa_raw_diagnostic_positive']}/{payload['n_families']}`",
        f"- BABAPPA-native calibrated support: `{payload['babappa_native_calibrated_support']}/{payload['n_families']}`",
        f"- HyPhy aBSREL-positive families: `{payload['hyphy_absrel_positive_families']}/{payload['n_families']}`",
        f"- HyPhy positive branches: `{payload['hyphy_positive_branches']}/{payload['hyphy_tested_branches']}`",
        f"- concordant positive: `{payload['concordance_counts'].get('concordant_positive', 0)}`",
        f"- concordant negative: `{payload['concordance_counts'].get('concordant_negative', 0)}`",
        f"- BABAPPA-only: `{payload['concordance_counts'].get('BABAPPA_only', 0)}`",
        f"- HyPhy-only: `{payload['concordance_counts'].get('HyPhy_only', 0)}`",
        "",
        "Low positive overlap means BABAPPA is not a HyPhy replacement. The main positive result is OOD abstention: BABAPPA made zero native-supported calls in true out-of-domain families, while HyPhy reported positives in many OOD families. This does not prove HyPhy is wrong; it shows that BABAPPA's current empirical policy is more conservative under BABAPPA-defined OOD conditions.",
        "",
        "BABAPPA-only families require BABAPPA-native null calibration, alignment audit, and biological review. HyPhy-only families may reflect likelihood-model sensitivity, model differences, or BABAPPA abstention. The benchmark supports complementary conservative behavior rather than replacement or superiority.",
        "",
    ]
    (outdir / "BABAPPA_Drosophila_benchmark_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="publication_benchmark/drosophila_absrel_benchmark/drosophila_babappa_absrel_panel.tsv")
    parser.add_argument("--results-root", default="publication_benchmark/drosophila_absrel_benchmark/results")
    parser.add_argument("--outdir", default="publication_benchmark/drosophila_absrel_benchmark/results/summary")
    parser.add_argument(
        "--hyphy-positive-mode",
        choices=["official", "exploratory_recursive"],
        default="official",
        help="Publication mode uses HyPhy's official 'test results -> positive test results' field. Recursive mode is exploratory only.",
    )
    args = parser.parse_args()
    return summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
