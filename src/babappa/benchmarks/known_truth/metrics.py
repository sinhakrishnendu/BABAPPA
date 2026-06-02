"""Known-truth benchmark metrics for BABAPPA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from babappa import __version__
from babappa.datasets.index import read_tsv, write_tsv

from .truth_schema import write_json


@dataclass(frozen=True)
class KnownTruthEvaluationConfig:
    truth: str
    scores: str
    outdir: str


@dataclass(frozen=True)
class KnownTruthCalibrationEvaluationConfig:
    truth: str
    scores: str
    outdir: str


def evaluate_known_truth_benchmark(config: KnownTruthEvaluationConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    truth_rows = read_tsv(Path(config.truth))
    scores_dir = Path(config.scores)
    gene_scores = _rows_by_family(read_tsv(scores_dir / "gene_support.tsv"), "score", "gene_support")
    applicability = _rows_by_family(read_tsv(scores_dir / "applicability.tsv"), "applicability_status")
    labels: List[int] = []
    scores: List[float] = []
    eval_rows: List[Dict[str, Any]] = []
    ood_false_calls = 0
    ood_total = 0
    ood_abstained = 0
    for truth in truth_rows:
        family_id = truth["family_id"]
        label = 1 if truth["truth_class"] == "positive" else 0
        score_row = gene_scores.get(family_id, {})
        score = _float(score_row.get("score", score_row.get("gene_support", 0.0)))
        called = _bool(score_row.get("called_positive", score >= 0.5))
        app = applicability.get(family_id, {}).get("applicability_status", truth.get("expected_applicability", ""))
        if truth.get("expected_applicability") == "out_of_domain" or app == "out_of_domain":
            ood_total += 1
            if not called:
                ood_abstained += 1
            if called and label == 0:
                ood_false_calls += 1
        labels.append(label)
        scores.append(score)
        eval_rows.append(
            {
                "family_id": family_id,
                "regime": truth["regime"],
                "truth_class": truth["truth_class"],
                "label": label,
                "score": f"{score:.6f}",
                "called_positive": str(called),
                "applicability": app,
                "saturation_tier": truth["saturation_tier"],
                "n_taxa": truth["n_taxa"],
                "n_codons": truth["n_codons"],
            }
        )
    confusion = _confusion(labels, scores, 0.5)
    metrics = {
        "auroc": _auroc(labels, scores),
        "auprc": _auprc(labels, scores),
        **confusion,
        "ood_total": ood_total,
        "ood_abstention_rate": _safe_div(ood_abstained, ood_total),
        "ood_false_call_rate": _safe_div(ood_false_calls, ood_total),
        "false_positives_in_ood_null_families": ood_false_calls,
    }
    stratified = _stratified_metrics(eval_rows, "saturation_tier")
    stratified.extend(_stratified_metrics(eval_rows, "applicability"))
    calibration_rows = _calibration_bins(labels, scores)
    branch_metrics = _branch_site_metrics(Path(config.truth), scores_dir)
    payload = {
        "known_truth_evaluation_version": __version__,
        "status": "ok",
        "n_families": len(truth_rows),
        "gene_level": metrics,
        "branch_site_level": branch_metrics,
        "truth_used_for_evaluation_only": True,
        "claim_boundary": "Metrics are simulation-known-truth validation metrics, not empirical discovery claims.",
    }
    write_json(outdir / "evaluation_summary.json", payload)
    write_tsv(outdir / "evaluation_summary.tsv", [{"metric": key, "value": value} for key, value in metrics.items()], ["metric", "value"])
    write_tsv(outdir / "stratified_metrics.tsv", stratified, ["stratum_type", "stratum", "n", "auroc", "auprc", "precision", "recall", "specificity", "fdr"])
    write_tsv(outdir / "confusion_tables.tsv", [{"threshold": 0.5, **confusion}], ["threshold", "tp", "fp", "tn", "fn", "precision", "recall", "specificity", "f1", "mcc", "fpr", "fnr", "empirical_fdr", "accuracy"])
    write_tsv(outdir / "calibration_table.tsv", calibration_rows, ["bin", "n", "mean_score", "observed_positive_rate"])
    write_tsv(outdir / "ood_abstention_table.tsv", [{"ood_total": ood_total, "ood_abstained": ood_abstained, "ood_false_calls": ood_false_calls, "ood_abstention_rate": metrics["ood_abstention_rate"], "ood_false_call_rate": metrics["ood_false_call_rate"]}], ["ood_total", "ood_abstained", "ood_false_calls", "ood_abstention_rate", "ood_false_call_rate"])
    write_tsv(outdir / "power_by_effect_size.tsv", _power_by_regime(eval_rows), ["regime", "n", "positive_n", "called_positive_n", "power"])
    (outdir / "evaluation_summary.md").write_text(_render_evaluation_md(payload), encoding="utf-8")
    return {"status": "ok", "outdir": str(outdir), "n_families": len(truth_rows), "auroc": metrics["auroc"], "auprc": metrics["auprc"]}


def evaluate_known_truth_calibration(config: KnownTruthCalibrationEvaluationConfig) -> Dict[str, Any]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    truth_rows = read_tsv(Path(config.truth))
    truth_label = {row["family_id"]: 1 if row["truth_class"] == "positive" else 0 for row in truth_rows}
    score_rows = read_tsv(Path(config.scores) / "gene_support.tsv")
    p_rows: List[Dict[str, Any]] = []
    p_values: List[float] = []
    for row in score_rows:
        score = _float(row.get("score", row.get("gene_support", 0.0)))
        p_like = max(0.0001, min(1.0, 1.0 - score))
        p_values.append(p_like)
        p_rows.append({"family_id": row["family_id"], "score": score, "p_like": p_like, "label": truth_label.get(row["family_id"], 0)})
    q_values = bh_qvalues(p_values)
    for row, q in zip(p_rows, q_values):
        row["q_value"] = q
        row["called_q_0_05"] = q <= 0.05
        row["called_q_0_10"] = q <= 0.10
    fdr_power = []
    for threshold in [0.05, 0.10]:
        called = [row for row in p_rows if row["q_value"] <= threshold]
        tp = sum(1 for row in called if int(row["label"]) == 1)
        fp = sum(1 for row in called if int(row["label"]) == 0)
        positives = sum(1 for row in p_rows if int(row["label"]) == 1)
        fdr_power.append(
            {
                "q_threshold": threshold,
                "called": len(called),
                "tp": tp,
                "fp": fp,
                "empirical_fdr": _safe_div(fp, len(called)),
                "power": _safe_div(tp, positives),
            }
        )
    payload = {
        "known_truth_calibration_evaluation_version": __version__,
        "status": "ok",
        "n_tests": len(p_rows),
        "fdr_power": fdr_power,
        "claim_boundary": "Q-values are evaluated against simulation truth; empirical claims require empirical calibration.",
    }
    write_json(outdir / "calibration_evaluation.json", payload)
    write_tsv(outdir / "qvalue_table.tsv", p_rows, ["family_id", "score", "p_like", "q_value", "label", "called_q_0_05", "called_q_0_10"])
    write_tsv(outdir / "fdr_power_table.tsv", fdr_power, ["q_threshold", "called", "tp", "fp", "empirical_fdr", "power"])
    (outdir / "calibration_evaluation.md").write_text(_render_calibration_md(payload), encoding="utf-8")
    return {"status": "ok", "outdir": str(outdir), "n_tests": len(p_rows)}


def bh_qvalues(p_values: List[float]) -> List[float]:
    n = len(p_values)
    if n == 0:
        return []
    ranked = sorted(enumerate(p_values), key=lambda item: item[1])
    q = [1.0] * n
    running = 1.0
    for rank_from_end, (idx, p_value) in enumerate(reversed(ranked), start=1):
        rank = n - rank_from_end + 1
        running = min(running, p_value * n / rank)
        q[idx] = min(1.0, running)
    return q


def _rows_by_family(rows: List[Dict[str, str]], *required_hint: str) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if row.get("family_id"):
            result[row["family_id"]] = row
    return result


def _branch_site_metrics(truth_manifest: Path, scores_dir: Path) -> Dict[str, Any]:
    truth_rows = read_tsv(truth_manifest)
    labels: Dict[Tuple[str, str, str], int] = {}
    for family in truth_rows:
        truth_path = Path(family["branch_site_truth_tsv"])
        if truth_path.exists():
            for row in read_tsv(truth_path):
                labels[(row["family_id"], row["branch"], row["site"])] = int(row.get("label", 0))
    score_path = scores_dir / "site_scores.tsv"
    if not score_path.exists():
        return {"status": "missing_site_scores"}
    y: List[int] = []
    s: List[float] = []
    for row in read_tsv(score_path):
        key = (row.get("family_id", ""), row.get("branch", ""), row.get("site", ""))
        if key in labels:
            y.append(labels[key])
            s.append(_float(row.get("score", row.get("branch_site_support", 0.0))))
    return {
        "n_branch_site_rows": len(y),
        "branch_site_auroc": _auroc(y, s),
        "branch_site_auprc": _auprc(y, s),
        **_confusion(y, s, 0.5),
    }


def _stratified_metrics(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "")), []).append(row)
    output = []
    for stratum, group in sorted(groups.items()):
        labels = [int(row["label"]) for row in group]
        scores = [float(row["score"]) for row in group]
        c = _confusion(labels, scores, 0.5)
        output.append(
            {
                "stratum_type": key,
                "stratum": stratum,
                "n": len(group),
                "auroc": _auroc(labels, scores),
                "auprc": _auprc(labels, scores),
                "precision": c["precision"],
                "recall": c["recall"],
                "specificity": c["specificity"],
                "fdr": c["empirical_fdr"],
            }
        )
    return output


def _power_by_regime(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["regime"]), []).append(row)
    output = []
    for regime, group in sorted(groups.items()):
        positives = [row for row in group if int(row["label"]) == 1]
        called = [row for row in positives if _bool(row["called_positive"])]
        output.append({"regime": regime, "n": len(group), "positive_n": len(positives), "called_positive_n": len(called), "power": _safe_div(len(called), len(positives))})
    return output


def _calibration_bins(labels: List[int], scores: List[float], n_bins: int = 10) -> List[Dict[str, Any]]:
    rows = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        idx = [j for j, score in enumerate(scores) if lo <= score < hi or (i == n_bins - 1 and score == 1.0)]
        rows.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": len(idx),
                "mean_score": _safe_div(sum(scores[j] for j in idx), len(idx)),
                "observed_positive_rate": _safe_div(sum(labels[j] for j in idx), len(idx)),
            }
        )
    return rows


def _confusion(labels: List[int], scores: List[float], threshold: float) -> Dict[str, Any]:
    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": _safe_div(tp * tn - fp * fn, mcc_denom),
        "fpr": _safe_div(fp, fp + tn),
        "fnr": _safe_div(fn, fn + tp),
        "empirical_fdr": _safe_div(fp, tp + fp),
        "accuracy": _safe_div(tp + tn, len(labels)),
    }


def _auroc(labels: List[int], scores: List[float]) -> float:
    positives = [(score, label) for label, score in zip(labels, scores) if label == 1]
    negatives = [(score, label) for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for p_score, _ in positives:
        for n_score, _ in negatives:
            if p_score > n_score:
                wins += 1.0
            elif p_score == n_score:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _auprc(labels: List[int], scores: List[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return float("nan")
    ranked = sorted(zip(scores, labels), reverse=True)
    area = 0.0
    tp = 0
    fp = 0
    prev_recall = 0.0
    for _score, label in ranked:
        if label:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / max(1, tp + fp)
        area += precision * (recall - prev_recall)
        prev_recall = recall
    return area


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _render_evaluation_md(payload: Dict[str, Any]) -> str:
    gene = payload["gene_level"]
    return (
        "# Known-Truth Benchmark Evaluation\n\n"
        f"Families: {payload['n_families']}\n\n"
        f"Gene AUROC: {gene['auroc']}\n\n"
        f"Gene AUPRC: {gene['auprc']}\n\n"
        f"OOD abstention rate: {gene['ood_abstention_rate']}\n\n"
        "These are simulation-known-truth metrics, not empirical discovery claims.\n"
    )


def _render_calibration_md(payload: Dict[str, Any]) -> str:
    lines = ["# Known-Truth Calibration Evaluation", "", f"Tests: {payload['n_tests']}", ""]
    for row in payload["fdr_power"]:
        lines.append(f"- q <= {row['q_threshold']}: FDR {row['empirical_fdr']}, power {row['power']}")
    lines.append("")
    lines.append(payload["claim_boundary"])
    lines.append("")
    return "\n".join(lines)

