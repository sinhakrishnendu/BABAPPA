"""Shared helpers for the lightweight known-truth BABAPPA/aBSREL benchmark."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


TSV_NA = "NA"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def read_config(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"config line is not key: value: {raw}")
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip("'\"")
    return data


def config_int(config: Dict[str, str], key: str, default: int) -> int:
    return int(config.get(key, default))


def config_float(config: Dict[str, str], key: str, default: float) -> float:
    return float(config.get(key, default))


def config_bool(config: Dict[str, str], key: str, default: bool = False) -> bool:
    value = config.get(key)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y"}


def resolve_outdir(config: Dict[str, str], override: str | None = None) -> Path:
    outdir = override or config.get("outdir")
    if not outdir:
        raise ValueError("missing outdir in config or --outdir")
    path = Path(outdir)
    if not path.is_absolute():
        path = repo_root() / path
    return path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split("\t")
        rows.append({key: values[i] if i < len(values) else "" for i, key in enumerate(header)})
    return rows


def write_tsv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(fieldnames)]
    for row in rows:
        lines.append("\t".join(str(row.get(field, "")) for field in fieldnames))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_fasta(path: Path) -> Dict[str, str]:
    records: Dict[str, str] = {}
    name: str | None = None
    seq: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records[name] = "".join(seq)
            name = line[1:].split()[0]
            seq = []
        else:
            seq.append(line.upper())
    if name is not None:
        records[name] = "".join(seq)
    return records


def write_fasta(path: Path, records: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for name, seq in records.items():
        lines.append(f">{name}")
        lines.extend(seq[i : i + 80] for i in range(0, len(seq), 80))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_command(argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [(score, label) for label, score in zip(labels, scores) if label == 1]
    negatives = [(score, label) for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = len(positives) * len(negatives)
    for ps, _ in positives:
        for ns, _ in negatives:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / total


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    if not any(labels):
        return None
    ranked = sorted(zip(scores, labels), reverse=True)
    hits = 0
    precisions: List[float] = []
    for i, (_score, label) in enumerate(ranked, start=1):
        if label == 1:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / sum(labels)


def classification_metrics(labels: Sequence[int], calls: Sequence[int]) -> Dict[str, Any]:
    tp = sum(1 for y, c in zip(labels, calls) if y == 1 and c == 1)
    tn = sum(1 for y, c in zip(labels, calls) if y == 0 and c == 0)
    fp = sum(1 for y, c in zip(labels, calls) if y == 0 and c == 1)
    fn = sum(1 for y, c in zip(labels, calls) if y == 1 and c == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    fdr = fp / (fp + tp) if fp + tp else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall_power": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        "fpr": fpr,
        "fnr": fnr,
        "empirical_fdr": fdr,
    }


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", TSV_NA, None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def deterministic_score(seed: int, family_id: str, truth_positive: bool, ood: bool = False) -> float:
    rng = random.Random(f"{seed}:{family_id}:score")
    base = rng.uniform(0.05, 0.35)
    if truth_positive:
        base += rng.uniform(0.25, 0.45)
    if ood:
        base *= 0.35
    return min(base, 0.99)
