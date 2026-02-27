from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .hash_utils import sha256_file, sha256_json
from .io import Alignment, read_fasta


STOP_CODONS = {"TAA", "TAG", "TGA"}
VALID_UNAMBIG_NT = {"A", "C", "G", "T"}
ALIGNMENT_SUFFIXES = {".fa", ".fasta", ".fas", ".fna", ".aln"}
CACHE_ENV = "BABAPPA_DATASET_CACHE"
ORTHOMAM_DEFAULT_BASE_URL = "https://orthomam.mbb.cnrs.fr/orthomam_v12/cds/"
ORTHOMAM_NT_SUBDIR = "omm_filtered_NT_CDS"
ORTHOMAM_TREE_SUBDIR = "trees"
SARS_ORF_WINDOWS: tuple[tuple[str, int, int], ...] = (
    ("sarscov2_orf1ab_chunk1", 266, 13483),
    ("sarscov2_orf1ab_chunk2", 13486, 21552),
    ("sarscov2_spike", 21563, 25384),
    ("sarscov2_n", 28274, 29533),
)


@dataclass
class FetchResult:
    dataset_json: Path
    metadata_tsv: Path
    fetch_manifest_json: Path
    synthetic_fallback: bool
    n_genes: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_url_text(url: str, *, timeout_sec: float) -> str:
    req = Request(url, headers={"User-Agent": "babappa/0.4 dataset-fetch"})
    with urlopen(req, timeout=float(timeout_sec)) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="replace")


def _extract_hrefs(html: str) -> list[str]:
    items = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        y = str(x).strip()
        if not y or y in {"./", "../"}:
            continue
        if y not in seen:
            seen.add(y)
            out.append(y)
    return out


def _download_with_resume(
    *,
    url: str,
    dst: Path,
    retries: int = 3,
    timeout_sec: float = 60.0,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if dst.exists() and dst.stat().st_size > 0:
        return
    last_err: Exception | None = None
    for _attempt in range(max(int(retries), 1)):
        try:
            start = int(tmp.stat().st_size) if tmp.exists() else 0
            req = Request(url, headers={"User-Agent": "babappa/0.4 dataset-fetch"})
            mode = "wb"
            if start > 0:
                req.add_header("Range", f"bytes={start}-")
                mode = "ab"
            with urlopen(req, timeout=float(timeout_sec)) as resp:
                with tmp.open(mode) as out:
                    shutil.copyfileobj(resp, out)
            tmp.replace(dst)
            return
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            last_err = exc
            time.sleep(1.0)
            continue
    raise RuntimeError(f"Download failed after retries: {url} ({last_err})")


def _extract_fasta_from_zip(zip_path: Path, out_fasta: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        cand = [n for n in names if Path(n).suffix.lower() in ALIGNMENT_SUFFIXES]
        if not cand:
            cand = names
        if not cand:
            raise ValueError(f"Zip file has no regular files: {zip_path}")
        pick = sorted(cand)[0]
        data = zf.read(pick)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    out_fasta.write_text(data.decode("utf-8", errors="replace"), encoding="utf-8")


def _normalize_key(text: str) -> str:
    return _sanitize_gene_id(str(text)).lower()


def _choose_matching_tree_url(marker_id: str, tree_urls: dict[str, str]) -> str | None:
    key = _normalize_key(marker_id)
    if key in tree_urls:
        return tree_urls[key]
    for k, url in tree_urls.items():
        if k.startswith(key) or key.startswith(k):
            return url
    return None


def _month_from_date(value: str | None) -> str:
    if value is None:
        return "unknown"
    v = str(value).strip()
    if len(v) >= 7 and v[4] == "-":
        return v[:7]
    return "unknown"


def _metadata_lookup_by_name(meta_df: pd.DataFrame, seq_name: str) -> dict[str, str]:
    if meta_df.empty:
        return {}
    candidates = ["strain", "accession", "name", "seq_name", "sequence_name", "sample", "id"]
    q = str(seq_name).strip()
    for col in candidates:
        if col not in meta_df.columns:
            continue
        sub = meta_df[meta_df[col].astype(str) == q]
        if sub.empty:
            continue
        row = sub.iloc[0].to_dict()
        return {str(k): "" if row[k] is None else str(row[k]) for k in row}
    return {}


def _stratified_subsample_records(
    *,
    records: list[tuple[str, str]],
    metadata_tsv: Path,
    max_samples: int | None,
    seed: int,
    stratify_columns: tuple[str, ...],
) -> list[tuple[str, str]]:
    if max_samples is None or int(max_samples) <= 0 or len(records) <= int(max_samples):
        return list(records)
    df = pd.read_csv(metadata_tsv, sep="\t", dtype=str).fillna("")
    if df.empty:
        records_sorted = sorted(records, key=lambda x: x[0])
        return records_sorted[: int(max_samples)]

    rows: list[dict[str, Any]] = []
    for name, seq in records:
        meta = _metadata_lookup_by_name(df, name)
        month = _month_from_date(meta.get("collection_date") or meta.get("date"))
        country = str(meta.get("country") or meta.get("geo_loc_name") or "unknown").strip() or "unknown"
        stratum_bits: list[str] = []
        for col in stratify_columns:
            c = str(col).strip().lower()
            if c in {"month", "year_month"}:
                stratum_bits.append(month)
            elif c in {"country", "geo", "geography"}:
                stratum_bits.append(country)
            else:
                v = str(meta.get(c, "unknown")).strip() or "unknown"
                stratum_bits.append(v)
        rows.append({"name": name, "seq": seq, "stratum": "|".join(stratum_bits)})
    rdf = pd.DataFrame(rows)
    groups = [g for _, g in rdf.groupby("stratum", sort=True)]
    rng = np.random.default_rng(int(seed))
    cap = int(np.ceil(int(max_samples) / max(len(groups), 1)))
    picks: list[tuple[str, str]] = []
    for g in groups:
        idx = np.arange(len(g), dtype=int)
        rng.shuffle(idx)
        choose = g.iloc[idx[: min(cap, len(g))]]
        for row in choose.itertuples(index=False):
            picks.append((str(row.name), str(row.seq)))
    picks = sorted(picks, key=lambda x: x[0])
    if len(picks) > int(max_samples):
        picks = picks[: int(max_samples)]
    return picks


def _datasets_bin() -> str | None:
    explicit = os.environ.get("BABAPPA_DATASETS_BIN")
    candidates: list[str] = []
    if explicit:
        candidates.append(str(Path(explicit).expanduser()))
    which_path = shutil.which("datasets")
    if which_path:
        candidates.append(str(which_path))
    candidates.append(str((Path.cwd() / ".conda_env" / "bin" / "datasets").resolve()))
    candidates.append(str((Path(__file__).resolve().parents[2] / ".conda_env" / "bin" / "datasets").resolve()))
    for item in candidates:
        p = Path(item)
        if p.exists() and os.access(p, os.X_OK):
            return str(p.resolve())
    return None


def _datasets_cli_available() -> bool:
    return _datasets_bin() is not None


def _parse_date_token(value: str | None) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if "T" in raw:
        raw = raw.split("T", 1)[0]
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(raw, fmt)
            if fmt == "%Y":
                dt = dt.replace(month=1, day=1)
            elif fmt == "%Y-%m":
                dt = dt.replace(day=1)
            return dt
        except Exception:
            continue
    return None


def _parse_date_range(value: str) -> tuple[datetime | None, datetime | None]:
    parts = str(value).split(":", 1)
    if len(parts) == 1:
        return _parse_date_token(parts[0]), None
    return _parse_date_token(parts[0]), _parse_date_token(parts[1])


def _extract_virus_summary_row(rec: dict[str, Any]) -> dict[str, str] | None:
    if not isinstance(rec, dict):
        return None
    acc = str(rec.get("accession") or rec.get("insdcAccession") or "").strip()
    if not acc:
        return None
    date = str(
        rec.get("collectionDate")
        or rec.get("collection_date")
        or rec.get("isolationDate")
        or rec.get("sampleCollectionDate")
        or ""
    ).strip()
    loc = rec.get("geoLocation") or rec.get("location") or rec.get("isolate")
    if isinstance(loc, dict):
        country = str(loc.get("country") or loc.get("name") or "").strip()
    else:
        country = str(loc or "").strip()
    return {"strain": acc, "collection_date": date, "country": country}


def _run_ncbi_datasets_summary(
    *,
    summary_jsonl: Path,
    host: str,
    complete_only: bool,
    released_after: str | None,
    timeout_sec: int,
) -> None:
    datasets_bin = _datasets_bin()
    if datasets_bin is None:
        raise ValueError("datasets CLI is not installed or not discoverable.")
    cmd = [
        datasets_bin,
        "summary",
        "virus",
        "genome",
        "taxon",
        "sars-cov-2",
        "--as-json-lines",
        "--limit",
        "all",
    ]
    if str(host).strip():
        cmd.extend(["--host", str(host).strip()])
    if bool(complete_only):
        cmd.append("--complete-only")
    if released_after:
        cmd.extend(["--released-after", str(released_after)])
    summary_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with summary_jsonl.open("w", encoding="utf-8") as out_handle:
        proc = subprocess.run(
            cmd,
            stdout=out_handle,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(timeout_sec),
        )
    if proc.returncode != 0:
        raise RuntimeError(
            "datasets CLI summary failed: "
            f"rc={proc.returncode}; stderr={(proc.stderr or '').strip()[-500:]}"
        )


def _select_summary_subset(
    *,
    summary_jsonl: Path,
    date_range: str,
    max_samples: int,
    stratify: str,
    seed: int,
) -> pd.DataFrame:
    start_dt, end_dt = _parse_date_range(str(date_range))
    rows: list[dict[str, str]] = []
    for line in summary_jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        row = _extract_virus_summary_row(rec)
        if row is None:
            continue
        col_dt = _parse_date_token(row.get("collection_date"))
        if start_dt is not None and (col_dt is None or col_dt < start_dt):
            continue
        if end_dt is not None and (col_dt is None or col_dt > end_dt):
            continue
        rows.append(row)
    if not rows:
        raise ValueError(
            f"No SARS-CoV-2 accessions matched date_range={date_range}. "
            "Adjust date range or provide imported FASTA/metadata."
        )
    df = pd.DataFrame(rows, columns=["strain", "collection_date", "country"]).fillna("")
    df = df[df["strain"].astype(str).str.len() > 0].drop_duplicates(subset=["strain"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid accession rows after filtering SARS summary output.")

    target_download_n = min(len(df), max(int(max_samples), int(max_samples) * 3))
    if len(df) <= target_download_n:
        return df.sort_values("strain").reset_index(drop=True)

    strat_cols = tuple(x.strip() for x in str(stratify).split(",") if x.strip())
    work = df.copy()
    strata: list[str] = []
    for row in work.itertuples(index=False):
        month = _month_from_date(getattr(row, "collection_date"))
        country = str(getattr(row, "country") or "unknown").strip() or "unknown"
        bits: list[str] = []
        for col in strat_cols:
            c = str(col).lower()
            if c in {"month", "year_month"}:
                bits.append(month)
            elif c in {"country", "geo", "geography"}:
                bits.append(country)
            else:
                bits.append("unknown")
        strata.append("|".join(bits) if bits else "all")
    work["stratum"] = strata
    groups = [g for _, g in work.groupby("stratum", sort=True)]
    rng = np.random.default_rng(int(seed))
    cap = int(np.ceil(target_download_n / max(len(groups), 1)))
    picks: list[pd.DataFrame] = []
    for g in groups:
        idx = np.arange(len(g), dtype=int)
        rng.shuffle(idx)
        picks.append(g.iloc[idx[: min(cap, len(g))]])
    out = pd.concat(picks, ignore_index=True)
    out = out.sort_values("strain").drop_duplicates(subset=["strain"]).reset_index(drop=True)
    if len(out) > target_download_n:
        out = out.iloc[:target_download_n].copy()
    return out[["strain", "collection_date", "country"]]


def _extract_ncbi_dataset_zip(
    *,
    zip_path: Path,
    out_fasta: Path,
    out_metadata_tsv: Path,
) -> None:
    tmp_dir = out_metadata_tsv.parent / "_ncbi_unpack_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    fasta_candidates = [
        p
        for p in sorted(tmp_dir.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in {".fna", ".fa", ".fasta"}
        and "cds" not in p.name.lower()
        and "protein" not in p.name.lower()
    ]
    if not fasta_candidates:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError("Could not find genome FASTA in datasets zip package.")

    seq_lines: list[str] = []
    for fp in fasta_candidates:
        seq_lines.extend(fp.read_text(encoding="utf-8", errors="replace").splitlines())
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    out_fasta.write_text("\n".join(seq_lines).strip() + "\n", encoding="utf-8")

    meta_rows: list[dict[str, str]] = []
    jsonl_candidates = [p for p in sorted(tmp_dir.rglob("*.jsonl")) if p.is_file()]
    for jp in jsonl_candidates:
        try:
            for line in jp.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    continue
                acc = str(rec.get("accession") or rec.get("insdcAccession") or "").strip()
                date = str(rec.get("collectionDate") or rec.get("collection_date") or "").strip()
                loc = rec.get("geoLocation") or rec.get("location") or rec.get("isolate")
                if isinstance(loc, dict):
                    country = str(loc.get("country") or loc.get("name") or "").strip()
                else:
                    country = str(loc or "").strip()
                if acc:
                    meta_rows.append({"strain": acc, "collection_date": date, "country": country})
        except Exception:
            continue
    if not meta_rows:
        # Ensure metadata file always exists even when package lacks report fields.
        out_metadata_tsv.write_text("strain\tcollection_date\tcountry\n", encoding="utf-8")
    else:
        pd.DataFrame(meta_rows).drop_duplicates().to_csv(out_metadata_tsv, sep="\t", index=False)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_ncbi_datasets_download(
    *,
    zip_out: Path,
    host: str,
    include_cds: bool,
    complete_only: bool,
    timeout_sec: int,
    accession_input: Path | None = None,
) -> None:
    datasets_bin = _datasets_bin()
    if datasets_bin is None:
        raise ValueError("datasets CLI is not installed or not discoverable.")
    if accession_input is not None:
        cmd = [
            datasets_bin,
            "download",
            "virus",
            "genome",
            "accession",
            "--inputfile",
            str(accession_input),
            "--filename",
            str(zip_out),
        ]
    else:
        cmd = [datasets_bin, "download", "virus", "genome", "taxon", "sars-cov-2", "--filename", str(zip_out)]
    if include_cds:
        cmd.extend(["--include", "cds", "--include", "genome", "--include", "protein"])
    else:
        cmd.extend(["--include", "genome"])
    if str(host).strip():
        cmd.extend(["--host", str(host).strip()])
    if bool(complete_only):
        cmd.append("--complete-only")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=int(timeout_sec),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "datasets CLI download failed: "
            f"rc={proc.returncode}; stderr={(proc.stderr or '').strip()[-500:]}"
        )


def _resolve_cache_dir() -> Path:
    candidates: list[Path] = []
    env = os.environ.get(CACHE_ENV)
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path.home() / ".cache" / "babappa" / "datasets")
    candidates.append(Path(tempfile.gettempdir()) / "babappa_dataset_cache")

    for root in candidates:
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / ".probe_write"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return root
        except Exception:
            continue
    raise OSError("Unable to initialize writable dataset cache directory.")


def dataset_cache_show() -> dict[str, Any]:
    cache_root = _resolve_cache_dir()
    entries: list[dict[str, Any]] = []
    for child in sorted(cache_root.iterdir()):
        if not child.is_dir():
            continue
        files = [p for p in child.rglob("*") if p.is_file()]
        size_bytes = int(sum(p.stat().st_size for p in files))
        entries.append(
            {
                "name": child.name,
                "path": str(child.resolve()),
                "n_files": len(files),
                "size_bytes": size_bytes,
            }
        )
    return {
        "cache_root": str(cache_root.resolve()),
        "entries": entries,
        "n_entries": len(entries),
    }


def dataset_cache_clear(name: str) -> dict[str, Any]:
    raw = str(name).strip()
    if not raw:
        raise ValueError("Cache name must be non-empty.")
    safe = raw.replace("\\", "/").strip("/")
    if not safe or ".." in safe.split("/"):
        raise ValueError(f"Invalid cache name: {name}")

    cache_root = _resolve_cache_dir().resolve()
    target = (cache_root / safe).resolve()
    try:
        target.relative_to(cache_root)
    except ValueError as exc:
        raise ValueError(f"Cache path escapes cache root: {target}") from exc

    if not target.exists():
        return {"removed": False, "path": str(target), "name": safe}
    if target == cache_root:
        raise ValueError("Refusing to clear the whole cache root.")
    shutil.rmtree(target)
    return {"removed": True, "path": str(target), "name": safe}


def _iter_alignment_files(path: Path) -> list[Path]:
    files = [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in ALIGNMENT_SUFFIXES]
    if not files:
        raise ValueError(f"No alignment files found under: {path}")
    return files


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_alignment(path: Path, aln: Alignment) -> None:
    lines: list[str] = []
    for name, seq in zip(aln.names, aln.sequences):
        lines.append(f">{name}")
        lines.append(seq)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _star_tree(names: list[str], branch_length: float = 0.1) -> str:
    leaves = [f"{name}:{branch_length}" for name in names]
    return f"({','.join(leaves)});"


def _balanced_tree(names: list[str], branch_length: float = 0.1) -> str:
    def _build(items: list[str]) -> str:
        if len(items) == 1:
            return f"{items[0]}:{branch_length}"
        mid = len(items) // 2
        left = _build(items[:mid])
        right = _build(items[mid:])
        return f"({left},{right}):{branch_length}"

    if not names:
        raise ValueError("Cannot build balanced tree without taxa.")
    return _build(list(names)) + ";"


def _slice_1based(seq: str, start_1based: int, end_1based: int) -> str:
    if start_1based < 1:
        raise ValueError(f"start_1based must be >=1, got {start_1based}")
    if end_1based < start_1based:
        raise ValueError(f"end_1based must be >= start_1based, got {end_1based} < {start_1based}")
    return seq[start_1based - 1 : end_1based]


def _default_neutral_model_payload(tree_path: Path, *, genetic_code_table: str = "standard") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_family": "GY94",
        "genetic_code_table": genetic_code_table,
        "tree_file": str(tree_path.resolve()),
        "kappa": 2.0,
        "omega": 1.0,
        "codon_frequencies_method": "F3x4",
        "frozen_values": True,
    }


def _ambiguous_codon_fraction(seq: str) -> float:
    seq_u = seq.upper().replace("U", "T")
    usable_len = len(seq_u) - (len(seq_u) % 3)
    if usable_len <= 0:
        return 1.0
    total = usable_len // 3
    ambiguous = 0
    for start in range(0, usable_len, 3):
        codon = seq_u[start : start + 3]
        if any(base not in VALID_UNAMBIG_NT for base in codon):
            ambiguous += 1
    return ambiguous / max(total, 1)


def _has_internal_stop(seq: str) -> bool:
    seq_u = seq.upper().replace("U", "T")
    usable_len = len(seq_u) - (len(seq_u) % 3)
    if usable_len < 6:
        return False
    last_start = usable_len - 3
    for start in range(0, last_start, 3):
        codon = seq_u[start : start + 3]
        if any(base not in VALID_UNAMBIG_NT for base in codon):
            continue
        if codon in STOP_CODONS:
            return True
    return False


def _filter_alignment(
    aln: Alignment,
    *,
    max_ambiguous_fraction: float,
    enforce_multiple_of_3: bool,
    remove_internal_stops: bool,
) -> tuple[Alignment, dict[str, int]]:
    keep_names: list[str] = []
    keep_seqs: list[str] = []
    dropped_ambiguous = 0
    dropped_stop = 0
    dropped_frame = 0

    for name, seq in zip(aln.names, aln.sequences):
        s = seq.upper().replace("U", "T")
        if enforce_multiple_of_3 and (len(s) % 3 != 0):
            dropped_frame += 1
            continue
        if remove_internal_stops and _has_internal_stop(s):
            dropped_stop += 1
            continue
        if _ambiguous_codon_fraction(s) > float(max_ambiguous_fraction):
            dropped_ambiguous += 1
            continue
        keep_names.append(name)
        keep_seqs.append(s)

    if not keep_names:
        raise ValueError("All sequences were dropped by QC filters.")
    out = Alignment(names=tuple(keep_names), sequences=tuple(keep_seqs))
    return out, {
        "n_input": aln.n_sequences,
        "n_retained": out.n_sequences,
        "dropped_ambiguous": dropped_ambiguous,
        "dropped_internal_stop": dropped_stop,
        "dropped_frame": dropped_frame,
    }


def _sanitize_gene_id(raw: str) -> str:
    out = []
    for ch in raw:
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("._")
    return cleaned or "gene"


def _load_provenance(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Provenance file must be JSON object: {p}")
    return payload


def _write_dataset_bundle(
    *,
    outdir: Path,
    track: str,
    source: str,
    dataset_name: str,
    release: str,
    genes: list[dict[str, Any]],
    tree_path: Path,
    neutral_model_payload: dict[str, Any],
    provenance: dict[str, Any],
    fetch_params: dict[str, Any],
    synthetic_fallback: bool,
    metadata_rows: list[dict[str, Any]],
    extra_metadata: dict[str, Any] | None = None,
) -> FetchResult:
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    metadata_tsv = outdir / "metadata.tsv"
    pd.DataFrame(metadata_rows).to_csv(metadata_tsv, sep="\t", index=False)

    neutral_path = outdir / "neutral_model.json"
    _write_json(neutral_path, neutral_model_payload)
    neutral_hash = sha256_file(neutral_path)

    dataset_payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset_name": dataset_name,
        "tree_path": str(tree_path.resolve()),
        "neutral_model_json": str(neutral_path.resolve()),
        "genes": genes,
        "metadata": {
            "source": source,
            "track": track,
            "release": release,
            "synthetic_fallback": bool(synthetic_fallback),
        },
    }
    if extra_metadata:
        dataset_payload["metadata"].update(extra_metadata)
    dataset_json = outdir / "dataset.json"
    _write_json(dataset_json, dataset_payload)

    fetch_manifest = {
        "schema_version": 1,
        "dataset_name": dataset_name,
        "source": source,
        "track": track,
        "release": release,
        "retrieved_at_utc": _utc_now_iso(),
        "synthetic_fallback": bool(synthetic_fallback),
        "fetch_parameters": fetch_params,
        "provenance": provenance,
        "paths": {
            "dataset_json": str(dataset_json.resolve()),
            "metadata_tsv": str(metadata_tsv.resolve()),
            "tree_path": str(tree_path.resolve()),
            "neutral_model_json": str(neutral_path.resolve()),
        },
        "hashes": {
            "dataset_json_sha256": sha256_file(dataset_json),
            "metadata_tsv_sha256": sha256_file(metadata_tsv),
            "tree_sha256": sha256_file(tree_path),
            "neutral_model_sha256": neutral_hash,
            "m0_hash": sha256_json(neutral_model_payload),
        },
        "n_genes": int(len(genes)),
    }
    fetch_manifest_json = outdir / "fetch_manifest.json"
    _write_json(fetch_manifest_json, fetch_manifest)
    return FetchResult(
        dataset_json=dataset_json,
        metadata_tsv=metadata_tsv,
        fetch_manifest_json=fetch_manifest_json,
        synthetic_fallback=bool(synthetic_fallback),
        n_genes=len(genes),
    )


def import_ortholog_dataset(
    *,
    source_dir: str | Path,
    outdir: str | Path,
    source_name: str,
    release: str,
    species_set: str,
    min_length_codons: int,
    max_genes: int,
    tree_file: str | Path | None = None,
    per_gene_tree_dir: str | Path | None = None,
    provenance_json: str | Path | None = None,
    seed: int = 42,
) -> FetchResult:
    src = Path(source_dir).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")
    out = Path(outdir).resolve()
    genes_dir = out / "genes"
    trees_dir = out / "trees"
    genes_dir.mkdir(parents=True, exist_ok=True)
    trees_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))

    files = _iter_alignment_files(src)
    rng.shuffle(files)

    selected_raw: list[dict[str, Any]] = []
    seen_gene_ids: set[str] = set()
    for f in files:
        try:
            aln = read_fasta(f)
        except Exception:
            continue
        if aln.length < int(min_length_codons) * 3:
            continue
        base_gene_id = _sanitize_gene_id(f.stem)
        gene_id = base_gene_id
        suffix = 2
        while gene_id in seen_gene_ids:
            gene_id = f"{base_gene_id}_{suffix}"
            suffix += 1
        seen_gene_ids.add(gene_id)
        selected_raw.append(
            {
                "gene_id": gene_id,
                "source_marker_id": str(f.stem),
                "source_file": str(f),
                "alignment": aln,
            }
        )
        if len(selected_raw) >= int(max_genes):
            break
    if not selected_raw:
        raise ValueError("No ortholog alignments selected after filtering.")

    target_taxa_n = 8 if str(species_set).lower() == "small8" else 16
    taxon_counts: Counter[str] = Counter()
    for row in selected_raw:
        taxon_counts.update(set(str(x) for x in row["alignment"].names))
    if not taxon_counts:
        raise ValueError("No taxa found in selected ortholog genes after filtering.")
    ranked_taxa = sorted(taxon_counts.items(), key=lambda x: (-int(x[1]), str(x[0])))
    chosen_n = min(int(target_taxa_n), len(ranked_taxa))
    chosen_taxa = [name for name, _ in ranked_taxa[:chosen_n]]

    filtered_raw = [
        row
        for row in selected_raw
        if set(chosen_taxa).issubset(set(str(x) for x in row["alignment"].names))
    ]
    while not filtered_raw and chosen_n > 1:
        chosen_n -= 1
        chosen_taxa = [name for name, _ in ranked_taxa[:chosen_n]]
        filtered_raw = [
            row
            for row in selected_raw
            if set(chosen_taxa).issubset(set(str(x) for x in row["alignment"].names))
        ]
    if not filtered_raw:
        raise ValueError("Unable to find a shared taxa core for ortholog import.")
    selected_raw = filtered_raw

    selected: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for row in selected_raw:
        gene_id = str(row["gene_id"])
        aln = row["alignment"]
        taxa_to_seq = {str(name): str(seq) for name, seq in zip(aln.names, aln.sequences)}
        pruned_names = [name for name in chosen_taxa if name in taxa_to_seq]
        pruned_seqs = [taxa_to_seq[name] for name in pruned_names]
        pruned = Alignment(names=tuple(pruned_names), sequences=tuple(pruned_seqs))
        dst = genes_dir / f"{gene_id}.fna"
        _write_alignment(dst, pruned)
        missing_frac = float(
            np.mean([_ambiguous_codon_fraction(seq) for seq in pruned.sequences]) if pruned.sequences else 0.0
        )
        selected.append(
            {
                "gene_id": gene_id,
                "alignment_path": str(dst.resolve()),
                "source_marker_id": str(row["source_marker_id"]),
            }
        )
        metadata_rows.append(
            {
                "gene_id": gene_id,
                "alignment_path": str(dst.resolve()),
                "length_nt": int(pruned.length),
                "n_taxa": int(pruned.n_sequences),
                "n_taxa_raw": int(aln.n_sequences),
                "n_taxa_dropped_by_intersection": int(aln.n_sequences - pruned.n_sequences),
                "missingness_fraction": missing_frac,
                "source_file": str(row["source_file"]),
                "sha256": sha256_file(dst),
            }
        )

    if tree_file is not None:
        tree_src = Path(tree_file).resolve()
        if not tree_src.exists():
            raise FileNotFoundError(f"tree_file not found: {tree_src}")
        tree_path = out / "tree.nwk"
        _copy_file(tree_src, tree_path)
        tree_policy = "fixed_species_tree"
    else:
        tree_path = out / "tree.nwk"
        tree_path.write_text(_balanced_tree(chosen_taxa) + "\n", encoding="utf-8")
        tree_policy = "fixed_species_tree_taxa_core_balanced"

    # Emit explicit per-gene tree artifacts so each unit has a stable tree hash/path.
    for row in selected:
        gene_id = str(row["gene_id"])
        gene_tree = trees_dir / f"{gene_id}.nwk"
        _copy_file(tree_path, gene_tree)
        row["tree_path"] = str(gene_tree.resolve())
    meta_by_gene = {str(r["gene_id"]): r for r in metadata_rows}
    for gene_id, row in meta_by_gene.items():
        gene_tree = trees_dir / f"{gene_id}.nwk"
        row["tree_path"] = str(gene_tree.resolve())
        row["tree_sha256"] = sha256_file(gene_tree)

    neutral_payload = _default_neutral_model_payload(tree_path)
    provenance = _load_provenance(provenance_json)
    fetch_params = {
        "source_dir": str(src),
        "source_name": source_name,
        "release": release,
        "species_set": species_set,
        "min_length_codons": int(min_length_codons),
        "max_genes": int(max_genes),
        "seed": int(seed),
        "tree_policy": tree_policy,
        "taxa_harmonization": "frequency_core",
        "target_taxa_n": int(target_taxa_n),
        "n_taxa_core": int(len(chosen_taxa)),
        "n_genes_before_taxa_core_filter": int(len(seen_gene_ids)),
        "n_genes_after_taxa_core_filter": int(len(selected)),
        "taxa_core": list(chosen_taxa),
    }
    return _write_dataset_bundle(
        outdir=out,
        track="ortholog",
        source=source_name,
        dataset_name=f"ortholog_{species_set}",
        release=release,
        genes=selected,
        tree_path=tree_path,
        neutral_model_payload=neutral_payload,
        provenance=provenance,
        fetch_params=fetch_params,
        synthetic_fallback=False,
        metadata_rows=metadata_rows,
        extra_metadata={
            "species_set": species_set,
            "tree_policy": tree_policy,
            "taxa_harmonization": "frequency_core",
            "target_taxa_n": int(target_taxa_n),
            "n_taxa_core": int(len(chosen_taxa)),
            "n_genes_before_taxa_core_filter": int(len(seen_gene_ids)),
            "n_genes_after_taxa_core_filter": int(len(selected)),
            "taxa_core": list(chosen_taxa),
        },
    )


def import_hiv_dataset(
    *,
    alignment: str | Path,
    outdir: str | Path,
    provenance_json: str | Path | None,
    recombination_policy: str,
    alignment_id: str | None = None,
    download_steps_text: str | None = None,
    gene: str = "env",
    subtype: str = "B",
    tree_file: str | Path | None = None,
    max_ambiguous_fraction: float = 0.05,
    enforce_multiple_of_3: bool = True,
    remove_internal_stops: bool = True,
) -> FetchResult:
    aln_path = Path(alignment).resolve()
    if not aln_path.exists():
        raise FileNotFoundError(f"Alignment not found: {aln_path}")
    out = Path(outdir).resolve()
    genes_dir = out / "genes"
    genes_dir.mkdir(parents=True, exist_ok=True)

    aln_raw = read_fasta(aln_path)
    aln_filtered, qc = _filter_alignment(
        aln_raw,
        max_ambiguous_fraction=float(max_ambiguous_fraction),
        enforce_multiple_of_3=bool(enforce_multiple_of_3),
        remove_internal_stops=bool(remove_internal_stops),
    )
    gene_id = _sanitize_gene_id(f"hiv_{gene}_{subtype}")
    gene_path = genes_dir / f"{gene_id}.fna"
    _write_alignment(gene_path, aln_filtered)

    if tree_file is not None:
        tree_src = Path(tree_file).resolve()
        if not tree_src.exists():
            raise FileNotFoundError(f"tree_file not found: {tree_src}")
        tree_path = out / "tree.nwk"
        _copy_file(tree_src, tree_path)
        tree_policy = "frozen_curated_tree"
    else:
        tree_path = out / "tree.nwk"
        tree_path.write_text(_star_tree(sorted(aln_filtered.names)) + "\n", encoding="utf-8")
        tree_policy = "frozen_star_tree_from_filtered_alignment"

    neutral_payload = _default_neutral_model_payload(tree_path)
    provenance = _load_provenance(provenance_json)
    metadata_rows = [
        {
            "gene_id": gene_id,
            "alignment_path": str(gene_path.resolve()),
            "length_nt": int(aln_filtered.length),
            "n_taxa": int(aln_filtered.n_sequences),
            "n_input_sequences": int(qc["n_input"]),
            "n_retained_sequences": int(qc["n_retained"]),
            "dropped_ambiguous": int(qc["dropped_ambiguous"]),
            "dropped_internal_stop": int(qc["dropped_internal_stop"]),
            "dropped_frame": int(qc["dropped_frame"]),
            "missingness_fraction": float(
                np.mean([_ambiguous_codon_fraction(s) for s in aln_filtered.sequences])
            ),
            "sha256": sha256_file(gene_path),
        }
    ]
    genes = [{"gene_id": gene_id, "alignment_path": str(gene_path.resolve())}]
    fetch_params = {
        "source": "hiv_import",
        "gene": gene,
        "subtype": subtype,
        "recombination_policy": recombination_policy,
        "alignment_id": None if alignment_id is None else str(alignment_id),
        "download_steps_text": None if download_steps_text is None else str(download_steps_text),
        "max_ambiguous_fraction": float(max_ambiguous_fraction),
        "enforce_multiple_of_3": bool(enforce_multiple_of_3),
        "remove_internal_stops": bool(remove_internal_stops),
        "tree_policy": tree_policy,
    }
    return _write_dataset_bundle(
        outdir=out,
        track="viral",
        source="hiv_import",
        dataset_name=f"hiv_{gene}_{subtype}",
        release="user_import",
        genes=genes,
        tree_path=tree_path,
        neutral_model_payload=neutral_payload,
        provenance=provenance,
        fetch_params=fetch_params,
        synthetic_fallback=False,
        metadata_rows=metadata_rows,
        extra_metadata={
            "virus": "HIV-1",
            "gene": gene,
            "subtype": subtype,
            "recombination_policy": recombination_policy,
            "tree_policy": tree_policy,
            "alignment_id": None if alignment_id is None else str(alignment_id),
        },
    )


def import_sarscov2_dataset(
    *,
    fasta: str | Path,
    metadata_tsv: str | Path,
    outdir: str | Path,
    provenance_json: str | Path | None,
    tree_file: str | Path | None = None,
    max_n_fraction: float = 0.01,
    max_samples: int | None = None,
    stratify: str | None = "month,country",
    stratify_seed: int = 1,
    enforce_multiple_of_3: bool = True,
    remove_internal_stops: bool = True,
) -> FetchResult:
    fasta_path = Path(fasta).resolve()
    meta_path = Path(metadata_tsv).resolve()
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata TSV not found: {meta_path}")

    out = Path(outdir).resolve()
    genes_dir = out / "genes"
    genes_dir.mkdir(parents=True, exist_ok=True)

    aln_raw = read_fasta(fasta_path)
    records = [(str(name), str(seq)) for name, seq in zip(aln_raw.names, aln_raw.sequences)]
    records.sort(key=lambda row: row[0])
    n_input_total = len(records)
    strat_cols = tuple(
        x.strip()
        for x in (str(stratify).split(",") if stratify is not None else [])
        if x.strip()
    )
    if max_samples is not None and int(max_samples) > 0:
        records = _stratified_subsample_records(
            records=records,
            metadata_tsv=meta_path,
            max_samples=int(max_samples),
            seed=int(stratify_seed),
            stratify_columns=strat_cols,
        )
    n_after_subsample = len(records)

    keep_names: list[str] = []
    keep_seqs: list[str] = []
    dropped_n = 0
    dropped_short = 0
    dropped_orf_frame = 0
    dropped_orf_stop = 0
    for name, seq in records:
        s = seq.upper().replace("U", "T")
        n_frac = float(s.count("N")) / max(len(s), 1)
        if n_frac > float(max_n_fraction):
            dropped_n += 1
            continue
        failed = False
        for _, start_1, end_1 in SARS_ORF_WINDOWS:
            if len(s) < int(end_1):
                dropped_short += 1
                failed = True
                break
            seg = _slice_1based(s, int(start_1), int(end_1))
            if bool(enforce_multiple_of_3) and (len(seg) % 3 != 0):
                dropped_orf_frame += 1
                failed = True
                break
            if bool(remove_internal_stops) and _has_internal_stop(seg):
                dropped_orf_stop += 1
                failed = True
                break
        if failed:
            continue
        keep_names.append(name)
        keep_seqs.append(s)

    if not keep_names:
        raise ValueError("All SARS-CoV-2 sequences were filtered by QC policy.")
    aln_filtered = Alignment(names=tuple(keep_names), sequences=tuple(keep_seqs))

    genes: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for gene_id, start_1, end_1 in SARS_ORF_WINDOWS:
        gpath = genes_dir / f"{gene_id}.fna"
        seg_aln = Alignment(
            names=tuple(keep_names),
            sequences=tuple(_slice_1based(seq, int(start_1), int(end_1)) for seq in keep_seqs),
        )
        _write_alignment(gpath, seg_aln)
        genes.append({"gene_id": gene_id, "alignment_path": str(gpath.resolve())})
        metadata_rows.append(
            {
                "gene_id": gene_id,
                "alignment_path": str(gpath.resolve()),
                "length_nt": int(seg_aln.length),
                "n_taxa": int(seg_aln.n_sequences),
                "n_input_sequences": int(n_input_total),
                "n_after_subsample": int(n_after_subsample),
                "n_retained_sequences": int(seg_aln.n_sequences),
                "dropped_high_n_fraction": int(dropped_n),
                "dropped_short_sequence": int(dropped_short),
                "dropped_orf_frame": int(dropped_orf_frame),
                "dropped_orf_internal_stop": int(dropped_orf_stop),
                "max_n_fraction": float(max_n_fraction),
                "enforce_multiple_of_3": bool(enforce_multiple_of_3),
                "remove_internal_stops": bool(remove_internal_stops),
                "orf_start_1based": int(start_1),
                "orf_end_1based": int(end_1),
                "missingness_fraction": float(np.mean([_ambiguous_codon_fraction(s) for s in seg_aln.sequences])),
                "sha256": sha256_file(gpath),
                "source_metadata_tsv": str(meta_path),
            }
        )

    if tree_file is not None:
        tree_src = Path(tree_file).resolve()
        if not tree_src.exists():
            raise FileNotFoundError(f"tree_file not found: {tree_src}")
        tree_path = out / "tree.nwk"
        _copy_file(tree_src, tree_path)
        tree_policy = "frozen_curated_tree"
    else:
        tree_path = out / "tree.nwk"
        tree_path.write_text(_star_tree(sorted(aln_filtered.names)) + "\n", encoding="utf-8")
        tree_policy = "frozen_star_tree_from_filtered_alignment"

    neutral_payload = _default_neutral_model_payload(tree_path)
    provenance = _load_provenance(provenance_json)
    fetch_params = {
        "source": "sarscov2_import",
        "max_n_fraction": float(max_n_fraction),
        "max_samples": None if max_samples is None else int(max_samples),
        "stratify": ",".join(strat_cols),
        "stratify_seed": int(stratify_seed),
        "n_input_sequences": int(n_input_total),
        "n_after_subsample": int(n_after_subsample),
        "n_retained_sequences": int(len(keep_names)),
        "dropped_high_n_fraction": int(dropped_n),
        "dropped_short_sequence": int(dropped_short),
        "dropped_orf_frame": int(dropped_orf_frame),
        "dropped_orf_internal_stop": int(dropped_orf_stop),
        "orf_windows": [
            {"gene_id": gid, "start_1based": int(s), "end_1based": int(e)}
            for gid, s, e in SARS_ORF_WINDOWS
        ],
        "tree_policy": tree_policy,
    }
    return _write_dataset_bundle(
        outdir=out,
        track="viral",
        source="sarscov2_import",
        dataset_name="sarscov2_2020",
        release="user_import",
        genes=genes,
        tree_path=tree_path,
        neutral_model_payload=neutral_payload,
        provenance=provenance,
        fetch_params=fetch_params,
        synthetic_fallback=False,
        metadata_rows=metadata_rows,
        extra_metadata={
            "virus": "SARS-CoV-2",
            "tree_policy": tree_policy,
            "source_metadata_tsv": str(meta_path),
            "orf_units": [x[0] for x in SARS_ORF_WINDOWS],
            "subsampling_policy": "deterministic_stratified_subsample",
        },
    )


def fetch_orthomam_dataset(
    *,
    release: str,
    species_set: str,
    min_length_codons: int,
    max_genes: int,
    outdir: str | Path,
    seed: int,
    tree_file: str | Path | None = None,
    mode: str = "cache",
    base_url: str | None = None,
    n_markers: int | None = None,
    marker_ids_file: str | Path | None = None,
    retries: int = 3,
    timeout_sec: float = 60.0,
) -> FetchResult:
    cache_root = _resolve_cache_dir()
    mode_l = str(mode).strip().lower()
    target_n = int(n_markers) if n_markers is not None else int(max_genes)
    if mode_l in {"cache", "cached"}:
        source_dir = cache_root / "orthomam" / str(release) / str(species_set)
        if not source_dir.exists():
            raise ValueError(
                "Ortholog fetch cache not found. Place curated alignments under "
                f"{source_dir} (or use --mode remote with internet access)."
            )
        return import_ortholog_dataset(
            source_dir=source_dir,
            outdir=outdir,
            source_name="orthomam",
            release=release,
            species_set=species_set,
            min_length_codons=min_length_codons,
            max_genes=target_n,
            tree_file=tree_file,
            provenance_json=None,
            seed=seed,
        )

    rel = str(release).strip()
    if base_url is None:
        if rel.lower().startswith("v"):
            digits = rel[1:]
            base = f"https://orthomam.mbb.cnrs.fr/orthomam_v{digits}/cds/"
        else:
            base = ORTHOMAM_DEFAULT_BASE_URL
    else:
        base = str(base_url).strip()
    if not base.endswith("/"):
        base += "/"
    nt_url = urljoin(base, ORTHOMAM_NT_SUBDIR + "/")
    trees_url = urljoin(base, ORTHOMAM_TREE_SUBDIR + "/")
    nt_html = _read_url_text(nt_url, timeout_sec=float(timeout_sec))
    tree_html = _read_url_text(trees_url, timeout_sec=float(timeout_sec))
    nt_links = [x for x in _extract_hrefs(nt_html) if str(x).lower().endswith(".zip")]
    if not nt_links:
        raise ValueError(f"No zip markers found in OrthoMaM index: {nt_url}")
    tree_links = [
        x
        for x in _extract_hrefs(tree_html)
        if Path(str(x)).suffix.lower() in {".nwk", ".tree", ".tre", ".newick"}
    ]
    tree_map = {_normalize_key(Path(str(x)).stem): urljoin(trees_url, str(x)) for x in tree_links}

    requested_ids: list[str] = []
    if marker_ids_file is not None:
        p = Path(marker_ids_file).resolve()
        if not p.exists():
            raise FileNotFoundError(f"marker_ids_file not found: {p}")
        requested_ids = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

    all_markers = sorted((Path(str(link)).stem, str(link)) for link in nt_links)
    selected: list[tuple[str, str]] = []
    if requested_ids:
        want = {_normalize_key(x) for x in requested_ids}
        for marker_id, link in all_markers:
            if _normalize_key(marker_id) in want:
                selected.append((marker_id, link))
    else:
        if mode_l == "small" and n_markers is None:
            target_n = min(target_n, 16)
        selected = all_markers[: int(target_n)]
    if not selected:
        raise ValueError("No OrthoMaM markers selected after applying inclusion rules.")

    cache_base = cache_root / "orthomam" / rel
    cache_nt = cache_base / ORTHOMAM_NT_SUBDIR
    cache_trees = cache_base / ORTHOMAM_TREE_SUBDIR
    extracted = cache_base / f"extracted_seed{int(seed)}_n{int(len(selected))}"
    if extracted.exists():
        shutil.rmtree(extracted)
    extracted.mkdir(parents=True, exist_ok=True)
    cache_nt.mkdir(parents=True, exist_ok=True)
    cache_trees.mkdir(parents=True, exist_ok=True)

    download_rows: list[dict[str, Any]] = []
    for marker_id, link in selected:
        aln_url = urljoin(nt_url, link)
        zip_path = cache_nt / Path(link).name
        _download_with_resume(url=aln_url, dst=zip_path, retries=int(retries), timeout_sec=float(timeout_sec))
        aln_out = extracted / f"{_sanitize_gene_id(marker_id)}.fna"
        _extract_fasta_from_zip(zip_path, aln_out)
        download_rows.append(
            {
                "kind": "alignment_zip",
                "marker_id": marker_id,
                "url": aln_url,
                "path": str(zip_path.resolve()),
                "sha256": sha256_file(zip_path),
            }
        )
        turl = _choose_matching_tree_url(marker_id, tree_map)
        if turl is not None:
            tname = Path(turl).name
            tpath = cache_trees / tname
            _download_with_resume(url=turl, dst=tpath, retries=int(retries), timeout_sec=float(timeout_sec))
            download_rows.append(
                {
                    "kind": "tree",
                    "marker_id": marker_id,
                    "url": turl,
                    "path": str(tpath.resolve()),
                    "sha256": sha256_file(tpath),
                }
            )

    res = import_ortholog_dataset(
        source_dir=extracted,
        outdir=outdir,
        source_name="OrthoMaM v12",
        release=rel,
        species_set=species_set,
        min_length_codons=min_length_codons,
        max_genes=target_n,
        tree_file=tree_file,
        per_gene_tree_dir=cache_trees,
        provenance_json=None,
        seed=seed,
    )

    dataset_payload = json.loads(res.dataset_json.read_text(encoding="utf-8"))
    dataset_payload.setdefault("metadata", {})
    dataset_payload["metadata"].update(
        {
            "source": "OrthoMaM v12",
            "base_url": base,
            "subdirs_used": [ORTHOMAM_NT_SUBDIR, ORTHOMAM_TREE_SUBDIR],
            "inclusion_rule": {
                "mode": mode_l,
                "seed": int(seed),
                "n_markers": int(len(selected)),
                "marker_ids": [x[0] for x in selected],
            },
        }
    )
    _write_json(res.dataset_json, dataset_payload)

    fetch_payload = json.loads(res.fetch_manifest_json.read_text(encoding="utf-8"))
    fetch_payload["source"] = "OrthoMaM v12"
    fetch_payload["base_url"] = base
    fetch_payload["subdirs_used"] = [ORTHOMAM_NT_SUBDIR, ORTHOMAM_TREE_SUBDIR]
    fetch_payload["download_manifest"] = download_rows
    fetch_payload.setdefault("fetch_parameters", {})
    fetch_payload["fetch_parameters"].update(
        {
            "mode": mode_l,
            "seed": int(seed),
            "n_markers": int(len(selected)),
            "marker_ids": [x[0] for x in selected],
            "cache_base": str(cache_base.resolve()),
            "tree_file_override": None if tree_file is None else str(Path(tree_file).resolve()),
        }
    )
    _write_json(res.fetch_manifest_json, fetch_payload)
    return res


def fetch_hiv_dataset(
    *,
    source: str,
    gene: str,
    subtype: str,
    outdir: str | Path,
    recombination_policy: str,
    tree_file: str | Path | None = None,
    max_ambiguous_fraction: float = 0.05,
    alignment_id: str | None = None,
) -> FetchResult:
    cache_root = _resolve_cache_dir()
    aln = (
        cache_root
        / "hiv"
        / f"source_{str(source).lower()}"
        / f"gene_{str(gene).lower()}"
        / f"subtype_{str(subtype).upper()}"
        / "alignment.fasta"
    )
    provenance = aln.with_name("provenance.json")
    if not aln.exists():
        raise ValueError(
            "Automated LANL fetch is not available in this environment. "
            "Use `babappa dataset import hiv --alignment ... --alignment-id ...`."
        )
    return import_hiv_dataset(
        alignment=aln,
        outdir=outdir,
        provenance_json=provenance if provenance.exists() else None,
        recombination_policy=recombination_policy,
        alignment_id=alignment_id,
        gene=gene,
        subtype=subtype,
        tree_file=tree_file,
        max_ambiguous_fraction=max_ambiguous_fraction,
    )


def fetch_sarscov2_dataset(
    *,
    source: str,
    date_range: str,
    max_samples: int,
    outdir: str | Path,
    tree_file: str | Path | None = None,
    max_n_fraction: float = 0.01,
    host: str = "human",
    complete_only: bool = True,
    include_cds: bool = True,
    stratify: str = "month,country",
    seed: int = 1,
    datasets_timeout_sec: int = 1800,
) -> FetchResult:
    cache_root = _resolve_cache_dir()
    key = str(date_range).replace(":", "_")
    base = cache_root / "sarscov2" / f"source_{str(source).lower()}" / key
    fasta = base / "genomes.fasta"
    metadata = base / "metadata.tsv"
    provenance = base / "provenance.json"

    src_l = str(source).strip().lower()
    if (not fasta.exists() or not metadata.exists()) and src_l in {
        "ncbi",
        "ncbi_datasets",
        "ncbi-virus",
        "datasets",
    }:
        if not _datasets_cli_available():
            raise ValueError(
                "datasets CLI is not installed. Install ncbi-datasets-cli or use "
                "`babappa dataset import sarscov2 --fasta ... --metadata ...`."
            )
        base.mkdir(parents=True, exist_ok=True)
        summary_jsonl = base / "summary.jsonl"
        selected_meta = base / "selected_metadata.tsv"
        accessions_txt = base / "selected_accessions.txt"
        zip_path = base / "sars_datasets.zip"
        start_dt, _end_dt = _parse_date_range(str(date_range))
        released_after = None if start_dt is None else start_dt.strftime("%Y-%m-%d")
        _run_ncbi_datasets_summary(
            summary_jsonl=summary_jsonl,
            host=host,
            complete_only=bool(complete_only),
            released_after=released_after,
            timeout_sec=int(datasets_timeout_sec),
        )
        selected_df = _select_summary_subset(
            summary_jsonl=summary_jsonl,
            date_range=str(date_range),
            max_samples=int(max_samples),
            stratify=str(stratify),
            seed=int(seed),
        )
        selected_df.to_csv(selected_meta, sep="\t", index=False)
        accessions_txt.write_text(
            "\n".join(str(x) for x in selected_df["strain"].astype(str).tolist()) + "\n",
            encoding="utf-8",
        )
        if zip_path.exists():
            zip_path.unlink()
        _run_ncbi_datasets_download(
            zip_out=zip_path,
            host=host,
            include_cds=bool(include_cds),
            complete_only=bool(complete_only),
            timeout_sec=int(datasets_timeout_sec),
            accession_input=accessions_txt,
        )
        _extract_ncbi_dataset_zip(zip_path=zip_path, out_fasta=fasta, out_metadata_tsv=metadata)
        # Keep deterministic selection metadata for downstream stratification.
        _copy_file(selected_meta, metadata)
        _write_json(
            provenance,
            {
                "source": "NCBI Datasets CLI",
                "command_hint": (
                    "datasets summary virus genome taxon sars-cov-2 --as-json-lines --limit all "
                    f"--released-after {released_after if released_after else 'NA'}; "
                    "datasets download virus genome accession --inputfile "
                    f"{accessions_txt} --filename {zip_path} "
                    f"--include {'cds,genome,protein' if include_cds else 'genome'}"
                ),
                "fetched_at_utc": _utc_now_iso(),
                "host": host,
                "complete_only": bool(complete_only),
                "include_cds": bool(include_cds),
                "date_range": str(date_range),
                "zip_sha256": sha256_file(zip_path),
                "summary_jsonl_sha256": sha256_file(summary_jsonl),
                "selected_metadata_sha256": sha256_file(selected_meta),
                "n_accessions_selected": int(len(selected_df)),
            },
        )
    if not fasta.exists() or not metadata.exists():
        raise ValueError(
            "SARS cache package not available. Use source=ncbi with datasets CLI "
            "or import with `babappa dataset import sarscov2 --fasta ... --metadata ...`."
        )
    res = import_sarscov2_dataset(
        fasta=fasta,
        metadata_tsv=metadata,
        outdir=outdir,
        provenance_json=provenance if provenance.exists() else None,
        tree_file=tree_file,
        max_n_fraction=max_n_fraction,
        max_samples=max_samples,
        stratify=stratify,
        stratify_seed=seed,
    )
    # Store fetch intent in manifest for traceability.
    manifest = json.loads(res.fetch_manifest_json.read_text(encoding="utf-8"))
    manifest["fetch_parameters"].update(
        {
            "source": source,
            "date_range": date_range,
            "max_samples": int(max_samples),
            "host": host,
            "complete_only": bool(complete_only),
            "include_cds": bool(include_cds),
            "stratify": str(stratify),
            "seed": int(seed),
        }
    )
    manifest["source"] = "NCBI Datasets" if src_l in {"ncbi", "ncbi_datasets", "datasets", "ncbi-virus"} else source
    _write_json(res.fetch_manifest_json, manifest)
    return res
