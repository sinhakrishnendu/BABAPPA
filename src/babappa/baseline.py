from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baseline_adapters import (
    CODEML_DOCKER_IMAGE,
    HYPHY_DOCKER_IMAGE,
    run_method_for_gene,
)
from .schemas import validate_dataset_payload


@dataclass
class BaselineRecord:
    gene_id: str
    method: str
    status: str
    reason: str
    p_value: float | None
    lrt_stat: float | None
    lnl1: float | None
    lnl0: float | None
    runtime_sec: float
    stderr_tail: str
    alternative_type: str | None = None
    replicate_id: str | None = None

    def to_row(self) -> list[str]:
        return [
            self.gene_id,
            self.method,
            self.status,
            self.reason,
            "" if self.p_value is None else f"{self.p_value:.10g}",
            "" if self.lrt_stat is None else f"{self.lrt_stat:.10g}",
            "" if self.lnl1 is None else f"{self.lnl1:.10g}",
            "" if self.lnl0 is None else f"{self.lnl0:.10g}",
            f"{self.runtime_sec:.6f}",
            self.stderr_tail.replace("\n", " | "),
            "" if self.alternative_type is None else self.alternative_type,
            "" if self.replicate_id is None else self.replicate_id,
        ]


@dataclass
class BaselineDoctorMethodResult:
    method: str
    status: str
    reason: str
    backend_hint: str
    p_value: float | None
    runtime_sec: float
    method_version: str | None = None
    container_image: str | None = None
    container_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "status": self.status,
            "reason": self.reason,
            "backend_hint": self.backend_hint,
            "p_value": self.p_value,
            "runtime_sec": self.runtime_sec,
            "method_version": self.method_version,
            "container_image": self.container_image,
            "container_digest": self.container_digest,
        }


@dataclass
class BaselineDoctorReport:
    methods: list[BaselineDoctorMethodResult]
    docker_available: bool
    singularity_available: bool
    timeout_sec: int

    @property
    def has_failures(self) -> bool:
        return any(m.status == "FAIL" for m in self.methods)

    def render(self) -> str:
        lines = [
            f"Docker available: {self.docker_available}",
            f"Singularity available: {self.singularity_available}",
            f"Toy baseline timeout (sec): {self.timeout_sec}",
            "",
        ]
        for m in self.methods:
            pv = "NA" if m.p_value is None else f"{m.p_value:.6g}"
            ver = m.method_version if m.method_version else "unknown"
            img = m.container_image if m.container_image else "NA"
            dig = m.container_digest if m.container_digest else "NA"
            lines.append(
                f"[{m.status}] {m.method}: {m.reason} | backend_hint={m.backend_hint} | version={ver} | image={img} | digest={dig} | p={pv} | runtime_sec={m.runtime_sec:.4f}"
            )
        lines.append("")
        lines.append("Baseline doctor summary: " + ("FAIL" if self.has_failures else "PASS"))
        return "\n".join(lines)


def load_dataset_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset file must be a JSON object: {p}")
    validate_dataset_payload(payload)
    return payload


def _docker_image_present(image: str) -> bool:
    if not shutil.which("docker"):
        return False
    proc = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def _docker_pull_image(image: str) -> None:
    subprocess.run(["docker", "pull", image], check=True, capture_output=True, text=True)


def _docker_image_digest(image: str) -> str | None:
    if not shutil.which("docker"):
        return None
    proc = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", image],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout.strip())
    except Exception:
        return None
    if isinstance(payload, list) and payload:
        first = payload[0]
        return None if first is None else str(first)
    return None


def _method_version(method: str) -> str:
    m = method.lower()
    if m == "codeml":
        return "paml-4.10.7"
    if m in {"busted", "relax"}:
        return "hyphy-2.5.63"
    return "unknown"


def _method_container_image(method: str) -> str | None:
    if method.lower() == "codeml":
        return CODEML_DOCKER_IMAGE
    if method.lower() in {"busted", "relax"}:
        return HYPHY_DOCKER_IMAGE
    return None


@contextmanager
def _backend_env(method: str, container: str) -> Any:
    m = method.lower()
    container_l = container.lower().strip()
    key = "BABAPPA_CODEML_BACKEND" if m == "codeml" else "BABAPPA_HYPHY_BACKEND"
    old = os.environ.get(key)
    if container_l and container_l != "auto":
        os.environ[key] = container_l
    try:
        yield
    finally:
        if old is None:
            if key in os.environ:
                del os.environ[key]
        else:
            os.environ[key] = old


def _write_toy_alignment(path: Path) -> None:
    # Build three related, codon-clean sequences with enough variation for HyPhy toy fitting.
    base_codons = ["GCT", "GAA", "TTC", "CAG", "AAC", "GGA", "TAT", "CCA", "GTT", "ATC"] * 10
    seq1 = list(base_codons)
    seq2 = list(base_codons)
    seq3 = list(base_codons)
    for i in range(0, len(base_codons), 9):
        seq2[i] = "GCC"
    for i in range(4, len(base_codons), 11):
        seq3[i] = "AAG"
    s1 = "".join(seq1)
    s2 = "".join(seq2)
    s3 = "".join(seq3)
    path.write_text(
        "\n".join(
            [
                ">T1",
                s1,
                ">T2",
                s2,
                ">T3",
                s3,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_baseline_doctor(
    *,
    methods: tuple[str, ...] = ("busted", "relax"),
    timeout_sec: int = 5,
    work_dir: str | Path | None = None,
    pull_images: bool = True,
    container: str = "auto",
) -> BaselineDoctorReport:
    docker_ok = shutil.which("docker") is not None
    singularity_ok = shutil.which("singularity") is not None

    method_list = [m.strip().lower() for m in methods if m.strip()]
    valid = {"codeml", "busted", "relax"}
    for m in method_list:
        if m not in valid:
            raise ValueError(f"Unsupported baseline method in doctor: {m}")

    if work_dir is None:
        base = Path(tempfile.mkdtemp(prefix="babappa_baseline_doctor_"))
    else:
        base = Path(work_dir).resolve()
        base.mkdir(parents=True, exist_ok=True)

    aln_path = base / "toy.fasta"
    tree_path = base / "toy.nwk"
    _write_toy_alignment(aln_path)
    tree_path.write_text("(T1:0.1,T2:0.1,T3:0.1);\n", encoding="utf-8")

    results: list[BaselineDoctorMethodResult] = []
    for method in method_list:
        container_l = container.lower().strip()
        if container_l not in {"auto", "docker", "singularity", "local"}:
            raise ValueError(f"Unsupported container mode: {container}")
        backend_hint = "unknown"
        if container_l == "local":
            backend_hint = "local"
        elif container_l == "docker":
            backend_hint = "docker"
        elif container_l == "singularity":
            backend_hint = "singularity"
        elif method == "codeml":
            if os.environ.get("BABAPPA_CODEML_BACKEND", "auto").strip().lower() == "local":
                backend_hint = "local"
            elif docker_ok:
                backend_hint = "docker"
            elif singularity_ok:
                backend_hint = "singularity"
        else:
            if os.environ.get("BABAPPA_HYPHY_BACKEND", "auto").strip().lower() == "local":
                backend_hint = "local"
            elif docker_ok:
                backend_hint = "docker"
            elif singularity_ok:
                backend_hint = "singularity"

        image = _method_container_image(method)
        digest = None
        version = _method_version(method)
        if backend_hint == "docker":
            if not docker_ok:
                results.append(
                    BaselineDoctorMethodResult(
                        method=method,
                        status="FAIL",
                        reason="docker_unavailable",
                        backend_hint=backend_hint,
                        p_value=None,
                        runtime_sec=0.0,
                        method_version=version,
                        container_image=image,
                        container_digest=None,
                    )
                )
                continue
        if backend_hint == "singularity":
            if not singularity_ok:
                results.append(
                    BaselineDoctorMethodResult(
                        method=method,
                        status="FAIL",
                        reason="singularity_unavailable",
                        backend_hint=backend_hint,
                        p_value=None,
                        runtime_sec=0.0,
                        method_version=version,
                        container_image=image,
                        container_digest=None,
                    )
                )
                continue

        if pull_images and backend_hint == "docker":
            try:
                if image is not None and not _docker_image_present(image):
                    _docker_pull_image(image)
                if image is not None:
                    digest = _docker_image_digest(image)
            except Exception as exc:
                results.append(
                    BaselineDoctorMethodResult(
                        method=method,
                        status="FAIL",
                        reason=f"docker_image_prepare_failed:{exc}",
                        backend_hint=backend_hint,
                        p_value=None,
                        runtime_sec=0.0,
                        method_version=version,
                        container_image=image,
                        container_digest=digest,
                    )
                )
                continue
        elif backend_hint == "docker" and image is not None:
            digest = _docker_image_digest(image)

        method_work = base / f"doctor_{method}"
        method_work.mkdir(parents=True, exist_ok=True)
        try:
            with _backend_env(method, container):
                recs = run_method_for_gene(
                    method=method,
                    alignment_path=aln_path,
                    tree_path=tree_path,
                    workdir=method_work,
                    foreground_taxon="T1",
                    run_site_model=False,
                    timeout_sec=int(timeout_sec),
                )
        except Exception as exc:
            results.append(
                BaselineDoctorMethodResult(
                    method=method,
                    status="FAIL",
                    reason=f"adapter_exception:{exc}",
                    backend_hint=backend_hint,
                    p_value=None,
                    runtime_sec=0.0,
                    method_version=version,
                    container_image=image,
                    container_digest=digest,
                )
            )
            continue

        ok = None
        for r in recs:
            p = r.p_value
            if r.status == "OK" and p is not None and 0.0 <= float(p) <= 1.0:
                ok = r
                break
        if ok is not None:
            results.append(
                BaselineDoctorMethodResult(
                    method=method,
                    status="PASS",
                    reason="parsed valid p-value on toy dataset",
                    backend_hint=backend_hint,
                    p_value=float(ok.p_value),
                    runtime_sec=float(ok.runtime_sec),
                    method_version=version,
                    container_image=image,
                    container_digest=digest,
                )
            )
        else:
            fail_reason = "; ".join(f"{r.method}:{r.reason}" for r in recs[:3]) if recs else "no_records"
            runtime = float(sum(float(r.runtime_sec) for r in recs)) if recs else 0.0
            results.append(
                BaselineDoctorMethodResult(
                    method=method,
                    status="FAIL",
                    reason=f"no_valid_p_value:{fail_reason}",
                    backend_hint=backend_hint,
                    p_value=None,
                    runtime_sec=runtime,
                    method_version=version,
                    container_image=image,
                    container_digest=digest,
                )
            )

    return BaselineDoctorReport(
        methods=results,
        docker_available=docker_ok,
        singularity_available=singularity_ok,
        timeout_sec=int(timeout_sec),
    )


def run_baseline_for_dataset(
    *,
    method: str,
    dataset: dict[str, Any],
    out_tsv: str | Path,
    work_dir: str | Path | None = None,
    foreground_taxon: str | None = None,
    foreground_branch_label: str | None = None,
    timeout_sec: int = 1800,
    container: str = "auto",
    jobs: int = 1,
) -> list[BaselineRecord]:
    tree_path = Path(str(dataset["tree_path"])).resolve()
    genes: list[dict[str, Any]] = list(dataset["genes"])
    method_options = dict(dataset.get("method_options") or {})

    base_work = Path(work_dir) if work_dir else Path(out_tsv).resolve().parent / "baseline_work"
    base_work.mkdir(parents=True, exist_ok=True)

    run_site_model = bool(method_options.get("run_site_model", True))
    default_foreground = (
        foreground_taxon
        or method_options.get("foreground_taxon")
        or method_options.get("foreground")
    )
    _ = foreground_branch_label  # reserved for future branch-label specific adapters

    n_jobs = int(jobs)
    if n_jobs <= 0:
        n_jobs = max(1, min(16, (os.cpu_count() or 1)))

    fallback_method = {
        "codeml": "codeml_branchsite",
        "busted": "busted",
        "relax": "relax",
    }.get(method.lower(), method.lower())

    def _run_one_gene(gene: dict[str, Any]) -> list[BaselineRecord]:
        gene_id = str(gene["gene_id"])
        alignment_path = Path(str(gene["alignment_path"])).resolve()
        gene_tree_path = (
            Path(str(gene["tree_path"])).resolve()
            if gene.get("tree_path") is not None
            else tree_path
        )
        job_dir = base_work / f"{method}_{gene_id}"
        job_dir.mkdir(parents=True, exist_ok=True)

        gene_foreground = (
            gene.get("foreground_taxon")
            or gene.get("foreground")
            or default_foreground
        )
        try:
            adapter_records = run_method_for_gene(
                method=method,
                alignment_path=alignment_path,
                tree_path=gene_tree_path,
                workdir=job_dir,
                foreground_taxon=(
                    None if gene_foreground is None else str(gene_foreground)
                ),
                run_site_model=run_site_model,
                timeout_sec=int(timeout_sec),
            )
        except Exception as exc:
            adapter_records = []
            adapter_records.append(
                {
                    "method": fallback_method,
                    "status": "FAIL",
                    "reason": f"adapter_exception:{exc}",
                    "p_value": None,
                    "lrt_stat": None,
                    "lnL1": None,
                    "lnL0": None,
                    "runtime_sec": 0.0,
                    "stderr_tail": "",
                }
            )

        gene_records: list[BaselineRecord] = []
        for adapter in adapter_records:
            if hasattr(adapter, "to_dict"):
                adapter_dict = adapter.to_dict()  # type: ignore[assignment]
            else:
                adapter_dict = dict(adapter)  # type: ignore[arg-type]
            status = str(adapter_dict.get("status", "FAIL")).upper()
            if status not in {"OK", "FAIL"}:
                status = "FAIL"
            record = BaselineRecord(
                gene_id=gene_id,
                method=str(adapter_dict.get("method", method.lower())),
                status=status,
                reason=str(adapter_dict.get("reason", "unknown")),
                p_value=(
                    None
                    if adapter_dict.get("p_value") is None
                    else float(adapter_dict.get("p_value"))
                ),
                lrt_stat=(
                    None
                    if adapter_dict.get("lrt_stat") is None
                    else float(adapter_dict.get("lrt_stat"))
                ),
                lnl1=(
                    None
                    if adapter_dict.get("lnL1") is None
                    else float(adapter_dict.get("lnL1"))
                ),
                lnl0=(
                    None
                    if adapter_dict.get("lnL0") is None
                    else float(adapter_dict.get("lnL0"))
                ),
                runtime_sec=float(adapter_dict.get("runtime_sec", 0.0)),
                stderr_tail=str(adapter_dict.get("stderr_tail", "")),
                alternative_type=None
                if gene.get("alternative_type") is None
                else str(gene.get("alternative_type")),
                replicate_id=None
                if gene.get("replicate_id") is None
                else str(gene.get("replicate_id")),
            )
            gene_records.append(record)
        return gene_records

    records: list[BaselineRecord] = []
    with _backend_env(method, container):
        if n_jobs == 1 or len(genes) <= 1:
            for gene in genes:
                records.extend(_run_one_gene(gene))
        else:
            ordered: list[tuple[int, list[BaselineRecord]]] = []
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                fut_to_idx = {
                    ex.submit(_run_one_gene, gene): idx
                    for idx, gene in enumerate(genes)
                }
                for fut in as_completed(fut_to_idx):
                    idx = int(fut_to_idx[fut])
                    try:
                        recs = fut.result()
                    except Exception as exc:
                        gene_id = str(genes[idx].get("gene_id", f"gene_{idx+1}"))
                        recs = [
                            BaselineRecord(
                                gene_id=gene_id,
                                method=fallback_method,
                                status="FAIL",
                                reason=f"worker_exception:{exc}",
                                p_value=None,
                                lrt_stat=None,
                                lnl1=None,
                                lnl0=None,
                                runtime_sec=0.0,
                                stderr_tail="",
                                alternative_type=None,
                                replicate_id=None,
                            )
                        ]
                    ordered.append((idx, recs))
            ordered.sort(key=lambda x: x[0])
            for _, recs in ordered:
                records.extend(recs)

    out_path = Path(out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "gene_id",
        "method",
        "status",
        "reason",
        "p_value",
        "lrt_stat",
        "lnL1",
        "lnL0",
        "runtime_sec",
        "stderr_tail",
        "alternative_type",
        "replicate_id",
    ]
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for record in records:
            handle.write("\t".join(record.to_row()) + "\n")

    return records
