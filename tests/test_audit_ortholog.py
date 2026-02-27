from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from babappa.audit_ortholog import (
    _frozen_energy_invariant_table,
    _null_uniformity_table,
    _testable_set_alignment_table,
)
from babappa.hash_utils import sha256_json


def test_null_uniformity_table_flags_severe_inflation() -> None:
    # Extreme left-tail concentration should be flagged.
    p = np.full(400, 0.001, dtype=float)
    table, severe = _null_uniformity_table(null_p=p, null_N=999)
    assert severe is True
    assert not table.empty
    metric_map = {str(r.metric): float(r.value) for r in table.itertuples(index=False)}
    assert metric_map["n_null"] == 400.0
    assert metric_map["tail_mass_p_le_0.05"] > 0.95


def test_testable_set_alignment_detects_mismatch() -> None:
    orth = pd.DataFrame(
        [
            {"gene_id": "g1", "babappa_status": "OK", "busted_status": "OK"},
            {"gene_id": "g2", "babappa_status": "OK", "busted_status": "FAIL"},
        ]
    )
    baseline = pd.DataFrame(
        [
            {"gene_id": "g1", "method": "busted", "status": "OK", "unit_kind": "full_gene"},
            {"gene_id": "g3", "method": "busted", "status": "OK", "unit_kind": "full_gene"},
        ]
    )
    table, mismatch, diff, testable_size = _testable_set_alignment_table(orth=orth, baseline_all=baseline)
    assert mismatch is True
    assert testable_size == 1
    assert not diff.empty
    m = {str(r.metric): int(r.value) for r in table.itertuples(index=False)}
    assert m["n_babappa_only_ok"] == 1
    assert m["n_busted_only_ok"] == 0
    assert m["n_baseline_gene_ids_not_in_ortholog_table"] == 1


def test_frozen_energy_invariant_detects_hash_drift(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    (pack / "manifests").mkdir(parents=True, exist_ok=True)
    (pack / "raw").mkdir(parents=True, exist_ok=True)
    (pack / "raw" / "frozen_model.json").write_text("{}", encoding="utf-8")
    (pack / "manifests" / "training_manifest.json").write_text(
        json.dumps({"model_hash": "EXPECTED", "m0_hash": "M0A", "phi_hash": "P1"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (pack / "raw" / "frozen_energy_runtime_hashes.tsv").write_text(
        "event\tgene_id\tenergy_model_hash\tphi_hash\tm0_hash\tstatus\treason\n"
        "after_training\t\tEXPECTED\tP1\tM0A\tOK\tmodel_frozen\n"
        "before_gene_eval\tg1\tEXPECTED\tP1\tM0A\tOK\texpected\n"
        "before_gene_eval\tg2\tEXPECTED\tP1\tM0A\tOK\texpected\n"
        "after_gene_calibration\tg1\tEXPECTED\tP1\tM0A\tOK\tok\n"
        "after_gene_calibration\tg2\tEXPECTED\tP1\tM0A\tOK\tok\n",
        encoding="utf-8",
    )

    babappa_raw = pd.DataFrame(
        [
            {"status": "OK", "p": 0.01, "model_hash": "WRONG", "M0_hash": "M0A", "phi_hash": "P1"},
            {"status": "OK", "p": 0.2, "model_hash": "WRONG", "M0_hash": "M0A", "phi_hash": "P1"},
        ]
    )

    table, fail, meta = _frozen_energy_invariant_table(pack_dir=pack, babappa_raw=babappa_raw)
    assert fail is True
    assert not table.empty
    assert meta["expected_model_hash"] == "EXPECTED"
    bad = table[table["check"] == "model_hash_matches_training_manifest"]
    assert not bad.empty
    assert bool(bad.iloc[0]["passed"]) is False


def test_frozen_energy_invariant_accepts_frozen_spec_m0_hash(tmp_path: Path) -> None:
    pack = tmp_path / "pack_ok"
    (pack / "manifests").mkdir(parents=True, exist_ok=True)
    (pack / "raw").mkdir(parents=True, exist_ok=True)
    neutral_spec = {
        "tree_newick": "(A:0.1,B:0.1);",
        "kappa": 2.0,
        "omega": 1.0,
        "simulator": "gy94",
    }
    m0_from_frozen = sha256_json(neutral_spec)
    (pack / "raw" / "frozen_model.json").write_text(
        json.dumps({"neutral_spec": neutral_spec}, indent=2) + "\n",
        encoding="utf-8",
    )
    # Deliberately different legacy/training payload hash.
    (pack / "manifests" / "training_manifest.json").write_text(
        json.dumps({"model_hash": "M1", "m0_hash": "M0_LEGACY", "phi_hash": "P1"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (pack / "raw" / "frozen_energy_runtime_hashes.tsv").write_text(
        "event\tgene_id\tenergy_model_hash\tphi_hash\tm0_hash\tstatus\treason\n"
        f"after_training\t\tM1\tP1\t{m0_from_frozen}\tOK\tmodel_frozen\n"
        f"before_gene_eval\tg1\tM1\tP1\t{m0_from_frozen}\tOK\texpected\n"
        f"before_gene_eval\tg2\tM1\tP1\t{m0_from_frozen}\tOK\texpected\n"
        f"after_gene_calibration\tg1\tM1\tP1\t{m0_from_frozen}\tOK\tok\n"
        f"after_gene_calibration\tg2\tM1\tP1\t{m0_from_frozen}\tOK\tok\n",
        encoding="utf-8",
    )
    babappa_raw = pd.DataFrame(
        [
            {"status": "OK", "p": 0.01, "model_hash": "M1", "M0_hash": m0_from_frozen, "phi_hash": "P1"},
            {"status": "OK", "p": 0.2, "model_hash": "M1", "M0_hash": m0_from_frozen, "phi_hash": "P1"},
        ]
    )
    _, fail, _ = _frozen_energy_invariant_table(pack_dir=pack, babappa_raw=babappa_raw)
    assert fail is False
