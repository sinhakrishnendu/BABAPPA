from __future__ import annotations

import pytest

from babappa.schemas import (
    load_schema,
    validate_dataset_payload,
    validate_energy_payload,
    validate_manifest_payload,
    validate_neutral_model_payload,
    validate_phi_spec_payload,
)


def test_schema_files_load() -> None:
    names = [
        "neutral_model.schema.json",
        "phi.schema.json",
        "energy.schema.json",
        "training_manifest.schema.json",
        "analysis_manifest.schema.json",
        "benchmark_manifest.schema.json",
        "dataset.schema.json",
    ]
    for name in names:
        schema = load_schema(name)
        assert isinstance(schema, dict)
        assert "$schema" in schema


def test_valid_payloads_pass_validation() -> None:
    neutral = {
        "schema_version": 1,
        "model_family": "GY94",
        "genetic_code_table": "standard",
        "tree_newick": "(A:0.1,B:0.1,C:0.1);",
        "kappa": 2.0,
        "omega": 1.0,
        "codon_frequencies_method": "F3x4",
    }
    validate_neutral_model_payload(neutral)

    phi = {
        "schema_version": 1,
        "name": "phi_default_v1",
        "feature_names": ["f1", "f2"],
    }
    validate_phi_spec_payload(phi)

    energy = {
        "schema_version": 3,
        "energy_family": "QuadraticEnergy",
        "feature_names": ["f1", "f2"],
        "mean": [0.0, 0.0],
        "covariance": [[1.0, 0.0], [0.0, 1.0]],
        "precision": [[1.0, 0.0], [0.0, 1.0]],
        "training_samples": 20,
        "training_alignments": 2,
        "max_training_length": 12,
        "ridge": 1e-5,
        "seed": 1,
        "objective": 0.1,
        "created_at_utc": "2026-02-24T00:00:00+00:00",
    }
    validate_energy_payload(energy)

    base = {
        "schema_version": 1,
        "command": "analyze",
        "tool_version": "0.4.0",
        "system": {"platform": "test"},
        "seed": 1,
        "seed_policy": "test",
    }
    train_manifest = dict(base)
    train_manifest.update({"m0_hash": "a", "phi_hash": "b", "model_hash": "c"})
    validate_manifest_payload(train_manifest, "training")

    analysis_manifest = dict(base)
    analysis_manifest.update(
        {"m0_hash": "a", "phi_hash": "b", "model_hash": "c", "tail": "right", "N": 199}
    )
    validate_manifest_payload(analysis_manifest, "analysis")

    benchmark_manifest = dict(base)
    benchmark_manifest.update({"grid": {"L_grid": [300]}, "pack_dir": "/tmp/pack"})
    validate_manifest_payload(benchmark_manifest, "benchmark")

    dataset = {
        "schema_version": 1,
        "tree_path": "/tmp/tree.nwk",
        "genes": [{"gene_id": "g1", "alignment_path": "/tmp/g1.fasta"}],
    }
    validate_dataset_payload(dataset)


def test_invalid_neutral_model_fails_fast() -> None:
    with pytest.raises(ValueError):
        validate_neutral_model_payload(
            {
                "schema_version": 1,
                "model_family": "GY94",
                "genetic_code_table": "standard",
                "tree_newick": "(A:0.1,B:0.1);",
                "kappa": 2.0,
                # omega missing
                "codon_frequencies_method": "F3x4",
            }
        )

    with pytest.raises(ValueError):
        validate_neutral_model_payload(
            {
                "schema_version": 1,
                "model_family": "GY94",
                "genetic_code_table": "standard",
                "tree_newick": "(A:0.1,B:0.1);",
                "kappa": 2.0,
                "omega": 1.2,
                "codon_frequencies_method": "F3x4",
            }
        )

    with pytest.raises(ValueError):
        validate_neutral_model_payload(
            {
                "schema_version": 1,
                "model_family": "GY94",
                "genetic_code_table": "standard",
                "tree_newick": "(A:0.1,B:0.1);",
                "kappa": 2.0,
                "omega": 1.0,
                # neither codon_frequencies nor codon_frequencies_method
            }
        )

