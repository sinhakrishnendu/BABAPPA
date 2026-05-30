import json

from babappa.simulate import SimulationConfig, simulate_families


REQUIRED_SUFFIXES = [
    ".fasta",
    ".treefile",
    ".truth.json",
    ".homology.tsv",
    ".events.tsv",
    ".meta.json",
]


def test_simulate_families_creates_manifest_and_family_files(tmp_path) -> None:
    outdir = tmp_path / "sim"
    summary = simulate_families(
        SimulationConfig(
            outdir=str(outdir),
            n_families=2,
            n_taxa=4,
            n_codons=30,
            seed=7,
        )
    )

    assert summary["status"] == "ok"
    assert (outdir / "manifest.json").exists()
    assert len(summary["family_ids"]) == 2

    for family_id in summary["family_ids"]:
        family_dir = outdir / "families" / family_id
        assert family_dir.exists()
        for suffix in REQUIRED_SUFFIXES:
            assert (family_dir / f"{family_id}{suffix}").exists()


def test_same_seed_gives_same_first_family_fasta(tmp_path) -> None:
    outdir_a = tmp_path / "same_a"
    outdir_b = tmp_path / "same_b"
    kwargs = {
        "n_families": 1,
        "n_taxa": 4,
        "n_codons": 30,
        "seed": 123,
    }

    simulate_families(SimulationConfig(outdir=str(outdir_a), **kwargs))
    simulate_families(SimulationConfig(outdir=str(outdir_b), **kwargs))

    assert _first_family_fasta(outdir_a) == _first_family_fasta(outdir_b)


def test_different_seed_gives_different_first_family_fasta(tmp_path) -> None:
    outdir_a = tmp_path / "seed_a"
    outdir_b = tmp_path / "seed_b"

    simulate_families(
        SimulationConfig(
            outdir=str(outdir_a),
            n_families=1,
            n_taxa=4,
            n_codons=30,
            seed=123,
        )
    )
    simulate_families(
        SimulationConfig(
            outdir=str(outdir_b),
            n_families=1,
            n_taxa=4,
            n_codons=30,
            seed=124,
        )
    )

    assert _first_family_fasta(outdir_a) != _first_family_fasta(outdir_b)


def test_truth_json_contains_gene_and_branch_labels(tmp_path) -> None:
    outdir = tmp_path / "truth"
    simulate_families(
        SimulationConfig(
            outdir=str(outdir),
            n_families=1,
            n_taxa=4,
            n_codons=30,
            seed=11,
            positive_rate=1.0,
        )
    )

    truth = _first_family_truth(outdir)

    assert "gene_label" in truth["labels"]
    assert truth["labels"]["gene_label"] == 1
    assert isinstance(truth["labels"]["branch_labels"], dict)
    assert sum(truth["labels"]["branch_labels"].values()) == 1


def test_positive_families_contain_selected_sites_1based(tmp_path) -> None:
    outdir = tmp_path / "positive"
    simulate_families(
        SimulationConfig(
            outdir=str(outdir),
            n_families=2,
            n_taxa=4,
            n_codons=30,
            seed=22,
            positive_rate=1.0,
            selected_site_fraction=0.1,
        )
    )

    for family_id in _family_ids(outdir):
        truth_path = outdir / "families" / family_id / f"{family_id}.truth.json"
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        assert truth["has_positive_selection"] is True
        assert truth["selected_sites_1based"]


def _family_ids(outdir):
    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    return manifest["family_ids"]


def _first_family_fasta(outdir) -> str:
    family_id = _family_ids(outdir)[0]
    fasta_path = outdir / "families" / family_id / f"{family_id}.fasta"
    return fasta_path.read_text(encoding="utf-8")


def _first_family_truth(outdir) -> dict:
    family_id = _family_ids(outdir)[0]
    truth_path = outdir / "families" / family_id / f"{family_id}.truth.json"
    return json.loads(truth_path.read_text(encoding="utf-8"))
