from pathlib import Path

import pytest

from babappa.datasets.index import read_tsv, write_tsv
from babappa.empirical.input_staging import (
    CdsFastaSanitizeConfig,
    ForegroundCandidateConfig,
    LocalPilotFileDiscoveryConfig,
    RealPilotBatchImportConfig,
    RealPilotFamilyImportConfig,
    RealPilotInputStagingConfig,
    RealPilotReadinessConfig,
    RealPilotTreeBuildingPlanConfig,
    discover_local_pilot_files,
    import_real_pilot_batch,
    import_real_pilot_family,
    list_foreground_candidates,
    plan_real_pilot_tree_building,
    prepare_real_pilot_inputs,
    sanitize_cds_fasta,
    validate_real_pilot_readiness,
)


FIELDS = [
    "panel_id",
    "gene_family",
    "species_group",
    "cds_fasta",
    "tree_file",
    "foreground",
    "expected_category",
    "reference_status",
    "notes",
]


def _valid_cds(path: Path, n_taxa: int = 6, n_codons: int = 60) -> None:
    seq = "ATG" + ("GCT" * (n_codons - 1))
    lines = []
    for idx in range(1, n_taxa + 1):
        lines.extend([f">taxon{idx}", seq])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tree(path: Path, n_taxa: int = 6) -> None:
    tips = ",".join(f"taxon{idx}:0.1" for idx in range(1, n_taxa + 1))
    path.write_text(f"({tips});\n", encoding="utf-8")


def _manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(path, rows, FIELDS)


def _row(panel_id: str, cds: str, tree: str, foreground: str = "taxon1") -> dict:
    return {
        "panel_id": panel_id,
        "gene_family": panel_id,
        "species_group": "synthetic",
        "cds_fasta": cds,
        "tree_file": tree,
        "foreground": foreground,
        "expected_category": "unknown",
        "reference_status": "planned",
        "notes": "",
    }


def test_prepare_real_pilot_inputs_creates_layout_and_missing_inputs(tmp_path: Path) -> None:
    workspace = tmp_path / "real"
    _manifest(workspace / "manifest" / "real_empirical_pilot_panel.tsv", [_row("fam1", "../input/cds/fam1.cds.fasta", "../input/trees/fam1.treefile")])
    result = prepare_real_pilot_inputs(
        RealPilotInputStagingConfig(str(workspace), "real_empirical_pilot_panel.tsv", str(workspace / "input_staging"))
    )
    assert result["missing_inputs"] == 2
    assert (workspace / "input" / "cds").is_dir()
    assert (workspace / "input_staging" / "missing_inputs.tsv").exists()
    missing = read_tsv(workspace / "input_staging" / "missing_inputs.tsv")
    assert missing[0]["suggested_path"].endswith("fam1.cds.fasta")


def test_import_real_pilot_family_copies_files_and_updates_manifest(tmp_path: Path) -> None:
    cds = tmp_path / "source.cds.fasta"
    tree = tmp_path / "source.treefile"
    _valid_cds(cds)
    _tree(tree)
    workspace = tmp_path / "real"
    result = import_real_pilot_family(
        RealPilotFamilyImportConfig(
            workspace=str(workspace),
            panel_id="fam1",
            gene_family="family one",
            species_group="synthetic",
            cds_fasta=str(cds),
            tree_file=str(tree),
            foreground="taxon1",
            expected_category="known_positive",
            reference_status="planned",
            notes="tiny valid import",
        )
    )
    assert result["status"] == "ok"
    rows = read_tsv(workspace / "manifest" / "real_empirical_pilot_panel.tsv")
    assert rows[0]["cds_fasta"] == "../input/cds/fam1.cds.fasta"
    assert (workspace / "input" / "trees" / "fam1.treefile").exists()


def test_import_real_pilot_batch_works_on_tiny_data(tmp_path: Path) -> None:
    cds = tmp_path / "source.cds.fasta"
    tree = tmp_path / "source.treefile"
    _valid_cds(cds)
    _tree(tree)
    batch = tmp_path / "batch.tsv"
    _manifest(batch, [_row("fam_batch", str(cds), str(tree))])
    result = import_real_pilot_batch(RealPilotBatchImportConfig(str(tmp_path / "real"), str(batch)))
    assert result["n_imported"] == 1
    assert (tmp_path / "real" / "input_staging" / "batch_import_report.tsv").exists()


def test_sanitize_cds_fasta_fails_non_codon_sequences_in_strict_mode(tmp_path: Path) -> None:
    bad = tmp_path / "bad.fasta"
    bad.write_text(">taxon1\nATGG\n>taxon2\nATGG\n>taxon3\nATGG\n", encoding="utf-8")
    result = sanitize_cds_fasta(CdsFastaSanitizeConfig(str(bad), str(tmp_path / "out.fasta"), str(tmp_path / "report.json"), "strict"))
    assert result["status"] == "fail"
    assert any(item.startswith("length_not_divisible_by_3") for item in result["failures"])


def test_sanitize_cds_fasta_detects_internal_stop_codons(tmp_path: Path) -> None:
    bad = tmp_path / "bad_stop.fasta"
    bad.write_text(">taxon1\nATGTAAGCT\n>taxon2\nATGTAAGCT\n>taxon3\nATGTAAGCT\n", encoding="utf-8")
    result = sanitize_cds_fasta(CdsFastaSanitizeConfig(str(bad), str(tmp_path / "out.fasta"), str(tmp_path / "report.json"), "strict"))
    assert result["status"] == "fail"
    assert any(item.startswith("internal_stop_codon") for item in result["failures"])


def test_sanitize_cds_fasta_detects_missing_start_codon(tmp_path: Path) -> None:
    bad = tmp_path / "bad_start.fasta"
    bad.write_text(">taxon1\nGCTGCTTAA\n>taxon2\nATGGCTTAA\n>taxon3\nATGGCTTAA\n", encoding="utf-8")
    result = sanitize_cds_fasta(CdsFastaSanitizeConfig(str(bad), str(tmp_path / "out.fasta"), str(tmp_path / "report.json"), "strict"))
    assert result["status"] == "fail"
    assert any(item.startswith("missing_start_codon") for item in result["failures"])


def test_list_foreground_candidates_reports_matching_taxa(tmp_path: Path) -> None:
    cds = tmp_path / "fam.cds.fasta"
    tree = tmp_path / "fam.treefile"
    _valid_cds(cds)
    _tree(tree)
    result = list_foreground_candidates(ForegroundCandidateConfig(str(cds), str(tree), str(tmp_path / "fg"), "taxon1"))
    assert result["status"] == "ok"
    assert result["matching_tips"] == 6
    assert result["foreground_valid"] is True


def test_plan_real_pilot_tree_building_writes_user_run_only_script(tmp_path: Path) -> None:
    workspace = tmp_path / "real"
    cds = workspace / "input" / "cds" / "fam1.cds.fasta"
    cds.parent.mkdir(parents=True)
    _valid_cds(cds)
    _manifest(workspace / "manifest" / "real_empirical_pilot_panel.tsv", [_row("fam1", "../input/cds/fam1.cds.fasta", "../input/trees/fam1.treefile")])
    result = plan_real_pilot_tree_building(
        RealPilotTreeBuildingPlanConfig(str(workspace), "real_empirical_pilot_panel.tsv", str(workspace / "tree_building_plan"))
    )
    script = (workspace / "tree_building_plan" / "build_missing_trees.sh").read_text()
    assert result["n_trees_to_build"] == 1
    assert "USER-RUN ONLY" in script
    assert "iqtree2" in script


def test_validate_real_pilot_readiness_false_when_files_missing(tmp_path: Path) -> None:
    workspace = tmp_path / "real"
    _manifest(workspace / "manifest" / "real_empirical_pilot_panel.tsv", [_row("fam1", "../input/cds/fam1.cds.fasta", "../input/trees/fam1.treefile")])
    result = validate_real_pilot_readiness(
        RealPilotReadinessConfig(str(workspace), "real_empirical_pilot_panel.tsv", str(workspace / "readiness"))
    )
    assert result["ready_to_run"] is False
    assert result["files_missing"] == 1


def test_validate_real_pilot_readiness_true_on_tiny_valid_panel(tmp_path: Path) -> None:
    workspace = tmp_path / "real"
    cds = workspace / "input" / "cds" / "fam1.cds.fasta"
    tree = workspace / "input" / "trees" / "fam1.treefile"
    cds.parent.mkdir(parents=True)
    tree.parent.mkdir(parents=True)
    _valid_cds(cds)
    _tree(tree)
    _manifest(workspace / "manifest" / "real_empirical_pilot_panel.tsv", [_row("fam1", "../input/cds/fam1.cds.fasta", "../input/trees/fam1.treefile")])
    result = validate_real_pilot_readiness(
        RealPilotReadinessConfig(str(workspace), "real_empirical_pilot_panel.tsv", str(workspace / "readiness"))
    )
    assert result["ready_to_run"] is True
    assert result["status"] == "ready"


def test_discover_local_pilot_files_suggests_candidate_pairs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "famA.cds.fasta").write_text(">taxon1\nATGGCT\n", encoding="utf-8")
    (data / "famA.treefile").write_text("(taxon1:0.1);\n", encoding="utf-8")
    result = discover_local_pilot_files(LocalPilotFileDiscoveryConfig(str(data), str(tmp_path / "discover")))
    assert result["n_fasta"] == 1
    assert result["n_tree"] == 1
    assert result["n_pair_suggestions"] == 1
