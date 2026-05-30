import json
from pathlib import Path

from babappa.datasets.index import read_tsv
from babappa.empirical.family_prefilter import (
    AddPrefilteredFamilyConfig,
    EmpiricalFamilyAcquisitionPlanConfig,
    EmpiricalFamilyPrefilterConfig,
    EmpiricalOODSummaryConfig,
    OODAwareFamilyBuildPlanConfig,
    TargetTaxaRecommendationConfig,
    add_prefiltered_family_to_pilot,
    plan_empirical_family_acquisition,
    plan_ood_aware_family_build,
    prefilter_empirical_family,
    recommend_target_taxa,
    summarize_empirical_ood,
)


def _write_fasta(path: Path, records: dict[str, str]) -> None:
    path.write_text("\n".join(f">{name}\n{seq}" for name, seq in records.items()) + "\n", encoding="utf-8")


def _write_tree(path: Path, labels: list[str]) -> None:
    path.write_text("(" + ",".join(f"{label}:0.1" for label in labels) + ");\n", encoding="utf-8")


def _moderate_records() -> dict[str, str]:
    base = ["ATG"] + ["GCT"] * 119
    records = {}
    for idx in range(1, 7):
        codons = list(base)
        codons[idx] = "GCC"
        codons[idx + 8] = "GCA"
        records[f"taxon{idx}"] = "".join(codons)
    return records


def test_prefilter_rejects_synthetic_high_pdistance_family(tmp_path: Path) -> None:
    records = {
        "taxon1": "ATG" + ("GCT" * 119),
        "taxon2": "ATG" + ("CGA" * 119),
        "taxon3": "ATG" + ("TTC" * 119),
        "taxon4": "ATG" + ("AAA" * 119),
        "taxon5": "ATG" + ("CCC" * 119),
        "taxon6": "ATG" + ("GGA" * 119),
    }
    fasta = tmp_path / "high.cds.fasta"
    tree = tmp_path / "high.treefile"
    _write_fasta(fasta, records)
    _write_tree(tree, list(records))
    result = prefilter_empirical_family(EmpiricalFamilyPrefilterConfig(str(fasta), str(tree), "taxon1", str(tmp_path / "prefilter")))
    assert result["decision"] in {"diagnostic_only", "reject_too_divergent"}
    assert result["mean_pdistance"] > 0.35


def test_prefilter_accepts_simple_moderate_divergence_family(tmp_path: Path) -> None:
    records = _moderate_records()
    fasta = tmp_path / "moderate.cds.fasta"
    tree = tmp_path / "moderate.treefile"
    _write_fasta(fasta, records)
    _write_tree(tree, list(records))
    result = prefilter_empirical_family(EmpiricalFamilyPrefilterConfig(str(fasta), str(tree), "taxon1", str(tmp_path / "prefilter")))
    assert result["decision"] == "accept"
    assert result["n_taxa"] == 6
    assert result["n_codons"] == 120


def test_recommend_target_taxa_writes_plant_close_template(tmp_path: Path) -> None:
    result = recommend_target_taxa(TargetTaxaRecommendationConfig("plant_close", str(tmp_path / "taxa")))
    rows = read_tsv(tmp_path / "taxa" / "recommended_target_taxa.tsv")
    assert result["status"] == "ok"
    assert "Brassicaceae" in result["recommendation"]
    assert any(row["taxon_label"] == "Arabidopsis_lyrata" for row in rows)


def test_plan_empirical_family_acquisition_writes_user_run_only_scripts(tmp_path: Path) -> None:
    taxa = tmp_path / "taxa.tsv"
    taxa.write_text("taxon_label\tensembl_dir_or_source_hint\tcategory\tnotes\ntaxon1\tsource\tclose\tok\n", encoding="utf-8")
    result = plan_empirical_family_acquisition(
        EmpiricalFamilyAcquisitionPlanConfig("fam1", "Arabidopsis_thaliana", "AT2G38470", str(taxa), str(tmp_path / "plan"))
    )
    assert result["executed"] is False
    for script in result["scripts"]:
        assert "USER-RUN ONLY" in (tmp_path / "plan" / script).read_text(encoding="utf-8")


def test_plan_ood_aware_family_build_includes_max_pdistance_gate(tmp_path: Path) -> None:
    taxa = tmp_path / "taxa.tsv"
    taxa.write_text("taxon_label\tensembl_dir_or_source_hint\tcategory\tnotes\ntaxon1\tsource\tclose\tok\n", encoding="utf-8")
    result = plan_ood_aware_family_build(
        OODAwareFamilyBuildPlanConfig("fam2", "Arabidopsis_thaliana", "AT2G38470", str(taxa), str(tmp_path / "ood"), 0.35, 6, 100)
    )
    script = (tmp_path / "ood" / "run_ood_aware_family_build.sh").read_text(encoding="utf-8")
    assert result["max_mean_pdistance"] == 0.35
    assert "MAX_MEAN_PDISTANCE=0.35" in script
    assert "USER-RUN ONLY" in script


def test_add_prefiltered_family_refuses_reject_too_divergent_unless_allowed(tmp_path: Path) -> None:
    prefilter_dir = tmp_path / "prefilter" / "fam"
    prefilter_dir.mkdir(parents=True)
    (prefilter_dir / "empirical_family_prefilter.json").write_text(
        json.dumps(
            {
                "decision": "reject_too_divergent",
                "cds_fasta": str(tmp_path / "fam.cds.fasta"),
                "tree_file": str(tmp_path / "fam.treefile"),
                "foreground": "taxon1",
            }
        ),
        encoding="utf-8",
    )
    result = add_prefiltered_family_to_pilot(
        AddPrefilteredFamilyConfig(str(tmp_path / "real"), str(prefilter_dir), "fam", "likely_positive")
    )
    assert result["status"] == "blocked"
    assert not (tmp_path / "real" / "manifest" / "real_empirical_pilot_panel.tsv").exists()


def test_ood_dashboard_summarizes_wrky_diagnostic_only_case(tmp_path: Path) -> None:
    prefilter_dir = tmp_path / "real" / "prefilter" / "WRKY_candidate_01"
    prefilter_dir.mkdir(parents=True)
    (prefilter_dir / "empirical_family_prefilter.json").write_text(
        json.dumps(
            {
                "n_taxa": 6,
                "n_codons": 474,
                "mean_pairwise_p_distance": 0.725799,
                "saturation_proxy": "extreme",
                "decision": "diagnostic_only",
                "recommended_action": "use this only as an OOD stress test",
            }
        ),
        encoding="utf-8",
    )
    result = summarize_empirical_ood(EmpiricalOODSummaryConfig(str(tmp_path / "real"), str(tmp_path / "real" / "ood_summary")))
    rows = read_tsv(tmp_path / "real" / "ood_summary" / "empirical_ood_summary.tsv")
    assert result["n_families"] == 1
    assert rows[0]["family"] == "WRKY_candidate_01"
    assert rows[0]["prefilter_decision"] == "diagnostic_only"


def test_docs_call_wrky_stress_test_not_discovery() -> None:
    text = Path("docs/OOD_AWARE_EMPIRICAL_FAMILY_SELECTION.md").read_text(encoding="utf-8")
    assert "WRKY_candidate_01" in text
    assert "stress-test/failure-mode" in text
    assert "not interpretable as a biological discovery" in text
