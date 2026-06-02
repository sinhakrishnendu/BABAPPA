import importlib.util
import json
from argparse import Namespace
from pathlib import Path

from babappa.datasets.index import read_tsv, write_tsv
from babappa.empirical.reference_eval import (
    HyphyReferenceParseConfig,
    MethodClaimReadinessConfig,
    parse_hyphy_reference,
    validate_method_claim_readiness,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_drosophila_summarizer():
    path = Path(__file__).resolve().parents[1] / "publication_benchmark" / "scripts" / "drosophila_05_summarize_babappa_absrel.py"
    spec = importlib.util.spec_from_file_location("drosophila_05_summarize_babappa_absrel", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hyphy_official_parser_negative_when_nested_p_values_positive(tmp_path: Path) -> None:
    hyphy = tmp_path / "hyphy"
    _write_json(
        hyphy / "absrel.json",
        {
            "test results": {"positive test results": 0, "tested": 4},
            "branch attributes": {"0": {"branch": {"Corrected P-value": 0.001}}},
        },
    )
    parse_hyphy_reference(HyphyReferenceParseConfig(str(hyphy), str(tmp_path / "parsed")))
    parsed = json.loads((tmp_path / "parsed" / "hyphy_reference_parse.json").read_text())
    absrel = parsed["parsed_outputs"][0]
    assert absrel["result_class"] == "negative"
    assert absrel["official_positive_test_results"] == 0


def test_hyphy_official_parser_positive_from_positive_test_results(tmp_path: Path) -> None:
    hyphy = tmp_path / "hyphy"
    _write_json(hyphy / "absrel.json", {"test results": {"positive test results": 2, "tested": 5}})
    parse_hyphy_reference(HyphyReferenceParseConfig(str(hyphy), str(tmp_path / "parsed")))
    parsed = json.loads((tmp_path / "parsed" / "hyphy_reference_parse.json").read_text())
    absrel = parsed["parsed_outputs"][0]
    assert absrel["result_class"] == "positive"
    assert absrel["official_positive_test_results"] == 2


def test_hyphy_missing_official_field_does_not_infer_in_official_mode(tmp_path: Path) -> None:
    hyphy = tmp_path / "hyphy"
    _write_json(hyphy / "absrel.json", {"branch attributes": {"0": {"branch": {"Corrected P-value": 0.001}}}})
    parse_hyphy_reference(HyphyReferenceParseConfig(str(hyphy), str(tmp_path / "parsed")))
    parsed = json.loads((tmp_path / "parsed" / "hyphy_reference_parse.json").read_text())
    absrel = parsed["parsed_outputs"][0]
    assert absrel["result_class"] == "inconclusive"
    assert "missing_official_positive_test_results" in absrel["warnings"]


def test_exploratory_recursive_mode_is_explicit_and_non_publication(tmp_path: Path) -> None:
    hyphy = tmp_path / "hyphy"
    _write_json(hyphy / "absrel.json", {"branch attributes": {"0": {"branch": {"Corrected P-value": 0.001}}}})
    parse_hyphy_reference(
        HyphyReferenceParseConfig(
            str(hyphy),
            str(tmp_path / "parsed"),
            hyphy_positive_mode="exploratory_recursive",
        )
    )
    parsed = json.loads((tmp_path / "parsed" / "hyphy_reference_parse.json").read_text())
    absrel = parsed["parsed_outputs"][0]
    assert absrel["result_class"] == "positive"
    assert "exploratory_recursive_mode_not_for_publication" in absrel["warnings"]


def test_publication_benchmark_uses_official_mode_by_default(tmp_path: Path) -> None:
    module = _load_drosophila_summarizer()
    panel = tmp_path / "panel.tsv"
    results = tmp_path / "results"
    write_tsv(
        panel,
        [{"panel_id": "family1", "gene_family": "gene", "benchmark_stratum": "strict_in_domain"}],
        ["panel_id", "gene_family", "benchmark_stratum"],
    )
    fam = results / "babappa" / "family1"
    fam.mkdir(parents=True)
    write_tsv(
        fam / "branch_predictions.tsv",
        [{"branch": "taxon1", "n_called_positive": "2", "max_prob_positive": "0.9"}],
        ["branch", "n_called_positive", "max_prob_positive"],
    )
    write_tsv(
        fam / "gene_summary.tsv",
        [{"max_gene_support": "0.9"}],
        ["max_gene_support"],
    )
    _write_json(fam / "prediction_manifest.json", {"applicability": "in_domain"})
    _write_json(
        fam / "babappa_native_null" / "babappa_native_null_summary.json",
        {"status": "ok", "evidence_class": "strong_babappa_native_support"},
    )
    _write_json(
        results / "hyphy_absrel" / "family1" / "absrel.json",
        {
            "test results": {"positive test results": 0, "tested": 1},
            "branch attributes": {"0": {"branch": {"Corrected P-value": 0.001}}},
        },
    )
    module.summarize(
        Namespace(
            panel=str(panel),
            results_root=str(results),
            outdir=str(tmp_path / "summary"),
            hyphy_positive_mode="official",
        )
    )
    summary = json.loads((tmp_path / "summary" / "benchmark_summary.json").read_text())
    assert summary["hyphy_positive_mode"] == "official"
    assert summary["hyphy_absrel_positive_families"] == 0
    rows = read_tsv(tmp_path / "summary" / "benchmark_family_results.tsv")
    assert rows[0]["concordance"] == "BABAPPA_only"


def test_method_claim_readiness_is_complementary_not_replacement(tmp_path: Path) -> None:
    _write_json(tmp_path / "simulation.json", {"status": "ok", "decision": "CONDITIONAL PASS"})
    _write_json(
        tmp_path / "drosophila.json",
        {
            "status": "ok",
            "n_families": 140,
            "babappa_native_calibrated_support": 14,
            "hyphy_absrel_positive_families": 73,
            "concordance_counts": {
                "concordant_positive": 3,
                "concordant_negative": 56,
                "BABAPPA_only": 11,
                "HyPhy_only": 70,
            },
            "overall_agreement": 0.421,
            "positive_agreement_against_hyphy": 0.041,
        },
    )
    _write_json(tmp_path / "wrky.json", {"decision_category": "diagnostic_positive_calibration_pending"})
    result = validate_method_claim_readiness(
        MethodClaimReadinessConfig(
            str(tmp_path / "simulation.json"),
            str(tmp_path / "drosophila.json"),
            str(tmp_path / "wrky.json"),
            str(tmp_path / "readiness"),
        )
    )
    payload = json.loads((tmp_path / "readiness" / "method_claim_readiness.json").read_text())
    assert result["ready_as_conservative_complementary_method"] is True
    assert payload["not_ready_as_replacement"] is True
    assert payload["needs_negative_controls"] is True
    assert payload["needs_matched_null_calibration"] is True


def test_benchmark_dataset_plan_contains_required_layers() -> None:
    text = Path("docs/BABAPPA_BENCHMARK_DATASET_PLAN.md").read_text(encoding="utf-8").lower()
    for phrase in [
        "simulated truth set",
        "empirical comparator set",
        "ood",
        "controls",
        "native-null calibration",
    ]:
        assert phrase in text


def test_benchmark_interpretation_report_has_comparator_boundary() -> None:
    text = Path("publication_benchmark/drosophila_corrected_summary/BABAPPA_Drosophila_benchmark_interpretation.md").read_text(encoding="utf-8")
    assert "external comparator, not ground truth" in text


def test_main_manuscript_avoids_forbidden_superiority_claims() -> None:
    text = Path("Manuscript/BABAPPA_method_paper_auxiliary_saturation.tex").read_text(encoding="utf-8").lower()
    forbidden = [
        "beats hyphy",
        "beats codeml",
        "matches hyphy",
        "matches codeml",
        "validated against hyphy",
        "validated against codeml",
        "superior to hyphy",
        "superior to codeml",
        "empirical discovery claim confirmed",
        "empirical positive selection discovered",
    ]
    assert not any(phrase in text for phrase in forbidden)
