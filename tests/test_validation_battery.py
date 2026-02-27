from __future__ import annotations

from pathlib import Path

from babappa.cli import main


def _write_alignment(path: Path, records: dict[str, str]) -> None:
    lines: list[str] = []
    for name, seq in records.items():
        lines.append(f">{name}")
        lines.append(seq)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_cli_validation_battery_smoke(tmp_path: Path) -> None:
    tree_path = tmp_path / "tree.nwk"
    tree_path.write_text("(A:0.1,B:0.1,C:0.1,D:0.1);\n", encoding="utf-8")

    neutral = tmp_path / "neutral.fasta"
    _write_alignment(
        neutral,
        {
            "A": "ATG" * 20,
            "B": "ATG" * 20,
            "C": "ATG" * 20,
            "D": "ATG" * 20,
        },
    )

    model_path = tmp_path / "model.json"
    assert main(["train", "--neutral", str(neutral), "--tree-file", str(tree_path), "--model", str(model_path), "--seed", "7"]) == 0

    genes_dir = tmp_path / "genes"
    genes_dir.mkdir(parents=True, exist_ok=True)
    _write_alignment(
        genes_dir / "g1.fasta",
        {
            "A": "ATG" * 20,
            "B": "ATG" * 20,
            "C": "ATG" * 20,
            "D": "ATG" * 20,
        },
    )
    _write_alignment(
        genes_dir / "g2.fasta",
        {
            "A": ("ATG" * 19) + "TTG",
            "B": ("ATG" * 19) + "CTG",
            "C": "ATG" * 20,
            "D": "ATG" * 20,
        },
    )

    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                f"model: {model_path}",
                "global_seed: 11",
                "N: 12",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    grid = tmp_path / "grid.yaml"
    grid.write_text(
        "\n".join(
            [
                "omega_mode: [fixed_1, fixed_omega_lt1]",
                "fixed_omega_lt1_values: [0.3]",
                "rate_heterogeneity: [none]",
                "codon_freqs: [uniform, empirical_from_alignment]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_validate_null = tmp_path / "out_validate_null"
    out_validate_freeze = tmp_path / "out_validate_freeze"
    out_pmin = tmp_path / "out_pmin"
    out_bias = tmp_path / "out_bias"
    out_sens = tmp_path / "out_sens"

    assert main(["validate-null", "--G", "3", "--N", "12", "--config", str(config), "--out", str(out_validate_null)]) == 0
    assert main(["validate-freeze", "--config", str(config), "--out", str(out_validate_freeze)]) == 0
    assert main(["pmin-audit", "--gene-set", str(genes_dir), "--N", "12", "--config", str(config), "--out", str(out_pmin)]) == 0
    assert main(["bias-audit", "--gene-set", str(genes_dir), "--N", "12", "--config", str(config), "--out", str(out_bias)]) == 0
    assert main(["sensitivity", "--gene-set", str(genes_dir), "--grid", str(grid), "--config", str(config), "--out", str(out_sens)]) == 0

    for outdir in [out_validate_null, out_validate_freeze, out_pmin, out_bias, out_sens]:
        assert (outdir / "raw" / "results.tsv").exists()
        assert (outdir / "report" / "report.pdf").exists()
        assert (outdir / "manifests" / "provenance_freeze.json").exists()
        assert (outdir / "checksums.txt").exists()
