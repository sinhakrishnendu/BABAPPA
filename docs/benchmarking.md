# BABAPPA Benchmarking

## Integrated benchmark orchestrator

Use the unified command:

```bash
babappa benchmark run --track TRACK --preset PRESET --outdir OUTDIR --seed 42
```

Tracks:
- `simulation`
- `ortholog`
- `viral`

## Canonical presets

Simulation:
- `null_full`
- `power_full`
- `simulation_null_full`
- `simulation_power_full`
- `null_smoke`
- `power_smoke`
- `simulation_null_paper`
- `simulation_power_paper`

Ortholog:
- `ORTHOMAM_SMALL8`
- `ORTHOMAM_MEDIUM16`
- `ORTHOMAM_SMALL8_PAPER`
- `ORTHOMAM_SMALL8_FULL`

Viral:
- `HIV_ENV_B`
- `SARS_2020_GLOBAL`
- `HIV_ENV_B_PAPER`
- `SARS_2020_PAPER`
- `HIV_ENV_B_FULL`
- `SARS_2020_FULL`

## Required paper-style runs

```bash
babappa baseline doctor --methods codeml,busted,relax --container docker --timeout 5
babappa benchmark run --track simulation --preset simulation_null_full --outdir results/simulation_null_full --seed 42 --include-baselines --baseline-container docker
babappa benchmark run --track simulation --preset simulation_power_full --outdir results/simulation_power_full --seed 42 --include-baselines --include-relax --baseline-container docker
babappa benchmark run --track ortholog --preset ORTHOMAM_SMALL8_FULL --outdir results/ortholog_small8 --seed 42 --include-baselines --baseline-container docker
babappa benchmark run --track viral --preset HIV_ENV_B_FULL --outdir results/viral_hiv_env_b --seed 42 --include-baselines --include-relax --baseline-container docker
```

Real-data tracks accept `--dataset-json` (otherwise synthetic fallback is used):

```bash
babappa benchmark run --track ortholog --preset ORTHOMAM_SMALL8 --dataset-json datasets/ortholog/ORTHOMAM_SMALL8/dataset.json --outdir results/ortholog_small8 --seed 42 --include-baselines
```

Empirical-only mode (strict no-fallback + QC gate):

```bash
babappa benchmark realdata --preset ortholog_real_v12 --data data/orthomam_small8 --N 999 --seed 1
```

If doctor reports failures, empirical runs stop unless `--allow-qc-fail` is provided.
`--outdir` defaults to `results/<preset>`. For results-only runs, use `--results-only` (or `--no-plots`).

Prepare dataset bundles:

```bash
babappa dataset import orthomam --source-dir curated/orthomam_small8 --release v13 --species-set small8 --min-length 300 --max-genes 800 --outdir data/orthomam_small8
babappa dataset import hiv --alignment-fasta env.fasta --alignment-id LANL_ALIGNMENT_ID --provenance provenance.json --recombination-policy subtype_filter --outdir data/hiv_env_b
babappa dataset import sarscov2 --fasta genomes.fasta --metadata metadata.tsv --provenance provenance.json --outdir data/sars_2020
babappa dataset fetch orthomam --version v12 --mode remote --n-markers 200 --seed 1 --outdir data/orthomam_v12_real
babappa dataset fetch sarscov2 --source ncbi --host human --complete-only --include-cds --stratify month,country --n 2000 --seed 1 --outdir data/sars_2020_real
babappa dataset cache --show
```

SARS imports generate ORF-level alignments (`ORF1ab` chunks, `Spike`, `N`) after QC so downstream viral realdata presets can produce ORF- and window-level outputs.

## Output pack layout

Every run writes:
- `raw/`
- `tables/`
- `figures/`
- `manifests/`
- `logs/`
- `scripts/rebuild_all.sh`
- `report/report.pdf`
- `checksums.txt`

## Rebuild from raw

```bash
bash results/simulation_null_full/scripts/rebuild_all.sh
```

Or:

```bash
babappa benchmark rebuild --outdir results/simulation_null_full
```

## FAST mode for local validation

For short smoke validations in constrained environments:

```bash
BABAPPA_BENCHMARK_FAST=1 babappa benchmark run --track simulation --preset null_full --outdir /tmp/babappa_demo/null --seed 42
```

This keeps the same pipeline and manifests but uses smaller grids.

## Notes on baselines

- Baseline failures are never dropped.
- Failed baseline rows are recorded with `status=FAIL` and explicit `reason` in raw and `T6_failure_rates.tsv`.
- Baseline comparison runs require `babappa baseline doctor` to pass unless `--allow-baseline-fail` is set.
- To obtain real baseline p-values, ensure local binaries or pinned containers for PAML/HyPhy are available.

## Calibration integrity checks (supplement-ready)

For ortholog packs, run:

```bash
babappa audit ortholog --pack results/ortholog_small8_publication_full_v9 --outdir results/ortholog_small8_publication_full_v9_audit --null_N 999 --null_G 2000 --seed 1
```

The audit now writes explicit integrity tables:
- `tables/frozen_energy_invariant.tsv`: verifies frozen-model hash and M0 hash consistency across analyzed rows.
- `tables/null_uniformity.tsv`: null p-value histogram/QQ companion metrics including continuous KS and discrete-grid KS diagnostics.
- `tables/testable_set_alignment.tsv`: checks BABAPPA and BUSTED testable-set alignment over the same ortholog unit universe.
- `tables/fdr_q_diff.tsv`: BH recomputation diff check against stored q-values.

Build/CI gating behavior:
- `babappa audit ortholog` exits non-zero on severe inflation, frozen-energy invariant failure, or calibration failure.
- `manifests/audit_manifest.json` records `severe_inflation`, `frozen_energy_invariant_fail`, `testable_set_mismatch`, and reproducibility metadata (`phi`, `M0`, `n`, `N`, seeds).
