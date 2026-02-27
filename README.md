# BABAPPA

BABAPPA (Bayesian Aggregated Burden Analysis via Probabilistic Projection and Approximation) is a Python engine and CLI for detecting distributed evolutionary burden from sequence alignments.

This implementation follows the manuscript-level workflow:

1. Build deterministic site representations from alignments.
2. Train a neutral energy model from neutral alignments or tree-driven neutral simulation.
3. Compute gene-level dispersion in energy space.
4. Calibrate significance with Monte Carlo null simulations (phylogenetic or Gaussian feature null).

## Hypothesis Scope

- `H0`: a gene alignment is generated under a specified neutral codon model on a specified phylogeny.
- `H1`: the gene departs from that neutral mechanism and induces unusual dispersion of sitewise neutral energies.

BABAPPA is a deviation-from-neutrality detector. A significant gene is not, by itself, proof of positive selection; model mis-specification and data artifacts can also trigger significance.

## Install

```bash
pip install babappa
```

For local development:

```bash
python -m pip install -e .
```

## Documentation

- `docs/biologist_quickstart.md`
- `docs/model_spec.md`
- `docs/benchmarking.md`
- `docs/interpretation.md`

## CLI

### 1) Validate dataset/model inputs (`doctor`)

```bash
babappa doctor \
  --genes-dir data/genes \
  --tree-file data/species_tree.nwk \
  --neutral-model-json data/neutral_model.json \
  --seed 42 \
  --output-dir results
```

### 2) Train from neutral FASTA alignments

```bash
babappa train \
  --neutral neutral_gene1.fasta neutral_gene2.fasta \
  --model babappa_model.json \
  --seed 42
```

### 3) Train from a neutral phylogenetic model (GY94)

```bash
babappa train \
  --tree "((A:0.1,B:0.1):0.05,C:0.12);" \
  --sim-replicates 300 \
  --sim-length 900 \
  --kappa 2.0 \
  --omega 1.0 \
  --model babappa_model_phylo.json \
  --seed 42
```

Use `--tree-file tree.nwk` instead of `--tree` to load Newick from disk.

### 4) Analyze one gene

```bash
babappa analyze \
  --model babappa_model_phylo.json \
  --alignment observed_gene.fasta \
  --calibration-mode auto \
  --tail right \
  --calibration-size 2000 \
  --output observed_gene_result.json \
  --manifest observed_gene_manifest.json
```

`--calibration-mode auto` selects phylogenetic calibration when the model was trained with a tree.
Available modes: `auto`, `phylo`, `gaussian`.
Tail options: `right`, `left`, `two-sided`.

### 5) Analyze many genes

```bash
babappa batch \
  --model babappa_model_phylo.json \
  --alignments gene_a.fasta gene_b.fasta gene_c.fasta \
  --calibration-mode auto \
  --tail right \
  --calibration-size 2000 \
  --output results.tsv
```

### 6) Apply BH FDR

```bash
babappa fdr \
  --input results.tsv \
  --output results_fdr.tsv \
  --p-column p \
  --q-column q_value
```

### 7) Run external baselines

```bash
babappa baseline run \
  --method busted \
  --input dataset.json \
  --out busted_results.tsv
```

`--method` choices: `busted`, `relax` (publication default) and `codeml` (legacy).

Validate baseline runtime/parsing readiness first:

```bash
babappa baseline doctor --hyphy --container docker --timeout 5
```

If baseline doctor fails, benchmark baseline-comparison runs stop unless `--allow-baseline-fail` is set.

### 8) Generate benchmark results packs (integrated tracks)

Simulation null calibration:

```bash
babappa benchmark run \
  --track simulation \
  --preset simulation_null_full \
  --outdir results/simulation_null_full \
  --seed 42 \
  --include-baselines
```

Simulation power with baselines:

```bash
babappa benchmark run \
  --track simulation \
  --preset simulation_power_full \
  --outdir results/simulation_power_full \
  --seed 42 \
  --include-baselines \
  --include-relax
```

Ortholog and viral real-data tracks:

```bash
babappa benchmark run --track ortholog --preset ORTHOMAM_SMALL8_FULL --outdir results/ortholog_small8 --seed 42 --include-baselines
babappa benchmark run --track viral --preset HIV_ENV_B_FULL --outdir results/viral_hiv_env_b --seed 42 --include-baselines --include-relax
```

Empirical-only runner (no synthetic fallback, strict QC by default):

```bash
babappa benchmark realdata \
  --preset ortholog_real_v12 \
  --data data/orthomam_small8 \
  --N 999 \
  --seed 1
```

`--outdir` is optional and defaults to `results/<preset>`. For results-only execution (no figures), add `--results-only` (alias: `--no-plots`).

By default, benchmark baseline-comparison runs require baseline doctor to pass. To continue anyway (recording failures), add:

```bash
--allow-baseline-fail
```

Real-data tracks also fail on doctor failures unless `--allow-qc-fail` is set.

Each benchmark emits a complete pack with `raw/`, `tables/`, `figures/`, `manifests/`, `logs/`, `scripts/rebuild_all.sh`, `checksums.txt`, and `report/report.pdf`.

Rebuild figures and tables from raw:

```bash
bash results/simulation_null_full/scripts/rebuild_all.sh
```

### 9) Explain one gene (biologist-facing interpretability)

```bash
babappa explain \
  --model model.json \
  --alignment gene.fasta \
  --calibration-size 999 \
  --tail right \
  --out explain_report.pdf \
  --json-out explain_report.json
```

### 10) Reproduce paper-oriented benchmark sequence

```bash
babappa reproduce \
  --mode publication \
  --outdir paper_runs \
  --seed 42 \
  --baseline-container docker
```

### 11) Prepare dataset bundles for ortholog/viral tracks

```bash
babappa dataset import orthomam \
  --source-dir curated/orthomam_small8 \
  --release v13 \
  --species-set small8 \
  --min-length 300 \
  --max-genes 800 \
  --outdir data/orthomam_small8

babappa dataset import hiv \
  --alignment-fasta env.fasta \
  --alignment-id LANL_ALIGNMENT_ID \
  --provenance provenance.json \
  --recombination-policy subtype_filter \
  --outdir data/hiv_env_b

babappa dataset import sarscov2 \
  --fasta genomes.fasta \
  --metadata metadata.tsv \
  --provenance provenance.json \
  --outdir data/sars_2020

# automated fetchers (network/backend permitting)
babappa dataset fetch orthomam --version v12 --mode remote --n-markers 200 --seed 1 --outdir data/orthomam_v12_real
babappa dataset fetch sarscov2 --source ncbi --host human --complete-only --include-cds --stratify month,country --n 2000 --seed 1 --outdir data/sars_2020_real
```

SARS import writes ORF-focused units (`ORF1ab` chunks, `Spike`, `N`) after QC filtering so realdata viral presets can run full-gene and sliding-window analyses.

Dataset cache utilities:

```bash
babappa dataset cache --show
babappa dataset clear-cache --name orthomam
```

Use paper presets directly in integrated benchmarking:
- `simulation_null_paper`
- `simulation_power_paper`
- `ORTHOMAM_SMALL8_PAPER`
- `HIV_ENV_B_PAPER`
- `SARS_2020_PAPER`

Full-scale aliases for empirical runs:
- `simulation_null_full`
- `simulation_power_full`
- `ORTHOMAM_SMALL8_FULL`
- `HIV_ENV_B_FULL`
- `SARS_2020_FULL`

## Statistical Test Definition

BABAPPA now uses a rank-based Monte Carlo p-value with add-one correction:

- right tail: `p = (1 + #{D_null >= D_obs}) / (N + 1)`
- left tail: `p = (1 + #{D_null <= D_obs}) / (N + 1)`
- two-sided: rank on `|D - c|` with `c` as pooled median of `{D_obs} ∪ {D_null}`

This avoids anti-conservative zero p-values and is aligned with finite-sample Monte Carlo testing principles when the statistic is symmetrically defined.

## Statistical Notes

- BABAPPA defines `n` as the number of training site-columns (rows in feature matrix) used for score matching.
- BABAPPA recommends `n > L_max` where `L_max` is the longest tested alignment length.
- BABAPPA recommends calibration size `N >= 1000` for stable Monte Carlo estimates.
- The model is trained once and frozen before calibration to preserve exchangeability.

## Default Method Specification

- Default representation map `phi` (8 features): gap fraction, ambiguous fraction, major fraction, minor fraction, normalized entropy, GC fraction, mean hydrophobicity, hydrophobicity SD.
- Default energy model `E_theta`: regularized quadratic/Gaussian score-matching form with closed-form precision estimate.
- Default phylogenetic neutral simulator: GY94-like codon CTMC on user-specified Newick tree (`omega=1` by default).

## Output Fields

- `dispersion`: observed variance of per-site energies.
- `mu0`, `sigma0`: null mean and standard deviation from calibration.
- `identifiability_index`: normalized dispersion `(D - mu0) / sigma0`.
- `p_value`: rank-based Monte Carlo p-value (tail specified by `tail`).
- `calibration_mode`: null calibration used (`phylo` or `gaussian`).
- `tail`: alternative direction (`right`, `left`, `two-sided`).
- `q_value`: Benjamini-Hochberg adjusted p-value (batch mode).

## Reproducibility

Use `--manifest path.json` with `train`, `analyze`, `batch`, or `benchmark` to write a reproducibility manifest containing command arguments, BABAPPA version, timestamp, and outputs.
