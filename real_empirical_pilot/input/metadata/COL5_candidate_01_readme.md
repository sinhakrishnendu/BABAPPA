# COL5_candidate_01 Input Staging

Staged from `gapdh_grass/col5.fasta` and `gapdh_grass/col5.protein.aln.fasta.treefile`.

## Files

- Raw CDS FASTA: `real_empirical_pilot/input/cds/COL5_candidate_01.cds.fasta`
- Tree: `real_empirical_pilot/input/trees/COL5_candidate_01.treefile`
- Panel manifest: `real_empirical_pilot/manifest/col5_candidate_panel.tsv`

## Sanity Check

- Sequence count: 7
- CDS contains gaps: no
- CDS lengths divisible by 3: yes
- CDS IDs match tree tips: yes
- Default foreground/reference taxon: `Cucsa.092180|Cucsa.092180.1`

## Caution

One sequence, `Medtr3g034680|Medtr3g034680.1`, is much shorter than the others. Treat this as an alignment-sensitive diagnostic panel until the empirical prefilter, method policy, and applicability reports show that the family is interpretable.

## Recommended Commands

Validate:

```bash
cd /Users/krishnendu/Documents/GitHub/BABAPPA
babappa validate-empirical-pilot-panel \
  --panel-manifest real_empirical_pilot/manifest/col5_candidate_panel.tsv \
  --outdir real_empirical_pilot/panel_validation_col5_candidate
```

Prefilter:

```bash
cd /Users/krishnendu/Documents/GitHub/BABAPPA
babappa prefilter-empirical-family \
  --cds-fasta real_empirical_pilot/input/cds/COL5_candidate_01.cds.fasta \
  --tree-file real_empirical_pilot/input/trees/COL5_candidate_01.treefile \
  --foreground 'Cucsa.092180|Cucsa.092180.1' \
  --outdir real_empirical_pilot/prefilter/COL5_candidate_01
```

Run BABAPPA diagnostic scoring:

```bash
cd /Users/krishnendu/Documents/GitHub/BABAPPA
babappa run-empirical-pilot-panel \
  --panel-manifest real_empirical_pilot/manifest/col5_candidate_panel.tsv \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir real_empirical_pilot/babappa_run_col5_candidate \
  --methods identity,mafft,babappalign,muscle \
  --device auto \
  --max-families 1
```
