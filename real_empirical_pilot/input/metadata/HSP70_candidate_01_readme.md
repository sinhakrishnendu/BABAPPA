# HSP70_candidate_01 Input Staging

Staged from `gapdh_grass/hsp70.fasta` and `gapdh_grass/hsp70.protein.aln.fasta.treefile`.

## Files

- Raw CDS FASTA: `real_empirical_pilot/input/cds/HSP70_candidate_01.cds.fasta`
- Tree: `real_empirical_pilot/input/trees/HSP70_candidate_01.treefile`
- Panel manifest: `real_empirical_pilot/manifest/hsp70_candidate_panel.tsv`

## Sanity Check

- Sequence count: 8
- CDS contains gaps: no
- CDS lengths divisible by 3: yes
- CDS IDs match tree tips: yes
- Default foreground/reference taxon: `Cucsa.010680|Cucsa.010680.1`

## Caution

This panel contains two `ChfasH1.6G167600` isoforms:

- `ChfasH1.6G167600|ChfasH1.6G167600.1`
- `ChfasH1.6G167600|ChfasH1.6G167600.2`

It also has large length differences, including `Psat06G0270900|Psat06G0270900-T1` at 1011 nt compared with several sequences near or above 1900 nt. Treat this as a paralogy/alignment-sensitive diagnostic panel until the empirical prefilter, method policy, and applicability reports support interpretation.

## Recommended Commands

Validate:

```bash
cd /Users/krishnendu/Documents/GitHub/BABAPPA
babappa validate-empirical-pilot-panel \
  --panel-manifest real_empirical_pilot/manifest/hsp70_candidate_panel.tsv \
  --outdir real_empirical_pilot/panel_validation_hsp70_candidate
```

Prefilter:

```bash
cd /Users/krishnendu/Documents/GitHub/BABAPPA
babappa prefilter-empirical-family \
  --cds-fasta real_empirical_pilot/input/cds/HSP70_candidate_01.cds.fasta \
  --tree-file real_empirical_pilot/input/trees/HSP70_candidate_01.treefile \
  --foreground 'Cucsa.010680|Cucsa.010680.1' \
  --outdir real_empirical_pilot/prefilter/HSP70_candidate_01
```

Run BABAPPA diagnostic scoring:

```bash
cd /Users/krishnendu/Documents/GitHub/BABAPPA
babappa run-empirical-pilot-panel \
  --panel-manifest real_empirical_pilot/manifest/hsp70_candidate_panel.tsv \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir real_empirical_pilot/babappa_run_hsp70_candidate \
  --methods identity,mafft,babappalign,muscle \
  --device auto \
  --max-families 1
```
