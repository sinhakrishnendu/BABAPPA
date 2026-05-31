# GAPDH_grass_conserved_01 Input Staging

Staged from `gapdh_grass/` on 2026-05-31.

## Files

- CDS/codon alignment: `real_empirical_pilot/input/cds/GAPDH_grass_conserved_01.cds.fasta`
- Codon MSA copy: `real_empirical_pilot/input/msas/GAPDH_grass_conserved_01.codon.aln.fasta`
- Protein MSA copy: `real_empirical_pilot/input/msas/GAPDH_grass_conserved_01.protein.aln.fasta`
- Tree: `real_empirical_pilot/input/trees/GAPDH_grass_conserved_01.treefile`
- Panel manifest: `real_empirical_pilot/manifest/gapdh_grass_control_panel.tsv`

## Sanity Check

- Sequence count: 9
- Aligned CDS length: 1257 nt
- Codon length divisible by 3: yes
- CDS IDs match tree tips: yes
- Foreground/default reference taxon: `LOC_Os02g07490_LOC_Os02g07490.1`

## Important Caution

`babappa prefilter-empirical-family` currently classifies this panel as `reject_possible_paralogy`.
The likely reason is that three `Phala.08G200500` transcript isoforms are present:

- `Phala.08G200500_Phala.08G200500.1`
- `Phala.08G200500_Phala.08G200500.2`
- `Phala.08G200500_Phala.08G200500.3`

For a clean conserved negative control, curate one ortholog/transcript per species and rebuild or prune the tree before interpreting BABAPPA scores. As staged, this family is suitable as a paralogy-risk diagnostic input, not as a clean negative-control conclusion.

## Commands

Validate staged input:

```bash
cd /Users/krishnendu/Documents/GitHub/BABAPPA
babappa validate-empirical-pilot-panel \
  --panel-manifest real_empirical_pilot/manifest/gapdh_grass_control_panel.tsv \
  --outdir real_empirical_pilot/panel_validation_gapdh_grass
```

Run the OOD/paralogy prefilter:

```bash
cd /Users/krishnendu/Documents/GitHub/BABAPPA
babappa prefilter-empirical-family \
  --cds-fasta real_empirical_pilot/input/cds/GAPDH_grass_conserved_01.cds.fasta \
  --tree-file real_empirical_pilot/input/trees/GAPDH_grass_conserved_01.treefile \
  --foreground LOC_Os02g07490_LOC_Os02g07490.1 \
  --outdir real_empirical_pilot/prefilter/GAPDH_grass_conserved_01
```

Run BABAPPA only if you intentionally want a paralogy-risk diagnostic:

```bash
cd /Users/krishnendu/Documents/GitHub/BABAPPA
babappa run-empirical-pilot-panel \
  --panel-manifest real_empirical_pilot/manifest/gapdh_grass_control_panel.tsv \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir real_empirical_pilot/babappa_run_gapdh_grass_control \
  --methods identity,mafft,babappalign,muscle \
  --device mps \
  --max-families 1
```
