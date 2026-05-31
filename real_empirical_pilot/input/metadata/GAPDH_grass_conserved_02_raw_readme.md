# GAPDH_grass_conserved_02_raw Input Staging

Staged from `gapdh_grass/GAPDH_grass.fasta` and `gapdh_grass/GAPDH_grass.protein.aln.fasta.treefile`.

## Files

- Raw CDS FASTA: `real_empirical_pilot/input/cds/GAPDH_grass_conserved_02_raw.cds.fasta`
- Tree: `real_empirical_pilot/input/trees/GAPDH_grass_conserved_02_raw.treefile`
- Panel manifest: `real_empirical_pilot/manifest/gapdh_grass_raw_control_panel.tsv`

## Sanity Check

- Sequence count: 6
- CDS contains gaps: no
- CDS lengths divisible by 3 after removing gaps: yes
- CDS IDs match tree tips: yes
- Default foreground/reference taxon: `LOC_Os02g07490_LOC_Os02g07490.1`

## Caveat

The updated raw panel removes the duplicate `Phala.08G200500` isoforms. The current limiting issue is divergence: `babappa prefilter-empirical-family` estimates high raw p-distance and classifies this as `diagnostic_only`. Treat it as a saturated diagnostic control unless the alignment-aware empirical run reports an in-domain applicability status.

## Recommended Commands

Validate:

```bash
cd /Users/krishnendu/Documents/GitHub/BABAPPA
babappa validate-empirical-pilot-panel \
  --panel-manifest real_empirical_pilot/manifest/gapdh_grass_raw_control_panel.tsv \
  --outdir real_empirical_pilot/panel_validation_gapdh_grass_raw
```

Prefilter:

```bash
cd /Users/krishnendu/Documents/GitHub/BABAPPA
babappa prefilter-empirical-family \
  --cds-fasta real_empirical_pilot/input/cds/GAPDH_grass_conserved_02_raw.cds.fasta \
  --tree-file real_empirical_pilot/input/trees/GAPDH_grass_conserved_02_raw.treefile \
  --foreground LOC_Os02g07490_LOC_Os02g07490.1 \
  --outdir real_empirical_pilot/prefilter/GAPDH_grass_conserved_02_raw
```

Run BABAPPA diagnostic scoring:

```bash
cd /Users/krishnendu/Documents/GitHub/BABAPPA
babappa run-empirical-pilot-panel \
  --panel-manifest real_empirical_pilot/manifest/gapdh_grass_raw_control_panel.tsv \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir real_empirical_pilot/babappa_run_gapdh_grass_raw_control \
  --methods identity,mafft,babappalign,muscle \
  --device mps \
  --max-families 1
```
