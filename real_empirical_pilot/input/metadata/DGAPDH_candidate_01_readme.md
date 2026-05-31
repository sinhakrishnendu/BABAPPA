# DGAPDH_candidate_01 Input Staging

Staged from `gapdh_grass/dgapdh.fasta` and `gapdh_grass/dgapdh.protein.aln.fasta.treefile`.

## Files

- Raw CDS FASTA: `real_empirical_pilot/input/cds/DGAPDH_candidate_01.cds.fasta`
- Tree: `real_empirical_pilot/input/trees/DGAPDH_candidate_01.treefile`
- Panel manifest: `real_empirical_pilot/manifest/dgapdh_candidate_panel.tsv`

## Sanity Check

- Sequence count: 8
- CDS contains gaps: no
- CDS lengths divisible by 3: yes
- Ambiguous bases: none detected
- CDS IDs match tree tips: yes
- Default foreground/reference taxon: `SMAR006602-PA`

## Notes

This is staged as a likely conserved negative-control candidate. It still requires the formal empirical prefilter, method-policy check, applicability gate, and scoring report before interpretation.

## Recommended Commands

Validate:

```bash
cd /Users/krishnendu/Documents/GitHub/BABAPPA
babappa validate-empirical-pilot-panel \
  --panel-manifest real_empirical_pilot/manifest/dgapdh_candidate_panel.tsv \
  --outdir real_empirical_pilot/panel_validation_dgapdh_candidate
```

Prefilter:

```bash
cd /Users/krishnendu/Documents/GitHub/BABAPPA
babappa prefilter-empirical-family \
  --cds-fasta real_empirical_pilot/input/cds/DGAPDH_candidate_01.cds.fasta \
  --tree-file real_empirical_pilot/input/trees/DGAPDH_candidate_01.treefile \
  --foreground SMAR006602-PA \
  --outdir real_empirical_pilot/prefilter/DGAPDH_candidate_01
```

Run BABAPPA diagnostic scoring:

```bash
cd /Users/krishnendu/Documents/GitHub/BABAPPA
babappa run-empirical-pilot-panel \
  --panel-manifest real_empirical_pilot/manifest/dgapdh_candidate_panel.tsv \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir real_empirical_pilot/babappa_run_dgapdh_candidate \
  --methods identity,mafft,babappalign,muscle \
  --device auto \
  --max-families 1
```
