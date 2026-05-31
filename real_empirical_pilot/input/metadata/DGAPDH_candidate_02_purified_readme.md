# DGAPDH_candidate_02_purified Input Staging

Staged from updated `gapdh_grass/dgapdh.fasta` and `gapdh_grass/dgapdh.protein.aln.fasta.treefile`.

## Files

- Raw CDS FASTA: `real_empirical_pilot/input/cds/DGAPDH_candidate_02_purified.cds.fasta`
- Tree: `real_empirical_pilot/input/trees/DGAPDH_candidate_02_purified.treefile`
- Panel manifest: `real_empirical_pilot/manifest/dgapdh_purified_panel.tsv`

## Sanity Check

- Sequence count: 7
- CDS contains gaps: no
- CDS lengths divisible by 3: yes
- Ambiguous bases: none detected
- CDS IDs match tree tips: yes
- Length range: 960-1023 nt
- Length ratio: approximately 1.066
- Default foreground/reference taxon: `SMAR006602-PA`

## Notes

This purified set removes the previous long-branch/outlier sequence `BL20647_evm1`. It is staged as a stronger likely negative-control candidate than `DGAPDH_candidate_01`, but still requires the empirical prefilter, method-policy check, applicability gate, and scoring report before interpretation.

## Recommended Commands

Validate:

```bash
cd /Users/krishnendu/Documents/GitHub/BABAPPA
babappa validate-empirical-pilot-panel \
  --panel-manifest real_empirical_pilot/manifest/dgapdh_purified_panel.tsv \
  --outdir real_empirical_pilot/panel_validation_dgapdh_purified
```

Prefilter:

```bash
cd /Users/krishnendu/Documents/GitHub/BABAPPA
babappa prefilter-empirical-family \
  --cds-fasta real_empirical_pilot/input/cds/DGAPDH_candidate_02_purified.cds.fasta \
  --tree-file real_empirical_pilot/input/trees/DGAPDH_candidate_02_purified.treefile \
  --foreground SMAR006602-PA \
  --outdir real_empirical_pilot/prefilter/DGAPDH_candidate_02_purified
```

Run BABAPPA diagnostic scoring:

```bash
cd /Users/krishnendu/Documents/GitHub/BABAPPA
babappa run-empirical-pilot-panel \
  --panel-manifest real_empirical_pilot/manifest/dgapdh_purified_panel.tsv \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir real_empirical_pilot/babappa_run_dgapdh_purified \
  --methods identity,mafft,babappalign,muscle \
  --device auto \
  --max-families 1
```
