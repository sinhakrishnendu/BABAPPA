# WRKY Real Empirical Diagnostic Pilot Result

## Decision

Software execution succeeded for the available real WRKY pilot family, but the biological interpretation is diagnostic only.

The WRKY family is out of domain for the current deployable model because the empirical input has very high pairwise p-distance (`0.725799`). BABAPPA therefore marked scores as `diagnostic_only=True`. This is not a positive-selection claim.

## Dataset

- panel_id: `WRKY_candidate_01`
- source note: Ensembl Plants BLASTP-derived WRKY33-like ortholog candidate set
- taxa: `Arabidopsis_thaliana`, `Brassica_rapa`, `Glycine_max`, `Oryza_sativa`, `Sorghum_bicolor`, `Brachypodium_distachyon`
- CDS FASTA: `real_empirical_pilot/input/cds/WRKY_candidate_01.cds.fasta`
- IQ-TREE tree: `real_empirical_pilot/input/trees/WRKY_candidate_01.treefile`
- foreground: `Arabidopsis_thaliana`

## Pipeline Status

- readiness: `ready_to_run=True` for the WRKY-only panel
- pilot run: `ok`
- families processed: `1`
- scoring: `ok`
- score rows: `9870`
- device: `mps`
- tier model: `extreme`
- simulation-matched calibration plan: `planned`
- reference comparison: pending

## Alignment Status

- `identity`: failed because raw CDS sequences are unaligned and unequal length
- `mafft`: ok
- `babappalign`: ok
- `muscle`: ok

## BABAPPA Diagnostic Scores

- applicability: `out_of_domain`
- reason: `very_high_p_distance:0.725799`
- diagnostic_only: `True`
- max gene support: `0`
- called positive branch-site rows: `0`

## Interpretation Boundary

This run validates that BABAPPA can ingest a real downloaded/tree-built empirical family and complete the guarded empirical scoring workflow. It does not validate empirical positive-selection inference for this family because the family is out of domain and reference workflows have not been run.

Recommended next action: run a closer, less saturated pilot family or a curated WRKY subset with lower divergence, then run codeml/HyPhy reference workflows for accepted in-domain or borderline families.
