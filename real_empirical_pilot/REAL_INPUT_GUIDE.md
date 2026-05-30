# Real Empirical Pilot Input Guide

This workspace is for staging a small guarded empirical pilot. It must not be used to fabricate biological data or to claim empirical positive selection.

## Recommended Panel Size

Use 8-12 families:

- 2-3 likely positive/adaptive candidate families
- 2-3 conserved likely negative families
- 1-2 alignment-sensitive families
- 1-2 saturated/divergent families
- 1 short/low-information family
- 1 paralogy-risk family if available

## Good First Candidate Sources

- WRKY transcription factors
- CONSTANS-like transcription factors
- immune genes
- detoxification/GST genes
- host-parasite interaction genes
- conserved housekeeping controls

Start each candidate with a close taxonomic panel. `WRKY_candidate_01` was a successful software stress test, but its mean pairwise p-distance was `0.725799`, which is too divergent for current empirical interpretation. Treat that family as an OOD/failure-mode probe, not as biological evidence.

## Input Requirements

- CDS FASTA may be unaligned or aligned, but it must be codon-valid.
- At least 6 taxa are preferred.
- At least 100 codons are preferred.
- Tree tips must match FASTA IDs.
- Foreground must be one of the taxa or a clearly documented branch label.
- Avoid obvious paralog mixtures unless the row is explicitly categorized as `paralogy_risk`.
- For the first interpretable pilot, aim for 6-10 close taxa, at least 100 codons, mean p-distance below about 0.35, and no obvious paralog mixture.

## Canonical Paths

For each manifest row, place files at:

```text
real_empirical_pilot/input/cds/<panel_id>.cds.fasta
real_empirical_pilot/input/trees/<panel_id>.treefile
```

Then update `real_empirical_pilot/manifest/real_empirical_pilot_panel.tsv`, or use:

```bash
babappa import-real-pilot-family ...
babappa import-real-pilot-batch ...
```

Before running BABAPPA scoring:

```bash
babappa validate-real-pilot-readiness --workspace real_empirical_pilot --manifest real_empirical_pilot_panel.tsv --outdir real_empirical_pilot/readiness
```

Only run the empirical pilot when `ready_to_run` is `true`.

Before adding a new real family to the scoring panel, run:

```bash
babappa prefilter-empirical-family --cds-fasta real_empirical_pilot/input/cds/<panel_id>.cds.fasta --tree-file real_empirical_pilot/input/trees/<panel_id>.treefile --foreground <foreground_taxon> --outdir real_empirical_pilot/prefilter/<panel_id>
```

Only import families with `accept` or `accept_with_caution` decisions as interpretable candidates. OOD families may be retained as diagnostic-only stress tests.
