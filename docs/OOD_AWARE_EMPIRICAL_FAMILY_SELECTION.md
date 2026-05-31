# OOD-Aware Empirical Family Selection

BABAPPA empirical mode is guarded and simulation-trained. It must abstain or mark results as diagnostic-only when a real family is outside the training and calibration envelope.

## Why This Layer Exists

`WRKY_candidate_01` processed successfully through the software pipeline: CDS/tree readiness, MAFFT, BABAPPAlign, MUSCLE, MPS scoring, and report generation all worked. It is therefore useful as a stress-test/failure-mode case.

It is not interpretable as a biological discovery. The family had mean pairwise p-distance `0.725799`, which is too high for current BABAPPA empirical use. This level of divergence suggests saturation risk, alignment sensitivity, and possible family-mix or orthology problems. BABAPPA should treat it as out-of-domain and diagnostic-only.

## Recommended First Pilot Envelope

For the first real empirical pilots, prefer:

- close taxonomic panel
- 6-10 taxa
- at least 100 codons
- mean pairwise p-distance ideally below `0.35`
- manageable gap and ambiguous-base burden
- foreground taxon present in FASTA and tree
- one likely ortholog per species
- no obvious paralog mixture

Broad plant panels can become saturated quickly. For Arabidopsis WRKY-like pilots, start with a Brassicaceae-heavy panel before mixing monocots, legumes, and deeply divergent angiosperms.

The close Brassicaceae redesign produced `WRKY_candidate_02_close`, which passed the OOD gate and became the first in-domain diagnostic empirical pilot. That success means the selection strategy worked; it does not by itself establish biological positive selection. The accepted family now has codeml/HyPhy-style reference comparison and a completed feature-level 100-null matched calibration, but those results are mixed: reference tools are negative, while the BABAPPA called-row burden is unusual against feature-level nulls. Manuscript-level interpretation still requires close-taxa controls and conservative biological review.

## Long-Run Handoff Policy

Codex does not execute heavy empirical calibration, broad scans, or long reference batches. OOD-aware acquisition and control workflows are written as USER-RUN scripts; the user runs them locally/offline and brings back summaries for interpretation.

## Commands

```bash
babappa prefilter-empirical-family --cds-fasta real_empirical_pilot/input/cds/WRKY_candidate_01.cds.fasta --tree-file real_empirical_pilot/input/trees/WRKY_candidate_01.treefile --foreground Arabidopsis_thaliana --outdir real_empirical_pilot/prefilter/WRKY_candidate_01
babappa recommend-target-taxa --pilot-type plant_close --outdir real_empirical_pilot/target_taxa_recommendations
babappa plan-ood-aware-family-build --family-id WRKY_candidate_02_close --query-species Arabidopsis_thaliana --query-gene-or-locus AT2G38470 --target-taxa-file real_empirical_pilot/target_taxa_recommendations/recommended_target_taxa.tsv --outdir real_empirical_pilot/acquisition_plans/WRKY_candidate_02_close --max-mean-pdistance 0.35 --min-taxa 6 --min-codons 100
babappa summarize-empirical-ood --workspace real_empirical_pilot --outdir real_empirical_pilot/ood_summary
```

The acquisition plans are USER-RUN ONLY. They are meant to help build a closer candidate family, then prefilter it before adding it to the empirical pilot manifest.

## Decision Classes

- `accept`: eligible for a guarded empirical pilot after reference planning.
- `accept_with_caution`: usable, but interpret with stronger alignment/reference caution.
- `reject_too_divergent`: reduce taxonomic breadth before scoring.
- `reject_too_short`: choose a longer coding sequence or another family.
- `reject_too_few_taxa`: add close orthologs.
- `reject_tree_mismatch`: repair FASTA/tree tips or foreground.
- `reject_possible_paralogy`: curate orthologs and remove obvious paralogs.
- `diagnostic_only`: useful as a stress test, not as an empirical positive-selection call.

Out-of-domain cases are not positive-selection calls.
