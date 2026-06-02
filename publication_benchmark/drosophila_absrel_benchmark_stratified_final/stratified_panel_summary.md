# Stratified Drosophila BABAPPA/aBSREL Benchmark Panel

- combined panel: `publication_benchmark/drosophila_absrel_benchmark_stratified_final/stratified_drosophila_babappa_absrel_panel.tsv`
- total families selected: `140`
- families requested per stratum: `20`

## Rationale

The panel is stratified by applicability-related properties so the publication benchmark can report:

- clean in-domain behavior
- near-boundary behavior
- OOD abstention and stress-test behavior
- concordance/discordance with HyPhy aBSREL within each stratum

This is more defensible than reporting a single average across heterogeneous orthologs.

## Strata

- `strict_in_domain`: Low-divergence, low-gap families expected to be BABAPPA in-domain. selected `20` families
- `relaxed_in_domain`: Moderate but still clean families near the upper in-domain range. selected `20` families
- `borderline_distance`: Low-gap families whose divergence is expected to push BABAPPA toward borderline. selected `20` families
- `borderline_gap`: Moderate-divergence families with elevated gap burden. selected `20` families
- `ood_high_distance`: Low-gap but high-divergence OOD stress-test families. selected `20` families
- `ood_high_gap`: High-gap OOD stress-test families with non-extreme divergence. selected `20` families
- `extreme_distance_stress`: Very high-divergence families used to test abstention and failure modes. selected `20` families
