# Drosophila BABAPPA vs HyPhy aBSREL Benchmark

- families summarized: `140`
- HyPhy positive mode: `official`
- BABAPPA raw diagnostic-positive: `17/140`
- BABAPPA-native calibrated support: `14/140`
- HyPhy aBSREL-positive families: `73/140`
- HyPhy positive branches: `185/1680`
- BABAPPA_only: `11`
- HyPhy_only: `70`
- concordant_negative: `56`
- concordant_positive: `3`
- overall agreement: `0.421`
- positive agreement against HyPhy: `0.041`
- negative agreement: `0.836`

## Concordance By Stratum

- `borderline_distance`: HyPhy_only=11, concordant_negative=9
- `borderline_gap`: BABAPPA_only=3, HyPhy_only=6, concordant_negative=11
- `extreme_distance_stress`: HyPhy_only=17, concordant_negative=3
- `ood_high_distance`: HyPhy_only=14, concordant_negative=6
- `ood_high_gap`: BABAPPA_only=2, HyPhy_only=9, concordant_negative=9
- `relaxed_in_domain`: BABAPPA_only=2, HyPhy_only=7, concordant_negative=11
- `strict_in_domain`: BABAPPA_only=4, HyPhy_only=6, concordant_negative=7, concordant_positive=3

## Applicability-Stratified Behavior

- `borderline`: families=48, BABAPPA-native=5, HyPhy-positive=18
- `in_domain`: families=40, BABAPPA-native=9, HyPhy-positive=16
- `out_of_domain`: families=52, BABAPPA-native=0, HyPhy-positive=39

Interpretation: this is a publication benchmark comparator. HyPhy aBSREL is an external branch-level likelihood reference, not BABAPPA's training target or ground truth. Concordance metrics are not sensitivity/specificity.
