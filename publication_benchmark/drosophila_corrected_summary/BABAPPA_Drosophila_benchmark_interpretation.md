# BABAPPA Drosophila Benchmark Interpretation

HyPhy aBSREL is an external comparator, not ground truth. Concordance metrics are therefore not sensitivity, specificity, power, or false-positive-rate estimates.

The corrected official parser uses HyPhy's own `test results -> positive test results` field. Under that parser, BABAPPA-native calibrated support showed limited positive overlap with HyPhy aBSREL but strong conservative behavior under BABAPPA-defined out-of-domain conditions.

- families: `140`
- BABAPPA raw diagnostic-positive: `17/140`
- BABAPPA-native calibrated support: `14/140`
- HyPhy aBSREL-positive families: `73/140`
- HyPhy positive branches: `185/1680`
- concordant positive: `3`
- concordant negative: `56`
- BABAPPA-only: `11`
- HyPhy-only: `70`

Low positive overlap means BABAPPA is not a HyPhy replacement. The main positive result is OOD abstention: BABAPPA made zero native-supported calls in true out-of-domain families, while HyPhy reported positives in many OOD families. This does not prove HyPhy is wrong; it shows that BABAPPA's current empirical policy is more conservative under BABAPPA-defined OOD conditions.

BABAPPA-only families require BABAPPA-native null calibration, alignment audit, and biological review. HyPhy-only families may reflect likelihood-model sensitivity, model differences, or BABAPPA abstention. The benchmark supports complementary conservative behavior rather than replacement or superiority.
