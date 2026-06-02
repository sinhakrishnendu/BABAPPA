# BABAPPA Statistical Testing Framework

BABAPPA’s evidence system is BABAPPA-native and simulation-trained. It should be reported as calibrated branch-site support, not as a codeml or HyPhy likelihood-ratio p-value.

## Evidence Layers

1. Explicit known-truth simulation benchmark: validates AUROC, AUPRC, FDR, power, calibration, and OOD abstention against known labels.
2. BABAPPA-native null calibration: estimates empirical p-like support for a user family or benchmark family under matched null conditions.
3. Empirical comparator benchmark: compares BABAPPA behavior with codeml/HyPhy outputs, without treating those outputs as truth.
4. Biological controls: conserved negative controls and curated candidate positives evaluate empirical false-call behavior and interpretability.

## Why HyPhy/codeml Are Not Truth

codeml and HyPhy are established likelihood-based reference methods. Their empirical outputs are useful comparators, but real biological datasets do not provide complete known branch-site labels. Sensitivity, specificity, and FDR should therefore be computed on simulated truth, not on empirical comparator calls.

## Native Calibration

BABAPPA-native calibration compares an observed score against matched null families. For publication-level empirical interpretation, calibration should be paired with:

- a defensible in-domain or borderline applicability status;
- sufficient null replicates;
- negative controls;
- external reference-method context;
- biological review of orthology, foreground choice, and alignment quality.

## Claim Boundaries

BABAPPA can support:

- simulation-validated method claims;
- conservative complementary method claims;
- OOD-gated diagnostic scoring;
- calibrated BABAPPA-native support after adequate null calibration.

BABAPPA should not currently be described as:

- a codeml replacement;
- a HyPhy replacement;
- a method proven to recover every likelihood-test positive;
- a source of unsupported empirical positive-selection discovery claims.

