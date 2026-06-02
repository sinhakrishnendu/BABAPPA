# Manuscript Rescue Verdict

## Verdict

The method is not wasted. The manuscript is salvageable as a conservative complementary software/method paper.

BABAPPA should not be presented as a replacement for codeml, HyPhy, or likelihood-ratio branch-site tests. The Drosophila benchmark shows limited positive overlap with HyPhy aBSREL, and that discordance must be reported transparently.

## What The Drosophila Benchmark Supports

The corrected Drosophila benchmark supports an OOD-gated conservatism narrative:

- BABAPPA-native calibrated support was present in 14 of 140 families.
- HyPhy aBSREL reported positives in 73 of 140 families.
- Concordant positives were limited to 3 families.
- Concordant negatives were 56 families.
- BABAPPA-only positives were 11 families.
- HyPhy-only positives were 70 families.
- BABAPPA made zero native-supported calls in true out-of-domain families.
- HyPhy reported positives in many BABAPPA-defined OOD families.

This does not prove HyPhy is wrong. It shows that BABAPPA behaves as a conservative, simulation-trained, applicability-aware scorer under its own OOD policy.

## Correct Manuscript Position

BABAPPA can be submitted as:

- a research-alpha branch-site support framework;
- a simulation-trained method with explicit branch-site truth validation;
- a method whose primary validation layer is a known-truth simulation benchmark;
- a conservative complementary empirical scorer;
- an OOD-gated tool that abstains or suppresses support outside its current training envelope;
- a platform for future calibrated empirical benchmarks.

BABAPPA should not be submitted as:

- a HyPhy replacement;
- a codeml replacement;
- a method proven to recover all likelihood-test positives;
- a standalone empirical discovery engine without controls and matched-null calibration.

## Required Before Stronger Claims

Before higher-impact empirical claims, BABAPPA needs:

- a completed paper-scale BABAPPA-BENCH-SIM-v1 known-truth benchmark;
- expanded negative controls;
- full matched-null calibration across representative empirical families;
- broader benchmark panels with curated biological interpretation;
- transparent BABAPPA-only and HyPhy-only failure-mode review;
- archived parser audits and benchmark manifests.

## Final Recommendation

Submit only as a conservative complementary method/software benchmark. The strongest current story is not high concordance with HyPhy; it is a guarded, simulation-trained, OOD-aware branch-site support system whose empirical positives are deliberately sparse and auditable.
