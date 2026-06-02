# BABAPPA Benchmark Dataset Plan

This document defines the benchmark suite needed to evaluate BABAPPA as a conservative, OOD-gated, simulation-trained complementary branch-site support framework. No empirical family is treated as absolute truth unless its truth is known by simulation or supported by a separately justified benchmark design.

## 1. Simulated Truth Set

The simulated truth layer is the only layer with known branch-site labels. BABAPPA-BENCH-SIM-v1 is the primary benchmark for this layer and should be used before empirical comparator claims.

Required strata:

- null families with no branch-site positive selection;
- weak, moderate, and strong branch-site positive families;
- low, moderate, high, and extreme saturation tiers;
- alignment perturbation tiers;
- OOD families with divergence, gap burden, taxon count, or length outside the deployable training envelope.

Required archived outputs:

- input CDS/MSA/tree files;
- explicit simulator branch-site truth;
- BABAPPA scores and calls;
- native-null calibration outputs;
- OOD/audit reports;
- parser and software commit metadata;
- benchmark manifest and checksums.

Primary metrics:

- AUROC;
- AUPRC;
- FDR;
- calibration;
- power by effect size and saturation tier;
- OOD abstention behavior.

The standard profiles are smoke, pilot, paper, and extended. Smoke exists for unit testing and command validation. Pilot, paper, and extended are user-run long jobs.

## 2. Empirical Comparator Set

The empirical comparator layer asks how BABAPPA behaves beside external methods, not whether either method is ground truth.

Initial strata:

- Drosophila single-copy orthologs;
- close-taxa plant WRKY and conserved controls;
- curated housekeeping controls;
- alignment-sensitive families;
- saturated families;
- paralogy-risk families.

Required archived outputs:

- input CDS/MSA/tree files;
- BABAPPA scores;
- BABAPPA-native null calibration;
- codeml outputs when feasible;
- HyPhy outputs;
- OOD/audit reports;
- parser mode and parser audit;
- manifest and checksums.

Primary metrics:

- family-level concordance/discordance;
- branch-level comparator summaries when official fields are unambiguous;
- BABAPPA-only and HyPhy/codeml-only case lists;
- OOD abstention and false-call suppression;
- qualitative failure-mode review.

## 3. Gold/Near-Gold Control Set

No empirical branch-site benchmark should be called absolute gold standard without independent evidence. The near-gold layer should include:

- negative housekeeping orthologs expected to remain negative or no-call;
- literature-supported immune/detox/adaptive candidates when available;
- close-taxa panels with manageable p-distance;
- independent biological review of foreground choice and orthology.

Positive literature-supported families must not be treated as perfect truth. They should be used as case studies with external references, BABAPPA-native calibration, and OOD reporting.

## 4. Output Archive

Each benchmark release should archive:

- input CDS/MSA/tree;
- panel manifest;
- BABAPPA branch-site, branch, and gene outputs;
- BABAPPA-native null summaries;
- codeml and HyPhy raw outputs and parsed summaries;
- OOD/applicability and feature-audit reports;
- parser mode and parser audit;
- benchmark summary tables;
- checksums and environment metadata.

Raw downloadable genome/transcriptome bundles and regenerable OrthoFinder intermediates should remain outside the source repository and be archived only when needed for reproducibility.

## 5. Metrics And Claim Boundaries

Use different metrics for different evidence layers:

- simulation truth: AUROC, AUPRC, FDR, power, calibration;
- empirical comparator: concordance/discordance, not sensitivity/specificity;
- OOD: abstention and false-call suppression;
- controls: empirical false-positive probes;
- matched-null calibration: BABAPPA-native empirical p-like support.

HyPhy and codeml are external comparators, not empirical ground truth. A low positive overlap does not automatically mean BABAPPA failed or HyPhy is wrong; it defines the current operating regime and the families that require deeper review.
