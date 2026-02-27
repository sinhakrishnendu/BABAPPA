# BABAPPA Interpretation

## What a significant BABAPPA result means
A low p-value indicates the observed gene has unusually high (or low, depending on tail) **dispersion of sitewise energy scores** relative to neutral expectations under fixed M0.

## What it does NOT mean by itself
- Not direct evidence of positive selection at a specific codon.
- Not direct branch attribution without branch-aware follow-up.
- Not a mechanistic functional claim.

## Recommended follow-up workflow
1. Rank genes by p/q and effect-scale summaries (`D_obs`, `mu0_hat`, `sigma0_hat`).
2. Cross-check with codeml branch-site/site-model and HyPhy BUSTED/RELAX.
3. Inspect alignment quality and recombination risk for top hits.
4. Map candidate residues to domains/structures.
5. Integrate phenotype/ecology metadata before biological claims.

## Reporting guidance
- Report methods conservatively: “distributed evolutionary burden signal under model M0.”
- Explicitly state model assumptions and potential misspecification.
- Include overlap/disagreement with codeml/HyPhy rather than over-claiming novelty.
