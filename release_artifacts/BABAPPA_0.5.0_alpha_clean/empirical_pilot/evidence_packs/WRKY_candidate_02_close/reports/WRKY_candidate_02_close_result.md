# WRKY Candidate 02 Close Empirical Diagnostic Result

## Decision

- software_succeeded: `True`
- biological claim: `not final`; diagnostic support only until simulation-matched calibration and reference workflows are complete
- applicability: `in_domain`
- diagnostic_only: `False`

## Input Acquisition

- source: Ensembl REST homology/id and sequence/id endpoints
- query: `Arabidopsis_thaliana AT2G38470 / WRKY33-like`
- sequences downloaded: `7`
- targets: `Arabidopsis_thaliana, Arabidopsis_lyrata, Arabidopsis_halleri, Arabis_alpina, Eutrema_salsugineum, Brassica_oleracea, Brassica_rapa_RO18`
- tree: `IQ-TREE on MAFFT protein alignment`

## OOD / Applicability

- prefilter decision: `accept`
- aligned mean p-distance: `0.101293`
- p-distance used by applicability: `0.097198` from `alignment_ensemble_mean`
- raw unaligned positional p-distance: `0.663538` (kept as warning only because sequences are unequal-length raw CDS)
- recommended tier model: `moderate`

## BABAPPA Scoring

- scoring status: `ok`
- device: `mps`
- tier model: `moderate`
- score rows: `10451`

| method | rows | max probability | mean probability | called positive | diagnostic only |
| --- | ---: | ---: | ---: | ---: | --- |
| mafft | 3416 | 0.177189 | 0.024472 | 2327 | False |
| babappalign | 3703 | 0.176992 | 0.024253 | 2454 | False |
| muscle | 3332 | 0.177136 | 0.024168 | 2173 | False |

## Interpretation

BABAPPA produced in-domain simulation-trained diagnostic branch-site support for this close WRKY family, but this is not a final positive-selection claim until simulation-matched calibration and codeml/HyPhy-style reference comparison are completed.

Do not use this as a final biological discovery claim yet. Run simulation-matched calibration and external reference workflows first.
