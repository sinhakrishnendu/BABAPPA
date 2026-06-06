# Positive-Control Data

Downloaded and prepared small control candidates for BABAPPA publication-revision benchmarks.

Source: official HyPhy tutorial dataset bundle from the HyPhy website/GitHub. The bundle includes alignments with appended Newick trees; these were split into separate codon MSA FASTA and `.treefile` files for BABAPPA.

Prepared controls:

- `known_positive_hiv_env`: HIV-1 env donor-recipient transmission dataset used in HyPhy aBSREL tutorials.
- `known_positive_ksr2_primate`: KSR2 primate dataset used in HyPhy BUSTED tutorials.
- `known_positive_abalone_lysin`: abalone lysin dataset used in HyPhy FEL/MEME/SLAC tutorials.
- `known_positive_influenza_h3_ha_optional`: influenza H3 HA trunk dataset used in HyPhy FUBAR tutorials; optional because it has 163 taxa.
- `known_negative_gapdh_grass`: local BABAPPA GAPDH grass negative-control candidate copied from existing user-curated inputs.

The plant R-gene/NLR row is intentionally left pending. No plant positive-control MSA/tree was fabricated.
