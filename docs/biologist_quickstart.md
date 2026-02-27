# BABAPPA Biologist Quickstart

This guide gives a practical neutral-to-discovery workflow for codon-aligned genes.

## 1) Build codon alignments
- Start from CDS nucleotide sequences for orthologous genes.
- Use codon-aware alignment tools (recommended: PRANK codon mode, MACSE).
- Keep reading frame and sequence lengths divisible by 3.
- Remove genes with unresolved ORFs, pervasive ambiguities, or obvious recombinant mosaics.

## 2) Build/fix species tree with branch lengths
- Create a rooted/unrooted Newick tree with branch lengths for the same taxa as your alignments.
- Resolve polytomies unless you explicitly allow them.

## 3) Define neutral model (M0) and freeze it
Create `neutral_model.json`:

```json
{
  "schema_version": 1,
  "model_family": "GY94",
  "genetic_code_table": "standard",
  "tree_file": "data/species_tree.nwk",
  "kappa": 2.0,
  "omega": 1.0,
  "codon_frequencies_method": "F3x4",
  "frozen_values": true
}
```

## 4) Validate inputs before any analysis

```bash
babappa doctor \
  --genes-dir data/genes \
  --tree-file data/species_tree.nwk \
  --neutral-model-json data/neutral_model.json \
  --taxa-policy strict \
  --seed 42 \
  --output-dir results
```

If this fails, fix the reported issues first.

## 5) Train frozen energy model
Recommended starting values for publication-grade runs:
- `n` (training samples): keep `Lmax/n <= 0.1` (warning threshold), never exceed `0.25` without explicit override.
- Use simulation training from M0 for robust phylogenetic calibration.

```bash
babappa train \
  --neutral-model-json data/neutral_model.json \
  --sim-replicates 400 \
  --sim-length 1200 \
  --seed 42 \
  --model results/frozen_energy_model.json \
  --manifest results/manifests/train_manifest.json
```

## 6) Analyze genes (single/batch)
Recommended Monte Carlo calibration:
- pilot: `N=199`
- manuscript-ready: `N=999` or larger

```bash
babappa batch \
  --model results/frozen_energy_model.json \
  --alignments data/genes/*.fasta \
  --calibration-mode phylo \
  --calibration-size 999 \
  --tail right \
  --seed 42 \
  --output results/babappa_results.tsv \
  --manifest results/manifests/batch_manifest.json
```

## 7) Apply BH FDR

```bash
babappa fdr \
  --input results/babappa_results.tsv \
  --output results/babappa_results_fdr.tsv \
  --p-column p \
  --q-column q_value
```

## 8) Biological interpretation and follow-up
- Significant BABAPPA genes indicate **excess dispersion of learned neutral-energy scores across sites** relative to M0.
- This is evidence of distributed evolutionary burden, not direct proof of positive selection at specific codons.
- Follow up with codon-level and branch-level analyses (codeml/HyPhy), structure mapping, and functional annotation.

## Common failure modes
- Bad codon alignment (frameshifts, length not divisible by 3).
- Internal stop codons not expected biologically.
- Taxa mismatch between alignments and tree.
- Recombination violating single-tree assumptions.
- Severe composition bias or model misspecification.
- Too-small training sample size (`L/n` too large).
