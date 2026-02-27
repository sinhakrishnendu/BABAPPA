# BABAPPA Practical Revision Notes

This file maps major reviewer critiques to concrete implementation changes in this repository.

## P0 Critiques Addressed

1. Monte Carlo p-value validity and exactness overclaim
- Implemented rank-based Monte Carlo p-values with add-one correction in [`src/babappa/stats.py`](./src/babappa/stats.py).
- Added explicit test tail control (`right`, `left`, `two-sided`) in engine and CLI.
- Replaced previous centered-threshold p-value logic with rank-based testing in [`src/babappa/engine.py`](./src/babappa/engine.py).

2. Missing end-to-end practical software
- Added full CLI support for training, analysis, batch testing, and benchmarking in [`src/babappa/cli.py`](./src/babappa/cli.py).
- Added reproducibility manifests (`--manifest`) for all main commands.

3. Missing empirical validation infrastructure
- Added built-in simulation benchmark workflow in [`src/babappa/evaluation.py`](./src/babappa/evaluation.py):
  - null calibration (type-I error),
  - power under biologically grounded omega-mixture alternatives,
  - comparison against a simple entropy baseline.

4. Underspecified neutral simulation and parameters
- Added explicit phylogenetic simulation modules:
  - Newick parser: [`src/babappa/phylo.py`](./src/babappa/phylo.py)
  - GY94-like codon model: [`src/babappa/codon.py`](./src/babappa/codon.py)
  - Neutral simulator and persisted neutral spec: [`src/babappa/neutral.py`](./src/babappa/neutral.py), [`src/babappa/model.py`](./src/babappa/model.py)

## P1 Practicality Improvements

1. Clarified training sample definition
- README now explicitly defines `n` as number of training site-columns (rows in the feature matrix).

2. Reproducibility and disclosure
- Model JSON stores neutral simulation spec, training mode, training sample counts, and seed.
- Manifest output captures arguments and outputs for exact reruns.

3. Scope and interpretation clarity
- README now includes explicit `H0/H1` scope and warns that significance is deviation-from-neutrality, not automatic proof of positive selection.

## Remaining Gaps (for full manuscript revision)

1. Direct benchmarking vs external tools (HyPhy/codeml/BUSTED/RELAX) is not yet automated in this repository.
2. Real-data case study pipeline is not yet bundled.
3. Manuscript and supplements outside this workspace still need line-by-line rewrite updates to match these software changes.
