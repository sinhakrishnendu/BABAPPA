from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codon import GY94Model, build_gy94_rate_matrix, transition_matrix
from .io import Alignment
from .phylo import TreeNode, parse_newick


@dataclass(frozen=True)
class NeutralSpec:
    tree_newick: str
    kappa: float = 2.0
    omega: float = 1.0
    codon_frequencies: dict[str, float] | None = None
    simulator: str = "gy94"

    def to_dict(self) -> dict[str, object]:
        return {
            "tree_newick": self.tree_newick,
            "kappa": self.kappa,
            "omega": self.omega,
            "codon_frequencies": self.codon_frequencies,
            "simulator": self.simulator,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "NeutralSpec":
        return cls(
            tree_newick=str(payload["tree_newick"]),  # type: ignore[index]
            kappa=float(payload.get("kappa", 2.0)),
            omega=float(payload.get("omega", 1.0)),
            codon_frequencies=(
                None
                if payload.get("codon_frequencies") is None
                else {str(k): float(v) for k, v in dict(payload["codon_frequencies"]).items()}  # type: ignore[index]
            ),
            simulator=str(payload.get("simulator", "gy94")),
        )


class GY94NeutralSimulator:
    def __init__(self, spec: NeutralSpec):
        if spec.simulator.lower() != "gy94":
            raise ValueError(f"Unsupported simulator: {spec.simulator}")
        self.spec = spec
        self.tree: TreeNode = parse_newick(spec.tree_newick)
        self.model: GY94Model = build_gy94_rate_matrix(
            kappa=spec.kappa,
            omega=spec.omega,
            codon_frequencies=spec.codon_frequencies,
        )
        self._transition_cache: dict[float, np.ndarray] = {}
        self._leaf_names = self.tree.leaf_names()
        if not self._leaf_names:
            raise ValueError("Neutral tree must contain at least one leaf.")

    @property
    def leaf_names(self) -> tuple[str, ...]:
        return tuple(self._leaf_names)

    def _sample_index(self, probs: np.ndarray, rng: np.random.Generator) -> int:
        cdf = np.cumsum(probs)
        x = float(rng.random())
        idx = int(np.searchsorted(cdf, x, side="right"))
        return min(idx, len(probs) - 1)

    def _transition(self, branch_length: float) -> np.ndarray:
        key = round(float(branch_length), 12)
        matrix = self._transition_cache.get(key)
        if matrix is None:
            matrix = transition_matrix(self.model.rate_matrix, float(branch_length))
            self._transition_cache[key] = matrix
        return matrix

    def _simulate_site(self, rng: np.random.Generator) -> dict[str, str]:
        codons = self.model.codons
        pi = self.model.frequencies
        root_idx = self._sample_index(pi, rng)
        states: dict[str, str] = {}

        def _walk(node: TreeNode, state_idx: int) -> None:
            if node.is_leaf:
                if not node.name:
                    raise ValueError("Leaf node is missing a name.")
                states[node.name] = codons[state_idx]
                return
            for child in node.children:
                trans = self._transition(child.length)
                child_idx = self._sample_index(trans[state_idx], rng)
                _walk(child, child_idx)

        _walk(self.tree, root_idx)
        return states

    def simulate_alignment(self, *, length_nt: int, seed: int | None = None) -> Alignment:
        if length_nt <= 0:
            raise ValueError("length_nt must be > 0")
        rng = np.random.default_rng(seed)
        n_codons = (length_nt + 2) // 3

        seq_buffers: dict[str, list[str]] = {name: [] for name in self._leaf_names}
        for _ in range(n_codons):
            site = self._simulate_site(rng)
            for leaf in self._leaf_names:
                seq_buffers[leaf].append(site[leaf])

        sequences = tuple("".join(seq_buffers[name])[:length_nt] for name in self._leaf_names)
        return Alignment(names=tuple(self._leaf_names), sequences=sequences)
