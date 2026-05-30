"""Small PyTorch smoke model for BABAPPA tensor shards."""

from __future__ import annotations

from babappa.training.neural_env import safe_import_torch

_torch, _torch_error = safe_import_torch()


if _torch is not None:

    class SmallGeneClassifier(_torch.nn.Module):
        """Minimal gene-level classifier for smoke-training infrastructure."""

        def __init__(
            self,
            vocab_size: int = 128,
            embedding_dim: int = 16,
            hidden_dim: int = 32,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.embedding = _torch.nn.Embedding(
                vocab_size, embedding_dim, padding_idx=0
            )
            self.mlp = _torch.nn.Sequential(
                _torch.nn.Linear(embedding_dim + 1, hidden_dim),
                _torch.nn.ReLU(),
                _torch.nn.Dropout(dropout),
                _torch.nn.Linear(hidden_dim, 1),
            )

        def forward(self, X):  # noqa: N803 - tensor name follows ML convention.
            codon_ids = X[..., 0].long()
            gap_indicator = X[..., 1].float().unsqueeze(-1)
            embedded = self.embedding(codon_ids)
            features = _torch.cat([embedded, gap_indicator], dim=-1)
            pooled = features.mean(dim=(1, 2))
            return self.mlp(pooled).squeeze(-1)


    class ContrastiveGeneClassifier(_torch.nn.Module):
        """Gene-level classifier with sparse taxon/site contrast pooling."""

        def __init__(
            self,
            vocab_size: int = 128,
            embedding_dim: int = 32,
            hidden_dim: int = 64,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.embedding = _torch.nn.Embedding(
                vocab_size, embedding_dim, padding_idx=0
            )
            token_dim = embedding_dim + 1
            pooled_dim = token_dim * 6
            second_hidden = max(1, hidden_dim // 2)
            self.mlp = _torch.nn.Sequential(
                _torch.nn.Linear(pooled_dim, hidden_dim),
                _torch.nn.ReLU(),
                _torch.nn.Dropout(dropout),
                _torch.nn.Linear(hidden_dim, second_hidden),
                _torch.nn.ReLU(),
                _torch.nn.Dropout(dropout),
                _torch.nn.Linear(second_hidden, 1),
            )

        def forward(self, X):  # noqa: N803 - tensor name follows ML convention.
            codon_ids = X[..., 0].long()
            gap_indicator = X[..., 1].float().unsqueeze(-1)
            embedded = self.embedding(codon_ids)
            features = _torch.cat([embedded, gap_indicator], dim=-1)

            global_mean = features.mean(dim=(1, 2))
            global_max = features.amax(dim=(1, 2))
            global_std = features.std(dim=(1, 2), unbiased=False)
            site_taxon_std_mean = features.std(dim=1, unbiased=False).mean(dim=1)
            site_taxon_max_mean = features.amax(dim=1).mean(dim=1)
            taxon_site_std_mean = features.std(dim=2, unbiased=False).mean(dim=1)
            pooled = _torch.cat(
                [
                    global_mean,
                    global_max,
                    global_std,
                    site_taxon_std_mean,
                    site_taxon_max_mean,
                    taxon_site_std_mean,
                ],
                dim=-1,
            )
            return self.mlp(pooled).squeeze(-1)


    class SaturationAwareGeneClassifier(_torch.nn.Module):
        """Contrastive gene-level classifier with saturation-tier conditioning."""

        uses_saturation_id = True

        def __init__(
            self,
            vocab_size: int = 128,
            embedding_dim: int = 32,
            hidden_dim: int = 64,
            dropout: float = 0.1,
            saturation_embedding_dim: int = 8,
            num_saturation_tiers: int = 5,
        ) -> None:
            super().__init__()
            self.embedding = _torch.nn.Embedding(
                vocab_size, embedding_dim, padding_idx=0
            )
            self.saturation_embedding = _torch.nn.Embedding(
                num_saturation_tiers,
                saturation_embedding_dim,
                padding_idx=0,
            )
            token_dim = embedding_dim + 1
            pooled_dim = token_dim * 6
            total_dim = pooled_dim + saturation_embedding_dim
            self.mlp = _torch.nn.Sequential(
                _torch.nn.Linear(total_dim, hidden_dim),
                _torch.nn.ReLU(),
                _torch.nn.Dropout(dropout),
                _torch.nn.Linear(hidden_dim, hidden_dim),
                _torch.nn.ReLU(),
                _torch.nn.Dropout(dropout),
                _torch.nn.Linear(hidden_dim, 1),
            )

        def forward(self, X, saturation_id=None):  # noqa: N803 - ML tensor name.
            codon_ids = X[..., 0].long()
            gap_indicator = X[..., 1].float().unsqueeze(-1)
            embedded = self.embedding(codon_ids)
            features = _torch.cat([embedded, gap_indicator], dim=-1)
            pooled = _contrastive_pool(features)

            if saturation_id is None:
                saturation_id = _torch.zeros(
                    X.shape[0],
                    dtype=_torch.long,
                    device=X.device,
                )
            saturation_id = saturation_id.long().clamp(
                min=0,
                max=self.saturation_embedding.num_embeddings - 1,
            )
            saturation_features = self.saturation_embedding(saturation_id)
            combined = _torch.cat([pooled, saturation_features], dim=-1)
            return self.mlp(combined).squeeze(-1)


    class SiteAttentionGeneClassifier(_torch.nn.Module):
        """Gene-level classifier that learns sparse codon-site attention."""

        def __init__(
            self,
            vocab_size: int = 128,
            embedding_dim: int = 32,
            hidden_dim: int = 64,
            dropout: float = 0.1,
            use_saturation_embedding: bool = False,
            saturation_embedding_dim: int = 8,
            num_saturation_tiers: int = 5,
        ) -> None:
            super().__init__()
            self.uses_saturation_id = bool(use_saturation_embedding)
            self.embedding = _torch.nn.Embedding(
                vocab_size, embedding_dim, padding_idx=0
            )
            token_dim = embedding_dim + 1
            site_dim = token_dim * 3
            global_dim = token_dim * 3
            self.site_mlp = _torch.nn.Sequential(
                _torch.nn.Linear(site_dim, hidden_dim),
                _torch.nn.ReLU(),
                _torch.nn.Dropout(dropout),
            )
            self.attention_head = _torch.nn.Linear(hidden_dim, 1)
            saturation_dim = 0
            if self.uses_saturation_id:
                self.saturation_embedding = _torch.nn.Embedding(
                    num_saturation_tiers,
                    saturation_embedding_dim,
                    padding_idx=0,
                )
                saturation_dim = saturation_embedding_dim
            total_dim = hidden_dim + global_dim + saturation_dim
            self.mlp = _torch.nn.Sequential(
                _torch.nn.Linear(total_dim, hidden_dim),
                _torch.nn.ReLU(),
                _torch.nn.Dropout(dropout),
                _torch.nn.Linear(hidden_dim, 1),
            )

        def forward(self, X, saturation_id=None):  # noqa: N803 - ML tensor name.
            features = self._token_features(X)
            site_hidden = self._site_hidden(features)
            attention_weights = _torch.softmax(
                self.attention_head(site_hidden).squeeze(-1),
                dim=1,
            )
            attended_sites = (site_hidden * attention_weights.unsqueeze(-1)).sum(dim=1)
            global_summary = _torch.cat(
                [
                    features.mean(dim=(1, 2)),
                    features.amax(dim=(1, 2)),
                    features.std(dim=(1, 2), unbiased=False),
                ],
                dim=-1,
            )
            parts = [attended_sites, global_summary]
            if self.uses_saturation_id:
                if saturation_id is None:
                    saturation_id = _torch.zeros(
                        X.shape[0],
                        dtype=_torch.long,
                        device=X.device,
                    )
                saturation_id = saturation_id.long().clamp(
                    min=0,
                    max=self.saturation_embedding.num_embeddings - 1,
                )
                parts.append(self.saturation_embedding(saturation_id))
            return self.mlp(_torch.cat(parts, dim=-1)).squeeze(-1)

        def get_attention_weights(self, X, saturation_id=None):  # noqa: N803
            """Return codon-site attention weights for later diagnostics."""
            del saturation_id
            features = self._token_features(X)
            site_hidden = self._site_hidden(features)
            return _torch.softmax(self.attention_head(site_hidden).squeeze(-1), dim=1)

        def _token_features(self, X):
            codon_ids = X[..., 0].long()
            gap_indicator = X[..., 1].float().unsqueeze(-1)
            embedded = self.embedding(codon_ids)
            return _torch.cat([embedded, gap_indicator], dim=-1)

        def _site_hidden(self, features):
            site_mean = features.mean(dim=1)
            site_max = features.amax(dim=1)
            site_std = features.std(dim=1, unbiased=False)
            site_summary = _torch.cat([site_mean, site_max, site_std], dim=-1)
            return self.site_mlp(site_summary)


    def _contrastive_pool(features):
        global_mean = features.mean(dim=(1, 2))
        global_max = features.amax(dim=(1, 2))
        global_std = features.std(dim=(1, 2), unbiased=False)
        site_taxon_std_mean = features.std(dim=1, unbiased=False).mean(dim=1)
        site_taxon_max_mean = features.amax(dim=1).mean(dim=1)
        taxon_site_std_mean = features.std(dim=2, unbiased=False).mean(dim=1)
        return _torch.cat(
            [
                global_mean,
                global_max,
                global_std,
                site_taxon_std_mean,
                site_taxon_max_mean,
                taxon_site_std_mean,
            ],
            dim=-1,
        )

else:

    class SmallGeneClassifier:  # type: ignore[no-redef]
        """Placeholder that explains the optional PyTorch dependency."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyTorch is not available. Install torch or use an environment "
                "containing torch."
            ) from None


    class ContrastiveGeneClassifier:  # type: ignore[no-redef]
        """Placeholder that explains the optional PyTorch dependency."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyTorch is not available. Install torch or use an environment "
                "containing torch."
            ) from None


    class SaturationAwareGeneClassifier:  # type: ignore[no-redef]
        """Placeholder that explains the optional PyTorch dependency."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyTorch is not available. Install torch or use an environment "
                "containing torch."
            ) from None


    class SiteAttentionGeneClassifier:  # type: ignore[no-redef]
        """Placeholder that explains the optional PyTorch dependency."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyTorch is not available. Install torch or use an environment "
                "containing torch."
            ) from None


def count_parameters(model) -> int:
    """Count trainable parameters in a PyTorch model."""
    return int(sum(parameter.numel() for parameter in model.parameters()))


def build_small_gene_classifier(
    vocab_size: int = 128,
    embedding_dim: int = 16,
    hidden_dim: int = 32,
    dropout: float = 0.1,
):
    """Build the intentionally small gene-level smoke classifier."""
    return SmallGeneClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def build_gene_classifier(
    architecture: str,
    vocab_size: int = 128,
    embedding_dim: int = 32,
    hidden_dim: int = 64,
    dropout: float = 0.1,
    saturation_embedding_dim: int = 8,
    num_saturation_tiers: int = 5,
):
    """Build a gene-level neural classifier by architecture name."""
    if architecture == "small":
        return SmallGeneClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if architecture == "contrastive":
        return ContrastiveGeneClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if architecture == "saturation_aware":
        return SaturationAwareGeneClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            saturation_embedding_dim=saturation_embedding_dim,
            num_saturation_tiers=num_saturation_tiers,
        )
    if architecture == "site_attention":
        return SiteAttentionGeneClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_saturation_embedding=False,
            saturation_embedding_dim=saturation_embedding_dim,
            num_saturation_tiers=num_saturation_tiers,
        )
    if architecture == "site_attention_saturation":
        return SiteAttentionGeneClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_saturation_embedding=True,
            saturation_embedding_dim=saturation_embedding_dim,
            num_saturation_tiers=num_saturation_tiers,
        )
    raise ValueError(
        "architecture must be one of: contrastive, saturation_aware, "
        "site_attention, site_attention_saturation, small"
    )
