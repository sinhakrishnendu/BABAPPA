"""Site-level neural model definitions."""

from __future__ import annotations

from babappa.training.neural_env import safe_import_torch

torch, _TORCH_IMPORT_ERROR = safe_import_torch()

if torch is not None:
    nn = torch.nn

    class SiteMLPClassifier(nn.Module):
        """Small MLP classifier for numeric site-level feature vectors."""

        def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):  # type: ignore[override]
            logits = self.network(x)
            return logits.squeeze(-1)

else:

    class SiteMLPClassifier:  # type: ignore[no-redef]
        """Placeholder that fails clearly when PyTorch is unavailable."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PyTorch is not available. Install torch or use an environment containing torch."
            ) from _TORCH_IMPORT_ERROR


def count_parameters(model) -> int:
    """Count trainable model parameters."""
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
