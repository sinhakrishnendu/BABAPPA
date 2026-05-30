"""Ranking-aware neural losses for BABAPPA gene-level training."""

from __future__ import annotations

from typing import Optional


VALID_LOSS_MODES = {"bce", "bce_rank", "focal", "focal_rank"}


def bce_logits_loss(logits, y, pos_weight=None, sample_weight=None):
    """Binary cross-entropy on logits with optional per-class/sample weighting."""
    import torch

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        y.float(),
        pos_weight=pos_weight,
        reduction="none",
    )
    if sample_weight is not None:
        loss = loss * sample_weight.float()
    return loss.mean()


def pairwise_rank_loss(logits, y, margin: float = 0.1):
    """Margin ranking loss over positive-negative pairs in a batch."""
    import torch

    y = y.float()
    positive_logits = logits[y >= 0.5]
    negative_logits = logits[y < 0.5]
    if positive_logits.numel() == 0 or negative_logits.numel() == 0:
        return logits.sum() * 0.0

    pair_losses = torch.relu(
        margin - positive_logits[:, None] + negative_logits[None, :]
    ).reshape(-1)
    max_pairs = 512
    if pair_losses.numel() > max_pairs:
        indices = torch.linspace(
            0,
            pair_losses.numel() - 1,
            steps=max_pairs,
            device=pair_losses.device,
        ).long()
        pair_losses = pair_losses[indices]
    return pair_losses.mean()


def focal_bce_logits_loss(
    logits,
    y,
    gamma: float = 2.0,
    pos_weight=None,
    sample_weight=None,
):
    """Stable focal BCE loss on logits."""
    import torch

    y = y.float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        y,
        pos_weight=pos_weight,
        reduction="none",
    )
    probs = torch.sigmoid(logits)
    pt = torch.where(y >= 0.5, probs, 1.0 - probs)
    focal_factor = (1.0 - pt).clamp(min=0.0, max=1.0).pow(gamma)
    loss = focal_factor * bce
    if sample_weight is not None:
        loss = loss * sample_weight.float()
    return loss.mean()


def combined_loss(
    logits,
    y,
    loss_mode: str,
    pos_weight=None,
    sample_weight=None,
    rank_weight: float = 0.2,
    focal_gamma: float = 2.0,
):
    """Combine BCE/focal classification loss with optional pairwise rank loss."""
    if loss_mode not in VALID_LOSS_MODES:
        allowed = ", ".join(sorted(VALID_LOSS_MODES))
        raise ValueError(f"loss_mode must be one of: {allowed}")

    if loss_mode in {"bce", "bce_rank"}:
        base_loss = bce_logits_loss(logits, y, pos_weight, sample_weight)
    else:
        base_loss = focal_bce_logits_loss(
            logits,
            y,
            gamma=focal_gamma,
            pos_weight=pos_weight,
            sample_weight=sample_weight,
        )

    if loss_mode in {"bce_rank", "focal_rank"} and rank_weight > 0:
        return base_loss + float(rank_weight) * pairwise_rank_loss(logits, y)
    return base_loss
