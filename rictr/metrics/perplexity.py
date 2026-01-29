from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def perplexity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute perplexity for language modeling.

    Args:
        logits: Model logits [B, S, V] or [B*S, V].
        labels: Target token ids [B, S] or [B*S].
        ignore_index: Label value to ignore (padding).

    Returns:
        Perplexity as a float.
    """
    # Flatten if needed
    if logits.dim() == 3:
        B, S, V = logits.shape
        logits = logits.view(-1, V)
        labels = labels.view(-1)

    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)
    return math.exp(loss.item())


def bits_per_character(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute bits per character (BPC) for character-level LM.

    Args:
        logits: Model logits.
        labels: Target token ids.
        ignore_index: Label value to ignore.

    Returns:
        BPC as a float.
    """
    if logits.dim() == 3:
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)
    return loss.item() / math.log(2)

