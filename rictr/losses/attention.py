from __future__ import annotations

import torch
import torch.nn.functional as F

# attention transfer loss
def attention_loss(
    student_attention: torch.Tensor,
    teacher_attention: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    #paper: (https://arxiv.org/abs/1612.03928)

    """
    Matches attention maps between teacher and student.
    Attention maps are typically the squared activations summed over channels.

    Args:
        student_attention: Student attention maps [B, H, W] or [B, S, S].
        teacher_attention: Teacher attention maps [B, H, W] or [B, S, S].
        normalize: If True, L2-normalize attention maps before comparison.

    Returns:
        Scalar attention loss.
    """
    # flatten spatial dimensions
    s_flat = student_attention.flatten(start_dim=1)
    t_flat = teacher_attention.flatten(start_dim=1)

    if normalize:
        s_flat = F.normalize(s_flat, p=2, dim=-1)
        t_flat = F.normalize(t_flat, p=2, dim=-1)

    return F.mse_loss(s_flat, t_flat)


def compute_attention_map(features: torch.Tensor) -> torch.Tensor:
    """attention map from feature tensor

    Creates a spatial attention map by summing squared activations
    across the channel dimension.

    Args:
        features: Feature tensor [B, C, H, W] or [B, S, D].

    Returns:
        Attention map [B, H, W] or [B, S].
    """
    if features.dim() == 4:
        # Conv features: [B, C, H, W] -> [B, H, W]
        return (features ** 2).sum(dim=1)
    elif features.dim() == 3:
        # Transformer features: [B, S, D] -> [B, S]
        return (features ** 2).sum(dim=-1)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {features.dim()}D")

