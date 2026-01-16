from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Temperature-scaled KL divergence for logit distillation.

    Computes KL(teacher || student) with temperature scaling.
    The T^2 factor preserves gradient magnitude across temperatures.

    Args:
        student_logits: Student model logits.
        teacher_logits: Teacher model logits.
        temperature: Softmax temperature. Higher = softer distributions.
        reduction: Loss reduction mode.

    Returns:
        Scalar KL divergence loss.
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    return F.kl_div(student_log_probs, teacher_probs, reduction=reduction) * (
        temperature * temperature
    )

