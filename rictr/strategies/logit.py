from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from ..losses.kl import kl_divergence


class SoftTarget: # soft target based knowledge distillation 
    # paper: (https://arxiv.org/abs/1503.02531)

    def __init__(
        self,
        *,
        temperature: float,
        alpha: float | None = None,
        task_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """
        Args:
            temperature: Softmax temperature. Higher = softer distributions.
            alpha: Blend factor for task loss. If None, pure distillation.
                When set, loss = alpha * task_loss + (1 - alpha) * distill_loss.
            task_loss: Supervised loss function. Default: cross_entropy.
        """
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

        self.temperature = temperature
        self.alpha = alpha
        self.task_loss = task_loss or F.cross_entropy

    def __call__(
        self,
        *,
        student_outputs: dict[str, torch.Tensor],
        teacher_outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        student_logits = student_outputs["logits"]
        teacher_logits = teacher_outputs["logits"]

        distill_loss = kl_divergence(
            student_logits,
            teacher_logits,
            temperature=self.temperature,
        )

        if self.alpha is None:
            return distill_loss

        if targets is None or "labels" not in targets:
            raise ValueError("targets with 'labels' required when alpha is set")

        task_loss = self.task_loss(student_logits, targets["labels"])

        return self.alpha * task_loss + (1.0 - self.alpha) * distill_loss

