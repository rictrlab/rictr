from __future__ import annotations

import torch

from .base import DistillationStrategy


class Composite: #can combine multiple distillation strategies with weighted sum
    # eg: logit + feature matching

    def __init__(
        self,
        strategies: list[tuple[DistillationStrategy, float]],
    ) -> None:
        """
        Args:
            strategies: List of (strategy, weight) tuples.
            weights are applied as multipliers to each strategy's loss.
        """
        if not strategies:
            raise ValueError("At least one strategy required")

        self.strategies = strategies

    def __call__(
        self,
        *,
        student_outputs: dict[str, torch.Tensor],
        teacher_outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        total = None

        for strategy, weight in self.strategies:
            loss = strategy(
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                targets=targets,
            )
            weighted = weight * loss

            if total is None:
                total = weighted
            else:
                total = total + weighted

        return total
