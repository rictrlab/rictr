from __future__ import annotations

from typing import Callable, Protocol

import torch


class DistillationLoss(Protocol):
    # takes student and teacher tensors and returns a scalar loss

    def __call__(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> torch.Tensor:
        ...


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

