from __future__ import annotations

from typing import Dict, Protocol

import torch


class DistillationStrategy(Protocol): #defines how teacher and student outputs are compared(strategy selection)

    def __call__(
        self,
        *,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
            # to be implemented
            # compute distillation loss -> return scalar loss tensor
        ...

