from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class LossTracker:
    #loss track over training

    values: List[float] = field(default_factory=list)

    def update(self, loss: torch.Tensor | float) -> None:
        #Record a loss value.
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        self.values.append(loss)

    def mean(self) -> float:
        # mean of tracked losses.
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def last(self) -> float | None:
        #most recent loss
        return self.values[-1] if self.values else None

    def min(self) -> float | None:
        #minimum loss
        return min(self.values) if self.values else None

    def reset(self) -> None:
        #clear tracked values.
        self.values.clear()

    def __len__(self) -> int:
        return len(self.values)

