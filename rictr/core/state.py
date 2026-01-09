from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
import torch


@dataclass
class StepOutput:
    # output from single training step
    loss: float
    step: int
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingState:
    # list to make mutable

    step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    history: List[StepOutput] = field(default_factory=list)

    def update(self, loss: torch.Tensor) -> StepOutput:
        # record step and reutrn output
        self.step += 1
        loss_val = loss.item()

        if loss_val < self.best_loss:
            self.best_loss = loss_val

        output = StepOutput(loss=loss_val, step=self.step)
        self.history.append(output)
        return output

    def advance_epoch(self) -> None:
        self.epoch += 1


Callback = Callable[[TrainingState, StepOutput], None]

