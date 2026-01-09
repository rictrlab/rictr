from __future__ import annotations

from typing import Iterable, List

import torch

from .distiller import Distiller
from .state import Callback, StepOutput, TrainingState


class Trainer:
    # training loop

    def __init__(
        self,
        distiller: Distiller,
        callbacks: List[Callback] | None = None,
    ) -> None:
        self.distiller = distiller
        self.callbacks = callbacks or []
        self.state = TrainingState()

    def train_epoch(
        self,
        dataloader: Iterable[dict[str, torch.Tensor]],
    ) -> float:
    # run one epoch of distillation
    # argument as dataloader dict
    # returns average loss for the epoch

        self.distiller.student.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            loss = self.distiller.distill_step(batch=batch)
            output = self.state.update(loss)

            for callback in self.callbacks:
                callback(self.state, output)

            total_loss += output.loss
            n_batches += 1

        self.state.advance_epoch()
        return total_loss / max(n_batches, 1)

    def train(
        self,
        dataloader: Iterable[dict[str, torch.Tensor]],
        epochs: int,
    ) -> List[float]:
    # run multiple epochs
    # gets argument as dataloader dict
    # returns list of average losses per epoch
        epoch_losses = []
        for _ in range(epochs):
            avg_loss = self.train_epoch(dataloader)
            epoch_losses.append(avg_loss)
        return epoch_losses

