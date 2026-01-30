from __future__ import annotations
from typing import TYPE_CHECKING, Dict
import torch
from torch import nn
if TYPE_CHECKING:
    from accelerate import Accelerator
from ..strategies.base import DistillationStrategy


# need smoe modification later

class AcceleratedDistiller:
    #distiller with Accelerate support for multi device training.


    def __init__(
        self,
        *,
        teacher: nn.Module,
        student: nn.Module,
        strategy: DistillationStrategy,
        optimizer: torch.optim.Optimizer,
        accelerator: "Accelerator",
    ) -> None:
        """
        Args:
            teacher: Teacher model.
            student: Student model.
            strategy: Distillation strategy.
            optimizer: Optimizer for student.
            accelerator: Accelerate Accelerator instance.
        """
        self.accelerator = accelerator

        # Prepare models and optimizer with accelerate
        self.student, self.optimizer = accelerator.prepare(student, optimizer)
        self.teacher = accelerator.prepare_model(teacher, evaluation_mode=True)
        self.strategy = strategy

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distill_step(
        self,
        *,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Execute a single distillation step with Accelerate.

        Args:
            batch: Input batch dict.

        Returns:
            Detached loss tensor.
        """
        self.optimizer.zero_grad()

        with torch.no_grad():
            teacher_outputs = self._forward(self.teacher, batch)

        student_outputs = self._forward(self.student, batch)

        loss = self.strategy(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            targets=batch,
        )

        self.accelerator.backward(loss)
        self.optimizer.step()

        return loss.detach()

    def _forward(
        self, model: nn.Module, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        outputs = model(**batch)
        return self._normalize_outputs(outputs)

    @staticmethod
    def _normalize_outputs(outputs) -> Dict[str, torch.Tensor]:
        if isinstance(outputs, dict):
            return outputs
        if hasattr(outputs, "_asdict"):
            return dict(outputs._asdict())
        if isinstance(outputs, (tuple, list)):
            return {"logits": outputs[0]}
        return {"logits": outputs}

