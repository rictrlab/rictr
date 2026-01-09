from __future__ import annotations
from typing import Any, Dict
import torch
from torch import nn
from ..strategies.base import DistillationStrategy

class Distiller:
    # to setup and manage distillation between student and teacher
    # freeze -> fwd pass -> loss to strategy -> distillation step

    def __init__(
        self,
        *,
        teacher: nn.Module,
        student: nn.Module,
        strategy: DistillationStrategy,
        optimizer: torch.optim.Optimizer,
        device: torch.device | None = None,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.strategy = strategy
        self.optimizer = optimizer
        self.device = device or torch.device("cpu")

        self._prepare_models()

    def _prepare_models(self) -> None:
        self.teacher.to(self.device)
        self.student.to(self.device)
        self.teacher.eval()
        self._freeze_teacher()

    def _freeze_teacher(self) -> None:
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distill_step(
        self,
        *,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        # single step distillation
        self.optimizer.zero_grad(set_to_none=True)

        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            teacher_outputs = self._forward(self.teacher, batch)

        student_outputs = self._forward(self.student, batch)

        loss = self.strategy(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            targets=batch,
        )

        loss.backward()
        self.optimizer.step()

        return loss.detach()

    def _forward(
        self, model: nn.Module, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        outputs = model(**batch)
        return self._normalize_outputs(outputs)

    @staticmethod
    def _normalize_outputs(outputs: Any) -> Dict[str, torch.Tensor]:
        # model outputs to dict format
        if isinstance(outputs, dict):
            return outputs

        if hasattr(outputs, "_asdict"):
            return dict(outputs._asdict())

        if isinstance(outputs, (tuple, list)):
            return {"logits": outputs[0]}

        return {"logits": outputs}

