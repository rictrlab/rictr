from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ..alignment.hooks import FeatureExtractor
from ..alignment.layer_map import LayerMap


class HiddenStateDistillation: #feature based knowledge distillation
     # matches intermediate layer activations between teacher and student
     #uses forward hooks to get features during the forward pass

    def __init__(
        self,
        *,
        teacher: nn.Module,
        student: nn.Module,
        layer_map: LayerMap,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
    ) -> None:
        """
        Args:
            teacher: Teacher model.
            student: Student model.
            layer_map: LayerMap defining teacher-student layer pairs and projectors.
            loss_fn: Loss function for comparing features. Default: MSE.
        """
        self.layer_map = layer_map
        self.loss_fn = loss_fn

        self._teacher_extractor = FeatureExtractor(teacher, layer_map.teacher_layers)
        self._student_extractor = FeatureExtractor(student, layer_map.student_layers)

    def __call__(
        self,
        *,
        student_outputs: dict[str, torch.Tensor],
        teacher_outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        losses = []

        t_features = self._teacher_extractor.features
        s_features = self._student_extractor.features

        for t_layer, s_layer in self.layer_map.pairs:
            t_feat = t_features[t_layer]
            s_feat = self.layer_map.project(s_layer, s_features[s_layer])

            losses.append(self.loss_fn(s_feat, t_feat))

        self._teacher_extractor.clear()
        self._student_extractor.clear()

        return torch.stack(losses).mean()

    def remove_hooks(self) -> None:
        #removes all registered hooks, call when distillation is done
        self._teacher_extractor.remove_hooks()
        self._student_extractor.remove_hooks()

