from __future__ import annotations

from typing import Dict, TypeAlias

import torch

# Common type aliases for knowledge distillation, ->

# batch of data as dict mapping names to tensors
BatchDict: TypeAlias = Dict[str, torch.Tensor]
# model op as dict mapping names to tensors
OutputDict: TypeAlias = Dict[str, torch.Tensor]
# logits tensor, typically [B, C] for classification or [B, S, V] for LM
LogitsType: TypeAlias = torch.Tensor
# feature tensor from intermediate layers
FeaturesType: TypeAlias = torch.Tensor

