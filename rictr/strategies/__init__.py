from .base import DistillationStrategy
from .logit import SoftTarget
from .hidden import HiddenStateDistillation
from .hybrid import Composite

__all__ = [
    "DistillationStrategy",
    "SoftTarget",
    "HiddenStateDistillation",
    "Composite",
]
