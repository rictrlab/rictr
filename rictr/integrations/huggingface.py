from __future__ import annotations

from typing import Any, Dict

import torch


def hf_to_dict(outputs: Any) -> Dict[str, torch.Tensor]:
    """Convert HuggingFace model outputs to a standard dict.

    Handles ModelOutput objects, dicts, and tuples.

    Args:
        outputs: HuggingFace model outputs.

    Returns:
        Dict with standard keys (logits, hidden_states, attentions).
    """
    # If already a dict
    if isinstance(outputs, dict):
        return outputs

    # HF ModelOutput (has .logits, .hidden_states, etc.)
    if hasattr(outputs, "logits"):
        result = {"logits": outputs.logits}

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            result["hidden_states"] = outputs.hidden_states

        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            result["attentions"] = outputs.attentions

        return result

    # Tuple output (legacy HF format)
    if isinstance(outputs, tuple):
        return {"logits": outputs[0]}

    return {"logits": outputs}


class HFOutputAdapter:
    #Adapter that wraps a HuggingFace model to output dicts.
    def __init__(self, model) -> None:
        """
        Args:
            model: HuggingFace model (transformers.PreTrainedModel).
        """
        self.model = model

    def __call__(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.model(*args, **kwargs)
        return hf_to_dict(outputs)

    def __getattr__(self, name: str):
        return getattr(self.model, name)

