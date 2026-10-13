from typing import Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from src.utils import get_device


def load_model_and_tokenizer(
    model_name_or_path: str,
    model_type: str = "causal-lm",
    dtype: torch.dtype = torch.float16,
    device: torch.device | None = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer from HF Hub or local folder, moving to device.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model ID or local directory.
    model_type : str
        "causal-lm" for decoder models or "classifier" for sequence classification models.
    dtype : torch.dtype
        Precision for model parameters.
    device : torch.device, optional
        Target device (defaults to get_device()).
    """
    if device is None:
        device = get_device()

    print(f"Loading tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {model_type} model on {device} ({dtype})...")
    model_cls = (
        AutoModelForSequenceClassification
        if model_type == "classifier"
        else AutoModelForCausalLM
    )
    model = model_cls.from_pretrained(
        model_name_or_path,
        dtype=dtype,
    ).to(device)
    model.eval()

    # Verify model parameters are on the requested device
    first_param = next(model.parameters())
    if first_param.device.type != device.type:
        raise RuntimeError(
            f"Device verification failed: expected {device.type}, got {first_param.device.type}"
        )

    return model, tokenizer
