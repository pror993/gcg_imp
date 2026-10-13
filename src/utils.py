from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase



def get_device() -> torch.device:
    """Return mps if available on Apple Silicon, else cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_memory() -> None:
    """Flush memory cache on the active backend."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_non_ascii_mask(
    tokenizer: PreTrainedTokenizerBase,
    vocab_size: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a boolean mask marking non-ASCII, unprintable, or special tokens.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        The tokenizer corresponding to the target model.
    vocab_size : int, optional
        Target vocabulary dimension (e.g. from the embedding matrix). If None,
        defaults to ``len(tokenizer)``.
    device : torch.device, optional
        Device to allocate the mask on.

    Returns
    -------
    mask : Tensor of shape ``(vocab_size,)``
        Boolean tensor where ``True`` indicates tokens to exclude.
    """
    total_vocab = vocab_size if vocab_size is not None else len(tokenizer)
    mask = torch.zeros(total_vocab, dtype=torch.bool, device=device)

    # Filter non-ASCII or unprintable replacement tokens
    num_to_check = min(total_vocab, len(tokenizer))
    for tok_id in range(num_to_check):
        token_str = tokenizer.decode([tok_id], skip_special_tokens=False)
        if not token_str.isascii() or "\ufffd" in token_str:
            mask[tok_id] = True

    # Any IDs outside tokenizer vocabulary (e.g. padded embedding table)
    if total_vocab > len(tokenizer):
        mask[len(tokenizer):] = True

    # Exclude all special tokens (e.g. BOS, EOS, PAD, UNK, chat delimiters)
    for s_id in tokenizer.all_special_ids:
        if 0 <= s_id < total_vocab:
            mask[s_id] = True

    return mask
