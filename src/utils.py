from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase



def get_device() -> torch.device:
    """Return cuda if available, mps if available on Apple Silicon, else cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
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


def expand_kv_cache(past_key_values: Any, batch_size: int) -> Any:
    """Expand precomputed batch-1 prefix KV cache to batch_size via non-copying views.

    Parameters
    ----------
    past_key_values : DynamicCache or tuple
        Precomputed key-value cache for the prompt prefix (batch dimension = 1).
    batch_size : int
        Target batch size for candidate forward passes.

    Returns
    -------
    expanded_cache : DynamicCache or tuple
        Cache expanded along the batch dimension.
    """
    if hasattr(past_key_values, "layers"):  # Transformers 5.x DynamicCache
        from transformers.cache_utils import DynamicCache
        new_cache = DynamicCache()
        for layer_idx, layer in enumerate(past_key_values.layers):
            k = layer.keys.expand(batch_size, -1, -1, -1)
            v = layer.values.expand(batch_size, -1, -1, -1)
            new_cache.update(k, v, layer_idx)
        return new_cache
    elif hasattr(past_key_values, "key_cache"):  # Transformers 4.x DynamicCache
        from transformers.cache_utils import DynamicCache
        new_cache = DynamicCache()
        for k, v in zip(past_key_values.key_cache, past_key_values.value_cache):
            new_cache.key_cache.append(k.expand(batch_size, -1, -1, -1))
            new_cache.value_cache.append(v.expand(batch_size, -1, -1, -1))
        return new_cache
    elif isinstance(past_key_values, tuple):  # Legacy tuple of (key, value)
        return tuple(
            (k.expand(batch_size, -1, -1, -1), v.expand(batch_size, -1, -1, -1))
            for k, v in past_key_values
        )
    return past_key_values
