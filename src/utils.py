import torch


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
