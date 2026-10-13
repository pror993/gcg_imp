"""
Benchmarking utilities: device synchronization, timing, memory querying, and formatting.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

import torch


def sync_device() -> None:
    """Synchronize the active accelerator backend to guarantee kernel completion."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


class Timer:
    """Accurate hardware-synchronized execution timer."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> Timer:
        sync_device()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        sync_device()
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0


def get_allocated_memory_mb() -> float:
    """Return currently allocated device memory in Megabytes (MB)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    elif torch.backends.mps.is_available():
        try:
            return torch.mps.current_allocated_memory() / (1024 * 1024)
        except Exception:
            return 0.0
    return 0.0


def format_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Format headers and rows into an aligned Markdown table."""
    str_rows = [[str(cell) for cell in row] for row in rows]
    col_widths = [len(h) for h in headers]

    for row in str_rows:
        for idx, cell in enumerate(row):
            if idx < len(col_widths):
                col_widths[idx] = max(col_widths[idx], len(cell))

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"

    body_lines = [
        "| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))) + " |"
        for row in str_rows
    ]

    return "\n".join([header_line, sep_line] + body_lines)
