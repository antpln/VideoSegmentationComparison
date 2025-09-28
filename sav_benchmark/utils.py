"""Utility helpers for device detection and memory bookkeeping."""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

try:
    import torch  # type: ignore[import]
except Exception:  # pragma: no cover - torch is optional
    torch = None


def device_str() -> str:
    """Return the preferred device string for torch-based predictors."""
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def cuda_sync() -> None:
    """Synchronize the CUDA stream when available."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_gpu_peaks() -> None:
    """Reset CUDA peak memory statistics if CUDA is available."""
    if torch is None or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def get_gpu_peaks() -> Tuple[Optional[int], Optional[int]]:
    """Return peak allocated and reserved memory in bytes when CUDA is active."""
    if torch is None or not torch.cuda.is_available():
        return None, None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


def to_mib(value: Optional[int]) -> Optional[float]:
    """Convert a byte value to mebibytes rounded to one decimal place."""
    if value is None:
        return None
    return round(value / (1024 * 1024), 1)


def expand_path(path: str) -> str:
    """Expand user (~) and environment variables in a filesystem path."""
    return os.path.expanduser(os.path.expandvars(path))


def maybe_compile_module(
    module: Any,
    *,
    mode: str | None = "reduce-overhead",
    backend: str | None = None,
) -> tuple[Any, bool]:
    """Attempt to compile a torch module via ``torch.compile``.

    Returns the (possibly compiled) module and a boolean indicating success.
    """

    if module is None or torch is None or not hasattr(torch, "compile"):
        return module, False

    if not isinstance(module, torch.nn.Module):
        return module, False

    try:
        compiled = torch.compile(module, mode=mode, backend=backend)
        return compiled, True
    except Exception as exc:  # pragma: no cover - backend specific
        print(f"[WARN] torch.compile failed: {exc}")
        return module, False
