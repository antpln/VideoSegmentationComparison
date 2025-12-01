"""Utility helpers for device detection, precision control, and memory housekeeping."""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
import gc
from typing import Any, Callable, Optional, Tuple

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
    device_count = torch.cuda.device_count()
    for idx in range(device_count):
        try:
            with torch.cuda.device(idx):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                if hasattr(torch.cuda, "reset_accumulated_memory_stats"):
                    torch.cuda.reset_accumulated_memory_stats()  # type: ignore[attr-defined]
                torch.cuda.empty_cache()
        except Exception:
            continue


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
    mode: Optional[str] = "reduce-overhead",
    backend: Optional[str] = None,
) -> Tuple[Any, bool]:
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


def cleanup_after_run() -> None:
    """Best-effort memory cleanup between inference runs."""

    gc.collect()
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for idx in range(device_count):
                try:
                    with torch.cuda.device(idx):
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        if hasattr(torch.cuda, "reset_accumulated_memory_stats"):
                            torch.cuda.reset_accumulated_memory_stats()  # type: ignore[attr-defined]
                except Exception:
                    continue
            # Drop any lingering inter-process handles and allocator bookkeeping.
            if hasattr(torch.cuda, "reset_peak_memory_stats"):
                torch.cuda.reset_peak_memory_stats()  # type: ignore[attr-defined]
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()  # type: ignore[attr-defined]
    except Exception:
        # Cleanup is best-effort; ignore backend-specific failures.
        pass


@dataclass(frozen=True)
class PrecisionContext:
    """Factory that yields autocast contexts for the requested precision."""

    precision: str
    device_type: Optional[str]
    dtype: Optional[Any]

    def __call__(self) -> Any:
        if self.device_type is None or self.dtype is None or torch is None:
            return nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.dtype)


def build_precision_context(precision: str, device: str) -> PrecisionContext:
    """Return a reusable precision context factory ensuring hardware support."""

    normalized = precision.lower()
    if normalized not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported precision '{precision}'. Choose from fp32, fp16, bf16.")

    if torch is None:
        if normalized != "fp32":
            raise ValueError("PyTorch is not available; only fp32 precision is supported.")
        return PrecisionContext(precision="fp32", device_type=None, dtype=None)

    if normalized == "fp32":
        return PrecisionContext(precision="fp32", device_type=None, dtype=None)

    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError(f"Precision '{precision}' requires an available CUDA device.")

    if normalized == "fp16":
        dtype = torch.float16  # type: ignore[attr-defined]
    else:  # bf16
        dtype = torch.bfloat16  # type: ignore[attr-defined]
        is_supported: Optional[Callable[[], bool]] = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(is_supported) and not is_supported():
            raise ValueError("Current CUDA device does not support bfloat16 autocast.")

    return PrecisionContext(precision=normalized, device_type="cuda", dtype=dtype)
