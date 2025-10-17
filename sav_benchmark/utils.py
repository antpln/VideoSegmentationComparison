"""Utility helpers for device detection and memory bookkeeping."""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

try:
    import torch  # type: ignore[import]
except Exception:  # pragma: no cover - torch is optional
    torch = None


def is_jetson() -> bool:
    """Detect if running on Jetson platform (unified memory architecture)."""
    # Check for Tegra SoC (Jetson identifier)
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    if os.path.exists("/sys/firmware/devicetree/base/model"):
        try:
            with open("/sys/firmware/devicetree/base/model", "r") as f:
                model = f.read().lower()
                if "jetson" in model or "tegra" in model:
                    return True
        except Exception:
            pass
    # Check CUDA device name for Tegra/Orin
    if torch is not None and torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0).lower()
            if "orin" in device_name or "xavier" in device_name or "tegra" in device_name:
                return True
        except Exception:
            pass
    return False


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
    """Return peak allocated and reserved memory in bytes when CUDA is active.
    
    Note: On Jetson (unified memory), this reports CUDA allocator's view of the 
    shared memory pool. For total system memory, use psutil.memory_info().
    """
    if torch is None or not torch.cuda.is_available():
        return None, None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


def get_memory_info() -> dict:
    """Get comprehensive memory information accounting for unified vs discrete architecture.
    
    Returns dict with keys:
        - is_unified: bool, True for Jetson/unified memory
        - gpu_alloc: int or None, CUDA allocated bytes
        - gpu_reserved: int or None, CUDA reserved bytes  
        - system_total: int or None, total system RAM
        - system_used: int or None, used system RAM
        - note: str, interpretation note for unified systems
    """
    import psutil
    
    unified = is_jetson()
    gpu_alloc, gpu_reserved = get_gpu_peaks()
    
    # System memory
    mem = psutil.virtual_memory()
    sys_total = mem.total
    sys_used = mem.used
    
    note = ""
    if unified and gpu_alloc is not None:
        note = (
            "Unified memory: GPU and system share same physical RAM. "
            "GPU alloc is subset of system used, not additive."
        )
    
    return {
        "is_unified": unified,
        "gpu_alloc": gpu_alloc,
        "gpu_reserved": gpu_reserved,
        "system_total": sys_total,
        "system_used": sys_used,
        "note": note,
    }


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
