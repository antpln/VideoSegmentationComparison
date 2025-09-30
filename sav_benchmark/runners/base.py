"""Base model runner abstractions for the SA-V benchmark.

The goal of this layer is to make adding a new model family uniform and low-effort.
A contributor only needs to:

1. Subclass `Model`.
2. Implement one or both of: `run_points(...)`, `run_bbox(...)` with the standard signature.
3. Instantiate the subclass and register it with `register_model_family` using its `exposed_runners()`.

The registration call remains backwards compatible because the existing registry
expects a mapping of prompt_type -> callable.

Each runner method MUST accept the canonical parameter list used elsewhere:

    (frames_24fps, prompt_frame_idx, prompt_mask, imgsz, weight_name, device,
     out_dir=None, overlay_name=None, clip_fps=24.0, compile_model=False,
     compile_mode="reduce-overhead", compile_backend=None)

and return the standard result dictionary.

This thin abstraction keeps the current functional runner implementations intact
while providing an obvious place for shared utilities or future common logic
(e.g., standardized setup/inference timing helpers or overlay helpers).
"""
from __future__ import annotations

from abc import ABC
from typing import Callable, Dict


class Model(ABC):
    """Abstract base class for model families.

    Subclasses implement any supported prompt styles and expose them through
    `exposed_runners()` for registry consumption.
    """

    def __init__(self, name: str):
        self.name = name

    # Optional hooks (subclasses implement the ones they support)
    # def run_points(...): ...
    # def run_bbox(...): ...

    def exposed_runners(self) -> Dict[str, Callable]:  # pragma: no cover - trivial
        mapping: Dict[str, Callable] = {}
        if hasattr(self, "run_points"):
            mapping["points"] = getattr(self, "run_points")  # type: ignore[assignment]
        if hasattr(self, "run_bbox"):
            mapping["bbox"] = getattr(self, "run_bbox")  # type: ignore[assignment]
        return mapping
