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
     out_dir=None, overlay_name=None, clip_fps=24.0, frame_stride=1, *, precision=None,
     compile_model=False, compile_mode="reduce-overhead", compile_backend=None)

and return the standard result dictionary.

This thin abstraction keeps the current functional runner implementations intact
while providing an obvious place for shared utilities or future common logic
(e.g., standardized setup/inference timing helpers or overlay helpers).
"""
from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


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

    def run_points(
        self,
        frames_24fps: List[Path],
        prompt_frame_idx: int,
        prompt_mask: "np.ndarray",
        imgsz: int,
        weight_name: str,
        device: str,
        out_dir: Optional[Path] = None,
        overlay_name: Optional[str] = None,
        clip_fps: float = 24.0,
        frame_stride: int = 1,
        *,
        precision: Any = None,
        max_clip_frames: Optional[int] = None,
        compile_model: bool = False,
        compile_mode: str | None = "reduce-overhead",
        compile_backend: str | None = None,
    ) -> Dict[str, object]:
        raise NotImplementedError(f"{self.name} does not support point prompts")

    def run_bbox(
        self,
        frames_24fps: List[Path],
        prompt_frame_idx: int,
        prompt_mask: "np.ndarray",
        imgsz: int,
        weight_name: str,
        device: str,
        out_dir: Optional[Path] = None,
        overlay_name: Optional[str] = None,
        clip_fps: float = 24.0,
        frame_stride: int = 1,
        *,
        precision: Any = None,
        max_clip_frames: Optional[int] = None,
        compile_model: bool = False,
        compile_mode: str | None = "reduce-overhead",
        compile_backend: str | None = None,
    ) -> Dict[str, object]:
        raise NotImplementedError(f"{self.name} does not support box prompts")

    def exposed_runners(self) -> Dict[str, Callable]:  # pragma: no cover - trivial
        mapping: Dict[str, Callable] = {}
        if type(self).run_points is not Model.run_points:
            mapping["points"] = getattr(self, "run_points")  # type: ignore[assignment]
        if type(self).run_bbox is not Model.run_bbox:
            mapping["bbox"] = getattr(self, "run_bbox")  # type: ignore[assignment]
        return mapping

    def register(self) -> Dict[str, Callable]:
        """Register the model family with the global runner registry."""
        from .registry import register_model_family

        runners = self.exposed_runners()
        register_model_family(self.name, runners)
        return runners
