"""Video writing and overlay helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import cv2  # type: ignore[import]
import numpy as np


def write_video_mp4(path_mp4: Path, frames_bgr: Iterable[np.ndarray], fps: float = 24.0) -> None:
    """Serialize a sequence of BGR frames to an H.264 mp4 file."""
    frames = list(frames_bgr)
    if not frames:
        return
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(path_mp4), fourcc, fps, (width, height))
    if not writer.isOpened():  # pragma: no cover - cv2 backend dependent
        raise RuntimeError(f"Could not open writer for {path_mp4}")

    try:
        for idx, frame in enumerate(frames):
            if frame.shape[:2] != (height, width):
                raise ValueError(f"Frame {idx} has shape {frame.shape[:2]}, expected {(height, width)}")
            if frame.dtype != np.uint8:
                raise ValueError(f"Frame {idx} has dtype {frame.dtype}, expected uint8")
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Frame {idx} must be 3-channel BGR")
            writer.write(frame)
    finally:
        writer.release()


def overlay_union(frame_bgr: np.ndarray, masks_bool: np.ndarray | None, alpha: float = 0.5) -> np.ndarray:
    """Blend the union of one or more masks onto a frame for visualization."""
    if masks_bool is None:
        return frame_bgr

    if masks_bool.ndim == 2:
        union = masks_bool
    else:
        # Collapse multi-mask tensors so any positive pixel is highlighted.
        union = np.any(masks_bool.astype(bool), axis=0)

    color = frame_bgr.copy()
    color[..., 1] = 255
    mask3 = np.dstack([union.astype(np.uint8) * 255] * 3)
    blended = np.where(mask3 > 0, (alpha * color + (1 - alpha) * frame_bgr).astype(frame_bgr.dtype), frame_bgr)
    return blended
