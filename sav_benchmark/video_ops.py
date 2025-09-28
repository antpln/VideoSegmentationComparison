"""Video writing and overlay helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2  # type: ignore[import]
import numpy as np


@dataclass
class FrameStream:
    """Streaming view over frames with optional resizing and optional square padding."""

    frame_paths: Sequence[Path]
    start_idx: int
    target_hw: Tuple[int, int]
    original_hw: Tuple[int, int]
    content_hw: Tuple[int, int]
    pad_hw: Tuple[int, int, int, int]  # top, bottom, left, right
    scale_values: Tuple[float, float]
    interpolation: int = cv2.INTER_LINEAR

    def generator(self) -> Iterator[np.ndarray]:
        """Yield frames at the configured inference resolution."""
        content_h, content_w = self.content_hw
        pad_top, pad_bottom, pad_left, pad_right = self.pad_hw
        target_h, target_w = self.target_hw

        for offset, path in enumerate(self.frame_paths[self.start_idx:]):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"Could not read frame {path}")
            if (content_h, content_w) != self.original_hw:
                frame = cv2.resize(
                    frame,
                    (content_w, content_h),
                    interpolation=self.interpolation,
                )
            if any(self.pad_hw):
                frame = cv2.copyMakeBorder(
                    frame,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
            if frame.shape[0] != target_h or frame.shape[1] != target_w:
                frame = cv2.resize(
                    frame,
                    (target_w, target_h),
                    interpolation=self.interpolation,
                )
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Frame {path} must be uint8 BGR")
            yield frame

    def original_generator(self) -> Iterator[np.ndarray]:
        """Yield frames without resizing starting from ``start_idx``."""
        for path in self.frame_paths[self.start_idx:]:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"Could not read frame {path}")
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Frame {path} must be uint8 BGR")
            yield frame

    @property
    def scale_xy(self) -> Tuple[float, float]:
        """Return (scale_x, scale_y) used to map original -> inference coords (before padding)."""
        return self.scale_values

    def pad_offsets(self) -> Tuple[int, int]:
        """Return (pad_x, pad_y) offsets applied after resizing (left, top)."""
        pad_top, _, pad_left, _ = self.pad_hw
        return pad_left, pad_top

    def content_shape(self) -> Tuple[int, int]:
        return self.content_hw


def _compute_target_hw(
    original_hw: Tuple[int, int],
    imgsz: Optional[int],
    override_hw: Optional[Tuple[int, int]] = None,
    *,
    force_square: bool = False,
) -> Tuple[int, int]:
    """Return the (H, W) resize target honoring aspect ratio and max dimension."""

    if override_hw is not None:
        target_h, target_w = override_hw
        target_h = max(1, int(target_h))
        target_w = max(1, int(target_w))
    else:
        if imgsz is None or imgsz <= 0:
            target_h, target_w = original_hw
        else:
            max_dim = max(original_hw)
            if imgsz >= max_dim:
                target_h, target_w = original_hw
            else:
                scale = imgsz / float(max_dim)
                target_h = max(1, int(round(original_hw[0] * scale)))
                target_w = max(1, int(round(original_hw[1] * scale)))

    if force_square:
        side = max(target_h, target_w)
        target_h = target_w = side

    # Ensure even dimensions for H.264 writers.
    target_h = _ensure_even(target_h)
    target_w = _ensure_even(target_w)

    return target_h, target_w


def _ensure_even(value: int) -> int:
    if value % 2 != 0:
        value += 1
    return value


def prepare_frame_stream(
    frame_paths: Sequence[Path],
    *,
    start_idx: int = 0,
    imgsz: Optional[int] = None,
    interpolation: int = cv2.INTER_LINEAR,
    target_hw: Optional[Tuple[int, int]] = None,
    force_square: bool = False,
) -> FrameStream:
    """Create a ``FrameStream`` starting at ``start_idx`` with optional resizing."""

    if start_idx < 0 or start_idx >= len(frame_paths):
        raise IndexError(f"start_idx {start_idx} out of range for {len(frame_paths)} frames")

    first_path = frame_paths[start_idx]
    first_frame = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise FileNotFoundError(f"Could not read frame {first_path}")

    original_hw = first_frame.shape[:2]
    target_hw = _compute_target_hw(
        original_hw,
        imgsz,
        override_hw=target_hw,
        force_square=force_square,
    )

    target_h, target_w = target_hw
    original_h, original_w = original_hw

    pad_top = pad_bottom = pad_left = pad_right = 0
    content_h, content_w = target_h, target_w
    scale_x = target_w / float(original_w)
    scale_y = target_h / float(original_h)

    if force_square:
        side = target_h  # same as target_w when force_square
        ratio = min(side / float(original_h), side / float(original_w))
        content_h = max(1, int(round(original_h * ratio)))
        content_w = max(1, int(round(original_w * ratio)))
        content_h = min(side, _ensure_even(content_h))
        content_w = min(side, _ensure_even(content_w))

        pad_top = max(0, (side - content_h) // 2)
        pad_bottom = max(0, side - content_h - pad_top)
        pad_left = max(0, (side - content_w) // 2)
        pad_right = max(0, side - content_w - pad_left)

        scale_x = content_w / float(original_w)
        scale_y = content_h / float(original_h)

    scale_values = (scale_x, scale_y)
    pad_hw = (pad_top, pad_bottom, pad_left, pad_right)

    return FrameStream(
        frame_paths=frame_paths,
        start_idx=start_idx,
        target_hw=target_hw,
        original_hw=original_hw,
        content_hw=(content_h, content_w),
        pad_hw=pad_hw,
        scale_values=scale_values,
        interpolation=interpolation,
    )


def write_video_mp4(path_mp4: Path, frames_bgr: Iterable[np.ndarray], fps: float = 24.0) -> None:
    """Serialize a sequence of BGR frames to an H.264 mp4 file."""

    iterator = iter(frames_bgr)
    try:
        first = next(iterator)
    except StopIteration:
        return

    if first.dtype != np.uint8 or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("First frame must be uint8 BGR")

    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(path_mp4), fourcc, fps, (width, height))
    if not writer.isOpened():  # pragma: no cover - cv2 backend dependent
        raise RuntimeError(f"Could not open writer for {path_mp4}")

    try:
        writer.write(first)
        for idx, frame in enumerate(iterator, start=1):
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


def overlay_video_frames(
    frame_paths: Sequence[Path],
    masks_seq: Sequence[Optional[np.ndarray]],
    *,
    output_path: Path,
    fps: float,
    target_hw: Optional[Tuple[int, int]] = None,
    interpolation: int = cv2.INTER_LINEAR,
    alpha: float = 0.5,
) -> str:
    """Stream overlays to disk without holding the entire clip in memory."""

    def _iter_frames() -> Iterator[np.ndarray]:
        total_masks = len(masks_seq)
        for idx, path in enumerate(frame_paths):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"Could not read frame {path}")

            if target_hw is not None and frame.shape[:2] != target_hw:
                frame = cv2.resize(frame, (target_hw[1], target_hw[0]), interpolation=interpolation)

            mask = masks_seq[idx] if idx < total_masks else None
            if mask is None:
                yield frame
                continue

            mask_arr = np.asarray(mask)
            if mask_arr.size == 0:
                yield frame
                continue
            if mask_arr.ndim > 2:
                mask_arr = np.any(mask_arr.astype(bool), axis=0)
            else:
                mask_arr = mask_arr.astype(bool)

            if mask_arr.shape != frame.shape[:2]:
                mask_arr = cv2.resize(
                    mask_arr.astype(np.uint8),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            yield overlay_union(frame, mask_arr, alpha=alpha)

    write_video_mp4(output_path, _iter_frames(), fps)
    return str(output_path)
