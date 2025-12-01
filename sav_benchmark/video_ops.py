"""Small utilities for writing videos and overlaying masks on frames.

This is a compact, dependency-light implementation that provides the
functions used by the runner modules: `overlay_union`, `write_video_mp4`,
`overlay_video_frames`, and `prepare_frame_stream`.

It's intentionally minimal: it uses OpenCV (`cv2`) for I/O and simple
alpha-blending for overlays so the runners can persist diagnostic videos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Generator, List, Optional, Tuple

import cv2  # type: ignore[import]
import numpy as np


def overlay_union(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
	"""Return a BGR image with the boolean `mask` overlaid in red.

	- frame: HxWx3 BGR uint8
	- mask: HxW boolean or uint8
	"""
	if mask is None:
		return frame
	mask_arr = np.asarray(mask)
	if mask_arr.dtype != np.bool_:
		mask_arr = mask_arr.astype(bool)
	# Ensure mask matches spatial dims
	if mask_arr.shape != frame.shape[:2]:
		mask_arr = cv2.resize(mask_arr.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

	overlay = frame.copy()
	# Red highlight for mask (BGR -> (0,0,255))
	overlay[mask_arr, 0] = (1.0 - alpha) * overlay[mask_arr, 0] + alpha * 0
	overlay[mask_arr, 1] = (1.0 - alpha) * overlay[mask_arr, 1] + alpha * 0
	overlay[mask_arr, 2] = (1.0 - alpha) * overlay[mask_arr, 2] + alpha * 255
	return overlay.astype(np.uint8)


def write_video_mp4(output_path: Path, frames: Iterable[np.ndarray], fps: float = 24.0) -> None:
	"""Write frames (list or generator) to an MP4 file using OpenCV.

	Accepts either a list of images or a generator. The codec uses MP4V
	for portability. If frames is empty the function creates a zero-length
	file by doing nothing.
	"""
	output_path = Path(output_path)
	# If frames is a generator, convert first to detect size
	iterator = iter(frames)
	try:
		first = next(iterator)
	except StopIteration:
		# Nothing to write
		return

	h, w = first.shape[:2]
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))
	if not writer.isOpened():
		raise RuntimeError(f"Failed to open video writer for {output_path}")
	writer.write(first)
	for frm in iterator:
		writer.write(frm)
	writer.release()


def overlay_video_frames(frames: Iterable[np.ndarray], masks_seq: Iterable[Optional[np.ndarray]], output_path: Path, fps: float = 24.0, target_hw: Optional[Tuple[int, int]] = None) -> str:
	"""Apply overlays to frames using masks_seq and write a video file.

	masks_seq may contain None entries; those frames are written without overlay.
	"""
	# Build an in-memory list (frames may already be a generator) to pair masks
	frame_list = list(frames)
	masks_list = list(masks_seq)
	overlays: List[np.ndarray] = []
	for idx, frame in enumerate(frame_list):
		mask = masks_list[idx] if idx < len(masks_list) else None
		if mask is None:
			overlays.append(frame)
			continue
		ov = overlay_union(frame, mask, alpha=0.5)
		overlays.append(ov)
	write_video_mp4(output_path, overlays, fps=fps)
	return str(output_path)


class _FrameStream:
	"""Simple helper that exposes the minimal API expected by runners.

	Attributes used by runners:
	- generator(): yields frames as BGR numpy arrays
	- target_hw: (H, W) used for inference
	- original_hw: (H, W) of source frames
	- scale_xy: (scale_x, scale_y) used to map prompt coords to inference coords
	- pad_offsets(): returns (pad_x, pad_y) (we keep zero padding)
	- content_shape(): returns (content_h, content_w) same as target
	"""

	def __init__(self, paths: Iterable[Path], imgsz: Optional[int] = None, target_hw: Optional[Tuple[int, int]] = None, force_square: bool = False):
		self._paths = list(paths)
		# read first frame to determine original size
		if not self._paths:
			self.original_hw = (0, 0)
			self.target_hw = target_hw or (0, 0)
		else:
			first = cv2.imread(str(self._paths[0]), cv2.IMREAD_COLOR)
			if first is None:
				raise FileNotFoundError(f"Could not read frame {self._paths[0]}")
			self.original_hw = (first.shape[0], first.shape[1])
			if target_hw is not None:
				self.target_hw = target_hw
			elif imgsz is not None:
				# square override
				side = int(imgsz)
				self.target_hw = (side, side)
			else:
				self.target_hw = self.original_hw
		oh, ow = self.original_hw
		th, tw = self.target_hw
		self.scale_xy = (float(tw) / float(ow) if ow else 1.0, float(th) / float(oh) if oh else 1.0)

	def generator(self) -> Generator[np.ndarray, None, None]:
		for p in self._paths:
			frm = cv2.imread(str(p), cv2.IMREAD_COLOR)
			if frm is None:
				continue
			# resize to target_hw if needed
			if frm.shape[:2] != (self.target_hw[0], self.target_hw[1]):
				frm = cv2.resize(frm, (self.target_hw[1], self.target_hw[0]), interpolation=cv2.INTER_LINEAR)
			yield frm

	def pad_offsets(self) -> Tuple[int, int]:
		return 0, 0

	def content_shape(self) -> Tuple[int, int]:
		return self.target_hw


def prepare_frame_stream(paths: Iterable[Path], imgsz: Optional[int] = None, target_hw: Optional[Tuple[int, int]] = None, force_square: bool = False) -> _FrameStream:
	return _FrameStream(paths, imgsz=imgsz, target_hw=target_hw, force_square=force_square)

