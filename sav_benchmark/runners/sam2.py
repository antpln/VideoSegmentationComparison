"""Runners for SAM2 variants."""

from __future__ import annotations

import os
import time
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2  # type: ignore[import]
import numpy as np
import psutil

try:
    from ultralytics.models.sam import SAM2VideoPredictor  # type: ignore[import]
except Exception:  # pragma: no cover - ultralytics optional
    SAM2VideoPredictor = None  # type: ignore

from ..prompts import extract_bbox_from_mask, extract_points_from_mask
from .base import Model
from ..utils import cuda_sync, get_gpu_peaks, maybe_compile_module, reset_gpu_peaks
from ..video_ops import overlay_video_frames, prepare_frame_stream, write_video_mp4


DEBUG_LOGS = os.getenv("SAV_BENCH_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _log_debug(message: str) -> None:
    if DEBUG_LOGS:
        print(message)


def _square_override(imgsz: Optional[int]) -> Optional[Tuple[int, int]]:
    if imgsz is None or imgsz <= 0:
        return None
    side = int(imgsz)
    if side % 2 != 0:
        side += 1
    return side, side


def _temp_video(prefix: str) -> Path:
    """Generate a unique mp4 path in the temp directory."""
    handle = Path(tempfile.gettempdir()) / f"{prefix}{time.time_ns()}.mp4"
    return handle


def _result_to_mask(result: object) -> Optional[np.ndarray]:
    """Convert a predictor result object into a boolean mask."""
    if result is None:
        return None
    masks_attr = getattr(result, "masks", None)
    if masks_attr is None:
        return None
    data = getattr(masks_attr, "data", None)
    if data is None:
        return None
    try:
        data_np = data.cpu().numpy().astype(bool)  # type: ignore[attr-defined]
    except AttributeError:
        data_np = np.asarray(data).astype(bool)
    return np.any(data_np, axis=0) if data_np.ndim == 3 else data_np


def _sam2_predictor(overrides: Dict[str, object]):
    if SAM2VideoPredictor is None:
        raise ImportError("Ultralytics SAM2VideoPredictor is not available")
    return SAM2VideoPredictor(overrides=overrides)


def _maybe_compile_predictor_model(
    predictor,
    *,
    compile_model: bool,
    compile_mode: Optional[str],
    compile_backend: Optional[str],
) -> None:
    if not compile_model:
        return
    module = getattr(predictor, "model", None)
    compiled, ok = maybe_compile_module(module, mode=compile_mode, backend=compile_backend)
    if ok:
        predictor.model = compiled




def _run_points(
    frames_24fps: List[Path],
    prompt_frame_idx: int,
    prompt_mask: np.ndarray,
    imgsz: int,
    weight_name: str,
    device: str,
    out_dir: Optional[Path] = None,
    overlay_name: Optional[str] = None,
    clip_fps: float = 24.0,
    num_points: int = 5,
    *,
    precision=None,
    max_clip_frames: Optional[int] = None,
    compile_model: bool = False,
    compile_mode: Optional[str] = "reduce-overhead",
    compile_backend: Optional[str] = None,
) -> Dict[str, object]:
    # Derive a SINGLE positive prompt point from the supplied mask (fairness across models).
    points, labels = extract_points_from_mask(prompt_mask, num_points=1)
    if not points:
        return {
            "secs": None,
            "fps": None,
            "latency_ms": None,
            "gpu_peak_alloc": None,
            "gpu_peak_reserved": None,
            "cpu_peak_rss": None,
            "masks_seq": None,
            "overlay": None,
            "frames": len(frames_24fps),
            "H": prompt_mask.shape[0],
            "W": prompt_mask.shape[1],
            "num_points": 0,
        }

    precision_scope = precision if precision is not None else (lambda: nullcontext())

    total_frames = len(frames_24fps)
    if prompt_frame_idx >= total_frames:
        raise IndexError(f"Prompt index {prompt_frame_idx} is out of range for {total_frames} frames")

    clip_end = total_frames
    if max_clip_frames is not None and max_clip_frames > 0:
        clip_end = min(total_frames, prompt_frame_idx + max_clip_frames)

    clipped_paths = frames_24fps[:clip_end]

    square_override = _square_override(imgsz)
    full_stream = prepare_frame_stream(
        clipped_paths,
        imgsz=imgsz,
        target_hw=square_override,
        force_square=True,
    )
    prompt_stream = prepare_frame_stream(
        clipped_paths,
        start_idx=prompt_frame_idx,
        imgsz=imgsz,
        target_hw=square_override,
        force_square=True,
    )
    infer_h, infer_w = prompt_stream.target_hw
    orig_h, orig_w = prompt_stream.original_hw
    _log_debug(
        f"[DEBUG SAM2 {overlay_name or ''}] resolution orig={orig_w}x{orig_h} -> infer={infer_w}x{infer_h}"
    )

    # SAM2 expects a video file; write the subsequence to a temporary mp4 using a generator.
    tmp_mp4 = _temp_video("sam2_points_")
    write_video_mp4(
        tmp_mp4,
        prompt_stream.generator(),
        clip_fps,
    )

    process = psutil.Process()
    reset_gpu_peaks()
    cpu_peak = process.memory_info().rss
    setup_start = time.perf_counter()

    # Configure the predictor for video tracking on the chosen weights.
    overrides = {
        "conf": 0.25,
        "task": "track",
        "mode": "predict",
        "imgsz": imgsz,
        "model": weight_name,
        "save": False,
        "device": device,
    }

    # Replace sub_masks with a dict for sliding window
    # Retain every predicted frame so evaluation covers the full propagated sequence.
    sub_masks: Dict[int, Optional[np.ndarray]] = {}
    inference_start: Optional[float] = None
    sub_frame_count = clip_end - prompt_frame_idx

    scale_x, scale_y = prompt_stream.scale_xy
    pad_x, pad_y = prompt_stream.pad_offsets()
    content_h, content_w = prompt_stream.content_shape()
    points_np = np.array([[p[0] * scale_x, p[1] * scale_y] for p in points], dtype=np.float32)
    points_np[:, 0] = np.clip(points_np[:, 0] + pad_x, 0, max(0, infer_w - 1))
    points_np[:, 1] = np.clip(points_np[:, 1] + pad_y, 0, max(0, infer_h - 1))
    points_list = points_np.tolist()
    labels_np = np.array(labels, dtype=np.int32)
    labels_list = labels_np.tolist()

    try:
        predictor = _sam2_predictor(overrides)
        # Optionally wrap the underlying module in torch.compile.
        _maybe_compile_predictor_model(
            predictor,
            compile_model=compile_model,
            compile_mode=compile_mode,
            compile_backend=compile_backend,
        )

        has_video_api = all(
            hasattr(predictor, attr)
            for attr in ("init_state", "add_new_points_or_box", "propagate_in_video")
        )

        if has_video_api:
            with precision_scope():
                # Seed the tracker with user prompts on the first frame.
                inference_state = predictor.init_state(str(tmp_mp4))
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    points=points_np,
                    labels=labels_np,
                )
                # Start timing AFTER predictor build, init_state, and prompt seeding.
                inference_start = time.perf_counter()

                # Collect masks produced for each propagated frame.
                for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
                    if mask_logits is not None and len(mask_logits) > 0:
                        mask = (mask_logits[0] > 0.0).cpu().numpy().astype(bool)
                        if mask.ndim > 2:
                            mask = np.any(mask, axis=0)
                        if mask.shape != (infer_h, infer_w):
                            mask = cv2.resize(
                                mask.astype(np.uint8),
                                (infer_w, infer_h),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                        if 0 <= frame_idx < sub_frame_count:
                            sub_masks[frame_idx] = mask
        else:
            # Fallback for newer Ultralytics builds that only expose the streaming API.
            with precision_scope():
                iterator = None
                for kwargs in (
                    {"stream": True, "points": points_list, "labels": labels_list},
                    {"points": points_list, "labels": labels_list},
                ):
                    try:
                        iterator = predictor(source=str(tmp_mp4), **kwargs)
                        break
                    except TypeError:
                        continue
                if iterator is None:
                    raise TypeError("SAM2VideoPredictor streaming interface not supported by this build")
                inference_start = time.perf_counter()

                for idx, result in enumerate(iterator):
                    mask = _result_to_mask(result)
                    if mask is not None:
                        if mask.ndim > 2:
                            mask = np.any(mask.astype(bool), axis=0)
                        else:
                            mask = mask.astype(bool)
                        if mask.shape != (infer_h, infer_w):
                            mask = cv2.resize(
                                mask.astype(np.uint8),
                                (infer_w, infer_h),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                    if mask is not None and idx < sub_frame_count:
                        sub_masks[idx] = mask
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] SAM2 points prediction failed: {exc}")
        sub_masks = [None] * sub_frame_count

    cuda_sync()
    if inference_start is None:  # If failure before timing began, treat inference time as 0.
        inference_start = time.perf_counter()
    duration = max(1e-9, time.perf_counter() - inference_start)
    setup_secs = inference_start - setup_start
    gpu_alloc, gpu_reserved = get_gpu_peaks()
    cpu_peak = max(cpu_peak, process.memory_info().rss)

    # Convert sub_masks dict or list to list for output (None for missing)
    if isinstance(sub_masks, dict):
        sub_masks_list = [sub_masks.get(i, None) for i in range(sub_frame_count)]
    else:
        sub_masks_list = [sub_masks[i] if i < len(sub_masks) else None for i in range(sub_frame_count)]
    masks_seq: List[Optional[np.ndarray]] = [None] * prompt_frame_idx + sub_masks_list

    overlay_path = None
    if out_dir and overlay_name:
        # Persist overlay videos only when requested.
        overlay_path = Path(out_dir) / f"{overlay_name}.mp4"
        overlay_video_frames(
            clipped_paths[prompt_frame_idx:clip_end],
            masks_seq[prompt_frame_idx:clip_end],
            output_path=overlay_path,
            fps=clip_fps,
            target_hw=full_stream.target_hw,
        )

    try:
        tmp_mp4.unlink(missing_ok=True)
    except OSError:
        pass

    fps = sub_frame_count / duration if sub_frame_count else 0.0
    latency_ms = 1000.0 / fps if fps > 0 else None
    return {
        "secs": duration,
        "fps": fps,
        "latency_ms": latency_ms,
        "gpu_peak_alloc": gpu_alloc,
        "gpu_peak_reserved": gpu_reserved,
        "cpu_peak_rss": cpu_peak,
        "masks_seq": masks_seq,
        "overlay": str(overlay_path) if overlay_path else None,
        "frames": clip_end - prompt_frame_idx,
        "processed_end_frame": clip_end,
        "H": orig_h,
        "W": orig_w,
        "infer_H": infer_h,
        "infer_W": infer_w,
        "num_points": len(points),
        "scale_x": scale_x,
        "scale_y": scale_y,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "content_H": content_h,
        "content_W": content_w,
        "setup_ms": round(setup_secs * 1000.0, 2),
    }


def _run_bbox(
    frames_24fps: List[Path],
    prompt_frame_idx: int,
    prompt_mask: np.ndarray,
    imgsz: int,
    weight_name: str,
    device: str,
    out_dir: Optional[Path] = None,
    overlay_name: Optional[str] = None,
    clip_fps: float = 24.0,
    *,
    precision=None,
    max_clip_frames: Optional[int] = None,
    compile_model: bool = False,
    compile_mode: Optional[str] = "reduce-overhead",
    compile_backend: Optional[str] = None,
) -> Dict[str, object]:
    bbox = extract_bbox_from_mask(prompt_mask)
    if bbox is None:
        return {
            "secs": None,
            "fps": None,
            "latency_ms": None,
            "gpu_peak_alloc": None,
            "gpu_peak_reserved": None,
            "cpu_peak_rss": None,
            "masks_seq": None,
            "overlay": None,
            "frames": len(frames_24fps),
            "H": 0,
            "W": 0,
            "infer_H": 0,
            "infer_W": 0,
        }

    precision_scope = precision if precision is not None else (lambda: nullcontext())

    total_frames = len(frames_24fps)
    if prompt_frame_idx >= total_frames:
        raise IndexError(f"Prompt index {prompt_frame_idx} is out of range for {total_frames} frames")

    clip_end = total_frames
    if max_clip_frames is not None and max_clip_frames > 0:
        clip_end = min(total_frames, prompt_frame_idx + max_clip_frames)

    clipped_paths = frames_24fps[:clip_end]

    square_override = _square_override(imgsz)
    full_stream = prepare_frame_stream(
        clipped_paths,
        imgsz=imgsz,
        target_hw=square_override,
        force_square=True,
    )
    prompt_stream = prepare_frame_stream(
        clipped_paths,
        start_idx=prompt_frame_idx,
        imgsz=imgsz,
        target_hw=square_override,
        force_square=True,
    )
    infer_h, infer_w = prompt_stream.target_hw
    orig_h, orig_w = prompt_stream.original_hw
    _log_debug(
        f"[DEBUG SAM2 {overlay_name or ''}] resolution orig={orig_w}x{orig_h} -> infer={infer_w}x{infer_h}"
    )

    # Persist the subsequence so SAM2VideoPredictor can stream it.
    tmp_mp4 = _temp_video("sam2_bbox_")
    write_video_mp4(tmp_mp4, prompt_stream.generator(), clip_fps)

    process = psutil.Process()
    reset_gpu_peaks()
    cpu_peak = process.memory_info().rss
    setup_start = time.perf_counter()

    overrides = {
        "conf": 0.25,
        "task": "track",
        "mode": "predict",
        "imgsz": imgsz,
        "model": weight_name,
        "save": False,
        "device": device,
    }

    # Replace sub_masks with a dict for sliding window
    # Keep the full prediction history; dropping frames would zero-out metrics later.
    sub_masks: Dict[int, Optional[np.ndarray]] = {}
    inference_start: Optional[float] = None
    sub_frame_count = clip_end - prompt_frame_idx

    scale_x, scale_y = prompt_stream.scale_xy
    pad_x, pad_y = prompt_stream.pad_offsets()
    content_h, content_w = prompt_stream.content_shape()
    x1, y1, x2, y2 = bbox
    scaled_x = sorted([x1 * scale_x + pad_x, x2 * scale_x + pad_x])
    scaled_y = sorted([y1 * scale_y + pad_y, y2 * scale_y + pad_y])
    bbox_np = np.array(
        [
            np.clip(scaled_x[0], 0, max(0, infer_w - 1)),
            np.clip(scaled_y[0], 0, max(0, infer_h - 1)),
            np.clip(scaled_x[1], 0, max(0, infer_w - 1)),
            np.clip(scaled_y[1], 0, max(0, infer_h - 1)),
        ],
        dtype=np.float32,
    )
    bbox_list = bbox_np.astype(float).tolist()
    try:
        predictor = _sam2_predictor(overrides)
        _maybe_compile_predictor_model(
            predictor,
            compile_model=compile_model,
            compile_mode=compile_mode,
            compile_backend=compile_backend,
        )

        has_video_api = all(
            hasattr(predictor, attr)
            for attr in ("init_state", "add_new_points_or_box", "propagate_in_video")
        )

        if has_video_api:
            with precision_scope():
                # Register the box prompt at frame zero before propagation.
                inference_state = predictor.init_state(str(tmp_mp4))
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    box=bbox_np,
                )
                inference_start = time.perf_counter()

                for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
                    if mask_logits is not None and len(mask_logits) > 0:
                        mask = (mask_logits[0] > 0.0).cpu().numpy().astype(bool)
                        if mask.ndim > 2:
                            mask = np.any(mask, axis=0)
                        if mask.shape != (infer_h, infer_w):
                            mask = cv2.resize(
                                mask.astype(np.uint8),
                                (infer_w, infer_h),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                        if 0 <= frame_idx < sub_frame_count:
                            sub_masks[frame_idx] = mask
        else:
            # Fallback for predictor builds exposing only the streaming interface.
            with precision_scope():
                iterator = None
                for kwargs in (
                    {"stream": True, "bboxes": [bbox_list]},
                    {"bboxes": [bbox_list]},
                    {"stream": True, "boxes": [bbox_list]},
                    {"boxes": [bbox_list]},
                ):
                    try:
                        iterator = predictor(source=str(tmp_mp4), **kwargs)
                        break
                    except TypeError:
                        continue
                if iterator is None:
                    raise TypeError("SAM2VideoPredictor streaming bbox interface not supported by this build")
                inference_start = time.perf_counter()

                for idx, result in enumerate(iterator):
                    mask = _result_to_mask(result)
                    if mask is not None:
                        if mask.ndim > 2:
                            mask = np.any(mask.astype(bool), axis=0)
                        else:
                            mask = mask.astype(bool)
                        if mask.shape != (infer_h, infer_w):
                            mask = cv2.resize(
                                mask.astype(np.uint8),
                                (infer_w, infer_h),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                    if mask is not None and idx < sub_frame_count:
                        sub_masks[idx] = mask
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] SAM2 bbox prediction failed: {exc}")
        sub_masks = [None] * sub_frame_count

    cuda_sync()
    if inference_start is None:
        inference_start = time.perf_counter()
    duration = max(1e-9, time.perf_counter() - inference_start)
    setup_secs = inference_start - setup_start
    gpu_alloc, gpu_reserved = get_gpu_peaks()
    cpu_peak = max(cpu_peak, process.memory_info().rss)

    if isinstance(sub_masks, dict):
        sub_masks_list = [sub_masks.get(i, None) for i in range(sub_frame_count)]
    else:
        sub_masks_list = [sub_masks[i] if i < len(sub_masks) else None for i in range(sub_frame_count)]
    masks_seq: List[Optional[np.ndarray]] = [None] * prompt_frame_idx + sub_masks_list

    overlay_path = None
    if out_dir and overlay_name:
        # Persist overlay videos only when requested.
        overlay_path = Path(out_dir) / f"{overlay_name}.mp4"
        overlay_video_frames(
            clipped_paths,
            masks_seq[:clip_end],
            output_path=overlay_path,
            fps=clip_fps,
            target_hw=full_stream.target_hw,
        )

    try:
        tmp_mp4.unlink(missing_ok=True)
    except OSError:
        pass

    fps = sub_frame_count / duration if sub_frame_count else 0.0
    latency_ms = 1000.0 / fps if fps > 0 else None

    return {
        "secs": duration,
        "fps": fps,
        "latency_ms": latency_ms,
        "gpu_peak_alloc": gpu_alloc,
        "gpu_peak_reserved": gpu_reserved,
        "cpu_peak_rss": cpu_peak,
        "masks_seq": masks_seq,
        "overlay": str(overlay_path) if overlay_path else None,
        "frames": clip_end - prompt_frame_idx,
        "H": orig_h,
        "W": orig_w,
        "infer_H": infer_h,
        "infer_W": infer_w,
        "processed_end_frame": clip_end,
        "bbox": bbox,
        "bbox_infer": bbox_list,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "content_H": content_h,
        "content_W": content_w,
        "setup_ms": round(setup_secs * 1000.0, 2),
    }


class SAM2(Model):
    """Concrete runner for Ultralytics SAM2 video models.

    The heavy lifting lives in `_run_points` / `_run_bbox` above. This wrapper
    simply binds those helpers as bound methods so the registry can expose them
    through the abstract base class.
    """

    def __init__(self) -> None:
        super().__init__("sam2")

    def run_points(
        self,
        frames_24fps: List[Path],
        prompt_frame_idx: int,
        prompt_mask: np.ndarray,
        imgsz: int,
        weight_name: str,
        device: str,
        out_dir: Optional[Path] = None,
        overlay_name: Optional[str] = None,
        clip_fps: float = 24.0,
        *,
        precision=None,
        max_clip_frames: Optional[int] = None,
    compile_model: bool = False,
    compile_mode: Optional[str] = "reduce-overhead",
    compile_backend: Optional[str] = None,
    ) -> Dict[str, object]:
        return _run_points(
            frames_24fps,
            prompt_frame_idx,
            prompt_mask,
            imgsz,
            weight_name,
            device,
            out_dir=out_dir,
            overlay_name=overlay_name,
            clip_fps=clip_fps,
            precision=precision,
            max_clip_frames=max_clip_frames,
            compile_model=compile_model,
            compile_mode=compile_mode,
            compile_backend=compile_backend,
        )

    def run_bbox(
        self,
        frames_24fps: List[Path],
        prompt_frame_idx: int,
        prompt_mask: np.ndarray,
        imgsz: int,
        weight_name: str,
        device: str,
        out_dir: Optional[Path] = None,
        overlay_name: Optional[str] = None,
        clip_fps: float = 24.0,
        *,
        precision=None,
        max_clip_frames: Optional[int] = None,
    compile_model: bool = False,
    compile_mode: Optional[str] = "reduce-overhead",
    compile_backend: Optional[str] = None,
    ) -> Dict[str, object]:
        return _run_bbox(
            frames_24fps,
            prompt_frame_idx,
            prompt_mask,
            imgsz,
            weight_name,
            device,
            out_dir=out_dir,
            overlay_name=overlay_name,
            clip_fps=clip_fps,
            precision=precision,
            max_clip_frames=max_clip_frames,
            compile_model=compile_model,
            compile_mode=compile_mode,
            compile_backend=compile_backend,
        )


SAM2_MODEL = SAM2()
SAM2_RUNNERS = SAM2_MODEL.register()
