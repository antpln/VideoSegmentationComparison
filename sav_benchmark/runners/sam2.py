"""Runners for SAM2 variants."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2  # type: ignore[import]
import numpy as np
import psutil
import tempfile

try:
    from ultralytics.models.sam import SAM2VideoPredictor  # type: ignore[import]
except Exception:  # pragma: no cover - ultralytics optional
    SAM2VideoPredictor = None  # type: ignore

from ..prompts import extract_bbox_from_mask, extract_points_from_mask
from .base import Model
from ..utils import cuda_sync, get_gpu_peaks, maybe_compile_module, reset_gpu_peaks
from ..video_ops import overlay_union, write_video_mp4


def _read_frames(frame_paths: Iterable[Path]) -> List[np.ndarray]:
    """Load frames as BGR numpy arrays."""
    frames: List[np.ndarray] = []
    for path in frame_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Could not read frame {path}")
        frames.append(frame)
    return frames


def _temp_video(prefix: str) -> Path:
    """Generate a unique mp4 path in the temp directory."""
    handle = Path(tempfile.gettempdir()) / f"{prefix}{time.time_ns()}.mp4"
    return handle


def _record_overlays(frames: List[np.ndarray], masks_seq: List[Optional[np.ndarray]], output_path: Path, fps: float) -> str:
    """Blend mask overlays onto frames to aid visual inspection."""
    if not frames:
        return ""
    overlays: List[np.ndarray] = []
    for idx, frame in enumerate(frames):
        mask = masks_seq[idx] if idx < len(masks_seq) else None
        if mask is None:
            overlays.append(frame)
            continue
        mask_arr = np.asarray(mask)
        if mask_arr.size == 0:
            overlays.append(frame)
            continue
        if mask_arr.ndim > 2:
            mask_arr = np.any(mask_arr.astype(bool), axis=0)
        else:
            mask_arr = mask_arr.astype(bool)
        if mask_arr.ndim != 2:
            overlays.append(frame)
            continue
        if mask_arr.shape != frame.shape[:2]:
            mask_arr = cv2.resize(
                mask_arr.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        overlays.append(overlay_union(frame, mask_arr, alpha=0.5))
    write_video_mp4(output_path, overlays, fps)
    return str(output_path)


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
    compile_mode: str | None,
    compile_backend: str | None,
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
    *,
    compile_model: bool = False,
    compile_mode: str | None = "reduce-overhead",
    compile_backend: str | None = None,
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

    frames = _read_frames(frames_24fps)
    # Only propagate from the prompt frame onwards.
    sub_frames = frames[prompt_frame_idx:]
    if not sub_frames:
        raise IndexError(f"Prompt index {prompt_frame_idx} is out of range for {len(frames)} frames")
    height, width = sub_frames[0].shape[:2]

    # SAM2 expects a video file; write the subsequence to a temporary mp4.
    tmp_mp4 = _temp_video("sam2_points_")
    write_video_mp4(tmp_mp4, sub_frames, clip_fps)

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

    sub_masks: List[Optional[np.ndarray]] = [None] * len(sub_frames)
    inference_start: float | None = None
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
            # Seed the tracker with user prompts on the first frame.
            inference_state = predictor.init_state(str(tmp_mp4))
            points_np = np.array(points, dtype=np.float32)
            labels_np = np.array(labels, dtype=np.int32)
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
                    if mask.shape != (height, width):
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    if 0 <= frame_idx < len(sub_masks):
                        sub_masks[frame_idx] = mask
        else:
            # Fallback for newer Ultralytics builds that only expose the streaming API.
            iterator = None
            for kwargs in (
                {"stream": True, "points": points, "labels": labels},
                {"points": points, "labels": labels},
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
                    if mask.shape != (height, width):
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                if mask is not None and idx < len(sub_masks):
                    sub_masks[idx] = mask
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] SAM2 points prediction failed: {exc}")
        sub_masks = [None] * len(sub_frames)

    cuda_sync()
    if inference_start is None:  # If failure before timing began, treat inference time as 0.
        inference_start = time.perf_counter()
    duration = max(1e-9, time.perf_counter() - inference_start)
    setup_secs = inference_start - setup_start
    gpu_alloc, gpu_reserved = get_gpu_peaks()
    cpu_peak = max(cpu_peak, process.memory_info().rss)

    masks_seq: List[Optional[np.ndarray]] = [None] * prompt_frame_idx + sub_masks

    overlay_path = None
    if out_dir and overlay_name:
        # Persist overlay videos only when requested.
        overlay_path = Path(out_dir) / f"{overlay_name}.mp4"
        _record_overlays(frames, masks_seq, overlay_path, clip_fps)

    try:
        tmp_mp4.unlink(missing_ok=True)
    except OSError:
        pass

    fps = len(sub_frames) / duration
    return {
    "secs": duration,
        "fps": fps,
        "latency_ms": 1000.0 / fps,
        "gpu_peak_alloc": gpu_alloc,
        "gpu_peak_reserved": gpu_reserved,
        "cpu_peak_rss": cpu_peak,
        "masks_seq": masks_seq,
        "overlay": str(overlay_path) if overlay_path else None,
        "frames": len(frames),
        "H": height,
        "W": width,
        "num_points": len(points),
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
    compile_model: bool = False,
    compile_mode: str | None = "reduce-overhead",
    compile_backend: str | None = None,
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
        }

    frames = _read_frames(frames_24fps)
    sub_frames = frames[prompt_frame_idx:]
    if not sub_frames:
        raise IndexError(f"Prompt index {prompt_frame_idx} is out of range for {len(frames)} frames")
    height, width = sub_frames[0].shape[:2]

    # Persist the subsequence so SAM2VideoPredictor can stream it.
    tmp_mp4 = _temp_video("sam2_bbox_")
    write_video_mp4(tmp_mp4, sub_frames, clip_fps)

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

    sub_masks: List[Optional[np.ndarray]] = [None] * len(sub_frames)
    inference_start: float | None = None
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
            # Register the box prompt at frame zero before propagation.
            inference_state = predictor.init_state(str(tmp_mp4))
            bbox_np = np.array(bbox, dtype=np.float32)
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
                    if mask.shape != (height, width):
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    if 0 <= frame_idx < len(sub_masks):
                        sub_masks[frame_idx] = mask
        else:
            # Fallback for predictor builds exposing only the streaming interface.
            iterator = None
            for kwargs in (
                {"stream": True, "bboxes": [bbox]},
                {"bboxes": [bbox]},
                {"stream": True, "boxes": [bbox]},
                {"boxes": [bbox]},
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
                    if mask.shape != (height, width):
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                if mask is not None and idx < len(sub_masks):
                    sub_masks[idx] = mask
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] SAM2 bbox prediction failed: {exc}")
        sub_masks = [None] * len(sub_frames)

    cuda_sync()
    if inference_start is None:
        inference_start = time.perf_counter()
    duration = max(1e-9, time.perf_counter() - inference_start)
    setup_secs = inference_start - setup_start
    gpu_alloc, gpu_reserved = get_gpu_peaks()
    cpu_peak = max(cpu_peak, process.memory_info().rss)

    masks_seq: List[Optional[np.ndarray]] = [None] * prompt_frame_idx + sub_masks

    overlay_path = None
    if out_dir and overlay_name:
        # Persist overlay videos only when requested.
        overlay_path = Path(out_dir) / f"{overlay_name}.mp4"
        _record_overlays(frames, masks_seq, overlay_path, clip_fps)

    try:
        tmp_mp4.unlink(missing_ok=True)
    except OSError:
        pass

    fps = len(sub_frames) / duration

    return {
        "secs": duration,
        "fps": fps,
        "latency_ms": 1000.0 / fps,
        "gpu_peak_alloc": gpu_alloc,
        "gpu_peak_reserved": gpu_reserved,
        "cpu_peak_rss": cpu_peak,
        "masks_seq": masks_seq,
        "overlay": str(overlay_path) if overlay_path else None,
        "frames": len(frames),
        "H": height,
        "W": width,
        "bbox": bbox,
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
        compile_model: bool = False,
        compile_mode: str | None = "reduce-overhead",
        compile_backend: str | None = None,
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
        compile_model: bool = False,
        compile_mode: str | None = "reduce-overhead",
        compile_backend: str | None = None,
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
            compile_model=compile_model,
            compile_mode=compile_mode,
            compile_backend=compile_backend,
        )


SAM2_MODEL = SAM2()
SAM2_RUNNERS = SAM2_MODEL.register()
