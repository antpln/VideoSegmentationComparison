"""Runners for EdgeTAM variants."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2  # type: ignore[import]
import numpy as np
import psutil

try:
    from EdgeTAM.sam2.build_sam import build_sam2_video_predictor  # type: ignore[import]
except Exception:  # pragma: no cover - EdgeTAM optional
    build_sam2_video_predictor = None  # type: ignore

from ..prompts import extract_bbox_from_mask, extract_points_from_mask
from ..utils import cuda_sync, expand_path, get_gpu_peaks, maybe_compile_module, reset_gpu_peaks
from ..video_ops import overlay_union, write_video_mp4


def _find_existing(paths: List[str]) -> Optional[Path]:
    """Return the first existing path from the provided candidate list."""
    for candidate in paths:
        resolved = Path(expand_path(candidate))
        if resolved.exists():
            return resolved
    return None


def _read_frame(path: Path) -> np.ndarray:
    """Load a single frame as a BGR numpy array."""
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Could not read frame {path}")
    return frame


def _record_overlays(frames: List[Path], masks_seq: List[Optional[np.ndarray]], output_path: Path, fps: float) -> str:
    """Write an overlay video that blends predictions with their source frames."""
    overlays: List[np.ndarray] = []
    for idx, frame_path in enumerate(frames):
        frame = _read_frame(frame_path)
        mask = masks_seq[idx] if idx < len(masks_seq) else None
        if mask is None:
            overlays.append(frame)
            continue
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        overlays.append(overlay_union(frame, mask, alpha=0.5))
    write_video_mp4(output_path, overlays, fps)
    return str(output_path)


def _init_predictor(weight_name: str) -> object:
    """Build an EdgeTAM video predictor using discovered config/weights."""
    if build_sam2_video_predictor is None:
        raise ImportError("EdgeTAM is not installed")

    config_candidates = [
        ("configs/edgetam.yaml", "./EdgeTAM/sam2/configs/edgetam.yaml"),
    ]
    checkpoint_candidates = [
        "./EdgeTAM/checkpoints/edgetam.pt",
        weight_name,
    ]

    config_name: Optional[str] = None
    for hydra_name, disk_path in config_candidates:
        resolved = Path(expand_path(disk_path))
        if resolved.exists():
            config_name = hydra_name
            break

    checkpoint_path = _find_existing(checkpoint_candidates)

    if config_name is None:
        raise FileNotFoundError(f"EdgeTAM config not found among: {[p for _, p in config_candidates]}")
    if checkpoint_path is None:
        raise FileNotFoundError(f"EdgeTAM checkpoint not found among: {checkpoint_candidates}")

    return build_sam2_video_predictor(config_file=config_name, ckpt_path=str(checkpoint_path))


def _maybe_compile_edgetam_predictor(
    predictor,
    *,
    compile_model: bool,
    compile_mode: str | None,
    compile_backend: str | None,
) -> None:
    """Optionally wrap EdgeTAM submodules in torch.compile for speed."""
    if not compile_model:
        return

    for attr in ("model", "sam2"):
        candidate = getattr(predictor, attr, None)
        compiled, ok = maybe_compile_module(candidate, mode=compile_mode, backend=compile_backend)
        if ok:
            setattr(predictor, attr, compiled)
            if attr == "sam2":
                inner = getattr(compiled, "model", None)
                inner_compiled, inner_ok = maybe_compile_module(inner, mode=compile_mode, backend=compile_backend)
                if inner_ok:
                    compiled.model = inner_compiled


def _verify_predictor_interfaces(predictor: object) -> None:
    """Ensure the required SAM2 video APIs are available."""
    required = ("init_state", "add_new_points_or_box", "propagate_in_video")
    missing = [name for name in required if not hasattr(predictor, name)]
    if missing:
        raise AttributeError(
            "EdgeTAM/SAM2 build is missing video tracking APIs: "
            + ", ".join(missing)
            + ". Update the dependency to a version that exposes video propagation."
        )


def run_with_points(
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
    compile_model: bool = False,
    compile_mode: str | None = "reduce-overhead",
    compile_backend: str | None = None,
) -> Dict[str, object]:
    _ = (imgsz, device)
    # Translate the mask into positive point prompts.
    points, labels = extract_points_from_mask(prompt_mask, num_points)
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

    predictor = _init_predictor(weight_name)
    _verify_predictor_interfaces(predictor)
    # Torch.compile can provide a modest boost on supported environments.
    _maybe_compile_edgetam_predictor(
        predictor,
        compile_model=compile_model,
        compile_mode=compile_mode,
        compile_backend=compile_backend,
    )
    # Focus only on frames from the prompting timestep forward.
    sub_frame_paths = frames_24fps[prompt_frame_idx:]
    first_frame = _read_frame(sub_frame_paths[0])
    height, width = first_frame.shape[:2]

    # EdgeTAM consumes contiguous mp4 clips.
    temp_video = Path(out_dir) / f"__tmp_edgetam_points_{overlay_name or 'clip'}.mp4" if out_dir else Path("__tmp_edgetam_points.mp4")
    writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"avc1"), clip_fps, (width, height))
    if not writer.isOpened():  # pragma: no cover
        raise RuntimeError(f"Could not open video writer for {temp_video}")
    try:
        for frame_path in sub_frame_paths:
            frame = _read_frame(frame_path)
            writer.write(frame)
    finally:
        writer.release()

    process = psutil.Process()
    reset_gpu_peaks()
    cpu_peak = process.memory_info().rss
    start = time.perf_counter()

    try:
        inference_state = predictor.init_state(str(temp_video))
        points_np = np.array(points, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.int32)
        predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=0, obj_id=1, points=points_np, labels=labels_np)

        sub_masks: List[Optional[np.ndarray]] = [None] * len(sub_frame_paths)
        # Propagate the annotations through the video and collect logits.
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            if mask_logits is None:
                continue
            logits = mask_logits[0] if isinstance(mask_logits, (list, tuple)) else mask_logits
            if logits is None:
                continue
            mask = (logits > 0.0).cpu().numpy().astype(bool)
            if 0 <= frame_idx < len(sub_masks):
                sub_masks[frame_idx] = mask
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] EdgeTAM points inference failed: {exc}")
        sub_masks = [None] * len(sub_frame_paths)

    cuda_sync()
    duration = max(1e-9, time.perf_counter() - start)
    gpu_alloc, gpu_reserved = get_gpu_peaks()
    cpu_peak = max(cpu_peak, process.memory_info().rss)

    masks_seq: List[Optional[np.ndarray]] = [None] * prompt_frame_idx + sub_masks

    overlay_path = None
    if out_dir and overlay_name:
        # Persist overlays only on demand.
        overlay_path = Path(out_dir) / f"{overlay_name}.mp4"
        _record_overlays(frames_24fps, masks_seq, overlay_path, clip_fps)

    try:
        temp_video.unlink(missing_ok=True)
    except OSError:
        pass

    fps = len(sub_frame_paths) / duration

    return {
        "secs": duration,
        "fps": fps,
        "latency_ms": 1000.0 / fps,
        "gpu_peak_alloc": gpu_alloc,
        "gpu_peak_reserved": gpu_reserved,
        "cpu_peak_rss": cpu_peak,
        "masks_seq": masks_seq,
        "overlay": str(overlay_path) if overlay_path else None,
        "frames": len(frames_24fps),
        "H": height,
        "W": width,
        "num_points": len(points),
    }


def run_with_bbox(
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
    _ = (imgsz, device)
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

    predictor = _init_predictor(weight_name)
    _verify_predictor_interfaces(predictor)
    _maybe_compile_edgetam_predictor(
        predictor,
        compile_model=compile_model,
        compile_mode=compile_mode,
        compile_backend=compile_backend,
    )
    frames = [_read_frame(path) for path in frames_24fps]
    sub_frames = frames[prompt_frame_idx:]
    height, width = sub_frames[0].shape[:2]

    # Prepare a contiguous mp4 clip for the predictor.
    temp_video = Path(out_dir) / f"__tmp_edgetam_bbox_{overlay_name or 'clip'}.mp4" if out_dir else Path("__tmp_edgetam_bbox.mp4")
    write_video_mp4(temp_video, sub_frames, clip_fps)

    process = psutil.Process()
    reset_gpu_peaks()
    cpu_peak = process.memory_info().rss
    start = time.perf_counter()

    try:
        inference_state = predictor.init_state(str(temp_video))
        predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=0, obj_id=1, box=np.array(bbox, dtype=np.float32))
        sub_masks: List[Optional[np.ndarray]] = [None] * len(sub_frames)
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            if mask_logits is None:
                continue
            logits = mask_logits[0] if isinstance(mask_logits, (list, tuple)) else mask_logits
            if logits is None:
                continue
            mask = (logits > 0.0).cpu().numpy().astype(bool)
            if 0 <= frame_idx < len(sub_masks):
                sub_masks[frame_idx] = mask
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] EdgeTAM bbox inference failed: {exc}")
        sub_masks = [None] * len(sub_frames)

    cuda_sync()
    duration = max(1e-9, time.perf_counter() - start)
    gpu_alloc, gpu_reserved = get_gpu_peaks()
    cpu_peak = max(cpu_peak, process.memory_info().rss)

    masks_seq: List[Optional[np.ndarray]] = [None] * prompt_frame_idx + sub_masks

    overlay_path = None
    if out_dir and overlay_name:
        # Persist overlays only on demand.
        overlay_path = Path(out_dir) / f"{overlay_name}.mp4"
        _record_overlays([Path(p) for p in frames_24fps], masks_seq, overlay_path, clip_fps)

    try:
        temp_video.unlink(missing_ok=True)
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
        "frames": len(frames_24fps),
        "H": height,
        "W": width,
        "bbox": bbox,
    }


EDGETAM_RUNNERS = {
    "points": run_with_points,
    "bbox": run_with_bbox,
}
