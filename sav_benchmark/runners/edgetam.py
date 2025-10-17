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
from .base import Model
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
        # Handle potential transposed (H,W) mismatch (e.g., logits came as (W,H)).
        if mask_arr.shape[0] == frame.shape[1] and mask_arr.shape[1] == frame.shape[0]:
            # Transpose to match frame (H,W)
            mask_arr = mask_arr.T
        if mask_arr.shape != frame.shape[:2]:
            mask_arr = cv2.resize(
                mask_arr.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        overlays.append(overlay_union(frame, mask_arr, alpha=0.5))
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
    compile_model: bool = False,
    compile_mode: str | None = "reduce-overhead",
    compile_backend: str | None = None,
    max_frames_in_mem: int = 600,  # NEW: limit number of frames in memory
) -> Dict[str, object]:
    _ = (imgsz, device)
    # Translate the mask into positive point prompts.
    # Use a single positive point for fair comparison with SAM2 point prompting.
    points, labels = extract_points_from_mask(prompt_mask, 1)
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

    # Fairness alignment with SAM2: start GPU peak tracking BEFORE predictor build
    # and separate setup (predictor build + clip write + init_state + prompt seeding)
    # from pure propagation inference time.
    process = psutil.Process()
    reset_gpu_peaks()
    cpu_peak = process.memory_info().rss
    setup_start = time.perf_counter()

    predictor = _init_predictor(weight_name)
    _verify_predictor_interfaces(predictor)
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

    inference_start: float | None = None

    try:
        inference_state = predictor.init_state(str(temp_video))
        points_np = np.array(points, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.int32)
        predictor.add_new_points_or_box(
            inference_state=inference_state, frame_idx=0, obj_id=1, points=points_np, labels=labels_np
        )
        # Start pure inference timing AFTER seeding (parity with SAM2 logic)
        inference_start = time.perf_counter()

        # Initialize counters for debugging
        mask_logits_count = 0
        positive_logits_count = 0
        
        # Replace sub_masks with a dict for sliding window
        sub_masks: Dict[int, Optional[np.ndarray]] = {}
        mask_indices: List[int] = []
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            if mask_logits is None or 1 not in obj_ids:
                continue
            mask_logits_count += 1
            if isinstance(mask_logits, (list, tuple)):
                idx = obj_ids.index(1)
                logits = mask_logits[idx]
            else:
                logits = mask_logits
            if logits is None:
                continue
            if hasattr(logits, "numel") and logits.numel() == 0:  # type: ignore[attr-defined]
                continue
            if hasattr(logits, "detach"):
                logits_np = logits.detach().cpu().numpy()
            elif hasattr(logits, "cpu"):
                logits_np = logits.cpu().numpy()
            else:
                logits_np = np.asarray(logits)
            if logits_np.size == 0:
                continue
            if mask_logits_count <= 5:
                print(
                    f"[DEBUG EdgeTAM logits] frame={frame_idx} shape={logits_np.shape} min={float(logits_np.min()):.4f} max={float(logits_np.max()):.4f}"
                )
            if np.count_nonzero(logits_np) > 0:
                positive_logits_count += 1
            tmp = logits_np
            thr = 0.5 if (tmp.min() >= 0.0 and tmp.max() <= 1.0) else 0.0
            mask_np = tmp > thr
            while mask_np.ndim > 2:
                if mask_np.shape[0] == 1:
                    mask_np = mask_np[0]
                else:
                    mask_np = np.any(mask_np, axis=0)
            if mask_np.ndim != 2:
                continue
            mask_np = mask_np.astype(bool)
            if mask_np.size == 0:
                continue
            if mask_np.shape != (height, width):
                mask_np = cv2.resize(
                    mask_np.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if 0 <= frame_idx < len(sub_frame_paths):
                sub_masks[frame_idx] = mask_np
                mask_indices.append(frame_idx)
                # Remove oldest if exceeding max_frames_in_mem
                if len(mask_indices) > max_frames_in_mem:
                    oldest = mask_indices.pop(0)
                    del sub_masks[oldest]
        # Convert sub_masks dict to list for output (None for missing)
        sub_masks_list = [sub_masks.get(i, None) for i in range(len(sub_frame_paths))]
        masks_seq: List[Optional[np.ndarray]] = [None] * prompt_frame_idx + sub_masks_list

        print(
            f"[DEBUG EdgeTAM {overlay_name or ''}] mask_logits present in {mask_logits_count} frames, positive entries in {positive_logits_count} frames; stored masks={sum(m is not None for m in sub_masks_list)}"
        )
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] EdgeTAM points inference failed: {exc}")
        sub_masks = [None] * len(sub_frame_paths)
        if inference_start is None:
            inference_start = time.perf_counter()

    cuda_sync()
    if inference_start is None:
        inference_start = time.perf_counter()
    duration = max(1e-9, time.perf_counter() - inference_start)
    setup_secs = inference_start - setup_start
    gpu_alloc, gpu_reserved = get_gpu_peaks()
    cpu_peak = max(cpu_peak, process.memory_info().rss)

    masks_seq: List[Optional[np.ndarray]] = [None] * prompt_frame_idx + sub_masks

    # Debug: log mask statistics
    valid_masks = sum(1 for m in masks_seq if m is not None)
    total_frames = len(masks_seq)
    print(f"[DEBUG EdgeTAM {overlay_name or ''} masks] {valid_masks}/{total_frames} frames have masks")

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
    "num_points": len(points),  # will be 1 under the single-point policy
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
    max_frames_in_mem: int = 3,  # NEW: limit number of frames in memory
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

    # Align fairness: track GPU peaks from predictor build; separate setup vs inference.
    process = psutil.Process()
    reset_gpu_peaks()
    cpu_peak = process.memory_info().rss
    setup_start = time.perf_counter()

    predictor = _init_predictor(weight_name)
    _verify_predictor_interfaces(predictor)
    _maybe_compile_edgetam_predictor(
        predictor,
        compile_model=compile_model,
        compile_mode=compile_mode,
        compile_backend=compile_backend,
    )

    sub_frame_paths = frames_24fps[prompt_frame_idx:]
    first_frame = _read_frame(sub_frame_paths[0])
    height, width = first_frame.shape[:2]

    temp_video = Path(out_dir) / f"__tmp_edgetam_bbox_{overlay_name or 'clip'}.mp4" if out_dir else Path("__tmp_edgetam_bbox.mp4")
    writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"avc1"), clip_fps, (width, height))
    if not writer.isOpened():  # pragma: no cover
        raise RuntimeError(f"Could not open video writer for {temp_video}")
    try:
        for frame_path in sub_frame_paths:
            frame = _read_frame(frame_path)
            writer.write(frame)
    finally:
        writer.release()

    inference_start: float | None = None

    try:
        inference_state = predictor.init_state(str(temp_video))
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            box=np.array(bbox, dtype=np.float32),
        )
        inference_start = time.perf_counter()
        sub_masks: List[Optional[np.ndarray]] = [None] * len(sub_frame_paths)
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            if mask_logits is None or 1 not in obj_ids:
                continue
            if isinstance(mask_logits, (list, tuple)):
                idx = obj_ids.index(1)
                logits = mask_logits[idx]
            else:
                logits = mask_logits
            if logits is None:
                continue
            if hasattr(logits, "numel") and logits.numel() == 0:  # type: ignore[attr-defined]
                continue
            if hasattr(logits, "detach"):
                logits_np = logits.detach().cpu().numpy()
            elif hasattr(logits, "cpu"):
                logits_np = logits.cpu().numpy()
            else:
                logits_np = np.asarray(logits)
            if logits_np.size == 0:
                continue
            mask_np = logits_np > 0.0
            if mask_np.ndim > 2:
                mask_np = np.any(mask_np, axis=0)
            if mask_np.ndim != 2:
                continue
            mask_np = mask_np.astype(bool)
            if mask_np.size == 0:
                continue
            if mask_np.shape != (height, width):
                mask_np = cv2.resize(
                    mask_np.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if 0 <= frame_idx < len(sub_masks):
                sub_masks[frame_idx] = mask_np
                # Remove oldest if exceeding max_frames_in_mem
                if len(sub_masks) > max_frames_in_mem:
                    oldest = next(iter(sub_masks))  # Get the first added (oldest) frame
                    del sub_masks[oldest]
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] EdgeTAM bbox inference failed: {exc}")
        sub_masks = [None] * len(sub_frame_paths)
        if inference_start is None:
            inference_start = time.perf_counter()

    cuda_sync()
    if inference_start is None:
        inference_start = time.perf_counter()
    duration = max(1e-9, time.perf_counter() - inference_start)
    setup_secs = inference_start - setup_start
    gpu_alloc, gpu_reserved = get_gpu_peaks()
    cpu_peak = max(cpu_peak, process.memory_info().rss)

    masks_seq: List[Optional[np.ndarray]] = [None] * prompt_frame_idx + sub_masks

    # Debug: log mask statistics
    valid_masks = sum(1 for m in masks_seq if m is not None)
    total_frames = len(masks_seq)
    print(f"[DEBUG EdgeTAM {overlay_name or ''} masks] {valid_masks}/{total_frames} frames have masks")

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
        "bbox": bbox,
        "setup_ms": round(setup_secs * 1000.0, 2),
    }


class EdgeTAM(Model):
    """Concrete runner that exposes EdgeTAM trackers via the standard prompts."""

    def __init__(self) -> None:
        super().__init__("edgetam")

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
        num_points: int = 5,
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
            num_points=num_points,
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


EDGETAM_MODEL = EdgeTAM()
EDGETAM_RUNNERS = EDGETAM_MODEL.register()
