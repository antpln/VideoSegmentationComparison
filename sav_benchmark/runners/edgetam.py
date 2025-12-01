"""Runners for EdgeTAM variants."""

from __future__ import annotations

import time
from pathlib import Path
import shutil
from typing import Dict, List, Optional

import cv2  # type: ignore[import]
import numpy as np
import psutil
import sys
from typing import Tuple
import traceback
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from EdgeTAM.sam2.build_sam import build_sam2_video_predictor  # type: ignore[import]
except Exception:  # pragma: no cover - EdgeTAM optional
    build_sam2_video_predictor = None  # type: ignore

from ..prompts import extract_bbox_from_mask, extract_points_from_mask
from .base import Model
from ..utils import cuda_sync, expand_path, get_gpu_peaks, maybe_compile_module, reset_gpu_peaks
from ..video_ops import overlay_union, write_video_mp4



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
                # Squeeze singleton dims (e.g., (1,1,H,W))
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
                # Handle swapped (W,H) by transposing if it exactly matches the swapped dims
                if mask_np.shape == (width, height):
                    mask_np = mask_np.T
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
            except Exception as per_frame_exc:
                print(f"[ERROR EdgeTAM] per-frame failure at frame {frame_idx}: {per_frame_exc}")
                print(traceback.format_exc())
        # Convert sub_masks dict to list for output (None for missing)
        sub_masks_list = [sub_masks.get(i, None) for i in range(len(sub_frame_paths))]

        print(
            f"[DEBUG EdgeTAM {overlay_name or ''}] mask_logits present in {mask_logits_count} frames, positive entries in {positive_logits_count} frames; stored masks={sum(m is not None for m in sub_masks_list)}"
        )
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] EdgeTAM points inference failed: {exc}")
        print(traceback.format_exc())
        sub_masks_list = [None] * len(sub_frame_paths)
        if inference_start is None:
            inference_start = time.perf_counter()

    cuda_sync()
    if inference_start is None:
        inference_start = time.perf_counter()
    duration = max(1e-9, time.perf_counter() - inference_start)
    setup_secs = inference_start - setup_start
    gpu_alloc, gpu_reserved = get_gpu_peaks()
    cpu_peak = max(cpu_peak, process.memory_info().rss)

    if 'sub_masks_list' not in locals():
        sub_masks_list = [None] * len(sub_frame_paths)
    masks_seq: List[Optional[np.ndarray]] = [None] * prompt_frame_idx + sub_masks_list

    # Debug: log mask statistics
    valid_masks = sum(1 for m in masks_seq if m is not None)
    total_frames = len(masks_seq)
    print(f"[DEBUG EdgeTAM {overlay_name or ''} masks] {valid_masks}/{total_frames} frames have masks")

    overlay_path = None
    if out_dir and overlay_name:
        # Persist overlays only on demand.
        overlay_path = Path(out_dir) / f"{overlay_name}.mp4"
        _record_overlays(frames_24fps, masks_seq, overlay_path, clip_fps)

    # Cleanup temporary JPEG frames directory
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
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
    compile_mode: Optional[str] = "reduce-overhead",
    compile_backend: Optional[str] = None,
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

    predictor = _init_predictor(weight_name, device=device, image_size=imgsz, compile_image_encoder=compile_model)
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

    # Write frames to a temporary JPEG directory instead of MP4 (avoids NVENC/NvMap allocs)
    temp_dir = Path(out_dir) / f"__tmp_edgetam_bbox_{overlay_name or 'clip'}_frames" if out_dir else Path("__tmp_edgetam_bbox_frames")
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame_path in enumerate(sub_frame_paths):
            frame = _read_frame(frame_path)
            cv2.imwrite(str(temp_dir / f"{idx:05d}.jpg"), frame)
    except Exception as e:
        raise RuntimeError(f"Failed to prepare temp JPEG frames at {temp_dir}: {e}")

    inference_start: Optional[float] = None

    try:
        # Keep frames and state on CPU to reduce upfront GPU allocations; load frames lazily
        inference_state = predictor.init_state(
            str(temp_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False,
        )
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

    # Cleanup temporary JPEG frames directory
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
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
            compile_model=compile_model,
            compile_mode=compile_mode,
            compile_backend=compile_backend,
        )


EDGETAM_MODEL = EdgeTAM()
EDGETAM_RUNNERS = EDGETAM_MODEL.register()


def _safe_image_size(image_size: Optional[int]) -> Optional[int]:
    if image_size is None:
        return None
    safe = max(64, int(round(int(image_size) / 64) * 64))
    if safe % 2 != 0:
        safe += 1
    return safe


def _find_existing(paths: List[str]) -> Optional[Path]:
    """Return the first existing path from the provided candidate list."""
    for candidate in paths:
        resolved = Path(expand_path(candidate))
        if resolved.exists():
            return resolved
    return None


def _gpu_debug_snapshot(stage: str) -> None:
    """Print a concise CPU/GPU memory snapshot for debugging."""
    if not DEBUG_LOGS:
        return
    try:
        vm = psutil.virtual_memory()
        _log_debug(
            f"[DEBUG mem {stage}] CPU used={vm.used/1e9:.2f}G avail={vm.available/1e9:.2f}G rss={psutil.Process().memory_info().rss/1e9:.2f}G"
        )
    except Exception:
        pass
    try:
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            dev = torch.cuda.current_device()
            total_free = None
            try:
                free_b, total_b = torch.cuda.mem_get_info(dev)
                total_free = (free_b / 1e9, total_b / 1e9)
            except Exception:
                total_free = None
            name = torch.cuda.get_device_name(dev)
            allocated = torch.cuda.memory_allocated(dev) / 1e9
            reserved = torch.cuda.memory_reserved(dev) / 1e9
            max_alloc = torch.cuda.max_memory_allocated(dev) / 1e9
            max_res = torch.cuda.max_memory_reserved(dev) / 1e9
            free_str = f" free={total_free[0]:.2f}G/{total_free[1]:.2f}G" if total_free else ""
            _log_debug(
                f"[DEBUG cuda {stage}] dev={dev} {name} alloc={allocated:.2f}G resv={reserved:.2f}G max_alloc={max_alloc:.2f}G max_resv={max_res:.2f}G{free_str}"
            )
    except Exception:
        pass


_RETRYABLE_CUDA_PATTERNS = (
    "CUDACachingAllocator",
    "NvMapMem",
    "NVML_SUCCESS",
)

_MAX_CUDA_RETRIES = 3


def _should_retry_cuda_error(exc: Exception) -> bool:
    message = str(exc)
    return any(token in message for token in _RETRYABLE_CUDA_PATTERNS)


def _init_predictor(weight_name: str, *, device: str = "cuda", image_size: Optional[int] = None, compile_image_encoder: bool = False) -> object:
    """Build an EdgeTAM video predictor using discovered config/weights.

    Honors the optional image_size to reduce memory footprint in a device-agnostic way.
    """
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

    hydra_overrides_extra = []
    if image_size is not None:
        # EdgeTAM config assumes image_size divisible by 64 (e.g., 1024 -> grids 64,16).
        safe_imgsz = _safe_image_size(image_size)
        if safe_imgsz is None:
            safe_imgsz = image_size
        if safe_imgsz != int(image_size):
            print(f"[WARN EdgeTAM] Adjusting image_size from {image_size} to nearest multiple-of-64: {safe_imgsz}")

        # Override image_size and corresponding RoPE attention sizes
        # feat_sizes for self-attention: stride-32 features = image_size / 32
        # q_sizes for cross-attention: stride-16 features = image_size / 16
        # k_sizes for cross-attention: stride-64 features = image_size / 64
        feat_size = safe_imgsz // 32
        q_size = safe_imgsz // 16
        k_size = safe_imgsz // 64
        
        hydra_overrides_extra.extend([
            f"++model.image_size={safe_imgsz}",
            f"++model.memory_attention.layer.self_attention.feat_sizes=[{feat_size},{feat_size}]",
            f"++model.memory_attention.layer.cross_attention.q_sizes=[{q_size},{q_size}]",
            f"++model.memory_attention.layer.cross_attention.k_sizes=[{k_size},{k_size}]",
        ])
    
    # Optionally compile image encoder for speed (adds compile overhead on first run)
    if compile_image_encoder:
        hydra_overrides_extra.append("++model.compile_image_encoder=true")

    return build_sam2_video_predictor(
        config_file=config_name,
        ckpt_path=str(checkpoint_path),
        device=device,
        hydra_overrides_extra=hydra_overrides_extra,
    )


def _maybe_compile_edgetam_predictor(
    predictor,
    *,
    compile_model: bool,
    compile_mode: Optional[str],
    compile_backend: Optional[str],
) -> None:
    """Note: EdgeTAM does not support full model compilation via torch.compile.
    
    Only the image encoder can be compiled via compile_image_encoder=True in the config.
    Attempting to compile the full predictor (which has @torch.inference_mode() decorators)
    will likely fail or provide no benefit.
    
    SAM 2 has vos_optimized=True flag for full compilation, but EdgeTAM doesn't support this.
    """
    if not compile_model:
        return
    
    print("[WARN EdgeTAM] Full model compilation via --compile_models is not supported by EdgeTAM.")
    print("[WARN EdgeTAM] Only image encoder compilation is supported (already enabled if --compile_models is set).")
    print("[WARN EdgeTAM] Methods decorated with @torch.inference_mode() cannot be compiled.")


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
    precision=None,
    max_clip_frames: Optional[int] = None,
    compile_model: bool = False,
    compile_mode: Optional[str] = "reduce-overhead",
    compile_backend: Optional[str] = None,
    _attempt: int = 1,
) -> Dict[str, object]:
    _ = ()
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
            "infer_H": prompt_mask.shape[0],
            "infer_W": prompt_mask.shape[1],
            "num_points": 0,
        }

    total_frames = len(frames_24fps)
    if prompt_frame_idx >= total_frames:
        raise IndexError(f"Prompt index {prompt_frame_idx} is out of range for {total_frames} frames")

    clip_end = total_frames
    if max_clip_frames is not None and max_clip_frames > 0:
        clip_end = min(total_frames, prompt_frame_idx + max_clip_frames)

    sub_frame_paths = frames_24fps[prompt_frame_idx:clip_end]
    sub_frame_count = len(sub_frame_paths)
    if sub_frame_count == 0:
        raise IndexError(f"Prompt index {prompt_frame_idx} is out of range for {total_frames} frames")

    safe_imgsz = _safe_image_size(imgsz)
    inference_hw = None
    if safe_imgsz is not None:
        inference_hw = (safe_imgsz, safe_imgsz)

    clipped_paths = frames_24fps[:clip_end]

    full_stream = prepare_frame_stream(
        clipped_paths,
        imgsz=imgsz,
        target_hw=inference_hw,
        force_square=True,
    )
    prompt_stream = prepare_frame_stream(
        clipped_paths,
        start_idx=prompt_frame_idx,
        imgsz=imgsz,
        target_hw=inference_hw,
        force_square=True,
    )
    height, width = prompt_stream.target_hw
    orig_height, orig_width = prompt_stream.original_hw
    scale_x, scale_y = prompt_stream.scale_xy
    pad_x, pad_y = prompt_stream.pad_offsets()
    content_h, content_w = prompt_stream.content_shape()
    content_h, content_w = prompt_stream.content_shape()
    precision_scope = precision if precision is not None else (lambda: nullcontext())

    # Fairness alignment with SAM2: start GPU peak tracking BEFORE predictor build
    # and separate setup (predictor build + clip write + init_state + prompt seeding)
    # from pure propagation inference time.
    process = psutil.Process()
    reset_gpu_peaks()
    cpu_peak = process.memory_info().rss
    setup_start = time.perf_counter()

    predictor = _init_predictor(weight_name, device=device, image_size=imgsz, compile_image_encoder=compile_model)
    _verify_predictor_interfaces(predictor)
    _maybe_compile_edgetam_predictor(
        predictor,
        compile_model=compile_model,
        compile_mode=compile_mode,
        compile_backend=compile_backend,
    )
    model_imgsz = getattr(predictor, "image_size", None)
    _log_debug(f"[DEBUG EdgeTAM] predictor constructed & compiled (image_size={model_imgsz}, device={device})")
    _gpu_debug_snapshot("post-predictor-build")

    # Write frames to a temporary JPEG directory instead of MP4 (avoids NVENC/NvMap allocs)
    temp_dir = Path(out_dir) / f"__tmp_edgetam_points_{overlay_name or 'clip'}_frames" if out_dir else Path("__tmp_edgetam_points_frames")
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(prompt_stream.generator()):
            cv2.imwrite(str(temp_dir / f"{idx:05d}.jpg"), frame)
        _log_debug(
            f"[DEBUG EdgeTAM] prepared temp frames at {temp_dir} with {sub_frame_count} images (orig={orig_height}x{orig_width} -> infer={height}x{width})"
        )
        _gpu_debug_snapshot("after-temp-frames")
    except Exception as e:
        raise RuntimeError(f"Failed to prepare temp JPEG frames at {temp_dir}: {e}")

    inference_start: Optional[float] = None
    sub_masks_list: List[Optional[np.ndarray]]

    # Separate try/except to pinpoint init_state failures
    try:
        # Keep frames and state on CPU to reduce upfront GPU allocations; load frames lazily
        _log_debug("[DEBUG EdgeTAM] entering init_state(...) with CPU offloading & async frames")
        _log_debug(f"[DEBUG EdgeTAM] predictor.image_size = {predictor.image_size}, imgsz arg = {imgsz}")
        _gpu_debug_snapshot("pre-init_state")
        with precision_scope():
            inference_state = predictor.init_state(
                str(temp_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                # Match example notebook behavior to avoid races and ensure deterministic shapes
                async_loading_frames=False,
            )
        _log_debug("[DEBUG EdgeTAM] init_state completed")
        _gpu_debug_snapshot("post-init_state")
    except Exception as e:
        print(f"[ERROR EdgeTAM] init_state failed: {e}")
        if DEBUG_LOGS:
            # Attempt a debug load to inspect frame tensor shapes
            try:
                from EdgeTAM.sam2.utils.misc import load_video_frames as _sam2_load_frames  # type: ignore

                imgs, vH, vW = _sam2_load_frames(
                    video_path=str(temp_dir),
                    image_size=model_imgsz or imgsz,
                    offload_video_to_cpu=True,
                    async_loading_frames=True,
                    compute_device=None if (torch is None or not torch.cuda.is_available()) else torch.device(device),
                )
                # Try to inspect first image shape
                try:
                    first = imgs[0]
                    _log_debug(
                        f"[DEBUG EdgeTAM] debug-load: first_frame_shape={getattr(first, 'shape', None)} video(HxW)={vH}x{vW}"
                    )
                except Exception:
                    _log_debug(f"[DEBUG EdgeTAM] debug-load: images_type={type(imgs)} video(HxW)={vH}x{vW}")
            except Exception as e2:
                _log_debug(f"[DEBUG EdgeTAM] secondary debug load failed: {e2}")
        raise

    try:
        points_np = np.array([[p[0] * scale_x + pad_x, p[1] * scale_y + pad_y] for p in points], dtype=np.float32)
        points_np[:, 0] = np.clip(points_np[:, 0], 0, max(0, width - 1))
        points_np[:, 1] = np.clip(points_np[:, 1], 0, max(0, height - 1))
        labels_np = np.array(labels, dtype=np.int32)
        # Keep every propagated mask so evaluation has the full sequence.
        sub_masks: Dict[int, Optional[np.ndarray]] = {}
        mask_logits_count = 0
        positive_logits_count = 0
        with precision_scope():
            predictor.add_new_points_or_box(
                inference_state=inference_state, frame_idx=0, obj_id=1, points=points_np, labels=labels_np
            )
            _log_debug("[DEBUG EdgeTAM] prompt seeded (points)")
            _gpu_debug_snapshot("post-add-prompt")
            # Start pure inference timing AFTER seeding (parity with SAM2 logic)
            inference_start = time.perf_counter()

            _log_debug("[DEBUG EdgeTAM] starting propagate_in_video()")
            _gpu_debug_snapshot("pre-propagate")
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
                try:
                    if mask_logits is None or 1 not in obj_ids:
                        continue
                    if frame_idx == 0:
                        _log_debug("[DEBUG EdgeTAM] first propagate yield received")
                        _gpu_debug_snapshot("first-yield")
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
                        _log_debug(
                            f"[DEBUG EdgeTAM logits] frame={frame_idx} shape={logits_np.shape} min={float(logits_np.min()):.4f} max={float(logits_np.max()):.4f}"
                        )
                    if np.count_nonzero(logits_np) > 0:
                        positive_logits_count += 1
                    tmp = logits_np
                    thr = 0.5 if (tmp.min() >= 0.0 and tmp.max() <= 1.0) else 0.0
                    mask_np = tmp > thr
                    # Squeeze singleton dims (e.g., (1,1,H,W))
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
                    # Handle swapped (W,H) by transposing if it exactly matches the swapped dims
                    if mask_np.shape == (width, height):
                        mask_np = mask_np.T
                        if mask_np.shape != (height, width):
                            mask_np = cv2.resize(
                                mask_np.astype(np.uint8),
                                (width, height),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                    if 0 <= frame_idx < sub_frame_count:
                        sub_masks[frame_idx] = mask_np
                except Exception as per_frame_exc:
                    print(f"[ERROR EdgeTAM] per-frame failure at frame {frame_idx}: {per_frame_exc}")
                    print(traceback.format_exc())
        # Convert sub_masks dict to list for output (None for missing)
        sub_masks_list = [sub_masks.get(i, None) for i in range(sub_frame_count)]

        _log_debug(
            f"[DEBUG EdgeTAM {overlay_name or ''}] mask_logits present in {mask_logits_count} frames, positive entries in {positive_logits_count} frames; stored masks={sum(m is not None for m in sub_masks_list)}"
        )
    except Exception as exc:  # pragma: no cover
        if _should_retry_cuda_error(exc) and _attempt < _MAX_CUDA_RETRIES:
            print(
                f"[WARN EdgeTAM] points inference attempt {_attempt} failed with allocator error: {exc}. Retrying."
            )
            cleanup_after_run()
            return _run_points(
                frames_24fps=frames_24fps,
                prompt_frame_idx=prompt_frame_idx,
                prompt_mask=prompt_mask,
                imgsz=imgsz,
                weight_name=weight_name,
                device=device,
                out_dir=out_dir,
                overlay_name=overlay_name,
                clip_fps=clip_fps,
                precision=precision,
                max_clip_frames=max_clip_frames,
                compile_model=compile_model,
                compile_mode=compile_mode,
                compile_backend=compile_backend,
                _attempt=_attempt + 1,
            )
        print(f"[ERROR] EdgeTAM points inference failed (attempt {_attempt}): {exc}")
        print(traceback.format_exc())
        sub_masks_list = [None] * sub_frame_count
        if inference_start is None:
            inference_start = time.perf_counter()

    cuda_sync()
    if inference_start is None:
        inference_start = time.perf_counter()
    duration = max(1e-9, time.perf_counter() - inference_start)
    setup_secs = inference_start - setup_start
    gpu_alloc, gpu_reserved = get_gpu_peaks()
    cpu_peak = max(cpu_peak, process.memory_info().rss)

    if 'sub_masks_list' not in locals():
        sub_masks_list = [None] * sub_frame_count
    masks_seq: List[Optional[np.ndarray]] = [None] * prompt_frame_idx + sub_masks_list

    # Debug: log mask statistics
    valid_masks = sum(1 for m in masks_seq if m is not None)
    total_masks = len(masks_seq)
    _log_debug(f"[DEBUG EdgeTAM {overlay_name or ''} masks] {valid_masks}/{total_masks} frames have masks")

    overlay_path = None
    if out_dir and overlay_name:
        # Persist overlays only on demand.
        overlay_path = Path(out_dir) / f"{overlay_name}.mp4"
        overlay_video_frames(
            clipped_paths,
            masks_seq[:clip_end],
            output_path=overlay_path,
            fps=clip_fps,
            target_hw=full_stream.target_hw,
        )

    # Cleanup temporary JPEG frames directory
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
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
        "H": orig_height,
        "W": orig_width,
        "infer_H": height,
        "infer_W": width,
        "processed_end_frame": clip_end,
        "num_points": len(points),  # will be 1 under the single-point policy
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
    _attempt: int = 1,
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
            "infer_H": 0,
            "infer_W": 0,
        }

    # Align fairness: track GPU peaks from predictor build; separate setup vs inference.
    process = psutil.Process()
    reset_gpu_peaks()
    cpu_peak = process.memory_info().rss
    setup_start = time.perf_counter()

    predictor = _init_predictor(weight_name, device=device, image_size=imgsz, compile_image_encoder=compile_model)
    _verify_predictor_interfaces(predictor)
    _maybe_compile_edgetam_predictor(
        predictor,
        compile_model=compile_model,
        compile_mode=compile_mode,
        compile_backend=compile_backend,
    )

    total_frames = len(frames_24fps)
    if prompt_frame_idx >= total_frames:
        raise IndexError(f"Prompt index {prompt_frame_idx} is out of range for {total_frames} frames")

    clip_end = total_frames
    if max_clip_frames is not None and max_clip_frames > 0:
        clip_end = min(total_frames, prompt_frame_idx + max_clip_frames)

    sub_frame_paths = frames_24fps[prompt_frame_idx:clip_end]
    sub_frame_count = len(sub_frame_paths)
    if sub_frame_count == 0:
        raise IndexError(f"Prompt index {prompt_frame_idx} is out of range for {total_frames} frames")

    safe_imgsz = _safe_image_size(imgsz)
    inference_hw = None
    if safe_imgsz is not None:
        inference_hw = (safe_imgsz, safe_imgsz)

    clipped_paths = frames_24fps[:clip_end]

    full_stream = prepare_frame_stream(
        clipped_paths,
        imgsz=imgsz,
        target_hw=inference_hw,
        force_square=True,
    )
    prompt_stream = prepare_frame_stream(
        clipped_paths,
        start_idx=prompt_frame_idx,
        imgsz=imgsz,
        target_hw=inference_hw,
        force_square=True,
    )
    height, width = prompt_stream.target_hw
    orig_height, orig_width = prompt_stream.original_hw
    scale_x, scale_y = prompt_stream.scale_xy
    pad_x, pad_y = prompt_stream.pad_offsets()
    precision_scope = precision if precision is not None else (lambda: nullcontext())

    # Write frames to a temporary JPEG directory instead of MP4 (avoids NVENC/NvMap allocs)
    temp_dir = Path(out_dir) / f"__tmp_edgetam_bbox_{overlay_name or 'clip'}_frames" if out_dir else Path("__tmp_edgetam_bbox_frames")
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(prompt_stream.generator()):
            cv2.imwrite(str(temp_dir / f"{idx:05d}.jpg"), frame)
    except Exception as e:
        raise RuntimeError(f"Failed to prepare temp JPEG frames at {temp_dir}: {e}")

    inference_start: Optional[float] = None
    bbox_np = np.array(bbox, dtype=np.float32)

    try:
        # Keep frames and state on CPU to reduce upfront GPU allocations; load frames lazily
        with precision_scope():
            inference_state = predictor.init_state(
                str(temp_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=False,
            )
        x1, y1, x2, y2 = bbox
        scaled_x = sorted([x1 * scale_x + pad_x, x2 * scale_x + pad_x])
        scaled_y = sorted([y1 * scale_y + pad_y, y2 * scale_y + pad_y])
        bbox_np = np.array(
            [
                np.clip(scaled_x[0], 0, max(0, width - 1)),
                np.clip(scaled_y[0], 0, max(0, height - 1)),
                np.clip(scaled_x[1], 0, max(0, width - 1)),
                np.clip(scaled_y[1], 0, max(0, height - 1)),
            ],
            dtype=np.float32,
        )
        # Capture every processed frame so evaluation can score the full clip.
        sub_masks: List[Optional[np.ndarray]] = [None] * sub_frame_count
        with precision_scope():
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                box=bbox_np,
            )
            inference_start = time.perf_counter()
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
                if 0 <= frame_idx < sub_frame_count:
                    sub_masks[frame_idx] = mask_np
    except Exception as exc:  # pragma: no cover
        if _should_retry_cuda_error(exc) and _attempt < _MAX_CUDA_RETRIES:
            print(
                f"[WARN EdgeTAM] bbox inference attempt {_attempt} failed with allocator error: {exc}. Retrying."
            )
            cleanup_after_run()
            return _run_bbox(
                frames_24fps=frames_24fps,
                prompt_frame_idx=prompt_frame_idx,
                prompt_mask=prompt_mask,
                imgsz=imgsz,
                weight_name=weight_name,
                device=device,
                out_dir=out_dir,
                overlay_name=overlay_name,
                clip_fps=clip_fps,
                precision=precision,
                max_clip_frames=max_clip_frames,
                compile_model=compile_model,
                compile_mode=compile_mode,
                compile_backend=compile_backend,
                _attempt=_attempt + 1,
            )
        print(f"[ERROR] EdgeTAM bbox inference failed (attempt {_attempt}): {exc}")
        print(traceback.format_exc())
        sub_masks = [None] * sub_frame_count
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
    total_masks = len(masks_seq)
    _log_debug(f"[DEBUG EdgeTAM {overlay_name or ''} masks] {valid_masks}/{total_masks} frames have masks")

    overlay_path = None
    if out_dir and overlay_name:
        # Persist overlays only on demand.
        overlay_path = Path(out_dir) / f"{overlay_name}.mp4"
        overlay_video_frames(
            clipped_paths,
            masks_seq[:clip_end],
            output_path=overlay_path,
            fps=clip_fps,
            target_hw=full_stream.target_hw,
        )

    # Cleanup temporary JPEG frames directory
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    except OSError:
        pass

    fps = sub_frame_count / duration if sub_frame_count else 0.0
    latency_ms = 1000.0 / fps if fps > 0 else None
    bbox_list = bbox_np.astype(float).tolist()

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
        "H": orig_height,
        "W": orig_width,
        "infer_H": height,
        "infer_W": width,
        "bbox": bbox,
        "bbox_infer": bbox_list,
        "processed_end_frame": clip_end,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "content_H": content_h,
        "content_W": content_w,
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
            num_points=num_points,
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


EDGETAM_MODEL = EdgeTAM()
EDGETAM_RUNNERS = EDGETAM_MODEL.register()
