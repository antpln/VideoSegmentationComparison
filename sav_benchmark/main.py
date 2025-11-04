"""Command-line entry point for the SA-V benchmark."""

from __future__ import annotations

import argparse
import csv
import sys
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import cv2  # type: ignore[import]

from .data_io import ensure_dir, list_annotated_indices_6fps, list_frames_24fps, load_mask_png, read_video_ids
from .metrics import j_and_proxy_jf
from .prompts import mask_centroid
from .runners import edgetam  # noqa: F401  (ensure registration side-effects)
from .runners import sam2  # noqa: F401
from .runners.registry import get_runner
from .synthetic import create_synthetic_test_data
from .utils import build_precision_context, cleanup_after_run, device_str, reset_gpu_peaks, to_mib

try:
    import torch  # type: ignore[import]
except Exception:  # pragma: no cover
    torch = None  # type: ignore


SAM2_WEIGHTS = {
    "sam2_tiny": "sam2.1_t.pt",
    "sam2_small": "sam2.1_s.pt",
    "sam2_base": "sam2.1_b.pt",
    "sam2_large": "sam2.1_l.pt",
}

EDGETAM_WEIGHTS = {
    "edgetam": "edgetam.pt",
}

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SA-V benchmark")
    parser.add_argument("--split_dir", type=str, default=None, help="Dataset root (sav_val or sav_test)")
    parser.add_argument("--test_mode", action="store_true", help="Use synthetic data for quick validation")
    parser.add_argument("--split_name", type=str, default="sav_val", help="Name of the split file to read")
    parser.add_argument(
        "--models",
        type=str,
        default="sam2_base_points,sam2_base_bbox,edgetam_points,edgetam_bbox",
        help="Comma-separated list of model_prompt combinations",
    )
    parser.add_argument("--weights_dir", type=str, default=".", help="Directory containing weight files")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--limit_videos", type=int, default=0, help="Limit number of videos (0 = all)")
    parser.add_argument("--limit_objects", type=int, default=0, help="Limit number of objects per video (0 = all)")
    parser.add_argument("--save_overlays", type=int, default=0, help="Write overlay mp4s when set to 1")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for overlays and summary")
    parser.add_argument("--compile_models", action="store_true", help="Compile models with torch.compile before inference")
    parser.add_argument("--compile_backend", type=str, default=None, help="Optional backend to use for torch.compile")
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        help="torch.compile mode (e.g., default, reduce-overhead, max-autotune)",
    )
    parser.add_argument(
        "--max_clip_frames",
        type=int,
        default=0,
        help="Process at most this many frames after the prompt (0 = full video)",
    )
    parser.add_argument("--shuffle_videos", action="store_true", help="Shuffle video order before limiting / processing")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling (implies --shuffle_videos)")
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Numerical precision for model execution",
    )
    parser.add_argument(
        "--disable_cudnn",
        action="store_true",
        help="Disable cuDNN to reduce conv workspace memory (slower but can avoid OOM)",
    )
    return parser.parse_args(args=argv)


def _unique_existing(paths: Iterable[Path]) -> List[Path]:
    seen: Set[Path] = set()
    result: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(path)
    return result


def _candidate_weight_paths(source: str, weights_dir: Path) -> List[Path]:
    source_path = Path(source)

    if source_path.is_absolute():
        return _unique_existing([source_path])

    search_roots = []
    for root in (weights_dir, REPO_ROOT, REPO_ROOT / "EdgeTAM"):
        if root.exists() and root.is_dir():
            search_roots.append(root)

    candidates: List[Path] = []

    # Direct joins (respecting any sub-directories in the relative path)
    for base in search_roots:
        candidates.append(base / source_path)

    # Fallback: search by filename under each root (similar to `find | grep`)
    filename = source_path.name
    visited: Set[Path] = set()
    for base in search_roots:
        try:
            for match in base.rglob(filename):
                if match not in visited:
                    candidates.append(match)
                    visited.add(match)
        except (OSError, PermissionError):
            continue
    found = _unique_existing(candidates)
    for path in found:
        print(f"  [INFO] Found weights: {path}")
    return found


def _resolve_models(args: argparse.Namespace, model_tags: List[str], weights_dir: Path) -> List[Tuple[str, str]]:
    """Map model tags to weight paths if available on disk."""
    model_map: Dict[str, str] = {}

    for model_name, weight in SAM2_WEIGHTS.items():
        for prompt in ("points", "bbox"):
            model_map[f"{model_name}_{prompt}"] = weight

    for model_name, weight in EDGETAM_WEIGHTS.items():
        for prompt in ("points", "bbox"):
            model_map[f"{model_name}_{prompt}"] = weight

    resolved: List[Tuple[str, str]] = []
    for tag in model_tags:
        weight_file = model_map.get(tag)
        if weight_file:
            candidates = _candidate_weight_paths(weight_file, weights_dir)
            if not candidates:
                print(
                    f"[WARN] Could not locate weights for {tag}: searched for '{weight_file}' under {weights_dir}"
                )
                continue
            resolved.append((tag, str(candidates[0])))
            continue

        print(f"[WARN] Unknown model tag: {tag} (skipping)")
    return resolved


def _select_runner(tag: str):
    if "_" not in tag:
        raise ValueError(f"Malformed model tag: {tag}")
    parts = tag.split("_")
    prompt_type = parts[-1]
    model_root = "_".join(parts[:-1])
    # Derive family name (strip size suffixes for sam2 family)
    if model_root.startswith("sam2"):
        family = "sam2"
    elif model_root == "edgetam":
        family = "edgetam"
    else:
        family = model_root
    runner = get_runner(family, prompt_type)
    if runner is None:
        raise ValueError(f"Unsupported model/prompt combination: {tag}")
    return runner


def _prepare_dataset(args: argparse.Namespace) -> Tuple[Path, Path]:
    """Resolve dataset/input directories and optionally fabricate synthetic data."""
    if args.test_mode:
        split_dir = Path("./test_synthetic_data")
        out_dir = Path(args.out_dir) if args.out_dir else Path("./test_outputs")
        create_synthetic_test_data(split_dir)
        args.limit_videos = 1
        args.limit_objects = 1
        args.imgsz = 256
    else:
        if not args.split_dir:
            raise SystemExit("--split_dir is required unless --test_mode is used")
        split_dir = Path(args.split_dir)
        if not (split_dir / "JPEGImages_24fps").exists():
            raise FileNotFoundError("Missing JPEGImages_24fps directory")
        if not (split_dir / "Annotations_6fps").exists():
            raise FileNotFoundError("Missing Annotations_6fps directory")
        out_dir = Path(args.out_dir) if args.out_dir else split_dir / "benchmark_outputs"
    ensure_dir(out_dir)
    return split_dir, out_dir


def _configure_torch(args: argparse.Namespace) -> None:
    """Apply conservative CUDA defaults when GPUs are available."""
    if torch is None or not torch.cuda.is_available():
        print("Using CPU")
        return
    print("CUDA available:", True, "Device:", torch.cuda.get_device_name(0))
    # Optional: disable cuDNN to reduce conv workspace peaks (slower but can avoid OOM)
    try:
        if args.disable_cudnn:
            torch.backends.cudnn.enabled = False
            print("cuDNN: disabled")
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:  # pragma: no cover
        pass


def _restore_mask_to_original(mask: Optional[np.ndarray], result: Dict[str, object]) -> Optional[np.ndarray]:
    if mask is None:
        return None

    mask_np = np.asarray(mask).astype(bool)
    orig_h = int(result.get("H") or mask_np.shape[0])
    orig_w = int(result.get("W") or mask_np.shape[1])

    content_h = int(result.get("content_H") or mask_np.shape[0])
    content_w = int(result.get("content_W") or mask_np.shape[1])
    pad_x = int(result.get("pad_x") or 0)
    pad_y = int(result.get("pad_y") or 0)

    y0 = max(0, min(pad_y, mask_np.shape[0]))
    y1 = max(0, min(pad_y + content_h, mask_np.shape[0]))
    x0 = max(0, min(pad_x, mask_np.shape[1]))
    x1 = max(0, min(pad_x + content_w, mask_np.shape[1]))

    if y1 > y0 and x1 > x0:
        cropped = mask_np[y0:y1, x0:x1]
    else:
        cropped = mask_np

    resized = cv2.resize(
        cropped.astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def run(args: argparse.Namespace) -> Path:
    split_dir, out_dir = _prepare_dataset(args)
    weights_dir = Path(args.weights_dir)
    model_tags = [tag.strip() for tag in args.models.split(",") if tag.strip()]
    models = _resolve_models(args, model_tags, weights_dir)
    if not models:
        raise SystemExit("No valid models selected")

    _configure_torch(args)
    device = device_str()
    precision_mode = build_precision_context(args.precision, device)
    if precision_mode.device_type and precision_mode.dtype is not None:
        dtype_name = str(precision_mode.dtype).split(".")[-1]
        print(f"Precision: {precision_mode.precision} ({precision_mode.device_type}, {dtype_name})")
    else:
        print(f"Precision: {precision_mode.precision}")

    video_ids = read_video_ids(split_dir, args.split_name)
    # Apply optional deterministic or random shuffling.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        args.shuffle_videos = True  # seed implies shuffle
    if args.shuffle_videos:
        random.shuffle(video_ids)
    if args.limit_videos > 0:
        video_ids = video_ids[: args.limit_videos]
    print(f"Found {len(video_ids)} videos.")

    summary_rows: List[Dict[str, object]] = []
    csv_path = out_dir / "sav_benchmark_summary.csv"

    for video_idx, video_id in enumerate(video_ids, start=1):
        print(f"\n[{video_idx}/{len(video_ids)}] Video: {video_id}")
        frames = list_frames_24fps(split_dir, video_id)
        if not frames:
            print("  No frames found, skipping.")
            continue
        annotations = list_annotated_indices_6fps(split_dir, video_id)
        if not annotations:
            print("  No annotations found, skipping.")
            continue

        objects = sorted(annotations.keys())
        if args.limit_objects > 0:
            objects = objects[: args.limit_objects]

        for obj_id in objects:
            frame_indices = sorted(annotations[obj_id])
            if not frame_indices:
                continue
            prompt_idx = frame_indices[0]
            mask_path = split_dir / "Annotations_6fps" / video_id / obj_id / f"{prompt_idx:05d}.png"
            prompt_mask = load_mask_png(mask_path)
            if prompt_mask is None:
                print(f"  [obj {obj_id}] missing mask for prompt frame {prompt_idx}, skipping object.")
                continue
            if mask_centroid(prompt_mask) is None:
                print(f"  [obj {obj_id}] empty mask at {prompt_idx}, skipping object.")
                continue

            gt_masks = []
            for frame_idx in frame_indices:
                gt_path = split_dir / "Annotations_6fps" / video_id / obj_id / f"{frame_idx:05d}.png"
                gt_mask = load_mask_png(gt_path)
                gt_masks.append(gt_mask)
            gt_mask_map = {idx: mask for idx, mask in zip(frame_indices, gt_masks)}

            for tag, weight_name in models:
                try:
                    runner = _select_runner(tag)
                except ValueError as exc:
                    print(f"    -> {exc}")
                    continue

                overlay_name = None
                if args.save_overlays:
                    overlay_name = f"{video_id}__obj{obj_id}__{tag}"

                print(f"  Model {tag} | obj {obj_id} | prompt frame {prompt_idx}")
                # Execute the selected runner and collect per-object telemetry.
                result = runner(
                    frames_24fps=frames,
                    prompt_frame_idx=prompt_idx,
                    prompt_mask=prompt_mask,
                    imgsz=args.imgsz,
                    weight_name=weight_name,
                    device=device,
                    out_dir=out_dir if args.save_overlays else None,
                    overlay_name=overlay_name,
                    clip_fps=24.0,
                    precision=precision_mode,
                    max_clip_frames=args.max_clip_frames,
                    compile_model=args.compile_models,
                    compile_mode=args.compile_mode,
                    compile_backend=args.compile_backend,
                )

                processed_end = result.get("processed_end_frame")
                scored_indices = [
                    idx
                    for idx in frame_indices
                    if processed_end is None or processed_end == 0 or idx < processed_end
                ]
                if not scored_indices:
                    print("    -> No frames selected for metric evaluation (clip truncated before annotations).")
                    continue

                predicted_masks: List[Optional[np.ndarray]] = []
                gt_eval_masks: List[Optional[np.ndarray]] = []
                mask_sequence = result.get("masks_seq")
                for frame_idx in scored_indices:
                    if mask_sequence is None or frame_idx >= len(mask_sequence) or frame_idx < 0:
                        predicted_masks.append(None)
                    else:
                        predicted_masks.append(
                            _restore_mask_to_original(mask_sequence[frame_idx], result)
                        )
                    gt_eval_masks.append(gt_mask_map.get(frame_idx))

                j_score, f_score = j_and_proxy_jf(predicted_masks, gt_eval_masks)

                row = {
                    "video": video_id,
                    "object": obj_id,
                    "model": tag,
                    "imgsz": args.imgsz,
                    "precision": precision_mode.precision,
                    "frames": result.get("frames"),
                    "processed_end_frame": processed_end,
                    "fps": None if result.get("fps") is None else round(result["fps"], 2),
                    "latency_ms": None if result.get("latency_ms") is None else round(result["latency_ms"], 1),
                    "gpu_peak_alloc_MiB": to_mib(result.get("gpu_peak_alloc")),
                    "gpu_peak_reserved_MiB": to_mib(result.get("gpu_peak_reserved")),
                    "cpu_peak_rss_MiB": to_mib(result.get("cpu_peak_rss")),
                    "J": None if j_score is None else round(j_score, 4),
                    "F": None if f_score is None else round(f_score, 4),
                    "JandF": None if (j_score is None or f_score is None) else round((j_score + f_score) / 2.0, 4),
                    "overlay": result.get("overlay"),
                    "input_H": result.get("H"),
                    "input_W": result.get("W"),
                    "infer_H": result.get("infer_H"),
                    "infer_W": result.get("infer_W"),
                }
                # Include setup time if provided by runner (e.g., SAM2 separation of build vs inference)
                if "setup_ms" in result:
                    row["setup_ms"] = result["setup_ms"]
                summary_rows.append(row)
                print(
                    f"    -> FPS {summary_rows[-1]['fps']}, J {summary_rows[-1]['J']}, F {summary_rows[-1]['F']}, "
                    f"mem GPU alloc {summary_rows[-1]['gpu_peak_alloc_MiB']} MiB"
                )
                reset_gpu_peaks()
                cleanup_after_run()

        if summary_rows:
            # Overwrite the CSV on each iteration so progress survives interruptions.
            fieldnames = list(summary_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)

    print("\nDone.")
    print("Summary CSV:", csv_path)
    if args.save_overlays:
        print("Overlays at:", out_dir)
    return csv_path


def main(argv: Optional[Iterable[str]] = None) -> Path:
    args = parse_args(argv)
    return run(args)


def run_simple_test() -> None:
    print("=" * 60)
    print("SIMPLE TEST MODE - Running basic functionality test")
    print("=" * 60)

    original_argv = sys.argv[:]
    sys.argv = ["sam_comparison.py", "--test_mode", "--models", "sam2_base_points", "--save_overlays", "0"]
    try:
        main()
        print("\n" + "=" * 60)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("The benchmark pipeline is working correctly.")
        print("=" * 60)
    except Exception as exc:  # pragma: no cover
        print(f"\n❌ TEST FAILED: {exc}")
        raise
    finally:
        sys.argv = original_argv
