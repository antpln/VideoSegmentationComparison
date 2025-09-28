"""Command-line entry point for the SA-V benchmark."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .data_io import ensure_dir, list_annotated_indices_6fps, list_frames_24fps, load_mask_png, read_video_ids
from .metrics import j_and_proxy_jf
from .prompts import mask_centroid
from .runners.edgetam import EDGETAM_RUNNERS
from .runners.sam2 import SAM2_RUNNERS
from .synthetic import create_synthetic_test_data
from .utils import device_str, to_mib

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
    return parser.parse_args(args=argv)


def _resolve_models(model_tags: List[str], weights_dir: Path) -> List[Tuple[str, str]]:
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
        if not weight_file:
            print(f"[WARN] Unknown model tag: {tag} (skipping)")
            continue
        weight_path = weights_dir / weight_file
        resolved.append((tag, str(weight_path) if weight_path.exists() else weight_file))
    return resolved


def _select_runner(tag: str):
    """Return the appropriate runner callable for the provided model tag."""
    if "_" not in tag:
        raise ValueError(f"Malformed model tag: {tag}")
    parts = tag.split("_")
    prompt_type = parts[-1]
    model_name = "_".join(parts[:-1])

    if model_name.startswith("sam2_"):
        runner = SAM2_RUNNERS.get(prompt_type)
    elif model_name == "edgetam":
        runner = EDGETAM_RUNNERS.get(prompt_type)
    else:
        runner = None

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


def _configure_torch() -> None:
    """Apply conservative CUDA defaults when GPUs are available."""
    if torch is None or not torch.cuda.is_available():
        print("Using CPU")
        return
    print("CUDA available:", True, "Device:", torch.cuda.get_device_name(0))
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:  # pragma: no cover
        pass


def run(args: argparse.Namespace) -> Path:
    split_dir, out_dir = _prepare_dataset(args)
    weights_dir = Path(args.weights_dir)
    model_tags = [tag.strip() for tag in args.models.split(",") if tag.strip()]
    models = _resolve_models(model_tags, weights_dir)
    if not models:
        raise SystemExit("No valid models selected")

    _configure_torch()
    device = device_str()

    video_ids = read_video_ids(split_dir, args.split_name)
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
                    compile_model=args.compile_models,
                    compile_mode=args.compile_mode,
                    compile_backend=args.compile_backend,
                )

                predicted_masks: List[Optional[np.ndarray]] = []
                if result.get("masks_seq") is None:
                    predicted_masks = [None] * len(gt_masks)
                else:
                    mask_sequence = result["masks_seq"]
                    for frame_idx in frame_indices:
                        predicted_masks.append(mask_sequence[frame_idx] if 0 <= frame_idx < len(mask_sequence) else None)

                j_score, jf_proxy = j_and_proxy_jf(predicted_masks, gt_masks)

                summary_rows.append(
                    {
                        "video": video_id,
                        "object": obj_id,
                        "model": tag,
                        "imgsz": args.imgsz,
                        "frames": result.get("frames"),
                        "fps": None if result.get("fps") is None else round(result["fps"], 2),
                        "latency_ms": None if result.get("latency_ms") is None else round(result["latency_ms"], 1),
                        "gpu_peak_alloc_MiB": to_mib(result.get("gpu_peak_alloc")),
                        "gpu_peak_reserved_MiB": to_mib(result.get("gpu_peak_reserved")),
                        "cpu_peak_rss_MiB": to_mib(result.get("cpu_peak_rss")),
                        "J": None if j_score is None else round(j_score, 4),
                        "JandF_proxy": None if jf_proxy is None else round(jf_proxy, 4),
                        "overlay": result.get("overlay"),
                    }
                )
                print(
                    f"    -> FPS {summary_rows[-1]['fps']}, J {summary_rows[-1]['J']}, "
                    f"mem GPU alloc {summary_rows[-1]['gpu_peak_alloc_MiB']} MiB"
                )

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
