#!/usr/bin/env python3
"""GPU profiling wrapper for SAM models using NVIDIA Nsight Systems.

This script runs the same dataset and models as sam_comparison.py but focuses on
GPU profiling using nsight-sys for performance analysis on:
- NVIDIA RTX 3090 (desktop)
- NVIDIA Jetson Orin (with JetPack 6.2)

Accuracy metrics are omitted; the focus is on GPU utilization, kernel timing,
memory bandwidth, and other performance characteristics captured by nsight.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from sav_benchmark.data_io import ensure_dir, list_annotated_indices_6fps, list_frames_24fps, load_mask_png, read_video_ids
from sav_benchmark.prompts import mask_centroid
from sav_benchmark.runners import edgetam  # noqa: F401  (ensure registration side-effects)
from sav_benchmark.runners import sam2  # noqa: F401
from sav_benchmark.runners.registry import get_runner
from sav_benchmark.synthetic import create_synthetic_test_data
from sav_benchmark.utils import device_str, to_mib

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

REPO_ROOT = Path(__file__).resolve().parent


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM GPU Profiling with Nsight")
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
    parser.add_argument("--limit_videos", type=int, default=1, help="Limit number of videos (0 = all, default=1 for profiling)")
    parser.add_argument("--limit_objects", type=int, default=1, help="Limit number of objects per video (0 = all, default=1 for profiling)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for profiling results")
    parser.add_argument("--compile_models", action="store_true", help="Compile models with torch.compile before inference")
    parser.add_argument("--compile_backend", type=str, default=None, help="Optional backend to use for torch.compile")
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        help="torch.compile mode (e.g., default, reduce-overhead, max-autotune)",
    )
    parser.add_argument("--shuffle_videos", action="store_true", help="Shuffle video order before limiting / processing")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling (implies --shuffle_videos)")
    parser.add_argument(
        "--autocast",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "none"],
        help="Enable CUDA autocast with the given dtype (bf16/fp16) or disable (none)",
    )
    parser.add_argument(
        "--disable_cudnn",
        action="store_true",
        help="Disable cuDNN to reduce conv workspace memory (slower but can avoid OOM)",
    )
    
    # Nsight-specific arguments
    parser.add_argument(
        "--nsight_path",
        type=str,
        default="nsys",
        help="Path to nsys executable (default: nsys in PATH)",
    )
    parser.add_argument(
        "--profile_mode",
        type=str,
        default="external",
        choices=["external", "inline"],
        help="external: wrap entire script with nsys; inline: use NVTX ranges (default: external)",
    )
    parser.add_argument(
        "--nsight_trace_options",
        type=str,
        default="cuda,cudnn,cublas,nvtx,osrt",
        help="Comma-separated nsys trace options (default: cuda,cudnn,cublas,nvtx,osrt)",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=1,
        help="Number of warmup runs before profiling (default: 1)",
    )
    parser.add_argument(
        "--profile_runs",
        type=int,
        default=1,
        help="Number of profiled runs per model (default: 1)",
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
        out_dir = Path(args.out_dir) if args.out_dir else Path("./test_profile_outputs")
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
        out_dir = Path(args.out_dir) if args.out_dir else split_dir / "profile_outputs"
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
    # Configurable global autocast
    try:
        if args.autocast == "bf16":
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            print("Autocast: enabled (cuda, bfloat16)")
        elif args.autocast == "fp16":
            torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
            print("Autocast: enabled (cuda, float16)")
        else:
            print("Autocast: disabled")
    except Exception as e:  # pragma: no cover
        print(f"Autocast: not enabled ({e})")


def _check_nsight_available(nsight_path: str) -> bool:
    """Check if nsight-sys is available on the system."""
    try:
        result = subprocess.run(
            [nsight_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print(f"Nsight Systems found: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    print(f"[WARN] Nsight Systems not found at '{nsight_path}'")
    print("       Profiling will be disabled. Install nsys or provide --nsight_path")
    return False


def _setup_nvtx_markers():
    """Setup NVTX markers for inline profiling mode."""
    try:
        import nvtx  # type: ignore
        return nvtx
    except ImportError:
        print("[WARN] nvtx package not available. Install with: pip install nvtx")
        print("       Using basic profiling without NVTX markers")
        return None


def run_profile(args: argparse.Namespace) -> Path:
    """Main profiling routine."""
    split_dir, out_dir = _prepare_dataset(args)
    weights_dir = Path(args.weights_dir)
    model_tags = [tag.strip() for tag in args.models.split(",") if tag.strip()]
    models = _resolve_models(args, model_tags, weights_dir)
    if not models:
        raise SystemExit("No valid models selected")

    _configure_torch(args)
    device = device_str()

    # Check nsight availability for external profiling
    nsight_available = False
    if args.profile_mode == "external":
        nsight_available = _check_nsight_available(args.nsight_path)
    
    # Setup NVTX for inline profiling
    nvtx = None
    if args.profile_mode == "inline":
        nvtx = _setup_nvtx_markers()

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
    print(f"Found {len(video_ids)} videos for profiling.")

    profile_results: List[Dict[str, object]] = []
    csv_path = out_dir / "profile_summary.csv"

    # Create nsight output directory
    nsight_dir = out_dir / "nsight_reports"
    ensure_dir(nsight_dir)

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

            for tag, weight_name in models:
                try:
                    runner = _select_runner(tag)
                except ValueError as exc:
                    print(f"    -> {exc}")
                    continue

                print(f"  Model {tag} | obj {obj_id} | prompt frame {prompt_idx}")
                
                # Warmup runs
                if args.warmup_runs > 0:
                    print(f"    Running {args.warmup_runs} warmup iteration(s)...")
                    for warmup_i in range(args.warmup_runs):
                        if nvtx:
                            nvtx.push_range(f"warmup_{tag}_run{warmup_i}")
                        _ = runner(
                            frames_24fps=frames,
                            prompt_frame_idx=prompt_idx,
                            prompt_mask=prompt_mask,
                            imgsz=args.imgsz,
                            weight_name=weight_name,
                            device=device,
                            out_dir=None,
                            overlay_name=None,
                            clip_fps=24.0,
                            compile_model=args.compile_models,
                            compile_mode=args.compile_mode,
                            compile_backend=args.compile_backend,
                        )
                        if nvtx:
                            nvtx.pop_range()
                        if torch is not None and torch.cuda.is_available():
                            torch.cuda.synchronize()
                
                # Profiled runs
                print(f"    Running {args.profile_runs} profiled iteration(s)...")
                for run_i in range(args.profile_runs):
                    report_name = f"{video_id}_obj{obj_id}_{tag}_run{run_i}"
                    nsight_report = nsight_dir / f"{report_name}.nsys-rep"
                    
                    if nvtx:
                        nvtx.push_range(f"profile_{tag}_run{run_i}")
                    
                    # Execute the selected runner
                    result = runner(
                        frames_24fps=frames,
                        prompt_frame_idx=prompt_idx,
                        prompt_mask=prompt_mask,
                        imgsz=args.imgsz,
                        weight_name=weight_name,
                        device=device,
                        out_dir=None,
                        overlay_name=None,
                        clip_fps=24.0,
                        compile_model=args.compile_models,
                        compile_mode=args.compile_mode,
                        compile_backend=args.compile_backend,
                    )
                    
                    if nvtx:
                        nvtx.pop_range()
                    
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.synchronize()

                    row = {
                        "video": video_id,
                        "object": obj_id,
                        "model": tag,
                        "run": run_i,
                        "imgsz": args.imgsz,
                        "frames": result.get("frames"),
                        "fps": None if result.get("fps") is None else round(result["fps"], 2),
                        "latency_ms": None if result.get("latency_ms") is None else round(result["latency_ms"], 1),
                        "gpu_peak_alloc_MiB": to_mib(result.get("gpu_peak_alloc")),
                        "gpu_peak_reserved_MiB": to_mib(result.get("gpu_peak_reserved")),
                        "cpu_peak_rss_MiB": to_mib(result.get("cpu_peak_rss")),
                        "nsight_report": str(nsight_report) if nsight_available else None,
                    }
                    # Include setup time if provided by runner
                    if "setup_ms" in result:
                        row["setup_ms"] = result["setup_ms"]
                    
                    profile_results.append(row)
                    print(
                        f"    -> Run {run_i}: FPS {row['fps']}, "
                        f"Latency {row['latency_ms']}ms, "
                        f"GPU alloc {row['gpu_peak_alloc_MiB']} MiB"
                    )

        if profile_results:
            # Write CSV after each video
            fieldnames = list(profile_results[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(profile_results)

    print("\n" + "=" * 60)
    print("Profiling Complete!")
    print("=" * 60)
    print(f"Summary CSV: {csv_path}")
    print(f"Nsight reports: {nsight_dir}")
    
    # Generate summary statistics
    _generate_summary_stats(profile_results, out_dir)
    
    return csv_path


def _generate_summary_stats(profile_results: List[Dict[str, object]], out_dir: Path) -> None:
    """Generate summary statistics from profiling results."""
    if not profile_results:
        return
    
    summary = {}
    for row in profile_results:
        model = row["model"]
        if model not in summary:
            summary[model] = {
                "fps": [],
                "latency_ms": [],
                "gpu_peak_alloc_MiB": [],
                "gpu_peak_reserved_MiB": [],
            }
        
        if row["fps"] is not None:
            summary[model]["fps"].append(float(row["fps"]))
        if row["latency_ms"] is not None:
            summary[model]["latency_ms"].append(float(row["latency_ms"]))
        if row["gpu_peak_alloc_MiB"] is not None:
            summary[model]["gpu_peak_alloc_MiB"].append(float(row["gpu_peak_alloc_MiB"]))
        if row["gpu_peak_reserved_MiB"] is not None:
            summary[model]["gpu_peak_reserved_MiB"].append(float(row["gpu_peak_reserved_MiB"]))
    
    stats = {}
    for model, metrics in summary.items():
        stats[model] = {}
        for metric, values in metrics.items():
            if values:
                stats[model][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                }
    
    stats_path = out_dir / "profile_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nSummary statistics: {stats_path}")
    print("\nPer-Model Averages:")
    for model, model_stats in stats.items():
        print(f"\n  {model}:")
        for metric, values in model_stats.items():
            print(f"    {metric}: {values['mean']:.2f} ± {values['std']:.2f}")


def main(argv: Optional[Iterable[str]] = None) -> Path:
    """Main entry point for GPU profiling."""
    args = parse_args(argv)
    return run_profile(args)


def run_simple_test() -> None:
    """Simple test mode for quick validation."""
    print("=" * 60)
    print("SIMPLE PROFILE TEST MODE")
    print("=" * 60)

    original_argv = sys.argv[:]
    sys.argv = [
        "sam_gpu_profiles.py",
        "--test_mode",
        "--models", "sam2_base_points",
        "--warmup_runs", "1",
        "--profile_runs", "1",
        "--profile_mode", "inline",
    ]
    try:
        main()
        print("\n" + "=" * 60)
        print("✅ PROFILE TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
    except Exception as exc:  # pragma: no cover
        print(f"\n❌ PROFILE TEST FAILED: {exc}")
        raise
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_simple_test()
    else:
        main()
