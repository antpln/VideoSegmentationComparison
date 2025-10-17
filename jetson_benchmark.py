#!/usr/bin/env python3
"""
Two-phase benchmark script optimized for Jetson Orin:
- Phase A (setup): Conservative memory settings, model warmup, data preprocessing
- Phase B (inference): Clean metrics with optimal throughput settings
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Phase identifier from environment
PHASE = os.environ.get("JETSON_BENCH_PHASE", "setup")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson-optimized two-phase benchmark")
    
    # Data & model configuration
    parser.add_argument("--split_dir", type=str, required=True, help="Dataset root (sav_val or sav_test)")
    parser.add_argument("--split_name", type=str, default="sav_val", help="Name of the split file")
    parser.add_argument("--models", type=str, default="sam2_base_points,edgetam_points",
                        help="Comma-separated model_prompt combinations")
    parser.add_argument("--weights_dir", type=str, default=".", help="Directory containing weight files")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size")
    
    # Experiment scope
    parser.add_argument("--limit_videos", type=int, default=0, help="Limit videos (0=all)")
    parser.add_argument("--limit_objects", type=int, default=0, help="Limit objects per video (0=all)")
    parser.add_argument("--shuffle_videos", action="store_true", help="Shuffle video order")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    # Output
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--save_overlays", type=int, default=0, help="Save overlay videos")
    
    # Jetson-specific tuning
    parser.add_argument("--workspace_mb", type=int, default=256, 
                        help="Max workspace size for setup phase (MB)")
    parser.add_argument("--max_frames_in_mem", type=int, default=600,
                        help="Max frames to keep in memory during inference")
    parser.add_argument("--enable_cudnn_tuning", action="store_true",
                        help="Auto-tune cuDNN benchmark setting in setup phase")
    parser.add_argument("--preprocessed_dir", type=str, default=None,
                        help="Directory for preprocessed video data")
    parser.add_argument("--skip_setup", action="store_true",
                        help="Skip setup phase (use existing preprocessed data)")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Only run setup phase")
    parser.add_argument("--autocast_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Dtype for autocast (enables flash attention with fp16/bfloat16; default=bfloat16)")
    
    return parser.parse_args()


def setup_phase_environment():
    """Configure conservative memory settings for setup phase."""
    print("=" * 80)
    print("PHASE A: SETUP (Conservative Memory)")
    print("=" * 80)
    
    try:
        import torch
        
        # Conservative settings to prevent memory spikes
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_grad_enabled(False)
        
        # Gentler memory allocator
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"
        
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Reset all memory stats
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
    except ImportError:
        print("PyTorch not available, skipping CUDA configuration")


def inference_phase_environment(cudnn_benchmark: bool = False):
    """Configure optimal settings for clean inference metrics."""
    print("=" * 80)
    print("PHASE B: INFERENCE (Clean Metrics)")
    print("=" * 80)
    
    try:
        import torch
        
        # Fast inference settings
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = not cudnn_benchmark
        
        # Allow TF32 for speed
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"cuDNN Benchmark: {cudnn_benchmark}")
            
            # Clean slate
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
    except ImportError:
        print("PyTorch not available")


def hard_cleanup():
    """Aggressive memory cleanup before process exit."""
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    gc.collect()


def test_cudnn_benchmark_setting(args: argparse.Namespace) -> bool:
    """
    Test if cuDNN benchmark=True works without OOM.
    Returns True if safe to use, False otherwise.
    """
    print("\n" + "=" * 80)
    print("Auto-tuning cuDNN benchmark setting...")
    print("=" * 80)
    
    try:
        import torch
        from sav_benchmark.runners.registry import get_runner
        
        if not torch.cuda.is_available():
            print("No CUDA available, skipping benchmark test")
            return False
        
        # Test with a small dummy input
        print("Testing benchmark=True with small input...")
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
        
        # Try a simple operation
        try:
            x = torch.randn(1, 3, 256, 256, device='cuda')
            conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
            with torch.inference_mode():
                for _ in range(5):  # Let cuDNN search
                    _ = conv(x)
            torch.cuda.synchronize()
            
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            print(f"✓ benchmark=True test passed (peak: {peak_mb:.1f} MB)")
            
            del x, conv
            torch.cuda.empty_cache()
            return True
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"✗ benchmark=True caused OOM, will use benchmark=False")
                torch.cuda.empty_cache()
                return False
            raise
            
    except Exception as e:
        print(f"✗ Benchmark test failed: {e}")
        return False


def preprocess_video_data(args: argparse.Namespace, preprocessed_dir: Path) -> Dict[str, Any]:
    """
    Preprocess video data in setup phase.
    Loads frames, extracts prompts, saves metadata.
    """
    print("\n" + "=" * 80)
    print("Preprocessing video data...")
    print("=" * 80)
    
    from sav_benchmark.data_io import (
        read_video_ids, list_frames_24fps, 
        list_annotated_indices_6fps, load_mask_png
    )
    from sav_benchmark.prompts import mask_centroid
    
    split_dir = Path(args.split_dir)
    video_ids = read_video_ids(split_dir, args.split_name)
    
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        args.shuffle_videos = True
    
    if args.shuffle_videos:
        import random
        random.shuffle(video_ids)
    
    if args.limit_videos > 0:
        video_ids = video_ids[:args.limit_videos]
    
    print(f"Processing {len(video_ids)} videos...")
    
    # Create preprocessed directory structure
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "video_ids": video_ids,
        "video_data": {},
        "split_dir": str(split_dir),
        "imgsz": args.imgsz,
    }
    
    for video_idx, video_id in enumerate(video_ids, 1):
        print(f"  [{video_idx}/{len(video_ids)}] {video_id}")
        
        frames = list_frames_24fps(split_dir, video_id)
        if not frames:
            print(f"    No frames, skipping")
            continue
            
        annotations = list_annotated_indices_6fps(split_dir, video_id)
        if not annotations:
            print(f"    No annotations, skipping")
            continue
        
        objects = sorted(annotations.keys())
        if args.limit_objects > 0:
            objects = objects[:args.limit_objects]
        
        video_meta = {
            "frames": [str(f) for f in frames],
            "objects": {}
        }
        
        for obj_id in objects:
            frame_indices = sorted(annotations[obj_id])
            if not frame_indices:
                continue
            
            prompt_idx = frame_indices[0]
            mask_path = split_dir / "Annotations_6fps" / video_id / obj_id / f"{prompt_idx:05d}.png"
            prompt_mask = load_mask_png(mask_path)
            
            if prompt_mask is None or mask_centroid(prompt_mask) is None:
                print(f"    Object {obj_id}: invalid prompt mask")
                continue
            
            # Store object metadata
            video_meta["objects"][obj_id] = {
                "prompt_idx": prompt_idx,
                "frame_indices": frame_indices,
                "prompt_mask_shape": prompt_mask.shape,
            }
            
            print(f"    Object {obj_id}: {len(frame_indices)} annotated frames")
        
        if video_meta["objects"]:
            metadata["video_data"][video_id] = video_meta
    
    # Save metadata
    metadata_path = preprocessed_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nPreprocessed data saved to: {preprocessed_dir}")
    print(f"Metadata: {metadata_path}")
    
    return metadata


def warmup_models(args: argparse.Namespace, model_tags: List[str]):
    """
    Warm up models in setup phase to trigger any compilation/optimization.
    Uses small dummy inputs to minimize memory usage.
    """
    print("\n" + "=" * 80)
    print("Warming up models...")
    print("=" * 80)
    
    try:
        import torch
        import numpy as np
        import inspect
        from pathlib import Path
        import shutil
        import gc

        # Create a tiny dummy frame for safe warmup
        warmup_dir = Path("./warmup_data")
        warmup_dir.mkdir(exist_ok=True)
        frame_path = warmup_dir / "00000.jpg"
        if not frame_path.exists():
            import cv2
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(frame_path), img)

        dummy_frames = [frame_path]
        dummy_mask = np.zeros((256, 256), dtype=bool)
        dummy_mask[100:150, 100:150] = True

        from sav_benchmark.main import _resolve_models, _select_runner
        from sav_benchmark.utils import device_str

        weights_dir = Path(args.weights_dir)
        models = _resolve_models(args, model_tags, weights_dir)
        device = device_str()

        for tag, weight_name in models:
            print(f"\n  Warming up {tag}...")
            try:
                runner = _select_runner(tag)

                # Candidate warmup kwargs (single-frame / small)
                warmup_kwargs = dict(
                    frames_24fps=dummy_frames,
                    prompt_frame_idx=0,
                    prompt_mask=dummy_mask,
                    imgsz=256,
                    weight_name=weight_name,
                    device=device,
                    out_dir=None,
                    overlay_name=None,
                    clip_fps=24.0,
                    compile_model=False,
                )

                # Inspect runner signature and filter kwargs to accepted params only
                try:
                    sig = inspect.signature(runner)
                    accepted = set(sig.parameters.keys())
                except Exception:
                    accepted = None

                filtered = {k: v for k, v in warmup_kwargs.items() if accepted is None or k in accepted}

                # Determine autocast dtype for warmup
                dtype_map = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }
                autocast_dtype = dtype_map.get(args.autocast_dtype, torch.bfloat16)
                use_autocast = args.autocast_dtype != "float32" and torch.cuda.is_available()

                # Attempt 1: single-frame warmup (preferred)
                try:
                    if use_autocast:
                        with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                            result = runner(**filtered)
                    else:
                        result = runner(**filtered)
                    
                    # Check if result is valid (has masks)
                    if isinstance(result, dict):
                        masks_seq = result.get("masks_seq", [])
                        valid_masks = sum(1 for m in masks_seq if m is not None) if masks_seq else 0
                        
                        if valid_masks == 0:
                            # Runner returned empty result (internal error)
                            print(f"    ! single-frame warmup returned 0 valid masks (internal error)")
                            raise RuntimeError("Warmup inference produced no masks")
                    
                    print("    ✓ Warmup: single-frame run succeeded")
                except Exception as e1:
                    print(f"    ! single-frame warmup failed: {e1}")

                    # Attempt 2: minimal load-only call (trigger model instantiation/weight load)
                    minimal = {}
                    if accepted:
                        if "weight_name" in accepted:
                            minimal["weight_name"] = weight_name
                        if "device" in accepted:
                            minimal["device"] = device
                        if "compile_model" in accepted:
                            minimal["compile_model"] = False

                    if minimal:
                        try:
                            _ = runner(**minimal)
                            print("    ✓ Warmup: minimal load-only succeeded")
                        except Exception as e2:
                            print(f"    ✗ Minimal load-only failed: {e2}; skipping warmup for this model")
                    else:
                        # Last-resort positional attempt
                        try:
                            _ = runner(weight_name)
                            print("    ✓ Warmup: positional load-only succeeded")
                        except Exception:
                            try:
                                _ = runner(weight_name, device)
                                print("    ✓ Warmup: positional load-only (weight_name,device) succeeded")
                            except Exception as e3:
                                print(f"    ✗ All warmup attempts failed: {e3}; skipping warmup for this model")

                # cleanup
                del runner
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"    ✗ Warmup skipped/failed for {tag}: {e}")
                continue

        # remove warmup files
        shutil.rmtree(warmup_dir, ignore_errors=True)

    except Exception as e:
        print(f"Model warmup failed: {e}")


def run_setup_phase(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Phase A: Setup with conservative memory settings.
    """
    setup_phase_environment()
    
    preprocessed_dir = Path(args.preprocessed_dir or "./preprocessed_data")
    
    # Test cuDNN benchmark if requested
    cudnn_benchmark_safe = False
    if args.enable_cudnn_tuning:
        cudnn_benchmark_safe = test_cudnn_benchmark_setting(args)
    
    # Preprocess data
    metadata = preprocess_video_data(args, preprocessed_dir)
    
    # Warm up models
    model_tags = [tag.strip() for tag in args.models.split(",") if tag.strip()]
    warmup_models(args, model_tags)
    
    # Save configuration for inference phase
    config = {
        "cudnn_benchmark": cudnn_benchmark_safe,
        "preprocessed_dir": str(preprocessed_dir),
        "args": vars(args),
    }
    
    config_path = preprocessed_dir / "inference_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Setup phase complete!")
    print(f"Configuration saved to: {config_path}")
    print(f"Recommended cuDNN benchmark setting: {cudnn_benchmark_safe}")
    print("=" * 80)
    
    return config


def run_inference_phase(args: argparse.Namespace, setup_config: Optional[Dict[str, Any]] = None):
    """
    Phase B: Clean inference with optimal settings.
    """
    preprocessed_dir = Path(args.preprocessed_dir or "./preprocessed_data")
    
    # Load setup configuration
    if setup_config is None:
        config_path = preprocessed_dir / "inference_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                setup_config = json.load(f)
        else:
            setup_config = {"cudnn_benchmark": False}
    
    cudnn_benchmark = setup_config.get("cudnn_benchmark", False)
    inference_phase_environment(cudnn_benchmark)
    
    # Load preprocessed metadata
    metadata_path = preprocessed_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Preprocessed metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\nLoaded metadata for {len(metadata['video_data'])} videos")
    
    # Run benchmark using preprocessed data
    from sav_benchmark.main import _resolve_models, _select_runner
    from sav_benchmark.data_io import load_mask_png, ensure_dir
    from sav_benchmark.metrics import j_and_proxy_jf
    from sav_benchmark.utils import device_str, to_mib
    import csv
    import numpy as np
    
    split_dir = Path(metadata["split_dir"])
    out_dir = Path(args.out_dir) if args.out_dir else split_dir / "jetson_benchmark_outputs"
    ensure_dir(out_dir)
    
    weights_dir = Path(args.weights_dir)
    model_tags = [tag.strip() for tag in args.models.split(",") if tag.strip()]
    models = _resolve_models(args, model_tags, weights_dir)
    device = device_str()
    
    summary_rows: List[Dict[str, Any]] = []
    csv_path = out_dir / "jetson_benchmark_summary.csv"
    
    # Process videos one at a time with memory cleanup between
    for video_idx, (video_id, video_meta) in enumerate(metadata["video_data"].items(), 1):
        print(f"\n[{video_idx}/{len(metadata['video_data'])}] Video: {video_id}")
        
        frames = [Path(f) for f in video_meta["frames"]]
        
        for obj_id, obj_meta in video_meta["objects"].items():
            prompt_idx = obj_meta["prompt_idx"]
            frame_indices = obj_meta["frame_indices"]
            
            # Load prompt mask
            mask_path = split_dir / "Annotations_6fps" / video_id / obj_id / f"{prompt_idx:05d}.png"
            prompt_mask = load_mask_png(mask_path)
            
            # Load ground truth masks
            gt_masks = []
            for frame_idx in frame_indices:
                gt_path = split_dir / "Annotations_6fps" / video_id / obj_id / f"{frame_idx:05d}.png"
                gt_mask = load_mask_png(gt_path)
                gt_masks.append(gt_mask)
            
            # Run each model
            for tag, weight_name in models:
                print(f"  Model {tag} | obj {obj_id} | prompt frame {prompt_idx}")
                
                try:
                    runner = _select_runner(tag)
                except ValueError as e:
                    print(f"    -> {e}")
                    continue
                
                overlay_name = None
                if args.save_overlays:
                    overlay_name = f"{video_id}__obj{obj_id}__{tag}"
                
                # Run inference with autocast for flash attention support
                import torch
                
                # Determine autocast dtype
                dtype_map = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }
                autocast_dtype = dtype_map.get(args.autocast_dtype, torch.bfloat16)
                
                # Use autocast if not float32 and CUDA is available
                use_autocast = args.autocast_dtype != "float32" and torch.cuda.is_available()
                
                if use_autocast:
                    with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
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
                            compile_model=False,  # Already warmed up
                        )
                else:
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
                        compile_model=False,  # Already warmed up
                    )
                
                # Evaluate predictions
                predicted_masks: List[Optional[np.ndarray]] = []
                if result.get("masks_seq") is None:
                    predicted_masks = [None] * len(gt_masks)
                else:
                    mask_sequence = result["masks_seq"]
                    for frame_idx in frame_indices:
                        predicted_masks.append(
                            mask_sequence[frame_idx] if 0 <= frame_idx < len(mask_sequence) else None
                        )
                
                j_score, jf_proxy = j_and_proxy_jf(predicted_masks, gt_masks)
                
                row = {
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
                    "cudnn_benchmark": cudnn_benchmark,
                }
                
                if "setup_ms" in result:
                    row["setup_ms"] = result["setup_ms"]
                
                summary_rows.append(row)
                print(
                    f"    -> FPS {row['fps']}, J {row['J']}, "
                    f"mem GPU alloc {row['gpu_peak_alloc_MiB']} MiB"
                )
                
                # Aggressive cleanup between models to prevent OOM
                del runner, result
                if 'predicted_masks' in locals():
                    del predicted_masks
                if 'mask_sequence' in locals():
                    del mask_sequence
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except ImportError:
                    pass
            
            # Additional cleanup between objects
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        
        # Cleanup between videos
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except ImportError:
            pass
        # Save progress after each video
        if summary_rows:
            fieldnames = list(summary_rows[0].keys())
            with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)
        
        # Aggressive cleanup between videos
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
    
    print("\n" + "=" * 80)
    print("Inference phase complete!")
    print(f"Results: {csv_path}")
    print("=" * 80)
    
    return csv_path


def main():
    args = parse_args()
    
    # Determine which phases to run
    run_setup = not args.skip_setup
    run_inference = not args.skip_inference
    
    setup_config = None
    
    try:
        if run_setup:
            setup_config = run_setup_phase(args)
            hard_cleanup()
            
            # If both phases requested, re-launch for inference
            if run_inference:
                print("\n" + "=" * 80)
                print("Relaunching for clean inference phase...")
                print("=" * 80 + "\n")
                time.sleep(2)  # Brief pause to ensure cleanup
                
                # Re-launch this script with skip_setup flag
                cmd = [sys.executable, __file__]
                for key, value in vars(args).items():
                    if key == "skip_setup":
                        continue
                    if value is None or value is False:
                        continue
                    if value is True:
                        cmd.append(f"--{key}")
                    else:
                        cmd.extend([f"--{key}", str(value)])
                cmd.append("--skip_setup")
                
                result = subprocess.run(cmd)
                sys.exit(result.returncode)
        
        elif run_inference:
            run_inference_phase(args, setup_config)
            hard_cleanup()
    
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
