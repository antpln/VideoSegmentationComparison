# GPU Profiling for SAM Models with NVIDIA Nsight

This directory contains scripts for GPU profiling of SAM2 and EdgeTAM models using NVIDIA Nsight Systems. The profiling setup is designed to work on both:
- **NVIDIA RTX 3090** (desktop GPU)
- **NVIDIA Jetson Orin** (with JetPack 6.2)

## Overview

The profiling scripts clone the experimental setup from `sam_comparison.py` but focus on GPU performance metrics rather than accuracy:
- GPU kernel execution times
- Memory bandwidth utilization
- CUDA API calls and synchronization
- cuDNN/cuBLAS operations
- Memory allocation patterns

## Files

- **`sam_gpu_profiles.py`**: Main profiling script (Python)
- **`run_nsight_profile.sh`**: Helper script for external nsys profiling (Bash)
- **`PROFILE_README.md`**: This documentation

## Requirements

### Software
1. **NVIDIA Nsight Systems**
   - Desktop: Download from [developer.nvidia.com/nsight-systems](https://developer.nvidia.com/nsight-systems)
   - Jetson: Included with JetPack 6.2
   
2. **Python Dependencies** (same as main benchmark)
   ```bash
   pip install torch torchvision numpy opencv-python pillow
   ```

3. **Optional: NVTX for inline profiling**
   ```bash
   pip install nvtx
   ```

### Hardware
- NVIDIA GPU with CUDA support
- Tested on: RTX 3090, Jetson Orin

## Usage

### Method 1: External Profiling with Nsight (Recommended)

Use the bash wrapper script to profile the entire execution:

```bash
# Basic usage with test data
./run_nsight_profile.sh --test_mode

# Profile on real dataset
./run_nsight_profile.sh \
    --split_dir /path/to/sav_val \
    --models sam2_base_points,sam2_base_bbox \
    --limit_videos 2 \
    --limit_objects 1 \
    --warmup_runs 2 \
    --profile_runs 3

# Profile with torch.compile
./run_nsight_profile.sh \
    --split_dir /path/to/sav_val \
    --models edgetam_points \
    --compile_models \
    --compile_mode max-autotune
```

**Output**: Creates `.nsys-rep` files in `nsight_profiles/` directory

### Method 2: Inline Profiling with NVTX Ranges

Run the script directly with NVTX markers:

```bash
python3 sam_gpu_profiles.py \
    --split_dir /path/to/sav_val \
    --models sam2_base_points \
    --profile_mode inline \
    --warmup_runs 1 \
    --profile_runs 3 \
    --limit_videos 1
```

Then manually wrap with nsys if needed:
```bash
nsys profile -o profile_inline \
    --trace=cuda,nvtx \
    python3 sam_gpu_profiles.py --profile_mode inline ...
```

### Method 3: Python-Only (No Nsight)

Run without nsight for basic timing metrics:

```bash
python3 sam_gpu_profiles.py \
    --split_dir /path/to/sav_val \
    --models sam2_tiny_points,sam2_small_points \
    --limit_videos 1 \
    --profile_mode inline
```

## Command-Line Arguments

### Dataset Options
- `--split_dir PATH`: Dataset root directory (required unless `--test_mode`)
- `--test_mode`: Use synthetic data for quick validation
- `--split_name NAME`: Split file name (default: `sav_val`)
- `--weights_dir PATH`: Directory containing model weights (default: `.`)

### Model Selection
- `--models LIST`: Comma-separated model tags (default: `sam2_base_points,sam2_base_bbox,edgetam_points,edgetam_bbox`)
  - Available: `sam2_tiny_*`, `sam2_small_*`, `sam2_base_*`, `sam2_large_*`, `edgetam_*`
  - Prompts: `*_points`, `*_bbox`
- `--imgsz SIZE`: Image size for inference (default: 768)

### Profiling Configuration
- `--profile_mode MODE`: `external` or `inline` (default: `external`)
- `--warmup_runs N`: Warmup iterations before profiling (default: 1)
- `--profile_runs N`: Number of profiled runs per model (default: 1)
- `--limit_videos N`: Limit videos for profiling (default: 1, 0=all)
- `--limit_objects N`: Limit objects per video (default: 1, 0=all)

### Nsight Options (for bash script)
Environment variables:
- `NSYS_CMD`: Path to nsys executable (default: `nsys`)
- `NSYS_TRACE`: APIs to trace (default: `cuda,cudnn,cublas,nvtx,osrt`)
- `NSYS_OUTPUT_DIR`: Output directory (default: `./nsight_profiles`)

### Torch Options
- `--compile_models`: Enable torch.compile
- `--compile_mode MODE`: Compile mode (`default`, `reduce-overhead`, `max-autotune`)
- `--compile_backend BACKEND`: Compile backend (optional)
- `--autocast TYPE`: Enable autocast (`bf16`, `fp16`, `none`)
- `--disable_cudnn`: Disable cuDNN (for OOM debugging)

### Output
- `--out_dir PATH`: Output directory for results (default: `<split_dir>/profile_outputs`)

## Output Files

### 1. Profile Summary CSV (`profile_summary.csv`)
Contains timing and memory metrics for each run:
- `video`, `object`, `model`, `run`: Identifiers
- `fps`, `latency_ms`: Throughput metrics
- `gpu_peak_alloc_MiB`, `gpu_peak_reserved_MiB`: GPU memory
- `cpu_peak_rss_MiB`: CPU memory
- `nsight_report`: Path to nsys report (if available)

### 2. Summary Statistics JSON (`profile_stats.json`)
Aggregated statistics per model:
```json
{
  "sam2_base_points": {
    "fps": {"mean": 12.34, "std": 0.56, "min": 11.5, "max": 13.2},
    "latency_ms": {"mean": 81.0, "std": 3.7, ...},
    ...
  }
}
```

### 3. Nsight Reports (`nsight_reports/*.nsys-rep`)
Binary profiling data for Nsight Systems GUI/CLI analysis.

## Analyzing Nsight Reports

### GUI Analysis
```bash
# Open in Nsight Systems GUI
nsys-ui nsight_reports/video001_obj0001_sam2_base_points_run0.nsys-rep
```

Key views:
- **Timeline**: GPU kernel execution, CPU threads, memory transfers
- **CUDA HW**: GPU utilization, SM activity
- **Memory**: Allocations, transfers, bandwidth

### CLI Analysis
```bash
# Summary statistics
nsys stats nsight_reports/*.nsys-rep

# CUDA kernel statistics
nsys stats --report cuda_gpu_kern_sum nsight_reports/*.nsys-rep

# Memory operations
nsys stats --report cuda_mem_sum nsight_reports/*.nsys-rep

# Export to SQLite for custom analysis
nsys export -t sqlite nsight_reports/*.nsys-rep
```

### Comparing Models
```bash
# Compare multiple runs
nsys stats \
    nsight_reports/video001_obj0001_sam2_base_points_run0.nsys-rep \
    nsight_reports/video001_obj0001_sam2_tiny_points_run0.nsys-rep \
    --report cuda_gpu_kern_sum
```

## Platform-Specific Notes

### RTX 3090
- Full bf16 support (use `--autocast bf16`)
- High memory bandwidth: watch for memory-bound kernels
- Consider `--compile_mode max-autotune` for best performance

### Jetson Orin (JetPack 6.2)
- Limited bf16 support: use `--autocast fp16` or `--autocast none`
- Unified memory architecture: CPU/GPU share memory
- Lower power constraints: may throttle under sustained load
- Use `nsys` from JetPack installation: `/opt/nvidia/nsight-systems/bin/nsys`

```bash
# Example for Jetson Orin
NSYS_CMD=/opt/nvidia/nsight-systems/bin/nsys \
./run_nsight_profile.sh \
    --split_dir /path/to/sav_val \
    --models sam2_tiny_points \
    --autocast fp16 \
    --limit_videos 1
```

## Troubleshooting

### "nsys: command not found"
- Install Nsight Systems or set `NSYS_CMD` environment variable
- Jetson: Check `/opt/nvidia/nsight-systems/bin/nsys`

### Out of Memory Errors
```bash
# Try these in order:
# 1. Use smaller model
--models sam2_tiny_points

# 2. Disable cuDNN
--disable_cudnn

# 3. Use fp16 instead of bf16
--autocast fp16

# 4. Reduce image size
--imgsz 512
```

### Profile Files Too Large
```bash
# Reduce traced APIs
NSYS_TRACE=cuda,nvtx ./run_nsight_profile.sh ...

# Disable CPU sampling
NSYS_SAMPLE=none ./run_nsight_profile.sh ...
```

### Compilation Errors with torch.compile
- Ensure PyTorch >= 2.0
- Try different backend: `--compile_backend inductor`
- Or disable: remove `--compile_models` flag

## Example Workflows

### Quick Test
```bash
# Verify everything works
./run_nsight_profile.sh --test_mode
```

### Compare Model Sizes
```bash
./run_nsight_profile.sh \
    --split_dir /data/sav_val \
    --models sam2_tiny_points,sam2_small_points,sam2_base_points \
    --limit_videos 3 \
    --limit_objects 1 \
    --profile_runs 5
```

### Profile with Compilation
```bash
./run_nsight_profile.sh \
    --split_dir /data/sav_val \
    --models sam2_base_points \
    --compile_models \
    --compile_mode max-autotune \
    --warmup_runs 3 \
    --profile_runs 10
```

### Jetson Power Efficiency Test
```bash
# Profile with different precision modes
for mode in fp16 none; do
    NSYS_OUTPUT_NAME="jetson_${mode}_%p" \
    ./run_nsight_profile.sh \
        --split_dir /data/sav_val \
        --models edgetam_points \
        --autocast $mode \
        --profile_runs 5
done
```

## Tips for Best Results

1. **Warmup is crucial**: Always use `--warmup_runs >= 1` to avoid cold-start effects
2. **Multiple runs**: Use `--profile_runs >= 3` for statistical significance
3. **Limit data**: For profiling, `--limit_videos 1 --limit_objects 1` is usually sufficient
4. **GPU synchronization**: Results include CUDA sync calls for accurate timing
5. **Compilation**: torch.compile adds overhead on first run, but improves subsequent runs

## Further Reading

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [CUDA Profiling Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Jetson Orin Optimization](https://docs.nvidia.com/jetson/)
