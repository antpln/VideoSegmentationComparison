# GPU Profiling Setup - Summary

## Created Files

This setup adds GPU profiling capabilities to the VideoSegmentationComparison project for benchmarking SAM2 and EdgeTAM models using NVIDIA Nsight Systems.

### 1. `sam_gpu_profiles.py` (Main Profiling Script)
**Purpose**: Python script that clones the experiment from `sam_comparison.py` but focuses on GPU profiling instead of accuracy metrics.

**Key Features**:
- Runs same datasets and models as original comparison
- Supports both external (nsys wrapper) and inline (NVTX) profiling modes
- Tracks GPU/CPU memory usage and timing metrics
- Configurable warmup and profiling runs
- Generates CSV summaries and JSON statistics
- Compatible with RTX 3090 and Jetson Orin

**Usage**:
```bash
python3 sam_gpu_profiles.py --split_dir /path/to/sav_val --models sam2_base_points
```

### 2. `run_nsight_profile.sh` (Nsight Wrapper Script)
**Purpose**: Bash helper script that wraps Python execution with `nsys` for comprehensive GPU profiling.

**Key Features**:
- Automatically configures nsys with optimal settings
- Checks for nsys availability
- Creates output directories
- Exports to multiple formats (SQLite, CSV)
- Customizable via environment variables
- Falls back to test mode if no args provided

**Usage**:
```bash
./run_nsight_profile.sh --split_dir /path/to/sav_val --models sam2_base_points
```

### 3. `PROFILE_README.md` (Documentation)
**Purpose**: Comprehensive documentation for GPU profiling setup.

**Contents**:
- Overview and requirements
- Platform-specific instructions (RTX 3090, Jetson Orin)
- Command-line argument reference
- Output file descriptions
- Nsight report analysis guide
- Troubleshooting section
- Example workflows

### 4. `profile_examples.sh` (Example Commands)
**Purpose**: Collection of ready-to-use profiling commands for various scenarios.

**Includes**:
- Quick tests
- RTX 3090 specific commands
- Jetson Orin specific commands
- Memory profiling examples
- Batch comparison workflows
- Custom nsight configurations
- Result analysis commands
- Remote profiling setup (Jetson → Desktop)

## Key Differences from `sam_comparison.py`

| Feature | sam_comparison.py | sam_gpu_profiles.py |
|---------|-------------------|---------------------|
| **Primary Goal** | Accuracy metrics (J, J&F) | GPU performance profiling |
| **Default Limits** | All videos/objects | 1 video, 1 object |
| **Metrics** | J, J&F, FPS, memory | FPS, latency, memory, GPU kernels |
| **Output** | Segmentation overlays, CSV | Nsight reports, timing CSV, stats JSON |
| **Nsight Support** | No | Yes (external + inline) |
| **NVTX Markers** | No | Yes (optional) |
| **Warmup Runs** | No | Yes (configurable) |
| **Multiple Runs** | Single run per object | Configurable runs for statistics |

## Profiling Modes

### External Mode (Recommended)
Wraps entire script execution with `nsys`:
```bash
./run_nsight_profile.sh --models sam2_base_points
```
- **Pros**: Complete system view, automatic kernel tracking
- **Cons**: Larger output files, includes Python overhead

### Inline Mode
Uses NVTX ranges for selective profiling:
```bash
python3 sam_gpu_profiles.py --profile_mode inline
```
- **Pros**: Precise timing of specific sections, smaller files
- **Cons**: Requires nvtx package, manual nsys wrapping if needed

## Platform-Specific Notes

### RTX 3090
- Full BF16 support: use `--autocast bf16`
- High memory: can run larger models (sam2_large)
- Recommended: `--compile_mode max-autotune` for best performance

### Jetson Orin (JetPack 6.2)
- Use FP16: `--autocast fp16`
- Smaller models recommended: sam2_tiny, sam2_small
- Nsys path: `/opt/nvidia/nsight-systems/bin/nsys`
- Unified memory architecture (shared CPU/GPU memory)

## Output Files Structure

```
profile_outputs/
├── profile_summary.csv          # Per-run metrics
├── profile_stats.json           # Aggregated statistics
└── nsight_reports/
    ├── video001_obj0001_sam2_base_points_run0.nsys-rep
    ├── video001_obj0001_sam2_base_points_run1.nsys-rep
    └── ...

nsight_profiles/                 # From run_nsight_profile.sh
├── sam_profile_<pid>_<host>.nsys-rep
├── sam_profile_<pid>_<host>.sqlite
└── ...
```

## Quick Start

### 1. Test Installation
```bash
./run_nsight_profile.sh --test_mode
```

### 2. Profile Single Model
```bash
./run_nsight_profile.sh \
    --split_dir /path/to/sav_val \
    --models sam2_base_points \
    --limit_videos 1 \
    --profile_runs 3
```

### 3. Analyze Results
```bash
# View summary
cat profile_outputs/profile_summary.csv

# View statistics
cat profile_outputs/profile_stats.json

# Analyze nsight report
nsys stats nsight_profiles/*.nsys-rep
nsys-ui nsight_profiles/*.nsys-rep  # GUI
```

## Integration with Existing Workflow

The profiling scripts use the same:
- Dataset structure (JPEGImages_24fps, Annotations_6fps)
- Model weights and loading
- Runner infrastructure (sav_benchmark/runners/)
- Prompt generation (points, bbox)
- Video/object selection logic

This ensures profiling results directly correspond to accuracy benchmarks from `sam_comparison.py`.

## Next Steps

1. **Run test mode** to verify installation:
   ```bash
   ./run_nsight_profile.sh --test_mode
   ```

2. **Profile baseline** on your target hardware:
   ```bash
   # RTX 3090
   ./run_nsight_profile.sh --split_dir /data/sav_val --models sam2_base_points
   
   # Jetson Orin
   NSYS_CMD=/opt/nvidia/nsight-systems/bin/nsys \
   ./run_nsight_profile.sh --split_dir /data/sav_val --models sam2_tiny_points --autocast fp16
   ```

3. **Compare models** to identify best performers:
   ```bash
   ./run_nsight_profile.sh \
       --models sam2_tiny_points,sam2_small_points,sam2_base_points \
       --profile_runs 5
   ```

4. **Optimize** based on nsight analysis:
   - Identify bottleneck kernels
   - Test with torch.compile
   - Adjust precision (bf16/fp16)
   - Tune image size

5. **Cross-platform comparison**:
   - Profile same model on RTX 3090 and Jetson Orin
   - Compare nsight reports side-by-side
   - Identify platform-specific bottlenecks

## Support

See `PROFILE_README.md` for detailed documentation, troubleshooting, and advanced usage.

For questions about the original benchmark, refer to the main project README.
