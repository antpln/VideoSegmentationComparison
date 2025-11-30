# Quick Setup Guide for NSIGHT Profiling

This branch (`nsight`) is set up for seamless GPU profiling on both **RTX 3090** and **Jetson Orin**.

## Quick Start (Clone and Run)

### 1. Clone the repository
```bash
git clone https://github.com/antpln/VideoSegementationComparison.git
cd VideoSegementationComparison
git checkout nsight
git submodule update --init --recursive  # For EdgeTAM
```

### 2. Install dependencies

**On RTX 3090 / Desktop:**
```bash
# Install PyTorch with CUDA support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip3 install -r requirements.txt

# Install Nsight Systems (if not already installed)
# Download from: https://developer.nvidia.com/nsight-systems
```

**On Jetson Orin:**
```bash
# PyTorch comes with JetPack 6.2
# Install other dependencies
pip3 install -r requirements.txt

# Nsight Systems is included with JetPack 6.2
export NSYS_CMD=/opt/nvidia/nsight-systems/bin/nsys
```

### 3. Verify setup
```bash
python3 setup_check.py
```
This will check all dependencies, CUDA availability, nsight installation, and model weights.

### 4. Download model weights

**SAM2 weights** (any size):
```bash
# Download from GitHub releases
wget https://github.com/facebookresearch/sam2/releases/download/v1.0/sam2.1_b.pt
# Or use Ultralytics CLI
yolo task=segment mode=predict model=sam2.1_b.pt
```

**EdgeTAM weights** (optional):
```bash
cd EdgeTAM
# Follow EdgeTAM setup instructions to download weights
# Place in EdgeTAM/checkpoints/edgetam.pt
```

### 5. Run a test profile
```bash
# Quick test with synthetic data
./run_nsight_profile.sh --test_mode

# Or directly
python3 sam_gpu_profiles.py --test_mode
```

### 6. Profile on real data

**RTX 3090:**
```bash
./run_nsight_profile.sh \
    --split_dir /path/to/sav_val \
    --models sam2_base_points \
    --autocast bf16 \
    --warmup_runs 2 \
    --profile_runs 5 \
    --limit_videos 2
```

**Jetson Orin:**
```bash
NSYS_CMD=/opt/nvidia/nsight-systems/bin/nsys \
./run_nsight_profile.sh \
    --split_dir /path/to/sav_val \
    --models sam2_tiny_points \
    --autocast fp16 \
    --warmup_runs 2 \
    --profile_runs 5 \
    --limit_videos 1
```

## What Gets Profiled

### Basic Hardware Metrics (already in main branch)
From `sav_benchmark/utils.py`:
- **GPU Peak Allocated Memory**: via `torch.cuda.max_memory_allocated()`
- **GPU Peak Reserved Memory**: via `torch.cuda.max_memory_reserved()`
- **CPU Peak RSS**: via `psutil.Process().memory_info().rss`
- **FPS & Latency**: timing around inference loop with `torch.cuda.synchronize()`

### New NSIGHT Statistics (this branch)
From `sam_gpu_profiles.py` + `run_nsight_profile.sh`:

**Kernel-Level Profiling:**
- GPU kernel execution times (per kernel)
- CUDA API call overhead
- cuDNN/cuBLAS operation timing
- GPU utilization (SM activity, memory bandwidth)

**Memory Profiling:**
- Memory allocation patterns
- Host-to-device and device-to-host transfers
- Memory bandwidth utilization
- CUDA memory API calls

**Timeline Analysis:**
- CPU/GPU overlap and synchronization
- Multi-stream execution
- Kernel launch overhead

**Advanced Metrics:**
- NVTX markers for custom code regions (inline mode)
- Warp occupancy and divergence
- Cache hit rates
- Power/thermal throttling events

## Output Files

After profiling, you'll have:

```
profile_outputs/
├── profile_summary.csv       # Per-run metrics (FPS, memory, latency)
├── profile_stats.json        # Aggregated statistics per model
└── nsight_reports/
    └── *.nsys-rep           # Binary Nsight reports

nsight_profiles/             # From run_nsight_profile.sh
└── *.nsys-rep              # System-wide profiling data
```

## Analyzing Results

### Quick CSV summary
```bash
cat profile_outputs/profile_summary.csv
cat profile_outputs/profile_stats.json | python3 -m json.tool
```

### Nsight CLI analysis
```bash
# Overall summary
nsys stats nsight_profiles/*.nsys-rep

# CUDA kernel summary
nsys stats --report cuda_gpu_kern_sum nsight_profiles/*.nsys-rep

# Memory operations
nsys stats --report cuda_mem_sum nsight_profiles/*.nsys-rep
```

### Nsight GUI (requires X11)
```bash
nsys-ui nsight_profiles/*.nsys-rep
```

### Export for custom analysis
```bash
# Export to SQLite
nsys export -t sqlite nsight_profiles/*.nsys-rep

# Export to CSV
nsys export -t csv nsight_profiles/*.nsys-rep
```

## Using the Makefile

The `Makefile.profile` provides shortcuts:

```bash
# Test mode
make -f Makefile.profile test

# Profile on 3090
make -f Makefile.profile profile-3090 SPLIT_DIR=/path/to/sav_val

# Profile on Jetson
make -f Makefile.profile profile-jetson SPLIT_DIR=/path/to/sav_val

# Compare models
make -f Makefile.profile compare-models SPLIT_DIR=/path/to/sav_val

# Show statistics
make -f Makefile.profile stats

# Analyze nsight reports
make -f Makefile.profile analyze
```

## Example Workflows

See `profile_examples.sh` for complete command examples covering:
- Quick tests
- Platform-specific profiling (3090 vs Jetson)
- Memory profiling
- Batch comparisons
- Custom nsight configurations
- Result analysis
- Remote profiling (Jetson → Desktop)

## Troubleshooting

### Out of Memory Errors
```bash
# Use smaller model
--models sam2_tiny_points

# Use FP16 instead of BF16
--autocast fp16

# Disable cuDNN
--disable_cudnn

# Reduce image size
--imgsz 512
```

### Nsight not found
```bash
# Set custom path
export NSYS_CMD=/path/to/nsys

# Or use inline mode (no nsys wrapper)
python3 sam_gpu_profiles.py --profile_mode inline ...
```

### Large profile files
```bash
# Reduce traced APIs
NSYS_TRACE=cuda,nvtx ./run_nsight_profile.sh ...

# Disable CPU sampling
NSYS_SAMPLE=none ./run_nsight_profile.sh ...
```

## Documentation

- `PROFILING_SETUP.md`: Complete overview of profiling setup
- `PROFILE_README.md`: Detailed usage guide with all options
- `profile_examples.sh`: Ready-to-run example commands
- `Makefile.profile`: Makefile shortcuts for common tasks

## Support

For questions about:
- **Profiling setup**: See `PROFILE_README.md` and `PROFILING_SETUP.md`
- **Main benchmark**: See main `README.md`
- **SAM2 models**: https://github.com/facebookresearch/sam2
- **EdgeTAM**: https://github.com/zhang-tao-whu/EdgeTAM
- **Nsight Systems**: https://docs.nvidia.com/nsight-systems/
