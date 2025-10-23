#!/bin/bash
# Example profiling commands for different scenarios
# Copy and modify these for your specific needs

# ==============================================================================
# QUICK TESTS
# ==============================================================================

# Test on synthetic data
./run_nsight_profile.sh --test_mode

# Test without nsight (Python timing only)
python3 sam_gpu_profiles.py --test_mode --profile_mode inline


# ==============================================================================
# RTX 3090 PROFILING
# ==============================================================================

# Profile SAM2 models with BF16
./run_nsight_profile.sh \
    --split_dir /path/to/sav_val \
    --models sam2_tiny_points,sam2_small_points,sam2_base_points,sam2_large_points \
    --autocast bf16 \
    --warmup_runs 2 \
    --profile_runs 5 \
    --limit_videos 2

# Profile with torch.compile optimization
./run_nsight_profile.sh \
    --split_dir /path/to/sav_val \
    --models sam2_base_points \
    --compile_models \
    --compile_mode max-autotune \
    --autocast bf16 \
    --warmup_runs 3 \
    --profile_runs 10

# Compare point vs bbox prompts
./run_nsight_profile.sh \
    --split_dir /path/to/sav_val \
    --models sam2_base_points,sam2_base_bbox \
    --limit_videos 3 \
    --profile_runs 5


# ==============================================================================
# JETSON ORIN PROFILING (JetPack 6.2)
# ==============================================================================

# Set Jetson-specific paths
export NSYS_CMD=/opt/nvidia/nsight-systems/bin/nsys
export NSYS_OUTPUT_DIR=/home/nvidia/profiles

# Profile EdgeTAM with FP16 (recommended for Jetson)
./run_nsight_profile.sh \
    --split_dir /path/to/sav_val \
    --models edgetam_points,edgetam_bbox \
    --autocast fp16 \
    --warmup_runs 2 \
    --profile_runs 5 \
    --limit_videos 1

# Profile smallest SAM2 model for edge deployment
./run_nsight_profile.sh \
    --split_dir /path/to/sav_val \
    --models sam2_tiny_points \
    --autocast fp16 \
    --imgsz 512 \
    --warmup_runs 1 \
    --profile_runs 5

# Test different precision modes
for precision in fp16 none; do
    NSYS_OUTPUT_NAME="jetson_${precision}_%p" \
    ./run_nsight_profile.sh \
        --split_dir /path/to/sav_val \
        --models sam2_tiny_points \
        --autocast $precision \
        --profile_runs 3
done


# ==============================================================================
# MEMORY PROFILING
# ==============================================================================

# Profile with detailed memory tracking
NSYS_TRACE=cuda,osrt \
./run_nsight_profile.sh \
    --split_dir /path/to/sav_val \
    --models sam2_large_points \
    --limit_videos 1 \
    --profile_runs 1

# Test memory optimization with cuDNN disabled
./run_nsight_profile.sh \
    --split_dir /path/to/sav_val \
    --models sam2_base_points \
    --disable_cudnn \
    --limit_videos 1


# ==============================================================================
# BATCH COMPARISONS
# ==============================================================================

# Compare all SAM2 sizes with points
./run_nsight_profile.sh \
    --split_dir /path/to/sav_val \
    --models sam2_tiny_points,sam2_small_points,sam2_base_points,sam2_large_points \
    --limit_videos 3 \
    --limit_objects 2 \
    --profile_runs 5 \
    --out_dir ./comparison_sam2_sizes

# Compare SAM2 vs EdgeTAM
./run_nsight_profile.sh \
    --split_dir /path/to/sav_val \
    --models sam2_base_points,edgetam_points \
    --limit_videos 5 \
    --profile_runs 5 \
    --out_dir ./comparison_sam2_vs_edgetam


# ==============================================================================
# CUSTOM NSIGHT SETTINGS
# ==============================================================================

# Minimal tracing (smaller files, faster)
NSYS_TRACE=cuda,nvtx \
NSYS_SAMPLE=none \
./run_nsight_profile.sh \
    --split_dir /path/to/sav_val \
    --models sam2_base_points

# Full tracing with CPU sampling
NSYS_TRACE=cuda,cudnn,cublas,nvtx,osrt \
NSYS_SAMPLE=cpu \
NSYS_BACKTRACE=dwarf \
./run_nsight_profile.sh \
    --split_dir /path/to/sav_val \
    --models sam2_base_points


# ==============================================================================
# INLINE PROFILING WITH NVTX
# ==============================================================================

# Run with NVTX markers (requires pip install nvtx)
nsys profile \
    -o inline_profile \
    --trace=cuda,nvtx \
    python3 sam_gpu_profiles.py \
        --split_dir /path/to/sav_val \
        --models sam2_base_points \
        --profile_mode inline \
        --warmup_runs 1 \
        --profile_runs 3


# ==============================================================================
# ANALYZING RESULTS
# ==============================================================================

# View in GUI (requires X11 or remote desktop)
# nsys-ui nsight_profiles/*.nsys-rep

# CLI statistics
nsys stats nsight_profiles/*.nsys-rep

# CUDA kernel summary
nsys stats --report cuda_gpu_kern_sum nsight_profiles/*.nsys-rep

# Memory operations summary
nsys stats --report cuda_mem_sum nsight_profiles/*.nsys-rep

# Export to SQLite for custom analysis
nsys export -t sqlite nsight_profiles/*.nsys-rep

# Export to CSV
nsys export -t csv nsight_profiles/*.nsys-rep


# ==============================================================================
# REMOTE PROFILING (Jetson to Desktop)
# ==============================================================================

# On Jetson: Run profiling (creates .nsys-rep files)
ssh nvidia@jetson-orin << 'EOF'
cd /path/to/VideoSegementationComparison
./run_nsight_profile.sh \
    --split_dir /data/sav_val \
    --models sam2_tiny_points \
    --autocast fp16 \
    --limit_videos 1
EOF

# Copy results to desktop
scp -r nvidia@jetson-orin:/path/to/VideoSegementationComparison/nsight_profiles ./jetson_profiles

# Analyze on desktop with GUI
nsys-ui ./jetson_profiles/*.nsys-rep
