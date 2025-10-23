#!/bin/bash
# Helper script to run GPU profiling with NVIDIA Nsight Systems
# 
# This script wraps the sam_gpu_profiles.py execution with nsys for external profiling.
# Supports both RTX 3090 (desktop) and Jetson Orin (JetPack 6.2)
#
# Usage:
#   ./run_nsight_profile.sh [python_script_args]
#
# Example:
#   ./run_nsight_profile.sh --split_dir /path/to/sav_val --models sam2_base_points --limit_videos 1

set -e

# Configuration
NSYS_CMD=${NSYS_CMD:-nsys}
PYTHON_CMD=${PYTHON_CMD:-python3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_SCRIPT="${SCRIPT_DIR}/sam_gpu_profiles.py"

# Default nsys options optimized for GPU profiling
# Customize these based on your needs:
# - trace: which APIs to trace (cuda, cudnn, cublas, nvtx, osrt)
# - sample: cpu or none
# - backtrace: none, fp, dwarf (dwarf gives detailed stack traces but larger files)
# - cuda-memory-usage: track CUDA memory allocations
NSYS_TRACE=${NSYS_TRACE:-"cuda,cudnn,cublas,nvtx,osrt"}
NSYS_SAMPLE=${NSYS_SAMPLE:-"none"}
NSYS_BACKTRACE=${NSYS_BACKTRACE:-"none"}
NSYS_OUTPUT_DIR=${NSYS_OUTPUT_DIR:-"./nsight_profiles"}
NSYS_OUTPUT_NAME=${NSYS_OUTPUT_NAME:-"sam_profile_%p_%h"}

# Create output directory
mkdir -p "${NSYS_OUTPUT_DIR}"

# Check if nsys is available
if ! command -v "${NSYS_CMD}" &> /dev/null; then
    echo "ERROR: ${NSYS_CMD} not found in PATH"
    echo ""
    echo "Please install NVIDIA Nsight Systems:"
    echo "  - Desktop: Download from https://developer.nvidia.com/nsight-systems"
    echo "  - Jetson: Included with JetPack 6.2"
    echo ""
    echo "Or set NSYS_CMD environment variable to the correct path"
    exit 1
fi

# Display nsys version
echo "Using Nsight Systems:"
"${NSYS_CMD}" --version
echo ""

# Check Python script exists
if [ ! -f "${PROFILE_SCRIPT}" ]; then
    echo "ERROR: Profile script not found: ${PROFILE_SCRIPT}"
    exit 1
fi

# Build nsys command
NSYS_FULL_CMD=(
    "${NSYS_CMD}"
    profile
    --trace="${NSYS_TRACE}"
    --sample="${NSYS_SAMPLE}"
    --backtrace="${NSYS_BACKTRACE}"
    --cuda-memory-usage=true
    --output="${NSYS_OUTPUT_DIR}/${NSYS_OUTPUT_NAME}"
    --force-overwrite=true
    --export=sqlite
    "${PYTHON_CMD}"
    "${PROFILE_SCRIPT}"
)

# Add user-provided arguments
NSYS_FULL_CMD+=("$@")

# If no arguments provided, run in test mode
if [ $# -eq 0 ]; then
    echo "No arguments provided, running in test mode..."
    NSYS_FULL_CMD+=(--test_mode)
fi

echo "Running profiling command:"
echo "${NSYS_FULL_CMD[@]}"
echo ""
echo "Output directory: ${NSYS_OUTPUT_DIR}"
echo "================================================"
echo ""

# Execute nsys profiling
"${NSYS_FULL_CMD[@]}"

echo ""
echo "================================================"
echo "Profiling complete!"
echo ""
echo "Report files saved to: ${NSYS_OUTPUT_DIR}"
echo ""
echo "To view reports:"
echo "  1. GUI: nsys-ui ${NSYS_OUTPUT_DIR}/<report>.nsys-rep"
echo "  2. CLI: nsys stats ${NSYS_OUTPUT_DIR}/<report>.nsys-rep"
echo ""
echo "To export to other formats:"
echo "  nsys export -t sqlite ${NSYS_OUTPUT_DIR}/<report>.nsys-rep"
echo "  nsys export -t csv ${NSYS_OUTPUT_DIR}/<report>.nsys-rep"
