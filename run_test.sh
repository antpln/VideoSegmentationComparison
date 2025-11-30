#!/bin/bash
# Test run script to verify profiling setup is working correctly
# This runs on synthetic data with minimal compute requirements

set -e

echo "=========================================="
echo "  NSIGHT PROFILING - TEST RUN"
echo "=========================================="
echo ""

# Detect platform
if [ -f /etc/nv_tegra_release ]; then
    PLATFORM="jetson"
    echo "Platform: Jetson Orin"
    NSYS_CMD=${NSYS_CMD:-/opt/nvidia/nsight-systems/bin/nsys}
    DEFAULT_AUTOCAST="fp16"
    DEFAULT_MODEL="sam2_tiny_points"
else
    PLATFORM="desktop"
    echo "Platform: Desktop (assuming RTX 3090 or similar)"
    NSYS_CMD=${NSYS_CMD:-nsys}
    DEFAULT_AUTOCAST="bf16"
    DEFAULT_MODEL="sam2_base_points"
fi

# Check if nsys is available
if ! command -v "${NSYS_CMD}" &> /dev/null; then
    echo "ERROR: nsys not found at '${NSYS_CMD}'"
    echo "Set NSYS_CMD environment variable or install Nsight Systems"
    exit 1
fi

echo "Nsight Systems: ${NSYS_CMD}"
echo "Default precision: ${DEFAULT_AUTOCAST}"
echo "Default model: ${DEFAULT_MODEL}"
echo ""

# Run setup check
echo "Running setup verification..."
python3 setup_check.py
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Setup check failed. Please resolve issues above."
    exit 1
fi

echo ""
echo "=========================================="
echo "  Running Test Profile"
echo "=========================================="
echo ""

# Run test profile with minimal settings
./run_nsight_profile.sh \
    --test_mode \
    --models "${DEFAULT_MODEL}" \
    --autocast "${DEFAULT_AUTOCAST}" \
    --warmup_runs 1 \
    --profile_runs 2

echo ""
echo "=========================================="
echo "  TEST RUN COMPLETE!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - test_profile_outputs/profile_summary.csv"
echo "  - test_profile_outputs/profile_stats.json"
echo "  - test_profile_outputs/nsight_reports/*.nsys-rep"
echo ""
echo "To view results:"
echo "  cat test_profile_outputs/profile_summary.csv"
echo "  cat test_profile_outputs/profile_stats.json"
echo ""
echo "To analyze nsight reports:"
echo "  nsys stats test_profile_outputs/nsight_reports/*.nsys-rep"
echo ""
echo "✓ Setup verified! Ready for full experiment runs."
