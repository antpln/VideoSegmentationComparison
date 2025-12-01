#!/bin/bash
# Full experiment run - profiles all models on real dataset
# Generates comprehensive CSV outputs for plotting and analysis

set -e

# Parse arguments
SPLIT_DIR=""
MODELS=""
VIDEOS=3
OBJECTS=2
RUNS=5
WARMUP=2
OUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --split_dir)
            SPLIT_DIR="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --videos)
            VIDEOS="$2"
            shift 2
            ;;
        --objects)
            OBJECTS="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --out_dir)
            OUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --split_dir <path> [--models <list>] [--videos N] [--objects N] [--runs N] [--warmup N]"
            exit 1
            ;;
    esac
done

# Detect platform
if [ -f /etc/nv_tegra_release ]; then
    PLATFORM="jetson"
    HOSTNAME="jetson-orin"
    NSYS_CMD=${NSYS_CMD:-/opt/nvidia/nsight-systems/bin/nsys}
    DEFAULT_AUTOCAST="fp16"
    DEFAULT_MODELS="sam2_tiny_points,sam2_tiny_bbox,sam2_small_points,sam2_small_bbox"
else
    PLATFORM="desktop"
    HOSTNAME=$(hostname)
    NSYS_CMD=${NSYS_CMD:-nsys}
    DEFAULT_AUTOCAST="bf16"
    DEFAULT_MODELS="sam2_tiny_points,sam2_tiny_bbox,sam2_small_points,sam2_small_bbox,sam2_base_points,sam2_base_bbox"
fi

# Use defaults if not specified
MODELS=${MODELS:-$DEFAULT_MODELS}

# Validate split_dir
if [ -z "${SPLIT_DIR}" ]; then
    echo "ERROR: --split_dir is required"
    echo ""
    echo "Usage: $0 --split_dir <path> [options]"
    echo ""
    echo "Options:"
    echo "  --split_dir <path>    Path to sav_val or sav_test dataset (required)"
    echo "  --models <list>       Comma-separated model list (default: platform-specific)"
    echo "  --videos <N>          Number of videos to profile (default: 3)"
    echo "  --objects <N>         Objects per video (default: 2)"
    echo "  --runs <N>            Profile runs per config (default: 5)"
    echo "  --warmup <N>          Warmup runs (default: 2)"
    echo ""
    echo "Example:"
    echo "  $0 --split_dir /data/sav_val --videos 5 --runs 10"
    exit 1
fi

if [ ! -d "${SPLIT_DIR}" ]; then
    echo "ERROR: Dataset directory not found: ${SPLIT_DIR}"
    exit 1
fi

# Accept either extracted JPEG frames or video_fps_24/videos_fps_24 layouts
if [ ! -d "${SPLIT_DIR}/JPEGImages_24fps" ] && [ ! -d "${SPLIT_DIR}/video_fps_24" ] && [ ! -d "${SPLIT_DIR}/videos_fps_24" ]; then
    echo "ERROR: Missing frame source under ${SPLIT_DIR}. Expected JPEGImages_24fps or video_fps_24/videos_fps_24"
    exit 1
fi

if [ ! -d "${SPLIT_DIR}/Annotations_6fps" ]; then
    echo "ERROR: Missing Annotations_6fps in ${SPLIT_DIR}"
    exit 1
fi

# Create output directory with timestamp and platform
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# If an explicit out dir was provided, use it; otherwise create under split_dir
if [ -n "${OUT_DIR}" ]; then
    OUT_DIR="${OUT_DIR}/nsight_experiment_${PLATFORM}_${TIMESTAMP}"
else
    OUT_DIR="${SPLIT_DIR}/nsight_experiment_${PLATFORM}_${TIMESTAMP}"
fi
mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "  NSIGHT PROFILING - FULL EXPERIMENT"
echo "=========================================="
echo ""
echo "Platform: ${PLATFORM} (${HOSTNAME})"
echo "Dataset: ${SPLIT_DIR}"
echo "Models: ${MODELS}"
echo "Videos: ${VIDEOS} | Objects per video: ${OBJECTS}"
echo "Profile runs: ${RUNS} | Warmup runs: ${WARMUP}"
echo "Precision: ${DEFAULT_AUTOCAST}"
echo "Output: ${OUT_DIR}"
echo ""
echo "=========================================="
echo ""

# Log configuration
cat > "${OUT_DIR}/experiment_config.txt" <<EOF
Experiment Configuration
========================
Timestamp: ${TIMESTAMP}
Platform: ${PLATFORM}
Hostname: ${HOSTNAME}
Dataset: ${SPLIT_DIR}
Models: ${MODELS}
Videos: ${VIDEOS}
Objects per video: ${OBJECTS}
Profile runs: ${RUNS}
Warmup runs: ${WARMUP}
Autocast: ${DEFAULT_AUTOCAST}
Nsight: ${NSYS_CMD}
Python: $(python3 --version)
CUDA: $(python3 -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "N/A")
GPU: $(python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "N/A")
EOF

cat "${OUT_DIR}/experiment_config.txt"
echo ""
echo "=========================================="
echo ""

# Start timer
START_TIME=$(date +%s)

# Run profiling
NSYS_OUTPUT_DIR="${OUT_DIR}/nsight_reports" \
./run_nsight_profile.sh \
    --split_dir "${SPLIT_DIR}" \
    --models "${MODELS}" \
    --autocast "${DEFAULT_AUTOCAST}" \
    --limit_videos "${VIDEOS}" \
    --limit_objects "${OBJECTS}" \
    --warmup_runs "${WARMUP}" \
    --profile_runs "${RUNS}" \
    --out_dir "${OUT_DIR}/profile_outputs"

# End timer
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

echo ""
echo "=========================================="
echo "  EXPERIMENT COMPLETE!"
echo "=========================================="
echo ""
echo "Duration: ${DURATION_MIN}m ${DURATION_SEC}s"
echo "Platform: ${PLATFORM}"
echo "Output directory: ${OUT_DIR}"
echo ""
echo "Generated files:"
echo "  📊 ${OUT_DIR}/profile_outputs/profile_summary.csv"
echo "  📊 ${OUT_DIR}/profile_outputs/profile_stats.json"
echo "  📁 ${OUT_DIR}/nsight_reports/*.nsys-rep"
echo "  📄 ${OUT_DIR}/experiment_config.txt"
echo ""
echo "Quick view:"
echo "  head -20 ${OUT_DIR}/profile_outputs/profile_summary.csv"
echo "  python3 -m json.tool ${OUT_DIR}/profile_outputs/profile_stats.json"
echo ""
echo "To copy to desktop for analysis:"
if [ "${PLATFORM}" = "jetson" ]; then
    echo "  # On Jetson, run:"
    echo "  tar -czf nsight_experiment_${PLATFORM}_${TIMESTAMP}.tar.gz -C ${SPLIT_DIR} nsight_experiment_${PLATFORM}_${TIMESTAMP}"
    echo "  # Then on desktop, run:"
    echo "  scp nvidia@jetson-orin:${SPLIT_DIR}/nsight_experiment_${PLATFORM}_${TIMESTAMP}.tar.gz ."
    echo "  tar -xzf nsight_experiment_${PLATFORM}_${TIMESTAMP}.tar.gz"
else
    echo "  # If running on remote desktop:"
    echo "  scp -r user@remote:${OUT_DIR} ."
fi
echo ""
echo "✓ Experiment data ready for analysis!"
